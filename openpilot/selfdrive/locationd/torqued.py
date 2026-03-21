#!/usr/bin/env python3
import os
import numpy as np
from collections import deque, defaultdict

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.locationd.helpers import PointBuckets, ParameterEstimator, PoseCalibrator, Pose
from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import TorqueEstimatorExt

HISTORY = 5  # secs
POINTS_PER_BUCKET = 1500
MIN_POINTS_TOTAL = 4000
MIN_POINTS_TOTAL_QLOG = 600
FIT_POINTS_TOTAL = 2000
FIT_POINTS_TOTAL_QLOG = 600
MIN_VEL = 15  # m/s
FRICTION_FACTOR = 1.5  # ~85% of data coverage
FACTOR_SANITY = 0.3
FACTOR_SANITY_QLOG = 0.5
FRICTION_SANITY = 0.5
FRICTION_SANITY_QLOG = 0.8
STEER_MIN_THRESHOLD = 0.02
MIN_FILTER_DECAY = 50
MAX_FILTER_DECAY = 250
LAT_ACC_THRESHOLD = 1
STEER_BUCKET_BOUNDS = [(-0.5, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]
MIN_BUCKET_POINTS = np.array([100, 300, 500, 500, 500, 500, 300, 100])
MIN_ENGAGE_BUFFER = 2  # secs

VERSION = 1  # bump this to invalidate old parameter caches
ALLOWED_CARS = ['toyota', 'hyundai', 'rivian', 'honda', 'volkswagen']


def slope2rot(slope):
  sin = np.sqrt(slope ** 2 / (slope ** 2 + 1))
  cos = np.sqrt(1 / (slope ** 2 + 1))
  return np.array([[cos, -sin], [sin, cos]])


class TorqueBuckets(PointBuckets):
  def add_point(self, x, y):
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, 1.0, y])
        break


class TorqueEstimator(ParameterEstimator, TorqueEstimatorExt):
  def __init__(self, CP, decimated=False, track_all_points=False):
    ParameterEstimator.__init__(self)
    TorqueEstimatorExt.__init__(self, CP)
    self.CP = CP
    self.hist_len = int(HISTORY / DT_MDL)
    self.lag = 0.0
    self.track_all_points = track_all_points  # for offline analysis, without max lateral accel or max steer torque filters
    if decimated:
      self.min_bucket_points: list[float] = (MIN_BUCKET_POINTS / 10).tolist()
      self.min_points_total = MIN_POINTS_TOTAL_QLOG
      self.fit_points = FIT_POINTS_TOTAL_QLOG
      self.factor_sanity = FACTOR_SANITY_QLOG
      self.friction_sanity = FRICTION_SANITY_QLOG

    else:
      self.min_bucket_points = MIN_BUCKET_POINTS.tolist()
      self.min_points_total = MIN_POINTS_TOTAL
      self.fit_points = FIT_POINTS_TOTAL
      self.factor_sanity = FACTOR_SANITY
      self.friction_sanity = FRICTION_SANITY

    self.offline_friction = 0.0
    self.offline_latAccelFactor = 0.0
    self.resets = 0.0
    self.use_params = CP.brand in ALLOWED_CARS and CP.lateralTuning.which() == 'torque'

    if CP.lateralTuning.which() == 'torque':
      self.offline_friction = CP.lateralTuning.torque.friction
      self.offline_latAccelFactor = CP.lateralTuning.torque.latAccelFactor

    self.calibrator = PoseCalibrator()

    TorqueEstimatorExt.initialize_custom_params(self, decimated)

    self.reset()

    initial_params = {
      'latAccelFactor': self.offline_latAccelFactor,
      'latAccelOffset': 0.0,
      'frictionCoefficient': self.offline_friction,
      'points': []
    }
    self.decay = MIN_FILTER_DECAY
    self.min_lataccel_factor = (1.0 - self.factor_sanity) * self.offline_latAccelFactor
    self.max_lataccel_factor = (1.0 + self.factor_sanity) * self.offline_latAccelFactor
    self.min_friction = (1.0 - self.friction_sanity) * self.offline_friction
    self.max_friction = (1.0 + self.friction_sanity) * self.offline_friction

    # try to restore cached params
    params = Params()
    self.params = params
    params_cache = params.get("CarParamsPrevRoute")
    torque_cache = params.get("LiveTorqueParameters")
    if params_cache is not None and torque_cache is not None:
      try:
        with log.Event.from_bytes(torque_cache) as log_evt:
          cache_ltp = log_evt.lateralTorqueParameters
        with car.CarParams.from_bytes(params_cache) as msg:
          cache_CP = msg
        if self.get_restore_key(cache_CP, cache_ltp.version) == self.get_restore_key(CP, VERSION):
          if cache_ltp.valid:
            initial_params = {
              'latAccelFactor': cache_ltp.latAccelFactorFiltered,
              'latAccelOffset': cache_ltp.latAccelOffsetFiltered,
              'frictionCoefficient': cache_ltp.frictionCoefficientFiltered
            }
          cached_points: list[list[float]] = [list(point) for point in cache_ltp.points]
          initial_params['points'] = cached_points
          self.decay = cache_ltp.decay
          self.filtered_points.load_points(cached_points)
          self._restore_ext_cache(cache_ltp)
    lateralTorqueParameters.latAccelFactorFiltered = float(self.filtered_params['latAccelFactor'].x)
    lateralTorqueParameters.latAccelOffsetFiltered = float(self.filtered_params['latAccelOffset'].x)
    lateralTorqueParameters.frictionCoefficientFiltered = float(self.filtered_params['frictionCoefficient'].x)
    lateralTorqueParameters.totalBucketPoints = len(self.filtered_points)
    lateralTorqueParameters.calPerc = self.filtered_points.get_valid_percent()
    lateralTorqueParameters.decay = self.decay
    lateralTorqueParameters.maxResets = self.resets
    self._extend_msg(msg, with_points)

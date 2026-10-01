"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time

import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log, custom

from opendbc.car import structs
from opendbc.sunnypilot.car.interfaces import get_steer_slew_schedule
from openpilot.common.params import ParamKeyFlag, Params
from openpilot.common.realtime import DT_CTRL
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.livedelay.helpers import get_fixed_lat_delay
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase
from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import LIVE_TORQUE_PARAMETERS_SP_SERVICE
from openpilot.sunnypilot.selfdrive.controls.lib.blinker_pause_lateral import BlinkerPauseLateral
from openpilot.sunnypilot.selfdrive.controls.lib.lane_change_smoothing import LaneChangeSmoothing
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque as LatControlTorqueV0
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v2 import LatControlTorque as LatControlTorqueV2
from openpilot.sunnypilot.selfdrive.controls.lib.steer_limit import classify
from openpilot.sunnypilot.selfdrive.controls.lib.torque_tune import MazdaTorqueV2Mode, resolved_mazda_v2_mode, resolved_tune_versions

# how long lateral control has to be inactive, continuously, before the torque tune follows
# a model change: a one-frame drop (a steer fault flicker) would otherwise swap and resume
# steering on a controller no inactive frame had primed
TUNE_SWAP_INACTIVE_FRAMES = round(0.5 / DT_CTRL)


# Drive-start params dump: every set key except DONT_LOG secrets and oversized blobs,
# published once on the first carControlSP of the drive so rlogs record the exact
# settings that produced them (CX-8 controller attribution took a day without this).
PARAM_DUMP_VALUE_CAP = 512


def build_param_dump(params: Params) -> list[dict]:
  dont_log = set(params.all_keys(flag=ParamKeyFlag.DONT_LOG))
  dump = []
  for key_raw in sorted(k for k in params.all_keys() if k is not None):
    if key_raw in dont_log:
      continue
    key = key_raw.decode()
    key_type = params.get_type(key)
    value = params.get(key)
    if value is None:
      continue
    raw = value if isinstance(value, bytes) else str(value).encode()
    if len(raw) > PARAM_DUMP_VALUE_CAP:
      continue
    # ParamKeyType ordinals match CarControlSP.ParamType exactly
    dump.append({"key": key, "type": int(key_type), "value": raw})
  return dump


def lateral_extension_of(lac):
  """Which torque-space extension currently owns the output, if any. v2's host
  disables the overrides, so this reads none there even with NNLC params set."""
  ext = getattr(lac, "extension", None)
  if ext is None or not ext.overrides_output:
    return custom.CarControlSP.LateralExtension.none
  if ext._nnlc_enabled:
    return custom.CarControlSP.LateralExtension.neuralNetwork
  if ext._jerk_aware_enabled:
    return custom.CarControlSP.LateralExtension.jerkAware
  return custom.CarControlSP.LateralExtension.none
from openpilot.sunnypilot.selfdrive.controls.lib.turn_assist import TurnAssistController


# TI recommended-settings seeds. Unset params only — explicit user picks persist.
# Only keys that DIFFER from upstream defaults belong here; unset-only seeding a key to
# its stock default is a dead write. Stock already covers:
# LagdToggle=1, LateralJerkTorqueController=0, CustomTorqueParams=0, TorqueParamsOverrideEnabled=0.
TI_TUNE_SEEDS = {
  "LiveTorqueParamsToggle": True,       # "Self-Tune": defaults OFF upstream, we want it learning
  "LaneChangeSmoothing": 2,              # Medium pace; the key is an option index, not a bool
  "LowSpeedTurnAssist": True,           # TI cars steer at standstill, so low-speed assist works
  # Owner-validated on the CX-8 (tracking RMSE -35%, override share halved vs static).
  # CX-5 matches its own exact NN model; the CX-8 fuzzy-matches MAZDA_CX5_2022's
  # (similarity 0.93, field-validated). Gate still requires a model match — cars
  # without one simply keep the PID path.
  "NeuralNetworkLateralControl": True,
}
# No delay seed on either TI platform: the owner has preferred both modes at different
# times ("Live Delay" in Aug, fixed 0.05 in the current NNLC drives) — there is no proven
# universal winner, so the stock LagdToggle=1 default governs fresh installs and explicit
# user picks persist untouched.
TI_PLATFORMS = ("MAZDA_CX5_2022", "MAZDA_CX8_2022")


def seed_ti_defaults(params: Params, CP) -> None:
  """TI cars get torque enforcement plus our recommended lateral defaults, seeded once.
  Unset params only — anything the user explicitly picked is left alone."""
  if not params.get_bool("TorqueInterceptorEnabled"):
    return
  if params.get("EnforceTorqueControl") is None:
    # enforcement off pins v0 via the resolver; the tune resolves to the declared 2.0
    # default once enforced. block=True: resolved_tune_version reads it right after, same call.
    params.put_bool("EnforceTorqueControl", True, block=True)
  if params.get("TorqueControlTune") is None:
    params.put("TorqueControlTune", 2.0, block=True)
  if CP.carFingerprint not in TI_PLATFORMS:
    return
  for key, value in TI_TUNE_SEEDS.items():
    if params.get(key) is None:
      params.put(key, value, block=True)


def torque_v2_mode_of(lac):
  mode = getattr(lac, "mazda_v2_mode", None)
  if mode == MazdaTorqueV2Mode.A:
    return custom.CarControlSP.TorqueV2Mode.modeA
  if mode == MazdaTorqueV2Mode.B:
    return custom.CarControlSP.TorqueV2Mode.modeB
  return custom.CarControlSP.TorqueV2Mode.notApplicable


class ControlsExt(ModelStateBase):
  lagd_toggle: bool

  def __init__(self, CP: structs.CarParams, params: Params):
    ModelStateBase.__init__(self)
    self.CP = CP
    self.params = params
    self.lagd_toggle = CP.lateralTuning.which() == 'torque' and params.get_bool("LagdToggle")
    if CP.lateralTuning.which() == 'torque' and not self.lagd_toggle:
      self.lat_delay = get_fixed_lat_delay(params, CP.steerActuatorDelay)
    self._param_update_time: float = 0.0
    self.blinker_pause_lateral = BlinkerPauseLateral()
    self.turn_assist = TurnAssistController(CP)
    self.lane_change_smoothing = LaneChangeSmoothing()

    # steer-limit classifier (lib/steer_limit.py): None on brands without carcontroller rate
    # limits and on angle-steered cars, where controlsd's flag is left untouched
    self._steer_slew_schedule = None
    if CP.steerControlType != structs.CarParams.SteerControlType.angle:
      self._steer_slew_schedule = get_steer_slew_schedule(CP)
    self._lat_active_last = False
    # frames in a row with CC.latActive false, as state_control decided it (note_lat_active);
    # the torque tune swap waits for TUNE_SWAP_INACTIVE_FRAMES of them
    self._inactive_frames = 0
    self._applied_torque_prev: float | None = None

    cloudlog.info("controlsd_ext is waiting for CarParamsSP")
    self.CP_SP = messaging.log_from_bytes(params.get("CarParamsSP", block=True), custom.CarParamsSP)
    cloudlog.info("controlsd_ext got CarParamsSP")

    self.sm_services_ext = ['radarState', 'selfdriveStateSP', LIVE_TORQUE_PARAMETERS_SP_SERVICE]
    self.pm_services_ext = ['carControlSP']
    self._param_dump_pending = True

  def initialize_lateral_control(self, lac, CI, dt):
    """One controller per model size, built once; every drive starts on the small model's."""
    # the enforce-off v0 forcing and the unset-param defaults both live in the resolver,
    # shared with the settings UIs so they gate on the tune that will actually run
    # TI cars default to tune v2, plus the torque-control enforcement the version
    # resolution hangs off of (without it every Mazda is pinned to v0 below).
    # Seeds fire only while the param is UNSET - an explicit user pick (including
    # enforce-off, or any tune version) is a set param and always persists.
    seed_ti_defaults(self.params, self.CP)

    versions = resolved_tune_versions(self.params, self.CP.lateralTuning.which() == 'torque')

    def build(version):
      if version == 0.0:
        return LatControlTorqueV0(self.CP, self.CP_SP, CI, dt)
      if version == 2.0:
        return LatControlTorqueV2(self.CP, self.CP_SP, CI, dt, mazda_v2_mode=resolved_mazda_v2_mode(self.params, self.CP))
      return lac

    built = {version: build(version) for version in dict.fromkeys(versions.values())}
    self._lac_by_size = {big: built[version] for big, version in versions.items()}
    self._lacs = tuple(built.values())
    return self._lac_by_size[False]

  def select_lateral_control(self, sm: messaging.SubMaster) -> None:
    """Runs at the end of every frame. modelV2.big says which model produced the frame; a
    change swaps self.LaC to the controller tuned for it, reset, but never while lateral
    control is active this frame. The idle controller holds whatever it had when it last
    steered (its request buffer, previous measurement, integrator): on the 2026-09-29 drives
    every big-to-small hand-back while steering stepped the commanded torque by 0.09 to 0.19
    of full scale in one tick. Since a hand-back no longer ends in a soft disable, the small
    model is carried by the tune that was steering until lateral has been inactive for
    TUNE_SWAP_INACTIVE_FRAMES in a row (a disengage, a blinker pause, a stop); a shorter drop
    keeps it. A swap to the big model only happens with nothing in control, and its second of
    no-entry is longer than the wait. From the next frame state_control runs the incoming
    controller, inactive until an engagement, which primes it as for any engagement, and
    pushes the live torque params, modelV2 and the lag into it. An engagement on the very
    frame after the swap would find it unprimed; priming it here would need the frame's
    CarState, which state_control has and this does not."""
    if len(self._lacs) == 1 or self._inactive_frames < TUNE_SWAP_INACTIVE_FRAMES:
      return
    big = bool(sm['modelV2'].big)
    lac = self._lac_by_size[big]
    if lac is self.LaC:
      return
    self.LaC = lac
    lac.reset()
    cloudlog.warning("controlsd: %s model, swapped to its torque tune", "big" if big else "small")

  def get_params_sp(self, sm: messaging.SubMaster) -> None:
    if time.monotonic() - self._param_update_time > PARAMS_UPDATE_PERIOD:
      self.blinker_pause_lateral.get_params()
      self.turn_assist.get_params()
      self.lane_change_smoothing.get_params()

      if self.CP.lateralTuning.which() == 'torque':
        self.lagd_toggle = self.params.get_bool("LagdToggle")
        if not self.lagd_toggle:
          self.lat_delay = get_fixed_lat_delay(self.params, self.CP.steerActuatorDelay)

      self._param_update_time = time.monotonic()

  def get_lat_active(self, sm: messaging.SubMaster) -> bool:
    self._lat_active_last = self._get_lat_active(sm)
    return self._lat_active_last

  def _get_lat_active(self, sm: messaging.SubMaster) -> bool:
    if self.blinker_pause_lateral.update(sm['carState']):
      return False

    ss_sp = sm['selfdriveStateSP']
    if ss_sp.mads.available:
      return bool(ss_sp.mads.active)

    # MADS not available, use stock state to engage
    return bool(sm['selfdriveState'].active)

  def reclassify_steer_limit(self, sm: messaging.SubMaster) -> None:
    """Runs after publish() has set this frame's steer_limited_by_safety from the raw torque
    mismatch. Replaces it with the classifier's directional, rail-aware flag (lib/steer_limit.py)
    before the next frame's LaC.update reads it. Torque tunes only; while lateral is inactive
    (get_lat_active's last value, the one publish() gated on) the flag is left as controlsd set it."""
    ext = getattr(self.LaC, 'extension', None)
    if ext is None or self._steer_slew_schedule is None:
      return
    if not self._lat_active_last:
      self._applied_torque_prev = None
      return
    v_ego = sm['carState'].vEgo
    applied = float(sm['carOutput'].actuatorsOutput.torque)
    bp, up, down = self._steer_slew_schedule
    rail_scale = ext.rail_scale_at(v_ego)
    limit = classify(ext.commanded_torque, applied, self._applied_torque_prev,
                     float(np.interp(v_ego, bp, up)), float(np.interp(v_ego, bp, down)),
                     rail_scale, self.steer_limited_by_safety, ext.last_error, ext.integrator)
    self.steer_limited_by_safety = limit.limited
    ext.set_actuator_state(applied, limit.at_rail)
    self._applied_torque_prev = applied

  def update_lateral_assist(self, sm: messaging.SubMaster, lat_active: bool, new_desired_curvature: float,
                            prev_desired_curvature: float, current_curvature: float) -> tuple[float, float]:
    """Low-speed turn assist + lane-change smoothing over the desired curvature,
    returning (new_desired_curvature, jerk_factor) for clip_curvature. The lateral
    maneuver mode's scripted commands must pass through the stock clip untouched."""
    self.note_lat_active(lat_active)
    if sm.valid['lateralManeuverPlan']:
      # clear the whole assist layer: state armed before maneuver mode (a held floor, a
      # lane-change arrest in progress) must not resume stale after it
      self.turn_assist.reset()
      # a lane-change unwind armed before maneuver mode must not resume stale after it
      self.lane_change_smoothing.reset()
      return new_desired_curvature, 1.0
    CS = sm['carState']
    model_v2 = sm['modelV2']
    new_desired_curvature = self.turn_assist.update(CS, lat_active, model_v2, new_desired_curvature, current_curvature)
    jerk_factor = self.lane_change_smoothing.update(CS, model_v2, lat_active, new_desired_curvature, prev_desired_curvature)
    return new_desired_curvature, jerk_factor

  def note_lat_active(self, lat_active: bool) -> None:
    self._inactive_frames = 0 if lat_active else self._inactive_frames + 1

  @staticmethod
  def get_lead_data(_lead, src: log.RadarState.LeadData) -> None:
    _lead.dRel = src.dRel
    _lead.yRel = src.yRel
    _lead.vRel = src.vRel
    _lead.aRel = src.deprecated.aRel
    _lead.vLead = src.vLead
    _lead.dPath = src.deprecated.dPath
    _lead.vLat = src.deprecated.vLat
    _lead.vLeadK = src.vLeadK
    _lead.aLeadK = src.aLeadK
    _lead.fcw = src.deprecated.fcw
    _lead.status = src.present
    _lead.aLeadTau = src.aLeadTau
    _lead.modelProb = src.modelProb
    _lead.radar = src.radar
    _lead.radarTrackId = src.radarTrackId

  def state_control_ext(self, sm: messaging.SubMaster) -> custom.CarControlSP:
    CC_SP = custom.CarControlSP.new_message()

    # Once per drive: the exact settings this log was produced under
    if self._param_dump_pending:
      CC_SP.params = build_param_dump(self.params)
      self._param_dump_pending = False
    CC_SP.zoompilot.lateralExtension = lateral_extension_of(getattr(self, "LaC", None))
    CC_SP.torqueV2Mode = torque_v2_mode_of(getattr(self, "LaC", None))

    self.get_lead_data(CC_SP.leadOne, sm['radarState'].leadOne)
    self.get_lead_data(CC_SP.leadTwo, sm['radarState'].leadTwo)

    # MADS state
    mads_src = sm['selfdriveStateSP'].mads
    CC_SP.mads.state = mads_src.state
    CC_SP.mads.enabled = mads_src.enabled
    CC_SP.mads.active = mads_src.active
    CC_SP.mads.available = mads_src.available

    # ICBM state
    icbm_src = sm['selfdriveStateSP'].intelligentCruiseButtonManagement
    CC_SP.intelligentCruiseButtonManagement.state = icbm_src.state
    CC_SP.intelligentCruiseButtonManagement.sendButton = icbm_src.sendButton
    CC_SP.intelligentCruiseButtonManagement.vTarget = icbm_src.vTarget

    # lateral assist telemetry, for offline validation of the turn hold and pace clamp
    CC_SP.turnAssist.holdCurvature = float(self.turn_assist.hold)
    CC_SP.turnAssist.leadCurvature = float(self.turn_assist.lead_applied)
    # lane-change pace clamp telemetry, for offline validation
    CC_SP.zoompilot.laneChangeSmoothing.jerkFactor = float(self.lane_change_smoothing.jerk_factor)

    return CC_SP

  @staticmethod
  def publish_ext(CC_SP: custom.CarControlSP, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    cc_sp_send = messaging.new_message('carControlSP')
    cc_sp_send.valid = sm['carState'].canValid
    cc_sp_send.carControlSP = CC_SP

    pm.send('carControlSP', cc_sp_send)

  def run_ext(self, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    CC_SP = self.state_control_ext(sm)
    self.publish_ext(CC_SP, sm, pm)
    self.reclassify_steer_limit(sm)

    # Speed-dependent torque: apply per-bin learned values to the lateral controller
    if (self.CP.lateralTuning.which() == 'torque'
        and sm.updated.get('lateralTorqueParameters', False)
        and sm.all_checks(['lateralTorqueParameters'])):
      tp = sm['lateralTorqueParameters']
      # torqued_ext publishes the bins beside every upstream message on the fork service;
      # one that has not checked out counts as no bins
      tp_sp = sm[LIVE_TORQUE_PARAMETERS_SP_SERVICE] if sm.all_checks([LIVE_TORQUE_PARAMETERS_SP_SERVICE]) else None
      # both sizes' controllers, so the idle one holds current bins when it takes over.
      # handles activation AND deactivation: useParams off or empty bins de-assert
      for lac in self._lacs:
        lac.extension.update_speed_dep_torque(tp, tp_sp)

    self.select_lateral_control(sm)

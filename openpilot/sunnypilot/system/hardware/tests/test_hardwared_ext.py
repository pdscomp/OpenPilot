"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from contextlib import nullcontext

import pytest

from openpilot.common.params import ParamKeyFlag
import openpilot.sunnypilot.selfdrive.car.stock_ecu_handback as handback_protocol
from openpilot.sunnypilot.selfdrive.car.stock_ecu_handback import (HANDBACK_WAIT_T, REQUEST_KEY,
                                                                   StockEcuHandBackServer)
from openpilot.sunnypilot.selfdrive.car.tests.fakes import FakeClock, FakeParams, answer
from openpilot.sunnypilot.system.hardware.hardwared_ext import HardwaredExt


@pytest.fixture(autouse=True)
def fake_lock(monkeypatch):
  # these are pure-state dictionary tests; the real flock is covered in test_stock_ecu_handback
  monkeypatch.setattr(handback_protocol, "_request_lock", lambda params: nullcontext())


def _ext(**values):
  params = FakeParams(**values)
  ext = HardwaredExt(params)
  clock = FakeClock()
  ext.handback.now = clock
  return ext, params, clock


class TestOnroadCycle:
  def test_clears_onroad_transition_params(self):
    params = FakeParams()
    HardwaredExt(params).prepare_onroad_entry()
    assert params.cleared == [ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION]

  def test_nothing_requested(self):
    ext, params, _ = _ext()
    assert not ext.update(started=True)
    assert params.get(REQUEST_KEY) is None

  def test_offroad_cycle_needs_no_handback(self):
    ext, params, _ = _ext(OnroadCycleRequested=True)
    assert ext.update(started=False)
    assert not params.get_bool("OnroadCycleRequested")
    assert params.get(REQUEST_KEY) is None
    assert params.cleared == [ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION]

  def test_onroad_cycle_waits_for_the_handback(self):
    ext, params, _ = _ext(OnroadCycleRequested=True)
    assert not ext.update(started=True)
    assert params.get(REQUEST_KEY)["id"] == 1
    assert sorted(params.get(REQUEST_KEY)["waiters"].values()) == ["hardwared"]
    assert params.get_bool("OnroadCycleRequested")  # still pending, not consumed
    assert not ext.update(started=True)
    answer(params, "restored")
    assert ext.update(started=True)
    assert not params.get_bool("OnroadCycleRequested")
    assert params.cleared == [ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION]

  def test_onroad_cycle_gives_up_waiting_with_no_card(self):
    ext, params, clock = _ext(OnroadCycleRequested=True)
    assert not ext.update(started=True)
    clock.t = HANDBACK_WAIT_T + 1
    assert ext.update(started=True)
    assert not params.get_bool("OnroadCycleRequested")

  def test_failed_handback_does_not_hold_the_cycle(self):
    # a cycle has no cancel control, so like a reboot it proceeds on any answer
    ext, params, _ = _ext(OnroadCycleRequested=True)
    ext.update(started=True)
    answer(params, "failed")
    assert ext.update(started=True)
    assert not params.get_bool("OnroadCycleRequested")


class TestForcedOffroad:
  def test_request_is_applied_at_once_when_offroad(self):
    ext, params, _ = _ext(OffroadModeRequested=True)
    assert not ext.update(started=False)  # not a cycle
    assert params.get_bool("OffroadMode")
    assert params.get_bool("OffroadModeRequested")  # the preference stays; it is the state

  def test_entering_waits_for_the_handback_when_onroad(self):
    ext, params, _ = _ext(OffroadModeRequested=True)
    assert not ext.update(started=True)
    assert not params.get_bool("OffroadMode")
    assert params.get(REQUEST_KEY)["id"] == 1
    answer(params, "restored")
    assert not ext.update(started=True)
    assert params.get_bool("OffroadMode")
    assert params.cleared == []

  def test_entering_holds_on_a_failed_handback(self):
    # the radar never came back: stopping now leaves the camera without radar frames, so the
    # request stays open and visible; the user withdraws it or a late recovery completes it
    ext, params, clock = _ext(OffroadModeRequested=True)
    ext.update(started=True)
    answer(params, "failed")
    clock.t = HANDBACK_WAIT_T * 2
    for _ in range(5):
      assert not ext.update(started=True)
    assert ext.handback.failed and not params.get_bool("OffroadMode")
    answer(params, "restored")
    ext.update(started=True)
    assert params.get_bool("OffroadMode")

  def test_withdrawn_request_resets_the_wait(self):
    ext, params, clock = _ext(OffroadModeRequested=True)
    ext.update(started=True)
    params.put_bool("OffroadModeRequested", False)
    ext.update(started=True)
    assert not ext.handback.pending
    assert params.get(REQUEST_KEY) is None
    assert not params.get_bool("OffroadMode")

  def test_exit_clears_the_old_session_before_pandad_sees_ignition(self):
    # nothing cleared CarParams / ControlsReady while the device sat forced offroad with the
    # ignition on: pandad would apply the old safety and open the relay seconds before controls
    ext, params, _ = _ext(OffroadMode=True, OffroadModeRequested=False)
    assert not ext.update(started=False)
    assert params.cleared == [ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION]
    assert not params.get_bool("OffroadMode")
    assert params.get(REQUEST_KEY) is None  # no hand-back on the way out: nothing was taken over

  def test_steady_states_do_nothing(self):
    for offroad in (True, False):
      ext, params, _ = _ext(OffroadMode=offroad, OffroadModeRequested=offroad)
      assert not ext.update(started=not offroad)
      assert params.cleared == [] and params.get(REQUEST_KEY) is None
      assert params.get_bool("OffroadMode") == offroad

  def test_alpha_edits_while_forced_offroad_touch_nothing(self):
    ext, params, _ = _ext(OffroadMode=True, OffroadModeRequested=True, AlphaLongitudinalEnabled=True)
    for _ in range(5):
      params.put_bool("AlphaLongitudinalEnabled", not params.get_bool("AlphaLongitudinalEnabled"))
      assert not ext.update(started=False)
    assert params.cleared == [] and params.get(REQUEST_KEY) is None and params.get_bool("OffroadMode")


class TestWithdrawalAfterAcceptance:
  """A forced-offroad request withdrawn after card claimed the hand-back cannot cancel the
  in-flight restoration: the record stands, hardwared drains it, then cycles onroad once."""

  def _accepted_request(self):
    ext, params, clock = _ext(OffroadModeRequested=True)
    assert not ext.update(started=True)
    server = StockEcuHandBackServer(params)
    server.update_params()
    assert params.get(REQUEST_KEY)["accepted"]
    params.put_bool("OffroadModeRequested", False)
    return ext, params, clock, server

  def test_drain_waits_for_restore_then_cycles_once(self):
    ext, params, clock, server = self._accepted_request()
    for _ in range(3):
      assert not ext.update(started=True)
      assert not params.get_bool("OnroadCycleRequested")
      assert not params.get_bool("OffroadMode")  # no offroad grant without intent
    assert params.get(REQUEST_KEY) is not None  # the record stands for card
    answer(params, "restored")
    assert not ext.update(started=True)
    assert params.get_bool("OnroadCycleRequested")
    for _ in range(3):  # exactly one cycle request, then the normal gated path owns it
      assert not ext.update(started=True)

  def test_drain_failure_stays_onroad_and_late_recovery_cycles(self):
    ext, params, clock, server = self._accepted_request()
    ext.update(started=True)
    answer(params, "failed")
    clock.t = HANDBACK_WAIT_T * 2
    for _ in range(5):
      assert not ext.update(started=True)
      assert not params.get_bool("OnroadCycleRequested")
      assert not params.get_bool("OffroadMode")
    assert ext.drain_failed
    answer(params, "restored")
    assert not ext.update(started=True)
    assert params.get_bool("OnroadCycleRequested")

  def test_re_requested_offroad_rejoins_the_live_generation(self):
    ext, params, clock, server = self._accepted_request()
    ext.update(started=True)  # withdraws into the drain
    request_id = params.get(REQUEST_KEY)["id"]
    params.put_bool("OffroadModeRequested", True)
    assert not ext.update(started=True)  # joins the same hand-back instead of starting over
    assert params.get(REQUEST_KEY)["id"] == request_id
    assert not ext.draining
    answer(params, "restored")
    assert not ext.update(started=True)
    assert params.get_bool("OffroadMode")

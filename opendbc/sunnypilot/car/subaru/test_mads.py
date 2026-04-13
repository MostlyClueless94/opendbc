from opendbc.car import structs
from opendbc.sunnypilot.car.subaru.mads import (
  MadsCarState,
  SUBARU_LKAS_OFF_STATE,
  SUBARU_MADS_LKAS_BUTTON_STATE,
  SUBARU_STOCK_LKAS_ACTIVE_STATES,
)


ButtonType = structs.CarState.ButtonEvent.Type


def _lkas_events(cur_btn: int, prev_btn: int):
  return MadsCarState.create_lkas_button_events(cur_btn, prev_btn, {SUBARU_MADS_LKAS_BUTTON_STATE: ButtonType.lkas})


def _assert_lkas_event(cur_btn: int, prev_btn: int):
  events = _lkas_events(cur_btn, prev_btn)
  assert len(events) == 1
  assert events[0].pressed
  assert events[0].type == ButtonType.lkas


def test_lkas_ready_toggle_creates_subipilot_mads_button_event():
  for cur_btn, prev_btn in (
    (SUBARU_MADS_LKAS_BUTTON_STATE, SUBARU_LKAS_OFF_STATE),
    (SUBARU_LKAS_OFF_STATE, SUBARU_MADS_LKAS_BUTTON_STATE),
  ):
    _assert_lkas_event(cur_btn, prev_btn)


def test_stock_lkas_active_states_do_not_create_mads_button_events():
  for stock_state in (2, 3):
    for previous_state in range(4):
      assert _lkas_events(stock_state, previous_state) == []


def test_stock_lkas_active_clear_creates_subipilot_mads_button_event():
  for stock_state in SUBARU_STOCK_LKAS_ACTIVE_STATES:
    for cur_btn in (SUBARU_LKAS_OFF_STATE, SUBARU_MADS_LKAS_BUTTON_STATE):
      _assert_lkas_event(cur_btn, stock_state)


def test_repeated_lkas_ready_state_does_not_create_mads_button_event():
  assert _lkas_events(SUBARU_MADS_LKAS_BUTTON_STATE, SUBARU_MADS_LKAS_BUTTON_STATE) == []


def test_lkas_state_transitions_that_are_not_physical_button_pulses_do_not_create_events():
  for cur_btn, prev_btn in (
    (SUBARU_LKAS_OFF_STATE, SUBARU_LKAS_OFF_STATE),
    (SUBARU_MADS_LKAS_BUTTON_STATE, SUBARU_MADS_LKAS_BUTTON_STATE),
    (2, 2),
    (3, 3),
    (2, SUBARU_MADS_LKAS_BUTTON_STATE),
    (3, SUBARU_MADS_LKAS_BUTTON_STATE),
  ):
    assert _lkas_events(cur_btn, prev_btn) == []

from opendbc.car import structs
from opendbc.sunnypilot.car.subaru.mads import MadsCarState, SUBARU_MADS_LKAS_BUTTON_STATE, SUBARU_STOCK_LKAS_ACTIVE_STATES


ButtonType = structs.CarState.ButtonEvent.Type


def _lkas_events(cur_btn: int, prev_btn: int):
  return MadsCarState.create_lkas_button_events(cur_btn, prev_btn, {SUBARU_MADS_LKAS_BUTTON_STATE: ButtonType.lkas})


def test_lkas_ready_state_creates_subipilot_mads_button_event():
  events = _lkas_events(SUBARU_MADS_LKAS_BUTTON_STATE, 0)

  assert len(events) == 1
  assert events[0].pressed
  assert events[0].type == ButtonType.lkas


def test_stock_lkas_active_states_do_not_create_mads_button_events():
  for stock_state in (2, 3):
    for previous_state in range(4):
      assert _lkas_events(stock_state, previous_state) == []


def test_stock_lkas_active_to_ready_creates_subipilot_mads_button_event():
  for stock_state in SUBARU_STOCK_LKAS_ACTIVE_STATES:
    events = _lkas_events(SUBARU_MADS_LKAS_BUTTON_STATE, stock_state)

    assert len(events) == 1
    assert events[0].pressed
    assert events[0].type == ButtonType.lkas


def test_repeated_lkas_ready_state_does_not_create_mads_button_event():
  assert _lkas_events(SUBARU_MADS_LKAS_BUTTON_STATE, SUBARU_MADS_LKAS_BUTTON_STATE) == []


def test_lkas_state_release_does_not_create_unknown_button_events():
  for previous_state in range(1, 4):
    assert _lkas_events(0, previous_state) == []

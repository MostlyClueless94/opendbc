"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from enum import StrEnum
from opendbc.car import Bus, structs

from opendbc.car.subaru.values import SubaruFlags
from opendbc.sunnypilot.mads_base import MadsCarStateBase
from opendbc.can.parser import CANParser

ButtonType = structs.CarState.ButtonEvent.Type
SUBARU_MADS_LKAS_BUTTON_STATE = 1
SUBARU_STOCK_LKAS_ACTIVE_STATES = (2, 3)


class MadsCarState(MadsCarStateBase):
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    super().__init__(CP, CP_SP)

  @staticmethod
  def create_lkas_button_events(cur_btn: int, prev_btn: int,
                                buttons_dict: dict[int, structs.CarState.ButtonEvent.Type]) -> list[structs.CarState.ButtonEvent]:
    events: list[structs.CarState.ButtonEvent] = []

    # State 2 is stock LKAS active. Ignore it so stock LKAS can be turned off without cycling MADS.
    if cur_btn == prev_btn or cur_btn != SUBARU_MADS_LKAS_BUTTON_STATE or prev_btn in SUBARU_STOCK_LKAS_ACTIVE_STATES:
      return events

    events.append(structs.CarState.ButtonEvent(pressed=True,
                                               type=buttons_dict.get(cur_btn, ButtonType.unknown)))
    return events

  def update_mads(self, ret: structs.CarState, can_parsers: dict[StrEnum, CANParser]) -> None:
    cp_cam = can_parsers[Bus.cam]

    self.prev_lkas_button = self.lkas_button
    if not self.CP.flags & SubaruFlags.PREGLOBAL:
      self.lkas_button = cp_cam.vl["ES_LKAS_State"]["LKAS_Dash_State"]

    ret.buttonEvents = self.create_lkas_button_events(self.lkas_button, self.prev_lkas_button, {1: ButtonType.lkas})

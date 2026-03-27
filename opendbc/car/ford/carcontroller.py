import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_hysteresis, structs
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import CAR, CarControllerParams, FordFlags
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX
from opendbc.car.lateral import ISO_LATERAL_ACCEL, apply_std_steer_angle_limits

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

# CAN FD limits:
# Limit to average banked road since safety doesn't have the roll
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees, 6% superelevation. higher actual roll raises lateral acceleration
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~2.4 m/s^2

F150_MODEL_LOOKAHEAD = 0.2
F150_CURVATURE_BLEND = 0.4
F150_CURVATURE_RATE_WINDOW = 0.3
F150_MODEL_DT = DT_CTRL * CarControllerParams.STEER_STEP
F150_PATH_OFFSET_MAX = 1.0
F150_PATH_ANGLE_MAX = 0.25
F150_CURVATURE_RATE_MAX = 0.001023
F150_PATH_OFFSET_GAIN = 0.03
F150_LANE_CHANGE_FACTOR_BP = [4.4, 40.23]
F150_LANE_CHANGE_FACTOR_V = [0.95, 0.85]
F150_PATH_OFFSET_RATE_BP = [5.0, 15.0, 25.0]
F150_PATH_OFFSET_RATE_V = [0.05, 0.025, 0.01]
F150_PATH_ANGLE_RATE_BP = [5.0, 15.0, 25.0]
F150_PATH_ANGLE_RATE_V = [0.003, 0.0015, 0.002]
F150_CURVATURE_RATE_RATE_BP = [5.0, 15.0, 25.0]
F150_CURVATURE_RATE_RATE_V = [0.0002, 0.00015, 0.0001]
F150_PATH_ANGLE_SPEED_BP = [0.0, 9.0, 15.0]
F150_PATH_ANGLE_SPEED_V = [0.0, 0.0, 1.0]
F150_LANELINE_CONFIDENCE_BP = [0.6, 0.8]
F150_LANELINE_WIDTH_BP = [3.75, 4.25]
F150_LANELINE_WIDTH_V = [0.81, 0.59]
F150_PATH_ANGLE_KP = 0.25
F150_PATH_ANGLE_KI = 0.05
F150_PATH_ANGLE_RESET_DURATION_FRAMES = int(round(1.5 / F150_MODEL_DT))
F150_POST_LANE_CHANGE_DURATION_FRAMES = 160
F150_POST_LANE_CHANGE_PATH_ANGLE_DELTA = 0.00125
F150_POST_LANE_CHANGE_PATH_OFFSET_DELTA = 0.00125
F150_POST_LANE_CHANGE_CURVATURE_RATE_DELTA = 0.0001


@dataclass
class FordCANFDPathSignals:
  curvature: float
  path_offset: float
  path_angle: float
  curvature_rate: float
  ramp_type: int
  precision_type: int


def _rate_limit_signal(value: float, last_value: float, max_delta: float) -> float:
  return float(np.clip(value, last_value - max_delta, last_value + max_delta))


def _interp_model_signal(values, times, lookahead: float) -> float:
  values_list = list(values)
  if len(values_list) == 0:
    return 0.0

  if len(values_list) == 1:
    return float(values_list[0])

  times_list = list(times)
  if len(times_list) != len(values_list):
    times_list = [F150_MODEL_DT * i for i in range(len(values_list))]

  clamped_lookahead = float(np.clip(lookahead, times_list[0], times_list[-1]))
  return float(np.interp(clamped_lookahead, times_list, values_list))


def compute_f150_blended_path_offset(model) -> float:
  position = getattr(model, "position", None)
  if position is None:
    return 0.0

  path_offset_position = _interp_model_signal(getattr(position, "y", ()), getattr(position, "t", ()), F150_MODEL_LOOKAHEAD)
  lane_lines = getattr(model, "laneLines", ())
  lane_line_probs = list(getattr(model, "laneLineProbs", ()))
  if len(lane_lines) < 3 or len(lane_line_probs) < 3:
    return path_offset_position

  left_lane = list(getattr(lane_lines[1], "y", ()))
  right_lane = list(getattr(lane_lines[2], "y", ()))
  if len(left_lane) == 0 or len(right_lane) == 0:
    return path_offset_position

  lane_center = (left_lane[0] + right_lane[0]) / 2.0
  lane_width = right_lane[0] - left_lane[0]
  lane_width_tolerance = float(np.interp(lane_width, F150_LANELINE_WIDTH_BP, F150_LANELINE_WIDTH_V))
  laneline_confidence = min(float(lane_line_probs[1]), float(lane_line_probs[2]), lane_width_tolerance)
  laneline_path_offset_scale = float(np.interp(laneline_confidence, F150_LANELINE_CONFIDENCE_BP, [0.0, 1.0]))

  return float((path_offset_position * (1.0 - laneline_path_offset_scale)) + (lane_center * laneline_path_offset_scale))


def compute_f150_path_angle_from_offset(path_offset: float, v_ego_raw: float, steering_pressed: bool, reset_requested: bool,
                                        last_path_angle: float, path_angle_integral: float,
                                        path_angle_reset_counter: int) -> tuple[float, float, int]:
  if steering_pressed:
    path_angle_reset_counter += 1
  else:
    path_angle_reset_counter = 0

  if path_angle_reset_counter > F150_PATH_ANGLE_RESET_DURATION_FRAMES:
    path_angle_integral = 0.0

  speed_factor = float(np.interp(v_ego_raw, F150_PATH_ANGLE_SPEED_BP, F150_PATH_ANGLE_SPEED_V))
  path_offset_error = 0.0 if reset_requested else (path_offset * F150_PATH_OFFSET_GAIN * speed_factor)
  path_angle_integral = float(np.clip(path_angle_integral + (F150_PATH_ANGLE_KI * F150_MODEL_DT * path_offset_error),
                                      -F150_PATH_ANGLE_MAX, F150_PATH_ANGLE_MAX))
  path_angle = (F150_PATH_ANGLE_KP * path_offset_error) + path_angle_integral
  path_angle = _rate_limit_signal(path_angle, last_path_angle,
                                  float(np.interp(abs(v_ego_raw), F150_PATH_ANGLE_RATE_BP, F150_PATH_ANGLE_RATE_V)))
  path_angle = float(np.clip(path_angle, -F150_PATH_ANGLE_MAX, F150_PATH_ANGLE_MAX))

  if reset_requested:
    path_angle = 0.0

  return path_angle, path_angle_integral, path_angle_reset_counter


def apply_f150_post_lane_change_transition(lane_change_active: bool, lane_change_last: bool, post_lane_change_active: bool,
                                           post_lane_change_timer: int, path_angle: float, path_offset: float,
                                           desired_curvature_rate: float, pre_lane_change_values: dict) -> tuple[float, float, float, bool, bool, int, dict]:
  if lane_change_last and not lane_change_active:
    post_lane_change_active = True
    post_lane_change_timer = 0
    pre_lane_change_values = {
      "path_angle": 0.0,
      "path_offset": 0.0,
      "desired_curvature_rate": 0.0,
    }

  lane_change_last = lane_change_active

  if post_lane_change_active:
    post_lane_change_timer += 1
    path_angle = _rate_limit_signal(path_angle, pre_lane_change_values["path_angle"], F150_POST_LANE_CHANGE_PATH_ANGLE_DELTA)
    path_offset = _rate_limit_signal(path_offset, pre_lane_change_values["path_offset"], F150_POST_LANE_CHANGE_PATH_OFFSET_DELTA)
    desired_curvature_rate = _rate_limit_signal(desired_curvature_rate, pre_lane_change_values["desired_curvature_rate"],
                                                F150_POST_LANE_CHANGE_CURVATURE_RATE_DELTA)
    pre_lane_change_values = {
      "path_angle": path_angle,
      "path_offset": path_offset,
      "desired_curvature_rate": desired_curvature_rate,
    }

    if post_lane_change_timer >= F150_POST_LANE_CHANGE_DURATION_FRAMES:
      post_lane_change_active = False

  return path_angle, path_offset, desired_curvature_rate, lane_change_last, post_lane_change_active, post_lane_change_timer, pre_lane_change_values


def apply_f150_post_reset_ramp(requested_curvature: float, apply_curvature_last: float, v_ego_raw: float,
                               lat_active: bool) -> tuple[float, bool]:
  apply_curvature = apply_std_steer_angle_limits(requested_curvature, apply_curvature_last, v_ego_raw, 0.0,
                                                 lat_active, CarControllerParams.ANGLE_LIMITS)
  curvature_error = abs(requested_curvature - apply_curvature)
  curvature_threshold = max(abs(requested_curvature) * 0.1, 0.001)
  return apply_curvature, curvature_error >= curvature_threshold


def compute_f150_canfd_path_signals(model, v_ego_raw: float, desired_curvature: float, lane_change_active: bool,
                                    lane_change_direction: int, steering_pressed: bool, steering_angle_deg: float,
                                    curvature_history: deque[float], last_path_offset: float, last_path_angle: float,
                                    last_curvature_rate: float, path_angle_integral: float,
                                    path_angle_reset_counter: int) -> tuple[FordCANFDPathSignals, float, int]:
  requested_curvature = desired_curvature
  path_offset = 0.0
  path_angle = 0.0
  desired_curvature_rate = 0.0
  precision_type = 1
  ramp_type = 2

  if model is None:
    return FordCANFDPathSignals(requested_curvature, path_offset, path_angle, desired_curvature_rate, ramp_type, precision_type), \
      path_angle_integral, path_angle_reset_counter

  orientation_rate = getattr(model, "orientationRate", None)
  if orientation_rate is None:
    return FordCANFDPathSignals(requested_curvature, path_offset, path_angle, desired_curvature_rate, ramp_type, precision_type), \
      path_angle_integral, path_angle_reset_counter

  predicted_curvature = _interp_model_signal(getattr(orientation_rate, "z", ()), getattr(orientation_rate, "t", ()),
                                             F150_MODEL_LOOKAHEAD) / max(v_ego_raw, 0.1)
  requested_curvature = ((1.0 - F150_CURVATURE_BLEND) * desired_curvature) + (F150_CURVATURE_BLEND * predicted_curvature)

  if lane_change_active:
    lane_change_factor = float(np.interp(v_ego_raw, F150_LANE_CHANGE_FACTOR_BP, F150_LANE_CHANGE_FACTOR_V))
    if lane_change_direction == 1 and requested_curvature < 0.0:
      requested_curvature *= lane_change_factor
    elif lane_change_direction == 2 and requested_curvature > 0.0:
      requested_curvature *= lane_change_factor
    precision_type = 0

  human_turn = steering_pressed and abs(steering_angle_deg) > 45.0
  reset_requested = human_turn or v_ego_raw < 0.1
  if reset_requested:
    requested_curvature = 0.0
    ramp_type = 3

  curvature_history.append(predicted_curvature)
  if len(curvature_history) > 1:
    history_dt = F150_CURVATURE_RATE_WINDOW if len(curvature_history) == curvature_history.maxlen else (len(curvature_history) - 1) * F150_MODEL_DT
    desired_curvature_rate = (curvature_history[-1] - curvature_history[0]) / max(history_dt * max(v_ego_raw, 0.1), F150_MODEL_DT)

  raw_path_offset = compute_f150_blended_path_offset(model)
  if lane_change_active or reset_requested:
    raw_path_offset = 0.0

  path_offset = _rate_limit_signal(raw_path_offset, last_path_offset,
                                   float(np.interp(v_ego_raw, F150_PATH_OFFSET_RATE_BP, F150_PATH_OFFSET_RATE_V)))
  path_angle, path_angle_integral, path_angle_reset_counter = compute_f150_path_angle_from_offset(
    path_offset, v_ego_raw, steering_pressed, reset_requested, last_path_angle, path_angle_integral, path_angle_reset_counter,
  )
  desired_curvature_rate = _rate_limit_signal(desired_curvature_rate, last_curvature_rate,
                                              float(np.interp(v_ego_raw, F150_CURVATURE_RATE_RATE_BP, F150_CURVATURE_RATE_RATE_V)))

  if lane_change_active or reset_requested:
    path_offset = 0.0
    path_angle = 0.0
    desired_curvature_rate = 0.0

  path_offset = float(np.clip(path_offset, -F150_PATH_OFFSET_MAX, F150_PATH_OFFSET_MAX))
  path_angle = float(np.clip(path_angle, -F150_PATH_ANGLE_MAX, F150_PATH_ANGLE_MAX))
  desired_curvature_rate = float(np.clip(desired_curvature_rate, -F150_CURVATURE_RATE_MAX, F150_CURVATURE_RATE_MAX))

  return FordCANFDPathSignals(requested_curvature, path_offset, path_angle, desired_curvature_rate, ramp_type, precision_type), \
    path_angle_integral, path_angle_reset_counter


def anti_overshoot(apply_curvature, apply_curvature_last, v_ego):
  diff = 0.1
  tau = 5  # 5s smooths over the overshoot
  dt = DT_CTRL * CarControllerParams.STEER_STEP
  alpha = 1 - np.exp(-dt / tau)

  lataccel = apply_curvature * (v_ego ** 2)
  last_lataccel = apply_curvature_last * (v_ego ** 2)
  last_lataccel = apply_hysteresis(lataccel, last_lataccel, diff)
  last_lataccel = alpha * lataccel + (1 - alpha) * last_lataccel

  output_curvature = last_lataccel / (max(v_ego, 1) ** 2)

  return float(np.interp(v_ego, [5, 10], [apply_curvature, output_curvature]))


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, steering_angle, lat_active, CP):
  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = np.clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                              current_curvature + CarControllerParams.CURVATURE_ERROR)

  # Curvature rate limit after driver torque limit
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, steering_angle, lat_active, CarControllerParams.ANGLE_LIMITS)

  # Ford Q4/CAN FD has more torque available compared to Q3/CAN so we limit it based on lateral acceleration.
  # Safety is not aware of the road roll so we subtract a conservative amount at all times
  if CP.flags & FordFlags.CANFD:
    # Limit curvature to conservative max lateral acceleration
    curvature_accel_limit = MAX_LATERAL_ACCEL / (max(v_ego_raw, 1) ** 2)
    apply_curvature = float(np.clip(apply_curvature, -curvature_accel_limit, curvature_accel_limit))

  return apply_curvature


def apply_creep_compensation(accel: float, v_ego: float) -> float:
  creep_accel = np.interp(v_ego, [1., 3.], [0.6, 0.])
  creep_accel = np.interp(accel, [0., 0.2], [creep_accel, 0.])
  accel -= creep_accel
  return float(accel)


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.CAN = fordcan.CanBus(CP)
    self.is_f150_canfd = CP.carFingerprint == CAR.FORD_F_150_MK14 and bool(CP.flags & FordFlags.CANFD)

    self.apply_curvature_last = 0
    self.anti_overshoot_curvature_last = 0
    self.path_offset_last = 0.0
    self.path_angle_last = 0.0
    self.curvature_rate_last = 0.0
    self.path_angle_integral = 0.0
    self.path_angle_reset_counter = 0
    self.post_reset_ramp_active = False
    self.reset_steering_last = False
    self.post_lane_change_active = False
    self.post_lane_change_timer = 0
    self.lane_change_last = False
    self.pre_lane_change_values = {
      "path_angle": 0.0,
      "path_offset": 0.0,
      "desired_curvature_rate": 0.0,
    }
    self.predicted_curvature_window: deque[float] = deque(maxlen=max(2, int(round(F150_CURVATURE_RATE_WINDOW / F150_MODEL_DT))))
    self.accel = 0.0
    self.gas = 0.0
    self.brake_request = False
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0
    self._model_sm = None
    self._latest_model = None

    if self.is_f150_canfd:
      try:
        import cereal.messaging as messaging

        self._model_sm = messaging.SubMaster(["modelV2"])
      except ImportError:
        self._model_sm = None

  def _update_model(self) -> None:
    if self._model_sm is None:
      return

    self._model_sm.update(0)
    if self._model_sm.updated["modelV2"]:
      self._latest_model = self._model_sm["modelV2"]

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []
    self._update_model()

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))

    ### lateral control ###
    # send steer msg at 20Hz
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      if self.is_f150_canfd:
        lane_change_active = False
        lane_change_direction = 0
        if self._latest_model is not None and hasattr(self._latest_model, "meta"):
          lane_change_active = getattr(self._latest_model.meta, "laneChangeState", 0) in (1, 2, 3)
          lane_change_direction = int(getattr(self._latest_model.meta, "laneChangeDirection", 0))

        ramp_type = 0
        precision_type = 1
        path_offset = 0.0
        path_angle = 0.0
        curvature_rate = 0.0
        requested_curvature = 0.0

        if CC.latActive:
          f150_signals, self.path_angle_integral, self.path_angle_reset_counter = compute_f150_canfd_path_signals(
            self._latest_model,
            CS.out.vEgoRaw,
            actuators.curvature,
            lane_change_active,
            lane_change_direction,
            CS.out.steeringPressed,
            CS.out.steeringAngleDeg,
            self.predicted_curvature_window,
            self.path_offset_last,
            self.path_angle_last,
            self.curvature_rate_last,
            self.path_angle_integral,
            self.path_angle_reset_counter,
          )
          requested_curvature = f150_signals.curvature
          path_offset = f150_signals.path_offset
          path_angle = f150_signals.path_angle
          curvature_rate = f150_signals.curvature_rate
          ramp_type = f150_signals.ramp_type
          precision_type = f150_signals.precision_type
        else:
          self.predicted_curvature_window.clear()
          self.path_angle_integral = 0.0
          self.path_angle_reset_counter = 0
          self.post_lane_change_active = False
          self.post_lane_change_timer = 0
          self.lane_change_last = False
          self.pre_lane_change_values = {
            "path_angle": 0.0,
            "path_offset": 0.0,
            "desired_curvature_rate": 0.0,
          }

        reset_requested = ramp_type == 3
        if reset_requested:
          self.post_reset_ramp_active = False
          self.predicted_curvature_window.clear()
          self.path_angle_integral = 0.0
          self.path_angle_reset_counter = 0
          self.post_lane_change_active = False
          self.post_lane_change_timer = 0
          self.pre_lane_change_values = {
            "path_angle": 0.0,
            "path_offset": 0.0,
            "desired_curvature_rate": 0.0,
          }
          self.lane_change_last = lane_change_active
        else:
          path_angle, path_offset, curvature_rate, self.lane_change_last, self.post_lane_change_active, \
            self.post_lane_change_timer, self.pre_lane_change_values = apply_f150_post_lane_change_transition(
              lane_change_active,
              self.lane_change_last,
              self.post_lane_change_active,
              self.post_lane_change_timer,
              path_angle,
              path_offset,
              curvature_rate,
              self.pre_lane_change_values,
            )

          if self.reset_steering_last:
            self.post_reset_ramp_active = True
            self.apply_curvature_last = 0.0
            self.anti_overshoot_curvature_last = 0.0

        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
        requested_curvature = apply_ford_curvature_limits(requested_curvature, self.apply_curvature_last,
                                                          current_curvature, CS.out.vEgoRaw, 0.,
                                                          CC.latActive, self.CP)

        if self.post_reset_ramp_active:
          self.apply_curvature_last, self.post_reset_ramp_active = apply_f150_post_reset_ramp(
            requested_curvature, self.apply_curvature_last, CS.out.vEgoRaw, CC.latActive,
          )
          path_offset = 0.0
          path_angle = 0.0
          curvature_rate = 0.0
        else:
          self.anti_overshoot_curvature_last = anti_overshoot(requested_curvature, self.anti_overshoot_curvature_last, CS.out.vEgoRaw)
          self.apply_curvature_last = apply_ford_curvature_limits(self.anti_overshoot_curvature_last, self.apply_curvature_last,
                                                                  current_curvature, CS.out.vEgoRaw, 0., CC.latActive, self.CP)

        self.reset_steering_last = reset_requested
        mode = 1 if CC.latActive else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode, -path_offset, -path_angle,
                                                     -self.apply_curvature_last, -curvature_rate, counter,
                                                     ramp_type=ramp_type, precision_type=precision_type))
        self.path_offset_last = path_offset
        self.path_angle_last = path_angle
        self.curvature_rate_last = curvature_rate
      else:
        # Bronco and some other cars consistently overshoot curv requests
        # Apply some deadzone + smoothing convergence to avoid oscillations
        if self.CP.carFingerprint in (CAR.FORD_BRONCO_SPORT_MK1, CAR.FORD_F_150_MK14):
          self.anti_overshoot_curvature_last = anti_overshoot(actuators.curvature, self.anti_overshoot_curvature_last, CS.out.vEgoRaw)
          apply_curvature = self.anti_overshoot_curvature_last
        else:
          apply_curvature = actuators.curvature

        # apply rate limits, curvature error limit, and clip to signal range
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
        self.apply_curvature_last = apply_ford_curvature_limits(apply_curvature, self.apply_curvature_last, current_curvature,
                                                                CS.out.vEgoRaw, 0., CC.latActive, self.CP)

        if self.CP.flags & FordFlags.CANFD:
          # TODO: extended mode
          # Ford uses four individual signals to dictate how to drive to the car. Curvature alone (limited to 0.02m/s^2)
          # can actuate the steering for a large portion of any lateral movements. However, in order to get further control on
          # steer actuation, the other three signals are necessary. Ford controls vehicles differently than most other makes.
          # A detailed explanation on ford control can be found here:
          # https://www.f150gen14.com/forum/threads/introducing-bluepilot-a-ford-specific-fork-for-comma3x-openpilot.24241/#post-457706
          mode = 1 if CC.latActive else 0
          counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
          can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode, 0., 0., -self.apply_curvature_last, 0., counter))
        else:
          can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive, 0., 0., -self.apply_curvature_last, 0.))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      accel = actuators.accel
      gas = accel

      if CC.longActive:
        # Compensate for engine creep at low speed.
        # Either the ABS does not account for engine creep, or the correction is very slow
        # TODO: verify this applies to EV/hybrid
        accel = apply_creep_compensation(accel, CS.out.vEgo)

        # The stock system has been seen rate limiting the brake accel to 5 m/s^3,
        # however even 3.5 m/s^3 causes some overshoot with a step response.
        accel = max(accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))

      accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      gas = float(np.clip(gas, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))

      # Both gas and accel are in m/s^2, accel is used solely for braking
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS

      # PCM applies pitch compensation to gas/accel, but we need to compensate for the brake/pre-charge bits
      accel_due_to_pitch = 0.0
      if len(CC.orientationNED) == 3:
        accel_due_to_pitch = math.sin(CC.orientationNED[1]) * ACCELERATION_DUE_TO_GRAVITY

      accel_pitch_compensated = accel + accel_due_to_pitch
      if accel_pitch_compensated > 0.3 or not CC.longActive:
        self.brake_request = False
      elif accel_pitch_compensated < 0.0:
        self.brake_request = True

      stopping = CC.actuators.longControlState == LongCtrlState.stopping
      # TODO: look into using the actuators packet to send the desired speed
      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, accel, stopping,
                                              self.brake_request, v_ego_kph=V_CRUISE_MAX))

      self.accel = accel
      self.gas = gas

    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # send lkas ui msg at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))

    # send acc ui msg at 5Hz or if ui state changes
    if hud_control.leadDistanceBars != self.lead_distance_bars_last:
      send_ui = True
      self.distance_bar_frame = self.frame

    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      show_distance_bars = self.frame - self.distance_bar_frame < 400
      can_sends.append(fordcan.create_acc_ui_msg(self.packer, self.CAN, self.CP, main_on, CC.latActive,
                                                 fcw_alert, CS.out.cruiseState.standstill, show_distance_bars,
                                                 hud_control, CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert
    self.lead_distance_bars_last = hud_control.leadDistanceBars

    new_actuators = actuators.as_builder()
    new_actuators.curvature = self.apply_curvature_last
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas

    self.frame += 1
    return new_actuators, can_sends

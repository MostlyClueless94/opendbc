import numpy as np
from openpilot.common.params import Params
from opendbc.can import CANPacker
from opendbc.car import Bus, make_tester_present_msg, structs
from opendbc.car.carlog import carlog
from opendbc.car.lateral import apply_center_deadzone, apply_driver_steer_torque_limits, apply_std_steer_angle_limits, common_fault_avoidance
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.subaru import subarucan
from opendbc.car.subaru.values import DBC, GLOBAL_ES_ADDR, CanBus, CarControllerParams, SubaruFlags

from opendbc.sunnypilot.car.subaru.stop_and_go import SnGCarController

# FIXME: These limits aren't exact. The real limit is more than likely over a larger time period and
# involves the total steering angle change rather than rate, but these limits work well for now
MAX_STEER_RATE = 25  # deg/s
MAX_STEER_RATE_FRAMES = 7  # tx control frames needed before torque can be cut
MADS_ONLY_MIN_SPEED = 0.44704  # m/s (1 mph)
MADS_ONLY_MAX_STEER_ANGLE = 120.0  # deg
ANGLE_DRIVER_OVERRIDE_HOLD_FRAMES = 10  # steering command frames (~200 ms with STEER_STEP=2)
ANGLE_DRIVER_OVERRIDE_RAMP_SPEED_MIN = 0
ANGLE_DRIVER_OVERRIDE_RAMP_SPEED_MAX = 6
ANGLE_DRIVER_OVERRIDE_RAMP_SPEED_DEFAULT = 4
ANGLE_DRIVER_OVERRIDE_RAMP_FRAMES = 36  # default Slow ramp (steering command frames, ~720 ms with STEER_STEP=2)
ANGLE_DRIVER_OVERRIDE_RAMP_FRAME_OPTIONS = [12, 18, 24, 30, 36, 42, 48]
ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_MIN = 0
ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_MAX = 6
ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_DEFAULT = 4
ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_EXPONENTS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5]
LOW_SPEED_SMOOTH_MAX_SPEED = 4.4704  # m/s (10 mph)
LOW_SPEED_SMOOTH_DEADBAND_MAX = 0.8  # deg at 0 mph
LOW_SPEED_SMOOTH_ALPHA_MIN = 0.35  # blend factor at 0 mph
SUBARU_TUNING_STRENGTH_MIN = -3
SUBARU_TUNING_STRENGTH_MAX = 4
SUBARU_TUNING_SCALE_POINTS = [SUBARU_TUNING_STRENGTH_MIN, 0, 3, SUBARU_TUNING_STRENGTH_MAX]
SUBARU_SMOOTHING_DEADBAND_SCALES = [0.70, 1.00, 1.35, 1.466]
SUBARU_SMOOTHING_ALPHA_SCALES = [1.20, 1.00, 0.80, 0.734]
SUBARU_CENTER_DAMPING_DEADBAND_SCALES = [0.70, 1.00, 1.45, 1.60]
SUBARU_CENTER_DAMPING_SIGN_FLIP_SCALES = [1.30, 1.00, 0.70, 0.60]
SUBARU_CENTER_DAMPING_ALPHA_SCALES = [1.20, 1.00, 0.75, 0.666]
LOW_SPEED_DELTA_DEADZONE_TARGET_MAX = 4.0  # deg, keep the experiment scoped near center
LOW_SPEED_DELTA_DEADZONE_STEER_MAX = 10.0  # deg, bypass real low-speed turns
LOW_SPEED_DELTA_DEADZONE_MAX = 2.0  # deg at 0 mph
LOW_SPEED_DELTA_DEADZONE_MIN = 0.75  # deg at 10 mph
LOW_SPEED_CENTER_DAMPING_TARGET_MAX = 3.0  # deg, only damp tiny near-center requests
LOW_SPEED_CENTER_DAMPING_STEER_MAX = 8.0  # deg, avoid muting real turns
LOW_SPEED_CENTER_DAMPING_DEADBAND_MAX = 0.8  # deg at 0 mph
LOW_SPEED_CENTER_DAMPING_DEADBAND_MIN = 0.25  # deg at 10 mph
LOW_SPEED_CENTER_DAMPING_SIGN_FLIP_DELTA_MIN = 0.2  # deg/frame at 0 mph
LOW_SPEED_CENTER_DAMPING_SIGN_FLIP_DELTA_MAX = 0.75  # deg/frame at 10 mph
LOW_SPEED_CENTER_DAMPING_ALPHA_MIN = 0.25  # first-order blend factor at 0 mph
LOW_SPEED_CENTER_DAMPING_ALPHA_MAX = 0.65  # first-order blend factor at 10 mph
LOW_SPEED_STRAIGHT_STABILITY_TARGET_MAX = 3.5  # deg, keep narrowly scoped to straight-ahead chatter
LOW_SPEED_STRAIGHT_STABILITY_STEER_MAX = 6.0  # deg, bypass real turning maneuvers
LOW_SPEED_STRAIGHT_CENTER_HOLD_TARGET_MAX = 1.0  # deg, hold near-center targets steady
LOW_SPEED_STRAIGHT_CENTER_HOLD_STEER_MAX = 2.5  # deg, measured wheel angle window for center hold
LOW_SPEED_STRAIGHT_SIGN_RELEASE_TARGET = 2.0  # deg, allow small opposite-sign moves only after persistence
LOW_SPEED_STRAIGHT_SIGN_RELEASE_FRAMES = 5  # 50 ms at 100 Hz
LOW_SPEED_STRAIGHT_SIGN_EPSILON = 0.05  # deg, ignore numerical noise around zero


class CarController(CarControllerBase, SnGCarController):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    SnGCarController.__init__(self, CP, CP_SP)
    self.apply_torque_last = 0
    self.apply_angle_last = 0

    self.cruise_button_prev = 0
    self.steer_rate_counter = 0
    self._debug_state = {}

    self.p = CarControllerParams(CP)
    self.packer = CANPacker(DBC[CP.carFingerprint][Bus.pt])
    self.params = Params()
    self.mc_subaru_smoothing_tune = False
    self.mc_subaru_smoothing_strength = 0
    self.mc_subaru_center_damping_strength = 0
    self.mc_subaru_manual_yield_resume_speed = ANGLE_DRIVER_OVERRIDE_RAMP_SPEED_DEFAULT
    self.mc_subaru_manual_yield_resume_softness = ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_DEFAULT
    self.angle_driver_override_hold_frames = 0
    self.angle_driver_override_ramp_frames = 0
    self.angle_driver_override_ramp_total_frames = ANGLE_DRIVER_OVERRIDE_RAMP_FRAMES
    self.angle_driver_override_ramp_start_angle = 0.0
    self.angle_driver_override_ramp_softness_exponent = ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_EXPONENTS[ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_DEFAULT]
    self.low_speed_straight_pending_direction = 0
    self.low_speed_straight_pending_frames = 0
    self._update_params()

  def _log_transition(self, key, value, message):
    if self._debug_state.get(key) != value:
      carlog.info(f"subaru[{self.CP.carFingerprint}] {message}")
      self._debug_state[key] = value

  def _get_int_param(self, key: str, default: int = 0) -> int:
    value = self.params.get(key, return_default=True)
    try:
      return int(value)
    except (TypeError, ValueError):
      return default

  @staticmethod
  def _get_strength_scale(strength: int, values: list[float]) -> float:
    return float(np.interp(strength, SUBARU_TUNING_SCALE_POINTS, values))

  @staticmethod
  def _get_resume_speed_frames(speed_setting: int) -> int:
    return ANGLE_DRIVER_OVERRIDE_RAMP_FRAME_OPTIONS[int(np.clip(
      speed_setting,
      ANGLE_DRIVER_OVERRIDE_RAMP_SPEED_MIN,
      ANGLE_DRIVER_OVERRIDE_RAMP_SPEED_MAX,
    ))]

  @staticmethod
  def _get_resume_softness_exponent(softness_setting: int) -> float:
    return ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_EXPONENTS[int(np.clip(
      softness_setting,
      ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_MIN,
      ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_MAX,
    ))]

  def _update_params(self):
    self.mc_subaru_smoothing_tune = self.params.get_bool("MCSubaruSmoothingTune")
    self.mc_subaru_smoothing_strength = int(np.clip(
      self._get_int_param("MCSubaruSmoothingStrength"),
      SUBARU_TUNING_STRENGTH_MIN,
      SUBARU_TUNING_STRENGTH_MAX,
    ))
    self.mc_subaru_center_damping_strength = int(np.clip(
      self._get_int_param("MCSubaruCenterDampingStrength"),
      SUBARU_TUNING_STRENGTH_MIN,
      SUBARU_TUNING_STRENGTH_MAX,
    ))
    self.mc_subaru_manual_yield_resume_speed = int(np.clip(
      self._get_int_param("MCSubaruManualYieldResumeSpeed", ANGLE_DRIVER_OVERRIDE_RAMP_SPEED_DEFAULT),
      ANGLE_DRIVER_OVERRIDE_RAMP_SPEED_MIN,
      ANGLE_DRIVER_OVERRIDE_RAMP_SPEED_MAX,
    ))
    self.mc_subaru_manual_yield_resume_softness = int(np.clip(
      self._get_int_param("MCSubaruManualYieldResumeSoftness", ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_DEFAULT),
      ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_MIN,
      ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_MAX,
    ))

  def _reset_angle_driver_override_ramp(self):
    self.angle_driver_override_ramp_frames = 0
    self.angle_driver_override_ramp_total_frames = ANGLE_DRIVER_OVERRIDE_RAMP_FRAMES
    self.angle_driver_override_ramp_start_angle = 0.0
    self.angle_driver_override_ramp_softness_exponent = ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_EXPONENTS[ANGLE_DRIVER_OVERRIDE_RAMP_SOFTNESS_DEFAULT]

  def _reset_angle_driver_override_state(self):
    self.angle_driver_override_hold_frames = 0
    self._reset_angle_driver_override_ramp()

  def _start_angle_driver_override_ramp(self, measured_angle: float):
    ramp_frames = self._get_resume_speed_frames(self.mc_subaru_manual_yield_resume_speed)
    self.angle_driver_override_ramp_frames = ramp_frames
    self.angle_driver_override_ramp_total_frames = ramp_frames
    self.angle_driver_override_ramp_start_angle = measured_angle
    self.angle_driver_override_ramp_softness_exponent = self._get_resume_softness_exponent(
      self.mc_subaru_manual_yield_resume_softness
    )

  def _update_angle_driver_override_state(self, steering_pressed: bool, lkas_allowed: bool) -> tuple[bool, bool]:
    if not lkas_allowed:
      self._reset_angle_driver_override_state()
      return False, False

    if steering_pressed:
      self.angle_driver_override_hold_frames = ANGLE_DRIVER_OVERRIDE_HOLD_FRAMES
      self._reset_angle_driver_override_ramp()
      return True, False

    if self.angle_driver_override_hold_frames > 0:
      self.angle_driver_override_hold_frames -= 1
      if self.angle_driver_override_hold_frames == 0:
        return True, True
      return True, False

    return False, False

  def _apply_angle_driver_override_ramp(self, live_steer_target: float) -> tuple[float, bool]:
    if self.angle_driver_override_ramp_frames <= 0:
      return live_steer_target, False

    progress = (self.angle_driver_override_ramp_total_frames - self.angle_driver_override_ramp_frames + 1) / \
      self.angle_driver_override_ramp_total_frames
    eased_progress = progress ** self.angle_driver_override_ramp_softness_exponent
    ramped_target = self.angle_driver_override_ramp_start_angle + eased_progress * (
      live_steer_target - self.angle_driver_override_ramp_start_angle
    )

    self.angle_driver_override_ramp_frames -= 1
    if self.angle_driver_override_ramp_frames <= 0:
      self._reset_angle_driver_override_ramp()

    return ramped_target, True

  def _get_angle_lkas_target(self, raw_target: float, CS, capture_lkas_target: bool) -> tuple[float, bool, float, bool, bool]:
    if capture_lkas_target and CS.out.vEgoRaw < LOW_SPEED_SMOOTH_MAX_SPEED:
      # Keep the existing MC low-speed stack intact, with the Lukas-inspired
      # near-center delta deadzone now enabled by default.
      steer_target = self._get_low_speed_smoothed_angle_target(raw_target, CS.out.vEgoRaw)
      steer_target, delta_deadzone_active, delta_deadzone = self._get_low_speed_delta_deadzone_target(steer_target, CS, capture_lkas_target)
      steer_target = self._get_low_speed_stable_angle_target(steer_target, CS)
      steer_target, center_damping_active, sign_flip_clamped = self._get_low_speed_center_damped_angle_target(steer_target, CS)
      return steer_target, delta_deadzone_active, delta_deadzone, center_damping_active, sign_flip_clamped

    self._reset_low_speed_straight_stability()
    return raw_target, False, 0.0, False, False

  def _get_low_speed_smoothed_angle_target(self, raw_target, v_ego):
    speed_factor = np.clip(v_ego / LOW_SPEED_SMOOTH_MAX_SPEED, 0.0, 1.0)
    deadband_scale = 1.0
    alpha_scale = 1.0
    if self.mc_subaru_smoothing_tune:
      deadband_scale = self._get_strength_scale(self.mc_subaru_smoothing_strength, SUBARU_SMOOTHING_DEADBAND_SCALES)
      alpha_scale = self._get_strength_scale(self.mc_subaru_smoothing_strength, SUBARU_SMOOTHING_ALPHA_SCALES)

    deadband = (1.0 - speed_factor) * LOW_SPEED_SMOOTH_DEADBAND_MAX * deadband_scale
    delta = raw_target - self.apply_angle_last

    if abs(delta) <= deadband:
      return self.apply_angle_last

    delta = delta - deadband if delta > 0 else delta + deadband
    alpha = np.interp(v_ego, [0.0, LOW_SPEED_SMOOTH_MAX_SPEED], [LOW_SPEED_SMOOTH_ALPHA_MIN, 1.0])
    alpha = float(np.clip(alpha * alpha_scale, 0.15, 1.0))
    return self.apply_angle_last + alpha * delta

  def _reset_low_speed_straight_stability(self):
    self.low_speed_straight_pending_direction = 0
    self.low_speed_straight_pending_frames = 0

  @staticmethod
  def _angle_direction(angle: float) -> int:
    if angle > LOW_SPEED_STRAIGHT_SIGN_EPSILON:
      return 1
    if angle < -LOW_SPEED_STRAIGHT_SIGN_EPSILON:
      return -1
    return 0

  def _get_low_speed_stable_angle_target(self, raw_target: float, CS) -> float:
    if CS.out.vEgoRaw >= LOW_SPEED_SMOOTH_MAX_SPEED or CS.out.standstill or CS.out.steeringPressed:
      self._reset_low_speed_straight_stability()
      return raw_target

    measured_angle = CS.out.steeringAngleDeg
    if abs(raw_target) > LOW_SPEED_STRAIGHT_STABILITY_TARGET_MAX or \
       abs(measured_angle) > LOW_SPEED_STRAIGHT_STABILITY_STEER_MAX or \
       abs(self.apply_angle_last) > LOW_SPEED_STRAIGHT_STABILITY_STEER_MAX:
      self._reset_low_speed_straight_stability()
      return raw_target

    if abs(raw_target) <= LOW_SPEED_STRAIGHT_CENTER_HOLD_TARGET_MAX and \
       abs(measured_angle) <= LOW_SPEED_STRAIGHT_CENTER_HOLD_STEER_MAX:
      self._reset_low_speed_straight_stability()
      return 0.0

    target_direction = self._angle_direction(raw_target)
    current_direction = self._angle_direction(self.apply_angle_last)
    if target_direction == 0:
      self._reset_low_speed_straight_stability()
      return 0.0

    needs_release = abs(raw_target) < LOW_SPEED_STRAIGHT_SIGN_RELEASE_TARGET and \
      (current_direction == 0 or current_direction != target_direction)
    if not needs_release:
      self._reset_low_speed_straight_stability()
      return raw_target

    if self.low_speed_straight_pending_direction != target_direction:
      self.low_speed_straight_pending_direction = target_direction
      self.low_speed_straight_pending_frames = 1
    else:
      self.low_speed_straight_pending_frames += 1

    if self.low_speed_straight_pending_frames < LOW_SPEED_STRAIGHT_SIGN_RELEASE_FRAMES:
      return 0.0 if current_direction == 0 else self.apply_angle_last

    self._reset_low_speed_straight_stability()
    return raw_target

  def _get_low_speed_delta_deadzone_target(self, raw_target: float, CS, lkas_request: bool):
    deadzone_active = lkas_request and \
      CS.out.vEgoRaw < LOW_SPEED_SMOOTH_MAX_SPEED and \
      not CS.out.standstill and not CS.out.steeringPressed and \
      abs(raw_target) <= LOW_SPEED_DELTA_DEADZONE_TARGET_MAX and \
      abs(CS.out.steeringAngleDeg) <= LOW_SPEED_DELTA_DEADZONE_STEER_MAX
    if not deadzone_active:
      return raw_target, False, 0.0

    delta = raw_target - self.apply_angle_last
    deadzone = float(np.interp(
      CS.out.vEgoRaw,
      [0.0, LOW_SPEED_SMOOTH_MAX_SPEED],
      [LOW_SPEED_DELTA_DEADZONE_MAX, LOW_SPEED_DELTA_DEADZONE_MIN],
    ))
    filtered_target = self.apply_angle_last + apply_center_deadzone(delta, deadzone)
    return filtered_target, True, deadzone

  def _get_low_speed_center_damped_angle_target(self, raw_target: float, CS):
    center_damping_active = CS.out.vEgoRaw < LOW_SPEED_SMOOTH_MAX_SPEED and \
      not CS.out.standstill and not CS.out.steeringPressed and \
      abs(raw_target) <= LOW_SPEED_CENTER_DAMPING_TARGET_MAX and \
      abs(CS.out.steeringAngleDeg) <= LOW_SPEED_CENTER_DAMPING_STEER_MAX
    if not center_damping_active:
      return raw_target, False, False

    deadband_scale = 1.0
    max_delta_scale = 1.0
    alpha_scale = 1.0
    if self.mc_subaru_smoothing_tune:
      deadband_scale = self._get_strength_scale(self.mc_subaru_center_damping_strength, SUBARU_CENTER_DAMPING_DEADBAND_SCALES)
      max_delta_scale = self._get_strength_scale(self.mc_subaru_center_damping_strength, SUBARU_CENTER_DAMPING_SIGN_FLIP_SCALES)
      alpha_scale = self._get_strength_scale(self.mc_subaru_center_damping_strength, SUBARU_CENTER_DAMPING_ALPHA_SCALES)

    deadband = np.interp(
      CS.out.vEgoRaw,
      [0.0, LOW_SPEED_SMOOTH_MAX_SPEED],
      [LOW_SPEED_CENTER_DAMPING_DEADBAND_MAX, LOW_SPEED_CENTER_DAMPING_DEADBAND_MIN],
    ) * deadband_scale
    if abs(raw_target) <= deadband:
      return 0.0, True, False

    target_direction = self._angle_direction(raw_target)
    current_direction = self._angle_direction(self.apply_angle_last)
    sign_flip_clamped = abs(self.apply_angle_last) <= LOW_SPEED_CENTER_DAMPING_TARGET_MAX and \
      current_direction != 0 and target_direction != 0 and current_direction != target_direction

    clamped_target = raw_target
    if sign_flip_clamped:
      max_delta = np.interp(
        CS.out.vEgoRaw,
        [0.0, LOW_SPEED_SMOOTH_MAX_SPEED],
        [LOW_SPEED_CENTER_DAMPING_SIGN_FLIP_DELTA_MIN, LOW_SPEED_CENTER_DAMPING_SIGN_FLIP_DELTA_MAX],
      ) * max_delta_scale
      clamped_target = float(np.clip(raw_target, self.apply_angle_last - max_delta, self.apply_angle_last + max_delta))

    alpha = np.interp(
      CS.out.vEgoRaw,
      [0.0, LOW_SPEED_SMOOTH_MAX_SPEED],
      [LOW_SPEED_CENTER_DAMPING_ALPHA_MIN, LOW_SPEED_CENTER_DAMPING_ALPHA_MAX],
    )
    alpha = float(np.clip(alpha * alpha_scale, 0.15, 1.0))
    filtered_target = self.apply_angle_last + alpha * (clamped_target - self.apply_angle_last)
    return filtered_target, True, sign_flip_clamped

  def handle_angle_lateral(self, CC, CS):
    # Angle-LKAS can hard fault during very low-speed MADS lateral-only maneuvers.
    # Keep MADS behavior above 1 mph, but block sharp parking-lot style steering in lateral-only mode.
    mads_only = CC.latActive and not CC.enabled
    mads_only_ok = CS.out.vEgoRaw > MADS_ONLY_MIN_SPEED and abs(CS.out.steeringAngleDeg) < MADS_ONLY_MAX_STEER_ANGLE
    lkas_allowed = CC.latActive and (CC.enabled or not mads_only or mads_only_ok) and \
      CS.out.gearShifter == structs.CarState.GearShifter.drive and not CS.out.standstill
    angle_driver_override, ramp_will_start = self._update_angle_driver_override_state(
      CS.out.steeringPressed,
      lkas_allowed,
    )
    lkas_request = lkas_allowed and not angle_driver_override
    capture_lkas_target = lkas_request or ramp_will_start

    inhibit_reason = "none"
    if not CC.latActive:
      inhibit_reason = "lat_inactive"
    elif angle_driver_override:
      inhibit_reason = "manual_override"
    elif CS.out.gearShifter != structs.CarState.GearShifter.drive:
      inhibit_reason = "gear_not_drive"
    elif CS.out.standstill:
      inhibit_reason = "standstill"
    elif mads_only and not mads_only_ok:
      inhibit_reason = "mads_below_min_speed" if CS.out.vEgoRaw <= MADS_ONLY_MIN_SPEED else "mads_angle_limit"

    self._log_transition("angle_lkas_inhibit", inhibit_reason, f"angle LKAS inhibit={inhibit_reason}")
    self._log_transition(
      "angle_driver_override_hold",
      self.angle_driver_override_hold_frames > 0,
      f"angle driver override hold active={self.angle_driver_override_hold_frames > 0} "
      f"frames={self.angle_driver_override_hold_frames} steeringPressed={CS.out.steeringPressed}",
    )

    steer_target, delta_deadzone_active, delta_deadzone, center_damping_active, sign_flip_clamped = \
      self._get_angle_lkas_target(CC.actuators.steeringAngleDeg, CS, capture_lkas_target)

    if ramp_will_start:
      self._start_angle_driver_override_ramp(CS.out.steeringAngleDeg)

    if lkas_request:
      steer_target, manual_override_ramp_active = self._apply_angle_driver_override_ramp(steer_target)
    else:
      manual_override_ramp_active = False

    handoff_active = angle_driver_override or ramp_will_start or manual_override_ramp_active
    self._log_transition(
      "angle_lkas_request",
      lkas_request,
      f"angle LKAS request={lkas_request} inhibit={inhibit_reason} target={steer_target:.2f} "
      f"lastApplied={self.apply_angle_last:.2f} measuredAngle={CS.out.steeringAngleDeg:.2f} "
      f"measuredRate={CS.out.steeringRateDeg:.2f} handoffActive={handoff_active} "
      f"rampActive={manual_override_ramp_active} latActive={CC.latActive} enabled={CC.enabled}",
    )

    self._log_transition(
      "angle_lkas_delta_deadzone",
      delta_deadzone_active,
      f"angle LKAS delta deadzone active={delta_deadzone_active} "
      f"deadzone={delta_deadzone:.2f} target={steer_target:.2f} last={self.apply_angle_last:.2f} "
      f"vEgo={CS.out.vEgoRaw:.2f} steeringAngle={CS.out.steeringAngleDeg:.2f}",
    )
    self._log_transition(
      "angle_lkas_center_damping",
      center_damping_active,
      f"angle LKAS low-speed center damping active={center_damping_active} "
      f"target={steer_target:.2f} vEgo={CS.out.vEgoRaw:.2f} steeringAngle={CS.out.steeringAngleDeg:.2f}",
    )
    self._log_transition(
      "angle_lkas_center_sign_flip_clamp",
      sign_flip_clamped,
      f"angle LKAS center sign-flip clamp active={sign_flip_clamped} "
      f"target={steer_target:.2f} last={self.apply_angle_last:.2f} vEgo={CS.out.vEgoRaw:.2f}",
    )
    self._log_transition(
      "angle_driver_override_ramp",
      manual_override_ramp_active,
      f"angle driver override ramp active={manual_override_ramp_active} "
      f"framesRemaining={self.angle_driver_override_ramp_frames} totalFrames={self.angle_driver_override_ramp_total_frames} "
      f"softnessExponent={self.angle_driver_override_ramp_softness_exponent:.2f} "
      f"start={self.angle_driver_override_ramp_start_angle:.2f} "
      f"steerTarget={steer_target:.2f}",
    )

    apply_steer = apply_std_steer_angle_limits(
      steer_target,
      self.apply_angle_last,
      CS.out.vEgoRaw,
      CS.out.steeringAngleDeg,
      lkas_request,
      self.p.ANGLE_LIMITS,
    )

    if not lkas_request:
      apply_steer = CS.out.steeringAngleDeg

    self.apply_angle_last = apply_steer
    return subarucan.create_steering_control_angle(self.packer, apply_steer, lkas_request)

  def handle_torque_lateral(self, CC, CS):
    apply_torque = int(round(CC.actuators.torque * self.p.STEER_MAX))

    new_torque = int(round(apply_torque))
    apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorque, self.p)

    if not CC.latActive:
      apply_torque = 0

    if self.CP.flags & SubaruFlags.PREGLOBAL:
      msg = subarucan.create_preglobal_steering_control(self.packer, self.frame // self.p.STEER_STEP, apply_torque, CC.latActive)
    else:
      apply_steer_req = CC.latActive

      if self.CP.flags & SubaruFlags.STEER_RATE_LIMITED:
        # Steering rate fault prevention
        self.steer_rate_counter, apply_steer_req = common_fault_avoidance(
          abs(CS.out.steeringRateDeg) > MAX_STEER_RATE,
          apply_steer_req,
          self.steer_rate_counter,
          MAX_STEER_RATE_FRAMES,
        )

      msg = subarucan.create_steering_control(self.packer, apply_torque, apply_steer_req)

    self.apply_torque_last = apply_torque
    return msg

  def update(self, CC, CC_SP, CS, now_nanos):
    if self.frame % 100 == 0:
      self._update_params()

    actuators = CC.actuators
    hud_control = CC.hudControl
    pcm_cancel_cmd = CC.cruiseControl.cancel

    can_sends = []

    # *** steering ***
    if (self.frame % self.p.STEER_STEP) == 0:
      if self.CP.flags & SubaruFlags.LKAS_ANGLE:
        can_sends.append(self.handle_angle_lateral(CC, CS))
      else:
        can_sends.append(self.handle_torque_lateral(CC, CS))

    # *** longitudinal ***

    if CC.longActive:
      apply_throttle = int(round(np.interp(actuators.accel, CarControllerParams.THROTTLE_LOOKUP_BP, CarControllerParams.THROTTLE_LOOKUP_V)))
      apply_rpm = int(round(np.interp(actuators.accel, CarControllerParams.RPM_LOOKUP_BP, CarControllerParams.RPM_LOOKUP_V)))
      apply_brake = int(round(np.interp(actuators.accel, CarControllerParams.BRAKE_LOOKUP_BP, CarControllerParams.BRAKE_LOOKUP_V)))

      # limit min and max values
      cruise_throttle = np.clip(apply_throttle, CarControllerParams.THROTTLE_MIN, CarControllerParams.THROTTLE_MAX)
      cruise_rpm = np.clip(apply_rpm, CarControllerParams.RPM_MIN, CarControllerParams.RPM_MAX)
      cruise_brake = np.clip(apply_brake, CarControllerParams.BRAKE_MIN, CarControllerParams.BRAKE_MAX)
    else:
      cruise_throttle = CarControllerParams.THROTTLE_INACTIVE
      cruise_rpm = CarControllerParams.RPM_MIN
      cruise_brake = CarControllerParams.BRAKE_MIN

    # *** alerts and pcm cancel ***
    if self.CP.flags & SubaruFlags.PREGLOBAL:
      if self.frame % 5 == 0:
        # 1 = main, 2 = set shallow, 3 = set deep, 4 = resume shallow, 5 = resume deep
        # disengage ACC when OP is disengaged
        if pcm_cancel_cmd:
          cruise_button = 1
        # turn main on if off and past start-up state
        elif not CS.out.cruiseState.available and CS.ready:
          cruise_button = 1
        else:
          cruise_button = CS.cruise_button

        # unstick previous mocked button press
        if cruise_button == 1 and self.cruise_button_prev == 1:
          cruise_button = 0
        self.cruise_button_prev = cruise_button

        can_sends.append(subarucan.create_preglobal_es_distance(self.packer, cruise_button, CS.es_distance_msg))

    else:
      if self.frame % 10 == 0:
        can_sends.append(subarucan.create_es_dashstatus(self.packer, self.frame // 10, CS.es_dashstatus_msg, CC.enabled,
                                                        self.CP.openpilotLongitudinalControl, CC.longActive, hud_control.leadVisible))

        can_sends.append(subarucan.create_es_lkas_state(self.packer, self.frame // 10, CS.es_lkas_state_msg, CC.enabled, hud_control.visualAlert,
                                                        hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                                                        hud_control.leftLaneDepart, hud_control.rightLaneDepart))

        if self.CP.flags & SubaruFlags.SEND_INFOTAINMENT:
          can_sends.append(subarucan.create_es_infotainment(self.packer, self.frame // 10, CS.es_infotainment_msg, hud_control.visualAlert))

      if self.CP.openpilotLongitudinalControl:
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_status(self.packer, self.frame // 5, CS.es_status_msg,
                                                      self.CP.openpilotLongitudinalControl, CC.longActive, cruise_rpm))

          can_sends.append(subarucan.create_es_brake(self.packer, self.frame // 5, CS.es_brake_msg,
                                                     self.CP.openpilotLongitudinalControl, CC.longActive, cruise_brake))

          can_sends.append(subarucan.create_es_distance(self.packer, self.frame // 5, CS.es_distance_msg, 0, pcm_cancel_cmd,
                                                        self.CP.openpilotLongitudinalControl, cruise_brake > 0, cruise_throttle))
      else:
        if pcm_cancel_cmd:
          if not (self.CP.flags & SubaruFlags.HYBRID):
            bus = CanBus.alt if self.CP.flags & SubaruFlags.GLOBAL_GEN2 else CanBus.main
            can_sends.append(subarucan.create_es_distance(self.packer, CS.es_distance_msg["COUNTER"] + 1, CS.es_distance_msg, bus, pcm_cancel_cmd))

      if self.CP.flags & SubaruFlags.DISABLE_EYESIGHT:
        # Tester present (keeps eyesight disabled)
        if self.frame % 100 == 0:
          can_sends.append(make_tester_present_msg(GLOBAL_ES_ADDR, CanBus.camera, suppress_response=True))

        # Create all of the other eyesight messages to keep the rest of the car happy when eyesight is disabled
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_highbeamassist(self.packer))

        if self.frame % 10 == 0:
          can_sends.append(subarucan.create_es_static_1(self.packer))

        if self.frame % 2 == 0:
          can_sends.append(subarucan.create_es_static_2(self.packer))

    can_sends.extend(SnGCarController.create_stop_and_go(self, self.packer, CC, CS, self.frame))

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last
    new_actuators.torque = self.apply_torque_last / self.p.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last

    self.frame += 1
    return new_actuators, can_sends

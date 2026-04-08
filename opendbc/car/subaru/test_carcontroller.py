import unittest
from types import SimpleNamespace

from openpilot.common.params import Params
from opendbc.car import structs
from opendbc.car.subaru import subarucan
from opendbc.car.subaru.carcontroller import (
  CarController,
  LOW_SPEED_SMOOTH_MAX_SPEED,
  MADS_ONLY_MIN_SPEED,
  SOFT_CAPTURE_LEVEL_PARAMS,
  SUBARU_TUNING_STRENGTH_MAX,
  SUBARU_TUNING_STRENGTH_MIN,
)
from opendbc.car.subaru.fingerprints import FW_VERSIONS
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR
from opendbc.car.tests.routes import routes


class TestSubaruCarController(unittest.TestCase):
  PARAM_KEYS = (
    "MCSubaruSmoothingTune",
    "MCSubaruSmoothingStrength",
    "MCSubaruCenterDampingTune",
    "MCSubaruCenterDampingStrength",
    "MCSubaruSoftCaptureEnabled",
    "MCSubaruSoftCaptureLevel",
  )

  def setUp(self):
    self.params = Params()
    for key in self.PARAM_KEYS:
      self.params.remove(key)

  def tearDown(self):
    for key in self.PARAM_KEYS:
      self.params.remove(key)

  @staticmethod
  def _build_cs(v_ego_raw, steering_angle_deg, steering_pressed=False, standstill=False, steering_rate_deg=0.0):
    return SimpleNamespace(out=SimpleNamespace(
      vEgoRaw=v_ego_raw,
      steeringAngleDeg=steering_angle_deg,
      steeringRateDeg=steering_rate_deg,
      gearShifter=structs.CarState.GearShifter.drive,
      standstill=standstill,
      steeringPressed=steering_pressed,
    ))

  @staticmethod
  def _build_cc(lat_active, enabled, steering_angle_deg):
    return SimpleNamespace(
      latActive=lat_active,
      enabled=enabled,
      actuators=SimpleNamespace(steeringAngleDeg=steering_angle_deg),
    )

  def _build_controller(
    self,
    *,
    soft_capture_enabled=False,
    soft_capture_level=1,
    smoothing_tune=False,
    smoothing_strength=0,
    center_damping_tune=False,
    center_damping_strength=0,
  ):
    self.params.put_bool("MCSubaruSoftCaptureEnabled", soft_capture_enabled)
    self.params.put("MCSubaruSoftCaptureLevel", str(soft_capture_level))
    self.params.put_bool("MCSubaruSmoothingTune", smoothing_tune)
    self.params.put("MCSubaruSmoothingStrength", str(smoothing_strength))
    self.params.put_bool("MCSubaruCenterDampingTune", center_damping_tune)
    self.params.put("MCSubaruCenterDampingStrength", str(center_damping_strength))
    CP = CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK_2023)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, CAR.SUBARU_OUTBACK_2023)
    return CarController({}, CP, CP_SP)

  def test_driver_press_immediately_yields_in_mads_only(self):
    controller = self._build_controller()
    cs = self._build_cs(9.5, 20.56, steering_pressed=True)
    cc = self._build_cc(True, False, 19.86)
    controller.apply_angle_last = cs.out.steeringAngleDeg

    msg = controller.handle_angle_lateral(cc, cs)
    expected = subarucan.create_steering_control_angle(controller.packer, cs.out.steeringAngleDeg, False)

    self.assertEqual(msg, expected)
    self.assertAlmostEqual(controller.apply_angle_last, cs.out.steeringAngleDeg)

  def test_driver_press_immediately_yields_in_full_engaged(self):
    controller = self._build_controller()
    cs = self._build_cs(9.5, 20.56, steering_pressed=True)
    cc = self._build_cc(True, True, 19.86)
    controller.apply_angle_last = cs.out.steeringAngleDeg

    msg = controller.handle_angle_lateral(cc, cs)
    expected = subarucan.create_steering_control_angle(controller.packer, cs.out.steeringAngleDeg, False)

    self.assertEqual(msg, expected)
    self.assertAlmostEqual(controller.apply_angle_last, cs.out.steeringAngleDeg)

  def test_release_after_manual_override_no_longer_uses_reclaim_hold_or_ramp(self):
    controller = self._build_controller(soft_capture_enabled=False)
    cc = self._build_cc(True, True, 14.0)
    pressed_cs = self._build_cs(8.0, 10.0, steering_pressed=True)
    released_cs = self._build_cs(8.0, 10.0, steering_pressed=False)
    controller.apply_angle_last = pressed_cs.out.steeringAngleDeg

    controller.handle_angle_lateral(cc, pressed_cs)
    msg = controller.handle_angle_lateral(cc, released_cs)
    inhibited = subarucan.create_steering_control_angle(controller.packer, released_cs.out.steeringAngleDeg, False)

    self.assertNotEqual(msg, inhibited)
    self.assertGreater(controller.apply_angle_last, released_cs.out.steeringAngleDeg)

  def test_soft_capture_disabled_is_noop(self):
    controller = self._build_controller(soft_capture_enabled=False, soft_capture_level=5)
    controller.soft_capture_frame = controller.frame

    self.assertEqual(controller._get_soft_capture_level(), 0)
    self.assertAlmostEqual(controller._get_soft_capture_angle(18.0, 10.0), 18.0)

  def test_fresh_engage_starts_soft_capture_ramp(self):
    baseline = self._build_controller(soft_capture_enabled=False)
    softened = self._build_controller(soft_capture_enabled=True, soft_capture_level=3)
    cc = self._build_cc(True, True, 14.0)
    cs = self._build_cs(8.0, 10.0)
    baseline.apply_angle_last = cs.out.steeringAngleDeg
    softened.apply_angle_last = cs.out.steeringAngleDeg

    baseline.handle_angle_lateral(cc, cs)
    softened.handle_angle_lateral(cc, cs)

    self.assertEqual(softened.soft_capture_frame, 0)
    self.assertTrue(softened.lat_active_prev)
    self.assertLess(softened.apply_angle_last, baseline.apply_angle_last)

  def test_soft_capture_higher_levels_reduce_initial_blend_delta(self):
    light = self._build_controller(soft_capture_enabled=True, soft_capture_level=1)
    medium = self._build_controller(soft_capture_enabled=True, soft_capture_level=3)
    maximum = self._build_controller(soft_capture_enabled=True, soft_capture_level=5)

    for controller in (light, medium, maximum):
      controller.soft_capture_frame = 0
      controller.frame = 0

    model_target = 20.0
    wheel_angle = 10.0
    light_delta = light._get_soft_capture_angle(model_target, wheel_angle) - wheel_angle
    medium_delta = medium._get_soft_capture_angle(model_target, wheel_angle) - wheel_angle
    max_delta = maximum._get_soft_capture_angle(model_target, wheel_angle) - wheel_angle

    self.assertGreater(light_delta, medium_delta)
    self.assertGreater(medium_delta, max_delta)

  def test_soft_capture_ramp_completes_to_full_model_target(self):
    controller = self._build_controller(soft_capture_enabled=True, soft_capture_level=3)
    ramp_frames, _ = SOFT_CAPTURE_LEVEL_PARAMS[3]
    controller.soft_capture_frame = 0
    controller.frame = ramp_frames

    self.assertAlmostEqual(controller._get_soft_capture_angle(18.0, 10.0), 18.0)

  def test_soft_capture_does_not_retrigger_on_manual_release_without_new_lat_active_edge(self):
    controller = self._build_controller(soft_capture_enabled=True, soft_capture_level=3)
    cc = self._build_cc(True, True, 14.0)
    released_cs = self._build_cs(8.0, 10.0, steering_pressed=False)
    pressed_cs = self._build_cs(8.0, 10.0, steering_pressed=True)
    controller.apply_angle_last = released_cs.out.steeringAngleDeg

    controller.handle_angle_lateral(cc, released_cs)
    self.assertEqual(controller.soft_capture_frame, 0)

    controller.handle_angle_lateral(cc, pressed_cs)
    controller.handle_angle_lateral(cc, released_cs)
    self.assertEqual(controller.soft_capture_frame, 0)

  def test_default_path_has_no_hidden_low_speed_shaping_when_optional_toggles_are_off(self):
    controller = self._build_controller(soft_capture_enabled=False, smoothing_tune=False, center_damping_tune=False)
    controller.apply_angle_last = 0.5
    cs = self._build_cs(3.0, 0.2)

    steer_target, center_damping_active, sign_flip_clamped = controller._get_angle_lkas_target(1.2, cs, True)

    self.assertAlmostEqual(steer_target, 1.2)
    self.assertFalse(center_damping_active)
    self.assertFalse(sign_flip_clamped)

  def test_smoothing_toggle_off_is_true_bypass(self):
    controller = self._build_controller(smoothing_tune=False, smoothing_strength=3)
    controller.apply_angle_last = 0.5

    self.assertAlmostEqual(controller._get_low_speed_smoothed_angle_target(1.8, 0.5), 1.8)

  def test_smoothing_toggle_on_changes_target(self):
    controller = self._build_controller(smoothing_tune=True, smoothing_strength=3)
    controller.apply_angle_last = 0.0

    self.assertLess(controller._get_low_speed_smoothed_angle_target(1.8, 0.5), 1.8)

  def test_center_damping_toggle_off_is_true_bypass(self):
    controller = self._build_controller(center_damping_tune=False, center_damping_strength=3)
    controller.apply_angle_last = 0.5
    cs = self._build_cs(3.0, 0.2)

    damped_target, active, sign_flip_clamped = controller._get_low_speed_center_damped_angle_target(1.8, cs)

    self.assertAlmostEqual(damped_target, 1.8)
    self.assertFalse(active)
    self.assertFalse(sign_flip_clamped)

  def test_center_damping_toggle_on_only_affects_near_center_requests(self):
    controller = self._build_controller(center_damping_tune=True, center_damping_strength=3)
    controller.apply_angle_last = 0.5
    cs = self._build_cs(3.0, 0.2)

    damped_target, active, sign_flip_clamped = controller._get_low_speed_center_damped_angle_target(1.8, cs)

    self.assertTrue(active)
    self.assertFalse(sign_flip_clamped)
    self.assertLess(damped_target, 1.8)

  def test_subaru_tuning_range_constants_stay_expected(self):
    self.assertEqual(SUBARU_TUNING_STRENGTH_MIN, -3)
    self.assertEqual(SUBARU_TUNING_STRENGTH_MAX, 4)

  def test_mads_only_below_one_mph_still_inhibits_angle_lkas(self):
    controller = self._build_controller()
    cs = self._build_cs(0.22352, 10.0)
    cc = self._build_cc(True, False, 14.0)
    controller.apply_angle_last = cs.out.steeringAngleDeg

    msg = controller.handle_angle_lateral(cc, cs)
    expected = subarucan.create_steering_control_angle(controller.packer, cs.out.steeringAngleDeg, False)

    self.assertEqual(msg, expected)
    self.assertAlmostEqual(controller.apply_angle_last, cs.out.steeringAngleDeg)

  def test_mads_only_just_above_one_mph_allows_angle_lkas(self):
    controller = self._build_controller(soft_capture_enabled=False)
    cs = self._build_cs(MADS_ONLY_MIN_SPEED + 0.01, 10.0)
    cc = self._build_cc(True, False, 14.0)
    controller.apply_angle_last = cs.out.steeringAngleDeg

    msg = controller.handle_angle_lateral(cc, cs)
    inhibited = subarucan.create_steering_control_angle(controller.packer, cs.out.steeringAngleDeg, False)

    self.assertNotEqual(msg, inhibited)
    self.assertGreater(controller.apply_angle_last, cs.out.steeringAngleDeg)

  def test_mads_only_standstill_still_inhibits_above_one_mph(self):
    controller = self._build_controller()
    cs = self._build_cs(MADS_ONLY_MIN_SPEED + 0.5, 10.0, standstill=True)
    cc = self._build_cc(True, False, 14.0)
    controller.apply_angle_last = cs.out.steeringAngleDeg

    msg = controller.handle_angle_lateral(cc, cs)
    expected = subarucan.create_steering_control_angle(controller.packer, cs.out.steeringAngleDeg, False)

    self.assertEqual(msg, expected)
    self.assertAlmostEqual(controller.apply_angle_last, cs.out.steeringAngleDeg)

  def test_mads_only_angle_limit_still_inhibits_above_one_mph(self):
    controller = self._build_controller()
    cs = self._build_cs(MADS_ONLY_MIN_SPEED + 0.5, 120.0)
    cc = self._build_cc(True, False, 124.0)
    controller.apply_angle_last = cs.out.steeringAngleDeg

    msg = controller.handle_angle_lateral(cc, cs)
    expected = subarucan.create_steering_control_angle(controller.packer, cs.out.steeringAngleDeg, False)

    self.assertEqual(msg, expected)
    self.assertAlmostEqual(controller.apply_angle_last, cs.out.steeringAngleDeg)

  def test_full_engaged_lateral_ignores_mads_only_low_speed_floor(self):
    controller = self._build_controller(soft_capture_enabled=False)
    cs = self._build_cs(0.22352, 10.0)
    cc = self._build_cc(True, True, 14.0)
    controller.apply_angle_last = cs.out.steeringAngleDeg

    msg = controller.handle_angle_lateral(cc, cs)
    inhibited = subarucan.create_steering_control_angle(controller.packer, cs.out.steeringAngleDeg, False)

    self.assertNotEqual(msg, inhibited)
    self.assertGreater(controller.apply_angle_last, cs.out.steeringAngleDeg)

  def test_optional_low_speed_logic_fades_out_by_ten_mph(self):
    controller = self._build_controller(smoothing_tune=True, smoothing_strength=3, center_damping_tune=True, center_damping_strength=3)
    controller.apply_angle_last = 0.5
    cs = self._build_cs(LOW_SPEED_SMOOTH_MAX_SPEED, 0.2)

    steer_target, center_damping_active, sign_flip_clamped = controller._get_angle_lkas_target(1.8, cs, True)

    self.assertAlmostEqual(steer_target, 1.8)
    self.assertFalse(center_damping_active)
    self.assertFalse(sign_flip_clamped)

  def test_outback_2023_angle_steering_route_still_present(self):
    route = next(route for route in routes if route.platform == CAR.SUBARU_OUTBACK_2023)
    self.assertEqual(route.platform, CAR.SUBARU_OUTBACK_2023)

  def test_crosstrek_2025_fw_versions_still_present(self):
    self.assertIn(CAR.SUBARU_CROSSTREK_2025, FW_VERSIONS)


if __name__ == "__main__":
  unittest.main()

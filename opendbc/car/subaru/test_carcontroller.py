import unittest
from types import SimpleNamespace

from opendbc.car import structs
from opendbc.car.lateral import apply_std_steer_angle_limits, get_max_angle_delta_vm
from opendbc.car.subaru import subarucan
from opendbc.car.subaru.carcontroller import CarController
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR


class TestSubaruCarController(unittest.TestCase):
  @staticmethod
  def _build_cs(v_ego_raw, steering_angle_deg, steering_pressed=False):
    return SimpleNamespace(out=SimpleNamespace(
      vEgoRaw=v_ego_raw,
      steeringAngleDeg=steering_angle_deg,
      gearShifter=structs.CarState.GearShifter.drive,
      standstill=False,
      steeringPressed=steering_pressed,
    ))

  @staticmethod
  def _build_cc(lat_active, enabled, steering_angle_deg):
    return SimpleNamespace(
      latActive=lat_active,
      enabled=enabled,
      actuators=SimpleNamespace(steeringAngleDeg=steering_angle_deg),
    )

  def _build_controller(self):
    CP = CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK_2023)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, CAR.SUBARU_OUTBACK_2023)
    return CarController({}, CP, CP_SP)

  def test_lkas_angle_holds_request_until_gap_is_legal(self):
    controller = self._build_controller()
    cs = self._build_cs(9.5, 20.56)
    cc = self._build_cc(True, True, 19.86)

    max_angle_delta = get_max_angle_delta_vm(cs.out.vEgoRaw, controller.VM, controller.p)
    controller.apply_angle_last = cs.out.steeringAngleDeg - (max_angle_delta + 1.0)

    msg = controller.handle_angle_lateral(cc, cs)
    expected = subarucan.create_steering_control_angle(controller.packer, cs.out.steeringAngleDeg, False)

    self.assertEqual(msg, expected)
    self.assertAlmostEqual(controller.apply_angle_last, cs.out.steeringAngleDeg)

  def test_lkas_angle_engages_after_gap_alignment(self):
    controller = self._build_controller()
    cs = self._build_cs(9.5, 20.56)
    cc = self._build_cc(True, True, 19.86)

    max_angle_delta = get_max_angle_delta_vm(cs.out.vEgoRaw, controller.VM, controller.p)
    controller.apply_angle_last = cs.out.steeringAngleDeg - (max_angle_delta * 0.5)
    controller.prev_lat_active = True

    msg = controller.handle_angle_lateral(cc, cs)
    expected_apply_steer = apply_std_steer_angle_limits(
      cc.actuators.steeringAngleDeg,
      cs.out.steeringAngleDeg - (max_angle_delta * 0.5),
      cs.out.vEgoRaw,
      cs.out.steeringAngleDeg,
      True,
      controller.p.ANGLE_LIMITS,
    )
    expected = subarucan.create_steering_control_angle(controller.packer, expected_apply_steer, True)

    self.assertEqual(msg, expected)
    self.assertAlmostEqual(controller.apply_angle_last, expected_apply_steer)

  def test_mads_manual_override_still_wins(self):
    controller = self._build_controller()
    cs = self._build_cs(9.5, 20.56, steering_pressed=True)
    cc = self._build_cc(True, False, 19.86)

    controller.apply_angle_last = cs.out.steeringAngleDeg
    controller.prev_lat_active = True

    msg = controller.handle_angle_lateral(cc, cs)
    expected = subarucan.create_steering_control_angle(controller.packer, cs.out.steeringAngleDeg, False)

    self.assertEqual(msg, expected)
    self.assertAlmostEqual(controller.apply_angle_last, cs.out.steeringAngleDeg)


if __name__ == "__main__":
  unittest.main()

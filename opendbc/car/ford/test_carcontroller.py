import unittest
from collections import deque
from types import SimpleNamespace

from opendbc.car.ford.carcontroller import compute_f150_canfd_path_signals


def _build_model(path_offset: float, path_angle: float, orientation_rate: float):
  times = [0.0, 0.2, 0.4]
  return SimpleNamespace(
    position=SimpleNamespace(y=[0.0, path_offset, path_offset], t=times),
    orientation=SimpleNamespace(z=[0.0, path_angle, path_angle], t=times),
    orientationRate=SimpleNamespace(z=[0.0, orientation_rate, orientation_rate], t=times),
  )


class TestFordCarController(unittest.TestCase):
  def test_f150_model_signals_go_nonzero_for_curve(self):
    model = _build_model(path_offset=0.2, path_angle=0.05, orientation_rate=0.02)
    signals = compute_f150_canfd_path_signals(
      model=model,
      v_ego_raw=20.0,
      desired_curvature=0.001,
      lane_change_active=False,
      steering_pressed=False,
      steering_angle_deg=0.0,
      curvature_history=deque(maxlen=6),
      last_path_offset=0.0,
      last_path_angle=0.0,
      last_curvature_rate=0.0,
    )

    self.assertNotEqual(signals.path_offset, 0.0)
    self.assertNotEqual(signals.path_angle, 0.0)
    self.assertGreater(signals.curvature, 0.001)
    self.assertEqual(signals.precision_type, 1)
    self.assertEqual(signals.ramp_type, 2)

  def test_f150_lane_change_zeroes_extra_path_fields(self):
    model = _build_model(path_offset=0.2, path_angle=0.05, orientation_rate=0.02)
    signals = compute_f150_canfd_path_signals(
      model=model,
      v_ego_raw=20.0,
      desired_curvature=0.001,
      lane_change_active=True,
      steering_pressed=False,
      steering_angle_deg=0.0,
      curvature_history=deque(maxlen=6),
      last_path_offset=0.0,
      last_path_angle=0.0,
      last_curvature_rate=0.0,
    )

    self.assertEqual(signals.path_offset, 0.0)
    self.assertEqual(signals.path_angle, 0.0)
    self.assertEqual(signals.curvature_rate, 0.0)
    self.assertEqual(signals.precision_type, 0)

  def test_f150_human_turn_requests_immediate_ramp_out(self):
    model = _build_model(path_offset=0.2, path_angle=0.05, orientation_rate=0.02)
    signals = compute_f150_canfd_path_signals(
      model=model,
      v_ego_raw=20.0,
      desired_curvature=0.002,
      lane_change_active=False,
      steering_pressed=True,
      steering_angle_deg=50.0,
      curvature_history=deque(maxlen=6),
      last_path_offset=0.0,
      last_path_angle=0.0,
      last_curvature_rate=0.0,
    )

    self.assertEqual(signals.curvature, 0.0)
    self.assertEqual(signals.path_offset, 0.0)
    self.assertEqual(signals.path_angle, 0.0)
    self.assertEqual(signals.curvature_rate, 0.0)
    self.assertEqual(signals.ramp_type, 3)


if __name__ == "__main__":
  unittest.main()

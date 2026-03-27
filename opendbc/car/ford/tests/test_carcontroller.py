from collections import deque
from types import SimpleNamespace
import unittest

from opendbc.car.ford.carcontroller import (
  apply_f150_post_lane_change_transition,
  apply_f150_post_reset_ramp,
  compute_f150_blended_path_offset,
  compute_f150_canfd_path_signals,
  compute_f150_path_angle_from_offset,
)


def build_model(position_y, left_lane, right_lane, lane_change_state=0, lane_change_direction=0):
  return SimpleNamespace(
    position=SimpleNamespace(y=[position_y, position_y], t=[0.0, 0.2]),
    orientation=SimpleNamespace(z=[0.0, 0.0], t=[0.0, 0.2]),
    orientationRate=SimpleNamespace(z=[0.0, 0.0], t=[0.0, 0.2]),
    laneLines=[
      SimpleNamespace(y=[0.0]),
      SimpleNamespace(y=[left_lane]),
      SimpleNamespace(y=[right_lane]),
      SimpleNamespace(y=[0.0]),
    ],
    laneLineProbs=[0.0, 0.95, 0.95, 0.0],
    meta=SimpleNamespace(laneChangeState=lane_change_state, laneChangeDirection=lane_change_direction),
  )


class TestFordCarController(unittest.TestCase):
  def test_f150_blended_path_offset_prefers_confident_lane_center(self):
    model = build_model(position_y=0.4, left_lane=-1.8, right_lane=1.6)
    blended_offset = compute_f150_blended_path_offset(model)

    self.assertLess(blended_offset, 0.0)
    self.assertAlmostEqual(blended_offset, -0.1, places=3)

  def test_f150_path_angle_builds_from_offset_error(self):
    path_angle, integral, reset_counter = compute_f150_path_angle_from_offset(
      path_offset=0.5,
      v_ego_raw=20.0,
      steering_pressed=False,
      reset_requested=False,
      last_path_angle=0.0,
      path_angle_integral=0.0,
      path_angle_reset_counter=0,
    )
    next_path_angle, next_integral, next_reset_counter = compute_f150_path_angle_from_offset(
      path_offset=0.5,
      v_ego_raw=20.0,
      steering_pressed=False,
      reset_requested=False,
      last_path_angle=path_angle,
      path_angle_integral=integral,
      path_angle_reset_counter=reset_counter,
    )

    self.assertGreater(path_angle, 0.0)
    self.assertGreater(next_path_angle, path_angle)
    self.assertGreater(next_integral, integral)
    self.assertEqual(next_reset_counter, 0)

  def test_f150_post_lane_change_transition_rate_limits_outputs(self):
    path_angle, path_offset, curvature_rate, lane_change_last, post_lane_change_active, post_lane_change_timer, pre_values = \
      apply_f150_post_lane_change_transition(
        lane_change_active=False,
        lane_change_last=True,
        post_lane_change_active=False,
        post_lane_change_timer=0,
        path_angle=0.1,
        path_offset=0.5,
        desired_curvature_rate=0.001,
        pre_lane_change_values={"path_angle": 0.0, "path_offset": 0.0, "desired_curvature_rate": 0.0},
      )

    self.assertTrue(post_lane_change_active)
    self.assertEqual(post_lane_change_timer, 1)
    self.assertFalse(lane_change_last)
    self.assertAlmostEqual(path_angle, 0.00125, places=6)
    self.assertAlmostEqual(path_offset, 0.00125, places=6)
    self.assertAlmostEqual(curvature_rate, 0.0001, places=7)
    self.assertAlmostEqual(pre_values["path_angle"], path_angle, places=6)

  def test_f150_post_reset_ramp_reenters_gradually(self):
    apply_curvature, ramp_active = apply_f150_post_reset_ramp(
      requested_curvature=0.01,
      apply_curvature_last=0.0,
      v_ego_raw=20.0,
      lat_active=True,
    )

    self.assertGreater(apply_curvature, 0.0)
    self.assertLess(apply_curvature, 0.01)
    self.assertTrue(ramp_active)

  def test_f150_lane_change_path_signals_zero_during_change(self):
    model = build_model(position_y=0.4, left_lane=-1.8, right_lane=1.6, lane_change_state=2, lane_change_direction=1)
    signals, integral, reset_counter = compute_f150_canfd_path_signals(
      model=model,
      v_ego_raw=20.0,
      desired_curvature=-0.01,
      lane_change_active=True,
      lane_change_direction=1,
      steering_pressed=False,
      steering_angle_deg=0.0,
      curvature_history=deque(maxlen=6),
      last_path_offset=0.0,
      last_path_angle=0.0,
      last_curvature_rate=0.0,
      path_angle_integral=0.0,
      path_angle_reset_counter=0,
    )

    self.assertEqual(signals.precision_type, 0)
    self.assertEqual(signals.path_offset, 0.0)
    self.assertEqual(signals.path_angle, 0.0)
    self.assertEqual(signals.curvature_rate, 0.0)
    self.assertLess(abs(signals.curvature), 0.01)
    self.assertGreaterEqual(integral, 0.0)
    self.assertEqual(reset_counter, 0)

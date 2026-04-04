from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CARSTATE = REPO_ROOT / "opendbc_repo/opendbc/car/subaru/carstate.py"
CARCONTROLLER = REPO_ROOT / "opendbc_repo/opendbc/car/subaru/carcontroller.py"


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def test_carstate_fault_logs_include_steering_and_cruise_context():
  source = _read(CARSTATE)
  assert 'steerFaultTemporary={ret.steerFaultTemporary} angle={ret.steeringAngleDeg:.2f}' in source
  assert 'steerFaultPermanent={ret.steerFaultPermanent} angle={ret.steeringAngleDeg:.2f}' in source
  assert 'rate={ret.steeringRateDeg:.2f}' in source
  assert 'torque={ret.steeringTorque:.2f}' in source
  assert 'torqueEps={ret.steeringTorqueEps:.2f}' in source
  assert 'cruiseEnabled={ret.cruiseState.enabled}' in source
  assert 'cruiseAvailable={ret.cruiseState.available}' in source


def test_carstate_publishes_subaru_brake_light_state_for_ui():
  source = _read(CARSTATE)
  assert 'ret_sp.brakeLightsAvailable = not (self.CP.flags & SubaruFlags.PREGLOBAL)' in source
  assert 'ret_sp.brakeLightsOn = ret.brakePressed' in source
  assert 'cp_cam.vl["ES_DashStatus"]["Brake_Lights"] != 0' in source
  assert 'cp_es_brake.vl["ES_Brake"]["Cruise_Brake_Lights"] != 0' in source
  assert 'cp_es_brake.vl["ES_Status"]["Brake_Lights"] != 0' in source


def test_carstate_publishes_subaru_cluster_speed_for_ui_toggle():
  source = _read(CARSTATE)
  assert 'self.cluster_speed_hyst_gap = CV.KPH_TO_MS / 2.' in source
  assert 'self.cluster_min_speed = CV.KPH_TO_MS / 2.' in source
  assert 'cluster_speed_kph = cp.vl["Brake_Pedal"]["Speed"]' in source
  assert 'cluster_speed_kph = max(cp.vl["Brake_Pedal"]["Speed"], cp_alt.vl["Brake_Pedal"]["Speed"])' in source
  assert 'ret.vEgoCluster = cluster_speed_kph * CV.KPH_TO_MS' in source


def test_carcontroller_request_logs_include_target_and_handoff_context():
  source = _read(CARCONTROLLER)
  assert 'angle LKAS request={lkas_request} inhibit={inhibit_reason} target={steer_target:.2f}' in source
  assert 'lastApplied={self.apply_angle_last:.2f}' in source
  assert 'measuredAngle={CS.out.steeringAngleDeg:.2f}' in source
  assert 'measuredRate={CS.out.steeringRateDeg:.2f}' in source
  assert 'handoffActive={handoff_active}' in source
  assert 'rampActive={manual_override_ramp_active}' in source


def test_carcontroller_no_longer_reads_chatter_toggle_param():
  source = _read(CARCONTROLLER)
  assert "MCSubaruChatterFix" not in source
  assert "mc_subaru_chatter_fix" not in source

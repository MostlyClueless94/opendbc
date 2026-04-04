from opendbc.car import structs
from opendbc.car.subaru.fingerprints import FW_VERSIONS


class TestSubaruFingerprint:
  def test_fw_version_format(self):
    for platform, fws_per_ecu in FW_VERSIONS.items():
      for (ecu, _, _), fws in fws_per_ecu.items():
        fw_size = len(fws[0])
        for fw in fws:
          assert len(fw) == fw_size, f"{platform} {ecu}: {len(fw)} {fw_size}"

  def test_carstate_sp_exposes_brake_light_fields(self):
    car_state_sp = structs.CarStateSP()
    assert hasattr(car_state_sp, "brakeLightsOn")
    assert hasattr(car_state_sp, "brakeLightsAvailable")

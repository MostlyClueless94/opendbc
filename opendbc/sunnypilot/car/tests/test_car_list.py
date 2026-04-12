import json
import unittest

from opendbc.sunnypilot.car.platform_list import get_car_list, CAR_LIST_JSON_OUT


class TestCarList(unittest.TestCase):
  def test_generator(self):
    generated_car_list = json.dumps(get_car_list(), indent=2, ensure_ascii=False)
    with open(CAR_LIST_JSON_OUT) as f:
      current_car_list = f.read()

    assert generated_car_list == current_car_list, "Run opendbc/sunnypilot/car/platform_list.py to update the car list"

  def test_subaru_angle_platforms_are_selectable(self):
    with open(CAR_LIST_JSON_OUT) as f:
      car_list = json.load(f)

    expected = {
      "Subaru Outback 2023-25": {
        "platform": "SUBARU_OUTBACK_2023",
        "year": ["2023", "2024", "2025"],
      },
      "Subaru Ascent 2023": {
        "platform": "SUBARU_ASCENT_2023",
        "year": ["2023"],
      },
      "Subaru Crosstrek 2024": {
        "platform": "SUBARU_CROSSTREK_2024",
        "year": ["2024"],
      },
      "Subaru Crosstrek 2025": {
        "platform": "SUBARU_CROSSTREK_2025",
        "year": ["2025"],
      },
    }

    for name, attrs in expected.items():
      assert name in car_list
      assert car_list[name]["brand"] == "subaru"
      assert car_list[name]["make"] == "Subaru"
      assert car_list[name]["platform"] == attrs["platform"]
      assert car_list[name]["year"] == attrs["year"]
      assert car_list[name]["package"] == "All"

  def test_dashcam_only_forester_2022_24_stays_hidden(self):
    with open(CAR_LIST_JSON_OUT) as f:
      car_list = json.load(f)

    assert "Subaru Forester 2022-24" not in car_list

  def test_wiki_only_subaru_rows_stay_hidden_until_data_backed(self):
    with open(CAR_LIST_JSON_OUT) as f:
      car_list = json.load(f)

    for name in (
      "Subaru Impreza 2025",
      "Subaru Legacy 2025",
      "Subaru Solterra 2023",
      "Subaru Crosstrek Hybrid 2026",
      "Subaru Forester 2023",
    ):
      assert name not in car_list

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from modules.api import weather_api


class WeatherApiTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 7, 18, 12, 0, tzinfo=timezone(timedelta(hours=7)))

    def test_city_and_relative_dates(self):
        cases = [
            ("thời tiết ở Đà Nẵng hôm nay", "Da Nang", 0),
            ("thời tiết Đà Nẵng ngày mai", "Da Nang", 1),
            ("thời tiết ngày kia ở Hà Nội", "Hanoi", 2),
            ("thời tiết Huế 3 ngày tới", "Huế", 3),
            ("thời tiết Đà Nẵng hôm qua", "Da Nang", -1),
        ]
        for query, expected_city, delta in cases:
            with self.subTest(query=query):
                city_text = query.split("thời tiết", 1)[1]
                self.assertEqual(weather_api.normalize_city_name(city_text), expected_city)
                self.assertEqual(
                    weather_api.resolve_weather_date(query, now=self.now),
                    self.now.date() + timedelta(days=delta),
                )

    def test_forecast_aggregates_requested_local_day(self):
        target = self.now.date() + timedelta(days=1)
        base = datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc)
        payload = {
            "city": {"name": "Da Nang", "timezone": 7 * 3600},
            "list": [
                {
                    "dt": int((base + timedelta(hours=hour)).timestamp()),
                    "main": {
                        "temp": 28 + hour / 10,
                        "temp_min": 27,
                        "temp_max": 30 + hour / 10,
                        "humidity": 75,
                    },
                    "weather": [{"description": "mưa nhẹ"}],
                    "pop": 0.6,
                }
                for hour in (0, 3, 6, 9, 12, 15)
            ],
        }

        with (
            patch.object(weather_api, "OPENWEATHER_API_KEY", "test"),
            patch.object(weather_api, "get_now", return_value=self.now),
            patch.object(
                weather_api,
                "_geocode_vietnam",
                return_value={"lat": 16.05, "lon": 108.2, "name": "Đà Nẵng, Việt Nam"},
            ),
            patch.object(weather_api, "_request_json", return_value=payload),
        ):
            result = weather_api.get_weather_forecast("Đà Nẵng", target)

        self.assertNotIn("error", result)
        self.assertEqual(result["date"], "19/07/2026")
        self.assertEqual(result["rain_probability"], 60)
        self.assertEqual(result["desc"], "mưa nhẹ")

    def test_geocoding_is_restricted_to_vietnam(self):
        payload = [
            {"name": "Da Nang", "country": "US", "lat": 1, "lon": 2},
            {
                "name": "Da Nang",
                "local_names": {"vi": "Đà Nẵng"},
                "country": "VN",
                "lat": 16.05,
                "lon": 108.2,
            },
        ]
        with patch.object(weather_api, "_request_json", return_value=payload):
            result = weather_api._geocode_vietnam("Đà Nẵng")

        self.assertEqual(result["lat"], 16.05)
        self.assertIn("Việt Nam", result["name"])

    def test_osm_fallback_resolves_district_name(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [
            {
                "name": "Xã Núi Thành",
                "display_name": "Xã Núi Thành, Thành phố Đà Nẵng, Việt Nam",
                "lat": "15.4342062",
                "lon": "108.6559260",
                "category": "boundary",
                "type": "administrative",
                "address": {"country_code": "vn"},
            }
        ]
        weather_api._geocode_vietnam_osm.cache_clear()
        with (
            patch.object(weather_api, "_request_json", return_value=[]),
            patch.object(weather_api.requests, "get", return_value=response) as request,
        ):
            result = weather_api._geocode_vietnam("Núi Thành, Đà Nẵng hôm nay")

        self.assertEqual(result["geocoder_source"], "OpenStreetMap")
        self.assertAlmostEqual(result["lat"], 15.4342062)
        self.assertEqual(request.call_args.kwargs["params"]["q"], "Núi Thành")


if __name__ == "__main__":
    unittest.main()

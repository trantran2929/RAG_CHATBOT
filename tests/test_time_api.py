import unittest
from datetime import datetime

import pytz

from modules.api.time_api import add_time, answer_time_query
from modules.nodes.processor import processor_instance


class TimeApiTests(unittest.TestCase):
    def setUp(self):
        tz = pytz.timezone("Asia/Ho_Chi_Minh")
        self.now = tz.localize(datetime(2026, 7, 18, 14, 30, 45))

    def test_time_intents_do_not_collide_with_tickers(self):
        expected = {
            "mấy giờ rồi": "time",
            "hôm nay là thứ mấy": "time",
            "ngày mai là ngày bao nhiêu": "time",
            "còn bao lâu đến cuối năm": "time",
            "giờ hiện tại": "time",
            "hôm qua là ngày nào": "time",
            "ngày 24 tháng 7 năm 2026 là thứ  mấy": "time",
            "tin tức hôm nay": "rag",
            "thời tiết Đà Nẵng ngày mai": "weather",
            "giá FPT ngày mai": "stock",
        }
        for query, intent in expected.items():
            with self.subTest(query=query):
                self.assertEqual(processor_instance.detect_intent(query), intent)

    def test_relative_time_answers(self):
        self.assertIn("Chủ Nhật", answer_time_query("ngày mai là thứ mấy", self.now))
        self.assertIn("17 tháng 07 năm 2026", answer_time_query("hôm qua là ngày nào", self.now))
        self.assertIn("21 tháng 07 năm 2026", answer_time_query("3 ngày nữa là ngày nào", self.now))
        self.assertIn("14:30:45", answer_time_query("mấy giờ rồi", self.now))
        self.assertEqual(
            answer_time_query("ngày 25/7/2026 là thứ mấy", self.now),
            "Ngày 25/07/2026 là Thứ Bảy.",
        )
        self.assertEqual(
            answer_time_query("ngày 24 tháng 7 năm 2026 là thứ  mấy", self.now),
            "Ngày 24/07/2026 là Thứ Sáu.",
        )
        self.assertEqual(
            answer_time_query("ngày 24 háng 7 năm 2026 là thứ mấy", self.now),
            "Ngày 24/07/2026 là Thứ Sáu.",
        )
        self.assertEqual(
            answer_time_query("ngày 24 thg 7 năm 2026 là thứ mấy", self.now),
            "Ngày 24/07/2026 là Thứ Sáu.",
        )
        self.assertEqual(
            answer_time_query("ngày 24 t 7 năm 2026 lag thứ mấy", self.now),
            "Ngày 24/07/2026 là Thứ Sáu.",
        )
        self.assertEqual(
            answer_time_query("ngay 25 th 7 nam 2026 là th mấy", self.now),
            "Ngày 25/07/2026 là Thứ Bảy.",
        )
        self.assertEqual(
            answer_time_query("ngày mai lag ngay may", self.now),
            "Ngày mai là ngày 19/07/2026.",
        )

    def test_add_year_from_leap_day(self):
        tz = pytz.timezone("Asia/Ho_Chi_Minh")
        leap_day = tz.localize(datetime(2024, 2, 29, 10, 0))
        self.assertEqual(add_time(leap_day, 1, "năm").date().isoformat(), "2025-02-28")


if __name__ == "__main__":
    unittest.main()

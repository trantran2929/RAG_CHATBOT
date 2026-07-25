import unittest

from modules.core.state import GlobalState
from modules.nodes.processor import processor_instance


class ProcessorTypoTests(unittest.TestCase):
    def test_abbreviated_date_and_typo_are_corrected(self):
        result = self.process("ngày 24 t 7 năm 2026 lag thứ mấy")
        self.assertEqual(
            result.corrected_query,
            "ngày 24 tháng 7 năm 2026 là thứ mấy",
        )
        self.assertEqual(result.intent, "time")

        result = self.process("ngay 25 th 7 nam 2026 là th mấy")
        self.assertEqual(
            result.corrected_query,
            "ngày 25 tháng 7 năm 2026 là thứ mấy",
        )
        self.assertEqual(result.intent, "time")

    def process(self, query: str) -> GlobalState:
        return processor_instance.process_query(GlobalState(user_query=query))

    def test_weather_phrase_typos_are_corrected(self):
        state = self.process("thơi tiet Đà Nẵng ngài mai")
        self.assertEqual(state.intent, "weather")
        self.assertEqual(state.corrected_query, "thời tiết Đà Nẵng ngày mai")
        self.assertEqual(state.user_query, "thơi tiet Đà Nẵng ngài mai")

    def test_missing_diacritics_still_match_finance_semantics(self):
        state = self.process("gia co phieu FPT hom nay")
        self.assertEqual(state.intent, "stock")
        self.assertIn("FPT", state.tickers)

    def test_correction_does_not_change_numbers_or_tickers(self):
        state = self.process("gia co phieu VCB ngay 24/7/2026")
        self.assertIn("VCB", state.corrected_query)
        self.assertIn("24/7/2026", state.corrected_query)

    def test_common_word_ticker_requires_explicit_context(self):
        self.assertNotIn("THU", processor_instance.detect_tickers("hôm nay là thứ mấy"))
        self.assertIn("THU", processor_instance.detect_tickers("giá mã THU"))


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import pandas as pd

from modules.api.forecast_api import (
    _format_next_session_verbose,
    format_forecast_text,
    get_full_forecast_answer,
    get_intraday_step_forecast_answer,
)
from modules.core.state import GlobalState
from modules.nodes.router import route_intent


def am_pack():
    return {
        "mode": "next_session",
        "next_session": "AM",
        "target_day": "2026-07-22",
        "open_band": {"px_mean": 67100, "px_lo": 65000, "px_hi": 69000},
        "open_direction": "tăng",
        "open_gap_pct": 0.15,
        "open_confidence": "uncertain",
        "gap": {"model_trained_through": "2026-07-21"},
    }


class ForecastApiTests(unittest.TestCase):
    def test_am_output_uses_vnd_and_model_freshness(self):
        text = _format_next_session_verbose("FPT", am_pack())
        self.assertIn("67,100.00 VNĐ", text)
        self.assertIn("Giá đóng cửa", text)
        self.assertNotIn("Giá mở cửa", text)
        self.assertIn("chưa chắc chắn", text)
        self.assertIn("model học tới phiên 2026-07-21", text)

    def test_next_close_pack_displays_cost_adjusted_signal(self):
        pack = {
            "mode": "next_session",
            "next_session": "AM",
            "target_day": "2026-07-27",
            "close_band": {"px_mean": 68000, "px_lo": 66000, "px_hi": 70000},
            "close_direction": "tăng",
            "close_return_pct": 1.2,
            "close_confidence": "uncertain",
            "trade_signal": "BUY",
        }
        text = _format_next_session_verbose("FPT", pack)
        self.assertIn("Giá đóng cửa", text)
        self.assertIn("**BUY**", text)

    def test_missing_band_does_not_crash(self):
        pack = {"mode": "next_session", "next_session": "AM"}
        self.assertEqual(_format_next_session_verbose("FPT", pack), "")
        self.assertIn("thiếu dải giá", format_forecast_text("FPT", {"mode": "broken"}))

    def test_pm_output_discloses_daily_fallback(self):
        pack = {
            "mode": "next_session",
            "next_session": "PM",
            "target_day": "2026-07-21",
            "pm_band": {"px_mean": 67500, "px_lo": 66000, "px_hi": 68500},
            "pm_direction": "tăng",
            "pm_gap_pct": 0.3,
            "pm_confidence": "low",
            "model": "AutoReg",
            "base_from": "last_close_daily",
        }
        text = _format_next_session_verbose("FPT", pack)
        self.assertIn("AutoReg", text)
        self.assertIn("thiếu intraday", text)
        self.assertIn("thấp", text)

    def test_intraday_fallback_is_not_presented_as_live_intraday(self):
        pack = {
            "mode": "in_session",
            "session": "AM",
            "next_step_dir": "giảm",
            "step_confidence": "low",
            "last_px": 67100,
            "path_pred": pd.Series([67000, 66900, 66800]),
            "source_used": "daily_fallback",
            "model": "AutoReg",
            "error": "Fallback daily (thiếu intraday).",
        }
        with patch("modules.api.forecast_api.smart_predict", return_value=pack):
            text = get_intraday_step_forecast_answer("FPT")
        self.assertIn("fallback dữ liệu ngày", text)
        self.assertIn("Lưu ý dữ liệu", text)

    def test_intraday_error_pack_returns_clear_error(self):
        pack = {
            "mode": "in_session",
            "next_step_dir": None,
            "error": "Không đủ dữ liệu cả intraday lẫn daily.",
        }
        with patch("modules.api.forecast_api.smart_predict", return_value=pack):
            text = get_intraday_step_forecast_answer("FPT")
        self.assertIn("Chưa thể dự báo nội phiên", text)
        self.assertIn("Không đủ dữ liệu", text)

    def test_full_answer_does_not_call_next_session_twice(self):
        with patch("modules.api.forecast_api.smart_predict", return_value=am_pack()), patch(
            "modules.api.forecast_api.predict_next_session"
        ) as next_mock:
            text = get_full_forecast_answer("FPT")
        next_mock.assert_not_called()
        self.assertIn("67,100.00 VNĐ", text)

    def test_router_connects_forecast_intent_to_api(self):
        state = GlobalState(
            user_query="dự báo phiên tới của FPT",
            corrected_query="dự báo phiên tới của FPT",
            intent="forecast",
            tickers=["FPT"],
        )
        with patch(
            "modules.nodes.router.get_next_session_forecast_answer",
            return_value="FORECAST_OK",
        ):
            result = route_intent(state)
        self.assertEqual(result.route_to, "api")
        self.assertEqual(result.api_type, "forecast_next_session")
        self.assertEqual(result.api_response, "FORECAST_OK")


if __name__ == "__main__":
    unittest.main()

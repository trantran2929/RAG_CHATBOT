import datetime as dt
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules.ML.pipeline import (
    MODEL_SCHEMA_VERSION,
    _build_exog_row_for_forecast,
    _fit_autoreg_returns,
    _model_is_stale,
    is_vn_holiday,
    signal_from_return,
)
from modules.ML.backtest import backtest_gap_model
from modules.api.stock_api import _stock_prices_to_vnd


class MLPipelineTests(unittest.TestCase):
    def test_legacy_and_old_models_are_stale(self):
        latest = dt.date(2026, 7, 17)
        self.assertTrue(_model_is_stale({}, latest))
        self.assertTrue(
            _model_is_stale(
                {"schema_version": MODEL_SCHEMA_VERSION, "last_train_date": "2025-10-22"},
                latest,
            )
        )
        self.assertFalse(
            _model_is_stale(
                {"schema_version": MODEL_SCHEMA_VERSION, "last_train_date": "2026-07-17"},
                latest,
            )
        )

    def test_autoreg_is_a_fitted_model_not_fixed_zero(self):
        idx = pd.bdate_range("2026-01-01", periods=80)
        returns = 0.001 + 0.004 * np.sin(np.arange(80) / 5)
        prices = pd.Series(50_000 * np.exp(np.cumsum(returns)), index=idx)
        result = _fit_autoreg_returns(prices, steps=3)
        self.assertEqual(len(result["returns"]), 3)
        self.assertGreater(result["sigma"], 0)
        self.assertGreaterEqual(result["lags"], 1)

    def test_fixed_vietnam_holiday_fallback(self):
        self.assertTrue(is_vn_holiday(dt.date(2026, 9, 2)))

    def test_stock_history_prices_are_exposed_as_vnd(self):
        raw = pd.DataFrame({"open": [67.0], "high": [68.0], "low": [66.0], "close": [67.1]})
        result = _stock_prices_to_vnd(raw)
        self.assertEqual(result.iloc[0]["close"], 67_100)

    def test_backtest_uses_only_prices_before_target(self):
        idx = pd.bdate_range("2025-01-01", periods=180)
        returns = 0.0005 + 0.005 * np.sin(np.arange(180) / 7)
        close = pd.Series(40_000 * np.exp(np.cumsum(returns)), index=idx)
        with patch("modules.ML.backtest.get_close_series", return_value=close), patch(
            "modules.ML.backtest._fit_predict_fold", return_value=0.001
        ):
            result = backtest_gap_model("FPT", test_days=10, use_exog=False)
        self.assertEqual(len(result), 10)
        self.assertTrue((pd.to_datetime(result["train_end"]) < result.index).all())
        self.assertIn("directional_accuracy", result.attrs["metrics"])
        self.assertEqual(
            result.attrs["metrics"]["target"],
            "next_session_close_to_close_log_return",
        )
        self.assertIn("baselines", result.attrs["metrics"])

    def test_signal_respects_cost_and_no_trade_zone(self):
        self.assertEqual(
            signal_from_return(0.004, round_trip_cost_bps=20, signal_buffer_bps=10),
            "BUY",
        )
        self.assertEqual(
            signal_from_return(-0.004, round_trip_cost_bps=20, signal_buffer_bps=10),
            "SELL",
        )
        self.assertEqual(
            signal_from_return(0.002, round_trip_cost_bps=20, signal_buffer_bps=10),
            "NO_TRADE",
        )

    def test_forecast_price_lag_uses_latest_observed_return(self):
        idx = pd.bdate_range("2026-01-01", periods=40)
        returns = np.linspace(0.001, 0.004, len(idx))
        close = pd.Series(50_000 * np.exp(np.cumsum(returns)), index=idx)
        last_idx = idx[-1]
        target_idx = pd.bdate_range(last_idx, periods=2)[-1]
        expected_latest = float(np.log(close.iloc[-1] / close.iloc[-2]))

        with patch("modules.ML.pipeline.build_news_features", return_value=pd.DataFrame()), patch(
            "modules.ML.pipeline.get_close_series", return_value=close
        ):
            row = _build_exog_row_for_forecast(
                "FPT",
                last_idx,
                ["ret_lag1"],
                [],
                {},
                target_idx=target_idx,
            )

        self.assertEqual(pd.Timestamp(row.index[0]), pd.Timestamp(target_idx))
        self.assertAlmostEqual(float(row.iloc[0]["ret_lag1"]), expected_latest)


if __name__ == "__main__":
    unittest.main()

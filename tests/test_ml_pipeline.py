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
from modules.ML.backtest_config import BacktestConfig
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

    def test_validation_and_final_test_do_not_overlap(self):
        idx = pd.bdate_range("2024-01-01", periods=400)
        synthetic_returns = 0.0005 + 0.004 * np.sin(np.arange(400) / 9)
        close = pd.Series(
            40_000 * np.exp(np.cumsum(synthetic_returns)),
            index=idx,
        )
        config = BacktestConfig(validation_days=60, final_test_days=40)
        with patch("modules.ML.backtest.get_close_series", return_value=close), patch(
            "modules.ML.backtest._fit_predict_fold", return_value=0.001
        ):
            validation = backtest_gap_model(
                "FPT",
                segment="validation",
                config=config,
                use_exog=False,
            )
            final_test = backtest_gap_model(
                "FPT",
                segment="final_test",
                config=config,
                use_exog=False,
            )

        self.assertEqual(len(validation), 60)
        self.assertEqual(len(final_test), 40)
        self.assertLess(validation.index.max(), final_test.index.min())
        self.assertEqual(
            validation.attrs["metrics"]["evaluation_segment"],
            "validation",
        )
        self.assertEqual(
            final_test.attrs["metrics"]["evaluation_segment"],
            "final_test",
        )

    def test_explicit_date_range_is_inclusive(self):
        idx = pd.bdate_range("2025-01-01", periods=180)
        close = pd.Series(
            50_000 * np.exp(np.cumsum(np.full(len(idx), 0.001))),
            index=idx,
        )
        start_date = idx[-10].date().isoformat()
        end_date = idx[-6].date().isoformat()
        with patch("modules.ML.backtest.get_close_series", return_value=close), patch(
            "modules.ML.backtest._fit_predict_fold", return_value=0.001
        ):
            result = backtest_gap_model(
                "FPT",
                start_date=start_date,
                end_date=end_date,
                use_exog=False,
            )

        self.assertEqual(len(result), 5)
        self.assertEqual(result.index[0].date().isoformat(), start_date)
        self.assertEqual(result.index[-1].date().isoformat(), end_date)

    def test_named_segment_cannot_be_combined_with_dates(self):
        idx = pd.bdate_range("2025-01-01", periods=300)
        close = pd.Series(
            50_000 * np.exp(np.cumsum(np.full(len(idx), 0.001))),
            index=idx,
        )
        with patch("modules.ML.backtest.get_close_series", return_value=close):
            with self.assertRaises(ValueError):
                backtest_gap_model(
                    "FPT",
                    segment="validation",
                    start_date=idx[-20].date().isoformat(),
                    use_exog=False,
                )

    def test_risk_filter_preserves_raw_signal_and_blocks_bear_buy(self):
        idx = pd.bdate_range("2024-01-01", periods=400)
        close = pd.Series(50_000 * np.exp(np.arange(400) * 0.001), index=idx)
        risk = pd.DataFrame(
            {
                "ret_1": 0.001,
                "ret_5": -0.01,
                "distance_ma20": -0.02,
                "volatility_5": 0.01,
                "volatility_20": 0.02,
                "previous_shock": 0.0,
                "vnindex_ret_5": -0.01,
                "vnindex_ret_60": -0.10,
                "vnindex_distance_ma200": -0.05,
                "vnindex_volatility_20": 0.02,
            },
            index=idx,
        )
        labeled = risk.copy()
        labeled["market_regime"] = "BEAR"
        labeled["high_volatility"] = False
        config = BacktestConfig(validation_days=10, final_test_days=10)
        with patch(
            "modules.ML.backtest.get_close_series", side_effect=[close, close]
        ), patch(
            "modules.ML.backtest._fit_predict_fold", return_value=0.01
        ), patch(
            "modules.ML.backtest.build_risk_features", return_value=risk
        ), patch(
            "modules.ML.backtest.fit_volatility_threshold", return_value=0.03
        ), patch(
            "modules.ML.backtest.label_market_regime", return_value=labeled
        ):
            result = backtest_gap_model(
                "FPT",
                segment="validation",
                config=config,
                use_exog=False,
                apply_risk_filtering=True,
            )

        self.assertTrue(result["raw_signal"].eq("BUY").all())
        self.assertTrue(result["final_signal"].eq("NO_TRADE").all())
        self.assertTrue(result["filter_reason"].str.contains("bear_regime").all())
        metrics = result.attrs["metrics"]
        self.assertEqual(metrics["raw_strategy"]["trade_count"], 10)
        self.assertEqual(metrics["filtered_strategy"]["trade_count"], 0)
        self.assertEqual(metrics["filter_report"]["blocked_count"], 10)
        self.assertIn("momentum_report", metrics)

    def test_filtered_final_test_requires_locked_volatility_threshold(self):
        idx = pd.bdate_range("2024-01-01", periods=400)
        close = pd.Series(50_000 * np.exp(np.arange(400) * 0.001), index=idx)
        risk = pd.DataFrame(index=idx)
        with patch(
            "modules.ML.backtest.get_close_series", side_effect=[close, close]
        ), patch("modules.ML.backtest.build_risk_features", return_value=risk):
            with self.assertRaises(ValueError):
                backtest_gap_model(
                    "FPT",
                    segment="final_test",
                    config=BacktestConfig(validation_days=10, final_test_days=10),
                    use_exog=False,
                    apply_risk_filtering=True,
                )

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

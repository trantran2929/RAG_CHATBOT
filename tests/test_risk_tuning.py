import unittest

import numpy as np
import pandas as pd

from modules.ML.backtest_config import BacktestConfig
from modules.ML.risk_tuning import (
    apply_config_to_validation,
    candidate_configs,
    tune_risk_filter,
)


def _validation_result():
    index = pd.bdate_range("2026-01-01", periods=20)
    result = pd.DataFrame(
        {
            "raw_signal": ["BUY"] * 20,
            "raw_position": [1] * 20,
            "raw_net_return": np.linspace(-0.01, 0.01, 20),
            "actual": np.linspace(-0.02, 0.02, 20),
            "previous_shock": [0.01] * 10 + [0.07] * 10,
            "vnindex_volatility_20": np.linspace(0.01, 0.05, 20),
            "market_regime": ["BULL"] * 10 + ["BEAR"] * 10,
            "distance_ma20": [0.02] * 10 + [-0.02] * 10,
            "ret_5": [0.01] * 10 + [-0.01] * 10,
            "high_volatility": False,
        },
        index=index,
    )
    result.attrs["metrics"] = {
        "evaluation_segment": "validation",
        "round_trip_cost_bps": 35.0,
        "risk_calibration_volatility": list(np.linspace(0.005, 0.04, 100)),
        "shock_history_before_evaluation": [0.01, 0.01],
    }
    return result


class RiskTuningTests(unittest.TestCase):
    def test_grid_is_intentionally_bounded(self):
        self.assertEqual(len(candidate_configs(BacktestConfig())), 21)

    def test_validation_config_is_applied_without_changing_raw_predictions(self):
        source = _validation_result()
        filtered, locked = apply_config_to_validation(source, BacktestConfig())
        pd.testing.assert_series_equal(filtered["raw_signal"], source["raw_signal"])
        self.assertIsNotNone(locked.fitted_volatility_threshold)
        self.assertGreater((filtered["final_signal"] == "NO_TRADE").sum(), 0)

    def test_tuning_returns_one_locked_selection(self):
        selected, table = tune_risk_filter(
            _validation_result(),
            candidate_configs(BacktestConfig()),
        )
        self.assertEqual(int(table["selected"].sum()), 1)
        self.assertIsNotNone(selected.fitted_volatility_threshold)

    def test_tuning_rejects_final_test_rows(self):
        result = _validation_result()
        result.attrs["metrics"]["evaluation_segment"] = "final_test"
        with self.assertRaises(ValueError):
            apply_config_to_validation(result, BacktestConfig())

    def test_grid_contains_unfiltered_baseline(self):
        configs = candidate_configs(BacktestConfig())

        baselines = [
            config for config in configs
            if not config.use_bear_filter
            and not config.use_ma20_filter
            and not config.use_volatility_filter
            and not config.use_shock_filter
        ]

        self.assertEqual(len(baselines), 1)

    def test_zero_cooldown_does_not_use_rolling_window_zero(self):
        config = BacktestConfig(cooldown_sessions=0)

        filtered, locked = apply_config_to_validation(
            _validation_result(),
            config,
        )

        self.assertFalse(filtered["shock_cooldown"].any())
        self.assertEqual(locked.cooldown_sessions, 0)


if __name__ == "__main__":
    unittest.main()

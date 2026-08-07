import unittest

import pandas as pd

from modules.ML.backtest_config import BacktestConfig
from modules.ML.risk_filter import apply_risk_filter, shock_cooldown_mask


class RiskFilterTests(unittest.TestCase):
    def setUp(self):
        self.config = BacktestConfig(shock_threshold=0.06)

    def test_bear_regime_blocks_buy_and_records_reason(self):
        result = apply_risk_filter(
            "BUY",
            pd.Series({"market_regime": "BEAR"}),
            self.config,
        )
        self.assertEqual(result.final_signal, "NO_TRADE")
        self.assertIn("bear_regime", result.reasons)

    def test_safe_buy_is_preserved(self):
        row = pd.Series(
            {
                "market_regime": "BULL",
                "distance_ma20": 0.02,
                "ret_5": 0.01,
                "high_volatility": False,
                "previous_shock": 0.01,
            }
        )
        result = apply_risk_filter("BUY", row, self.config)
        self.assertEqual(result.final_signal, "BUY")
        self.assertEqual(result.reasons, [])

    def test_non_buy_signal_is_not_rewritten(self):
        result = apply_risk_filter("SELL", pd.Series(dtype=object), self.config)
        self.assertEqual(result.final_signal, "SELL")

    def test_multiple_reasons_are_auditable(self):
        row = pd.Series(
            {
                "market_regime": "BEAR",
                "distance_ma20": -0.1,
                "ret_5": -0.08,
                "high_volatility": True,
                "previous_shock": 0.07,
            }
        )
        result = apply_risk_filter("BUY", row, self.config)
        self.assertEqual(len(result.reasons), 3)
        self.assertIn("|", result.reason_text)

    def test_cooldown_reason_blocks_buy(self):
        row = pd.Series({"market_regime": "BULL", "shock_cooldown": True})
        result = apply_risk_filter("BUY", row, self.config)
        self.assertEqual(result.final_signal, "NO_TRADE")
        self.assertIn("shock_cooldown", result.reasons)

    def test_cooldown_never_reads_target_return(self):
        index = pd.bdate_range("2026-01-01", periods=4)
        returns = pd.Series([0.0, 0.07, 0.0, 0.0], index=index)
        mask = shock_cooldown_mask(returns, threshold=0.06, sessions=1)
        self.assertFalse(bool(mask.iloc[1]))
        self.assertTrue(bool(mask.iloc[2]))
        self.assertFalse(bool(mask.iloc[3]))


if __name__ == "__main__":
    unittest.main()

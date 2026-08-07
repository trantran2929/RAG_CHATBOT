import unittest

import numpy as np
import pandas as pd

from modules.ML.risk_features import (
    build_risk_features,
    fit_volatility_threshold,
    label_market_regime,
)


def _prices(periods=320, start="2024-01-01", drift=0.001):
    index = pd.bdate_range(start, periods=periods)
    returns = drift + 0.004 * np.sin(np.arange(periods) / 11)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=index)


class RiskFeatureTests(unittest.TestCase):
    def test_target_row_uses_previous_session_return(self):
        stock = _prices()
        market = _prices(drift=0.0005)
        features = build_risk_features(stock, market)
        target = stock.index[-1]
        expected = np.log(stock.iloc[-2] / stock.iloc[-3])
        self.assertAlmostEqual(features.loc[target, "ret_1"], expected)

    def test_future_stock_change_does_not_change_earlier_features(self):
        stock = _prices()
        market = _prices(drift=0.0005)
        original = build_risk_features(stock, market)
        changed = stock.copy()
        changed.iloc[-1] *= 10
        recalculated = build_risk_features(changed, market)
        pd.testing.assert_frame_equal(original.iloc[:-1], recalculated.iloc[:-1])

    def test_market_alignment_never_backfills_future_observation(self):
        stock = _prices(periods=40)
        market = _prices(periods=35, start=stock.index[5].date().isoformat())
        features = build_risk_features(stock, market)
        self.assertTrue(features.iloc[:5]["vnindex_ret_5"].isna().all())

    def test_volatility_threshold_excludes_cutoff_and_future(self):
        index = pd.bdate_range("2025-01-01", periods=10)
        frame = pd.DataFrame(
            {"vnindex_volatility_20": np.arange(1, 11, dtype=float)},
            index=index,
        )
        cutoff = index[5]
        threshold = fit_volatility_threshold(frame, 0.5, end_date=cutoff)
        self.assertEqual(threshold, 3.0)

    def test_market_regime_and_high_volatility_are_independent(self):
        index = pd.bdate_range("2025-01-01", periods=3)
        frame = pd.DataFrame(
            {
                "vnindex_distance_ma200": [0.1, -0.1, 0.1],
                "vnindex_ret_60": [0.2, -0.2, -0.1],
                "vnindex_volatility_20": [0.01, 0.05, 0.01],
            },
            index=index,
        )
        result = label_market_regime(frame, volatility_threshold=0.03)
        self.assertEqual(result.iloc[0]["market_regime"], "BULL")
        self.assertEqual(result.iloc[1]["market_regime"], "BEAR")
        self.assertEqual(result.iloc[2]["market_regime"], "SIDEWAYS")
        self.assertTrue(bool(result.iloc[1]["high_volatility"]))


if __name__ == "__main__":
    unittest.main()

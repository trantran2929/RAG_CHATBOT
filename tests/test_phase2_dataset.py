import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modules.ML.phase2_config import Phase2DatasetConfig
from modules.ML.phase2_dataset import (
    assign_time_split,
    audit_ohlcv,
    build_phase2_dataset,
    build_symbol_dataset,
)
from modules.ML.phase2_features import (
    build_market_features,
    build_stock_features,
    make_target,
)


def _ohlcv(periods=520, start="2024-01-01", slope=0.001):
    index = pd.bdate_range(start, periods=periods)
    close = 50_000.0 * np.exp(np.arange(periods) * slope)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0 + np.arange(periods) * 100.0,
        },
        index=index,
    )


class Phase2DatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = Phase2DatasetConfig(
            symbols=("FPT",), lookback_days=520,
            validation_days=125, final_test_days=125,
            min_train_rows=60,
        )

    def test_target_uses_configured_after_cost_threshold(self):
        close = pd.Series([100.0, 100.6, 100.7, 100.0])
        target = make_target(close, 0.005)
        self.assertEqual(target.loc[1, "target"], "UP")
        self.assertEqual(target.loc[2, "target"], "NO_TRADE")
        self.assertEqual(target.loc[3, "target"], "DOWN")

    def test_target_day_change_does_not_change_its_features(self):
        source = _ohlcv()
        target_date = source.index[300]
        before = build_stock_features(source).loc[target_date].copy()
        changed = source.copy()
        changed.loc[target_date:, "close"] *= 1.5
        after = build_stock_features(changed).loc[target_date]
        pd.testing.assert_series_equal(before, after)

    def test_future_market_observation_is_never_backfilled(self):
        stock = _ohlcv()
        market = _ohlcv(slope=0.0005)
        target_date = stock.index[300]
        before = build_market_features(stock, market).loc[target_date].copy()
        changed = market.copy()
        changed.loc[target_date:, "close"] *= 2.0
        after = build_market_features(stock, changed).loc[target_date]
        pd.testing.assert_series_equal(before, after)

    def test_splits_are_disjoint_and_have_locked_lengths(self):
        index = pd.bdate_range("2024-01-01", periods=320)
        split = assign_time_split(index, self.config)
        self.assertEqual(int(split.eq("validation").sum()), 125)
        self.assertEqual(int(split.eq("final_test").sum()), 125)
        self.assertEqual(int(split.eq("train").sum()), 70)

    def test_final_targets_are_masked_by_default(self):
        dataset = build_symbol_dataset("FPT", _ohlcv(), _ohlcv(slope=0.0005), self.config)
        final = dataset[dataset["split"] == "final_test"]
        validation = dataset[dataset["split"] == "validation"]
        self.assertEqual(len(final), 125)
        self.assertTrue(final["target"].isna().all())
        self.assertTrue(final["future_return"].isna().all())
        self.assertTrue(validation["target"].notna().all())

    def test_audit_reports_duplicate_and_suspicious_return(self):
        frame = _ohlcv(20)
        frame.iloc[10, frame.columns.get_loc("close")] *= 1.2
        duplicated = pd.concat([frame, frame.iloc[[0]]])
        report = audit_ohlcv("FPT", duplicated)
        self.assertGreater(report["duplicate_dates"], 0)
        self.assertGreater(report["absolute_return_over_10pct"], 0)

    def test_builder_is_reproducible(self):
        stock = _ohlcv()
        market = _ohlcv(slope=0.0005)

        def provider(symbol, days):
            return market.copy() if symbol == "VNINDEX" else stock.copy()

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            d1, _, m1 = build_phase2_dataset(self.config, Path(first), provider=provider)
            d2, _, m2 = build_phase2_dataset(self.config, Path(second), provider=provider)
            pd.testing.assert_frame_equal(d1, d2)
            self.assertEqual(m1["dataset_sha256"], m2["dataset_sha256"])
            self.assertEqual(
                (Path(first) / "phase2_dataset.csv").read_bytes(),
                (Path(second) / "phase2_dataset.csv").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()

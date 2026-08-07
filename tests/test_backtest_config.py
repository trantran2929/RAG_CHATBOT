import json
import tempfile
import unittest
from pathlib import Path

from modules.ML.backtest_config import (
    BacktestConfig,
    load_backtest_config,
    save_backtest_config,
)


class BacktestConfigTests(unittest.TestCase):
    def test_config_round_trip_preserves_locked_values(self):
        config = BacktestConfig(
            validation_days=80,
            final_test_days=60,
            round_trip_cost_bps=50,
            signal_buffer_bps=20,
            shock_threshold=0.065,
            volatility_percentile=0.90,
            cooldown_sessions=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locked_config.json"
            saved_path = save_backtest_config(config, path)
            loaded = load_backtest_config(saved_path)

        self.assertEqual(config, loaded)
        self.assertTrue(config.__dataclass_params__.frozen)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            BacktestConfig(validation_days=4)
        with self.assertRaises(ValueError):
            BacktestConfig(round_trip_cost_bps=-1)
        with self.assertRaises(ValueError):
            BacktestConfig(volatility_percentile=1.0)

    def test_loader_rejects_non_object_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_backtest_config(path)


if __name__ == "__main__":
    unittest.main()

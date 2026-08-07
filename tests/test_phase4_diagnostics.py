import unittest

import pandas as pd

from modules.ML.phase4_diagnostics import _diagnostic_metrics


class Phase4DiagnosticTests(unittest.TestCase):
    def test_direction_accuracy_is_computed_only_on_actions(self):
        frame = pd.DataFrame({
            "signal": ["BUY", "SELL", "NO_TRADE"],
            "target": ["UP", "UP", "DOWN"],
            "future_return": [0.02, 0.01, -0.02],
        })
        metrics = _diagnostic_metrics(frame, overall_accuracy=0.5)
        self.assertEqual(metrics["actions"], 2)
        self.assertAlmostEqual(metrics["action_direction_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["buy_direction_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["sell_direction_accuracy"], 0.0)

    def test_small_group_is_not_flagged(self):
        frame = pd.DataFrame({
            "signal": ["BUY"] * 5, "target": ["DOWN"] * 5,
            "future_return": [-0.01] * 5,
        })
        metrics = _diagnostic_metrics(frame, overall_accuracy=0.5)
        self.assertFalse(metrics["diagnostic_flag"])


if __name__ == "__main__":
    unittest.main()

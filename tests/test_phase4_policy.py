import unittest

import numpy as np

from modules.ML.phase4_policy import probability_to_signal, wilson_lower_bound


class Phase4PolicyTests(unittest.TestCase):
    def test_signal_requires_threshold_and_winning_margin(self):
        probabilities = np.array([
            [0.10, 0.20, 0.70],
            [0.65, 0.20, 0.15],
            [0.34, 0.33, 0.33],
            [0.40, 0.39, 0.21],
        ])
        signals = probability_to_signal(probabilities, threshold=0.36, margin=0.03)
        self.assertEqual(signals.tolist(), ["BUY", "SELL", "NO_TRADE", "NO_TRADE"])

    def test_wilson_rewards_evidence_not_tiny_perfect_sample(self):
        self.assertGreater(wilson_lower_bound(60, 100), wilson_lower_bound(3, 3))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from modules.ML.phase3_analysis import expected_calibration_error, multiclass_brier


class Phase3AnalysisTests(unittest.TestCase):
    def test_perfect_probabilities_have_zero_brier_and_ece(self):
        actual = np.array(["DOWN", "NO_TRADE", "UP"])
        probabilities = np.eye(3)
        self.assertAlmostEqual(multiclass_brier(actual, probabilities), 0.0)
        self.assertAlmostEqual(expected_calibration_error(actual, probabilities), 0.0)

    def test_uncertain_probabilities_have_positive_brier(self):
        actual = np.array(["DOWN", "UP"])
        probabilities = np.full((2, 3), 1 / 3)
        self.assertGreater(multiclass_brier(actual, probabilities), 0.0)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from modules.ML.phase4_ensemble import blend_probabilities


class Phase4EnsembleTests(unittest.TestCase):
    def test_blend_probabilities_remain_normalized(self):
        xgb = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])
        blended = blend_probabilities(xgb, np.array([0.01, -0.01]), 0.25)
        np.testing.assert_allclose(blended.sum(axis=1), 1.0)
        self.assertGreater(blended[0, 2], xgb[0, 2])
        self.assertGreater(blended[1, 0], xgb[1, 0])

    def test_invalid_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            blend_probabilities(np.array([[0.2, 0.3, 0.5]]), np.array([0.1]), 1.5)


if __name__ == "__main__":
    unittest.main()

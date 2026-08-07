import unittest

from modules.ML.phase3_selection import choose_feature_set


class Phase3SelectionTests(unittest.TestCase):
    def _summary(self, cv_bal=0.36, cv_f1=0.35, val_bal=0.36, val_loss=1.10):
        return {
            "cv_mean_balanced_accuracy": cv_bal, "cv_mean_macro_f1": cv_f1,
            "validation_balanced_accuracy": val_bal, "validation_log_loss": val_loss,
        }

    def test_compact_is_selected_only_when_every_rule_passes(self):
        selected, failed = choose_feature_set(self._summary(), self._summary(0.37, 0.36, 0.358, 1.105))
        self.assertEqual(selected, "compact")
        self.assertEqual(failed, [])

    def test_full_is_selected_when_compact_cv_is_worse(self):
        selected, failed = choose_feature_set(self._summary(), self._summary(0.35, 0.36, 0.37, 1.05))
        self.assertEqual(selected, "full")
        self.assertIn("cv_balanced_accuracy_not_worse", failed)


if __name__ == "__main__":
    unittest.main()

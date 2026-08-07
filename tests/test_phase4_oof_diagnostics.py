import unittest

import pandas as pd


class Phase4OOFDiagnosticTests(unittest.TestCase):
    def test_train_cutoff_must_precede_oof_date(self):
        frame = pd.DataFrame({
            "train_max_date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
        })
        self.assertTrue((frame["train_max_date"] < frame["date"]).all())


if __name__ == "__main__":
    unittest.main()

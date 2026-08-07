import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modules.ML.phase3_config import Phase3Config
from modules.ML.phase3_evaluation import expanding_date_folds, load_phase3_dataset


class Phase3EvaluationTests(unittest.TestCase):
    def _frame(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        rows = []
        labels = ("DOWN", "NO_TRADE", "UP")
        for index, date in enumerate(dates):
            split = "train" if index < 40 else "validation" if index < 50 else "final_test"
            rows.append({
                "symbol": "FPT", "date": date, "ret_1": index / 100,
                "market_regime": "bull", "future_return": np.nan if split == "final_test" else 0.01,
                "target": np.nan if split == "final_test" else labels[index % 3], "split": split,
            })
        return pd.DataFrame(rows)

    def test_loader_keeps_only_train_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dataset.csv"
            self._frame().to_csv(path, index=False)
            loaded = load_phase3_dataset(path)
        self.assertEqual(set(loaded["split"]), {"train", "validation"})
        self.assertEqual(len(loaded), 50)

    def test_loader_rejects_exposed_final_target(self):
        frame = self._frame()
        frame.loc[frame["split"].eq("final_test"), "target"] = "UP"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dataset.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "final-test target"):
                load_phase3_dataset(path)

    def test_expanding_folds_never_train_on_future_dates(self):
        frame = self._frame().query("split == 'train'").reset_index(drop=True)
        config = Phase3Config(cv_folds=2, min_train_dates=30)
        folds = list(expanding_date_folds(frame, config))
        self.assertEqual(len(folds), 2)
        for _, train_index, validation_index in folds:
            self.assertLess(frame.loc[train_index, "date"].max(), frame.loc[validation_index, "date"].min())


if __name__ == "__main__":
    unittest.main()

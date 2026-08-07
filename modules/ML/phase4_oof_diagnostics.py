"""Leakage-safe raw compact-XGBoost OOF diagnostics by symbol and signal side."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modules.ML.phase3_config import Phase3Config
from modules.ML.phase3_evaluation import _fit_predict, expanding_date_folds, load_phase3_dataset
from modules.ML.phase3_models import LABELS
from modules.ML.phase4_diagnostics import _diagnostic_metrics


def build_compact_oof(dataset_path: Path, phase3_lock_path: Path) -> pd.DataFrame:
    lock = json.loads(Path(phase3_lock_path).read_text(encoding="utf-8"))
    if lock.get("selected_feature_set") != "compact":
        raise ValueError("phase3 lock không chọn compact")
    data = load_phase3_dataset(dataset_path)
    train = data[data["split"].eq("train")].copy().reset_index(drop=True)
    config = Phase3Config(**lock["config"])
    columns = lock["feature_columns"]
    rows = []
    for fold, train_index, validation_index in expanding_date_folds(train, config):
        fold_train = train.loc[train_index]
        fold_validation = train.loc[validation_index]
        _, predicted, probabilities = _fit_predict(
            "xgboost", fold_train, fold_validation, columns, config
        )
        signals = np.where(
            predicted == "UP", "BUY",
            np.where(predicted == "DOWN", "SELL", "NO_TRADE"),
        )
        fold_rows = fold_validation[["symbol", "date", "future_return", "target", "market_regime"]].copy()
        fold_rows["fold"] = fold
        fold_rows["train_max_date"] = fold_train["date"].max()
        fold_rows["prediction"] = predicted
        fold_rows["signal"] = signals
        for index, label in enumerate(LABELS):
            fold_rows[f"prob_{label.lower()}"] = probabilities[:, index]
        rows.append(fold_rows)
    result = pd.concat(rows, ignore_index=True).sort_values(
        ["date", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    if not (result["train_max_date"] < result["date"]).all():
        raise RuntimeError("OOF leakage: train_max_date không nhỏ hơn target date")
    return result


def _group_report(frame, column, overall_accuracy, min_actions=20):
    reports = []
    for group, subset in frame.groupby(column, observed=True, sort=True):
        metrics = _diagnostic_metrics(subset, overall_accuracy)
        metrics["diagnostic_flag"] = (
            metrics["actions"] >= min_actions
            and metrics["accuracy_gap_vs_overall"] <= -0.05
        )
        reports.append({column: group, **metrics})
    return pd.DataFrame(reports)


def run_oof_diagnostics(dataset_path: Path, phase3_lock_path: Path, output_dir: Path):
    oof = build_compact_oof(dataset_path, phase3_lock_path)
    overall = _diagnostic_metrics(oof, 0.0)
    overall_accuracy = overall["action_direction_accuracy"]
    overall["accuracy_gap_vs_overall"] = 0.0
    overall["diagnostic_flag"] = False
    by_symbol = _group_report(oof, "symbol", overall_accuracy)
    by_side = _group_report(oof[oof["signal"].isin(["BUY", "SELL"])], "signal", overall_accuracy)
    by_fold = _group_report(oof, "fold", overall_accuracy, min_actions=30)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output / "compact_xgboost_train_oof_predictions.csv", index=False)
    by_symbol.to_csv(output / "train_oof_by_symbol.csv", index=False)
    by_side.to_csv(output / "train_oof_by_signal_side.csv", index=False)
    by_fold.to_csv(output / "train_oof_by_fold.csv", index=False)
    flagged_symbols = by_symbol[by_symbol["diagnostic_flag"]]
    payload = {
        "analysis_type": "raw_compact_xgboost_argmax_oof_no_tuning",
        "oof_rows": int(len(oof)), "folds": int(oof["fold"].nunique()),
        "overall": overall,
        "flag_rule": {"min_actions": 20, "accuracy_gap_lte": -0.05},
        "flagged_symbols": flagged_symbols[[
            "symbol", "actions", "action_direction_accuracy",
            "accuracy_gap_vs_overall", "incorrect_actions",
        ]].to_dict("records"),
        "calibrator_used": False,
        "reason_calibrator_not_used": "would refit on and diagnose the same OOF probabilities",
        "policy_changed": False, "final_test_read": False,
    }
    (output / "train_oof_diagnostic_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload, by_symbol, by_side, by_fold


__all__ = ["build_compact_oof", "run_oof_diagnostics"]

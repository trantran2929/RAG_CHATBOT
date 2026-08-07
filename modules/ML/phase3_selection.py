"""Paired fold comparison and locking of the Phase 3 XGBoost feature set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from modules.ML.phase3_analysis import ABLATION_GROUPS
from modules.ML.phase3_config import Phase3Config
from modules.ML.phase3_evaluation import (
    _fit_predict, classification_metrics, expanding_date_folds, load_phase3_dataset,
)
from modules.ML.phase3_models import feature_columns


COMPACT_GROUP = "stock_returns"
MAX_VALIDATION_BALANCED_ACCURACY_DROP = 0.005
MAX_VALIDATION_LOG_LOSS_INCREASE = 0.01


def choose_feature_set(full_summary, compact_summary) -> tuple[str, list[str]]:
    checks = {
        "cv_balanced_accuracy_not_worse":
            compact_summary["cv_mean_balanced_accuracy"] >= full_summary["cv_mean_balanced_accuracy"],
        "cv_macro_f1_not_worse":
            compact_summary["cv_mean_macro_f1"] >= full_summary["cv_mean_macro_f1"],
        "validation_balanced_accuracy_within_tolerance":
            compact_summary["validation_balanced_accuracy"] >= (
                full_summary["validation_balanced_accuracy"] - MAX_VALIDATION_BALANCED_ACCURACY_DROP
            ),
        "validation_log_loss_within_tolerance":
            compact_summary["validation_log_loss"] <= (
                full_summary["validation_log_loss"] + MAX_VALIDATION_LOG_LOSS_INCREASE
            ),
    }
    return ("compact" if all(checks.values()) else "full"), [key for key, passed in checks.items() if not passed]


def _summary(folds, validation_metrics):
    return {
        "cv_mean_balanced_accuracy": float(pd.DataFrame(folds)["balanced_accuracy"].mean()),
        "cv_mean_macro_f1": float(pd.DataFrame(folds)["macro_f1"].mean()),
        "cv_mean_log_loss": float(pd.DataFrame(folds)["log_loss"].mean()),
        "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_log_loss": validation_metrics["log_loss"],
    }


def compare_and_lock(dataset_path: Path, output_dir: Path, lock_path: Path, config=None, *, force=False):
    config = config or Phase3Config()
    lock_path = Path(lock_path)
    if lock_path.exists() and not force:
        raise FileExistsError(f"lock đã tồn tại: {lock_path}; dùng --force chỉ khi có quyết định audit mới")
    data = load_phase3_dataset(dataset_path)
    train = data[data["split"].eq("train")].copy()
    validation = data[data["split"].eq("validation")].copy()
    full_columns = feature_columns(data)
    removed = set(ABLATION_GROUPS[COMPACT_GROUP])
    compact_columns = [column for column in full_columns if column not in removed]
    fold_rows = []
    for fold, train_index, validation_index in expanding_date_folds(train, config):
        fold_train, fold_validation = train.loc[train_index], train.loc[validation_index]
        for candidate, columns in (("full", full_columns), ("compact", compact_columns)):
            _, predicted, probabilities = _fit_predict(
                "xgboost", fold_train, fold_validation, columns, config
            )
            fold_rows.append({"fold": fold, "candidate": candidate, **classification_metrics(
                fold_validation["target"], predicted, probabilities
            )})
    validation_rows = []
    for candidate, columns in (("full", full_columns), ("compact", compact_columns)):
        _, predicted, probabilities = _fit_predict("xgboost", train, validation, columns, config)
        validation_rows.append({"candidate": candidate, **classification_metrics(
            validation["target"], predicted, probabilities
        )})
    folds = pd.DataFrame(fold_rows)
    validation_table = pd.DataFrame(validation_rows)
    summaries = {}
    for candidate in ("full", "compact"):
        summaries[candidate] = _summary(
            folds[folds["candidate"].eq(candidate)].to_dict("records"),
            validation_table[validation_table["candidate"].eq(candidate)].iloc[0].to_dict(),
        )
    selected, failed_checks = choose_feature_set(summaries["full"], summaries["compact"])
    selected_columns = full_columns if selected == "full" else compact_columns
    payload = {
        "model": "xgboost", "selected_feature_set": selected,
        "excluded_feature_group": None if selected == "full" else COMPACT_GROUP,
        "feature_columns": selected_columns, "feature_count": len(selected_columns),
        "comparison": summaries, "failed_compact_checks": failed_checks,
        "selection_rule": {
            "compact_cv_balanced_accuracy_not_worse": True,
            "compact_cv_macro_f1_not_worse": True,
            "max_validation_balanced_accuracy_drop": MAX_VALIDATION_BALANCED_ACCURACY_DROP,
            "max_validation_log_loss_increase": MAX_VALIDATION_LOG_LOSS_INCREASE,
        },
        "config": config.to_dict(), "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
        "final_test_read": False,
    }
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "xgboost_full_vs_compact_folds.csv", index=False)
    validation_table.to_csv(output / "xgboost_full_vs_compact_validation.csv", index=False)
    (output / "xgboost_full_vs_compact_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload, folds, validation_table


__all__ = ["choose_feature_set", "compare_and_lock"]

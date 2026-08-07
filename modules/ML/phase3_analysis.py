"""Calibration, model importance and bounded ablation for Phase 3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from modules.ML.phase3_config import Phase3Config
from modules.ML.phase3_evaluation import (
    _fit_predict, classification_metrics, expanding_date_folds, load_phase3_dataset,
)
from modules.ML.phase3_models import LABELS, feature_columns


ABLATION_GROUPS = {
    "stock_returns": ("ret_1", "ret_2", "ret_5", "ret_10", "ret_20", "intraday_return", "overnight_gap"),
    "stock_risk": ("range_1", "volatility_5", "volatility_10", "volatility_20", "atr_14", "drawdown_20", "drawdown_60"),
    "stock_trend": ("distance_ma20", "distance_ma50", "distance_ma200", "rsi_14"),
    "volume": ("relative_volume_5", "relative_volume_20"),
    "market": (
        "vnindex_ret_1", "vnindex_ret_5", "vnindex_ret_20", "vnindex_ret_60",
        "vnindex_volatility_5", "vnindex_volatility_20", "vnindex_distance_ma20",
        "vnindex_distance_ma200", "relative_strength_5", "relative_strength_20",
        "rolling_beta_60", "rolling_correlation_60", "market_regime",
    ),
}


def multiclass_brier(actual, probabilities) -> float:
    encoded = np.zeros_like(probabilities, dtype=float)
    for row, label in enumerate(actual):
        encoded[row, LABELS.index(label)] = 1.0
    return float(np.mean(np.sum((probabilities - encoded) ** 2, axis=1)))


def expected_calibration_error(actual, probabilities, bins: int = 10) -> float:
    actual = np.asarray(actual)
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            accuracy = np.mean(predicted[mask] == actual[mask])
            error += mask.mean() * abs(accuracy - confidence[mask].mean())
    return float(error)


def _probability_metrics(actual, probabilities):
    predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    metrics = classification_metrics(actual, predicted, probabilities)
    metrics["brier_multiclass"] = multiclass_brier(actual, probabilities)
    metrics["ece_10_bins"] = expected_calibration_error(actual, probabilities)
    return metrics


def run_calibration(dataset_path: Path, output_dir: Path, model_name: str, config=None):
    config = config or Phase3Config()
    data = load_phase3_dataset(dataset_path)
    train = data[data["split"].eq("train")].copy()
    validation = data[data["split"].eq("validation")].copy()
    columns = feature_columns(data)
    oof_probabilities, oof_targets = [], []
    for _, train_index, validation_index in expanding_date_folds(train, config):
        fold_train, fold_validation = train.loc[train_index], train.loc[validation_index]
        _, _, probabilities = _fit_predict(model_name, fold_train, fold_validation, columns, config)
        oof_probabilities.append(probabilities)
        oof_targets.extend(fold_validation["target"].tolist())
    calibrator = LogisticRegression(max_iter=1000, random_state=config.random_state)
    calibrator.fit(np.vstack(oof_probabilities), oof_targets)
    _, _, raw_probabilities = _fit_predict(model_name, train, validation, columns, config)
    calibrated = calibrator.predict_proba(raw_probabilities)
    aligned = np.zeros_like(raw_probabilities)
    for index, label in enumerate(calibrator.classes_):
        aligned[:, LABELS.index(label)] = calibrated[:, index]
    payload = {
        "model": model_name, "method": "OOF multinomial sigmoid",
        "calibration_fit_split": "train_oof", "evaluation_split": "validation",
        "oof_rows": len(oof_targets), "raw": _probability_metrics(validation["target"], raw_probabilities),
        "calibrated": _probability_metrics(validation["target"], aligned),
        "final_test_read": False,
    }
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / f"{model_name}_calibration_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run_feature_importance(dataset_path: Path, output_dir: Path, model_name: str, config=None):
    config = config or Phase3Config()
    data = load_phase3_dataset(dataset_path)
    train = data[data["split"].eq("train")].copy()
    columns = feature_columns(data)
    model, _, _ = _fit_predict(model_name, train, train.iloc[:1], columns, config)
    transformed = model.named_steps["preprocess"].get_feature_names_out()
    estimator = model.named_steps["model"]
    if model_name == "xgboost":
        values = estimator.feature_importances_
        kind = "xgboost_gain_weight"
    else:
        values = np.abs(estimator.coef_).mean(axis=0)
        kind = "mean_absolute_standardized_coefficient"
    report = pd.DataFrame({"feature": transformed, "importance": values})
    report = report.sort_values("importance", ascending=False).reset_index(drop=True)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    report.to_csv(output / f"{model_name}_feature_importance.csv", index=False)
    metadata = {"model": model_name, "importance_type": kind, "fit_split": "train", "final_test_read": False}
    (output / f"{model_name}_feature_importance_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_ablation(dataset_path: Path, output_dir: Path, model_name: str, config=None):
    config = config or Phase3Config()
    data = load_phase3_dataset(dataset_path)
    train = data[data["split"].eq("train")].copy()
    validation = data[data["split"].eq("validation")].copy()
    all_columns = feature_columns(data)
    rows = []
    experiments = {"full": all_columns}
    for group, removed in ABLATION_GROUPS.items():
        experiments[f"without_{group}"] = [column for column in all_columns if column not in removed]
    for experiment, columns in experiments.items():
        _, predicted, probabilities = _fit_predict(model_name, train, validation, columns, config)
        metrics = classification_metrics(validation["target"], predicted, probabilities)
        rows.append({"experiment": experiment, "features": len(columns), **metrics})
    report = pd.DataFrame(rows)
    full = report.loc[report["experiment"].eq("full")].iloc[0]
    report["delta_balanced_accuracy"] = report["balanced_accuracy"] - full["balanced_accuracy"]
    report["delta_macro_f1"] = report["macro_f1"] - full["macro_f1"]
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    report.to_csv(output / f"{model_name}_ablation_report.csv", index=False)
    return report


__all__ = [
    "ABLATION_GROUPS", "multiclass_brier", "expected_calibration_error",
    "run_calibration", "run_feature_importance", "run_ablation",
]

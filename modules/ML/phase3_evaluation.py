"""Validation-only evaluation for Phase 3; final-test rows are never consumed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

from modules.ML.phase3_config import Phase3Config
from modules.ML.phase3_models import LABELS, build_logistic, build_xgboost, feature_columns


def load_phase3_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"symbol", "date", "target", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset thiếu cột: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("dataset trùng khóa symbol/date")
    allowed = {"train", "validation", "final_test"}
    if not set(frame["split"].unique()).issubset(allowed):
        raise ValueError("dataset có split không hợp lệ")
    final = frame[frame["split"].eq("final_test")]
    if final["target"].notna().any():
        raise ValueError("final-test target đã bị lộ; từ chối chạy Pha 3")
    visible = frame[frame["split"].isin(["train", "validation"])].copy()
    if visible["target"].isna().any():
        raise ValueError("train/validation có target trống")
    if not set(visible["target"]).issubset(set(LABELS)):
        raise ValueError("target có nhãn không hợp lệ")
    return visible.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)


def expanding_date_folds(train: pd.DataFrame, config: Phase3Config):
    dates = np.array(sorted(train["date"].unique()))
    remaining = len(dates) - config.min_train_dates
    if remaining < config.cv_folds:
        raise ValueError("không đủ ngày train cho time-series cross-validation")
    blocks = np.array_split(dates[config.min_train_dates:], config.cv_folds)
    for fold, validation_dates in enumerate(blocks, start=1):
        cutoff = validation_dates[0]
        train_index = train.index[train["date"] < cutoff]
        validation_index = train.index[train["date"].isin(validation_dates)]
        yield fold, train_index, validation_index


def _fit_predict(model_name, train, evaluate, columns, config):
    if model_name == "logistic":
        model = build_logistic(train, columns, c=config.logistic_c, random_state=config.random_state)
        y_train = train["target"]
    elif model_name == "xgboost":
        model = build_xgboost(train, columns, config)
        mapping = model.phase3_label_to_int
        y_train = train["target"].map(mapping)
    else:
        raise ValueError(f"model không hỗ trợ: {model_name}")
    model.fit(train[columns], y_train)
    probabilities = model.predict_proba(evaluate[columns])
    if model_name == "xgboost":
        predictions = np.array(LABELS)[probabilities.argmax(axis=1)]
    else:
        classes = model.named_steps["model"].classes_
        predictions = classes[probabilities.argmax(axis=1)]
        aligned = np.zeros((len(evaluate), len(LABELS)))
        for index, label in enumerate(classes):
            aligned[:, LABELS.index(label)] = probabilities[:, index]
        probabilities = aligned
    return model, predictions, probabilities


def classification_metrics(actual, predicted, probabilities):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)),
        "log_loss": float(log_loss(actual, probabilities, labels=LABELS)),
        "rows": int(len(actual)),
    }


def evaluate_phase3(dataset_path: Path, output_dir: Path, model_name: str, config=None):
    config = config or Phase3Config()
    data = load_phase3_dataset(dataset_path)
    train = data[data["split"].eq("train")].copy()
    validation = data[data["split"].eq("validation")].copy()
    columns = feature_columns(data)
    folds = []
    for fold, train_index, validation_index in expanding_date_folds(train, config):
        fold_train, fold_validation = train.loc[train_index], train.loc[validation_index]
        _, predicted, probabilities = _fit_predict(
            model_name, fold_train, fold_validation, columns, config
        )
        folds.append({"fold": fold, **classification_metrics(
            fold_validation["target"], predicted, probabilities
        )})
    model, predicted, probabilities = _fit_predict(model_name, train, validation, columns, config)
    validation_metrics = classification_metrics(validation["target"], predicted, probabilities)
    majority = train["target"].mode().iloc[0]
    validation_metrics["majority_accuracy"] = float((validation["target"] == majority).mean())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = validation[["symbol", "date", "target"]].copy()
    predictions["prediction"] = predicted
    for index, label in enumerate(LABELS):
        predictions[f"prob_{label.lower()}"] = probabilities[:, index]
    predictions.to_csv(output / f"{model_name}_validation_predictions.csv", index=False)
    payload = {
        "model": model_name, "config": config.to_dict(),
        "feature_columns": columns, "cv_folds": folds,
        "validation": validation_metrics,
        "final_test_read": False,
        "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
    }
    (output / f"{model_name}_validation_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload, model


__all__ = ["load_phase3_dataset", "expanding_date_folds", "evaluate_phase3"]

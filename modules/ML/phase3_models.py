"""Leakage-safe Phase 3 classifiers and feature preprocessing."""

from __future__ import annotations

from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LABELS = ("DOWN", "NO_TRADE", "UP")
NON_FEATURE_COLUMNS = {"symbol", "date", "future_return", "target", "split"}


def feature_columns(frame) -> list[str]:
    return [column for column in frame.columns if column not in NON_FEATURE_COLUMNS]


def _preprocessor(frame, columns: Sequence[str]) -> ColumnTransformer:
    categorical = [column for column in columns if str(frame[column].dtype) in {"object", "string"}]
    numeric = [column for column in columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )


def build_logistic(frame, columns: Sequence[str], *, c: float, random_state: int) -> Pipeline:
    return Pipeline([
        ("preprocess", _preprocessor(frame, columns)),
        ("model", LogisticRegression(
            C=c, class_weight="balanced", max_iter=2000,
            random_state=random_state,
        )),
    ])


def build_xgboost(frame, columns: Sequence[str], config) -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost chưa được cài; hãy rebuild service ingestion sau khi cập nhật requirements.txt"
        ) from exc
    label_to_int = {label: index for index, label in enumerate(LABELS)}
    model = XGBClassifier(
        objective="multi:softprob", num_class=len(LABELS),
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        subsample=config.xgb_subsample,
        colsample_bytree=config.xgb_colsample_bytree,
        eval_metric="mlogloss", random_state=config.random_state,
        n_jobs=1,
    )
    pipeline = Pipeline([("preprocess", _preprocessor(frame, columns)), ("model", model)])
    pipeline.phase3_label_to_int = label_to_int
    return pipeline


__all__ = ["LABELS", "feature_columns", "build_logistic", "build_xgboost"]

"""Auditable and reproducible Phase 2 dataset builder."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from modules.ML.phase2_config import Phase2DatasetConfig
from modules.ML.phase2_features import (
    MARKET_FEATURE_COLUMNS,
    STOCK_FEATURE_COLUMNS,
    build_market_features,
    build_stock_features,
    clean_ohlcv,
    make_target,
)
from modules.api.stock_api import get_prices_df


DataProvider = Callable[[str, int], pd.DataFrame]


def audit_ohlcv(symbol: str, frame: pd.DataFrame) -> Dict[str, object]:
    """Return diagnostics without silently deleting suspicious market moves."""
    raw = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(raw.index))
    duplicates = int(index.duplicated(keep=False).sum())
    missing_columns = sorted(set(("open", "high", "low", "close", "volume")) - set(raw.columns))
    if missing_columns:
        return {
            "symbol": symbol, "first_date": None, "last_date": None,
            "row_count": int(len(raw)), "duplicate_dates": duplicates,
            "missing_ohlcv": int(len(raw)), "non_positive_price": int(len(raw)),
            "negative_volume": int(len(raw)), "absolute_return_over_10pct": 0,
            "near_limit_sessions": 0, "missing_columns": "|".join(missing_columns),
        }
    numeric = raw[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    returns = numeric["close"].pct_change(fill_method=None)
    return {
        "symbol": symbol,
        "first_date": index.min().date().isoformat() if len(index) else None,
        "last_date": index.max().date().isoformat() if len(index) else None,
        "row_count": int(len(raw)),
        "duplicate_dates": duplicates,
        "missing_ohlcv": int(numeric.isna().any(axis=1).sum()),
        "non_positive_price": int((numeric[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        "negative_volume": int((numeric["volume"] < 0).sum()),
        "absolute_return_over_10pct": int((returns.abs() > 0.10).sum()),
        "near_limit_sessions": int((returns.abs() >= 0.065).sum()),
        "missing_columns": "",
    }


def assign_time_split(index: pd.Index, config: Phase2DatasetConfig) -> pd.Series:
    """Chronological, disjoint split; final test always remains the last block."""
    count = len(index)
    reserved = config.validation_days + config.final_test_days
    required = reserved + config.min_train_rows
    if count < required:
        raise ValueError(
            "không đủ hàng sau feature warm-up: "
            f"cần ít nhất {required}, hiện có {count}"
        )
    train_end = count - reserved
    validation_end = count - config.final_test_days
    split = pd.Series("train", index=index, dtype="object")
    split.iloc[train_end:validation_end] = "validation"
    split.iloc[validation_end:] = "final_test"
    return split


def build_symbol_dataset(
    symbol: str,
    stock_ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    config: Phase2DatasetConfig,
    *,
    include_final_targets: bool = False,
) -> pd.DataFrame:
    stock = clean_ohlcv(stock_ohlcv, f"{symbol}_ohlcv")
    stock_features = build_stock_features(stock)
    market_features = build_market_features(stock, market_ohlcv)
    target = make_target(stock["close"], config.target_threshold)
    dataset = stock_features.join(market_features).join(target)
    dataset.insert(0, "symbol", str(symbol).strip().upper())
    dataset.insert(1, "date", dataset.index)
    dataset["split"] = assign_time_split(dataset.index, config)
    feature_columns = list(STOCK_FEATURE_COLUMNS) + list(MARKET_FEATURE_COLUMNS)
    numeric_features = [c for c in feature_columns if c != "market_regime"]
    dataset[numeric_features] = dataset[numeric_features].replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=numeric_features + ["market_regime", "target"])
    if dataset.empty:
        raise ValueError(f"{symbol}: không còn hàng sau feature warm-up")
    # Re-assign after warm-up so every symbol retains exactly 125/125 reserved rows.
    dataset["split"] = assign_time_split(dataset.index, config)
    if not include_final_targets:
        final_mask = dataset["split"].eq("final_test")
        dataset.loc[final_mask, ["future_return", "target"]] = [np.nan, pd.NA]
    return dataset.reset_index(drop=True)


def _stable_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")


def build_phase2_dataset(
    config: Phase2DatasetConfig,
    output_dir: Path,
    *,
    provider: DataProvider = get_prices_df,
    include_final_targets: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    market = provider("VNINDEX", max(900, config.lookback_days))
    datasets, audits = [], []
    for symbol in config.symbols:
        stock = provider(symbol, config.lookback_days)
        audits.append(audit_ohlcv(symbol, stock))
        datasets.append(
            build_symbol_dataset(
                symbol, stock, market, config,
                include_final_targets=include_final_targets,
            )
        )
    dataset = pd.concat(datasets, ignore_index=True)
    dataset = dataset.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)
    audit = pd.DataFrame(audits).sort_values("symbol").reset_index(drop=True)
    dataset_bytes = _stable_csv_bytes(dataset)
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    metadata = {
        "feature_version": config.feature_version,
        "config": config.to_dict(),
        "news": "disabled" if not config.news_enabled else "not_implemented",
        "include_final_targets": bool(include_final_targets),
        "final_targets_masked": not include_final_targets,
        "feature_columns": list(STOCK_FEATURE_COLUMNS) + list(MARKET_FEATURE_COLUMNS),
        "row_count": int(len(dataset)),
        "rows_by_split": {str(k): int(v) for k, v in dataset.groupby("split").size().items()},
        "dataset_sha256": dataset_hash,
    }
    visible = dataset[dataset["split"].isin(["train", "validation"])].copy()
    target_counts = (
        visible.groupby(["split", "target"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    regime_counts = (
        visible.groupby(["split", "market_regime"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    date_ranges = (
        dataset.groupby("split", observed=True)["date"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    report = {
        "target_counts_train_validation_only": target_counts.to_dict(orient="records"),
        "regime_counts_train_validation_only": regime_counts.to_dict(orient="records"),
        "date_ranges": [
            {
                "split": str(row["split"]),
                "min": pd.Timestamp(row["min"]).date().isoformat(),
                "max": pd.Timestamp(row["max"]).date().isoformat(),
                "count": int(row["count"]),
            }
            for _, row in date_ranges.iterrows()
        ],
        "final_target_statistics_read": False,
    }
    (output / "phase2_dataset.csv").write_bytes(dataset_bytes)
    audit.to_csv(output / "phase2_data_audit.csv", index=False, lineterminator="\n")
    (output / "phase2_dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "phase2_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset, audit, metadata


__all__ = [
    "audit_ohlcv", "assign_time_split", "build_symbol_dataset",
    "build_phase2_dataset",
]

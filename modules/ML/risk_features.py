"""Leakage-safe risk features used by Phase 1 validation filters."""

from typing import Optional

import numpy as np
import pandas as pd


RISK_FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "distance_ma20",
    "volatility_5",
    "volatility_20",
    "previous_shock",
    "vnindex_ret_5",
    "vnindex_ret_60",
    "vnindex_distance_ma200",
    "vnindex_volatility_20",
]


def _clean_close(series: pd.Series, name: str) -> pd.Series:
    if series is None:
        raise ValueError(f"{name} không được để trống")
    close = pd.to_numeric(series, errors="coerce").dropna().astype("float64")
    close = close[close > 0]
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index)).tz_localize(None)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if len(close) < 2:
        raise ValueError(f"{name} không đủ dữ liệu giá")
    return close


def build_risk_features(
    stock_close: pd.Series,
    market_close: pd.Series,
) -> pd.DataFrame:
    """
    Build features for target t using information available through t-1.

    Market data are forward-filled onto stock sessions. Backfill is forbidden
    because it would move a future observation into an earlier target row.
    """
    stock = _clean_close(stock_close, "stock_close")
    market = _clean_close(market_close, "market_close")
    stock_ret = np.log(stock / stock.shift(1))
    market_ret = np.log(market / market.shift(1))

    features = pd.DataFrame(index=stock.index)
    features["ret_1"] = stock_ret.shift(1)
    features["ret_5"] = np.log(stock / stock.shift(5)).shift(1)
    features["distance_ma20"] = (
        stock / stock.rolling(20, min_periods=20).mean() - 1.0
    ).shift(1)
    features["volatility_5"] = stock_ret.rolling(5, min_periods=5).std().shift(1)
    features["volatility_20"] = (
        stock_ret.rolling(20, min_periods=20).std().shift(1)
    )
    features["previous_shock"] = stock_ret.abs().shift(1)

    market_features = pd.DataFrame(index=market.index)
    market_features["vnindex_ret_5"] = np.log(market / market.shift(5)).shift(1)
    market_features["vnindex_ret_60"] = np.log(market / market.shift(60)).shift(1)
    market_features["vnindex_distance_ma200"] = (
        market / market.rolling(200, min_periods=200).mean() - 1.0
    ).shift(1)
    market_features["vnindex_volatility_20"] = (
        market_ret.rolling(20, min_periods=20).std().shift(1)
    )

    # Only carry an already-observed market row forward to a stock session.
    market_aligned = market_features.reindex(stock.index).ffill()
    features = features.join(market_aligned, how="left")
    return features[RISK_FEATURE_COLUMNS].astype("float64")


def fit_volatility_threshold(
    features: pd.DataFrame,
    percentile: float,
    *,
    end_date: Optional[pd.Timestamp] = None,
) -> float:
    """Fit a VNINDEX volatility threshold without seeing end_date or later."""
    if not 0 < float(percentile) < 1:
        raise ValueError("percentile phải nằm trong (0, 1)")
    sample = features
    if end_date is not None:
        cutoff = pd.Timestamp(end_date).tz_localize(None)
        sample = sample.loc[pd.DatetimeIndex(sample.index) < cutoff]
    values = pd.to_numeric(
        sample.get("vnindex_volatility_20"),
        errors="coerce",
    ).dropna()
    if values.empty:
        raise ValueError("Không đủ dữ liệu quá khứ để fit volatility threshold")
    return float(values.quantile(float(percentile)))


def label_market_regime(
    features: pd.DataFrame,
    volatility_threshold: float,
) -> pd.DataFrame:
    """Attach BULL/BEAR/SIDEWAYS plus an independent high-volatility flag."""
    if not np.isfinite(volatility_threshold) or volatility_threshold < 0:
        raise ValueError("volatility_threshold không hợp lệ")
    out = features.copy()
    bull = (
        (out["vnindex_distance_ma200"] > 0)
        & (out["vnindex_ret_60"] > 0)
    )
    bear = (
        (out["vnindex_distance_ma200"] < 0)
        & (out["vnindex_ret_60"] < 0)
    )
    out["market_regime"] = "SIDEWAYS"
    out.loc[bull, "market_regime"] = "BULL"
    out.loc[bear, "market_regime"] = "BEAR"
    out["high_volatility"] = (
        out["vnindex_volatility_20"] > float(volatility_threshold)
    ).fillna(False)
    return out


__all__ = [
    "RISK_FEATURE_COLUMNS",
    "build_risk_features",
    "fit_volatility_threshold",
    "label_market_regime",
]

"""Point-in-time OHLCV and market features for Phase 2."""

from typing import Iterable

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close", "volume")
STOCK_FEATURE_COLUMNS = (
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20",
    "range_1", "intraday_return", "overnight_gap",
    "volatility_5", "volatility_10", "volatility_20",
    "distance_ma20", "distance_ma50", "distance_ma200",
    "relative_volume_5", "relative_volume_20",
    "drawdown_20", "drawdown_60", "rsi_14", "atr_14",
)
MARKET_FEATURE_COLUMNS = (
    "vnindex_ret_1", "vnindex_ret_5", "vnindex_ret_20", "vnindex_ret_60",
    "vnindex_volatility_5", "vnindex_volatility_20",
    "vnindex_distance_ma20", "vnindex_distance_ma200",
    "relative_strength_5", "relative_strength_20",
    "rolling_beta_60", "rolling_correlation_60", "market_regime",
)


def clean_ohlcv(frame: pd.DataFrame, name: str = "ohlcv") -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"{name} không có dữ liệu")
    missing = set(PRICE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} thiếu cột: {sorted(missing)}")
    out = frame.loc[:, PRICE_COLUMNS].copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for column in PRICE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    invalid_price = (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
    out.loc[invalid_price, ["open", "high", "low", "close"]] = np.nan
    out.loc[out["volume"] < 0, "volume"] = np.nan
    return out


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-change.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.where(loss.ne(0.0), 100.0)


def build_stock_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build target-t features using observations available through t-1."""
    data = clean_ohlcv(ohlcv, "stock_ohlcv")
    close, volume = data["close"], data["volume"]
    log_close = np.log(close)
    daily_return = log_close.diff()
    raw = pd.DataFrame(index=data.index)
    for window in (1, 2, 5, 10, 20):
        raw[f"ret_{window}"] = log_close.diff(window)
    raw["range_1"] = np.log(data["high"] / data["low"])
    raw["intraday_return"] = np.log(close / data["open"])
    raw["overnight_gap"] = np.log(data["open"] / close.shift(1))
    for window in (5, 10, 20):
        raw[f"volatility_{window}"] = daily_return.rolling(window, min_periods=window).std()
    for window in (20, 50, 200):
        raw[f"distance_ma{window}"] = close / close.rolling(window, min_periods=window).mean() - 1.0
    for window in (5, 20):
        raw[f"relative_volume_{window}"] = volume / volume.rolling(window, min_periods=window).mean() - 1.0
    for window in (20, 60):
        raw[f"drawdown_{window}"] = close / close.rolling(window, min_periods=window).max() - 1.0
    raw["rsi_14"] = _rsi(close, 14)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    raw["atr_14"] = true_range.rolling(14, min_periods=14).mean() / close
    return raw.loc[:, STOCK_FEATURE_COLUMNS].shift(1).replace([np.inf, -np.inf], np.nan)


def build_market_features(stock_ohlcv: pd.DataFrame, market_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Align already-observed VNINDEX features to stock sessions without backfill."""
    stock = clean_ohlcv(stock_ohlcv, "stock_ohlcv")
    market = clean_ohlcv(market_ohlcv, "market_ohlcv")
    stock_ret = np.log(stock["close"]).diff()
    market_log = np.log(market["close"])
    market_ret = market_log.diff()
    raw = pd.DataFrame(index=market.index)
    for window in (1, 5, 20, 60):
        raw[f"vnindex_ret_{window}"] = market_log.diff(window)
    for window in (5, 20):
        raw[f"vnindex_volatility_{window}"] = market_ret.rolling(window, min_periods=window).std()
    for window in (20, 200):
        raw[f"vnindex_distance_ma{window}"] = (
            market["close"] / market["close"].rolling(window, min_periods=window).mean() - 1.0
        )
    market_known = raw.shift(1).reindex(stock.index).ffill()
    aligned_market_return = market_ret.shift(1).reindex(stock.index).ffill()
    stock_known_return = stock_ret.shift(1)
    for window in (5, 20):
        market_known[f"relative_strength_{window}"] = (
            np.log(stock["close"]).diff(window).shift(1)
            - market_log.diff(window).shift(1).reindex(stock.index).ffill()
        )
    market_known["rolling_beta_60"] = (
        stock_known_return.rolling(60, min_periods=60).cov(aligned_market_return)
        / aligned_market_return.rolling(60, min_periods=60).var()
    )
    market_known["rolling_correlation_60"] = stock_known_return.rolling(
        60, min_periods=60
    ).corr(aligned_market_return)
    bull = (market_known["vnindex_distance_ma200"] > 0) & (market_known["vnindex_ret_60"] > 0)
    bear = (market_known["vnindex_distance_ma200"] < 0) & (market_known["vnindex_ret_60"] < 0)
    market_known["market_regime"] = "SIDEWAYS"
    market_known.loc[bull, "market_regime"] = "BULL"
    market_known.loc[bear, "market_regime"] = "BEAR"
    return market_known.loc[:, MARKET_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def make_target(close: pd.Series, threshold: float) -> pd.DataFrame:
    """Label return realized at t; callers must never include it as a feature."""
    if threshold < 0:
        raise ValueError("target threshold không được âm")
    clean = pd.to_numeric(close, errors="coerce").astype("float64")
    future_return = np.log(clean / clean.shift(1))
    label = pd.Series("NO_TRADE", index=clean.index, dtype="object")
    label.loc[future_return > threshold] = "UP"
    label.loc[future_return < -threshold] = "DOWN"
    label.loc[future_return.isna()] = pd.NA
    return pd.DataFrame({"future_return": future_return, "target": label}, index=clean.index)


__all__ = [
    "PRICE_COLUMNS", "STOCK_FEATURE_COLUMNS", "MARKET_FEATURE_COLUMNS",
    "clean_ohlcv", "build_stock_features", "build_market_features", "make_target",
]

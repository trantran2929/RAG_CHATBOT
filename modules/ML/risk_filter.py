"""Deterministic Phase 1 risk filter; no model fitting occurs here."""

from dataclasses import dataclass
from typing import List

import pandas as pd

from modules.ML.backtest_config import BacktestConfig


@dataclass(frozen=True)
class FilterResult:
    final_signal: str
    reasons: List[str]

    @property
    def reason_text(self) -> str:
        return "|".join(self.reasons)


def apply_risk_filter(
    raw_signal: str,
    row: pd.Series,
    config: BacktestConfig,
) -> FilterResult:
    """Convert a risky BUY to NO_TRADE while preserving the raw signal."""
    signal = str(raw_signal).strip().upper()
    if signal != "BUY":
        return FilterResult(signal, [])

    reasons: List[str] = []
    if config.use_bear_filter and row.get("market_regime") == "BEAR":
        reasons.append("bear_regime")
    if (
        config.use_ma20_filter
        and float(row.get("distance_ma20", 0.0)) < 0
        and float(row.get("ret_5", 0.0)) < 0
    ):
        reasons.append("below_ma20_negative_ret5")
    if config.use_volatility_filter and bool(row.get("high_volatility", False)):
        reasons.append("high_volatility")
    if config.use_shock_filter and bool(row.get("shock_cooldown", False)):
        reasons.append("shock_cooldown")
    if (
        config.use_shock_filter
        and config.cooldown_sessions == 0
        and float(row.get("previous_shock", 0.0)) >= config.shock_threshold
    ):
        reasons.append("previous_shock")

    return FilterResult("NO_TRADE" if reasons else "BUY", reasons)


def shock_cooldown_mask(
    returns: pd.Series,
    threshold: float,
    sessions: int,
) -> pd.Series:
    """
    Mark target t when any of the previous N sessions was a shock.

    shift(1) is mandatory: the target return itself is never inspected.
    """
    if sessions < 0:
        raise ValueError("sessions không được âm")
    if threshold <= 0:
        raise ValueError("threshold phải > 0")
    shock = (pd.to_numeric(returns, errors="coerce").abs().ge(float(threshold)).astype("boolean"))
    if sessions == 0:
        return pd.Series(False, index=returns.index, dtype=bool)

    shifted = shock.shift(1)
    return (
        shifted
        .rolling(sessions, min_periods=1)
        .max()
        .fillna(False)
        .astype(bool)
    )


__all__ = ["FilterResult", "apply_risk_filter", "shock_cooldown_mask"]

"""Validation-only tuning utilities that reuse cached SARIMAX predictions."""

from dataclasses import replace
from itertools import product
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from modules.ML.backtest import _filter_report, _strategy_metrics
from modules.ML.backtest_config import BacktestConfig
from modules.ML.risk_filter import apply_risk_filter


def candidate_configs(base: BacktestConfig) -> List[BacktestConfig]:

    config = [
        replace(
            base,
            shock_threshold=shock,
            volatility_percentile=percentile,
            cooldown_sessions=cooldown,
            fitted_volatility_threshold=None,
        )
        for shock, percentile, cooldown in product(
            (0.05, 0.06, 0.07),
            (0.90, 0.95, 0.975),
            (0, 1),
        )
    ]

    #Ablation: không dùng volatility filter
    config.append(
        replace(
            base,
            use_volatility_filter=False,
            fitted_volatility_threshold=None,
        )
    )

    #Ablation: không dùng shock filter
    config.append(
        replace(
            base,
            use_shock_filter=False,
            fitted_volatility_threshold=None,
        )
    )

    #Baseline: chỉ giữ nguyên tín hiệu SARIMAX
    config.append(
        replace(
            base,
            use_bear_filter=False,
            use_ma20_filter=False,
            use_volatility_filter=False,
            use_shock_filter=False,
            fitted_volatility_threshold=None,
        )
    )

    return config


def apply_config_to_validation(
    validation_result: pd.DataFrame,
    config: BacktestConfig,
) -> Tuple[pd.DataFrame, BacktestConfig]:
    """Apply a filter config without refitting or changing SARIMAX forecasts."""
    if validation_result.attrs.get("metrics", {}).get("evaluation_segment") != "validation":
        raise ValueError("Risk tuning chỉ được chạy trên segment='validation'")
    required = {
        "raw_signal", "raw_position", "raw_net_return", "actual",
        "previous_shock", "vnindex_volatility_20", "market_regime",
    }
    missing = required.difference(validation_result.columns)
    if missing:
        raise ValueError(f"Validation result thiếu risk columns: {sorted(missing)}")

    metrics = validation_result.attrs["metrics"]
    calibration = pd.Series(metrics.get("risk_calibration_volatility", []), dtype=float).dropna()
    if calibration.empty:
        raise ValueError("Thiếu volatility calibration sample trước validation")
    threshold = float(calibration.quantile(config.volatility_percentile))
    locked = replace(config, fitted_volatility_threshold=threshold)

    out = validation_result.copy()
    out["high_volatility"] = out["vnindex_volatility_20"] > threshold
    if config.cooldown_sessions == 0:
        out["shock_cooldown"] = False
    else:
        prefix = list(
            metrics.get("shock_history_before_evaluation", [])
        )
        needed_prefix = config.cooldown_sessions - 1
        prefix_tail = (
            prefix[-needed_prefix:]
            if needed_prefix else []
        )

        previous = pd.Series(
            prefix_tail + out["previous_shock"].tolist(),
            dtype="float64",
        )

        cooldown = (
            previous
            .rolling(
                config.cooldown_sessions,
                min_periods=1,
            )
            .max()
            .ge(config.shock_threshold)
        )

        out["shock_cooldown"] = (
            cooldown
            .iloc[-len(out):]
            .to_numpy(dtype=bool)
        )

    final_signals, reasons = [], []
    for _, row in out.iterrows():
        filtered = apply_risk_filter(row["raw_signal"], row, locked)
        final_signals.append(filtered.final_signal)
        reasons.append(filtered.reason_text)
    out["final_signal"] = final_signals
    out["filter_reason"] = reasons
    out["final_position"] = out["final_signal"].eq("BUY").astype(int)
    cost = float(metrics.get("round_trip_cost_bps", 0.0)) / 10_000.0
    out["final_gross_return"] = out["final_position"] * out["actual"]
    out["final_net_return"] = out["final_gross_return"] - out["final_position"] * cost
    out["signal"] = out["final_signal"]
    out["position"] = out["final_position"]
    out["gross_return"] = out["final_gross_return"]
    out["net_return"] = out["final_net_return"]
    return out, locked


def tune_risk_filter(
    validation_result: pd.DataFrame,
    configs: Iterable[BacktestConfig],
    min_coverage: float = 0.02,
) -> Tuple[BacktestConfig, pd.DataFrame]:
    rows = []
    locked_configs = []
    for config in configs:
        filtered, locked = apply_config_to_validation(validation_result, config)
        strategy = _strategy_metrics(filtered, "final_position", "final_net_return")
        report = _filter_report(filtered)
        score = (
            strategy["cumulative_net_return"]
            + 0.25 * strategy["sharpe"]
            + strategy["max_drawdown"]
        )
        rows.append({
            "coverage": strategy["coverage"],
            "net_return": strategy["cumulative_net_return"],
            "sharpe": strategy["sharpe"],
            "max_drawdown": strategy["max_drawdown"],
            "blocked_winners": report["blocked_winners"],
            "blocked_losers": report["blocked_losers"],
            "score": score,
        })
        locked_configs.append(locked)
    table = pd.DataFrame(rows)
    eligible = table.index[table["coverage"] >= float(min_coverage)]
    candidates = eligible if len(eligible) else table.index
    best_index = int(table.loc[candidates, "score"].idxmax())
    table["selected"] = False
    table.loc[best_index, "selected"] = True
    return locked_configs[best_index], table


__all__ = ["candidate_configs", "apply_config_to_validation", "tune_risk_filter"]

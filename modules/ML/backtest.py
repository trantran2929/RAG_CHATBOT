"""Point-in-time walk-forward evaluation for the production SARIMAX predictor."""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.api.stock_api import get_close_series
from modules.ML.backtest_config import BacktestConfig
from modules.ML.metrics import mae, rmse
from modules.ML.pipeline import (
    _align_exog_to_y,
    _apply_scaler,
    _signal_threshold_return,
    _standardize_df,
    signal_from_return,
)
from modules.ML.predictors.sarimax_exog import arima_select_fit
from modules.ML.risk_features import (
    build_risk_features,
    fit_volatility_threshold,
    label_market_regime,
)
from modules.ML.risk_filter import apply_risk_filter, shock_cooldown_mask


_VALID_SEGMENTS = {"all", "validation", "final_test"}
_MIN_TRAIN_RETURNS = 30


def _resolve_evaluation_positions(
    returns: pd.Series,
    *,
    segment: str,
    test_days: int,
    validation_days: int,
    final_test_days: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple[int, int]:
    """
    Return an inclusive start and exclusive end position for evaluation.

    Date bounds are inclusive and intentionally cannot be combined with a named
    validation/final-test segment, preventing ambiguous or overlapping splits.
    """
    normalized_segment = str(segment).strip().lower()
    if normalized_segment not in _VALID_SEGMENTS:
        allowed = ", ".join(sorted(_VALID_SEGMENTS))
        raise ValueError(f"segment phải là một trong: {allowed}")
    if test_days < 5:
        raise ValueError("test_days phải >= 5")
    if validation_days < 5 or final_test_days < 5:
        raise ValueError("validation_days và final_test_days phải >= 5")
    if returns.empty:
        raise ValueError("Không có return để tạo evaluation split")

    has_date_bounds = start_date is not None or end_date is not None
    if has_date_bounds:
        if normalized_segment != "all":
            raise ValueError(
                "start_date/end_date chỉ dùng với segment='all'; "
                "không kết hợp với validation/final_test"
            )
        index = pd.DatetimeIndex(pd.to_datetime(returns.index)).tz_localize(None)
        start_ts = (
            pd.Timestamp(start_date).tz_localize(None)
            if start_date is not None
            else index[0]
        )
        end_ts = (
            pd.Timestamp(end_date).tz_localize(None)
            if end_date is not None
            else index[-1]
        )
        if start_ts > end_ts:
            raise ValueError("start_date phải <= end_date")
        selected = np.flatnonzero((index >= start_ts) & (index <= end_ts))
        if len(selected) == 0:
            raise ValueError("Khoảng ngày không chứa phiên giao dịch nào")
        start, end = int(selected[0]), int(selected[-1] + 1)
    elif normalized_segment == "validation":
        end = len(returns) - final_test_days
        start = end - validation_days
    elif normalized_segment == "final_test":
        end = len(returns)
        start = end - final_test_days
    else:
        end = len(returns)
        start = end - test_days

    if start < _MIN_TRAIN_RETURNS:
        required = (end - start) + _MIN_TRAIN_RETURNS
        raise ValueError(
            "Không đủ dữ liệu trước evaluation segment để train walk-forward; "
            f"cần ít nhất {required} return, hiện có {len(returns)}"
        )
    if end <= start:
        raise ValueError("Evaluation segment rỗng")
    return start, end


def _fit_predict_fold(
    returns: pd.Series,
    target_pos: int,
    exog: Optional[pd.DataFrame],
) -> float:
    """Fit the same SARIMAX selection used by production and forecast one row."""
    y_train = returns.iloc[:target_pos]
    y_model = pd.Series(
        y_train.to_numpy(dtype="float64"),
        index=pd.RangeIndex(len(y_train)),
    )

    X_train = None
    X_next = None
    use_exog = exog is not None and not exog.empty
    if use_exog:
        fold_raw = exog.reindex(returns.index).fillna(0.0)
        X_train_raw = fold_raw.iloc[:target_pos]
        X_next_raw = fold_raw.iloc[[target_pos]]
        use_exog = bool(np.abs(X_train_raw.to_numpy(dtype=float)).sum() > 0)
        if use_exog:
            X_train, scaler = _standardize_df(X_train_raw)
            X_next = _apply_scaler(X_next_raw, scaler)[X_train.columns]
            X_train = X_train.reset_index(drop=True)

    fit, _, _ = arima_select_fit(
        y_model,
        d=0,
        max_p=max(1, int(os.getenv("SARIMAX_MAX_P", "2"))),
        max_q=max(1, int(os.getenv("SARIMAX_MAX_Q", "2"))),
        trends=("n", "c"),
        exog=X_train if use_exog else None,
    )
    forecast = fit.get_forecast(steps=1, exog=X_next if use_exog else None)
    return float(np.asarray(forecast.predicted_mean, dtype=float).reshape(-1)[0])


def _strategy_metrics(
    result: pd.DataFrame,
    position_col: str = "position",
    return_col: str = "net_return",
) -> Dict[str, float]:
    traded = result[result[position_col] != 0]
    wins = traded[return_col] > 0
    gross_profit = float(traded.loc[traded[return_col] > 0, return_col].sum())
    gross_loss = abs(float(traded.loc[traded[return_col] < 0, return_col].sum()))
    equity = np.exp(result[return_col].cumsum())
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    std = float(result[return_col].std(ddof=1))
    sharpe = (
        float(np.sqrt(252.0) * result[return_col].mean() / std)
        if std > 1e-12
        else 0.0
    )
    return {
        "coverage": float(len(traded) / len(result)) if len(result) else 0.0,
        "trade_count": int(len(traded)),
        "win_rate": float(wins.mean()) if len(traded) else 0.0,
        "cumulative_net_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 1e-12 else None,
    }


def _filter_report(result: pd.DataFrame) -> Dict[str, float]:
    blocked = result[
        result["raw_signal"].eq("BUY") & result["final_signal"].eq("NO_TRADE")
    ]
    kept = result[
        result["raw_signal"].eq("BUY") & result["final_signal"].eq("BUY")
    ]
    return {
        "blocked_count": int(len(blocked)),
        "blocked_winners": int((blocked["actual"] > 0).sum()),
        "blocked_losers": int((blocked["actual"] < 0).sum()),
        "blocked_actual_return": float(blocked["actual"].sum()),
        "kept_count": int(len(kept)),
        "kept_winners": int((kept["actual"] > 0).sum()),
        "kept_losers": int((kept["actual"] < 0).sum()),
        "kept_actual_return": float(kept["actual"].sum()),
    }


def backtest_gap_model(
    symbol: str,
    test_days: int = 250,
    segment: str = "all",
    validation_days: Optional[int] = None,
    final_test_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[BacktestConfig] = None,
    round_trip_cost_bps: Optional[float] = None,
    signal_buffer_bps: Optional[float] = None,
    use_exog: bool = True,
    allow_short: bool = False,
    apply_risk_filtering: bool = False,
    volatility_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Walk forward one market session at a time using production SARIMAX logic.

    Target is next-session close-to-close log-return. At target t, training data
    ends at t-1 and every exogenous row is shifted to contain only information
    available before t.
    """
    active_config = config or BacktestConfig()
    validation_days = int(
        active_config.validation_days
        if validation_days is None
        else validation_days
    )
    final_test_days = int(
        active_config.final_test_days
        if final_test_days is None
        else final_test_days
    )
    if config is not None:
        if round_trip_cost_bps is None:
            round_trip_cost_bps = active_config.round_trip_cost_bps
        if signal_buffer_bps is None:
            signal_buffer_bps = active_config.signal_buffer_bps

    requested_evaluation_days = max(
        test_days,
        validation_days + final_test_days,
    )
    close = get_close_series(
        symbol,
        days=max(730, requested_evaluation_days + 180),
    )
    close = pd.to_numeric(close, errors="coerce").dropna().astype("float64")

    returns = np.log(close / close.shift(1)).dropna()
    start, end = _resolve_evaluation_positions(
        returns,
        segment=segment,
        test_days=test_days,
        validation_days=validation_days,
        final_test_days=final_test_days,
        start_date=start_date,
        end_date=end_date,
    )
    exog = _align_exog_to_y(symbol.upper(), returns, shift=1) if use_exog else None
    risk_features = None
    fitted_threshold = volatility_threshold
    if apply_risk_filtering:
        market_close = get_close_series(
            "VNINDEX",
            days=max(900, requested_evaluation_days + 400),
        )
        risk_features = build_risk_features(close, market_close)
        if fitted_threshold is None:
            fitted_threshold = active_config.fitted_volatility_threshold
        if fitted_threshold is None:
            if str(segment).strip().lower() == "final_test":
                raise ValueError(
                    "final_test yêu cầu fitted_volatility_threshold đã khóa từ validation"
                )
            fitted_threshold = fit_volatility_threshold(
                risk_features,
                active_config.volatility_percentile,
                end_date=returns.index[start],
            )

        risk_features = label_market_regime(
            risk_features,
            fitted_threshold,
        )
        cooldown = shock_cooldown_mask(
            returns,
            active_config.shock_threshold,
            active_config.cooldown_sessions,
        ).reindex(risk_features.index)
        risk_features["shock_cooldown"] = (
            cooldown
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
    cost_return = max(
        0.0,
        float(
            os.getenv("TRADING_ROUND_TRIP_COST_BPS", "35")
            if round_trip_cost_bps is None
            else round_trip_cost_bps
        ),
    ) / 10_000.0
    threshold = _signal_threshold_return(round_trip_cost_bps, signal_buffer_bps)

    rows = []
    for target_pos in range(start, end):
        pred = _fit_predict_fold(returns, target_pos, exog)
        actual = float(returns.iloc[target_pos])
        raw_signal = signal_from_return(pred, round_trip_cost_bps, signal_buffer_bps)
        final_signal = raw_signal
        filter_reason = ""
        risk_values = {}
        if apply_risk_filtering:
            risk_row = risk_features.reindex([returns.index[target_pos]]).iloc[0]
            filtered = apply_risk_filter(raw_signal, risk_row, active_config)
            final_signal = filtered.final_signal
            filter_reason = filtered.reason_text
            risk_values = risk_row.to_dict()
        # Vietnamese cash equities are evaluated long-only by default. SELL
        # therefore means exit/avoid unless an explicitly shortable instrument
        # is being tested.
        raw_position = 1 if raw_signal == "BUY" else -1 if raw_signal == "SELL" and allow_short else 0
        final_position = 1 if final_signal == "BUY" else -1 if final_signal == "SELL" and allow_short else 0
        raw_gross_return = float(raw_position * actual)
        raw_net_return = raw_gross_return - (cost_return if raw_position else 0.0)
        final_gross_return = float(final_position * actual)
        final_net_return = final_gross_return - (cost_return if final_position else 0.0)
        rows.append(
            {
                "date": returns.index[target_pos],
                "pred": pred,
                "actual": actual,
                "train_end": returns.index[target_pos - 1],
                "raw_signal": raw_signal,
                "final_signal": final_signal,
                "filter_reason": filter_reason,
                "raw_position": raw_position,
                "final_position": final_position,
                "raw_gross_return": raw_gross_return,
                "raw_net_return": raw_net_return,
                "final_gross_return": final_gross_return,
                "final_net_return": final_net_return,
                "signal": final_signal,
                "position": final_position,
                "gross_return": final_gross_return,
                "net_return": final_net_return,
                "correct_direction": int(np.sign(pred) == np.sign(actual)),
                **risk_values,
            }
        )

    result = pd.DataFrame(rows).set_index("date")
    always_up_accuracy = float((result["actual"] > 0).mean())
    previous_direction = np.sign(returns.shift(1).reindex(result.index))
    valid_previous = previous_direction.notna()
    previous_accuracy = float(
        (
            previous_direction.loc[valid_previous]
            == np.sign(result.loc[valid_previous, "actual"])
        ).mean()
    )
    result["momentum_direction"] = previous_direction
    result["sarimax_direction"] = np.sign(result["pred"])
    result["momentum_agreement"] = np.where(
        result["sarimax_direction"].eq(result["momentum_direction"]),
        "AGREE",
        "DISAGREE",
    )
    raw_metrics = _strategy_metrics(result, "raw_position", "raw_net_return")
    filtered_metrics = _strategy_metrics(result, "final_position", "final_net_return")
    momentum_report = {}
    for name, group in result.groupby("momentum_agreement"):
        momentum_report[str(name)] = {
            "sessions": int(len(group)),
            "directional_accuracy": float(group["correct_direction"].mean()),
            "raw_net_return": float(np.exp(group["raw_net_return"].sum()) - 1.0),
            "final_net_return": float(np.exp(group["final_net_return"].sum()) - 1.0),
        }
    result.attrs["metrics"] = {
        "rmse": rmse(result["actual"], result["pred"]),
        "mae": mae(result["actual"], result["pred"]),
        "directional_accuracy": float(result["correct_direction"].mean()),
        "test_size": int(len(result)),
        "evaluation_segment": str(segment).strip().lower(),
        "evaluation_start": pd.Timestamp(result.index[0]).date().isoformat(),
        "evaluation_end": pd.Timestamp(result.index[-1]).date().isoformat(),
        "validation_days": validation_days,
        "final_test_days": final_test_days,
        "method": "expanding_window_production_sarimax",
        "target": "next_session_close_to_close_log_return",
        "feature_timing": "information_available_through_previous_session",
        "uses_exog": bool(exog is not None and not exog.empty),
        "signal_threshold_return": threshold,
        "round_trip_cost_bps": cost_return * 10_000.0,
        "allow_short": bool(allow_short),
        "risk_filtering": bool(apply_risk_filtering),
        "fitted_volatility_threshold": fitted_threshold,
        "risk_calibration_volatility": (
            pd.to_numeric(
                risk_features.loc[
                    risk_features.index < returns.index[start],
                    "vnindex_volatility_20",
                ],
                errors="coerce",
            ).dropna().tail(500).tolist()
            if risk_features is not None
            else []
        ),
        "shock_history_before_evaluation": (
            returns.abs().iloc[max(0, start - 10):start].tolist()
        ),
        "raw_strategy": raw_metrics,
        "filtered_strategy": filtered_metrics,
        "filter_report": _filter_report(result),
        "momentum_report": momentum_report,
        "baselines": {
            "always_up_directional_accuracy": always_up_accuracy,
            "previous_session_directional_accuracy": previous_accuracy,
            "buy_and_hold_return": float(np.exp(result["actual"].sum()) - 1.0),
        },
        **filtered_metrics,
    }
    return result


def print_backtest_report(result: pd.DataFrame) -> None:
    metrics = result.attrs.get("metrics", {})
    baselines = metrics.get("baselines", {})
    raw = metrics.get("raw_strategy", {})
    filtered = metrics.get("filtered_strategy", {})
    print("===== WALK-FORWARD PRODUCTION SARIMAX =====")
    print(f"Target: {metrics.get('target')}")
    print(f"RMSE: {metrics.get('rmse'):.6f}")
    print(f"MAE: {metrics.get('mae'):.6f}")
    print(f"Directional accuracy: {metrics.get('directional_accuracy'):.2%}")
    print(f"Coverage: {metrics.get('coverage'):.2%}")
    print(f"Trades: {metrics.get('trade_count')}")
    print(f"Win rate after costs: {metrics.get('win_rate'):.2%}")
    print(f"Net return: {metrics.get('cumulative_net_return'):.2%}")
    print(f"Sharpe: {metrics.get('sharpe'):.3f}")
    print(f"Max drawdown: {metrics.get('max_drawdown'):.2%}")
    print(f"Always-up accuracy: {baselines.get('always_up_directional_accuracy'):.2%}")
    print(
        "Previous-session accuracy: "
        f"{baselines.get('previous_session_directional_accuracy'):.2%}"
    )
    print(f"Buy-and-hold return: {baselines.get('buy_and_hold_return'):.2%}")
    print(f"Test sessions: {metrics.get('test_size')}")
    if metrics.get("risk_filtering"):
        print("----- RISK FILTER COMPARISON -----")
        print(f"Raw trades: {raw.get('trade_count')}")
        print(f"Filtered trades: {filtered.get('trade_count')}")
        print(f"Raw net return: {raw.get('cumulative_net_return'):.2%}")
        print(f"Filtered net return: {filtered.get('cumulative_net_return'):.2%}")
        print(f"Raw max drawdown: {raw.get('max_drawdown'):.2%}")
        print(f"Filtered max drawdown: {filtered.get('max_drawdown'):.2%}")
        print(f"Filter report: {metrics.get('filter_report')}")


def plot_prediction(df: pd.DataFrame):
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["actual"], label="Actual return")
    plt.plot(df.index, df["pred"], label="Predicted return")
    plt.legend()
    plt.title("Walk-forward predicted vs actual returns")
    plt.show()

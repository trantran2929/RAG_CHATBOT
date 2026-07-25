"""Point-in-time walk-forward evaluation for the production SARIMAX predictor."""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.api.stock_api import get_close_series
from modules.ML.metrics import mae, rmse
from modules.ML.pipeline import (
    _align_exog_to_y,
    _apply_scaler,
    _signal_threshold_return,
    _standardize_df,
    signal_from_return,
)
from modules.ML.predictors.sarimax_exog import arima_select_fit


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


def _strategy_metrics(result: pd.DataFrame) -> Dict[str, float]:
    traded = result[result["position"] != 0]
    wins = traded["net_return"] > 0
    gross_profit = float(traded.loc[traded["net_return"] > 0, "net_return"].sum())
    gross_loss = abs(float(traded.loc[traded["net_return"] < 0, "net_return"].sum()))
    equity = np.exp(result["net_return"].cumsum())
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    std = float(result["net_return"].std(ddof=1))
    sharpe = (
        float(np.sqrt(252.0) * result["net_return"].mean() / std)
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


def backtest_gap_model(
    symbol: str,
    test_days: int = 250,
    round_trip_cost_bps: Optional[float] = None,
    signal_buffer_bps: Optional[float] = None,
    use_exog: bool = True,
    allow_short: bool = False,
) -> pd.DataFrame:
    """
    Walk forward one market session at a time using production SARIMAX logic.

    Target is next-session close-to-close log-return. At target t, training data
    ends at t-1 and every exogenous row is shifted to contain only information
    available before t.
    """
    if test_days < 5:
        raise ValueError("test_days phải >= 5")

    close = get_close_series(symbol, days=max(730, test_days + 180))
    close = pd.to_numeric(close, errors="coerce").dropna().astype("float64")
    if len(close) < test_days + 30:
        raise ValueError("Không đủ dữ liệu cho backtest walk-forward.")

    returns = np.log(close / close.shift(1)).dropna()
    exog = _align_exog_to_y(symbol.upper(), returns, shift=1) if use_exog else None
    start = len(returns) - test_days
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
    for target_pos in range(start, len(returns)):
        pred = _fit_predict_fold(returns, target_pos, exog)
        actual = float(returns.iloc[target_pos])
        signal = signal_from_return(pred, round_trip_cost_bps, signal_buffer_bps)
        # Vietnamese cash equities are evaluated long-only by default. SELL
        # therefore means exit/avoid unless an explicitly shortable instrument
        # is being tested.
        position = 1 if signal == "BUY" else -1 if signal == "SELL" and allow_short else 0
        gross_return = float(position * actual)
        net_return = gross_return - (cost_return if position else 0.0)
        rows.append(
            {
                "date": returns.index[target_pos],
                "pred": pred,
                "actual": actual,
                "train_end": returns.index[target_pos - 1],
                "signal": signal,
                "position": position,
                "gross_return": gross_return,
                "net_return": net_return,
                "correct_direction": int(np.sign(pred) == np.sign(actual)),
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
    result.attrs["metrics"] = {
        "rmse": rmse(result["actual"], result["pred"]),
        "mae": mae(result["actual"], result["pred"]),
        "directional_accuracy": float(result["correct_direction"].mean()),
        "test_size": int(len(result)),
        "method": "expanding_window_production_sarimax",
        "target": "next_session_close_to_close_log_return",
        "feature_timing": "information_available_through_previous_session",
        "uses_exog": bool(exog is not None and not exog.empty),
        "signal_threshold_return": threshold,
        "round_trip_cost_bps": cost_return * 10_000.0,
        "allow_short": bool(allow_short),
        "baselines": {
            "always_up_directional_accuracy": always_up_accuracy,
            "previous_session_directional_accuracy": previous_accuracy,
            "buy_and_hold_return": float(np.exp(result["actual"].sum()) - 1.0),
        },
        **_strategy_metrics(result),
    }
    return result


def print_backtest_report(result: pd.DataFrame) -> None:
    metrics = result.attrs.get("metrics", {})
    baselines = metrics.get("baselines", {})
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


def plot_prediction(df: pd.DataFrame):
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["actual"], label="Actual return")
    plt.plot(df.index, df["pred"], label="Predicted return")
    plt.legend()
    plt.title("Walk-forward predicted vs actual returns")
    plt.show()

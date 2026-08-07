"""Fixed diagnostics for a rejected Phase 4 validation policy; no tuning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROUND_TRIP_COST = 35.0 / 10_000.0
MIN_FLAG_ACTIONS = 10
FLAG_ACCURACY_GAP = -0.05


def _diagnostic_metrics(frame: pd.DataFrame, overall_accuracy: float) -> dict:
    action = frame["signal"].isin(["BUY", "SELL"])
    buy = frame["signal"].eq("BUY")
    sell = frame["signal"].eq("SELL")
    correct = (buy & frame["target"].eq("UP")) | (sell & frame["target"].eq("DOWN"))
    actions = int(action.sum())
    accuracy = float(correct[action].mean()) if actions else 0.0
    buy_net = frame.loc[buy, "future_return"] - ROUND_TRIP_COST
    gap = accuracy - overall_accuracy
    return {
        "rows": int(len(frame)), "actions": actions,
        "buys": int(buy.sum()), "sells": int(sell.sum()),
        "no_trade": int(frame["signal"].eq("NO_TRADE").sum()),
        "action_coverage": actions / len(frame) if len(frame) else 0.0,
        "action_direction_accuracy": accuracy,
        "accuracy_gap_vs_overall": gap,
        "incorrect_actions": int((action & ~correct).sum()),
        "buy_direction_accuracy": float(frame.loc[buy, "target"].eq("UP").mean()) if buy.any() else 0.0,
        "sell_direction_accuracy": float(frame.loc[sell, "target"].eq("DOWN").mean()) if sell.any() else 0.0,
        "buy_win_rate_after_cost": float((buy_net > 0).mean()) if len(buy_net) else 0.0,
        "buy_mean_return_after_cost": float(buy_net.mean()) if len(buy_net) else 0.0,
        "no_trade_missed_move_rate": float(
            frame.loc[frame["signal"].eq("NO_TRADE"), "target"].isin(["UP", "DOWN"]).mean()
        ) if frame["signal"].eq("NO_TRADE").any() else 0.0,
        "diagnostic_flag": actions >= MIN_FLAG_ACTIONS and gap <= FLAG_ACCURACY_GAP,
    }


def _group_report(frame, group_column, overall_accuracy):
    rows = []
    for group, subset in frame.groupby(group_column, observed=True, sort=True):
        rows.append({group_column: group, **_diagnostic_metrics(subset, overall_accuracy)})
    return pd.DataFrame(rows)


def run_diagnostics(dataset_path: Path, signals_path: Path, policy_path: Path, output_dir: Path):
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    if policy.get("approved_for_signal_use") is not False:
        raise ValueError("diagnostics này yêu cầu baseline policy REJECTED")
    dataset = pd.read_csv(dataset_path)
    validation = dataset[dataset["split"].eq("validation")][
        ["symbol", "date", "market_regime", "future_return", "target"]
    ].copy()
    signals = pd.read_csv(signals_path)[["symbol", "date", "signal"]].copy()
    validation["date"] = pd.to_datetime(validation["date"], errors="raise")
    signals["date"] = pd.to_datetime(signals["date"], errors="raise")
    frame = validation.merge(signals, on=["symbol", "date"], how="left", validate="one_to_one")
    if len(frame) != len(validation) or frame["signal"].isna().any():
        raise ValueError("signals không phủ đủ validation")
    overall = _diagnostic_metrics(frame, 0.0)
    overall_accuracy = overall["action_direction_accuracy"]
    overall["accuracy_gap_vs_overall"] = 0.0
    overall["diagnostic_flag"] = False
    by_symbol = _group_report(frame, "symbol", overall_accuracy)
    by_regime = _group_report(frame, "market_regime", overall_accuracy)
    by_side = []
    for side in ("BUY", "SELL"):
        subset = frame[frame["signal"].eq(side)]
        metrics = _diagnostic_metrics(subset, overall_accuracy)
        by_side.append({"signal_side": side, **metrics})
    by_side = pd.DataFrame(by_side)
    flagged = pd.concat([
        by_symbol[by_symbol["diagnostic_flag"]].assign(group_type="symbol").rename(columns={"symbol": "group"}),
        by_regime[by_regime["diagnostic_flag"]].assign(group_type="market_regime").rename(columns={"market_regime": "group"}),
    ], ignore_index=True)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    by_symbol.to_csv(output / "diagnostic_by_symbol.csv", index=False)
    by_regime.to_csv(output / "diagnostic_by_market_regime.csv", index=False)
    by_side.to_csv(output / "diagnostic_by_signal_side.csv", index=False)
    flagged.to_csv(output / "diagnostic_flagged_groups.csv", index=False)
    payload = {
        "policy_status": "REJECTED", "analysis_type": "fixed_no_tuning",
        "overall": overall,
        "flag_rule": {"min_actions": MIN_FLAG_ACTIONS, "accuracy_gap_lte": FLAG_ACCURACY_GAP},
        "flagged_groups": flagged[["group_type", "group", "actions", "action_direction_accuracy",
                                    "accuracy_gap_vs_overall", "incorrect_actions"]].to_dict("records")
            if len(flagged) else [],
        "recommended_action": "diagnose_only_do_not_change_policy_or_open_final_test",
        "final_test_read": False,
    }
    (output / "phase4_diagnostic_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload, by_symbol, by_regime, by_side


__all__ = ["run_diagnostics", "_diagnostic_metrics"]

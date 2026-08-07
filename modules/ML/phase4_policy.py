"""Lock compact calibration and select a bounded validation-only signal policy."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from modules.ML.phase3_config import Phase3Config
from modules.ML.phase3_evaluation import _fit_predict, expanding_date_folds, load_phase3_dataset
from modules.ML.phase3_models import LABELS


PROBABILITY_THRESHOLDS = (0.36, 0.40, 0.44, 0.48)
PROBABILITY_MARGINS = (0.00, 0.03, 0.06)
MIN_ACTIONS = 30
MIN_SIDE_ACTIONS = 10
ROUND_TRIP_COST = 35.0 / 10_000.0
MIN_ACCEPTED_ACTION_ACCURACY = 0.50
MIN_ACCEPTED_BUY_WIN_RATE = 0.50


def probability_to_signal(probabilities, threshold: float, margin: float):
    probabilities = np.asarray(probabilities, dtype=float)
    down = probabilities[:, LABELS.index("DOWN")]
    no_trade = probabilities[:, LABELS.index("NO_TRADE")]
    up = probabilities[:, LABELS.index("UP")]
    buy = (up >= threshold) & ((up - np.maximum(down, no_trade)) >= margin)
    sell = (down >= threshold) & ((down - np.maximum(up, no_trade)) >= margin)
    return np.where(buy, "BUY", np.where(sell, "SELL", "NO_TRADE"))


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = wins / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    adjustment = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return float((centre - adjustment) / denominator)


def evaluate_policy(validation, probabilities, threshold, margin):
    signals = probability_to_signal(probabilities, threshold, margin)
    actual = validation["target"].to_numpy()
    action = signals != "NO_TRADE"
    correct = ((signals == "BUY") & (actual == "UP")) | ((signals == "SELL") & (actual == "DOWN"))
    actions = int(action.sum()); buys = int((signals == "BUY").sum()); sells = int((signals == "SELL").sum())
    correct_actions = int((correct & action).sum())
    rows = validation[["symbol", "date", "future_return", "target"]].copy()
    rows["signal"] = signals
    rows["net_long_return"] = np.where(
        rows["signal"].eq("BUY"), rows["future_return"] - ROUND_TRIP_COST, 0.0
    )
    daily = rows.groupby("date", observed=True)["net_long_return"].mean()
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    sharpe = 0.0 if daily.std(ddof=0) == 0 else float(daily.mean() / daily.std(ddof=0) * np.sqrt(252))
    buy_rows = rows[rows["signal"].eq("BUY")]
    return {
        "threshold": threshold, "margin": margin, "rows": int(len(rows)),
        "actions": actions, "buys": buys, "sells": sells,
        "action_coverage": actions / len(rows),
        "action_direction_accuracy": correct_actions / actions if actions else 0.0,
        "wilson_lower_bound": wilson_lower_bound(correct_actions, actions),
        "buy_win_rate_after_cost": float((buy_rows["net_long_return"] > 0).mean()) if buys else 0.0,
        "long_only_net_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "long_only_sharpe": sharpe,
        "long_only_max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "eligible": actions >= MIN_ACTIONS and buys >= MIN_SIDE_ACTIONS and sells >= MIN_SIDE_ACTIONS,
    }, rows


def _calibrated_compact_probabilities(train, validation, columns, config):
    oof_probabilities, oof_targets = [], []
    for _, train_index, validation_index in expanding_date_folds(train, config):
        fold_train, fold_validation = train.loc[train_index], train.loc[validation_index]
        _, _, probabilities = _fit_predict("xgboost", fold_train, fold_validation, columns, config)
        oof_probabilities.append(probabilities)
        oof_targets.extend(fold_validation["target"].tolist())
    calibrator = LogisticRegression(max_iter=1000, random_state=config.random_state)
    calibrator.fit(np.vstack(oof_probabilities), oof_targets)
    base_model, _, raw = _fit_predict("xgboost", train, validation, columns, config)
    calibrated = calibrator.predict_proba(raw)
    aligned = np.zeros_like(raw)
    for index, label in enumerate(calibrator.classes_):
        aligned[:, LABELS.index(label)] = calibrated[:, index]
    return base_model, calibrator, aligned, len(oof_targets)


def lock_phase4(dataset_path: Path, phase3_lock_path: Path, output_dir: Path, *, force=False):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    policy_lock = output / "phase4_locked_policy.json"
    calibrator_path = output / "compact_xgboost_calibrator.joblib"
    if (policy_lock.exists() or calibrator_path.exists()) and not force:
        raise FileExistsError("Pha 4 đã được khóa; chỉ dùng --force khi dataset/đặc tả được audit lại")
    phase3_lock = json.loads(Path(phase3_lock_path).read_text(encoding="utf-8"))
    if phase3_lock.get("selected_feature_set") != "compact":
        raise ValueError("phase3 lock không chọn compact")
    dataset_hash = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    if dataset_hash != phase3_lock.get("dataset_sha256"):
        raise ValueError("dataset hash không khớp phase3 lock")
    data = load_phase3_dataset(dataset_path)
    train = data[data["split"].eq("train")].copy()
    validation = data[data["split"].eq("validation")].copy()
    columns = phase3_lock["feature_columns"]
    config = Phase3Config(**phase3_lock["config"])
    base_model, calibrator, probabilities, oof_rows = _calibrated_compact_probabilities(
        train, validation, columns, config
    )
    candidates, signal_tables = [], {}
    for threshold in PROBABILITY_THRESHOLDS:
        for margin in PROBABILITY_MARGINS:
            metrics, rows = evaluate_policy(validation, probabilities, threshold, margin)
            candidates.append(metrics); signal_tables[(threshold, margin)] = rows
    table = pd.DataFrame(candidates)
    eligible = table[table["eligible"]]
    if eligible.empty:
        raise RuntimeError("VALIDATION_REJECTED: không policy nào đủ số BUY/SELL tối thiểu")
    selected = eligible.sort_values(
        ["wilson_lower_bound", "action_direction_accuracy", "action_coverage"],
        ascending=[False, False, False], kind="mergesort",
    ).iloc[0].to_dict()
    rejection_reasons = []
    if selected["action_direction_accuracy"] < MIN_ACCEPTED_ACTION_ACCURACY:
        rejection_reasons.append("action_direction_accuracy_below_50pct")
    if selected["buy_win_rate_after_cost"] < MIN_ACCEPTED_BUY_WIN_RATE:
        rejection_reasons.append("buy_win_rate_after_cost_below_50pct")
    approved = not rejection_reasons
    selected_rows = signal_tables[(selected["threshold"], selected["margin"])]
    artifact = {
        "base_model": base_model, "calibrator": calibrator,
        "labels": LABELS, "feature_columns": columns,
        "dataset_sha256": dataset_hash,
    }
    joblib.dump(artifact, calibrator_path)
    calibrator_hash = hashlib.sha256(calibrator_path.read_bytes()).hexdigest()
    payload = {
        "model": "compact_xgboost_calibrated", "calibration_method": "train_oof_multinomial_sigmoid",
        "calibration_oof_rows": oof_rows, "selected_policy": selected,
        "grid": {"thresholds": PROBABILITY_THRESHOLDS, "margins": PROBABILITY_MARGINS,
                 "min_actions": MIN_ACTIONS, "min_side_actions": MIN_SIDE_ACTIONS},
        "selection_metric": "wilson_lower_bound_action_direction_accuracy",
        "validation_status": "ACCEPTED" if approved else "REJECTED",
        "approved_for_signal_use": approved,
        "rejection_reasons": rejection_reasons,
        "acceptance_gate": {
            "min_action_direction_accuracy": MIN_ACCEPTED_ACTION_ACCURACY,
            "min_buy_win_rate_after_cost": MIN_ACCEPTED_BUY_WIN_RATE,
        },
        "sell_semantics": "directional_or_exit_only_no_short_pnl",
        "round_trip_cost_bps_for_buy_report": 35.0,
        "dataset_sha256": dataset_hash, "calibrator_sha256": calibrator_hash,
        "phase3_lock": str(phase3_lock_path), "final_test_read": False,
    }
    table.to_csv(output / "phase4_policy_grid.csv", index=False)
    selected_rows.to_csv(output / "phase4_validation_signals.csv", index=False)
    policy_lock.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload, table


__all__ = ["probability_to_signal", "wilson_lower_bound", "evaluate_policy", "lock_phase4"]

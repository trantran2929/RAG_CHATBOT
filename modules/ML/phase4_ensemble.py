"""Validation-only compact XGBoost + raw SARIMAX ensemble candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modules.ML.phase3_evaluation import load_phase3_dataset
from modules.ML.phase3_models import LABELS
from modules.ML.phase4_policy import evaluate_policy, probability_to_signal


ENSEMBLE_WEIGHTS = (0.15, 0.25, 0.35)
MIN_ACTIONS = 30
MIN_SIDE_ACTIONS = 10


def load_sarimax_validation(directory: Path, symbols) -> pd.DataFrame:
    directory = Path(directory)
    if "final" in str(directory).lower():
        raise ValueError("từ chối nguồn SARIMAX có tên final")
    rows = []
    for symbol in symbols:
        path = directory / f"{symbol}_validation_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"date", "pred", "actual"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} thiếu cột {sorted(required - set(frame.columns))}")
        frame = frame[["date", "pred", "actual"]].copy()
        frame.insert(0, "symbol", symbol)
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result.duplicated(["symbol", "date"]).any():
        raise ValueError("SARIMAX validation trùng symbol/date")
    return result


def blend_probabilities(xgb_probabilities, sarimax_predictions, weight: float):
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight phải nằm trong [0, 1]")
    vote = np.zeros_like(xgb_probabilities)
    vote[:, LABELS.index("NO_TRADE")] = (np.asarray(sarimax_predictions) == 0).astype(float)
    vote[:, LABELS.index("UP")] = (np.asarray(sarimax_predictions) > 0).astype(float)
    vote[:, LABELS.index("DOWN")] = (np.asarray(sarimax_predictions) < 0).astype(float)
    return (1.0 - weight) * np.asarray(xgb_probabilities) + weight * vote


def _calibrated_probabilities(artifact, validation, columns):
    raw = artifact["base_model"].predict_proba(validation[columns])
    calibrated = artifact["calibrator"].predict_proba(raw)
    aligned = np.zeros_like(raw)
    for index, label in enumerate(artifact["calibrator"].classes_):
        aligned[:, LABELS.index(label)] = calibrated[:, index]
    return aligned


def evaluate_ensemble_candidates(
    dataset_path: Path, calibrator_path: Path, baseline_policy_path: Path,
    sarimax_dir: Path, output_dir: Path,
):
    baseline = json.loads(Path(baseline_policy_path).read_text(encoding="utf-8"))
    if baseline.get("approved_for_signal_use") is not False:
        raise ValueError("baseline phải là policy REJECTED được giữ nguyên")
    dataset_hash = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    if dataset_hash != baseline.get("dataset_sha256"):
        raise ValueError("dataset hash không khớp baseline Pha 4")
    calibrator_hash = hashlib.sha256(Path(calibrator_path).read_bytes()).hexdigest()
    if calibrator_hash != baseline.get("calibrator_sha256"):
        raise ValueError("calibrator hash không khớp baseline Pha 4")
    artifact = joblib.load(calibrator_path)
    if artifact.get("dataset_sha256") != dataset_hash:
        raise ValueError("calibrator hash dataset không khớp")
    data = load_phase3_dataset(dataset_path)
    validation = data[data["split"].eq("validation")].copy().sort_values(
        ["date", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    symbols = sorted(validation["symbol"].unique())
    sarimax = load_sarimax_validation(sarimax_dir, symbols)
    merged = validation.merge(sarimax, on=["symbol", "date"], how="left", validate="one_to_one")
    if merged["pred"].isna().any() or len(merged) != len(validation):
        raise ValueError("SARIMAX không phủ đủ validation XGBoost")
    if not np.allclose(merged["future_return"], merged["actual"], atol=1e-9, rtol=1e-7):
        raise ValueError("actual SARIMAX không khớp future_return dataset")
    probabilities = _calibrated_probabilities(artifact, merged, artifact["feature_columns"])
    selected_policy = baseline["selected_policy"]
    threshold = float(selected_policy["threshold"]); margin = float(selected_policy["margin"])
    candidates, signal_tables = [], {}
    for weight in ENSEMBLE_WEIGHTS:
        blended = blend_probabilities(probabilities, merged["pred"].to_numpy(), weight)
        metrics, rows = evaluate_policy(merged, blended, threshold, margin)
        metrics.update({"candidate": f"blend_{weight:.2f}", "sarimax_weight": weight})
        candidates.append(metrics); signal_tables[metrics["candidate"]] = rows
    base_signals = probability_to_signal(probabilities, threshold, margin)
    sarimax_direction = np.where(merged["pred"] > 0, "BUY", np.where(merged["pred"] < 0, "SELL", "NO_TRADE"))
    gated = probabilities.copy()
    disagreement = (base_signals != "NO_TRADE") & (base_signals != sarimax_direction)
    gated[disagreement] = np.array([0.0, 1.0, 0.0])
    metrics, rows = evaluate_policy(merged, gated, threshold, margin)
    metrics.update({"candidate": "agreement_gate", "sarimax_weight": None})
    candidates.append(metrics); signal_tables["agreement_gate"] = rows
    table = pd.DataFrame(candidates)
    table["sample_eligible"] = (
        (table["actions"] >= MIN_ACTIONS) & (table["buys"] >= MIN_SIDE_ACTIONS)
        & (table["sells"] >= MIN_SIDE_ACTIONS)
    )
    table["passes_acceptance_gate"] = (
        table["sample_eligible"] & (table["action_direction_accuracy"] >= 0.50)
        & (table["buy_win_rate_after_cost"] >= 0.50)
        & (table["wilson_lower_bound"] > float(selected_policy["wilson_lower_bound"]))
    )
    accepted = table[table["passes_acceptance_gate"]]
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "phase4_ensemble_candidate_grid.csv", index=False)
    payload = {
        "baseline_policy": str(baseline_policy_path), "baseline_status": "REJECTED",
        "threshold_and_margin_reused": {"threshold": threshold, "margin": margin},
        "sarimax_source": str(sarimax_dir), "sarimax_input": "raw_pred_only",
        "candidates": table.to_dict("records"), "dataset_sha256": dataset_hash,
        "final_test_read": False,
    }
    if accepted.empty:
        payload.update({"validation_status": "REJECTED", "approved_for_signal_use": False,
                        "selected_candidate": None})
    else:
        selected = accepted.sort_values(
            ["wilson_lower_bound", "action_direction_accuracy", "action_coverage"],
            ascending=[False, False, False], kind="mergesort",
        ).iloc[0].to_dict()
        payload.update({"validation_status": "ACCEPTED", "approved_for_signal_use": True,
                        "selected_candidate": selected})
        signal_tables[selected["candidate"]].to_csv(
            output / "phase4_ensemble_validation_signals.csv", index=False
        )
        (output / "phase4_ensemble_candidate_lock.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (output / "phase4_ensemble_candidate_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload, table


__all__ = ["load_sarimax_validation", "blend_probabilities", "evaluate_ensemble_candidates"]

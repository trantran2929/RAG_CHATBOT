"""CLI for validation-only XGBoost + SARIMAX ensemble candidates."""

import argparse
from pathlib import Path

from modules.ML.phase4_ensemble import evaluate_ensemble_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="models/phase2_audited/phase2_dataset.csv")
    parser.add_argument("--calibrator", default="models/phase4/compact_xgboost_calibrator.joblib")
    parser.add_argument("--baseline-policy", default="models/phase4/phase4_locked_policy.json")
    parser.add_argument("--sarimax-dir", default="models/phase1_candidate_v3")
    parser.add_argument("--output-dir", default="models/phase4_ensemble")
    args = parser.parse_args()
    report, table = evaluate_ensemble_candidates(
        Path(args.dataset), Path(args.calibrator), Path(args.baseline_policy),
        Path(args.sarimax_dir), Path(args.output_dir),
    )
    print("===== PHASE 4 ENSEMBLE CANDIDATES =====")
    print(table.to_string(index=False))
    print(f"Validation status: {report['validation_status']}")
    print(f"Approved for signal use: {report['approved_for_signal_use']}")
    if report["selected_candidate"]:
        print(f"Selected candidate: {report['selected_candidate']['candidate']}")
    else:
        print("Candidate lock created: NO")
    print("Final test read: NO")


if __name__ == "__main__":
    main()

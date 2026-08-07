"""CLI for validation-only Phase 3 calibration, importance and ablation."""

import argparse
from pathlib import Path

from modules.ML.phase3_analysis import run_ablation, run_calibration, run_feature_importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", choices=("calibration", "importance", "ablation", "all"))
    parser.add_argument("--model", choices=("logistic", "xgboost"), default="xgboost")
    parser.add_argument("--dataset", default="models/phase2_audited/phase2_dataset.csv")
    parser.add_argument("--output-dir", default="models/phase3")
    args = parser.parse_args()
    dataset, output = Path(args.dataset), Path(args.output_dir)
    if args.analysis in {"calibration", "all"}:
        result = run_calibration(dataset, output, args.model)
        print("===== CALIBRATION =====")
        print(f"Raw log loss: {result['raw']['log_loss']:.4f}")
        print(f"Calibrated log loss: {result['calibrated']['log_loss']:.4f}")
        print(f"Raw ECE: {result['raw']['ece_10_bins']:.4f}")
        print(f"Calibrated ECE: {result['calibrated']['ece_10_bins']:.4f}")
    if args.analysis in {"importance", "all"}:
        result = run_feature_importance(dataset, output, args.model)
        print("===== TOP 10 FEATURES =====")
        print(result.head(10).to_string(index=False))
    if args.analysis in {"ablation", "all"}:
        result = run_ablation(dataset, output, args.model)
        print("===== ABLATION =====")
        print(result.to_string(index=False))
    print("Final test read: NO")


if __name__ == "__main__":
    main()

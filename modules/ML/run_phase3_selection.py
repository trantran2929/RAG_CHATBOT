"""CLI to compare and lock full versus compact XGBoost without final-test access."""

import argparse
from pathlib import Path

from modules.ML.phase3_selection import compare_and_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="models/phase2_audited/phase2_dataset.csv")
    parser.add_argument("--output-dir", default="models/phase3")
    parser.add_argument("--lock-path", default="models/phase3/phase3_locked_config.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report, folds, validation = compare_and_lock(
        Path(args.dataset), Path(args.output_dir), Path(args.lock_path), force=args.force
    )
    print("===== PHASE 3 FULL VS COMPACT =====")
    print(folds.to_string(index=False))
    print("===== VALIDATION =====")
    print(validation.to_string(index=False))
    print(f"Locked selection: {report['selected_feature_set'].upper()}")
    if report["failed_compact_checks"]:
        print("Failed compact checks: " + ", ".join(report["failed_compact_checks"]))
    print(f"Lock path: {args.lock_path}")
    print("Final test read: NO")


if __name__ == "__main__":
    main()

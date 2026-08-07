"""CLI to lock compact calibration and validation signal policy."""

import argparse
from pathlib import Path

from modules.ML.phase4_policy import lock_phase4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="models/phase2_audited/phase2_dataset.csv")
    parser.add_argument("--phase3-lock", default="models/phase3/phase3_locked_config.json")
    parser.add_argument("--output-dir", default="models/phase4")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report, table = lock_phase4(
        Path(args.dataset), Path(args.phase3_lock), Path(args.output_dir), force=args.force
    )
    print("===== PHASE 4 POLICY GRID =====")
    print(table.to_string(index=False))
    print("===== LOCKED POLICY =====")
    for key, value in report["selected_policy"].items():
        print(f"{key}: {value}")
    print(f"Validation status: {report['validation_status']}")
    print(f"Approved for signal use: {report['approved_for_signal_use']}")
    if report["rejection_reasons"]:
        print("Rejection reasons: " + ", ".join(report["rejection_reasons"]))
    print("Final test read: NO")


if __name__ == "__main__":
    main()

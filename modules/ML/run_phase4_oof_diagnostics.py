"""CLI for raw compact-XGBoost train OOF diagnostics."""

import argparse
from pathlib import Path

from modules.ML.phase4_oof_diagnostics import run_oof_diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="models/phase2_audited/phase2_dataset.csv")
    parser.add_argument("--phase3-lock", default="models/phase3/phase3_locked_config.json")
    parser.add_argument("--output-dir", default="models/phase4_oof_diagnostics")
    args = parser.parse_args()
    summary, by_symbol, by_side, by_fold = run_oof_diagnostics(
        Path(args.dataset), Path(args.phase3_lock), Path(args.output_dir)
    )
    print("===== TRAIN OOF BY SYMBOL =====")
    print(by_symbol.to_string(index=False))
    print("===== TRAIN OOF BY BUY/SELL =====")
    print(by_side.to_string(index=False))
    print("===== TRAIN OOF BY FOLD =====")
    print(by_fold.to_string(index=False))
    print("===== FLAGGED SYMBOLS =====")
    print(summary["flagged_symbols"] or "NONE")
    print("Calibrator used: NO")
    print("Policy changed: NO")
    print("Final test read: NO")


if __name__ == "__main__":
    main()

"""CLI for fixed validation diagnostics of the rejected Phase 4 policy."""

import argparse
from pathlib import Path

from modules.ML.phase4_diagnostics import run_diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="models/phase2_audited/phase2_dataset.csv")
    parser.add_argument("--signals", default="models/phase4/phase4_validation_signals.csv")
    parser.add_argument("--policy", default="models/phase4/phase4_locked_policy.json")
    parser.add_argument("--output-dir", default="models/phase4_diagnostics")
    args = parser.parse_args()
    summary, by_symbol, by_regime, by_side = run_diagnostics(
        Path(args.dataset), Path(args.signals), Path(args.policy), Path(args.output_dir)
    )
    print("===== DIAGNOSTIC BY SYMBOL =====")
    print(by_symbol.to_string(index=False))
    print("===== DIAGNOSTIC BY MARKET REGIME =====")
    print(by_regime.to_string(index=False))
    print("===== DIAGNOSTIC BY SIGNAL SIDE =====")
    print(by_side.to_string(index=False))
    print("===== FLAGGED GROUPS =====")
    print(summary["flagged_groups"] or "NONE")
    print("Policy changed: NO")
    print("Final test read: NO")


if __name__ == "__main__":
    main()

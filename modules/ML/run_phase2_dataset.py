"""CLI that builds the Phase 2 dataset without revealing final-test targets."""

import argparse
from pathlib import Path

from modules.ML.phase2_config import Phase2DatasetConfig
from modules.ML.phase2_dataset import build_phase2_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="FPT,VCB,HPG,PNJ,SHS")
    parser.add_argument("--lookback-days", type=int, default=1100)
    parser.add_argument("--output-dir", default="models/phase2")
    args = parser.parse_args()
    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    config = Phase2DatasetConfig(symbols=symbols, lookback_days=args.lookback_days)
    dataset, audit, metadata = build_phase2_dataset(config, Path(args.output_dir))
    print("===== PHASE 2 DATASET =====")
    print(f"Rows: {len(dataset)}")
    print(f"Rows by split: {metadata['rows_by_split']}")
    print(f"SHA256: {metadata['dataset_sha256']}")
    print("Final-test targets: MASKED")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()

"""CLI for Phase 3 cross-validation and locked validation evaluation."""

import argparse
from pathlib import Path

from modules.ML.phase3_evaluation import evaluate_phase3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("logistic", "xgboost"))
    parser.add_argument("--dataset", default="models/phase2_audited/phase2_dataset.csv")
    parser.add_argument("--output-dir", default="models/phase3")
    args = parser.parse_args()
    report, _ = evaluate_phase3(Path(args.dataset), Path(args.output_dir), args.model)
    print(f"===== PHASE 3 {args.model.upper()} =====")
    for fold in report["cv_folds"]:
        print(f"Fold {fold['fold']}: balanced_accuracy={fold['balanced_accuracy']:.2%}, macro_f1={fold['macro_f1']:.3f}")
    metrics = report["validation"]
    print(f"Validation accuracy: {metrics['accuracy']:.2%}")
    print(f"Validation balanced accuracy: {metrics['balanced_accuracy']:.2%}")
    print(f"Validation macro F1: {metrics['macro_f1']:.3f}")
    print(f"Majority baseline accuracy: {metrics['majority_accuracy']:.2%}")
    print("Final test read: NO")


if __name__ == "__main__":
    main()

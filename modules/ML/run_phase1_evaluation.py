"""CLI for validation tuning and locked final testing; never tunes on final test."""

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from modules.ML.backtest import backtest_gap_model, print_backtest_report
from modules.ML.backtest_config import (
    BacktestConfig,
    load_backtest_config,
    save_backtest_config,
)
from modules.ML.risk_tuning import candidate_configs, tune_risk_filter


DEFAULT_SYMBOLS = ["FPT", "VCB", "HPG", "PNJ", "SHS"]


def _parse_symbols(raw: str):
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def run_validation(symbols, output_dir: Path) -> BacktestConfig:
    """Tune each validation result, then choose the grid row with best mean score."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = BacktestConfig()
    grid = candidate_configs(base)
    reports = []
    locked_by_symbol = []
    for symbol in symbols:
        print(f"\n===== VALIDATION {symbol} =====")
        result = backtest_gap_model(
            symbol,
            segment="validation",
            config=base,
            use_exog=False,
            apply_risk_filtering=True,
        )
        result.to_csv(output_dir / f"{symbol}_validation_predictions.csv")
        _, report = tune_risk_filter(result, grid)
        report = report.add_prefix(f"{symbol}_")
        reports.append(report)
        # Preserve symbol-specific fitted threshold for diagnostics only.
        locked_by_symbol.append([
            tune_risk_filter(result, [cfg])[0] for cfg in grid
        ])

    combined = pd.concat(reports, axis=1)
    score_cols = [c for c in combined if c.endswith("_score")]
    coverage_cols = [c for c in combined if c.endswith("_coverage")]
    combined["mean_score"] = combined[score_cols].mean(axis=1)
    combined["mean_coverage"] = combined[coverage_cols].mean(axis=1)
    eligible = combined.index[
        (combined["mean_coverage"] >= 0.02)
        & (combined["mean_score"] > 0)
    ]
    if len(eligible) == 0:
        combined["selected"] = False
        combined.to_csv(output_dir / "validation_tuning_report.csv", index_label="grid_index")
        config_path = output_dir / "phase1_locked_config.json"
        if config_path.exists():
            config_path.unlink()

        raise RuntimeError(
            "VALIDATION_REJECTED: không cấu hình nào đồng thời đạt "
            "mean_coverage >= 2% và mean_score > 0; "
            "locked config không được tạo"
        )

    best = int(combined.loc[eligible, "mean_score"].idxmax())
    combined["selected"] = False
    combined.loc[best, "selected"] = True
    combined.to_csv(output_dir / "validation_tuning_report.csv", index_label="grid_index")

    # VNINDEX threshold should be identical for aligned periods; median is robust
    # to small provider/date differences between symbol histories.
    thresholds = [items[best].fitted_volatility_threshold for items in locked_by_symbol]
    selected = replace(
        grid[best],
        fitted_volatility_threshold=float(pd.Series(thresholds).median()),
    )
    save_backtest_config(selected, output_dir / "phase1_locked_config.json")
    print(f"\nLocked grid index: {best}")
    print(selected)
    return selected


def run_final_test(symbols, output_dir: Path, config_path: Path) -> None:
    """Evaluate a previously locked config; this function performs no tuning."""
    config = load_backtest_config(config_path)
    if config.fitted_volatility_threshold is None:
        raise ValueError("Locked config thiếu fitted_volatility_threshold")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for symbol in symbols:
        print(f"\n===== FINAL TEST {symbol} =====")
        result = backtest_gap_model(
            symbol,
            segment="final_test",
            config=config,
            use_exog=False,
            apply_risk_filtering=True,
        )
        print_backtest_report(result)
        result.to_csv(output_dir / f"{symbol}_final_test.csv")
        metrics = result.attrs["metrics"]
        summaries.append({
            "symbol": symbol,
            **metrics["filtered_strategy"],
            "directional_accuracy": metrics["directional_accuracy"],
        })
    pd.DataFrame(summaries).to_csv(output_dir / "final_test_summary.csv", index=False)


def run_cost_stress(symbols, output_dir: Path, config_path: Path, costs) -> None:
    """Re-evaluate the locked filter under predeclared cost scenarios."""
    locked = load_backtest_config(config_path)
    if locked.fitted_volatility_threshold is None:
        raise ValueError("Locked config thiếu fitted_volatility_threshold")
    for cost in costs:
        scenario = replace(locked, round_trip_cost_bps=float(cost))
        scenario_dir = output_dir / f"cost_{float(cost):g}bps"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        for symbol in symbols:
            result = backtest_gap_model(
                symbol,
                segment="final_test",
                config=scenario,
                use_exog=False,
                apply_risk_filtering=True,
            )
            result.to_csv(scenario_dir / f"{symbol}.csv")
            metrics = result.attrs["metrics"]
            summaries.append({"symbol": symbol, **metrics["filtered_strategy"]})
        pd.DataFrame(summaries).to_csv(scenario_dir / "summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validation", "final_test", "stress"))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--output-dir", default="models/phase1")
    parser.add_argument("--config", default="models/phase1/phase1_locked_config.json")
    parser.add_argument("--costs", default="35,50,75")
    args = parser.parse_args()
    symbols = _parse_symbols(args.symbols)
    output_dir = Path(args.output_dir)
    if args.mode == "validation":
        run_validation(symbols, output_dir)
    elif args.mode == "final_test":
        run_final_test(symbols, output_dir, Path(args.config))
    else:
        costs = [float(value) for value in args.costs.split(",") if value.strip()]
        run_cost_stress(symbols, output_dir, Path(args.config), costs)


if __name__ == "__main__":
    main()

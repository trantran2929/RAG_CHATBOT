"""Serializable, validated configuration for Phase 1 backtests."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, Union


PathLike = Union[str, os.PathLike]


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters selected on validation and locked before the final test."""

    validation_days: int = 125
    final_test_days: int = 125

    round_trip_cost_bps: float = 35.0
    signal_buffer_bps: float = 15.0

    shock_threshold: float = 0.06
    volatility_percentile: float = 0.75
    fitted_volatility_threshold: Optional[float] = None
    cooldown_sessions: int = 1

    use_bear_filter: bool = True
    use_ma20_filter: bool = True
    use_volatility_filter: bool = True
    use_shock_filter: bool = True

    def __post_init__(self) -> None:
        if self.validation_days < 5:
            raise ValueError("validation_days phải >= 5")
        if self.final_test_days < 5:
            raise ValueError("final_test_days phải >= 5")
        if self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps không được âm")
        if self.signal_buffer_bps < 0:
            raise ValueError("signal_buffer_bps không được âm")
        if not 0 < self.shock_threshold < 1:
            raise ValueError("shock_threshold phải nằm trong (0, 1)")
        if not 0 < self.volatility_percentile < 1:
            raise ValueError("volatility_percentile phải nằm trong (0, 1)")
        if (
            self.fitted_volatility_threshold is not None
            and self.fitted_volatility_threshold < 0
        ):
            raise ValueError("fitted_volatility_threshold không được âm")
        if self.cooldown_sessions < 0:
            raise ValueError("cooldown_sessions không được âm")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_backtest_config(config: BacktestConfig, path: PathLike) -> Path:
    """Atomically save the exact validation-selected configuration as JSON."""
    if not isinstance(config, BacktestConfig):
        raise TypeError("config phải là BacktestConfig")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_backtest_config(path: PathLike) -> BacktestConfig:
    """Load and validate a previously locked configuration."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("File cấu hình phải chứa một JSON object")
    return BacktestConfig(**payload)


__all__ = [
    "BacktestConfig",
    "save_backtest_config",
    "load_backtest_config",
]

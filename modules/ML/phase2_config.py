"""Versioned configuration for the leakage-safe Phase 2 dataset."""

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class Phase2DatasetConfig:
    symbols: Tuple[str, ...] = ("FPT", "VCB", "HPG", "PNJ", "SHS")
    # get_prices_df receives calendar lookback days, not trading sessions.
    lookback_days: int = 1100
    validation_days: int = 125
    final_test_days: int = 125
    min_train_rows: int = 250
    round_trip_cost_bps: float = 35.0
    signal_buffer_bps: float = 15.0
    feature_version: int = 1
    news_enabled: bool = False

    def __post_init__(self) -> None:
        normalized = tuple(str(s).strip().upper() for s in self.symbols if str(s).strip())
        if not normalized:
            raise ValueError("symbols không được để trống")
        if len(set(normalized)) != len(normalized):
            raise ValueError("symbols không được trùng nhau")
        object.__setattr__(self, "symbols", normalized)
        if self.lookback_days < self.validation_days + self.final_test_days + self.min_train_rows:
            raise ValueError("lookback_days không đủ cho train/validation/final_test")
        if self.validation_days < 5 or self.final_test_days < 5:
            raise ValueError("validation_days và final_test_days phải >= 5")
        if self.min_train_rows < 30:
            raise ValueError("min_train_rows phải >= 30")
        if self.round_trip_cost_bps < 0 or self.signal_buffer_bps < 0:
            raise ValueError("chi phí và signal buffer không được âm")
        if self.feature_version < 1:
            raise ValueError("feature_version phải >= 1")

    @property
    def target_threshold(self) -> float:
        return (self.round_trip_cost_bps + self.signal_buffer_bps) / 10_000.0

    def to_dict(self):
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["target_threshold"] = self.target_threshold
        return payload


__all__ = ["Phase2DatasetConfig"]

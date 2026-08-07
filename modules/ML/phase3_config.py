"""Locked configuration for Phase 3 validation-only classifiers."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Phase3Config:
    random_state: int = 42
    cv_folds: int = 4
    min_train_dates: int = 150
    logistic_c: float = 0.25
    xgb_n_estimators: int = 250
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.03
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    def __post_init__(self) -> None:
        if self.cv_folds < 2:
            raise ValueError("cv_folds phải >= 2")
        if self.min_train_dates < 30:
            raise ValueError("min_train_dates phải >= 30")
        if self.logistic_c <= 0:
            raise ValueError("logistic_c phải > 0")

    def to_dict(self):
        return asdict(self)


__all__ = ["Phase3Config"]

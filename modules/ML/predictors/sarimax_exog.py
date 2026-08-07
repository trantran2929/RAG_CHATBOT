import numpy as np
import pandas as pd
from typing import Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def _fit_is_valid(result) -> bool:
    converged = bool(result.mle_retvals.get("converged", False))
    finite_aic = np.isfinite(float(result.aic))
    finite_params = np.isfinite(np.asarray(result.params, dtype=float)).all()
    return converged and finite_aic and finite_params

def _fit_one(
    y: pd.Series,
    order: Tuple[int, int, int],
    trend: str,
    exog=None
):
    """
    Fit 1 mô hình SARIMAX với order=(p,d,q), trend ('n' hoặc 'c'),
    và exog (có thể None).
    Ưu tiên solver LBFGS, fallback Powell nếu LBFGS fail.
    """
    X = None
    if exog is not None:
        X = np.asarray(exog, dtype="float64")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

    model = SARIMAX(
        endog=y,
        order=order,
        trend=trend,
        exog=X,
        enforce_stationarity=False,
        enforce_invertibility=False,
        concentrate_scale=True,
    )

    errors = []
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                result = model.fit(method=method, maxiter=2000, disp=False)
            if _fit_is_valid(result):
                return result
            else:
                errors.append(f"{method}: not converged")
        except Exception as exc:
            errors.append(f"{method}: {exc}")

    raise RuntimeError(
        f"SARIMAX order={order}, trend={trend} failed: {'; '.join(errors)}")


def arima_select_fit(
    y: pd.Series,
    d: int = 0,
    max_p: int = 3,
    max_q: int = 3,
    trends=("n", "c"),
    exog=None
):
    """
    Thử nhiều cấu hình (p,d,q) với trend khác nhau.
    Chọn mô hình có AIC thấp nhất.
    Nếu tất cả struct fail thì fallback (1,d,0) với trend 'n'.
    """
    best, best_ic, best_order, best_trend = None, np.inf, None, None

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            # loại bỏ (0, d, 0) hoàn toàn phẳng vì quá trivial
            if p == 0 and q == 0:
                continue

            for tr in trends:
                try:
                    res = _fit_one(y, (p, d, q), tr, exog=exog)
                    ic = float(res.aic)
                    if _fit_is_valid(res) and ic < best_ic:
                        best, best_ic = res, ic
                        best_order, best_trend = (p, d, q), tr
                except Exception:
                    continue

    if best is None:
        best = _fit_one(y, (1, d, 0), "n", exog=exog)
        best_order, best_trend = (1, d, 0), "n"

    return best, best_order, best_trend

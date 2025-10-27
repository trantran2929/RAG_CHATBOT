# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime as dt
import pytz
from typing import Optional, List, Tuple, Dict

from modules.api.stock_api import (
    get_close_series,
    get_prices_df,
    get_intraday_df,
    get_time_vn,
)
from modules.api.time_api import get_now
from modules.ML.features import build_news_features
from modules.ML.predictors.sarimax_exog import arima_select_fit
from modules.ML.registry import save_model_meta, load_model_meta
from modules.ML.metrics import rmse as _rmse, mae as _mae

ICT = pytz.timezone("Asia/Ho_Chi_Minh")

VN_HOLIDAYS: set[str] = set()


# ===== Lịch giao dịch / phiên =====
def is_vn_holiday(d: dt.date) -> bool:
    return d.strftime("%Y-%m-%d") in VN_HOLIDAYS

def next_trading_day(d: dt.date) -> dt.date:
    nxt = d + dt.timedelta(days=1)
    while nxt.weekday() >= 5 or is_vn_holiday(nxt):
        nxt += dt.timedelta(days=1)
    return nxt

def _session_status(now: Optional[dt.datetime] = None) -> str:
    now = (now or get_now()).astimezone(ICT)
    if now.weekday() >= 5 or is_vn_holiday(now.date()):
        return "closed"
    t = now.time()
    if t < dt.time(9, 0):   return "pre_open"
    if t < dt.time(11,30):  return "morning"
    if t < dt.time(13, 0):  return "lunch"
    if t < dt.time(15, 0):  return "afternoon"
    return "post_close"

def pick_target_trading_day(now: Optional[dt.datetime] = None) -> dt.date:
    now = (now or get_now()).astimezone(ICT)
    st = _session_status(now)
    if st in ("pre_open","morning","lunch","afternoon"):
        return now.date()
    return next_trading_day(now.date())


# ===== Tiện ích xử lý chuỗi giá & exog =====
def _safe_numeric(df_like):
    if isinstance(df_like, pd.Series):
        out = pd.to_numeric(df_like, errors="coerce")
        out = out.ffill().bfill().fillna(0.0)
        return out.astype("float64")

    if isinstance(df_like, pd.DataFrame):
        out = df_like.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.ffill().bfill().fillna(0.0)
        out = out.astype("float64").infer_objects(copy=False)
        return out

    raise TypeError(f"_safe_numeric expects Series or DataFrame, got {type(df_like)}")

def _to_returns(close_series: pd.Series) -> pd.Series:
    r = np.log(close_series / close_series.shift(1))
    return r.dropna()

def _standardize_df(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    stats = {}
    X2 = X.copy()
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype("float64")
        mu = float(s.mean())
        sd = float(s.std())
        if sd <= 1e-12:
            X2[c] = 0.0
            stats[c] = {"mu": mu, "sd": 1.0}
        else:
            X2[c] = (s - mu) / sd
            stats[c] = {"mu": mu, "sd": sd}
    return X2.astype("float64"), stats

def _apply_scaler(X: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    Xo = pd.DataFrame(index=X.index)
    for c, v in stats.items():
        if c in X.columns:
            s = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype("float64")
            denom = v["sd"] if v["sd"] > 1e-12 else 1.0
            Xo[c] = (s - v["mu"]) / denom
        else:
            Xo[c] = 0.0
    return Xo.astype("float64")

def _add_price_lag_features(symbol: str, X: pd.DataFrame, lags=(1,2,5)) -> pd.DataFrame:
    close = get_close_series(symbol, days=max(500, len(X) + 60))
    if close is None or len(close) < 30:
        return X
    ret = np.log(close / close.shift(1)).dropna()
    ret.index = pd.DatetimeIndex(ret.index).tz_localize(None)

    X2 = X.copy()
    for L in lags:
        feat = ret.shift(1).rolling(L).sum()
        X2[f"ret_lag{L}"] = (
            feat.reindex(X2.index)
                .fillna(0.0)
                .astype("float64")
        )
    return X2

def _align_exog_to_y(symbol: str,
                     y: pd.Series,
                     add_index: Optional[List[str]] = None,
                     shift: int = 1) -> pd.DataFrame:
    if y.empty:
        return pd.DataFrame(index=y.index)

    start_ts = int(pd.Timestamp(y.index[0], tz=ICT).timestamp())
    end_ts   = int(pd.Timestamp(y.index[-1], tz=ICT).timestamp())

    feats = build_news_features(
        symbol,
        start_ts,
        end_ts,
        add_index=add_index or ["VNINDEX","VN30"]
    )
    if feats.empty:
        return pd.DataFrame(index=y.index)

    X = feats.set_index("date")
    X.index = pd.DatetimeIndex(X.index).tz_localize(None)

    X = X.reindex(pd.DatetimeIndex(y.index).tz_localize(None)).fillna(0.0)

    if shift:
        X = X.shift(1).fillna(0.0)

    X = _add_price_lag_features(symbol, X, lags=(1,2,5))

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype("float64")

    return X.astype("float64")

def _build_exog_row_for_forecast(symbol: str,
                                 last_idx: pd.Timestamp,
                                 feat_cols: List[str],
                                 add_index: Optional[List[str]],
                                 scaler: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    end_ts = int(pd.Timestamp(last_idx, tz=ICT).timestamp())
    start_ts = end_ts - 3*24*3600

    feats = build_news_features(
        symbol,
        start_ts,
        end_ts,
        add_index=add_index or ["VNINDEX","VN30"]
    )

    if feats.empty:
        X_raw = pd.DataFrame(index=[pd.Timestamp(last_idx)],
                             columns=feat_cols).fillna(0.0)
    else:
        X_raw = feats.set_index("date")
        X_raw.index = pd.DatetimeIndex(X_raw.index).tz_localize(None)
        X_raw = X_raw.reindex([pd.Timestamp(last_idx)]).fillna(0.0)

    X_raw = _add_price_lag_features(symbol, X_raw, lags=(1,2,5))

    for c in feat_cols:
        if c not in X_raw.columns:
            X_raw[c] = 0.0

    X_raw = X_raw[feat_cols].astype("float64")

    if scaler and len(scaler) > 0:
        return _apply_scaler(X_raw, scaler)[feat_cols]

    return X_raw

def _price_from_ret(base_px: float, ret_mean: float, ret_ci: List[float]) -> dict:
    mean_px = base_px * np.exp(ret_mean)
    lo_px   = base_px * np.exp(ret_ci[0])
    hi_px   = base_px * np.exp(ret_ci[1])
    return {
        "px_mean": float(mean_px),
        "px_lo": float(lo_px),
        "px_hi": float(hi_px),
    }

def _dir_from_gap(last_close: float, band: dict, eps: float = 1e-9):
    px_mean = float(band["px_mean"])
    px_lo   = float(band["px_lo"])
    px_hi   = float(band["px_hi"])
    gap_pct = 100.0 * (px_mean - last_close) / (last_close + eps)

    if px_mean > last_close + eps:
        arrow = "tăng"
    elif px_mean < last_close - eps:
        arrow = "giảm"
    else:
        arrow = "không thay đổi"

    if px_lo > last_close + eps:
        conf = "up_confident"
    elif px_hi < last_close - eps:
        conf = "down_confident"
    else:
        conf = "uncertain"

    return arrow, float(np.round(gap_pct, 3)), conf


# ===== intraday momentum helper =====
def _get_intraday_best(symbol: str, debug: bool = False) -> pd.DataFrame:
    sym = symbol.upper()
    source_candidates = [None, "VCI", "TCBS", "MSN"]
    intervals = ["1m", "5m", "15m"]
    day_opts = [1, 2]

    for src in source_candidates:
        for iv in intervals:
            for d in day_opts:
                try:
                    df = get_intraday_df(sym, source=src, interval=iv, days=d, debug=debug)
                    if df is not None and not df.empty and "close" in df.columns:
                        if debug:
                            print(f"[intraday_best] picked src={src or 'default'} iv={iv} days={d} rows={len(df)}")
                        return df
                except Exception as e:
                    if debug:
                        print(f"[intraday_best] src={src or 'default'} iv={iv} days={d} error: {e}")
                    continue

    if debug:
        print("[intraday_best] no intraday found → returning empty")

    return pd.DataFrame(columns=["open","high","low","close","volume"])


# ====== TRAIN MODEL (SARIMAX) ======
def train_gap_model(symbol: str,
                    lookback_days: int = 365,
                    add_index: Optional[List[str]] = None):
    """
    Huấn luyện SARIMAX dự báo log-return phiên kế tiếp.

    Kết quả trả về LUÔN gồm 3 phần:
    (fit, meta, eval_report)
    """

    sym = symbol.upper()

    # 1. close -> log-return
    close_series = get_close_series(sym, days=lookback_days)
    if close_series is None or len(close_series) < 60:
        raise ValueError("Không đủ dữ liệu close để train.")
    r = _to_returns(close_series)
    if r.empty or len(r) < 30:
        raise ValueError("Không đủ dữ liệu returns để train.")

    # 2. exog (tin tức, sentiment, lags)
    X_raw = _align_exog_to_y(
        sym, r,
        add_index=add_index or ["VNINDEX","VN30"],
        shift=1,
    )

    use_exog = (not X_raw.empty) and bool((np.abs(X_raw.values).sum() > 0))

    if use_exog:
        X_std, scaler = _standardize_df(X_raw)
        fit, order, trend = arima_select_fit(
            r,
            d=0,
            max_p=3,
            max_q=3,
            trends=("n","c"),
            exog=X_std,
        )
        feat_cols = list(X_std.columns)
        X_in_fit = X_std
    else:
        fit, order, trend = arima_select_fit(
            r,
            d=0,
            max_p=3,
            max_q=3,
            trends=("n","c"),
            exog=None,
        )
        feat_cols, scaler = [], {}
        X_in_fit = None

    # 3. đánh giá in-sample
    if use_exog:
        y_pred_in = fit.predict(start=r.index[0], end=r.index[-1], exog=X_in_fit)
    else:
        y_pred_in = fit.predict(start=r.index[0], end=r.index[-1])

    y_true = _safe_numeric(r)
    y_pred_in = _safe_numeric(y_pred_in)

    rmse_val = _rmse(y_true, y_pred_in)
    mae_val  = _mae(y_true, y_pred_in)
    aic_val  = float(fit.aic)

    # 4. one-step-ahead preview
    endog_index = pd.DatetimeIndex(fit.model.data.row_labels).tz_localize(None)
    last_idx = endog_index[-1]

    X_next = None
    if use_exog and feat_cols:
        X_next = _build_exog_row_for_forecast(
            sym,
            last_idx,
            feat_cols,
            add_index or ["VNINDEX","VN30"],
            scaler,
        )

    fc = fit.get_forecast(steps=1, exog=X_next)
    pm = fc.predicted_mean
    if hasattr(pm, "iloc"):
        ret_hat_next = float(pm.iloc[0])
    else:
        ret_hat_next = float(np.asarray(pm).ravel()[0])

    last_close = float(close_series.iloc[-1])
    next_price_est = float(last_close * np.exp(ret_hat_next))

    # 5. meta để lưu
    meta = {
        "symbol": sym,
        "order": list(order),
        "trend": trend,
        "use_exog": use_exog,
        "feature_cols": feat_cols,
        "scaler": scaler,
        "train_len": int(len(r)),
        "timestamp": get_time_vn(),
        "target": "gap_ret",
        "add_index": add_index or ["VNINDEX","VN30"],

        "aic": aic_val,
        "rmse_in_sample": rmse_val,
        "mae_in_sample": mae_val,

        "last_close": last_close,
        "ret_hat_next": ret_hat_next,
        "next_price_est": next_price_est,
    }

    # 6. lưu ra ổ đĩa
    mpath, jpath = save_model_meta(sym, "gap", fit, meta)

    # 7. in debug
    print(f"- Model saved: {mpath}")
    print(f"- Meta saved : {jpath}\n")

    print("--- META ---")
    for k,v in meta.items():
        print(f"{k}: {v}")

    print("\n--- Hiệu suất in-sample ---")
    print(f"RMSE (log-return): {rmse_val:.6f}")
    print(f"MAE  (log-return): {mae_val:.6f}")
    print(f"AIC              : {aic_val}\n")

    print("Close series tail:")
    print(close_series.tail())
    print("")
    print(
        f"Last close: {last_close:,.3f}  "
        f"→ Dự báo next(px): {next_price_est:,.3f} "
        f"(ret_hat={ret_hat_next:+.5f})"
    )

    eval_report = {
        "rmse": rmse_val,
        "mae": mae_val,
        "aic": aic_val,
        "last_close": last_close,
        "ret_hat_next": ret_hat_next,
        "next_price_est": next_price_est,
    }

    # QUAN TRỌNG: luôn trả về 3 giá trị
    return fit, meta, eval_report


# ====== FORECAST GAP (dùng model đã lưu) ======
def forecast_gap(symbol: str, alpha: float = 0.10):
    """
    Load model 'gap' và dự báo log-return tiếp theo + CI.
    Nếu model chưa tồn tại -> tự train rồi chạy lại.
    """
    sym = symbol.upper()

    fit, meta = load_model_meta(sym, "gap")
    if fit is None:
        # train mới, nhận về (fit_trained, meta_trained, eval_report)
        fit_trained, meta_trained, _ = train_gap_model(sym, lookback_days=365)
        # sau train_gap_model đã save xuống ổ đĩa rồi,
        # nên ta load lại để chắc chắn đồng bộ
        fit, meta = load_model_meta(sym, "gap")
        # nếu vẫn None thì coi như lỗi nghiêm trọng
        if fit is None:
            raise RuntimeError("Không thể load model sau khi train.")

    use_exog   = bool(meta.get("use_exog"))
    feat_cols  = meta.get("feature_cols", [])
    scaler     = meta.get("scaler", {})
    add_index  = meta.get("add_index", ["VNINDEX","VN30"])

    X1 = None
    if use_exog and feat_cols:
        endog_index = pd.DatetimeIndex(fit.model.data.row_labels).tz_localize(None)
        last_idx = endog_index[-1]
        X1 = _build_exog_row_for_forecast(
            sym,
            last_idx,
            feat_cols,
            add_index,
            scaler,
        )

    fc = fit.get_forecast(steps=1, exog=X1)
    pm = fc.predicted_mean
    if hasattr(pm, "iloc"):
        mean = float(pm.iloc[0])
    else:
        mean = float(np.asarray(pm).ravel()[0])

    ci_df = fc.conf_int(alpha=alpha)
    arr = np.asarray(ci_df.values).reshape(-1)
    ci_lo, ci_hi = float(arr[0]), float(arr[-1])

    df_px = get_prices_df(sym, days=5)
    last_close = float(df_px["close"].iloc[-1])

    return {
        "symbol": sym,
        "gap_ret_mean": mean,
        "gap_ret_ci": [ci_lo, ci_hi],
        "last_close": last_close,
        "use_exog": use_exog,
    }


# ====== Dự báo full phiên kế tiếp (AM/PM) ======
def _fallback_am_pm():
    return {
        "AM": {"ret_pred": 0.0, "ret_ci": [-0.005, 0.005]},
        "PM": {"ret_pred": 0.0, "ret_ci": [-0.005, 0.005]},
    }

def predict_tomorrow_full_exog(symbol: str, alpha: float = 0.10):
    sym = symbol.upper()
    gap = forecast_gap(sym, alpha=alpha)
    ampm = _fallback_am_pm()

    open_band = _price_from_ret(
        gap["last_close"],
        gap["gap_ret_mean"],
        gap["gap_ret_ci"],
    )
    am_band   = _price_from_ret(
        open_band["px_mean"],
        ampm["AM"]["ret_pred"],
        ampm["AM"]["ret_ci"],
    )
    pm_band   = _price_from_ret(
        am_band["px_mean"],
        ampm["PM"]["ret_pred"],
        ampm["PM"]["ret_ci"],
    )

    last_close = float(gap["last_close"])
    open_dir, open_gap_pct, open_conf = _dir_from_gap(last_close, open_band)

    target = pick_target_trading_day()

    return {
        "target_day": target,
        "gap": gap,
        "ampm": ampm,
        "bands": {"OPEN_am": open_band, "AM_px": am_band, "PM_px": pm_band},
        "open_direction": open_dir,
        "open_gap_pct": open_gap_pct,
        "open_confidence": open_conf,
        "timestamp": get_time_vn(),
        "mode": "out_of_session"
    }


# ====== Dự báo trong phiên vs trước phiên ======
def direction_from_return(x: float, eps: float = 1e-6) -> str:
    return "tăng" if x > eps else "giảm" if x < -eps else "không thay đổi"

def _next_trading_session(now: Optional[dt.datetime] = None) -> Tuple[dt.date, str]:
    now = (now or get_now()).astimezone(ICT)
    st = _session_status(now)
    if st == "pre_open":               return now.date(), "AM"
    if st in ("morning","lunch"):      return now.date(), "PM"
    if st == "afternoon":              return next_trading_day(now.date()), "AM"
    if st in ("post_close","closed"):  return next_trading_day(now.date()), "AM"
    return next_trading_day(now.date()), "AM"

def _confidence_from_mu_sigma(mu: float, sigma: float) -> str:
    ratio = abs(mu) / (sigma + 1e-9)
    if ratio >= 1.0: return "high"
    if ratio >= 0.5: return "medium"
    return "low"

def predict_next_step_in_session(symbol: str, source: str = "VCI"):
    sym = symbol.upper()
    now = get_now().astimezone(ICT)
    st = _session_status(now)

    if st not in ("morning","afternoon"):
        return {
            "error": "Phiên chưa mở, hãy dùng predict_next_session().",
            "next_step_dir": None,
            "path_pred": pd.Series(dtype=float),
            "mode": "in_session"
        }

    intraday = _get_intraday_best(sym)
    if intraday is not None and not intraday.empty and "close" in intraday.columns:
        now_naive = now.replace(tzinfo=None)
        intraday = intraday[intraday.index <= now_naive]

        close = intraday["close"].astype("float64").dropna()
        if len(close) >= 6:
            ret = np.log(close/close.shift(1)).dropna()
            N = min(5, len(ret))
            mu = float(ret.tail(N).mean())
            sig = float(ret.tail(N).std() or 0.0)

            last_px = float(close.iloc[-1])
            step1 = last_px * float(np.exp(mu))
            path = [step1]
            for _ in range(2):
                path.append(path[-1] * float(np.exp(mu)))
            series_pred = pd.Series(path, index=pd.RangeIndex(1, 1+len(path), name="t+step"))

            direction = direction_from_return(mu)
            step_conf = _confidence_from_mu_sigma(mu, sig)

            return {
                "error": None,
                "session": "AM" if st == "morning" else "PM",
                "next_step_dir": direction,
                "ret_mean": mu,
                "ret_std": sig,
                "step_confidence": step_conf,
                "last_px": last_px,
                "path_pred": series_pred,
                "source_used": "intraday",
                "mode": "in_session"
            }

    df_daily = get_prices_df(sym, days=6)
    cls = df_daily["close"].dropna()
    if len(cls) >= 3:
        r = np.log(cls/cls.shift(1)).dropna()
        mu = float(r.tail(3).mean())
        sig = float(r.tail(3).std() or 0.0)
        last_px = float(cls.iloc[-1])
        step1 = last_px * float(np.exp(mu))
        path = [step1, step1 * float(np.exp(mu)), step1 * float(np.exp(2*mu))]
        series_pred = pd.Series(path, index=pd.RangeIndex(1, 4, name="t+step"))
        direction = direction_from_return(mu)
        step_conf = "low"  # fallback

        return {
            "error": "Fallback daily (thiếu intraday).",
            "session": "AM" if st == "morning" else "PM",
            "next_step_dir": direction,
            "ret_mean": mu,
            "ret_std": sig,
            "step_confidence": step_conf,
            "last_px": last_px,
            "path_pred": series_pred,
            "source_used": "daily_fallback",
            "mode": "in_session"
        }

    return {
        "error": "Không đủ dữ liệu cả intraday lẫn daily.",
        "next_step_dir": None,
        "path_pred": pd.Series(dtype=float),
        "mode": "in_session"
    }

def predict_next_session(symbol: str, alpha: float = 0.10, source: str = "VCI"):
    sym = symbol.upper()
    now = get_now().astimezone(ICT)
    target_day, next_sess = _next_trading_session(now)

    if next_sess == "AM":
        pack = predict_tomorrow_full_exog(sym, alpha=alpha)
        open_band = pack["bands"]["OPEN_am"]
        last_close = float(pack["gap"]["last_close"])
        open_dir, open_gap_pct, open_conf = _dir_from_gap(last_close, open_band)
        return {
            "mode": "next_session",
            "next_session": "AM",
            "target_day": target_day,
            "open_band": open_band,
            "gap": pack["gap"],
            "open_direction": open_dir,
            "open_gap_pct": open_gap_pct,
            "open_confidence": open_conf,
            "timestamp": get_time_vn(),
            "note": "AM dùng band OPEN (ước lượng khoảng mở cửa)."
        }

    intraday = get_intraday_df(sym, source=source, interval="5m", days=1)
    base_px = None
    if intraday is not None and not intraday.empty and "close" in intraday.columns:
        am_part = intraday[intraday.index.time <= dt.time(11, 30)]
        if am_part is not None and not am_part.empty:
            base_px = float(am_part["close"].dropna().iloc[-1])
    if base_px is None:
        df = get_prices_df(sym, days=5)
        base_px = float(df["close"].dropna().iloc[-1])

    ampm = {"PM": {"ret_pred": 0.0, "ret_ci": [-0.005, 0.005]}}
    pm_band = _price_from_ret(base_px, ampm["PM"]["ret_pred"], ampm["PM"]["ret_ci"])
    pm_dir, pm_gap_pct, pm_conf = _dir_from_gap(base_px, pm_band)

    return {
        "mode": "next_session",
        "next_session": "PM",
        "target_day": target_day,
        "pm_band": pm_band,
        "pm_direction": pm_dir,
        "pm_gap_pct": pm_gap_pct,
        "pm_confidence": pm_conf,
        "base_from": "AM_close" if (intraday is not None and not intraday.empty) else "last_close_daily",
        "timestamp": get_time_vn(),
        "note": "PM dựa trên giá kết thúc buổi sáng và band PM mặc định."
    }

def smart_predict(symbol: str, alpha: float = 0.10, source: str = "VCI"):
    now = get_now().astimezone(ICT)
    st = _session_status(now)

    if st in ("morning","afternoon"):
        step = predict_next_step_in_session(symbol, source=source)
        return {
            "mode": "in_session",
            "session": "AM" if st=="morning" else "PM",
            "symbol": symbol.upper(),
            "timestamp": get_time_vn(),
            **step
        }

    nxt = predict_next_session(symbol, alpha=alpha, source=source)
    return {
        "mode": "out_of_session",
        "symbol": symbol.upper(),
        "timestamp": get_time_vn(),
        **nxt
    }


__all__ = [
    "train_gap_model","forecast_gap",
    "predict_tomorrow_full_exog","smart_predict",
    "predict_next_session","predict_next_step_in_session",
    "direction_from_return","pick_target_trading_day"
]

"""
VNStock Data API (ICT timezone, TTL cache, robust normalize)
- discover_market_indices, get_index_detail
- get_stock_quote, get_top_stocks
- get_history_prices
- get_history_df_vnstock, get_prices_df, get_close_series
- get_intraday_df  (ticks → OHLCV)
- format_* helpers (text output)
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="vnai.scope.profile")

from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Iterable, List, Dict
import time, re

import numpy as np
import pandas as pd
import pytz

from vnstock import Quote, Trading, Screener
from modules.api.time_api import get_now  # tz-aware ICT

# ========= Time config (ICT)
VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")
DATE_FMT = "%d-%m-%Y"
DATETIME_FMT = "%d-%m-%Y %H:%M:%S"
MIN_YEAR = 2005

def get_time_vn() -> str:
    return get_now().strftime(DATETIME_FMT)

def _today_vn() -> datetime:
    return get_now()  # tz-aware ICT

def _ymd(dt_: datetime) -> str:
    if dt_.tzinfo:
        dt_ = dt_.astimezone(VN_TZ)
    return dt_.strftime("%Y-%m-%d")

# ========= TTL Cache decorator
def TTLCache(ttl_seconds: int = 300, verbose: bool = False):
    def decorator(func):
        cache: Dict[tuple, tuple] = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (func.__name__, args, frozenset(kwargs.items()))
            t = time.time()
            if key in cache:
                val, ts = cache[key]
                if t - ts < ttl_seconds:
                    if verbose:
                        print(f"[Cache Hit] {func.__name__}")
                    return val
                cache.pop(key, None)
            if verbose:
                print(f"[Cache Miss] {func.__name__}")
            val = func(*args, **kwargs)
            cache[key] = (val, t)
            return val
        return wrapper
    return decorator

# ========= Symbol helpers
def _sanitize_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    return re.sub(r"[^A-Z0-9]", "", s)

def _validate_symbol(symbol: str):
    if not (3 <= len(symbol) <= 10):
        raise ValueError(f"Symbol không hợp lệ sau sanitize: {repr(symbol)}")

# ========= Daily normalize
def _normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    time_col = next((c for c in ("time","date","tradingdate") if c in df.columns), None)

    if time_col:
        s = df[time_col]
        if np.issubdtype(getattr(s, "dtype", object), np.number):
            s_num = pd.to_numeric(s, errors="coerce")
            unit = "ms" if (s_num.max() or 0) > 1e12 else "s"
            idx = (
                pd.to_datetime(s_num, unit=unit, utc=True)
                .dt.tz_convert(VN_TZ).dt.tz_localize(None)
            )
        else:
            idx = pd.to_datetime(s, errors="coerce")
        df = (
            df.assign(_dt=idx)
              .dropna(subset=["_dt"])
              .drop(columns=[time_col])
              .set_index("_dt")
        )
    else:
        idx = pd.to_datetime(df.index, errors="coerce")
        df.index = idx
        df = df[~df.index.isna()]

    for col in ("open","high","low","close","volume"):
        if col not in df.columns:
            alt = next((k for k in df.columns if str(k).lower().startswith(col)), None)
            df[col] = pd.to_numeric(df.get(alt, np.nan), errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    today_ict = _today_vn().date()
    df = df[(df.index.year >= MIN_YEAR) & (df.index.date <= today_ict)]
    df = df.sort_index().dropna(subset=["close"])
    return df[["open","high","low","close","volume"]]

# ========= Quote history fallback
def _quote_history_with_fallback(symbol: str, start: str, end: str) -> pd.DataFrame:
    for src in ("VCI","TCBS","MSN"):
        try:
            q = Quote(symbol=symbol, source=src)
            raw = q.history(start=start, end=end)
            dfn = _normalize_history_df(raw)
            if dfn is not None and not dfn.empty:
                return dfn
        except Exception:
            continue
    raise ValueError(f"Không lấy được lịch sử cho {symbol} từ VCI/TCBS/MSN")

# ========= Indices
_CANDIDATE_INDICES: List[str] = [
    "VNINDEX","VN30","VN100","VNALL","VNMID","VNSML",
    "HNX","HNX30","UPCOM","UPCOM10",
    "VNFINLEAD","VNFINSELECT","VNDIAMOND","VN30TR",
]

@TTLCache(ttl_seconds=6*3600)
def discover_market_indices(
    candidates: Iterable[str] | None = None,
    min_points: int = 1
) -> List[str]:
    idxs = list(candidates or _CANDIDATE_INDICES)
    ok: List[str] = []
    end = _today_vn(); start = end - timedelta(days=7)
    for code in idxs:
        try:
            df = _quote_history_with_fallback(code, _ymd(start), _ymd(end))
            if df is not None and not df.empty and len(df) >= min_points:
                ok.append(code)
        except Exception:
            pass
    prio = {c:i for i,c in enumerate(_CANDIDATE_INDICES)}
    ok.sort(key=lambda x: prio.get(x, 999))
    return ok

# ========= Index detail
@TTLCache(ttl_seconds=300)
def get_index_detail(index_code: str = "VNINDEX") -> dict:
    try:
        index_code = _sanitize_symbol(index_code); _validate_symbol(index_code)
        end = _today_vn(); start = end - timedelta(days=7)
        df = _quote_history_with_fallback(index_code, _ymd(start), _ymd(end))
        if df is None or df.empty:
            raise ValueError("Không có dữ liệu chỉ số.")
        last = df.iloc[-1]; prev = df.iloc[-2] if len(df)>1 else last
        change = float(last["close"]) - float(prev["close"])
        pct = (change/float(prev["close"])*100) if float(prev["close"]) else 0.0
        return {
            "ticker": index_code.upper(),
            "price": round(float(last["close"]), 2),
            "change": round(change, 2),
            "percent_change": round(pct, 2),
            "open": round(float(last.get("open", 0) or 0), 2),
            "high": round(float(last.get("high", 0) or 0), 2),
            "low":  round(float(last.get("low", 0) or 0), 2),
            "volume": int(last.get("volume", 0) or 0),
            "timestamp": get_time_vn(),
            "source": "VNStock",
        }
    except Exception as e:
        print(f"[VNStock] Lỗi chỉ số {index_code}: {e}")
        return {"ticker": index_code, "error": str(e)}

# ========= Stock quote (snapshot + fallback)
@TTLCache(ttl_seconds=60)
def get_stock_quote(symbol: str) -> dict:
    """
    Trả về dict thông tin giá cho 1 mã.
    Có cờ fallback=True nếu chỉ lấy được giá đóng cửa gần nhất.
    """
    symbol = _sanitize_symbol(symbol)
    if not symbol:
        return {"symbol": symbol, "error": "Symbol rỗng."}
    try:
        _validate_symbol(symbol)
    except ValueError as ve:
        return {"symbol": symbol, "error": str(ve)}

    def _get_recent_market_date(sym: str) -> Optional[str]:
        try:
            end_ = _today_vn(); start_ = end_ - timedelta(days=7)
            hist_df = _quote_history_with_fallback(sym, _ymd(start_), _ymd(end_))
            if hist_df is None or hist_df.empty:
                return None
            last_idx = hist_df.index[-1]
            if isinstance(last_idx, (pd.Timestamp, datetime)):
                return last_idx.strftime(DATE_FMT)
            try:
                return datetime.strptime(str(last_idx)[:10], "%Y-%m-%d").strftime(DATE_FMT)
            except Exception:
                return str(last_idx)
        except Exception:
            return None

    # 1. realtime board
    try:
        trading = Trading()
        df = trading.price_board(symbols_list=[symbol])
        if df is None or df.empty:
            raise ValueError("Không có dữ liệu cổ phiếu.")
        df.columns = [f"{a}_{b}" for a,b in df.columns]
        row = df.iloc[0].to_dict()

        price = row.get("match_match_price")
        ref   = row.get("match_reference_price") or row.get("listing_ref_price")
        ceil  = row.get("match_ceiling_price")   or row.get("listing_ceiling")
        floor = row.get("match_floor_price")     or row.get("listing_floor")
        vol   = (
            row.get("match_accumulated_volume")
            or row.get("match_match_vol")
            or 0
        )

        if not price or price == 0:
            raise ValueError("match_price = 0")

        chg = float(price) - float(ref) if ref else 0.0
        pct = (chg/float(ref)*100) if ref else 0.0

        market_date_str = _get_recent_market_date(symbol)

        return {
            "symbol": symbol,
            "price": float(price),
            "open": float(ref) if ref is not None else None,
            "high": float(ceil) if ceil is not None else None,
            "low":  float(floor) if floor is not None else None,
            "change": round(chg,2),
            "percent_change": round(pct,2),
            "volume": int(vol or 0),
            "timestamp": get_time_vn(),   # lúc query
            "market_date": market_date_str,  # ngày giao dịch thực sự
            "source": "VNStock.Trading",
            "fallback": False,
        }

    except Exception as e:
        print(f"[VNStock] Lỗi realtime {symbol}: {e}")

        # 2. Fallback dùng lịch sử
        try:
            end = _today_vn(); start = end - timedelta(days=7)
            df = _quote_history_with_fallback(symbol, _ymd(start), _ymd(end))
            if df is not None and not df.empty:
                last = df.iloc[-1]
                op = float(last.get("open",0) or 0)
                cl = float(last.get("close",0) or 0)
                chg = cl - op
                pct = (chg/op*100) if op else 0.0
                try:
                    idx_val = str(last.name)
                    fb_date = datetime.strptime(
                        idx_val[:10], "%Y-%m-%d"
                    ).strftime(DATE_FMT)
                except Exception:
                    fb_date = "phiên gần nhất"
                return {
                    "symbol": symbol,
                    "price": cl or None,
                    "open": op or None,
                    "high": float(last.get("high",0) or 0),
                    "low":  float(last.get("low",0) or 0),
                    "change": round(chg,2),
                    "percent_change": round(pct,2),
                    "volume": int(last.get("volume",0) or 0),
                    "timestamp": get_time_vn(),
                    "fallback": True,
                    "fallback_date": fb_date,
                    "market_date": fb_date,
                    "source": "VNStock.Quote(Fallback)",
                }
        except Exception as ex:
            print(f"[VNStock] Fallback Quote lỗi {symbol}: {ex}")

        return {"symbol": symbol, "price": None, "error": "Không có dữ liệu giao dịch gần đây."}

# ========= Top movers
@TTLCache(ttl_seconds=300)
def get_top_stocks(limit: int = 10, direction: str = "up") -> list:
    """
    direction: "up" lấy top tăng %, "down" lấy top giảm % (âm nhiều nhất).
    """
    try:
        screener_df = Screener().stock(params={"exchangeName":"HOSE,HNX,UPCOM"}, limit=3000)
        if screener_df is None or screener_df.empty:
            raise ValueError("Không có dữ liệu sàng lọc.")
        growth_col = next((c for c in screener_df.columns if "growth" in c.lower()), None)
        if not growth_col:
            raise ValueError("Không tìm thấy cột tăng trưởng giá.")
        screener_df[growth_col] = pd.to_numeric(
            screener_df[growth_col],
            errors="coerce"
        ).astype(float)

        ascending = True if direction == "down" else False
        top_df = screener_df.sort_values(growth_col, ascending=ascending).head(limit)

        out = []
        for _, row in top_df.iterrows():
            out.append({
                "symbol": (row.get("ticker") or row.get("symbol") or "").upper(),
                "exchange": row.get("exchange","") or "",
                "price": row.get("price_near_realtime") or row.get("close_price"),
                "pct_change": round(float(row[growth_col] or 0.0), 2),
                "volume": int(row.get("avg_trading_value_10d", 0) or 0),
                "timestamp": get_time_vn(),
            })
        return out
    except Exception as e:
        print(f"[VNStock] Lỗi top movers: {e}")
        return []

def format_top_stocks(direction: str = "up", limit: int = 5) -> str:
    arr = get_top_stocks(limit=limit, direction=direction)
    if not arr:
        label = "Tăng" if direction == "up" else "Giảm"
        return f"🚫 Top {label} tạm thời không có dữ liệu."

    header_emoji = "🚀 Top tăng mạnh" if direction == "up" else "⚠️ Top giảm mạnh"
    lines = [header_emoji + ":"]
    for item in arr:
        sym = item.get("symbol","?")
        px = item.get("price", None)
        pct = item.get("pct_change", 0.0)
        vol = item.get("volume", 0)
        lines.append(
            f"- {sym}: {px} ({pct:+.2f}%), GTGD~{vol:,}"
        )
    return "\n".join(lines)

# ========= History (dict)
@TTLCache(ttl_seconds=300)
def get_history_prices(symbol: str, days: int = 7) -> dict:
    symbol = _sanitize_symbol(symbol)
    try:
        _validate_symbol(symbol)
    except ValueError as ve:
        return {"symbol": symbol, "error": str(ve)}

    try:
        end = _today_vn(); start = end - timedelta(days=days+14)
        df = _quote_history_with_fallback(symbol, _ymd(start), _ymd(end))
        if df is None or df.empty:
            raise ValueError("Không có dữ liệu lịch sử.")

        df["change"] = df["close"].diff()
        df["pct_change"] = df["close"].pct_change() * 100
        df = df.dropna()

        recs = []
        tail = df.tail(days) if len(df)>days else df
        for idx,row in tail.iterrows():
            if isinstance(idx,(pd.Timestamp,datetime)):
                trade_date = idx.strftime(DATE_FMT)
            else:
                try:
                    trade_date = datetime.strptime(str(idx)[:10], "%Y-%m-%d").strftime(DATE_FMT)
                except Exception:
                    trade_date = str(idx)
            recs.append({
                "date": trade_date,
                "open": float(row.get("open",0) or 0),
                "high": float(row.get("high",0) or 0),
                "low":  float(row.get("low",0) or 0),
                "close":float(row.get("close",0) or 0),
                "volume": int(row.get("volume",0) or 0),
                "change": round(float(row.get("change",0) or 0),2),
                "pct_change": round(float(row.get("pct_change",0) or 0),2),
            })
        return {
            "symbol": symbol,
            "history": recs,
            "days": days,
            "timestamp": get_time_vn(),
            "source": "VNStock.Quote(FallbackOK)",
        }
    except Exception as e:
        print(f"[VNStock] Lỗi lịch sử {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}

# --- tz helper
def _to_vn_aware(dt_: datetime) -> datetime:
    if dt_ is None:
        return _today_vn()
    if dt_.tzinfo is None:
        return VN_TZ.localize(dt_)
    return dt_.astimezone(VN_TZ)

@TTLCache(ttl_seconds=300)
def get_history_df_vnstock(symbol: str,
                            start: Optional[str] = None,
                            end: Optional[str] = None) -> pd.DataFrame:
    symbol = _sanitize_symbol(symbol); _validate_symbol(symbol)

    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else _today_vn()
    end_dt = _to_vn_aware(end_dt)
    now_ict = _today_vn()
    end_dt = min(end_dt, now_ict)

    start_dt = datetime.strptime(start, "%Y-%m-%d") if start else (end_dt - timedelta(days=370))
    start_dt = _to_vn_aware(start_dt)

    df = _quote_history_with_fallback(symbol, _ymd(start_dt), _ymd(end_dt))
    df = _normalize_history_df(df)
    if df.empty:
        raise ValueError(f"VNStock: không có dữ liệu hợp lệ cho {symbol} (normalize)")
    df = df.asfreq("D").ffill()
    return df[["open","high","low","close","volume"]]

@TTLCache(ttl_seconds=300)
def get_prices_df(symbol: str, days: int = 365) -> pd.DataFrame:
    end = _today_vn()
    start = end - timedelta(days=days + 14)
    df = get_history_df_vnstock(symbol, start=_ymd(start), end=_ymd(end))
    today_ict = _today_vn().date()
    df = df[df.index.date <= today_ict]
    return df.tail(days).asfreq("D").ffill()[["open","high","low","close","volume"]]

@TTLCache(ttl_seconds=300)
def get_close_series(symbol: str, days: int = 365) -> pd.Series:
    df = get_prices_df(symbol, days=days)
    s = df["close"].astype("float64")
    s = s[s > 0].dropna()
    return s

# ========= Intraday helpers
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

def _to_ict_naive_index(s: pd.Series | pd.Index) -> pd.DatetimeIndex:
    ser = pd.Series(s) if isinstance(s, pd.Index) else s

    if is_numeric_dtype(ser):
        s_num = pd.to_numeric(ser, errors="coerce")
        valid = s_num.notna()
        if not valid.any():
            return pd.DatetimeIndex([], dtype="datetime64[ns]")
        unit = "ms" if s_num[valid].max() > 1e12 else "s"
        di = pd.to_datetime(s_num, unit=unit, utc=True)
        return pd.DatetimeIndex(di).tz_convert(VN_TZ).tz_localize(None)

    if is_datetime64_any_dtype(ser):
        di = pd.DatetimeIndex(pd.to_datetime(ser, errors="coerce"))
        if di.tz is not None:
            di = di.tz_convert(VN_TZ).tz_localize(None)
        return di

    di = pd.DatetimeIndex(pd.to_datetime(ser, errors="coerce", utc=True))
    di = di.tz_convert(VN_TZ).tz_localize(None)
    return di

def _to_min_rule(interval: str) -> str:
    iv = (interval or "").strip().lower()
    if iv.endswith("min"):
        return iv
    if iv.endswith("m"):
        return iv[:-1] + "min"
    return "5min"

def _normalize_ticks_df(ticks: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    if ticks is None or ticks.empty:
        return pd.DataFrame(columns=["price", "volume"])

    df = ticks.copy()
    df.columns = [str(c).lower() for c in df.columns]

    time_col = next((c for c in ("time", "datetime", "timestamp") if c in df.columns), None)
    idx_raw = df[time_col] if time_col else df.index
    idx = _to_ict_naive_index(idx_raw)
    mask = ~pd.isna(idx)
    if not mask.any():
        if debug:
            print("[ticks] all NaT after time parse.")
        return pd.DataFrame(columns=["price", "volume"])

    df = df.loc[mask].copy()
    df.index = idx[mask]

    price_col = next(
        (c for c in ("price","match_price","last_price","matched_price","p")
         if c in df.columns),
        None
    )
    vol_col   = next(
        (c for c in (
            "volume","vol","match_volume","matched_volume",
            "accumulated_volume","qtty","quantity","volumn","v"
        ) if c in df.columns),
        None
    )

    out = pd.DataFrame(index=df.index)
    out["price"]  = pd.to_numeric(df.get(price_col, np.nan), errors="coerce")
    out["volume"] = pd.to_numeric(df.get(vol_col, 0), errors="coerce").fillna(0.0)

    out = out.dropna(subset=["price"], how="all")
    return out.sort_index()

def _normalize_intraday_ohlc_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or getattr(raw, "empty", True):
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    df = raw.copy()
    df.columns = [str(c).lower() for c in df.columns]

    time_col = next((c for c in ("time","datetime","timestamp","tradingtime","date","_dt")
                     if c in df.columns), None)
    idx = _to_ict_naive_index(df[time_col]) if time_col else _to_ict_naive_index(df.index)

    col_map = {
        "open":   ["open","o","match_open","first_price"],
        "high":   ["high","h","match_high","highest"],
        "low":    ["low","l","match_low","lowest"],
        "close":  ["close","c","match_price","last_price","price"],
        "volume": ["volume","vol","match_volume","accumulated_volume","qtty","quantity"],
    }
    out = {}
    for std, cands in col_map.items():
        found = next((c for c in cands if c in df.columns), None)
        out[std] = pd.to_numeric(df.get(found, np.nan), errors="coerce")

    res = pd.DataFrame(out, index=idx).sort_index()
    res = res.dropna(subset=["close"], how="all")
    return res.astype({
        "open":"float64","high":"float64","low":"float64",
        "close":"float64","volume":"float64"
    })

def _resample_ticks_to_ohlcv(
    ticks: pd.DataFrame,
    interval: str = "5m",
    debug: bool = False
) -> pd.DataFrame:
    if ticks is None or ticks.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    rule = _to_min_rule(interval)
    df = ticks.sort_index()

    try:
        ohlc = df["price"].resample(rule).ohlc()
        try:
            vol = df["volume"].resample(rule).sum(min_count=1)
        except TypeError:
            vol = df["volume"].resample(rule).sum()
        out = pd.concat([ohlc, vol.rename("volume")], axis=1)
        out = out.dropna(subset=["close"], how="all")
        if not out.empty:
            return out.astype({
                "open":"float64","high":"float64","low":"float64",
                "close":"float64","volume":"float64"
            })
    except Exception:
        pass

    floored = df.index.floor(rule)
    gb = df.groupby(floored)
    out2 = pd.DataFrame({
        "open":  gb["price"].first(),
        "high":  gb["price"].max(),
        "low":   gb["price"].min(),
        "close": gb["price"].last(),
        "volume": gb["volume"].sum(min_count=1)
                  if hasattr(gb["volume"], "sum")
                  else gb["volume"].sum(),
    }).dropna(subset=["close"], how="all").sort_index()
    return out2.astype({
        "open":"float64","high":"float64","low":"float64",
        "close":"float64","volume":"float64"
    })

def _finalize_intraday(
    raw: pd.DataFrame,
    interval: str,
    days: int,
    debug: bool = False
) -> pd.DataFrame:
    """
    Nhận ticks hoặc OHLC-like → chuẩn hoá về OHLCV + cắt theo days.
    """
    if raw is None or getattr(raw, "empty", True):
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    cols = {c.lower() for c in raw.columns}
    is_ohlc_like = len({"open","high","low","close"} & cols) >= 2

    if is_ohlc_like:
        df = _normalize_intraday_ohlc_df(raw)
    else:
        ticks_norm = _normalize_ticks_df(raw, debug=debug)
        df = _resample_ticks_to_ohlcv(ticks_norm, interval=interval, debug=debug)

    if not df.empty and days:
        cutoff = _today_vn().date() - timedelta(days=max(0, int(days) - 1))
        df = df[df.index.date >= cutoff]

    return df

@TTLCache(ttl_seconds=60)
def get_intraday_df(
    symbol: str,
    interval: str = "5m",
    days: int = 1,
    source: str | None = None,
    debug: bool = False
) -> pd.DataFrame:
    """
    Lấy intraday OHLCV:
      - Ưu tiên: Quote(symbol=SYM).intraday()
      - sau đó Quote().intraday(symbol=SYM)
      - thử các source VCI, TCBS, MSN
      - fallback Trading/Data
    """
    sym = _sanitize_symbol(symbol)
    try:
        _validate_symbol(sym)
    except ValueError:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    # 1) Quote(symbol=...).intraday()
    try:
        q = Quote(symbol=sym) if source is None else Quote(symbol=sym, source=source)
        raw = q.intraday()
        df = _finalize_intraday(raw, interval, days, debug=debug)
        if not df.empty:
            return df
    except Exception:
        pass

    # 2) Quote().intraday(symbol=SYM)
    try:
        q2 = Quote() if source is None else Quote(source=source)
        raw2 = q2.intraday(symbol=sym)
        df2 = _finalize_intraday(raw2, interval, days, debug=debug)
        if not df2.empty:
            return df2
    except Exception:
        pass

    # 3) thử nguồn cụ thể
    for src in ("VCI","TCBS","MSN"):
        try:
            qx = Quote(symbol=sym, source=src)
            rawx = qx.intraday()
            dfx = _finalize_intraday(rawx, interval, days, debug=debug)
            if not dfx.empty:
                return dfx
        except Exception:
            continue

    # 4) Trading().intraday
    try:
        tr = Trading()
        if hasattr(tr, "intraday"):
            raw_t = tr.intraday(symbols_list=[sym], interval=interval, days=days)
            df_t = _finalize_intraday(raw_t, interval, days, debug=debug)
            if not df_t.empty:
                return df_t
    except Exception:
        pass

    # 5) Data().intraday
    try:
        from vnstock import Data
        d = Data()
        if hasattr(d, "intraday"):
            raw_d = d.intraday(symbol=sym, interval=interval, days=days)
            df_d = _finalize_intraday(raw_d, interval, days, debug=debug)
            if not df_d.empty:
                return df_d
    except Exception:
        pass

    return pd.DataFrame(columns=["open","high","low","close","volume"])

# ========= Format helpers =========

def _same_trading_day(market_date_str: str) -> bool:
    """
    Kiểm tra xem market_date ('24-10-2025') có cùng ngày ICT hiện tại không.
    """
    if not market_date_str:
        return False
    now = _today_vn()
    today_str = now.strftime(DATE_FMT)
    return market_date_str == today_str

def format_stock_info(symbol: str) -> str:
    """
    Chuỗi mô tả giá mã cổ phiếu:
    - Nếu fallback: dùng "Giá đóng cửa phiên X".
    - Nếu không fallback:
        + Nếu market_date == hôm nay -> "Giá hiện tại".
        + Ngược lại -> "Giá cuối phiên <market_date> ... (thị trường đã đóng)".
    """
    data = get_stock_quote(symbol)
    if "error" in data or not data.get("price"):
        return f"⚠️ Không thể lấy dữ liệu cho mã **{symbol.upper()}**."

    price = data["price"]
    market_date = data.get("market_date")
    ts_query = data.get("timestamp")

    if data.get("fallback", False):
        headline = (
            f"📊 **{data['symbol']}** — Giá đóng cửa phiên "
            f"{data.get('fallback_date','gần nhất')}: {price:,} VNĐ"
        )
        note = " (dữ liệu lấy từ lịch sử)"
    else:
        if _same_trading_day(market_date):
            headline = (
                f"📊 **{data['symbol']}** — Giá hiện tại: {price:,} VNĐ"
            )
            note = ""
        else:
            headline = (
                f"📊 **{data['symbol']}** — Giá cuối phiên {market_date}: "
                f"{price:,} VNĐ"
            )
            note = " (thị trường đã đóng)"

    body = (
        f"{headline}  \n"
        f"• Mở cửa: {data.get('open')} | Cao nhất: {data.get('high')} | Thấp nhất: {data.get('low')}  \n"
        f"• Thay đổi: {data.get('change',0):+.2f} ({data.get('percent_change',0):+.2f}%)  \n"
        f"• Khối lượng: {data.get('volume',0):,}  \n"
        f"🕒 Truy vấn lúc: {ts_query}{note}"
    )
    return body

@TTLCache(ttl_seconds=300)
def format_market_summary(indices: list[str] = None) -> str:
    """
    Tóm tắt thị trường:
    - VNINDEX, VN30, HNX, UPCOM
    - Top tăng / top giảm (3 mã mỗi nhóm)
    - Gắn timestamp
    """
    try:
        if indices is None:
            indices = ["VNINDEX", "VN30", "HNX", "UPCOM"]

        summaries = []
        for idx in indices:
            data = get_index_detail(idx)
            if "error" in data:
                continue
            emoji = "📈" if data["change"] > 0 else "📉" if data["change"] < 0 else "⏸️"
            summaries.append(
                f"{emoji} **{data['ticker']}**: {data['price']:,} điểm "
                f"({data['change']:+.2f}, {data['percent_change']:+.2f}%)"
            )

        if not summaries:
            return "⚠️ Không thể lấy dữ liệu thị trường."

        return (
            f"📊 **TỔNG QUAN THỊ TRƯỜNG VIỆT NAM**  \n" +
            "  \n".join(summaries) + "  \n\n" +
            f"{format_top_stocks('up', 3)}  \n\n" +
            f"{format_top_stocks('down', 3)}  \n" +
            f"🕒 Cập nhật: {get_time_vn()}"
        )
    except Exception as e:
        print(f"[VNStock] Lỗi tóm tắt thị trường: {e}")
        return "⚠️ Không thể lấy dữ liệu thị trường."

def format_history_text(symbol: str, days: int, data: dict) -> str:
    """
    Chuỗi mô tả lịch sử price N ngày.
    """
    if not data or "error" in data:
        return f"📜 {symbol.upper()} — lịch sử {days} ngày: chưa khả dụng."
    hist = data.get("history") or []
    if not hist:
        return f"📜 {symbol.upper()} — lịch sử {days} ngày: trống."

    try:
        first, last = hist[0], hist[-1]
        c0 = float(first.get("close", 0) or 0)
        c1 = float(last.get("close", 0) or 0)
        pct = ((c1 - c0) / c0 * 100.0) if c0 else 0.0
        return (
            f"📜 **{symbol.upper()} — lịch sử {days} ngày**\n"
            f"• Từ {first.get('date')} → {last.get('date')}\n"
            f"• Đóng cửa: {c0:,.2f} → {c1:,.2f} (**{pct:+.2f}%**)\n"
            f"• Bản đầy đủ gồm {len(hist)} phiên (OHLC, khối lượng)."
        )
    except Exception:
        return (
            f"📜 {symbol.upper()} — lịch sử {days} ngày:"
            f" có {len(hist)} phiên."
        )


__all__ = [
    "DATE_FMT","DATETIME_FMT","TTLCache","get_time_vn",
    "discover_market_indices","get_index_detail","get_stock_quote",
    "get_top_stocks","get_history_prices",
    "get_history_df_vnstock","get_prices_df","get_close_series",
    "get_intraday_df",
    "format_stock_info","format_top_stocks",
    "format_market_summary","format_history_text",
    "_same_trading_day",
]

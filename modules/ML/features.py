import numpy as np
import pandas as pd
import pytz
from typing import List, Optional
from qdrant_client import models

from modules.utils.services import qdrant_services  # client Qdrant chia sẻ trong project

ICT = pytz.timezone("Asia/Ho_Chi_Minh")


def _day_from_epoch_s(ts) -> pd.Timestamp:
    """
    Epoch seconds / Timestamp (UTC or tz-aware) -> Timestamp (ICT timezone)
    normalized to 00:00, naive.
    """
    if isinstance(ts, pd.Timestamp):
        t = ts.tz_convert(ICT) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(ICT)
    else:
        t = pd.Timestamp.fromtimestamp(int(ts), tz="UTC").tz_convert(ICT)
    # normalize() = set time to 00:00 same day
    return t.normalize().tz_localize(None)


def _date_range_index(start_ts: int, end_ts: int) -> pd.DatetimeIndex:
    start_day = _day_from_epoch_s(start_ts)
    end_day = _day_from_epoch_s(end_ts)
    return pd.date_range(start_day, end_day, freq="D")


def _cap_pm1(x: float) -> float:
    """
    Clip sentiment về [-1, 1] để tránh outlier.
    """
    return float(np.clip(x, -1.0, 1.0))


def _agg_block(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    Gộp tin theo ngày -> đếm news_count / pos_count / ...
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                f"{prefix}news_count",
                f"{prefix}pos_count",
                f"{prefix}neg_count",
                f"{prefix}neu_count",
                f"{prefix}mean_sent",
                f"{prefix}sum_sent",
            ]
        )

    g = df.groupby("date", as_index=False).agg(
        **{
            f"{prefix}news_count": ("date", "count"),
            f"{prefix}pos_count": ("label", lambda s: (s == "pos").sum()),
            f"{prefix}neg_count": ("label", lambda s: (s == "neg").sum()),
            f"{prefix}neu_count": ("label", lambda s: (s == "neu").sum()),
            f"{prefix}mean_sent": ("sentiment", "mean"),
            f"{prefix}sum_sent": ("sentiment", "sum"),
        }
    )

    # ép numeric sạch
    for c in g.columns:
        if c != "date":
            g[c] = (
                pd.to_numeric(g[c], errors="coerce")
                .fillna(0.0)
                .astype("float64")
            )

    return g


def _scroll_all(
    collection: str,
    flt: Optional[models.Filter],
    with_payload=True,
    limit: int = 2048
):
    """
    Scroll toàn bộ dữ liệu phù hợp filter từ Qdrant
    (không bị giới hạn top-k như search),
    để gom tin trong khoảng thời gian.
    """
    all_pts = []
    next_page = None

    while True:
        pts, next_page = qdrant_services.client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            with_payload=with_payload,
            with_vectors=False,
            limit=limit,
            offset=next_page,
        )

        if not pts:
            break

        all_pts.extend(pts)
        if next_page is None:
            break

    return all_pts


def _payload_rows(points, want_root=False):
    """
    Convert list điểm từ Qdrant -> DataFrame cột:
      date, label, sentiment, (root_id?)
    - date được chuẩn hoá về ngày ICT (naive 00:00)
    - sentiment đã clamp [-1,1]
    - muốn loại trùng bài báo thì dùng root_id
    """
    rows = []
    for p in points:
        pl = p.payload or {}
        ts = pl.get("time_ts")
        if ts is None:
            continue

        rows.append({
            "date": _day_from_epoch_s(ts),
            "label": (pl.get("label") or "neu"),
            "sentiment": _cap_pm1(float(pl.get("sentiment") or 0.0)),
            **({"root_id": pl.get("root_id")} if want_root else {}),
        })

    df = pd.DataFrame(rows)

    # Loại duplicate của cùng 1 bài (root_id) trong cùng 1 ngày
    # để tránh spam/phát tán lại bài cũ.
    if not df.empty and ("root_id" in df.columns):
        df = (
            df
            .drop_duplicates(subset=["date", "root_id"])
            .drop(columns=["root_id"])
        )

    return df


def build_news_features(
    symbol: str,
    start_ts: int,
    end_ts: int,
    add_index: Optional[List[str]] = None,
    collection: Optional[str] = None,
    reindex_full: bool = True,
) -> pd.DataFrame:
    """
    Trích xuất đặc trưng tin tức/sentiment theo ngày cho 1 mã chứng khoán:
      - news_count / pos_count / neg_count / neu_count
      - mean_sent / sum_sent
    và (tuỳ chọn) các chỉ số thị trường (VD: VNINDEX, VN30) với prefix idx_<INDEX>_*

    Điều kiện:
    - Chỉ tính tin trong [start_ts, end_ts]
    - Loại trùng trong cùng 1 ngày qua root_id
    - Nếu reindex_full=True: đảm bảo mỗi ngày trong dải đều có dòng,
      các missing day -> fill 0.0

    OUTPUT columns tối thiểu:
    [date,
     news_count, pos_count, neg_count, neu_count, mean_sent, sum_sent,
     idx_VNINDEX_news_count, ... (nếu add_index có VNINDEX),
     ...]
    """
    coll = collection or getattr(qdrant_services, "collection_name", "cafef_articles")
    sym = symbol.upper()
    add_index = (add_index or [])

    want_fields = ["time_ts", "label", "sentiment", "root_id"]

    # -------------------
    # Tin liên quan mã cổ phiếu
    filt_symbol = models.Filter(
        must=[
            models.FieldCondition(
                key="symbols",
                match=models.MatchAny(any=[sym])
            ),
            models.FieldCondition(
                key="time_ts",
                range=models.Range(gte=int(start_ts), lte=int(end_ts))
            ),
        ]
    )
    pts_sym = _scroll_all(coll, filt_symbol, with_payload=want_fields)
    df_sym = _payload_rows(pts_sym, want_root=True)
    g_sym = _agg_block(df_sym, prefix="")

    dfs = [g_sym] if not g_sym.empty else []

    # -------------------
    # Tin vĩ mô / chỉ số thị trường bổ sung
    for idx_code in add_index:
        idx_code = idx_code.upper()
        filt_idx = models.Filter(
            must=[
                models.FieldCondition(
                    key="index_codes",
                    match=models.MatchAny(any=[idx_code])
                ),
                models.FieldCondition(
                    key="time_ts",
                    range=models.Range(gte=int(start_ts), lte=int(end_ts))
                ),
            ]
        )

        pts_idx = _scroll_all(coll, filt_idx, with_payload=want_fields)
        df_idx = _payload_rows(pts_idx, want_root=True)
        g_idx = _agg_block(df_idx, prefix=f"idx_{idx_code}_")

        if not g_idx.empty:
            dfs.append(g_idx)

    # -------------------
    # Gộp tất cả
    if not dfs:
        out = pd.DataFrame(columns=[
            "date",
            "news_count", "pos_count", "neg_count", "neu_count",
            "mean_sent", "sum_sent",
        ])
    else:
        out = dfs[0]
        for k in range(1, len(dfs)):
            out = out.merge(dfs[k], on="date", how="outer")

        for c in out.columns:
            if c != "date":
                out[c] = (
                    pd.to_numeric(out[c], errors="coerce")
                    .fillna(0.0)
                    .astype("float64")
                )

    # -------------------
    # Reindex full dải ngày để tránh missing
    if reindex_full:
        full_idx = _date_range_index(start_ts, end_ts)
        out = (
            out
            .set_index("date")
            .reindex(full_idx)
            .reset_index()
            .rename(columns={"index": "date"})
        )
        for c in out.columns:
            if c != "date":
                out[c] = (
                    pd.to_numeric(out[c], errors="coerce")
                    .fillna(0.0)
                    .astype("float64")
                )

    out = out.sort_values("date").reset_index(drop=True)
    return out

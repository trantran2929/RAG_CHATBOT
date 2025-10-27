import os
from typing import List, Dict, Optional
from qdrant_client import models
from modules.utils.services import qdrant_services, embedder_services, sentiment_services

def _collection_name() -> str:
    return os.getenv(
        "QDRANT_COLLECTION",
        getattr(qdrant_services, "collection_name", "cafef_articles"),
    )

def _neutral_pack() -> Dict[str, float | str]:
    return {"label": "neu", "sentiment": 0.0}

def _infer_sentiment_batch(batch_docs: List[Dict]) -> List[Dict]:
    """
    Nhận list docs -> trả list kết quả sentiment theo thứ tự.
    Nếu không có model / lỗi -> trả neutral.
    """
    if not batch_docs:
        return []
    if not sentiment_services or not hasattr(sentiment_services, "analyze_batch"):
        return [_neutral_pack() for _ in batch_docs]
    try:
        items = [
            {
                "title": d.get("title", "") or "",
                "summary": d.get("summary", "") or "",
                "content": d.get("content", "") or "",
            }
            for d in batch_docs
        ]
        out = sentiment_services.analyze_batch(items)
        fixed: List[Dict] = []
        for r in out:
            if isinstance(r, dict) and "label" in r and "sentiment" in r:
                try:
                    fixed.append(
                        {
                            "label": str(r.get("label", "neu")),
                            "sentiment": float(r.get("sentiment", 0.0)),
                        }
                    )
                except Exception:
                    fixed.append(_neutral_pack())
            else:
                fixed.append(_neutral_pack())
        if len(fixed) != len(batch_docs):
            fixed = fixed[: len(batch_docs)] + [_neutral_pack()] * max(
                0, len(batch_docs) - len(fixed)
            )
        return fixed
    except Exception as e:
        print(f"[Loader] Sentiment error: {e}")
        return [_neutral_pack() for _ in batch_docs]

def _stable_point_id(d: Dict, j: int) -> str:
    """Ưu tiên d['id']; nếu thiếu, sinh id ổn định từ url|title|time_ts|j."""
    import hashlib
    sid = d.get("id")
    if sid:
        return str(sid)
    m = hashlib.md5()
    m.update((d.get("url", "") or "").encode("utf-8"))
    m.update(b"|")
    m.update((d.get("title", "") or "").encode("utf-8"))
    m.update(b"|")
    m.update(str(int(d.get("time_ts", 0))).encode("utf-8"))
    m.update(b"|")
    m.update(str(j).encode("utf-8"))
    return m.hexdigest()

def load_to_vector_db(
    docs: List[Dict],
    collection_name: Optional[str] = None,
    batch_size: int = 128,
) -> int:
    """
    - Yêu cầu: mỗi doc cần có 'content' và 'time_ts'
    - Gán sentiment/label theo batch.
    - Upsert vào Qdrant dưới dạng vector dense + sparse.
    """
    if not docs:
        return 0

    coll = collection_name or _collection_name()
    print(f"[Loader] {len(docs)} docs → collection='{coll}'")

    valid: List[Dict] = []
    for d in docs:
        txt = (d.get("content") or d.get("summary") or d.get("title") or "").strip()
        if not txt:
            continue
        if "time_ts" not in d:
            continue

        d = dict(d)
        d["content"] = txt
        d["symbols"] = d.get("symbols", []) or []
        d["index_codes"] = d.get("index_codes", []) or []
        if "time" not in d:
            d["time"] = ""  
        valid.append(d)

    if not valid:
        print("[Loader] 0 docs hợp lệ.")
        return 0

    total = 0
    for start in range(0, len(valid), batch_size):
        batch = valid[start : start + batch_size]
        senti_res = _infer_sentiment_batch(batch)

        texts = [b["content"] for b in batch]

        dense_vecs = embedder_services.encode_dense(texts)

        sparse_vecs = embedder_services.encode_sparse(texts)

        points: List[models.PointStruct] = []
        for j, d in enumerate(batch):
            pid = _stable_point_id(d, j)
            sp = sparse_vecs[j]
            s_out = senti_res[j] if j < len(senti_res) else _neutral_pack()

            payload = {
                "id": pid,
                "title": d.get("title", "") or "",
                "url": d.get("url", "") or "",
                "time": d.get("time", "") or "",
                "time_ts": int(d.get("time_ts", 0)),
                "summary": d.get("summary", "") or "",
                "content": d.get("content", "") or "",
                "symbols": list(d.get("symbols", []) or []),
                "index_codes": list(d.get("index_codes", []) or []),
                "sentiment": float(s_out.get("sentiment", 0.0)),
                "label": str(s_out.get("label", "neu")),
                "source": d.get("source", "cafef") or "cafef",
            }

            points.append(
                models.PointStruct(
                    id=pid,
                    vector={
                        "dense_vector": dense_vecs[j],
                        "sparse_vector": models.SparseVector(
                            indices=[int(x) for x in sp["indices"]],
                            values=[float(v) for v in sp["values"]],
                        ),
                    },
                    payload=payload,
                )
            )

        qdrant_services.client.upsert(collection_name=coll, points=points)
        total += len(points)

    print(f"[Loader] Upserted {total} points → '{coll}'")
    return total
import os, time, traceback, hashlib
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Set, List, Dict
from modules.ingestion.crawler import crawl_cafef_stock
from modules.ingestion.preprocess import preprocess_articles


_ML_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-refresh")
_ML_FUTURE: Future | None = None


def _get_qdrant_client():
    """Create a lightweight client without importing embedding/reranking models."""
    from qdrant_client import QdrantClient

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def _get_existing_ids_from_qdrant(
    collection_name: str | None,
    batch_size: int = 1024,
    max_points: int = 200_000,
) -> Set[str]:
    """
    Lấy toàn bộ các point id đã có trong Qdrant (tối đa max_points).
    Dùng để chống nạp trùng.
    """
    existing_ids: Set[str] = set()
    client = _get_qdrant_client()
    coll = collection_name or os.getenv("QDRANT_COLLECTION", "cafef_articles")

    offset = None
    fetched_total = 0

    while True:
        scroll_res, next_page = client.scroll(
            collection_name=coll,
            limit=batch_size,
            with_payload=["url", "content"],
            offset=offset,
        )

        if not scroll_res:
            break

        for point in scroll_res:
            # point.id có thể là str hoặc int -> ép về str để so sánh
            existing_ids.add(str(point.id))
            payload = point.payload or {}
            fingerprint = _doc_fingerprint(payload)
            if fingerprint:
                existing_ids.add(fingerprint)

        fetched_total += len(scroll_res)
        if fetched_total >= max_points:
            break

        if not next_page:
            break
        offset = next_page

    print(f"[Ingestion] Đã load {len(existing_ids)} khóa ID/fingerprint để kiểm tra trùng.")
    return existing_ids


def _doc_fingerprint(doc: Dict) -> str:
    url = str(doc.get("url") or "").strip()
    content = str(doc.get("content") or "").strip()
    if not url or not content:
        return ""
    digest = hashlib.sha256(f"{url}|{content}".encode("utf-8")).hexdigest()
    return f"doc:{digest}"


def _filter_new_docs(
    docs: List[Dict],
    existing_ids: Set[str],
    min_time_ts: int | None = None,
) -> List[Dict]:
    """
    Giữ lại:
    - doc có id chưa tồn tại trong Qdrant
    - doc có time_ts >= min_time_ts (nếu min_time_ts được cung cấp)
    """
    fresh_docs: List[Dict] = []

    for d in docs:
        pid = str(d.get("id", "")).strip()
        if not pid:
            continue

        if pid in existing_ids:
            continue
        fingerprint = _doc_fingerprint(d)
        if fingerprint and fingerprint in existing_ids:
            continue

        if min_time_ts is not None:
            try:
                ts_val = int(d.get("time_ts", 0))
            except (TypeError, ValueError):
                continue
            if ts_val < min_time_ts:
                # quá cũ → bỏ qua
                continue

        fresh_docs.append(d)

    return fresh_docs


def run_ingestion_cycle(
    collection_name: str | None = None,
    max_pages: int = 1,
    max_age_days: int = 3,
    now_ts: int | None = None,
) -> Dict[str, int]:
    """Run exactly one crawl/filter/upsert cycle and return observable counts."""
    cycle_now = int(now_ts if now_ts is not None else time.time())
    cutoff_ts = cycle_now - max(0, int(max_age_days)) * 24 * 3600

    raw_articles = crawl_cafef_stock(max_pages=max_pages)
    print(f"[Ingestion] Crawl được {len(raw_articles)} bài gốc")
    chunked_docs = preprocess_articles(raw_articles, max_words=400)
    print(f"[Ingestion] Tách thành {len(chunked_docs)} chunk docs")

    if not chunked_docs:
        return {"articles": len(raw_articles), "chunks": 0, "new_docs": 0, "upserted": 0}

    existing_ids = _get_existing_ids_from_qdrant(collection_name)
    new_docs = _filter_new_docs(
        docs=chunked_docs,
        existing_ids=existing_ids,
        min_time_ts=cutoff_ts,
    )
    print(f"[Ingestion] Sau khi lọc, còn {len(new_docs)} docs mới cần upsert")

    upserted = 0
    if new_docs:
        from modules.ingestion.loader import load_to_vector_db
        upserted = load_to_vector_db(new_docs, collection_name=collection_name)
        print(f"[Ingestion] ✅ Đã upsert {upserted} doc MỚI vào `{collection_name or '[default]'}`")
    else:
        print("[Ingestion] Không có doc mới (toàn trùng hoặc quá cũ). Bỏ qua upsert.")

    return {
        "articles": len(raw_articles),
        "chunks": len(chunked_docs),
        "new_docs": len(new_docs),
        "upserted": int(upserted),
    }


def _refresh_models(symbols: List[str]) -> Dict[str, str]:
    from modules.ML.pipeline import refresh_stale_models
    return refresh_stale_models(symbols)


def _schedule_model_refresh(symbols: List[str]) -> None:
    """Do not let slow SARIMAX fitting block the next news crawl cycle."""
    global _ML_FUTURE
    if _ML_FUTURE is not None and not _ML_FUTURE.done():
        print("[ML] Refresh trước vẫn đang chạy; bỏ qua lần kích hoạt này.")
        return
    if _ML_FUTURE is not None:
        try:
            print(f"[ML] Refresh status: {_ML_FUTURE.result()}")
        except Exception as exc:
            print(f"[ML] Refresh trước bị lỗi: {exc}")
    print(f"[ML] Bắt đầu refresh nền: {', '.join(symbols)}")
    _ML_FUTURE = _ML_EXECUTOR.submit(_refresh_models, symbols)


def run_scheduler():
    coll = os.getenv("QDRANT_COLLECTION")
    interval = int(os.getenv("INGEST_INTERVAL", 3600))
    max_pages = int(os.getenv("CRAWL_MAX_PAGES", 1))
    retrain_enabled = os.getenv("ML_RETRAIN_AFTER_INGEST", "1") == "1"
    retrain_symbols = [
        s.strip().upper()
        for s in os.getenv("ML_RETRAIN_SYMBOLS", "FPT,HPG,VCB,SHS,THU").split(",")
        if s.strip()
    ]

    max_age_days = int(os.getenv("MAX_NEWS_AGE_DAYS", 3))

    while True:
        try:
            print("\n[Ingestion] Bắt đầu vòng đồng bộ tin tức mới...")

            stats = run_ingestion_cycle(coll, max_pages, max_age_days)
            print(f"[Ingestion] Cycle stats: {stats}")

            if stats["upserted"] > 0 and retrain_enabled and retrain_symbols:
                _schedule_model_refresh(retrain_symbols)

        except Exception as e:
            print(f"[Ingestion] ❌ LỖI: {e}")
            traceback.print_exc()

        print(f"[Ingestion] Sleeping {interval}s...\n")
        time.sleep(interval)


if __name__ == "__main__":
    run_scheduler()

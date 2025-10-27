import os, time, traceback
from typing import Set, List, Dict
from modules.ingestion.crawler import crawl_cafef_stock
from modules.ingestion.preprocess import preprocess_articles
from modules.ingestion.loader import load_to_vector_db
from modules.utils.services import qdrant_services


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
    coll = collection_name or qdrant_services.collection_name

    offset = None
    fetched_total = 0

    while True:
        scroll_res, next_page = qdrant_services.client.scroll(
            collection_name=coll,
            limit=batch_size,
            with_payload=False,
            offset=offset,
        )

        if not scroll_res:
            break

        for point in scroll_res:
            # point.id có thể là str hoặc int -> ép về str để so sánh
            existing_ids.add(str(point.id))

        fetched_total += len(scroll_res)
        if fetched_total >= max_points:
            break

        if not next_page:
            break
        offset = next_page

    print(f"[Ingestion] Đã load {len(existing_ids)} point_id từ Qdrant để kiểm tra trùng.")
    return existing_ids


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

        if min_time_ts is not None:
            ts_val = int(d.get("time_ts", 0))
            if ts_val < min_time_ts:
                # quá cũ → bỏ qua
                continue

        fresh_docs.append(d)

    return fresh_docs


def run_scheduler():
    coll = os.getenv("QDRANT_COLLECTION")
    interval = int(os.getenv("INGEST_INTERVAL", 3600))
    max_pages = int(os.getenv("CRAWL_MAX_PAGES", 1))

    max_age_days = int(os.getenv("MAX_NEWS_AGE_DAYS", 3))
    cutoff_ts = int(
        (time.time() - max_age_days * 24 * 3600)
    )

    while True:
        try:
            print("\n[Ingestion] Bắt đầu vòng đồng bộ tin tức mới...")

            # 1) Crawl thô
            raw_articles = crawl_cafef_stock(max_pages=max_pages)
            print(f"[Ingestion] Crawl được {len(raw_articles)} bài gốc")

            # 2) Tiền xử lý → chunk docs với schema cố định
            chunked_docs = preprocess_articles(raw_articles, max_words=220)
            print(f"[Ingestion] Tách thành {len(chunked_docs)} chunk docs")

            if not chunked_docs:
                print("[Ingestion] Không có dữ liệu hợp lệ sau preprocess. Ngủ tiếp.")
                time.sleep(interval)
                continue

            # 3) Lấy danh sách ID đã tồn tại từ Qdrant để chống trùng
            existing_ids = _get_existing_ids_from_qdrant(coll)

            # 4) Lọc chỉ giữ lại tin MỚI + GẦN ĐÂY
            new_docs = _filter_new_docs(
                docs=chunked_docs,
                existing_ids=existing_ids,
                min_time_ts=cutoff_ts,
            )

            print(f"[Ingestion] Sau khi lọc, còn {len(new_docs)} docs mới cần upsert")

            if new_docs:
                n = load_to_vector_db(new_docs, collection_name=coll)
                print(
                    f"[Ingestion] ✅ Đã upsert {n} doc MỚI vào `{coll or '[default]'}`"
                )
            else:
                print("[Ingestion] Không có doc mới (toàn trùng hoặc quá cũ). Bỏ qua upsert.")

        except Exception as e:
            print(f"[Ingestion] ❌ LỖI: {e}")
            traceback.print_exc()

        print(f"[Ingestion] Sleeping {interval}s...\n")
        time.sleep(interval)


if __name__ == "__main__":
    run_scheduler()

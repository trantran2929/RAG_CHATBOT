import uuid
from typing import List, Dict
from modules.utils.services import qdrant_services, embedder_services
from qdrant_client import models
from datetime import datetime
from dateutil import parser
import pytz, hashlib, time

def normalize_time_str(time_str: str) -> dict:
    """
    Chuyển time string sang dict {time_str, time_ts}
    - Cố gắng parse nhiều format khác nhau
    - Nếu không parse được thì trả về None cho time_ts
    """
    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")

    if not time_str or not isinstance(time_str, str):
        now =datetime.now(vn_tz)
        return {"time_str": now.strftime("%d-%m-%Y, %H:%M:%S"), "time_ts": int(now.timestamp())}

    try:
        #hỗ trợ ISO, dd-mm-YYYY, dd/mm/YYYY ...
        dt = parser.parse(time_str, dayfirst=True)

        # Nếu datetime chưa có tzinfo thì gán Asia/Ho_Chi_Minh
        if not dt.tzinfo:
            dt = vn_tz.localize(dt)
        else:
            dt = dt.astimezone(vn_tz)

        return {
            "time_str": dt.strftime("%d-%m-%Y %H:%M:%S"),
            "time_ts": int(dt.timestamp())
        }

    except Exception as e:
        now = datetime.now(vn_tz)
        print(f"Parse lỗi ({time_str}): {e}")
        return {
            "time_str": now.strftime("%d-%m-%Y %H:%M:%S"),
            "time_ts": int(now.timestamp())
        }


def load_to_vector_db(docs: List[Dict], collection_name: str = None, batch_size: int = 100) -> int:
    """Nhận docs (có thể gồm nhiều chunk), tạo dense + sparse embeddings, upsert vào Qdrant."""
    if not docs:
        return 0

    start_all = time.time()
    coll = collection_name or qdrant_services.collection_name

    # Không loại trùng theo URL — vì mỗi chunk là 1 point riêng biệt
    print(f"[Loader] Nhận {len(docs)} đoạn văn cần upload lên Qdrant...")

    #Lọc bỏ các doc trống nội dung
    valid_docs = []
    for doc in docs:
        content_text = (
            doc.get("content") or
            doc.get("summary") or
            doc.get("title") or
            ""
        ).strip()
        if not content_text:
            continue

        doc["content"] = content_text
        valid_docs.append(doc)
    
    if not valid_docs:
        print("[Loader] Tất cả tài liệu đều rỗng, không có gì upload")
        return 0
    
    print(f"[Loader] Sau khi lọc còn {len(valid_docs)} tài liệu hợp lệ.")

    corpus = [doc.get("content", "") for doc in docs]
    embedder_services.fit_bm25(corpus)

    total = 0
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        text = [doc.get("content", "") for doc in batch]

        dense_vecs = embedder_services.encode_dense(text)
        sparse_vecs = embedder_services.encode_sparse(text)

        points = []
        for i, doc in enumerate(batch):
            # Tạo id duy nhất theo (url + index + md5)
            base_key = f"{doc.get('url','')}_{i}_{uuid.uuid4()}"
            point_id = hashlib.md5(base_key.encode("utf-8")).hexdigest()

            sparse_raw = sparse_vecs[i]
            sparse_vec = models.SparseVector(
                indices=sparse_raw["indices"],
                values=sparse_raw["values"]
            )

            time_info = normalize_time_str(doc.get("time", ""))

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense_vector": dense_vecs[i],
                        "sparse_vector": sparse_vec
                    },
                    payload={
                        "id": point_id,
                        "title": doc.get("title", ""),
                        "time": time_info["time_str"],
                        "time_ts": time_info["time_ts"],
                        "summary": doc.get("summary", ""),
                        "url": doc.get("url", ""),
                        "content": doc.get("content", ""),
                        "source": doc.get("source", "cafef"),
                        "chunk_index": i,   
                    }
                )
            )

        qdrant_services.client.upsert(collection_name=coll, points=points)
        total += len(points)

    elapsed = round(time.time() - start_all, 2)
    print(f"[Loader] Đã upload {total} chunks lên Qdrant trong {elapsed}s\n")
    return total
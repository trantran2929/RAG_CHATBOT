import uuid
from typing import List, Dict
from modules.nodes.embedder import embed_query
from modules.nodes.services import qdrant_services
from qdrant_client.models import VectorParams

def ensure_collection(collection_name: str, vector_size: int = 384):
    collections = [c.name for c in qdrant_services.client.get_collection().collections]
    if collection_name not in collections:
        print(f"{collection_name} chưa tồn tại, tiến hành tạo mới")
        qdrant_services.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance="Cosine")
        )
    else:
        print(f"Collection {collection_name} đã tồn tại")

def load_to_vector_db(docs: List[Dict], collection_name: str = "defaulr_collection", vector_size: int = 384) -> int:
    """
    Nhận danh sách docs trả về, gọi emb để tạo vector và lưu vào qdrant
    Trả về số lượng docs đã nạp.
    """
    if not docs:
        return 0
    ensure_collection(collection_name,vector_size=vector_size)

    texts = [doc.get("content","") for doc in docs]
    metadata = []
    ids = []

    for doc in docs:
        meta = {
            "title": doc.get("title", ""),
            "time": doc.get("time",""),
            "summary": doc.get("summary",""),
            "link": doc.get("link","")
        }
        metadata.append(meta)
        ids.append(doc.get("id") or str(uuid.uuid4()))

    vectors = embed_query(texts)

    qdrant_services.client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": ids[i],
                "vector": vectors[i],
                "payload": metadata[i]
            }
            for i in range(len(docs))
        ]
    )
    print(f"Đã nạp {len(docs)} documents vào collection {collection_name}.")

    return len(docs)

def delete_by_collection(collection_name: str):
    try:
        qdrant_services.client.delete_collection(collection_name=collection_name)
        print(f"Đã xóa collection {collection_name}")
    except Exception as e:
        print(f"Lỗi xóa collection {collection_name}: {e}")


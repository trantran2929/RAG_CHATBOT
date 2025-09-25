import uuid
from typing import List, Dict
from rag_note.modules.utils.services import qdrant_services, embedder_services

def load_to_vector_db(docs: List[Dict], collection_name: str = None) -> int:
    """Nhận docs, tạo dense + sparse embeddings, upsert vào Qdrant."""
    if not docs:
        return 0

    coll = collection_name or qdrant_services.collection_name
    texts = [doc.get("content", "") for doc in docs]

    dense_vecs = embedder_services.encode_dense(texts)
    sparse_vecs = embedder_services.encode_sparse(texts)

    if len(dense_vecs) != len(texts) or len(sparse_vecs) != len(texts):
        raise ValueError("Embedding size mismatch với số docs!")

    ids = [doc.get("id") or str(uuid.uuid4()) for doc in docs]

    points = []
    for i, doc in enumerate(docs):
        points.append({
            "id": ids[i],
            "vector": {
                "dense_vector": dense_vecs[i],
                "sparse_vector": sparse_vecs[i],
            },
            "payload": {
                "title": doc.get("title", ""),
                "time": doc.get("time", ""),
                "summary": doc.get("summary", ""),
                "url": doc.get("url", ""),     
                "content": doc.get("content", ""),
                "source": doc.get("source", "cafef")  
            },
        })

    qdrant_services.client.upsert(
        collection_name=coll,
        points=points,
    )
    print(f"[VectorDB] Đã nạp {len(docs)} documents vào collection `{coll}`.")
    return len(docs)

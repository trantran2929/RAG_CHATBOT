import uuid
from typing import List, Dict
from modules.utils.services import qdrant_services, embedder_services
from qdrant_client import models

def load_to_vector_db(docs: List[Dict], collection_name: str = None, batch_size: int = 100) -> int:
    """Nhận docs, tạo dense + sparse embeddings, upsert vào Qdrant."""
    if not docs:
        return 0

    coll = collection_name 

    corpus = [doc.get("content","") for doc in docs]
    embedder_services.fit_bm25(corpus)
    total = 0

    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        text = [doc.get("content", "") for doc in batch]

        dense_vecs = embedder_services.encode_dense(text)
        sparse_vecs = embedder_services.encode_sparse(text)
        points = []
        for i, doc in enumerate(batch):
            sparse_raw = sparse_vecs[i]
            sparse_vec = models.SparseVector(
                indices=sparse_raw["indices"],
                values=sparse_raw["values"]
            )

            point_id = str(uuid.uuid4())

            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense_vector": dense_vecs[i],
                    "sparse_vector": sparse_vec
                },
                payload={
                    "id": doc.get("id", point_id),
                    "title": doc.get("title", ""),
                    "time": doc.get("time", ""),
                    "summary": doc.get("summary", ""),
                    "url": doc.get("url", ""),
                    "content": doc.get("content", ""),
                    "source": doc.get("source", "cafef")
                }
            )
            points.append(point)

        qdrant_services.client.upsert(
            collection_name=coll,
            points=points,
        )
        total += len(batch)
        print(f"[VectorDB] Đã nạp {len(batch)} documents vào collection `{coll}`.")
    return total

from qdrant_client import QdrantClient
from qdrant_client.http import models
from uuid import uuid4
import time
from .embedder import get_embedding

qdrant_client = QdrantClient(host="localhost", port=6333, prefer_grpc=False)

knowledge_collection = "rag_knowledge"
chat_collection = "chat_history"

# qdrant_client.delete_collection(collection_name=knowledge_collection)
# Tạo collection
for col_name, size in [(knowledge_collection, 384), (chat_collection,384)]:
    if col_name not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.recreate_collection(
            collection_name=col_name,
            vectors_config={"size": 384, "distance": "Cosine"}
        )

def save_knowledge(documents: list[str]):
    for idx, doc in enumerate(documents):
        vec = get_embedding(doc).squeeze().tolist()
        qdrant_client.upsert(
            collection_name=knowledge_collection,
            points=[{"id": idx, "vector": vec, "payload": {"text": doc}}]
        )

def save_message(role, content):
    vec = get_embedding(content).squeeze().tolist()
    qdrant_client.upsert(
        collection_name=chat_collection,
        points=[{
            "id": str(uuid4()),
            "vector": vec,
            "payload": {
                "role": role.lower(),
                "content": content,
                "timestamp": time.time()
            }
        }]
    )

def load_history(top_k=20):
    resp, _ = qdrant_client.scroll(collection_name=chat_collection, limit=top_k, with_payload=True)
    points = getattr(resp, "result", resp)  # tương thích với các phiên bản client
    if not points:
        return []
    points = sorted(resp, key=lambda x: x.payload.get("timestamp",0))
    return [{"role": p.payload["role"], "content": p.payload["content"]} for p in points]

# Xóa hết point trong collection giữ lại collection
def clear_history():
    qdrant_client.delete(
        collection_name=chat_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[])   # filter trống = match tất cả
        )
    )
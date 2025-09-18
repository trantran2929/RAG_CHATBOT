from .vector_db import qdrant_client, knowledge_collection
import uuid

# Xem thông tin collection
info = qdrant_client.get_collection(knowledge_collection)
print("collection info:", info)
try:
    print("vectors config:", info.config)
except Exception:
    print("Không lấy được info.config.params.vectors — in dir(info):", dir(info))

#Upsert test point (hợp lệ)
vec0 = [0.0] * 384
qdrant_client.upsert(
    collection_name=knowledge_collection,
    points=[{
        "id": str(uuid.uuid4()),
        "vector": vec0,   
        "payload": {"text": "DBG ZERO"}
    }]
)

res = qdrant_client.query_points(
    collection_name=knowledge_collection,
    query=vec0,   
    limit=1,
    with_payload=True
)

print("query result:", res)

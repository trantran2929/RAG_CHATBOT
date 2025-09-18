from .vector_db import qdrant_client, knowledge_collection
from .embedder import get_embedding

def retrieve_context(query: str, top_k=2):
    query_vec = get_embedding(query)
    if hasattr(query_vec, "ndim") and query_vec.ndim > 1:
        query_vec = query_vec[0]
    query_vec = query_vec.tolist()
    results = qdrant_client.search(
        collection_name=knowledge_collection,
        query_vector=query_vec,
        limit=top_k,
        # vector_name="default",
        with_payload=["text"]
    )
    return [hit.payload["text"] for hit in results]

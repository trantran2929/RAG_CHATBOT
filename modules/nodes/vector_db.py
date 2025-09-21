from .services import qdrant_services
from qdrant_client.http import models
import numpy as np
from modules.utils.debug import add_debug_info
from datetime import datetime
import uuid


def _get_next_version(client, collection_name, question):
    """
    Lấy version tiếp theo cho một câu hỏi.
    """
    hits, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(
                key="question", match=models.MatchValue(value=question)
            )]
        ),
        limit=100  # lấy nhiều để tìm version cao nhất
    )
    if hits:
        last = max([p.payload.get("version", 1) for p in hits])
        return last + 1
    return 1


def search_and_store(state):
    """
    Tìm trong Qdrant sử dụng state.query_embedding.
    Lưu kết quả vào state.search_results.
    Đồng thời upsert vector + answer vào Qdrant với version + timestamp.
    """
    if state.query_embedding is None or (
        isinstance(state.query_embedding, np.ndarray) and state.query_embedding.size == 0
    ):
        raise ValueError("query_embedding chưa được tính, chạy embedder.")

    client = qdrant_services.client
    collection_name = qdrant_services.collection_name

    # Lấy vector dạng list
    if isinstance(state.query_embedding, np.ndarray):
        query_vector = (
            state.query_embedding[0].tolist()
            if state.query_embedding.ndim > 1
            else state.query_embedding.tolist()
        )
    else:
        query_vector = state.query_embedding

    # Search top-k từ knowledge base
    result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
        with_payload=True
    )

    state.search_results = [
        {
            "question": r.payload.get("question"),
            "answer": r.payload.get("answer"),
            "version": r.payload.get("version"),
            "timestamp": r.payload.get("timestamp"),
            "score": r.score,
        }
        for r in result
    ]

    # Nếu có final_answer → lưu vào Qdrant với version mới
    if getattr(state, "final_answer", None):
        version = _get_next_version(client, collection_name, state.clean_query)
        payload = {
            "type": "query",
            "question": state.clean_query,
            "answer": state.final_answer,
            "session_id": state.session_id,
            "version": version,
            "timestamp": datetime.utcnow().isoformat()
        }

        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=query_vector,
                    payload=payload
                )
            ]
        )

    add_debug_info(state, "search_result_count", len(state.search_results))
    return state


def get_all_versions(question):
    """
    Truy xuất toàn bộ version của một câu hỏi trong Qdrant.
    """
    client = qdrant_services.client
    collection_name = qdrant_services.collection_name

    hits, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(
                key="question", match=models.MatchValue(value=question)
            )]
        ),
        limit=100
    )

    return sorted(
        [
            {
                "answer": h.payload.get("answer"),
                "version": h.payload.get("version"),
                "timestamp": h.payload.get("timestamp"),
            }
            for h in hits
        ],
        key=lambda x: x["version"]
    )
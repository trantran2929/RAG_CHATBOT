from modules.core.state import GlobalState
from .services import qdrant_services
from modules.utils.debug import add_debug_info
import numpy as np


def search_vector_db(state: GlobalState, top_k: int = 5) -> GlobalState:
    """
    Dùng query_embedding trong state để search trong Knowledge Base (Qdrant).
    Lưu kết quả search vào state.search_results.
    """

    if state.query_embedding is None:
        add_debug_info(state, "vector_db", "Không tìm thấy embedding, skip search")
        state.search_results = []
        return state

    try:
        # Đảm bảo embedding là list[float]
        if isinstance(state.query_embedding, np.ndarray):
            query_vector = state.query_embedding.astype(float).tolist()
        elif isinstance(state.query_embedding, list):
            query_vector = [float(x) for x in state.query_embedding]
        else:
            raise ValueError("query_embedding không hợp lệ")

        # Search vector trong Qdrant (Knowledge Base)
        results = qdrant_services.client.search(
            collection_name=qdrant_services.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        if not results:
            add_debug_info(state, "vector_db_search", "Không tìm thấy kết quả trong KB")
            state.search_results = []
            return state

        # Chuẩn hóa kết quả
        state.search_results = [
            {"id": r.id, "score": r.score, "payload": r.payload}
            for r in results
        ]

        add_debug_info(
            state, 
            "vector_db_results", 
            f"Tìm thấy {len(state.search_results)} kết quả từ KB"
        )

    except Exception as e:
        add_debug_info(state, "vector_db_error", f"Lỗi: {str(e)}")
        state.search_results = []

    return state
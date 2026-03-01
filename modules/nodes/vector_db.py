# modules/nodes/vector_db.py
from modules.core.state import GlobalState
from modules.utils.services import qdrant_services
from time import perf_counter
import pytz
from datetime import datetime, timedelta
from qdrant_client import models
from modules.utils.time_utils import resolve_time_window


def normalize_score(score: float) -> float:
    """
    Chuẩn hóa score cosine similarity [-1,1] thành [0,1]
    (phục vụ debug / phân tích, không dùng để RRF ở đây)
    """
    return (score + 1) / 2 if score is not None else 0.0


def _search_modality(
    vec,
    using: str,
    top_k: int,
    search_filter: models.Filter,
):
    """
    Helper: gọi Qdrant cho 1 modality (dense hoặc sparse).
    - using = "dense_vector" hoặc "sparse_vector"
    """
    if vec is None:
        return []

    if using == "sparse_vector":
        query = models.SparseVector(indices=vec["indices"], values=vec["values"])
    else:
        query = vec

    result = qdrant_services.client.query_points(
        collection_name=qdrant_services.collection_name,
        query=query,
        using=using,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )

    hits = []
    for rank, p in enumerate(result.points, start=1):
        payload = p.payload or {}
        hits.append(
            {
                "id": p.id,
                "rank": rank, 
                "score": round(normalize_score(p.score), 4),
                "title": payload.get("title", ""),
                "time": payload.get("time", ""),
                "time_ts": payload.get("time_ts", 0),
                "url": payload.get("url", ""),
                "content": (payload.get("content") or "")[:1000],
            }
        )
    return hits


def search_vector_db(state: GlobalState, top_k: int = 5) -> GlobalState:
    """
    NHIỆM VỤ:
    - CHỈ search Qdrant cho từng modality (dense / sparse).
    - Kết quả thô được lưu vào:
        - state.search_results_dense
        - state.search_results_sparse
    - state.search_results = dense + sparse (để debug tổng hợp).
    - Filter thời gian được thực hiện bằng resolve_time_window.
    """
    route_to = getattr(state, "route_to", "")
    if route_to not in ["rag", "hybrid"]:
        state.search_results = []
        state.search_results_dense = []
        state.search_results_sparse = []
        state.add_debug("vector_db", "Skipped (route not rag/hybrid)")
        state.llm_status = "vector_db_skipped"
        return state

    if not state.query_embedding:
        state.search_results = []
        state.search_results_dense = []
        state.search_results_sparse = []
        state.add_debug("vector_db", "No embedding")
        state.llm_status = "vector_db_no_embedding"
        return state

    dense_vec = state.query_embedding.get("dense_vector")
    sparse_vec_list = state.query_embedding.get("sparse_vector")

    # Lấy sparse vector đầu tiên nếu encode_sparse trả list
    sparse_vec = None
    if sparse_vec_list and isinstance(sparse_vec_list, list) and len(sparse_vec_list) > 0:
        sparse_vec = sparse_vec_list[0]

    if dense_vec is None and sparse_vec is None:
        state.search_results = []
        state.search_results_dense = []
        state.search_results_sparse = []
        state.add_debug("vector_db", "No dense/sparse vectors")
        return state

    # Dùng helper chung để lấy (start_ts, end_ts)
    # - Nếu state.time_filter có Processor detect từ câu hỏi thì dùng luôn.
    # - Nếu không thì default [now - 72h, now]
    start_ts, end_ts = resolve_time_window(state, default_hours=72)

    search_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="time_ts",
                range=models.Range(gte=start_ts, lte=end_ts),
            )
        ]
    )

    try:
        start_t = perf_counter()

        dense_hits = _search_modality(
            dense_vec, using="dense_vector", top_k=top_k, search_filter=search_filter
        )

        sparse_hits = _search_modality(
            sparse_vec, using="sparse_vector", top_k=top_k, search_filter=search_filter
        )

        elapsed = round(perf_counter() - start_t, 3)

        state.search_results_dense = dense_hits
        state.search_results_sparse = sparse_hits
        state.search_results = dense_hits + sparse_hits 

        state.llm_status = "vector_db_success"
        state.add_debug("vector_db_dense", len(dense_hits))
        state.add_debug("vector_db_sparse", len(sparse_hits))
        state.add_debug("vector_db_time_sec", elapsed)
        state.add_debug("vector_db_filter_start", start_ts)
        state.add_debug("vector_db_filter_end", end_ts)
        state.add_debug("vector_db_route", route_to)

        return state

    except Exception as e:
        state.search_results = []
        state.search_results_dense = []
        state.search_results_sparse = []
        state.llm_status = "vector_db_error"
        state.add_debug("vector_db_exception", str(e))
        return state

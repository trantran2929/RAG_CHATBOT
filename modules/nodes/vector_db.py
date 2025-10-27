# modules/nodes/vector_db.py
from modules.core.state import GlobalState
from modules.utils.services import qdrant_services, reranker_services
from time import perf_counter
import pytz
from datetime import datetime, timedelta
from qdrant_client import models


def normalize_score(score: float) -> float:
    """
    Chuẩn hóa score cosine similarity [-1,1] thành [0,1]
    """
    return (score + 1) / 2 if score is not None else 0.0


def search_vector_db(state: GlobalState, top_k: int = 5, alpha: float = 0.7) -> GlobalState:
    """
    Hybrid search (dense + sparse) trong Qdrant + reranker sắp xếp lại.
    - Chỉ chạy khi route_to in {"rag", "hybrid"}
    - Nếu có time_filter trong state -> lọc theo mốc thời gian đó
    - Nếu không -> mặc định chỉ lấy tin 3 ngày gần đây
    - Sau đó rerank bằng CrossEncoder.

    Kết quả:
    - state.search_results = list các hits [{id, score, title, time, ...}, ...]
    - debug_info cập nhật chi tiết
    """
    route_to = getattr(state, "route_to", "")
    if route_to not in ["rag", "hybrid"]:
        state.search_results = []
        state.add_debug("vector_db", "Skipped (route not rag/hybrid)")
        state.llm_status = "vector_db_skipped"
        return state

    if not state.query_embedding:
        state.search_results = []
        state.add_debug("vector_db", "No embedding")
        state.llm_status = "vector_db_no_embedding"
        return state

    dense_vec = state.query_embedding.get("dense_vector")
    sparse_vec = state.query_embedding.get("sparse_vector")

    if dense_vec is None and sparse_vec is None:
        state.search_results = []
        state.add_debug("vector_db", "No dense/sparse vectors")
        return state

    # Xác định filter thời gian
    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    now_vn = datetime.now(vn_tz)

    if getattr(state, "time_filter", None):
        start_ts, end_ts = state.time_filter
    else:
        start_ts = int((now_vn - timedelta(days=3)).timestamp())
        end_ts = int(now_vn.timestamp())

    search_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="time_ts",
                range=models.Range(gte=start_ts, lte=end_ts)
            )
        ]
    )

    try:
        start_t = perf_counter()
        prefetches = []

        if dense_vec is not None:
            prefetches.append(
                models.Prefetch(
                    query=dense_vec,
                    using="dense_vector",
                    limit=top_k
                )
            )

        if sparse_vec and isinstance(sparse_vec, list) and len(sparse_vec) > 0:
            sv = sparse_vec[0]
            prefetches.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sv["indices"],
                        values=sv["values"]
                    ),
                    using="sparse_vector",
                    limit=top_k,
                )
            )

        if not prefetches:
            state.add_debug("vector_db", "No prefetch queries")
            state.search_results = []
            return state

        result = qdrant_services.client.query_points(
            collection_name=qdrant_services.collection_name,
            prefetch=prefetches,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=search_filter,
            with_payload=True
        )

        hits = []
        for p in result.points:
            score = normalize_score(p.score)
            payload = p.payload or {}
            hits.append({
                "id": p.id,
                "score": round(score, 4),
                "title": payload.get("title", ""),
                "time": payload.get("time", ""),
                "time_ts": payload.get("time_ts", 0),
                "url": payload.get("url", ""),
                "content": (payload.get("content") or "")[:1000]
            })
            
        if hits and getattr(state, "use_reranker", True):
            hits = reranker_services.rerank(state.user_query, hits)
            state.add_debug("reranker_status", "applied")
        else:
            state.add_debug("reranker_status", "skipped")

        elapsed = round(perf_counter() - start_t, 3)

        state.search_results = hits
        state.llm_status = "vector_db_success"

        state.add_debug("vector_db_results", len(hits))
        state.add_debug("vector_db_time_sec", elapsed)
        state.add_debug("vector_db_filter_start", start_ts)
        state.add_debug("vector_db_filter_end", end_ts)
        state.add_debug("vector_db_route", route_to)

        return state

    except Exception as e:
        state.search_results = []
        state.llm_status = "vector_db_error"
        state.add_debug("vector_db_exception", str(e))
        return state

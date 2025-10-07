from modules.core.state import GlobalState
from modules.utils.services import qdrant_services
import pytz
from datetime import datetime, timedelta
from qdrant_client import models

def normalize_score(score: float) -> float:
    """
    Chuẩn hóa score cosine similarity [-1,1] thành [0,1]
    """
    return (score + 1) /2 if score is not None else 0.0

def search_vector_db(state: GlobalState, top_k: int = 5, alpha: float = 0.7) -> GlobalState:
    """
    Hybrid search (dense + sparse) trong Qdrant.
    - Nếu có time_filter: lọc theo thời gian cụ thể (hôm qua, hôm nay, tuần trước,...)
    - Nếu không có mặc định lấy dữ liệu trong 3 ngày gần nhất
    - Nếu query đã được api xử lý (state.api_response): bỏ qua vector search
    """
    if getattr(state,"api_response",None):
        state.add_debug("vector_db", "Skipped")
        state.search_results = []
        return state
    
    if not state.query_embedding:
        state.add_debug("vector_db", "No embedding")
        state.search_results = []
        print(f"No embedding for query: {state.user_query}", flush=True)
        return state

    dense_vec = state.query_embedding.get("dense_vector")
    sparse_vec = state.query_embedding.get("sparse_vector")

    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    now_vn = datetime.now(vn_tz)
    now_utc = now_vn.astimezone(pytz.utc)

    if getattr(state, "time_filter", None):
        start_ts, end_ts = state.time_filter
    else:
        start_ts = int((now_utc-timedelta(days=3)).timestamp())
        end_ts = int(now_utc.timestamp())

    search_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="time_ts",
                range=models.Range(gte=start_ts, lte=end_ts)
            )
        ]
    )

    dense_results, sparse_results = [], []

    try:
        if dense_vec is not None:
            dense_results = qdrant_services.client.search(
                collection_name=qdrant_services.collection_name,
                query_vector=("dense_vector", dense_vec),
                limit=top_k,
                with_payload=True,
                query_filter=search_filter
            )

        if sparse_vec is not None:
            sparse_results = qdrant_services.client.search(
                collection_name=qdrant_services.collection_name,
                query_vector=("sparse_vector", sparse_vec),
                limit=top_k,
                with_payload=True,
                query_filter=search_filter
            )
            
        if dense_results and sparse_results:
            combined = {}
            for r in dense_results:
                combined[r.id] = {
                    "id": r.id,
                    "score": normalize_score(r.score)* alpha,
                    "payload": r.payload or {}
                }
            for r in sparse_results:
                if r.id in combined:
                    combined[r.id]["score"] += r.score * (1 - alpha)
                else:
                    combined[r.id] = {
                        "id": r.id,
                        "score": normalize_score(r.score) * (1 - alpha),
                        "payload": r.payload or {}
                    }
            state.search_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]

        else:
            base_results = dense_results if dense_results else sparse_results
            state.search_results = [
                {
                    "id": r.id,
                    "score": normalize_score(r.score),
                    "payload": r.payload or {}
                } for r in base_results
            ]

        state.add_debug("vector_db_results", f"Found {len(state.search_results)} results")
        if getattr(state,"debug",False):
            print(f"[VectorDB] Found {len(state.search_results)} results", flush=True)
            for r in state.search_results[:3]:
                print({
                    "id": r["id"],
                    "score": round(r["score"],4),
                    "title": (r["payload"] or {}).get("title",""),
                    "time": (r["payload"] or {}).get("time",""),
                    "time_ts": (r["payload"] or {}).get("time_ts", "")
                }, flush=True)

    except Exception as e:
        state.search_results = []
        print("Error during vector DB search:", str(e), flush=True)
        state.add_debug("vector_db_error", str(e))

    return state

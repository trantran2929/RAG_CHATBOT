# modules/nodes/retriever.py
from modules.core.state import GlobalState
from datetime import datetime, timedelta
import pytz

RRF_K = 60  


def _rrf(rank: int | None) -> float:
    """RRF score cho 1 rank, rank bắt đầu từ 1."""
    if not rank:
        return 0.0
    return 1.0 / (RRF_K + rank)


def retrieve_documents(state: GlobalState, max_hours: int = 48) -> GlobalState:
    """
    NHIỆM VỤ:
    - Nhận search_results_dense & search_results_sparse từ vector_db.
    - Áp dụng RRF fusion:
        rrf_score = RRF(dense_rank) + RRF(sparse_rank)
    - Chuẩn hóa thành `state.retrieved_docs` để reranker/prompt_builder dùng.
    """
    if getattr(state, "route_to", "") not in ["rag", "hybrid"]:
        state.retrieved_docs = []
        state.context = ""
        state.llm_status = "retriever_skipped"
        state.add_debug("retriever", "skipped_non_rag_route")
        return state

    dense_hits = getattr(state, "search_results_dense", []) or []
    sparse_hits = getattr(state, "search_results_sparse", []) or []

    if not dense_hits and not sparse_hits:
        state.retrieved_docs = []
        state.context = ""
        state.llm_status = "retriever_no_docs"
        state.add_debug("retriever", "no_search_results")
        return state

    fused: dict[tuple, dict] = {}

    def _key(hit):
        # key = (id, time_ts) để tránh trùng
        return (hit.get("id"), hit.get("time_ts", 0))

    # Gom thông tin dense
    for h in dense_hits:
        k = _key(h)
        doc = fused.get(k, {})
        doc.update(
            {
                "id": h.get("id"),
                "title": h.get("title", ""),
                "time": h.get("time", ""),
                "time_ts": h.get("time_ts", 0),
                "url": h.get("url", ""),
                "content": (h.get("content") or "").strip(),
            }
        )
        doc["dense_rank"] = h.get("rank")
        doc["dense_score"] = h.get("score", 0.0)
        fused[k] = doc

    # Gom thông tin sparse
    for h in sparse_hits:
        k = _key(h)
        doc = fused.get(k, {})
        base_content = doc.get("content", "") or ""
        # ghép nội dung sparse vào (tránh trùng lặp lớn)
        merged_content = (base_content + " " + (h.get("content") or "")).strip()
        doc.update(
            {
                "id": h.get("id"),
                "title": h.get("title", ""),
                "time": h.get("time", ""),
                "time_ts": h.get("time_ts", 0),
                "url": h.get("url", ""),
                "content": merged_content,
            }
        )
        doc["sparse_rank"] = h.get("rank")
        doc["sparse_score"] = h.get("score", 0.0)
        fused[k] = doc

    docs = []
    for _, d in fused.items():
        time_ts = d.get("time_ts", 0)

        dense_rank = d.get("dense_rank")
        sparse_rank = d.get("sparse_rank")

        rrf_score = _rrf(dense_rank) + _rrf(sparse_rank)

        docs.append(
            {
                "id": d.get("id"),
                "title": d.get("title", ""),
                "time": d.get("time", ""),
                "time_ts": time_ts,
                "url": d.get("url", ""),
                "content": (d.get("content") or "")[:2500],
                "score": round(rrf_score, 6),   # dùng RRF làm score tổng
                "rrf_score": round(rrf_score, 6),
                "dense_rank": dense_rank,
                "sparse_rank": sparse_rank,
                "dense_score": d.get("dense_score"),
                "sparse_score": d.get("sparse_score"),
            }
        )

    docs.sort(key=lambda x: (x.get("rrf_score", 0.0), x.get("time_ts", 0)), reverse=True)

    state.retrieved_docs = docs
    state.context = ""  
    state.llm_status = "retriever_success"

    state.add_debug("retriever_docs", len(docs))
    state.add_debug(
        "retriever_top_titles",
        [d["title"] for d in docs[:5]],
    )

    return state

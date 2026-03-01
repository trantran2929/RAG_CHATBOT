from modules.core.state import GlobalState
from modules.utils.services import reranker_services

def rerank_documents(state: GlobalState, top_k: int | None = None) -> GlobalState:
    """
    Rerank danh sách state.retrieved_docs dựa trên user_query.
    - Nếu reranker không khả dụng → giữ nguyên thứ tự.
    - Để prompt_builder tự build lại context theo thứ tự mới, ta xóa state.context.
    """
    if getattr(state, "route_to", "") not in ["rag", "hybrid"]:
        state.add_debug("reranker", "skipped_non_rag_route")
        state.llm_status = "reranker_skipped"
        return state

    docs = getattr(state, "retrieved_docs", []) or []
    if not docs:
        state.add_debug("reranker", "no_docs")
        state.llm_status = "reranker_no_docs"
        return state

    try:
        inputs = [
            {
                "id": d.get("id"),
                "title": d.get("title", ""),
                "content": d.get("content", ""),
                "score": d.get("score", 0.0),
                "time_ts": d.get("time_ts", 0),
                "url": d.get("url", ""),
            }
            for d in docs
        ]

        ranked = reranker_services.rerank(state.user_query, inputs)
        ranked.sort(key=lambda x: x.get("rerank_score", x.get("score", 0.0)), reverse=True)

        if top_k is not None and top_k > 0:
            ranked = ranked[:top_k]

        state.retrieved_docs = ranked
        state.context = ""  
        state.llm_status = "reranker_success"
        state.add_debug("reranker_status", "applied")
        state.add_debug("reranker_docs", len(ranked))
        return state

    except Exception as e:
        state.add_debug("reranker_error", str(e))
        state.llm_status = "reranker_error"
        return state

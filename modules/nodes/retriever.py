from modules.utils.debug import add_debug_info

def retrieve_documents(state, top_k: int = 2):
    """
    Chuyển đổi state.search_results -> state.retrieved_docs
    thành danh sách dict {id, score, text, version, timestamp}
    """
    search_results = getattr(state, "search_results", None)
    if not search_results:
        state.retrieved_docs = []
        add_debug_info(state, "retrieved_docs_count", 0)
        return state

    # Lấy top_k kết quả
    top_results = search_results[:top_k]

    docs = []
    for r in top_results:
        payload = getattr(r, "payload", {}) or {}

        # Ưu tiên lấy text từ các field khác nhau
        text = (
            payload.get("text")
            or payload.get("content")
            or payload.get("doc")
            or ""
        )

        docs.append({
            "id": getattr(r, "id", None),
            "score": getattr(r, "score", None),
            "text": text,
            "version": payload.get("version"),
            "timestamp": payload.get("timestamp"),
        })

    state.retrieved_docs = docs
    add_debug_info(state, "retrieved_docs_count", len(docs))
    return state
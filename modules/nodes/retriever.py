from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info

def retrieve_documents(state: GlobalState, max_chars: int = 1500) -> GlobalState:
    """
    Lấy documents (payload) từ Qdrant search_results
    và tổng hợp lại context cho LLM.
    """
    if not state.search_results:
        state.retrieved_docs = []
        state.context = ""
        add_debug_info(state, "retriever", "Không có tài liệu nào được tìm thấy")
        return state

    docs = []
    context_parts = []

    for hit in state.search_results:
        payload = hit.get("payload", {}) or {}
        score = hit.get("score", 0.0)

        content = payload.get("content") or ""
        if not content or len(content) < 20:  # fallback nếu content quá ngắn
            content = payload.get("summary") or ""

        title = payload.get("title", "")
        link = payload.get("link", "")
        time = payload.get("time", "")

        if content:
            docs.append({
                "id": hit.get("id"),
                "score": score,
                "title": title,
                "time": time,
                "link": link,
                "content": content
            })
            # meta source: có thể bỏ score nếu không muốn lộ cho LLM
            context_parts.append(
                f"[{title} | {time}]\n{content}\n(Source: {link})"
            )

    # Ghép context
    context_text = "\n\n".join(context_parts)

    # Giới hạn độ dài nhưng ưu tiên không cắt giữa đoạn
    if len(context_text) > max_chars:
        truncated = []
        total = 0
        for part in context_parts:
            if total + len(part) > max_chars:
                break
            truncated.append(part)
            total += len(part)
        context_text = "\n\n".join(truncated) + "\n...\n[Context truncated]"

    # cập nhật state
    state.retrieved_docs = docs
    state.context = context_text

    add_debug_info(state, "retriever_docs", len(docs))
    add_debug_info(state, "retriever_context_len", len(state.context))
    add_debug_info(state, "retriever_titles", [d["title"] for d in docs])

    return state

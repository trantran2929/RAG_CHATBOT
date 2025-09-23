from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info


def retrieve_documents(state: GlobalState) -> GlobalState:
    """
    Lấy documents (payload) từ Qdrant search_results
    và tổng hợp lại context.
    """
    if not state.search_results:
        state.retrieved_docs = []
        state.context = ""
        add_debug_info(state, "retriever", "Không có tài liệu nào được tìm thấy")
        return state

    docs = []
    context_parts = []

    for hit in state.search_results:
        payload = hit.get("payload", {})
        content = payload.get("content", "")

        if content:
            docs.append(payload)
            context_parts.append(content)

    # cập nhật state
    state.retrieved_docs = docs
    state.context = "\n".join(context_parts)

    add_debug_info(state, "retriever_docs", len(docs))
    add_debug_info(state, "retriever_context_len", len(state.context))

    return state

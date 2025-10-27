from modules.utils.services import embedder_services
from modules.core.state import GlobalState

def embed_query(state: GlobalState) -> GlobalState:
    """
    - Sinh embedding cho truy vấn (dense + sparse)
    - Chỉ chạy khi route_to = 'rag' hoặc 'hybrid'
    """
    if getattr(state, "route_to", "") not in ["rag", "hybrid"]:
        state.add_debug("embed_status", "skipped_non_rag_route")
        state.query_embedding = None
        state.llm_status = "embed_skipped"
        return state

    query = state.processed_query or state.user_query
    if not query:
        state.query_embedding = None
        state.add_debug("embed_status", "empty_query")
        state.llm_status = "embed_failed"
        return state

    try:
        dense_vec = embedder_services.encode_dense([query])[0]
    except Exception as e:
        dense_vec = None
        state.add_debug("embed_dense_error", str(e))

    sparse_vec = None
    try:
        sparse_vec = embedder_services.encode_sparse([query])
        state.add_debug("embed_sparse_status", "ok")
    except Exception as e:
        sparse_vec = None
        state.add_debug("embed_sparse_error", str(e))

    if dense_vec is None and sparse_vec is None:
        state.query_embedding = None
        state.llm_status = "embed_failed"
        state.add_debug("embed_status", "failed_both_none")
        return state

    state.query_embedding = {
        "dense_vector": dense_vec,
        "sparse_vector": sparse_vec,
    }
    state.llm_status = "embed_success"
    state.add_debug("embed_status", "success")
    state.add_debug("embed_query_text", query)
    state.add_debug("embed_route", getattr(state, "route_to", "unknown"))
    state.add_debug("embed_dense_dim", len(dense_vec) if dense_vec else 0)

    return state

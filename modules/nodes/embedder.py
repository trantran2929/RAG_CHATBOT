from modules.utils.services import embedder_services
from modules.core.state import GlobalState

def embed_query(state: GlobalState)->GlobalState:
    query = state.processed_query or state.user_query
    if not query:
        state.query_embedding = None
        state.add_debug("embed_status", "empty_query")
        return state

    try: 
        dense_vec = embedder_services.encode_dense([query])[0]
        state.add_debug("embed_dense_dim", len(dense_vec))

        try:
            sparse_vec = embedder_services.encode_sparse([query])
            state.add_debug("embed_sparse_status", "ok")
        except Exception:
            sparse_vec = None
            state.add_debug("embed_sparse_error", str(e))

        state.query_embedding = {
            "dense_vector": dense_vec,
            "sparse_vector": sparse_vec,
        }

        state.llm_status = "embed_success"
        state.add_debug("embed_status", "success")
    
    except Exception as e:
        state.query_embedding = None
        state.llm_status = "embed_failed"
        state.add_debug("embed_error", str(e))
        
    return state
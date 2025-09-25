from modules.utils.services import embedder_services
from modules.core.state import GlobalState

def embed_query(state: GlobalState)->GlobalState:
    query = state.processed_query or state.user_query
    if not query:
        state.query_embedding = None
        return state

    dense_vec = embedder_services.encode_dense([query])[0]

    state.query_embedding = {
        "dense_vector": dense_vec,
    }
    return state

from modules.core.state import GlobalState
from modules.utils.services import qdrant_services
from modules.utils.debug import add_debug_info
from qdrant_client import models

def search_vector_db(state: GlobalState, top_k: int = 5, alpha: float = 0.7) -> GlobalState:
    """
    Hybrid search (dense + sparse + bit) trong Qdrant.
    """
    if not state.query_embedding:
        add_debug_info(state, "vector_db", "No embedding")
        state.search_results = []
        return state

    dense_vec = state.query_embedding.get("dense_vector")

    try:
        results = qdrant_services.client.search(
            collection_name=qdrant_services.collection_name,
            prefetch=[
                models.Prefetch(
                    query={"vector": dense_vec},
                    using="dense_vector"
                ),
                models.Prefetch(
                    query={"text": state.user_query},
                    using="bm25"
                ),
                models.Prefetch(
                    query={"vector": dense_vec},
                    using="binary"      #quantized
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        state.search_results = [
            {"id": r.id, "score": r.score, "payload": r.payload} for r in results
        ]
        add_debug_info(state, "vector_db_results", f"Found {len(results)}")
    except Exception as e:
        state.search_results = []
        add_debug_info(state, "vector_db_error", str(e))

    return state

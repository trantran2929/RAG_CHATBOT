from modules.core.state import GlobalState
from modules.utils.services import qdrant_services
from modules.utils.debug import add_debug_info
from qdrant_client import models

def search_vector_db(state: GlobalState, top_k: int = 5, alpha: float = 0.7) -> GlobalState:
    """
    Hybrid search (dense + sparse) trong Qdrant.
    """
    if not state.query_embedding:
        add_debug_info(state, "vector_db", "No embedding")
        state.search_results = []
        return state

    dense_vec = state.query_embedding.get("dense_vector")
    sparse_vec = state.query_embedding.get("sparse_vector")

    try:
        results = qdrant_services.client.search(
            collection_name=qdrant_services.collection_name,
            query = models.FusionQuery(
                queries=[
                    models.Query(
                        vector=dense_vec,
                        top_k=top_k,
                        weight=alpha,
                        using="dense_vector"
                    ),
                    models.Query(
                        vector={"sparse_vector": sparse_vec} if sparse_vec else {},
                        top_k=top_k,
                        weight=1 - alpha,
                        using="sparse_vector"
                    )
                ],
                fusion=models.Fusion.WeightedSum
            ),
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

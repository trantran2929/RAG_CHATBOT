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
        print("No query embedding available.", flush=True)
        return state

    dense_vec = state.query_embedding.get("dense_vector")
    sparse_vec = state.query_embedding.get("sparse_vector")

    dense_results, sparse_results = [], []

    try:
        if dense_vec is not None:
            dense_results = qdrant_services.client.search(
                collection_name=qdrant_services.collection_name,
                query_vector=("dense_vector", dense_vec),
                limit=top_k,
                with_payload=True,
            )
            print("Dense search results:", dense_results, flush=True)
            for r in dense_results:
                print({
                    "id": r.id,
                    "score": r.score,
                    "title": r.payload.get("title",""),
                    "payload_preview": str(r.payload)[:200]  
                }, flush=True)
        if sparse_vec:
            try:
                sparse_results = qdrant_services.client.search(
                    collection_name=qdrant_services.collection_name,
                    query_vector=("sparse_vector", sparse_vec),
                    limit=top_k,
                    with_payload=True,
                )
                print("Sparse search results:", sparse_results, flush=True)
                for r in sparse_results:
                    print({
                        "id": r.id,
                        "score": r.score,
                        "title": r.payload.get("title",""),
                        "payload_preview": str(r.payload)[:200]  
                    }, flush=True)
            except Exception as e:
                print("Error during sparse search:", str(e), flush=True)
            
        if dense_results and sparse_results:
            combined = {}
            for r in dense_results:
                combined[r.id] = {
                    "id": r.id,
                    "score": r.score * alpha,
                    "payload": r.payload or {}
                }
            for r in sparse_results:
                if r.id in combined:
                    combined[r.id]["score"] += r.score * (1 - alpha)
                else:
                    combined[r.id] = {
                        "id": r.id,
                        "score": r.score * (1 - alpha),
                        "payload": r.payload
                    }
            state.search_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]

        else:
            # fallback lấy dense hoặc sparse
            base_results = dense_results if dense_results else sparse_results
            state.search_results = [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload
                } for r in base_results
            ]
        

        print("=== RAW SEARCH RESULTS FROM QDRANT ===", flush=True)
        for r in state.search_results:
            print({
                "id": r["id"],
                "score": r["score"],
                "title": r["payload"].get("title",""),
                "payload_preview": str(r["payload"])[:200] 
            }, flush=True)

        add_debug_info(state, "vector_db_results", f"Found {len(state.search_results)} results")
    except Exception as e:
        state.search_results = []
        print("Error during vector DB search:", str(e), flush=True)
        add_debug_info(state, "vector_db_error", str(e))

    return state

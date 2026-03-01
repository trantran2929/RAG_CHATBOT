# -*- coding: utf-8 -*-
"""
Bản đã sửa:
- Không import response_node ở top-level → tránh circular import
- Tất cả các hàm cần response_node sẽ import bên trong (lazy import)
- Giữ nguyên toàn bộ logic debug/test node
"""

from modules.core.state import GlobalState
from modules.nodes.cache import load_cache, save_cache
from modules.nodes.processor import processor_query
from modules.nodes.router import route_intent
from modules.nodes.embedder import embed_query
from modules.nodes.vector_db import search_vector_db
from modules.nodes.retriever import retrieve_documents
from modules.nodes.reranker import rerank_documents
from modules.nodes.prompt_builder import build_prompt


def add_debug_info(state, key: str, value):
    """Safe helper to attach debug info into state.debug_info"""
    try:
        if hasattr(state, "debug_info") and isinstance(state.debug_info, dict):
            state.debug_info[key] = value
    except Exception:
        pass


def _sep(title: str):
    print("\n" + "=" * 80)
    print(f"🧩 {title}")
    print("=" * 80 + "\n")


# ========================================================================
# TEST PROCESSOR
# ========================================================================
def test_processor_node(user_query: str) -> GlobalState:
    _sep("TEST PROCESSOR NODE")
    state = GlobalState(user_query=user_query, debug=True)

    state = load_cache(state)
    print(f"- History từ cache: {len(state.conversation_history)} messages")

    state = processor_query(state)

    print(f"user_query      : {state.user_query}")
    print(f"processed_query : {state.processed_query}")
    print(f"lang            : {state.lang}")
    print(f"intent          : {state.intent}")
    print(f"tickers         : {state.tickers}")
    print(f"time_filter     : {state.time_filter}")
    print(f"is_greeting     : {state.is_greeting}")
    print(f"cache_key       : {state.cache_key}")
    print(f"debug_info      : {state.debug_info}")

    return state


# ========================================================================
# TEST ROUTER
# ========================================================================
def test_router_node(user_query: str) -> GlobalState:
    _sep("TEST ROUTER NODE")
    state = GlobalState(user_query=user_query, debug=True)

    state = load_cache(state)
    state = processor_query(state)
    state = route_intent(state)

    print(f"user_query  : {state.user_query}")
    print(f"intent      : {state.intent}")
    print(f"tickers     : {state.tickers}")
    print(f"route_to    : {state.route_to}")
    print(f"api_type    : {state.api_type}")
    print(f"api_response: {state.api_response}")
    print(f"llm_status  : {state.llm_status}")
    print(f"debug_info  : {state.debug_info}")

    return state


# ========================================================================
# TEST EMBEDDER
# ========================================================================
def test_embedder_node(user_query: str) -> GlobalState:
    _sep("TEST EMBEDDER NODE")
    state = GlobalState(user_query=user_query, debug=True)

    state = load_cache(state)
    state = processor_query(state)
    state = route_intent(state)

    if state.route_to not in ["rag", "hybrid"]:
        print("⚠ Ép route_to='rag' để test embedder")
        state.route_to = "rag"

    state = embed_query(state)

    print(f"user_query        : {state.user_query}")
    print(f"route_to          : {state.route_to}")
    print(f"llm_status        : {state.llm_status}")

    if state.query_embedding:
        dv = state.query_embedding.get("dense_vector")
        sv = state.query_embedding.get("sparse_vector")
        print(f"dense_dim         : {len(dv) if dv is not None else 0}")
        print(f"sparse_vector_len : {len(sv[0]['indices']) if sv else 0}")
    else:
        print("❌ Không có query_embedding")

    print(f"debug_info        : {state.debug_info}")
    return state


# ========================================================================
# TEST VECTOR DB
# ========================================================================
def test_vector_db_node(user_query: str, top_k: int = 5) -> GlobalState:
    _sep("TEST VECTOR_DB NODE")
    state = GlobalState(user_query=user_query, debug=True)

    state = load_cache(state)
    state = processor_query(state)
    state = route_intent(state)
    if state.route_to not in ["rag", "hybrid"]:
        state.route_to = "rag"

    state = embed_query(state)
    state = search_vector_db(state, top_k=top_k)

    print(f"user_query          : {state.user_query}")
    print(f"intent              : {state.intent}")
    print(f"route_to            : {state.route_to}")
    print(f"llm_status          : {state.llm_status}")
    print(f"#dense_hits         : {len(state.search_results_dense)}")
    print(f"#sparse_hits        : {len(state.search_results_sparse)}")

    return state


# ========================================================================
# TEST RETRIEVER
# ========================================================================
def test_retriever_node(user_query: str, top_k: int = 5) -> GlobalState:
    _sep("TEST RETRIEVER NODE")
    state = GlobalState(user_query=user_query, debug=True)

    state = load_cache(state)
    state = processor_query(state)
    state = route_intent(state)
    if state.route_to not in ["rag", "hybrid"]:
        state.route_to = "rag"

    state = embed_query(state)
    state = search_vector_db(state, top_k=top_k)
    state = retrieve_documents(state)

    print(f"llm_status         : {state.llm_status}")
    print(f"#retrieved_docs    : {len(state.retrieved_docs)}")

    return state


# ========================================================================
# TEST RERANKER
# ========================================================================
def test_reranker_node(user_query: str, top_k: int = 5) -> GlobalState:
    _sep("TEST RERANKER NODE")
    state = GlobalState(user_query=user_query, debug=True)

    state = load_cache(state)
    state = processor_query(state)
    state = route_intent(state)
    if state.route_to not in ["rag", "hybrid"]:
        state.route_to = "rag"

    state = embed_query(state)
    state = search_vector_db(state, top_k=top_k)
    state = retrieve_documents(state)
    state = rerank_documents(state, top_k=top_k)

    print(f"llm_status          : {state.llm_status}")
    print(f"#retrieved_docs     : {len(state.retrieved_docs)}")

    return state


# ========================================================================
# TEST PROMPT BUILDER
# ========================================================================
def test_prompt_builder_node(user_query: str, top_k: int = 5) -> GlobalState:
    _sep("TEST PROMPT_BUILDER NODE")
    state = GlobalState(user_query=user_query, debug=True)

    state = load_cache(state)
    state = processor_query(state)
    state = route_intent(state)

    if state.route_to not in ["rag", "hybrid"]:
        state.route_to = "rag"

    state = embed_query(state)
    state = search_vector_db(state, top_k=top_k)
    state = retrieve_documents(state)
    state = rerank_documents(state, top_k=top_k)
    state = build_prompt(state)

    print(f"intent             : {state.intent}")
    print(f"route_to           : {state.route_to}")
    print(f"prompt_len         : {len(state.prompt)}")

    return state


# ========================================================================
# TEST RESPONSE NODE — Lazy import tại đây!
# ========================================================================
def test_response_node(user_query: str, top_k: int = 5) -> GlobalState:
    _sep("TEST RESPONSE NODE (FULL PATH)")
    from modules.nodes.response_generator import response_node  # LAZY IMPORT

    state = GlobalState(user_query=user_query, debug=True)

    state = load_cache(state)
    state = processor_query(state)
    state = route_intent(state)

    if state.route_to in ["rag", "hybrid"]:
        state = embed_query(state)
        state = search_vector_db(state, top_k=top_k)
        state = retrieve_documents(state)
        state = rerank_documents(state, top_k=top_k)
        state = build_prompt(state)

    state = response_node(state)
    state = save_cache(state)

    print(f"intent        : {state.intent}")
    print(f"final_answer  : {state.final_answer[:400]}")

    return state


# ========================================================================
# DEBUG FULL PIPELINE — Lazy import response_node
# ========================================================================
def debug_full_pipeline(user_query: str) -> GlobalState:
    _sep("DEBUG FULL PIPELINE")
    from modules.nodes.response_generator import response_node  # LAZY IMPORT

    state = GlobalState(user_query=user_query, debug=True)

    state = load_cache(state)
    print(f"[load_cache] history = {len(state.conversation_history)}")

    state = processor_query(state)
    print(f"[processor] intent={state.intent}")

    state = route_intent(state)
    print(f"[router] route_to={state.route_to}")

    if state.route_to in ["rag", "hybrid"]:
        state = embed_query(state)
        state = search_vector_db(state, top_k=5)
        state = retrieve_documents(state)
        state = rerank_documents(state, top_k=5)
        state = build_prompt(state)

    state = response_node(state)
    print(f"[response_node]")

    state = save_cache(state)
    print(f"[save_cache]")

    print("\n>>> FINAL ANSWER")
    print("-" * 80)
    print(state.final_answer)
    print("-" * 80)

    return state


# Chạy CLI
if __name__ == "__main__":
    while True:
        try:
            q = input("\nNhập câu hỏi (enter để thoát): ").strip()
        except EOFError:
            break
        if not q:
            break
        debug_full_pipeline(q)

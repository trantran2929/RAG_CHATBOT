from langgraph.graph import StateGraph, START, END
from .state import GlobalState
from modules.nodes.cache import load_cache, save_cache
from modules.nodes.processor import processor_query
from modules.nodes.embedder import embed_query
from modules.nodes.vector_db import search_vector_db
from modules.nodes.retriever import retrieve_documents
from modules.nodes.prompt_builder import build_prompt
from modules.nodes.response_generator import response_node
from modules.utils.debug import debug_summary_node
from modules.nodes.router import route_intent

def build_graph():
    workflow = StateGraph(GlobalState)

    workflow.add_node("load_cache", load_cache)
    workflow.add_node("processor", processor_query)
    workflow.add_node("router", route_intent)
    workflow.add_node("embedder", embed_query)
    workflow.add_node("vector_db", search_vector_db)
    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("prompt_builder", build_prompt)
    workflow.add_node("response_node", response_node)
    workflow.add_node("debug_summary", debug_summary_node)
    workflow.add_node("save_cache", save_cache)

    workflow.add_edge(START, "load_cache")
    workflow.add_edge("load_cache", "processor")
    workflow.add_edge("processor", "router")
    workflow.add_conditional_edges(
        "router", 
        lambda state: getattr(state, "route_to", "rag"),
        {
            "api":"response_node",
            "rag": "embedder"
        }
    )
    workflow.add_edge("embedder", "vector_db")
    workflow.add_edge("vector_db", "retriever")
    workflow.add_edge("retriever", "prompt_builder")
    workflow.add_edge("prompt_builder", "response_node")
    workflow.add_edge("response_node", "debug_summary")
    workflow.add_edge("debug_summary", "save_cache")
    workflow.add_edge("save_cache", END)

    return workflow.compile()

from langgraph.graph import StateGraph, END, START

from .state import GlobalState
from modules.nodes.cache import load_memory, save_memory
from modules.nodes.processor import processor_query
from modules.nodes.embedder import embed_query
from modules.nodes.vector_db import search_and_store
from modules.nodes.retriever import retrieve_documents
from modules.nodes.prompt_builder import build_prompt
from modules.nodes.generator_response import generate_response
from modules.nodes.response import chatbot_response


def build_graph():
    """
    Xây dựng đồ thị xử lý RAG pipeline bằng langgraph.
    """
    workflow = StateGraph(GlobalState)

    # Định nghĩa các node
    workflow.add_node("load_memory", load_memory)                # lấy hội thoại cũ từ Redis
    workflow.add_node("processor", processor_query)              # xử lý query
    workflow.add_node("embedder", embed_query)                   # embedding
    workflow.add_node("vector_db", search_and_store)             # search KB
    workflow.add_node("retriever", retrieve_documents)           # lấy snippets
    workflow.add_node("prompt_builder", build_prompt)            # build prompt (thêm context + history)
    workflow.add_node("response_generator", generate_response)   # gọi LLM
    workflow.add_node("save_memory", save_memory)                # lưu lại hội thoại vào Redis
    workflow.add_node("response", chatbot_response)              # trả lời cuối

    # Nối các bước theo workflow
    workflow.add_edge(START, "load_memory")
    workflow.add_edge("load_memory", "processor")
    # bỏ qua retrieval nếu small-talk
    workflow.add_conditional_edges(
        "processor",
        lambda s: "prompt_builder" if getattr(s, "skip_retrieval", False) else "embedder",
        {"prompt_builder": "prompt_builder", "embedder": "embedder"}
    )
    workflow.add_edge("embedder", "vector_db")
    workflow.add_edge("vector_db", "retriever")
    workflow.add_edge("retriever", "prompt_builder")
    workflow.add_edge("prompt_builder", "response_generator")
    workflow.add_edge("response_generator", "save_memory")
    workflow.add_edge("save_memory", "response")
    workflow.add_edge("response", END)

    return workflow.compile()
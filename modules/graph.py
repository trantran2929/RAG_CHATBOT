from langgraph.graph import StateGraph, START, END
from .state import State
def build_graph(response_fn):
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot_response", response_fn)
    graph_builder.add_edge(START, "chatbot_response")
    graph_builder.add_edge("chatbot_response", END)
    return graph_builder.compile()
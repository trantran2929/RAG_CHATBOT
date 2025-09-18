import streamlit as st
import os
import sys
# Thêm folder gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from modules.graph import build_graph

from langchain_core.messages import HumanMessage, AIMessage

from modules.processor import detect_language
from modules.retriever import retrieve_context
from modules.prompt_builder import build_prompt
from modules.llm import generate_response
from modules.vector_db import save_message, load_history, clear_history
from modules.state import State

# Hàm xử lý tin nhắn cho Gradio 
def chatbot_response(state:State): 
    user_input = state.messages[-1].content
    lang = detect_language(user_input)
    history = load_history(top_k=6) 
    
     #RAG: Truy xuất dữ liệu nền
    context = "\n".join(retrieve_context(user_input, top_k=3))

    prompt = build_prompt(lang, context, history, user_input)
    response = generate_response(prompt)
        
    # Lưu dạng role-based
    save_message("User", user_input)
    save_message("assistant", response)
    return {"messages": [AIMessage(content=response)]}


graph = build_graph(chatbot_response)

st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Chatbot AI")

history = load_history()
# Hiển thị lịch sử
for msg in history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input chat
if user_input := st.chat_input("Nhập tin nhắn..."):
    out = graph.invoke({"messages": [HumanMessage(content=user_input)]})
    response = out["messages"][-1].content

    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        st.write(response)

if st.button("Xóa lịch sử"):
    clear_history()
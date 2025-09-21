import streamlit as st
import os
import sys
import json
import uuid

# Thêm folder gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.core.graph import build_graph
from modules.core.state import GlobalState
from modules.nodes.services import redis_services

st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Chatbot AI")

graph = build_graph()

# Khởi tạo session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def display_history():
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Load hội thoại từ Redis 
key = f"chat:{st.session_state.session_id}"
cached = redis_services.client.get(key)
if cached:
    try:
        st.session_state.chat_history = json.loads(cached)
    except Exception:
        pass

display_history()

# Input chat
if user_input := st.chat_input("Nhập tin nhắn..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    # Tạo state mới cho query
    state = GlobalState(
        raw_query=user_input,
        session_id=st.session_state.session_id,
        messages=st.session_state.chat_history
    )

    # Chạy pipeline
    state = graph.invoke(state)
    response = state.get("final_answer", "Xin lỗi, không có câu trả lời")

    # Cập nhật session_state
    # st.session_state.chat_history.append({"role": "assistant", "content": response})

    redis_services.client.set(key, json.dumps(st.session_state.chat_history, ensure_ascii=False), ex=3600)
    display_history()

# Nút xóa hội thoại
if st.button("Xóa lịch sử"):
    st.session_state.chat_history = []
    redis_services.client.delete(key)
    st.rerun()
import streamlit as st
import os
import sys
import json
import uuid

# Thêm folder gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.core.graph import build_graph
from modules.core.state import GlobalState
from modules.utils.services import redis_services

# UI CONFIG
st.set_page_config(page_title="Chatbot AI", layout="wide")
st.title("🤖 Chatbot AI")

graph = build_graph()

# Khởi tạo session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# SIDEBAR 
with st.sidebar:
    st.subheader("⚙️ Tùy chọn")
    st.write(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🗑️ Xóa toàn bộ lịch sử trong Redis"):
        try:
            redis_services.client.flushdb()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Đã xóa toàn bộ dữ liệu!")
            st.info("Trang sẽ tự động reload...")
            import time
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi khi xóa dữ liệu: {str(e)}")

    if st.button("🔍 Debug Redis Keys"):
        try:
            all_keys = list(redis_services.client.scan_iter("*"))
            if all_keys:
                st.write("**Keys trong Redis:**")
                for key in all_keys:
                    if isinstance(key, bytes):
                        key = key.decode("utf-8")
                    st.code(key)
            else:
                st.write("Redis trống!")
        except Exception as e:
            st.error(f"Lỗi khi debug: {str(e)}")

# LOAD CACHE 
key = f"chat:{st.session_state.session_id}"
cached = redis_services.client.get(key)
if cached:
    try:
        loaded_data = json.loads(cached)
        if isinstance(loaded_data, list):
            st.session_state.chat_history = loaded_data
        elif isinstance(loaded_data, dict) and "history" in loaded_data:
            st.session_state.chat_history = loaded_data.get("history", [])
        else:
            st.session_state.chat_history = []
    except Exception as e:
        st.warning(f"Không thể load cache: {str(e)}")
        st.session_state.chat_history = []

# ========== HELPERS ==========
def render_sources(sources):
    """Hiển thị danh sách nguồn tham khảo."""
    if not sources:
        return
    with st.expander("📚 Nguồn tham khảo"):
        for src in sources:
            title = src.get("title", "Không tiêu đề")
            link = src.get("link", "")
            time = src.get("time", "")
            score = src.get("score", None)

            meta = f"{title} ({time})"
            if score is not None:
                meta += f" | score={score:.3f}"

            if link:
                st.markdown(f"- [{meta}]({link})")
            else:
                st.markdown(f"- {meta}")

def display_history():
    """Render lịch sử hội thoại."""
    for msg in st.session_state.chat_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                render_sources(msg.get("sources"))

# ========== MAIN UI ==========
display_history()

if user_input := st.chat_input("💬 Nhập tin nhắn của bạn..."):
    user_msg = {"role": "user", "content": user_input}
    st.session_state.chat_history.append(user_msg)

    state = GlobalState(
        user_query=user_input,
        session_id=st.session_state.session_id,
        messages=st.session_state.chat_history.copy()
    )

    with st.chat_message("assistant"):
        with st.spinner("Đang tạo câu trả lời..."):
            try:
                state = graph.invoke(state)
                response = state.get("final_answer", "❌ Xin lỗi, không có câu trả lời.")

                # Chỉ giữ metadata sources
                sources = []
                if getattr(state, "retrieved_docs", None):
                    for doc in state.retrieved_docs:
                        sources.append({
                            "title": doc.get("title", ""),
                            "link": doc.get("link", ""),
                            "time": doc.get("time", ""),
                            "score": doc.get("score", None),
                        })

                assistant_msg = {"role": "assistant", "content": response}
                if sources:
                    assistant_msg["sources"] = sources

                st.session_state.chat_history.append(assistant_msg)

                # Hiển thị output + nguồn
                st.markdown(response)
                render_sources(sources)

                # Lưu cache
                try:
                    redis_services.client.set(
                        key,
                        json.dumps(st.session_state.chat_history, ensure_ascii=False),
                        ex=3600
                    )
                except Exception as e:
                    st.warning(f"Không thể lưu vào Redis: {str(e)}")

            except Exception as e:
                st.error(f"Lỗi khi xử lý: {str(e)}")
                error_msg = {"role": "assistant", "content": f"❌ Lỗi: {str(e)}"}
                st.session_state.chat_history.append(error_msg)

    st.rerun()

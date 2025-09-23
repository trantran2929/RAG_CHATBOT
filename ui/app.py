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
            # Xóa tất cả keys Redis
            redis_services.client.flushdb()
            
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.success("Đã xóa toàn bộ dữ liệu!")
            st.info("Trang sẽ tự động reload...")
            
            # Delay một chút rồi rerun để đảm bảo Redis đã clear
            import time
            time.sleep(0.5)
            st.rerun()
            
        except Exception as e:
            st.error(f"Lỗi khi xóa dữ liệu: {str(e)}")

    # Debug button 
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

# load cache sau khi khởi tạo session
key = f"chat:{st.session_state.session_id}"
# LOAD CACHE 
cached = redis_services.client.get(key)
if cached:
    try:
        loaded_data = json.loads(cached)
        
        # Case 1: Format từ Streamlit (list)
        if isinstance(loaded_data, list):
            st.session_state.chat_history = loaded_data
            
        # Case 2: Format từ LangGraph (dict với history/final_answer)
        elif isinstance(loaded_data, dict):
            if "history" in loaded_data:
                st.session_state.chat_history = loaded_data.get("history", [])
            else:
                # Reset về empty nếu format không đúng
                st.session_state.chat_history = []
        else:
            st.session_state.chat_history = []
            
    except Exception as e:
        st.warning(f"Không thể load cache: {str(e)}")
        st.session_state.chat_history = []

# HIỂN THỊ LỊCH SỬ CHAT
def display_history():
    for msg in st.session_state.chat_history:
        if not isinstance(msg, dict):
            continue
            
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Nếu assistant, chỉ show final_answer
        if role == "assistant":
            # Clean up response format
            if "Assistant:" in content:
                content = content.split("Assistant:")[-1]
            if "User:" in content:
                content = content.split("User:")[0]
            content = content.strip()
            
        with st.chat_message(role):
            st.markdown(content)

display_history()

# INPUT & XỬ LÝ CHAT 
if user_input := st.chat_input("💬 Nhập tin nhắn của bạn..."):
    # Hiển thị tin nhắn user ngay lập tức
    user_msg = {"role": "user", "content": user_input}
    st.session_state.chat_history.append(user_msg)
    
    # Update key nếu cần
    key = f"chat:{st.session_state.session_id}"

    # Tạo state cho pipeline
    state = GlobalState(
        user_query=user_input,
        session_id=st.session_state.session_id,
        messages=st.session_state.chat_history.copy()  # Copy để tránh reference issues
    )

    # Loading khi chatbot trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tạo câu trả lời..."):
            try:
                state = graph.invoke(state)
                response = state.get("final_answer", "❌ Xin lỗi, không có câu trả lời.")
                st.markdown(response)
                
                # Lưu vào lịch sử
                assistant_msg = {"role": "assistant", "content": response}
                st.session_state.chat_history.append(assistant_msg)
                
                # Lưu Redis với error handling
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
                # Vẫn lưu lỗi vào lịch sử để user biết
                error_msg = {"role": "assistant", "content": f"❌ Lỗi: {str(e)}"}
                st.session_state.chat_history.append(error_msg)

    st.rerun()
import json
from .services import redis_services
from modules.utils.debug import add_debug_info

def chatbot_response(state):
    """
    Cập nhật hội thoại với câu trả lời mới vào Redis.
    - Chỉ lưu messages (user + assistant)
    - Redis TTL = 3600s (1h)
    """
    key = f"chat:{state.session_id}"

    # final_answer có thể là string hoặc dict {"answer": "..."}
    if isinstance(state.final_answer, dict):
        answer_text = state.final_answer.get("answer", "Xin lỗi, không có câu trả lời")
    else:
        answer_text = state.final_answer or "Xin lỗi, không có câu trả lời"

    # Thêm message mới vào state
    state.add_message("assistant", answer_text)

    try:
        # Lưu vào Redis: chỉ messages
        redis_services.client.set(
            key,
            json.dumps(state.messages, ensure_ascii=False),
            ex=3600
        )
        add_debug_info(state, "redis_status", "saved_success")
    except Exception as e:
        add_debug_info(state, "redis_status", f"error: {e}")

    return state
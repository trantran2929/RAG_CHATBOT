import json
from .services import redis_services


def load_memory(state):
    """
    Load lịch sử hội thoại từ Redis bằng state.conversation_history.
    """
    if not getattr(state, "session_id", None):
        state.session_id = "default_session"

    redis_client = redis_services.client
    key = f"chat:{state.session_id}"

    cached = redis_client.get(key)
    if cached:
        try:
            state.conversation_history = json.loads(cached)
        except Exception:
            state.conversation_history = []
    else:
        state.conversation_history = []

    return state


def save_memory(state, TTL=3600):
    """
    Thêm state.messages và final_answer vào conversation_history và lưu.
    """
    if not getattr(state, "session_id", None):
        state.session_id = "default_session"

    redis_client = redis_services.client
    key = f"chat:{state.session_id}"

    history = getattr(state, "conversation_history", [])
    # Thêm tin nhắn hiện tại vào state.messages
    for m in state.messages:
        if m not in history:
            history.append(m)

    if state.final_answer:
        history.append({"role":"assistant", "content": state.final_answer})
    
    try:
        redis_client.set(
            key,
            json.dumps(history, ensure_ascii=False),
            ex=TTL
        )
    except Exception as e:
        print(f"[memory.save] Redis error: {e}")
    state.conversation_history = history
    return state


import json
from modules.core.state import GlobalState
from modules.utils.services import redis_services

CACHE_TTL = 3600

def load_cache(state: GlobalState) -> GlobalState:
    """Load conversation history từ Redis nếu có."""
    if not state.session_id:
        return state

    cache_key = f"chat:{state.session_id}"
    cached = redis_services.client.get(cache_key)

    if cached:
        try:
            data = json.loads(cached)
            if isinstance(data, list):
                state.conversation_history = data
            elif isinstance(data, dict) and "history" in data:
                state.conversation_history = data.get("history", [])
            else:
                state.conversation_history = []
        except json.JSONDecodeError:
            state.conversation_history = []
    else:
        state.conversation_history = []

    state.from_cache = bool(state.conversation_history)
    return state


def save_cache(state: GlobalState) -> GlobalState:
    """Save conversation history vào Redis."""
    if not state.session_id:
        return state

    cache_key = f"chat:{state.session_id}"

    cleaned_history = []
    for msg in state.conversation_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "").lower()
        content = (msg.get("content") or msg.get("query") or msg.get("answer") or "").strip()
        if role in ["user", "assistant"] and content:
            cleaned_history.append({"role": role, "content": content})
    redis_services.client.setex(
        cache_key,
        CACHE_TTL,
        json.dumps(cleaned_history, ensure_ascii=False)
    )
    return state

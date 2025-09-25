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
    redis_services.client.setex(
        cache_key,
        CACHE_TTL,
        json.dumps(state.conversation_history, ensure_ascii=False)
    )
    return state

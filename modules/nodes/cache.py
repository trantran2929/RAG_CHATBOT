import json
from modules.core.state import GlobalState
from .services import redis_services

CACHE_TTL = 3600

def load_cache(state: GlobalState) -> GlobalState:
    """
    Load lịch sử hội thoại từ Redis vào state.conversation_history.
    Sử dụng consistent key format với Streamlit app.
    """
    if not state.session_id:
        return state
    
    cache_key = f"chat:{state.session_id}"
    cached = redis_services.client.get(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            # Handle cả 2 format:
            # Format 1: Streamlit format - list of messages
            if isinstance(data, list):
                state.conversation_history = data
                state.from_cache = True
                
            # Format 2: LangGraph format - dict với history
            elif isinstance(data, dict) and "history" in data:
                state.conversation_history = data.get("history", [])
                state.from_cache = True
                
            else:
                state.conversation_history = []
                state.from_cache = False
                
        except json.JSONDecodeError:
            state.conversation_history = []
            state.from_cache = False
    else:
        state.conversation_history = []
        state.from_cache = False
    
    return state


def save_cache(state: GlobalState) -> GlobalState:
    """
    Lưu conversation_history vào Redis cache.
    (Không append thêm vì đã được update ở response node)
    """
    if not state.session_id:
        return state
    
    cache_key = f"chat:{state.session_id}"
    # cached_data = {
    #     "history": state.conversation_history,
    #     "final_answer": state.final_answer
    # }
    redis_services.client.setex(
        cache_key,
        CACHE_TTL,
        json.dumps(state.conversation_history, ensure_ascii=False)
    )
    return state

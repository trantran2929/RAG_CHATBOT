import json, os, errno
from modules.core.state import GlobalState
from modules.utils.services import redis_services

LOCAL_CACHE_DIR = os.getenv("LOCAL_CACHE_DIR", "./.cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

def _local_path(session_id: str) -> str:
    return os.path.join(LOCAL_CACHE_DIR, f"chat_{session_id}.json")

def _safe_load_local(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass
    return None


def _safe_save_local(path: str, data):
    """Ghi file cache local, nuốt lỗi để không chặn flow."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        if e.errno != errno.EEXIST:
            # nuốt lỗi im lặng
            pass
    except Exception:
        pass

def load_cache(state: GlobalState) -> GlobalState:
    """Load conversation history từ Redis nếu có, nếu không dùng file cục bộ."""
    if not state.session_id:
        return state

    cache_key = f"chat:{state.session_id}"

    cached = None
    try:
        if getattr(redis_services, "client", None) is not None:
            cached = redis_services.client.get(cache_key)
    except Exception:
        cached = None

    if not cached:
        data = _safe_load_local(_local_path(state.session_id))
    else:
        try:
            data = json.loads(cached)
        except Exception:
            data = None

    if isinstance(data, list):
        state.conversation_history = data
    elif isinstance(data, dict) and "history" in (data or {}):
        state.conversation_history = data.get("history", [])
    else:
        state.conversation_history = []

    state.from_cache = bool(state.conversation_history)
    return state


def save_cache(state: GlobalState) -> GlobalState:
    """Save conversation history vào Redis, lỗi thì lưu vào file cục bộ."""
    if not state.session_id:
        return state

    cache_key = f"chat:{state.session_id}"

    cleaned_history = []
    for msg in state.conversation_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "").lower()
        content = (
            msg.get("content")
            or msg.get("query")
            or msg.get("answer")
            or ""
        ).strip()
        if role in ["user", "assistant"] and content:
            cleaned_history.append({"role": role, "content": content})

    wrote = False
    try:
        if getattr(redis_services, "client", None) is not None:
            redis_services.client.set(
                cache_key,
                json.dumps(cleaned_history, ensure_ascii=False),
                ex=7 * 24 * 3600,
            )
            wrote = True
    except Exception:
        wrote = False

    if not wrote:
        _safe_save_local(_local_path(state.session_id), cleaned_history)

    return state
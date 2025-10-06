import datetime
import pytz
from modules.utils.logger import log_debug, log_info, log_error  # ✅ import logger helpers

# Hàm ghi thông tin debug chi tiết vào state + logger
def add_debug_info(state, key, value):
    """
    Ghi thông tin debug vào state.debug_info và log ra file (qua logger).
    Có kiểm tra an toàn và giới hạn độ dài log.
    """
    try:
        # Ghi vào state nếu có add_debug()
        if hasattr(state, "add_debug") and callable(state.add_debug):
            state.add_debug(key, value)
        elif hasattr(state, "debug_info") and isinstance(state.debug_info, dict):
            state.debug_info[key] = value

        # In log qua logger
        val_str = str(value)
        if len(val_str) > 400:
            val_str = val_str[:400] + "... [truncated]"

        if getattr(state, "debug", False):
            log_debug(f"{key}: {val_str}", state)
        else:
            # Ghi nhẹ vào file mà không spam console
            log_info(f"{key}: {val_str}")

    except Exception as e:
        log_error(f"[add_debug_info] Lỗi khi ghi debug key={key}", e)

# Tóm tắt pipeline (API / RAG / Greeting)

def debug_summary_node(state):
    """
    Node tóm tắt toàn bộ pipeline (API / RAG / Greeting).
    Ghi vào debug_info và log qua logger.
    """
    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.datetime.now(vn_tz).strftime("%d/%m/%Y %H:%M:%S")

    route = getattr(state, "route", "unknown")
    api_type = getattr(state, "api_type", None)
    llm_status = getattr(state, "llm_status", "unknown")
    n_docs = len(getattr(state, "retrieved_docs", []))
    prompt_len = len(getattr(state, "prompt", "") or "")
    response_len = len(getattr(state, "final_answer", "") or "")
    time_filter = getattr(state, "time_filter", None)
    tickers = getattr(state, "tickers", [])
    latency_ms = round((datetime.datetime.now().timestamp() - state.timestamp) * 1000, 2)

    summary = {
        "timestamp": now,
        "session_id": state.session_id,
        "route": route,
        "api_type": api_type,
        "llm_status": llm_status,
        "retrieved_docs": n_docs,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "tickers": tickers,
        "time_filter": time_filter,
        "latency_ms": latency_ms,
    }

    # Ghi debug info vào state
    add_debug_info(state, "summary", summary)

    # Log tóm tắt pipeline qua logger
    if getattr(state, "debug", False):
        log_debug("\n=== [PIPELINE SUMMARY] ===", state)
        log_debug(f"🕒 {now}", state)
        log_debug(f"📍 Route: {route} ({api_type or 'N/A'})", state)
        log_debug(f"💬 Query: {state.user_query}", state)
        log_debug(f"📚 Retrieved Docs: {n_docs}", state)
        log_debug(f"🧠 Prompt len: {prompt_len} | Response len: {response_len}", state)
        if tickers:
            log_debug(f"🔖 Tickers: {tickers}", state)
        if time_filter:
            start, end = time_filter
            log_debug(f"🗓️ Time Filter: {start} → {end}", state)
        log_debug(f"✅ LLM Status: {llm_status}", state)
        log_debug(f"⚡ Latency: {latency_ms} ms", state)
        log_debug("===========================\n", state)
    else:
        log_info(f"[SUMMARY] route={route}, llm={llm_status}, docs={n_docs}, latency={latency_ms}ms")

    return state

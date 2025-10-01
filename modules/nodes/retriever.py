from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info
from datetime import datetime, timedelta

def debug_print(state: GlobalState, key: str, value):
    add_debug_info(state, key, value)
    print(f"DEBUG: {key} = {value}", flush=True) 

def retrieve_documents(state: GlobalState, max_chars: int = 1500, max_hours: int = 48) -> GlobalState:
    """
    Lấy documents (payload) từ Qdrant search_results
    và tổng hợp lại context cho LLM.
    """
    if not state.search_results:
        state.retrieved_docs = []
        state.context = ""
        print("No search results found.", flush=True)
        add_debug_info(state, "retriever", "Không có tài liệu nào được tìm thấy")
        return state

    docs = []
    context_parts = []

    now = datetime.utcnow()
    time_limit = now - timedelta(hours=max_hours)

    for hit in state.search_results:
        payload = hit.get("payload", {}) or {}
        score = hit.get("score", 0.0)

        content = (payload.get("content") or payload.get("summary") or payload.get("text") or "").strip()
        if not content or len(content) < 20:
            print("Skipping document due to insufficient content length.")
            continue

        title = payload.get("title", "")
        url = payload.get("url", "") or payload.get("link", "") or ""
        time = payload.get("time", "")
        time_dt = None
        if time:
            try:
                time_dt = datetime.strptime(time, "%d-%m-%Y %H:%M:%S")
            except Exception:
                time_dt = None
        debug_print(state, f"retriever_hit_{hit.get('id')}", {
            "score": score,
            "title": title,
            "time": time,
            "url": url,
            "content_len": len(content)
        })
        if content and (time_dt is None or time_dt >= time_limit):
            docs.append({
                "id": hit.get("id"),
                "score": score,
                "title": title,
                "time": time,
                "url": url,
                "content": content
            })
            context_parts.append(
                f"[{title} | {time}]\n{content}\n(Source: {url})"
            )

    # Ghép context
    context_text = "\n\n".join(context_parts)

    # Giới hạn độ dài nhưng ưu tiên không cắt giữa đoạn
    if len(context_text) > max_chars:
        truncated = []
        total = 0
        for part in context_parts:
            if total + len(part) > max_chars:
                break
            truncated.append(part)
            total += len(part)
        context_text = "\n\n".join(truncated) + "\n...\n[Context truncated]"

    # cập nhật state
    state.retrieved_docs = docs
    state.context = context_text

    print("=== RETRIEVED DOCS ===", flush=True)
    for d in state.retrieved_docs:
        print({
            "id": d.get("id"),
            "score": d.get("score"),
            "title": d.get("title"),
            "time": d.get("time"),
            "url": d.get("url"),
            "content_preview": d.get("content")[:200]  
        }, flush=True)

    print("=== FINAL CONTEXT PASSED TO PROMPT ===", flush=True)
    print(state.context[:500], flush=True)  

    debug_print(state, "retriever_docs", len(docs))
    debug_print(state, "retriever_context_len", len(state.context))
    debug_print(state, "retriever_titles", [d["title"] for d in docs])


    return state

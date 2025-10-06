from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info
from datetime import datetime, timedelta
from modules.api.time_api import get_datetime_context
import pytz

def retrieve_documents(state: GlobalState, max_chars: int = 1500, max_hours: int = 48) -> GlobalState:
    """
    Lấy documents (payload) từ Qdrant search_results và tổng hợp lại context cho LLM.
    - Nếu query là API (state.api_response != None) thì bỏ qua
    - Nếu có yếu tố thời gian thì thêm ngữ cảnh thời gian vào đầu context
    - Giới hạn context theo max_chars và max_hours.
    """

    if getattr(state,"api_response",None):
        state.retrieved_docs = []
        state.context = ""
        add_debug_info(state, "retriever", "Skipped")
        return state
    
    if not state.search_results:
        state.retrieved_docs = []
        state.context = ""
        add_debug_info(state, "retriever", "Không có tài liệu nào được tìm thấy")
        return state

    docs = []
    context_parts = []
    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(vn_tz)
    min_ts = int((now - timedelta(hours=max_hours)).timestamp())

    for hit in state.search_results:
        payload = hit.get("payload", {}) or {}
        score = float(hit.get("score", 0.0)) if hit.get("score") is not None else 0.0

        content = (payload.get("content") or payload.get("summary") or payload.get("text") or "").strip()
        if not content or len(content) < 20:
            continue

        title = payload.get("title", "")
        url = payload.get("url", "") or payload.get("link", "") or ""
        time = payload.get("time", "")
        time_ts = payload.get("time_ts",None)

        if time_ts and isinstance(time_ts, (int, float)) and time_ts < min_ts:
            continue

        if content:
            docs.append({
                "id": hit.get("id"),
                "score": score,
                "title": title,
                "time": time,
                "time_ts": time_ts,
                "url": url,
                "content": content
            })
            context_parts.append(
                f"[{title} | {time}]\n{content}\n(Source: {url})"
            )

        add_debug_info(state, f"retriever_hit_{hit.get('id')}", {
            "score": round(score, 4),
            "title": title,
            "time": time,
            "time_ts": time_ts,
            "url": url,
            "content_len": len(content)
        })

    context_text = "\n\n".join(context_parts)

    if len(context_text) > max_chars:
        truncated = []
        total = 0
        for part in context_parts:
            if total + len(part) > max_chars:
                break
            truncated.append(part)
            total += len(part)
        context_text = "\n\n".join(truncated) + "\n...\n[Context truncated]"

    state.retrieved_docs = docs
    state.context = context_text.strip()

    if getattr(state,"debug", False):
        print("*** RETRIEVED DOCS ***", flush=True)
        for d in state.retrieved_docs:
            print({
                "id": d.get("id"),
                "score": d.get("score"),
                "title": d.get("title"),
                "time": d.get("time"),
                "time_ts": d.get("time_ts"),
                "url": d.get("url"),
                "content_preview": d.get("content")[:200]  
            }, flush=True)

        print("*** FINAL CONTEXT PASSED TO PROMPT ***", flush=True)
        print(state.context[:500], flush=True)  

    add_debug_info(state, "retriever_docs", len(docs))
    add_debug_info(state, "retriever_context_len", len(state.context))
    add_debug_info(state, "retriever_titles", [d["title"] for d in docs])

    if getattr(state, "debug", False):
        print(f"[Retriever] Retrieved {len(docs)} docs", flush=True)
        for d in docs:
            print(f"  - {d['title']} | {d['time']} | score={d['score']:.3f}", flush=True)

    return state

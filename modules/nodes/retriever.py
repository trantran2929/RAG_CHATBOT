from modules.core.state import GlobalState
from datetime import datetime, timedelta
import pytz
from collections import defaultdict

def retrieve_documents(state: GlobalState, max_chars: int = 12000, max_hours: int = 48) -> GlobalState:
    """
    Lấy documents (payload) từ Qdrant search_results và tổng hợp lại context cho LLM.
    - Nếu query là API (state.api_response != None) thì bỏ qua
    - Nếu có yếu tố thời gian thì thêm ngữ cảnh thời gian vào đầu context
    - Giới hạn context theo max_chars và max_hours.
    """

    if getattr(state, "route_to", "") not in ["rag", "hybrid"]:
        state.retrieved_docs = []
        state.context = ""
        state.add_debug("retriever", "Skipped")
        state.llm_status = "retriever_skipped"
        return state
    
    if not state.search_results:
        state.retrieved_docs = []
        state.context = ""
        state.add_debug("retriever", "Không có tài liệu nào được tìm thấy")
        state.llm_status = "retriever_no_docs"
        return state

    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(vn_tz)
    min_ts = int((now - timedelta(hours=max_hours)).timestamp())

    # Gom chunk theo url hoặc title
    grouped = defaultdict(list)
    for hit in state.search_results:
        url = (hit.get("url") or "").strip()
        title = (hit.get("title") or "").strip()
        key = url if url else title
        grouped[key].append(hit)
    
    merged_docs = []
    for key, group in grouped.items():
        best = max(group, key=lambda x: x.get("score",0.0))
        merged_content = " ".join(
            (d.get("content") or d.get("summary") or d.get("title") or "")
            for d in group
            if d.get("content")
        ).strip()

        time_ts = best.get("time_ts")

        if time_ts and isinstance(time_ts, (int, float)) and time_ts < min_ts:
            continue

        merged_docs.append({
            "id": best.get("id"),
            "score": best.get("score", 0.0),
            "title": best.get("title", ""),
            "time": best.get("time", ""),
            "time_ts": best.get("time_ts"),
            "url": best.get("url", ""),
            "content": merged_content[:2000]
        })

    merged_docs.sort(key=lambda x: (x["time_ts"] or 0, -x["score"]))

    context_parts = []
    total_len = 0
    for d in merged_docs:
        part = f"[{d['title']} | {d['time']}] | score={round(d['score'],3)}]\n{d['content']}\n(Source: {d['url']})"
        if total_len + len(part) > max_chars:
            break
        context_parts.append(part)
        total_len += len(part)

    context_text = "\n\n".join(context_parts)
    state.retrieved_docs = merged_docs
    state.context = context_text.strip()

    if getattr(state,"debug", False):
        print("*** RETRIEVED DOCS ***", flush=True)
        for d in state.retrieved_docs:
            print({
                "id": d.get("id"),
                "score": round(d.get("score", 0.0), 3),
                "title": d.get("title"),
                "time": d.get("time"),
                "time_ts": d.get("time_ts"),
                "url": d.get("url"),
                "content_preview": d.get("content")[:800]  
            }, flush=True)

        print("*** FINAL CONTEXT PASSED TO PROMPT ***", flush=True)
        print(state.context[:800], flush=True)  
        print(f"\n[Retriever] Retrieved {len(state.retrieved_docs)} docs", flush=True)

    # state.add_debug("retriever_docs", len(merged_docs))
    # state.add_debug("retriever_context_len", len(state.context))
    # state.add_debug("retriever_titles", [d["title"] for d in merged_docs])
    state.llm_status = "retriever_success"

    return state

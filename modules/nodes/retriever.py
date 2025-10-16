from modules.core.state import GlobalState
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
from modules.api.time_api import get_datetime_context

def retrieve_documents(state: GlobalState, max_chars: int = 12000, max_hours: int = 48) -> GlobalState:
    """
    Lấy documents (payload) từ Qdrant search_results và tổng hợp lại context cho LLM.
    - Nếu query là API (state.api_response != None) thì bỏ qua
    - Nếu có yếu tố thời gian thì thêm ngữ cảnh thời gian vào đầu context
    - Giới hạn context theo max_chars và max_hours.
    - Gom nhóm theo url/title, loại trùng
    - Thêm nhãn thời gian cập nhật vào đầu context để model biết mốc dữ liệu
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
    use_time_filter = bool(getattr(state, "time_filter", None))

    # Gom chunk theo url hoặc title
    grouped = defaultdict(list)
    for hit in state.search_results:
        title = (hit.get("title") or "").strip()
        time_ts = hit.get("time_ts") or 0 
        key = f"{title}_{time_ts}"
        grouped[key].append(hit)
    
    merged_docs = []
    for key, group in grouped.items():
        best = max(group, key=lambda x: x.get("rerank_score",x.get("score",0.0)))
        title = best.get("title", f"Tài liệu {best.get('id', '')}").strip()
        time_ts = best.get("time_ts")

        if not use_time_filter and time_ts < min_ts:
            continue
        merged_content = " ".join(
            list(dict.fromkeys(
                (
                    d.get("content") 
                    or d.get("summary") 
                    or d.get("title") 
                    or ""
                ).strip()
                for d in group 
            ))
        )

        merged_docs.append({
            "id": best.get("id"),
            "score": round(best.get("score", 0.0), 4),
            "rerank_score": round(best.get("rerank_score", best.get("score", 0.0)), 4),
            "title": title,
            "time": best.get("time", ""),
            "time_ts": time_ts,
            "url": best.get("url", ""),
            "content": merged_content[:2500]
        })

    merged_docs.sort(key=lambda x: (-(x.get("time_ts") or 0), x.get("rerank_score", -x.get("score", 0))))

    if getattr(state, "intent", "") == "market":
        market_keywords = ["chứng khoán", "thị trường", "vnindex", "vn30", "vĩ mô", "dòng tiền", "ngành"]
        merged_docs = [d for d in merged_docs if any(kw in d["content"].lower() for kw in market_keywords)] or merged_docs

    context_parts = []
    total_len = 0
    min_docs = 3
    for i,d in enumerate(merged_docs):
        part = (
            f"[{d['title']} | {d['time']}] "
            f"(score={d['score']}, rerank={d['rerank_score']})\n"
            f"{d['content']}"
        )

        if len(part) > max_chars * 0.5:
            part = part[: int(max_chars * 0.5)] + "..."
        
        context_parts.append(part)
        total_len += len(part)
        if total_len > max_chars and i>=min_docs-1:
            break

    if not context_parts:
        context_parts.append("(Không tìm thấy nội dung tin tức phù hợp.)")

    context_text = "\n\n".join(context_parts).strip()

    context_header = f"[Tin tức cập nhập đến: {get_datetime_context()}]\n\n"
    context_text = context_header + context_text

    state.retrieved_docs = merged_docs
    state.context = context_text

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
                "content_preview": d.get("content")[:1000]  
            }, flush=True)

        print("*** FINAL CONTEXT PASSED TO PROMPT ***", flush=True)
        print(state.context[:800], flush=True)  
        print(f"\n[Retriever] Retrieved {len(state.retrieved_docs)} docs", flush=True)

    # state.add_debug("retriever_docs", len(merged_docs))
    # state.add_debug("retriever_context_len", len(state.context))
    # state.add_debug("retriever_titles", [d["title"] for d in merged_docs])
    state.llm_status = "retriever_success"

    return state

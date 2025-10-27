from modules.core.state import GlobalState
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
from modules.api.time_api import get_datetime_context

def retrieve_documents(state: GlobalState, max_chars: int = 12000, max_hours: int = 48) -> GlobalState:
    if getattr(state, "route_to", "") not in ["rag", "hybrid"]:
        state.retrieved_docs = []
        state.context = ""
        state.llm_status = "retriever_skipped"
        state.add_debug("retriever", "skipped_non_rag_route")
        return state

    if not state.search_results:
        state.retrieved_docs = []
        state.context = ""
        state.llm_status = "retriever_no_docs"
        state.add_debug("retriever", "no_search_results")
        return state

    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(vn_tz)
    min_ts = int((now - timedelta(hours=max_hours)).timestamp())
    use_time_filter = bool(getattr(state, "time_filter", None))

    grouped = defaultdict(list)
    for hit in state.search_results:
        title = (hit.get("title") or "").strip()
        time_ts = hit.get("time_ts") or 0
        key = f"{title}_{time_ts}"
        grouped[key].append(hit)

    merged_docs = []
    for key, group in grouped.items():
        best = max(group, key=lambda x: x.get("rerank_score", x.get("score", 0.0)))
        title = best.get("title", f"Tài liệu {best.get('id', '')}").strip()
        time_ts = best.get("time_ts") or 0

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

    merged_docs.sort(
        key=lambda x: (-(x.get("time_ts") or 0), x.get("rerank_score", -x.get("score", 0)))
    )

    if getattr(state, "intent", "") == "market":
        market_keywords = [
            "chứng khoán", "thị trường", "vnindex", "vn30", "vĩ mô", "dòng tiền", "ngành"
        ]
        filtered = [
            d for d in merged_docs
            if any(kw in d["content"].lower() for kw in market_keywords)
        ]
        if filtered:
            merged_docs = filtered

    context_parts = []
    total_len = 0
    min_docs = 3

    for i, d in enumerate(merged_docs):
        part = (
            f"[{d['title']} | {d['time']}] "
            f"(score={d['score']}, rerank={d['rerank_score']})\n"
            f"{d['content']}"
        )

        if len(part) > max_chars * 0.5:
            part = part[: int(max_chars * 0.5)] + "..."

        context_parts.append(part)
        total_len += len(part)
        if total_len > max_chars and i >= min_docs - 1:
            break

    if not context_parts:
        context_parts.append("(Không tìm thấy nội dung tin tức phù hợp.)")

    context_text = "\n\n".join(context_parts).strip()

    context_header = f"[Tin tức cập nhật đến: {get_datetime_context()}]\n\n"
    context_text = context_header + context_text

    state.retrieved_docs = merged_docs
    state.context = context_text
    state.llm_status = "retriever_success"

    state.add_debug("retriever_docs", len(merged_docs))
    state.add_debug("retriever_context_len", len(state.context))
    state.add_debug(
        "retriever_titles",
        [d["title"] for d in merged_docs[:5]]
    )

    return state

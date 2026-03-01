# modules/nodes/prompt_builder.py
from modules.core.state import GlobalState
from modules.api.time_api import get_datetime_context

SYSTEM_INSTRUCTION = """
Bạn là trợ lý AI tài chính Việt Nam, chuyên phân tích xu hướng thị trường, cổ phiếu và tin tức.
1. Tóm tắt tin tức chứng khoán (cổ phiếu, chỉ số, ngành nghề) Việt Nam. 
2. Phân tích xu hướng thị trường dựa trên dữ liệu và tin tức Việt Nam.
3. Nếu người dùng hỏi về mã cổ phiếu, phải tập trung phân tích đúng mã đó (ví dụ: TCB → Techcombank), không nhầm sang mã khác.
4. Cung cấp thông tin mã cổ phiếu ở Việt Nam.
"""

CONSTRAINTS = """
- Trả lời bằng TIẾNG VIỆT, ngắn gọn, tự nhiên.
- KHÔNG in mã code, KHÔNG trả JSON thô.
- KHÔNG nói “Tôi không phải chuyên gia tài chính”.
- Giữ nguyên các ký hiệu như VNINDEX, VCB, VN30.
- KHÔNG khuyến nghị mua/bán, KHÔNG kêu gọi đầu tư.
- Chỉ trả lời nội dung cuối cùng dưới mục **Assistant:**, đừng lặp lại hướng dẫn.
"""

INTENT_TASKS = {
    "market": "Phân tích xu hướng cổ phiếu/thị trường bằng cách kết hợp dữ liệu API (giá cổ phiếu, thị trường) và tin tức gần nhất.",
    "stock": "Tóm tắt thông tin giá/hành vi cổ phiếu hiện tại hoặc lịch sử.",
    "weather": "Cung cấp thông tin thời tiết tại địa điểm được hỏi.",
    "time": "Cung cấp thông tin thời gian hiện tại hoặc mốc thời gian cụ thể.",
    "rag": "Phân tích dựa trên ngữ cảnh tin tức hoặc dữ liệu trong vector DB.",
    "greeting": "Phản hồi chào hỏi hoặc giới thiệu khả năng trợ lý tài chính.",
}


def _build_context_from_docs(
    docs,
    max_docs: int = 5,
    max_context_chars: int = 4000,
) -> str:
    """
    Convert retrieved_docs → đoạn context text gọn gàng cho LLM.
    Ưu tiên:
    - Tin mới (do retriever + reranker đã sort).
    - Không vượt quá max_context_chars.
    """
    if not docs:
        return ""

    parts = [f"[Tin tức cập nhật đến: {get_datetime_context()}]"]
    total_len = len(parts[0])
    used = 0

    for i, d in enumerate(docs, start=1):
        if used >= max_docs:
            break

        title = (d.get("title") or "").strip()
        ts = (d.get("time") or "").strip()
        body = (d.get("content") or "").strip().replace("\n", " ")
        if len(body) > 1000:
            body = body[:1000].strip() + "..."

        score = d.get("score") or d.get("rrf_score") or 0.0
        line = f"{i}. [{ts}] {title} (score={round(score, 4)})\n   {body}"

        if total_len + len(line) > max_context_chars and used >= 2:
            break

        parts.append(line)
        total_len += len(line)
        used += 1

    return "\n".join(parts).strip()


def build_prompt(state: GlobalState, max_context_chars: int = 5000) -> GlobalState:
    """
    NHIỆM VỤ:
    - Chỉ chạy khi route_to in {"rag", "hybrid"}.
    - Lấy `retrieved_docs` (đã RRF + rerank CrossEncoder nếu có) → build context.
    - Ghép: Instruction + Constraints + Bối cảnh thời gian + Context + API + History + Task.
    - Đặt prompt vào state.prompt để response_node dùng gọi LLM.
    """
    if getattr(state, "route_to", "") not in ["rag", "hybrid"]:
        state.prompt = ""
        state.llm_status = "prompt_skipped"
        state.add_debug("prompt_builder", "skipped_non_rag_route")
        return state

    if getattr(state, "conversation_history", None) is None:
        state.conversation_history = []

    lang = getattr(state, "lang", "vi")
    intent = getattr(state, "intent", "rag")
    task = INTENT_TASKS.get(intent, "Phản hồi thông tin theo từng loại intent.")
    user_input = (state.user_query or state.processed_query or "").strip()

    docs_list = getattr(state, "retrieved_docs", []) or []
    context = _build_context_from_docs(
        docs_list,
        max_docs=5,
        max_context_chars=max_context_chars - 1000,  
    )

    if not context:
        context = (getattr(state, "context", "") or "").strip()

    if context and len(context) > max_context_chars:
        context = context[:max_context_chars].strip() + "..."

    prompt_parts = [
        f"## Instruction:\n{SYSTEM_INSTRUCTION.strip()}",
        f"## Constraints:\n{CONSTRAINTS.strip()}",
        f"## Bối cảnh thời gian:\n{get_datetime_context().strip()}\n",
    ]

    if context:
        prompt_parts.append(
            "## Retrieved Context:\n"
            "Dưới đây là một số tin tức / dữ kiện gần nhất. Hãy dùng nội dung này để mô tả "
            "bối cảnh thị trường và cổ phiếu, nhưng KHÔNG đưa ra khuyến nghị đầu tư:\n"
        )
        prompt_parts.append(context)
    else:
        prompt_parts.append(
            "## Retrieved Context:\n"
            "(Chưa thu thập được tin tức/bối cảnh liên quan.)"
        )

    if getattr(state, "api_response", None):
        prompt_parts.append("## Dữ liệu API:\n" + state.api_response.strip())

    history_msgs = (getattr(state, "conversation_history", []) or [])
    if history_msgs:
        lines = []
        for msg in history_msgs:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "").strip()
            if role in ["User", "Assistant"] and content:
                lines.append(f"**{role}:** {content}")
        if lines:
            prompt_parts.append("## Conversation History:\n" + "\n".join(lines))

    prompt_parts.append(f"## Task Type:\nIntent: {intent}\nMô tả: {task}\n")
    prompt_parts.append(f"## Task Input:\n**User:** {user_input}\n")

    if lang == "vi":
        lang_instruction = (
            "- Luôn trả lời bằng TIẾNG VIỆT tự nhiên.\n"
            "- Chỉ xuất phần trả lời cho người dùng, đừng lặp lại hướng dẫn."
        )
    else:
        lang_instruction = (
            f"- Trả lời bằng ngôn ngữ '{lang}'.\n"
            "- Chỉ xuất phần trả lời cuối cùng."
        )

    state.prompt = "\n\n".join(
        [
            lang_instruction,
            "\n".join(prompt_parts),
            "## Task Output:\n**Assistant:**",
        ]
    ).strip()

    state.llm_status = "prompt_built_success"
    state.add_debug("prompt_intent", intent)
    state.add_debug("prompt_status", "built_success")
    state.add_debug("prompt_has_context", bool(context))
    state.add_debug("prompt_context_len", len(context))

    return state

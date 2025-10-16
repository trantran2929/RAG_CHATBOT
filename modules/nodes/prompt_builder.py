from modules.core.state import GlobalState
from modules.api.time_api import get_datetime_context

SYSTEM_INSTRUCTION = """
Bạn là **trợ lý AI tài chính Việt Nam**, chuyên phân tích xu hướng thị trường, cổ phiếu và tin tức.
1. Khi intent = 'market' thì KẾT HỢP dữ liệu API (giá cổ phiếu, thị trường) với tin tức (Context).
2. Khi intent = 'stock' thì Trình bày thông tin giá cổ phiếu ngắn gọn.
3. KHÔNG khuyến nghị đầu tư tuyệt đối.
"""

CONSTRAINTS = """
- Luôn trả lời bằng TIẾNG VIỆT, ngắn gọn, tự nhiên.
- KHÔNG dùng mã code, KHÔNG in cấu trúc JSON.
- KHÔNG nói “Tôi không phải chuyên gia tài chính”.
- Giữ nguyên các ký hiệu như VNINDEX, VCB, VN30,...
"""

INTENT_TASKS = {
    "market": "Phân tích xu hướng cổ phiếu hoặc thị trường bằng cách kết hợp dữ liệu API và tin tức gần nhất.",
    "stock": "Cung cấp thông tin giá cổ phiếu hiện tại hoặc lịch sử.",
    "weather": "Cung cấp thông tin thời tiết tại địa điểm được hỏi.",
    "time": "Cung cấp thông tin thời gian hiện tại hoặc mốc thời gian cụ thể.",
    "rag": "Phân tích tổng hợp dựa trên ngữ cảnh tin tức hoặc dữ liệu trong vector DB.",
    "greeting": "Phản hồi chào hỏi hoặc giới thiệu khả năng trợ lý tài chính."
}


def build_prompt(state: GlobalState, max_context_chars: int = 5000) -> GlobalState:
    """
    Sinh prompt động cho chatbot đa-intent (tài chính, thời tiết, thời gian, tin tức, v.v.)
    """
    if getattr(state, "route_to", "") not in ["rag", "hybrid"]:
        state.prompt = ""
        state.add_debug("prompt_builder", "Skipped (no LLM required)")
        state.llm_status = "prompt_skipped"
        return state

    lang = getattr(state, "lang", "vi")
    intent = getattr(state, "intent", "rag")
    task = INTENT_TASKS.get(intent, "Phản hồi thông tin tổng quát.")
    user_input = (state.user_query or state.processed_query or "").strip()

    prompt_parts = []
    prompt_parts = [
        f"## Instruction:\n{SYSTEM_INSTRUCTION.strip()}",
        f"## Constraints:\n{CONSTRAINTS.strip()}",
        f"## Bối cảnh thời gian:\n{get_datetime_context().strip()}\n"
    ]
    context = getattr(state, "context", "").strip()
    if context:
        prompt_parts.append("\n## Retrieved Context:\nDưới đây là các tin tức gần nhất, hãy ưu tiên sử dụng chúng để phân tích thị trường:\n")
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "..."
        prompt_parts.append(context)
    else:
        prompt_parts.append("\n## Retrieved Context:\n(Tin tức gần đây chưa khả dụng hoặc không phù hợp.)")

    if getattr(state, "api_response", None):
        prompt_parts.append("## Dữ liệu API:\n" + state.api_response.strip())
    # prompt_parts.append("""
    #     ### Ví dụ minh họa cách kết hợp dữ liệu giá & tin tức:
    #     **Dữ liệu giá:**
    #     📊 VCB — Giá hiện tại: 62,500 VNĐ (-0.95%)

    #     **Tin tức:**
    #     VN-Index xuất hiện tín hiệu tạo đỉnh ngắn hạn, nhóm chứng khoán báo lỗ quý 3/2025.

    #     **Trả lời mẫu:**
    #     Hôm nay cổ phiếu VCB giảm 0.95%, diễn biến cùng xu hướng điều chỉnh chung của thị trường
    #     khi VN-Index có tín hiệu tạo đỉnh ngắn hạn. Một số thông tin tiêu cực từ nhóm chứng khoán
    #     gây áp lực chốt lời, khiến dòng tiền trở nên thận trọng. Xu hướng ngắn hạn: giảm nhẹ.
    #     """)

    history_msgs = (getattr(state, "conversation_history", []) or [])[-5:]
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
        lang_instruction = "- Luôn trả lời bằng TIẾNG VIỆT nếu có từ ngữ chuyên ngành (VNINDEX, VN30,..) thì giữ nguyên."
    else:
        lang_instruction = f"- Ưu tiên trả lời bằng ngôn ngữ '{lang}'."

    state.prompt = "\n\n".join([
        lang_instruction,
        "\n".join(prompt_parts),
        "## Task Output:",
        "**Assistant:**"
    ]).strip()

    state.llm_status = "prompt_built_success"
    state.add_debug("prompt_intent", intent)
    state.add_debug("prompt_status", "built_success")

    return state

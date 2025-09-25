from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info

SYSTEM_INSTRUCTION = """
Bạn là trợ lý AI chuyên về tài chính.
Nhiệm vụ:
1. Tóm tắt tin tức chứng khoán (cổ phiếu, chỉ số, ngành nghề).
2. Phân tích xu hướng thị trường dựa trên dữ liệu và tin tức.
3. Cung cấp thông tin thị trường crypto (BTC, ETH, Altcoin).
4. Đưa ra gợi ý đầu tư crypto theo hướng tham khảo, KHÔNG phải lời khuyên tuyệt đối.
5. Nếu người dùng chỉ chào hỏi → trả lời ngắn gọn, thân thiện.
6. Tuyệt đối KHÔNG sinh code block (```).
"""

def build_prompt(state: GlobalState, max_context_chars: int = 1800) -> GlobalState:
    lang = state.lang or "vi"
    prompt_parts = []

    # Role
    if state.role:
        prompt_parts.append(f"Role: {state.role}\n")
    # Conversation History
    history_msgs = state.conversation_history or state.messages
    if history_msgs:
        history_lines = []
        for msg in history_msgs:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user").capitalize()
            content = (
                msg.get("content")
                or msg.get("query")
                or msg.get("answer")
                or ""
            ).strip()
            if content:
                history_lines.append(f"{role}: {content}")
        if history_lines:
            prompt_parts.append("Conversation History:\n" + "\n".join(history_lines))
    examples = [
        ("Thị trường chứng khoán Mỹ hôm nay thế nào?",
         "Dow Jones tăng 1.2%, S&P500 giảm 0.3%. Nhóm công nghệ dẫn dắt xu hướng tăng."),
        ("BTC hôm nay giá bao nhiêu?",
         "BTC đang ở mức 70,200 USD, tăng 5% trong 24h.")
    ]
    examples_lines = []
    for input_text, output_text in examples:
        examples_lines.append(f"User: {input_text}\nAssistant: {output_text}")
    # if examples_lines:
    #     prompt_parts.append("Ví dụ tham khảo:\n" + "\n\n".join(examples_lines))
    # Retrieved Context
    retrieved_docs = getattr(state, "retrieved_docs", []) or []
    if retrieved_docs:
        context_parts = []
        for doc in retrieved_docs:
            text = (doc.get("content") or doc.get("summary") or "").strip()
            if text:
                title = doc.get("title", "")
                link = doc.get("link", "")
                time = doc.get("time", "")

                meta = " | ".join(filter(None, [title, time, f"Link: {link}" if link else None]))
                context_parts.append(f"[{meta}]\n{text}")

        context_text = "\n\n".join(context_parts)
        if len(context_text) > max_context_chars:
            context_text = context_text[:max_context_chars] + "...\n[Context truncated]"
        prompt_parts.append("Retrieved Context:\n" + context_text)
    else:
        prompt_parts.append(
            "Retrieved Context: Không có tài liệu liên quan " 
            "-> Hãy dựa vào kiến thức chung (LLM) để trả lời."
            )
    # User query
    user_input = (state.user_query or state.processed_query or "").strip()
    prompt_parts.append(f"User query: {user_input}")

    if lang == "vi":
        lang_instruction = "- Luôn trả lời bằng tiếng Việt."
    else:
        lang_instruction = f"- Ưu tiên trả lời bằng ngôn ngữ '{lang}'."

    joined_parts = "\n\n".join(prompt_parts)
    # Combine final system prompt
    system_prompt = f"""
        {SYSTEM_INSTRUCTION}
        {lang_instruction}

        {joined_parts}

        Assistant:
    """

    state.prompt = system_prompt.strip()

    print(state.prompt)

    # Debug
    add_debug_info(state, "prompt_length", len(system_prompt))
    add_debug_info(state, "prompt_preview", system_prompt[:400])
    add_debug_info(state, "history_count", len(history_msgs))
    add_debug_info(state, "context_docs_count", len(retrieved_docs))

    return state

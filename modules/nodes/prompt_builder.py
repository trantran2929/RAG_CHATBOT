from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info

SYSTEM_INSTRUCTION = """
Bạn là trợ lý AI. Nhiệm vụ:
1. Trả lời ngắn gọn, chính xác, dựa trên Context + Conversation history.
2. Nếu KHÔNG tìm thấy thông tin trong Context, hãy dựa vào kiến thức chung (LLM) để trả lời, KHÔNG được bỏ trống hoặc chỉ báo lỗi.
3. Không lặp lại câu hỏi hoặc thêm thông tin không liên quan.
4. Nếu người dùng chỉ chào hỏi (ví dụ: "xin chào", "hello") → chỉ trả lời lời chào ngắn gọn, thân thiện (ví dụ: "Xin chào! Tôi có thể giúp gì cho bạn?..."), KHÔNG sinh code.
5. Khi người dùng hỏi ngôn ngữ nào thì trả về ngôn ngữ đó (ví dụ: A: Hello, B: Hello! Can I help you,...).
6. Tuyệt đối KHÔNG sinh code block (``` hoặc ```python), trừ khi user query rõ ràng liên quan đến lập trình (ví dụ: code, python, function, viết chương trình...).
7. Nếu user yêu cầu “giải thích từng bước” hoặc “nêu chi tiết” → hãy phân tích theo bước. Nếu không thì chỉ trả lời đáp án cuối cùng.
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

    advanced_examples = [
        # Self-Ask
        ("Obama bao nhiêu tuổi khi trở thành tổng thống?",
         "Sub-question 1: Obama sinh năm nào? → 1961\n"
         "Sub-question 2: Ông làm tổng thống năm nào? → 2009\n"
         "Kết quả: 2009 - 1961 = 48\n"
         "Đáp án: 48 tuổi."),
        
        # COD
        ("Nếu có 3 quả táo, ăn 1 thì còn bao nhiêu?",
         "Bước 1: Ban đầu có 3 quả\n"
         "Bước 2: Ăn 1 quả\n"
         "Bước 3: 3 - 1 = 2\n"
         "Đáp án: 2."),
        
        # COK
        ("Tại sao nước biển mặn?",
         "Nước mưa hòa tan khoáng chất\n"
         "Ion muối ra biển\n"
         "Nước bay hơi, muối tích tụ\nĐáp án: Do muối tích tụ lâu dài."),
        
        # SG-ICL
        ("Dịch sang tiếng Việt: 'I love programming'",
         "Ví dụ tự tạo:\n - Hello → Xin chào\n - Good morning → Chào buổi sáng\n"
         "Áp dụng: 'I love programming' → 'Tôi yêu lập trình'.")
    ]

    short_examples = [
        ("Hà Nội là thủ đô của nước nào?", "Hà Nội là thủ đô của Việt Nam."),
        ("Thủ đô của Pháp là gì?", "Thủ đô của Pháp là Paris."),
        ("Ai là tổng thống Mỹ năm 2020?", "Tổng thống Mỹ năm 2020 là Donald Trump."),
        ("Tháp Eiffel nằm ở đâu?", "Tháp Eiffel nằm ở Paris, Pháp."),
        ("Việt Nam có bao nhiêu tỉnh?", "Việt Nam có 63 tỉnh thành.")
    ]

    code_examples = [
        ("Viết code Python tính giai thừa của một số.",
         "```python\n"
         "def factorial(n):\n"
         "    return 1 if n <= 1 else n * factorial(n-1)\n"
         "```")
    ]
    
    examples_lines = []
    if state.mode in ["math_step_by_step", "translation"]:
        for input_text, output_text in advanced_examples:
            examples_lines.append(f"User: {input_text}\nAssistant: {output_text}")
    elif state.mode == "code_generation":
        for input_text, output_text in code_examples:
            examples_lines.append(f"User: {input_text}\nAssistant: {output_text}")
    else:  
        for input_text, output_text in short_examples:
            examples_lines.append(f"User: {input_text}\nAssistant: {output_text}")
    if examples_lines:
        prompt_parts.append("Ví dụ tham khảo:\n" + "\n\n".join(examples_lines))
    # Retrieved Context
    retrieved_docs = getattr(state, "retrieved_docs", []) or []
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

    if context_text:
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

    # Debug
    add_debug_info(state, "prompt_length", len(system_prompt))
    add_debug_info(state, "prompt_preview", system_prompt[:400])
    add_debug_info(state, "history_count", len(history_msgs))
    add_debug_info(state, "context_docs_count", len(retrieved_docs))

    return state

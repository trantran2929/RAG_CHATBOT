def build_prompt(state):
    """
    Xây dựng prompt cho LLM:
    - context (retrieved docs từ Qdrant, kèm version/timestamp)
    - conversation history (từ Redis)
    - current user input (clean_query)
    """
    lang = state.lang or "vi"

    if getattr(state, "retrieved_docs", None):
        context_parts = []
        for doc in state.retrieved_docs:
            meta = f"(id={doc.get('id')}, ver={doc.get('version')}, ts={doc.get('timestamp')})"
            text = doc.get("text", "")
            context_parts.append(f"{meta}: {text}")
        context = "\n".join(context_parts)
    else:
        context = "[Không có tài liệu liên quan trong Qdrant]"

    # Lịch sử hội thoại Redis + current messages
    history_msgs = (state.conversation_history or []) + (state.messages or [])
    history_text = ""
    for msg in history_msgs[-10:]:  
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "") or msg.get("query", "") or msg.get("answer", "")
        history_text += f"{role}: {content}\n"

    # Câu hỏi hiện tại
    user_input = state.clean_query or state.raw_query or ""

    # Prompt hệ thống
    system_prompt = f"""
    Bạn là trợ lý AI. Nhiệm vụ:
    - Trả lời ngắn gọn, chính xác, dựa trên Context + Conversation history.
    - Nếu không chắc hoặc không có trong Context, hãy nói rõ: "Tôi chưa tìm thấy thông tin trong kiến thức hiện tại."
    - Tránh lặp lại câu hỏi hoặc thêm thông tin không liên quan.
    - Nếu là chào hỏi → trả lời ngắn gọn, thân thiện.
    - Nếu là yêu cầu viết code → chỉ đưa code chính xác, kèm giải thích ngắn gọn (nếu cần).
    - Luôn trả lời bằng ngôn ngữ '{lang}'.

    Context (retrieved from Qdrant):
    {context}

    Conversation history (from Redis):
    {history_text}

    User question:
    {user_input}

    Assistant:
    """

    state.prompt = system_prompt.strip()
    return state
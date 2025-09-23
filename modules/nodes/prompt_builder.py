from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info
def build_prompt(state: GlobalState)->GlobalState:
    """
    Tạo prompt cho LLM dựa trên:
    - role (system instructions)
    - conversation history
    - few-shot examples
    - retrieved context
    - user query
    """
    lang = state.lang or "vi"
    prompt_parts = []

    if state.role:
        prompt_parts.append(f"Role: {state.role}\n")

    history_msgs = []
    # Merge conversation_history và messages, tránh duplicate
    if state.conversation_history:
        history_msgs.extend(state.conversation_history)
    elif state.messages:  # Chỉ dùng messages nếu không có conversation_history
        history_msgs.extend(state.messages)
    
    if history_msgs:
        history_lines = []
        for msg in history_msgs:
            if not isinstance(msg, dict):
                continue
                
            role = msg.get('role', 'user').capitalize()
            
            # Lấy content từ nhiều field khả dĩ
            content = (
                msg.get('content') or 
                msg.get('query') or 
                msg.get('answer') or 
                ""
            ).strip()
            
            if content:  # Chỉ add nếu có content
                history_lines.append(f"{role}: {content}")
        
        if history_lines:
            prompt_parts.append(f"Conversation History:\n" + "\n".join(history_lines))

    # Few-shot
    if state.examples:
        examples_lines = []
        for ex in state.examples:
            if isinstance(ex, dict):
                input_text = ex.get('input', '').strip()
                output_text = ex.get('output', '').strip()
                if input_text and output_text:
                    examples_lines.append(f"User: {input_text}\nAssistant: {output_text}")
        
        if examples_lines:
            prompt_parts.append(f"Examples:\n" + "\n\n".join(examples_lines))

    # Retrieved context
    retrieved_docs = getattr(state, "retrieved_docs", None) or []
    if retrieved_docs:
        context_parts = []
        for doc in state.retrieved_docs:
            text = (doc.get("text") or doc.get("content") or "").strip()
            if text:
                meta = f"(id={doc.get('id')}, ver={doc.get('version')}, ts={doc.get('timestamp')})"
                context_parts.append(f"{meta}\n{text}")
        if context_parts: 
            prompt_parts.append("Retrieved context:" + "\n".join(context_parts))
        else:
            prompt_parts.append("Retrieved Context: Không có tài liệu liên quan")
    else:
        prompt_parts.append("Retrieved Context: Không có tài liệu liên quan")

    # User query
    user_input = (state.user_query or state.processed_query or "").strip()
    prompt_parts.append(f"User query: {user_input}")

    cobined_parts = "\n\n".join(filter(None, prompt_parts))
    # Prompt hệ thống
    system_prompt = f"""
    Bạn là trợ lý AI. Nhiệm vụ:
    - Trả lời ngắn gọn, chính xác, dựa trên Context + Conversation history.
    - Nếu không chắc hoặc không có trong Context, hãy nói rõ: "Tôi chưa tìm thấy thông tin trong kiến thức hiện tại."
    - Tránh lặp lại câu hỏi hoặc thêm thông tin không liên quan.
    - Nếu là chào hỏi → trả lời ngắn gọn, thân thiện.
    - Nếu là yêu cầu viết code → chỉ đưa code chính xác, kèm giải thích ngắn gọn (nếu cần).
    - Luôn trả lời bằng ngôn ngữ '{lang}'.

    {cobined_parts}

    Assistant:
    """

    state.prompt = system_prompt.strip()
    add_debug_info(state, "prompt_length", len(system_prompt))
    add_debug_info(state,"prompt_preview", system_prompt[:400])
    add_debug_info(state, "history_count", len(history_msgs))
    add_debug_info(state, "context_docs_count", len(retrieved_docs))
    return state
def build_prompt(lang: str, context: str, history: list, user_input: str):
    # Tạo system prompt động theo ngôn ngữ
    system_prompt = f"""
    Bạn là trợ lý AI. Khi người dùng hỏi, hãy:
    - Trả lời ngắn gọn, tập trung vào câu hỏi của người dùng.
    - Không lặp lại hướng dẫn hoặc dữ liệu nền.
    - Nếu là chào hỏi, chỉ cần đáp lại ngắn gọn, thân thiện (ví dụ: 'Xin chào 👋 Bạn cần hỗ trợ gì?').
    - Nếu là yêu cầu viết code, chỉ trả về đoạn code đúng duy nhất và giải thích ngắn gọn nếu cần.
    - Không lặp lại câu hỏi, không thêm thông tin không liên quan, không lan man.
    - Các câu hỏi khác trả lời đúng trọng tâm.
    - Trả lời bằng ngôn ngữ '{lang}'.

    Dữ liệu nền:
    {context}
    """
    prompt = system_prompt
    for msg in history:
        role = (msg.get("role") or "user").capitalize()
        content = msg.get("content") or ""
        prompt += f"\n{role}: {content}"

    # Thêm tin nhắn mới từ người dùng 
    prompt += f"\nNgười dùng hỏi: {user_input}\nTrợ lý AI trả lời:"

    return prompt
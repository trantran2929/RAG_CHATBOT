from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info
from datetime import datetime

SYSTEM_INSTRUCTION = """
***Bạn là trợ lý AI chuyên về tài chính. Hãy thực hiện các nhiệm vụ sau:***
1. Tóm tắt tin tức chứng khoán (cổ phiếu, chỉ số, ngành nghề) Việt Nam.
2. Phân tích xu hướng thị trường dựa trên dữ liệu và tin tức Việt Nam.
3. Cung cấp thông tin mã cổ phiếu ở Việt Nam.
4. Đưa ra gợi ý đầu tư theo hướng tham khảo, KHÔNG phải lời khuyên tuyệt đối.
"""
CONSTRAINTS = """
***Các quy tắc ứng xử:***
1. Không sử dung ngôn ngữ lập trình, mã code(python, js, html, markdown...), các câu lệnh (if/else,print,...).
2. Không được thêm tiền tố như `text`, `json`, `yaml`, `tool_code` hoặc bất kỳ định dạng nào khác.
3. Trả lời trực tiếp, đúng vào câu hỏi.
4. KHÔNG thêm disclaimer kiểu "Tôi không phải chuyên gia tài chính...", heading hoặc nội dung ngoài yêu cầu.
5. Dùng thông tin từ Retrieved context nếu có, nhưng không lặp lại nguyên văn.
"""

def build_prompt(state: GlobalState, max_context_chars: int = 1800) -> GlobalState:
    lang = state.lang or "vi"
    prompt_parts = []

    prompt_parts.append(f"## Instruction:\n {SYSTEM_INSTRUCTION.strip()}\n\n")
    
    prompt_parts.append(f"## Constraints:\n {CONSTRAINTS.strip()}\n\n")

    examples = """
    ## Examples (chỉ minh họa định dạng hội thoại, KHÔNG lặp lại nội dung ví dụ, chỉ dùng để tham khảo cấu trúc):

    **- Example 1:**
    **User:** Hãy cho tôi thông tin về một mã cổ phiếu.  
    **Assistant:** Tóm tắt ngắn gọn tin tức liên quan đến mã cổ phiếu đó.

    **- Example 2:**
    **User:** Thị trường chứng khoán Việt Nam có xu hướng gì nổi bật?  
    **Assistant:** Phân tích xu hướng dựa trên tin tức và dữ liệu, ngắn gọn, súc tích.
    """
    # examples_lines = []
    # for idx, (input_text, output_text) in enumerate(examples, start=1):
    #     examples_lines.append(
    #     f"**Example {idx}:**\n"
    #     f"- User: {input_text}\n"
    #     f"- Assistant: {output_text}"
    # )
        
    # user_input = (state.user_query or state.processed_query or "").strip()
    # if user_input:
    #     examples_lines.append(f"**User:** {user_input}\n**Assistant:**")
    prompt_parts.append(examples.strip())

    # Conversation History
    history_msgs = state.conversation_history or []
    history_lines = []
    if history_msgs:
        for msg in history_msgs:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "").capitalize()
            if role not in ["User", "Assistant"]:
                continue
            content = (
                msg.get("content")
                or msg.get("query")
                or msg.get("answer")
                or ""
            ).strip()
            if content:
                if role == "User":
                    history_lines.append(f"**User:** {content}")
                elif role == "Assistant":
                    history_lines.append(f"**Assistant:** {content}\n")
    if history_lines:
        prompt_parts.append("## Conversation History:\n" + "\n".join(history_lines))
    # else:
    #     prompt_parts.append("## Conversation History:\nKhông có lịch sử hội thoại.")
            
    # Retrieved Context
    retrieved_docs = getattr(state, "retrieved_docs", []) or []
    if retrieved_docs:
        sorted_docs = sorted(
            retrieved_docs,
            key=lambda d: (
                datetime.strptime(d.get("time", ""), "%d-%m-%Y %H:%M:%S")
                if d.get("time") else datetime.min, 
                d.get("score", 0.0)
            ),
            reverse=True
        )
        context_parts = []
        for doc in sorted_docs:
            text = (doc.get("content") or doc.get("summary") or "").strip()
            if text:
                if len(text) > 500:
                    text = text[:500] + "..."
                title = doc.get("title", "")
                url = doc.get("url", "")
                time = doc.get("time", "")
                score = doc.get("score", None)

                meta = " | ".join(filter(None, [title, time,f"score={score:.3f}" if score is not None else None, f"Link: {url}" if url else None]))
                context_parts.append(f"[{meta}]\n{text}")

        context_text = "\n\n".join(context_parts)
        # if len(context_text) > max_context_chars:
        #     context_text = context_text[:max_context_chars] + "...\n[Context truncated]"
        prompt_parts.append("## Retrieved Context:\n" + context_text)
    else:
        prompt_parts.append("## Retrieved Context:\nKhông có tài liệu nào được lấy từ Qdrant (-> kiểm tra retriever).")
    # User query
    user_input = (state.user_query or state.processed_query or "").strip()
    prompt_parts.append("\n## Task Input:")
    prompt_parts.append(f"**User:** {user_input}")

    if lang == "vi":
        lang_instruction = "Không được sử dụng ngôn ngữ khác ngoài tiếng Việt."
    else:
        lang_instruction = f"- Ưu tiên trả lời bằng ngôn ngữ '{lang}'."

    joined_parts = "\n\n".join(prompt_parts)

    state.prompt = "\n\n".join([lang_instruction, joined_parts,"## Task Output:","**Assistant:**\n"]).strip()
    print(state.prompt)

    # Debug
    add_debug_info(state, "prompt_length", len(state.prompt))
    add_debug_info(state, "prompt_preview", state.prompt[:400])
    add_debug_info(state, "history_count", len(history_msgs))
    add_debug_info(state, "context_docs_count", len(retrieved_docs))

    return state

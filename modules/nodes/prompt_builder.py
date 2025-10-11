from modules.core.state import GlobalState
from modules.api.time_api import get_datetime_context

SYSTEM_INSTRUCTION = """
***Bạn là trợ lý AI chuyên về tài chính. Hãy thực hiện các nhiệm vụ sau:***
1. Tóm tắt tin tức chứng khoán (cổ phiếu, chỉ số, ngành nghề) Việt Nam. 
2. Phân tích xu hướng thị trường dựa trên dữ liệu và tin tức Việt Nam.
3. Nếu người dùng hỏi về mã cổ phiếu, phải tập trung phân tích đúng mã đó (ví dụ: TCB → Techcombank), không nhầm sang mã khác.
4. Cung cấp thông tin mã cổ phiếu ở Việt Nam.
5. Đưa ra gợi ý đầu tư theo hướng tham khảo, KHÔNG phải lời khuyên tuyệt đối.
"""
CONSTRAINTS = """
***Các quy tắc ứng xử:***
1. Không sử dung ngôn ngữ lập trình, mã code(python, js, html, markdown...), các câu lệnh (if/else,print,...).
2. Không được thêm tiền tố như `text`, `json`, `yaml`, `tool_code` hoặc bất kỳ định dạng nào khác.
3. KHÔNG thêm disclaimer kiểu "Tôi không phải chuyên gia tài chính...", heading hoặc nội dung ngoài yêu cầu.
4. Dùng thông tin từ Retrieved context nếu có, nhưng không lặp lại nguyên văn.
"""

def build_prompt(state: GlobalState, max_context_chars: int = 1800) -> GlobalState:

    if getattr(state, "route_to", "") not in ["rag", "hybrid"]:
        state.prompt = ""
        state.add_debug("prompt_builder", "Skipped")
        state.llm_status = "prompt_skipped"
        return state

    lang = state.lang or "vi"
    prompt_parts = []

    prompt_parts.append(f"## Instruction:\n {SYSTEM_INSTRUCTION.strip()}\n\n")
    
    prompt_parts.append(f"## Constraints:\n {CONSTRAINTS.strip()}\n\n")

    prompt_parts.append("## Current Date/Time Context:\n" + get_datetime_context()+ "\n(Luôn sử dụng thông tin trên khi người dùng hỏi về ngày, giờ, hôm qua, mai, tuần trước/sau.)")

    examples = """
    ## Examples (chỉ minh họa định dạng hội thoại, KHÔNG lặp lại nội dung ví dụ, chỉ dùng để tham khảo cấu trúc):

    **- Example 1:**
    **User:** Hãy cho tôi thông tin về một mã cổ phiếu.  
    **Assistant:** Tóm tắt ngắn gọn tin tức liên quan đến mã cổ phiếu đó.

    **- Example 2:**
    **User:** Thị trường chứng khoán Việt Nam có xu hướng gì nổi bật?  
    **Assistant:** Phân tích xu hướng dựa trên tin tức và dữ liệu, ngắn gọn, súc tích.
    """
    prompt_parts.append(examples.strip())

    # Conversation History
    history_msgs = (state.conversation_history or [])[-5:]
    if history_msgs:
        history_lines = []
        for msg in history_msgs:
            role = msg.get("role", "").capitalize()
            if role in ["User", "Assistant"]:
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

    if getattr(state, "api_response", None) and getattr(state, "route_to", "") not in ["rag", "hybrid"]:
        prompt_parts.append("## Stock Data:\n" + state.api_response.strip() + "\n")
            
    # Retrieved Context
    retrieved_docs = getattr(state, "retrieved_docs", []) or []
    if retrieved_docs:
        context_parts = []
        for doc in sorted(retrieved_docs, key=lambda d: d.get("score", 0.0), reverse=True):
            text = (doc.get("content") or doc.get("summary") or "").strip()
            if text:
                if len(text) > 500:
                    text = text[:500] + "..."
                title = doc.get("title", "")
                url = doc.get("url", "")
                time = doc.get("time", "")
                score = doc.get("score", None)

                meta = " | ".join(filter(None, [
                    title, time,
                    f"score={score:.3f}" if score is not None else None, 
                    f"Link: {url}" if url else None
                ]))
                context_parts.append(f"[{meta}]\n{text}")

        context_text = "\n\n".join(context_parts)
    # if len(context_text) > max_context_chars:
    #     context_text = context_text[:max_context_chars] + "...\n"
        prompt_parts.append("## Retrieved Context:\n" + context_text)
    else:
        prompt_parts.append("## Retrieved Context:\nKhông có tài liệu nào được lấy từ Qdrant (-> kiểm tra retriever).")
    
    intent_instructions = {
        "stock": "Phân tích hoặc tóm tắt dữ liệu cổ phiếu, chỉ số hoặc biến động thị trường Việt Nam.",
        "news": "Tóm tắt nhanh tin tức tài chính/chứng khoán Việt Nam gần đây.",
        "weather": "Trả lời thông tin thời tiết Việt Nam.",
        "time": "Trả lời về ngày giờ, thời gian hoặc lịch hiện tại.",
        "rag": "Phân tích câu hỏi tổng quát bằng cách dùng Context."
    }
    intent_task = intent_instructions.get(state.intent or "rag", "Trả lời câu hỏi tài chính tổng quát.")
    if state.intent == "news":
        prompt_parts.append("""
            ## Output Guidelines:
            - Hãy tóm tắt 3–5 câu tin tài chính hoặc chứng khoán Việt Nam gần nhất từ Context.
            - Nếu có nhiều tin, ưu tiên tin về các công ty lớn (VIX, SSI, HPG, VCB,...).
            - Nếu không có tin nổi bật, hãy nói rõ “Không tìm thấy tin tài chính đáng chú ý hôm nay”.
            - Không viết placeholder, không lặp lại tiêu đề, không thêm lời khuyên đầu tư.
            - Chỉ dùng dữ liệu có trong Context.
                """)
    prompt_parts.append(f"## Task Type:\nIntent: {state.intent}\nMô tả: {intent_task}")

    # User query
    user_input = (state.user_query or state.processed_query or "").strip()
    prompt_parts.append(f"\n## Task Input:\n**User:** {user_input}")

    if lang == "vi":
        lang_instruction = "- Luôn trả lời bằng TIẾNG VIỆT nếu có từ ngữ chuyên ngành (VNINDEX, VN30,..) thì giữ nguyên dạng gốc"
    else:
        lang_instruction = f"- Ưu tiên trả lời bằng ngôn ngữ '{lang}'."

    joined_parts = "\n\n".join(prompt_parts)

    state.prompt = "\n\n".join([lang_instruction, joined_parts,"## Task Output:","**Assistant:**\n"]).strip()

    print(state.prompt)

    if getattr(state,"debug", False):
        print("*** FINAL PROMPT SAMPLE ***")
        print(state.prompt[:600], flush=True)

    # Debug
    # state.add_debug("prompt_length", len(state.prompt))
    # state.add_debug("prompt_preview", state.prompt[:400])
    # state.add_debug("history_count", len(history_msgs))
    # state.add_debug("context_docs_count", len(retrieved_docs))
    # state.add_debug("prompt_bulder_status", "RAG prompt built")

    return state

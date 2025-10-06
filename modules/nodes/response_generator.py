from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info
from modules.utils.services import llm_services
from datetime import datetime
import re

def response_node(state: GlobalState, max_history: int = 50) -> GlobalState:
    """
    - Nếu có API response (giờ, thời tiết, chứng khoán) → trả luôn
    - Nếu là greeting → trả lời chào
    - Ngược lại gọi LLM sinh phản hồi từ prompt
    - Chuẩn hóa output, lưu vào conversation_history
    """
    if getattr(state, "api_response", None):
        assistant_msg = state.api_response
        state.response = state.final_answer = assistant_msg

        if state.user_query:
            state.conversation_history.append({"role": "user", "content": state.user_query})
        state.conversation_history.append({"role": "assistant", "content": assistant_msg})

        add_debug_info(state, "llm_status", "api_response")
        add_debug_info(state, "router", "API")
        return state

    if getattr(state, "is_greeting", False):
        assistant_msg = "Chào bạn! Tôi có thể giúp gì cho bạn?"
        state.response = state.raw_response = state.final_answer = assistant_msg

        if state.user_query:
            state.conversation_history.append({"role": "user", "content": state.user_query})
        state.conversation_history.append({"role": "assistant", "content": assistant_msg})
        add_debug_info(state, "llm_status", "greeting")
        add_debug_info(state, "router", "API")
        return state
    
    if not state.prompt:
        add_debug_info(state, "llm_status", "empty_prompt")
        state.response = state.final_answer = "Prompt trống, không thể sinh câu trả lời."
        return state

    try:
        if not getattr(llm_services, "generator", None):
            add_debug_info(state, "llm_status", "generator_not_initialized")
            state.final_answer = "LLM chưa được khởi tạo."
            return state

        outputs = llm_services.generator(
            state.prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2,
            return_full_text=False,
        )

        if isinstance(outputs, str):
            text = outputs.strip()
        elif isinstance(outputs, list):
            if len(outputs) > 0 and isinstance(outputs[0], dict):
                text = outputs[0].get("generated_text") or outputs[0].get("text") or ""
            else:
                text = str(outputs)
        elif isinstance(outputs, dict):
            text = outputs.get("generated_text") or outputs.get("text") or ""
        else:
            text = ""

        assistant_msg = re.sub(r"^(```+|---+)", "", text).strip()
        assistant_msg = re.sub(r"http\S+", "(link)", assistant_msg)

        stop_words = [
                "Assistant:", "User:", "Người dùng:",
                "Trợ lý:", "Retrieved Context:", "Answer:", "Conversation History:"
            ]
        for stop_word in stop_words:
            if stop_word in assistant_msg:
                assistant_msg = assistant_msg.split(stop_word)[-1].strip()

        if not assistant_msg:
            assistant_msg = "Xin lỗi, tôi không thể tạo phản hồi."

    except Exception as e:
        add_debug_info(state, "llm_status", "error")
        add_debug_info(state, "llm_error", str(e))
        state.final_answer = "Đã xảy ra lỗi khi gọi LLM."
        state.response = ""
        return state

    if state.user_query:
        state.conversation_history.append({"role": "user", "content": state.user_query})

    entry = {"role": "assistant", "content": assistant_msg}
    if getattr(state, "retrieved_docs", None):
        entry["sources"] = [
            {
                "title": d.get("title", ""),
                "url": d.get("url", ""),
                "time": d.get("time", ""),
                "score": float(d.get("score", 0.0)) if d.get("score") else None,
            }
            for d in state.retrieved_docs
        ]
    state.conversation_history.append(entry)
    state.conversation_history = state.conversation_history[-max_history:]

    state.final_answer = state.response = assistant_msg
    add_debug_info(state, "llm_status", "success")
    add_debug_info(state, "route", "RAG")
    add_debug_info(state, "llm_response_len", len(assistant_msg))

    return state

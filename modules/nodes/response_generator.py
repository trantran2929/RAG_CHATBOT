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
    if getattr(state, "route_to", "") not in ["rag", "hybrid"]:
        state.mark_api_response(
            api_type=state.api_type or "generic_api",
            result=state.api_response,
            text=state.api_response
        )

        if state.user_query:
            state.conversation_history.append({"role": "user", "content": state.user_query})
        state.conversation_history.append({"role": "assistant", "content": state.api_response})

        add_debug_info(state, "router", "API")
        add_debug_info(state, "intent", state.intent)
        return state

    if getattr(state, "is_greeting", False):
        assistant_msg = "Chào bạn! Tôi có thể giúp gì cho bạn?"
        state.set_final_answer(assistant_msg, route="Greeting")

        if state.user_query:
            state.conversation_history.append({"role": "user", "content": state.user_query})
        state.conversation_history.append({"role": "assistant", "content": assistant_msg})
        add_debug_info(state, "llm_status", "greeting")
        add_debug_info(state, "router", "Greeting")
        return state
    
    if not state.prompt:
        msg = "Prompt trống, không thể sinh câu trả lời."
        state.set_final_answer(msg, route="RAG")
        add_debug_info(state, "llm_status", "empty_prompt")
        return state

    try:
        if not getattr(llm_services, "generator", None):
            msg = "LLM chưa được khởi tạo."
            state.set_final_answer(msg, route="RAG")
            add_debug_info(state, "llm_status", "generator_not_initialized")
            return state

        outputs = llm_services.generator(
            state.prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            repetition_penalty=1.2,
            return_full_text=False
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

        assistant_msg = re.sub(r"```[\s\S]*?```", "", text).strip()
        assistant_msg = re.sub(r"^(```+|---+)", "", text).strip()
        assistant_msg = re.sub(r"http\S+", "(link)", assistant_msg)

        stop_words = [
                "Assistant:", "User:", "Người dùng:",
                "Trợ lý:", "Retrieved Context:", "Answer:", "Conversation History:"
            ]
        for stop_word in stop_words:
            if stop_word in assistant_msg:
                assistant_msg = assistant_msg.split(stop_word)[-1].strip()

        if not assistant_msg or len(assistant_msg.split()) < 10:
            # Nếu LLM trả rỗng hoặc quá ngắn -> fallback tự động tóm tắt context
            context_docs = getattr(state, "retrieved_docs", [])
            if context_docs:
                fallback_summary = []
                for doc in context_docs[:3]:  # chỉ lấy top 3 tin
                    title = doc.get("title", "")
                    time = doc.get("time", "")
                    snippet = (doc.get("content_preview") or doc.get("summary") or doc.get("content") or "")
                    snippet = snippet[:150].strip()
                    fallback_summary.append(f"- {title} ({time}): {snippet}...")
                assistant_msg = (
                    "Tóm tắt nhanh các tin nổi bật hôm nay:\n" + "\n".join(fallback_summary)
                    + "\n\nThị trường đang phản ứng với các yếu tố trên, nhà đầu tư tiếp tục theo dõi xu hướng nhóm ngân hàng và chứng khoán."
                )
            else:
                assistant_msg = "Không có dữ liệu tin tức nào để tóm tắt hôm nay."


    except Exception as e:
        add_debug_info(state, "llm_status", "error")
        add_debug_info(state, "llm_error", str(e))
        state.set_final_answer("Đã xảy ra lỗi khi gọi LLM.", route="RAG")
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
    # add_debug_info(state, "llm_status", "success")
    # add_debug_info(state, "route", "RAG")
    # add_debug_info(state, "llm_response_len", len(assistant_msg))
    # add_debug_info(state, "intent", state.intent)
    # add_debug_info(state, "timestamp", datetime.now().isoformat())

    return state

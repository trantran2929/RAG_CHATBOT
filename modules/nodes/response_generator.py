from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info
from modules.utils.services import llm_services

def response_node(state: GlobalState, max_history: int = 50) -> GlobalState:
    """
    - Gọi LLM sinh phản hồi
    - Chuẩn hóa output
    - Cập nhật conversation_history + debug
    - Đưa ra UI
    """

    if not state.prompt:
        add_debug_info(state, "llm_status", "empty_prompt")
        state.final_answer = "Prompt trống, không thể sinh câu trả lời."
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

        if not outputs or "generated_text" not in outputs[0]:
            state.final_answer = "Không nhận được phản hồi từ mô hình."
            add_debug_info(state, "llm_status", "empty_output")
        else:
            state.raw_response = outputs[0]["generated_text"].strip()
            assistant_msg = state.raw_response

            stop_words = [
                "Assistant:", "User:", "Người dùng:",
                "Trợ lý:", "Retrieved Context:", "Answer:", "Conversation History:"
            ]
            for stop_word in stop_words:
                if stop_word in assistant_msg:
                    assistant_msg = assistant_msg.split(stop_word)[-1].strip()

            user_msg = state.user_query or state.processed_query
            if user_msg:
                if not state.conversation_history or state.conversation_history[-1].get("content") != user_msg:
                    state.conversation_history.append({"role": "user", "content": user_msg})

            if assistant_msg:
                if not state.conversation_history or state.conversation_history[-1].get("content") != assistant_msg:
                    entry = {"role": "assistant", "content": assistant_msg}
                    if getattr(state, "retrieved_docs", None):
                        entry["sources"] = [
                            {
                                "title": doc.get("title", ""),
                                "link": doc.get("link", ""),
                                "time": doc.get("time", ""),
                                "score": float(doc.get("score", 0.0)) if doc.get("score") is not None else None,
                            }
                            for doc in state.retrieved_docs
                        ]
                    state.conversation_history.append(entry)

            # Giữ lại max_history
            if len(state.conversation_history) > max_history:
                state.conversation_history = state.conversation_history[-max_history:]

            state.final_answer = assistant_msg
            state.response = assistant_msg
            add_debug_info(state, "llm_status", "success")
            add_debug_info(state, "llm_preview", assistant_msg[:200])

    except Exception as e:
        state.raw_response = ""
        state.final_answer = "Đã xảy ra lỗi khi gọi LLM."
        state.response = ""
        add_debug_info(state, "llm_status", "error")
        add_debug_info(state, "llm_error", str(e))

    return state

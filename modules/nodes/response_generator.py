from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info
from modules.utils.services import llm_services
from datetime import datetime
import re

def response_node(state: GlobalState, max_history: int = 50) -> GlobalState:
    """
    - Gọi LLM sinh phản hồi
    - Chuẩn hóa output
    - Cập nhật conversation_history + debug
    - Đưa ra UI
    """

    if getattr(state, "is_greeting", False):
        assistant_msg = "Chào bạn! Tôi có thể giúp gì cho bạn?"
        state.response = state.raw_response = state.final_answer = assistant_msg

        conv_pair = []
        if state.user_query:
            conv_pair.append({"role": "user", "content": state.user_query})
        conv_pair.append({"role": "assistant", "content": assistant_msg})
        state.conversation_history.extend(conv_pair)
        add_debug_info(state, "llm_status", "greeting")
        return state
    user_msg = (state.user_query or state.processed_query or "").strip().lower()
    
    if "hôm nay" in user_msg and "ngày" in user_msg:
        today = datetime.now().strftime("%d-%m-%Y")
        assistant_msg = f"Hôm nay là ngày {today}."
        state.response = state.final_answer = state.raw_response = assistant_msg
        
        conv_pair = []
        if state.user_query:
            conv_pair.append({"role": "user", "content": state.user_query})
        conv_pair.append({"role": "assistant", "content": assistant_msg})
        state.conversation_history.extend(conv_pair)
        add_debug_info(state, "llm_status", "date_response")
        return state
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

        if outputs or "generated_text" in outputs[0]:
            state.raw_response = outputs[0]["generated_text"].strip()
            assistant_msg = state.raw_response

            assistant_msg = assistant_msg.replace("```", "").replace("---","").replace("python", "").strip()

            assistant_msg = re.sub(r"\[.*?\]", "", assistant_msg, flags=re.DOTALL)
            assistant_msg = re.sub(r"score=\S+", "", assistant_msg)
            assistant_msg = re.sub(r"Link:\s*\S+", "", assistant_msg)
            assistant_msg = re.sub(r"http\S+", "", assistant_msg)

            stop_words = [
                "Assistant:", "User:", "Người dùng:",
                "Trợ lý:", "Retrieved Context:", "Answer:", "Conversation History:"
            ]
            for stop_word in stop_words:
                if stop_word in assistant_msg:
                    assistant_msg = assistant_msg.split(stop_word)[-1].strip()
                
            assistant_msg = re.sub(r"(Ok(ay)?|Alright|Sure|Got it)[\.,!?]?$", "", assistant_msg, flags=re.IGNORECASE).strip()

            # Xóa cụm tiếng Anh không cần thiết
            assistant_msg = re.sub(r"(Ok(ay)?|Alright|Sure|Got it)[\.,!?]?$", "", assistant_msg, flags=re.IGNORECASE).strip()
            assistant_msg = re.sub(r"Okay.*$", "", assistant_msg, flags=re.IGNORECASE).strip()
            assistant_msg = re.sub(r"vui lòng.*đúng nhất cho người dùng.*", "", assistant_msg, flags=re.IGNORECASE).strip()

            # Nếu vẫn thừa nhiều tiếng Anh → cắt
            if re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}\s+[A-Za-z]{3,}", assistant_msg):
                parts = re.split(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}\s+[A-Za-z]{3,}", assistant_msg)
                assistant_msg = parts[0].strip()

            if not assistant_msg:
                assistant_msg = "Xin lỗi, tôi không thể tạo phản hồi."
        else:
            state.final_answer = "Không nhận được phản hồi từ mô hình."
            add_debug_info(state, "llm_status", "empty_output")
        if not assistant_msg and user_msg in ["có", "vâng", "tiếp tục", "đồng ý", "yes", "ok", "okay"]:
            last_assistant = ""
            for msg in reversed(state.conversation_history):
                if msg.get("role") == "assistant" and msg.get("content"):
                    last_assistant = msg["content"]
                    break
            assistant_msg = (
                "Tôi sẽ tiếp tục từ câu trả lời trước: " + last_assistant
            )

        conv_pair = []
        if state.user_query:
            conv_pair.append({"role": "user", "content": state.user_query})

        if assistant_msg:
            entry = {"role": "assistant", "content": assistant_msg}
            if getattr(state, "retrieved_docs", None):
                entry["sources"] = [
                    {
                        "title": doc.get("title", ""),
                        "url": doc.get("url", ""),
                        "time": doc.get("time", ""),
                        "score": float(doc.get("score", 0.0)) if doc.get("score") is not None else None,
                    }
                    for doc in state.retrieved_docs
                ]
            conv_pair.append(entry)
        if conv_pair:
            state.conversation_history.extend(conv_pair)

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

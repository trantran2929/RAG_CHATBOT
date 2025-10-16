from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info
from modules.utils.services import llm_services
from datetime import datetime
import re

def response_node(state: GlobalState, max_history: int = 50) -> GlobalState:
    """
    Node cuối cùng sinh phản hồi cho người dùng.
    Quy tắc:
    - Nếu route_to = API → trả ngay api_response
    - Nếu greeting → trả lời chào thân thiện
    - Nếu RAG / Hybrid → gọi LLM sinh nội dung từ prompt
    """
    route = getattr(state, "route_to", "")
    intent = getattr(state, "intent", "rag")

    if route not in ["rag", "hybrid"]:
        result = state.api_response or "Không có phản hồi API"
        state.mark_api_response(
            api_type=state.api_type or "api",
            result=result,
            text=result
        )
        if state.user_query:
            state.conversation_history.append({"role": "user", "content": state.user_query})
        state.conversation_history.append({"role": "assistant", "content": result})
        state.final_answer = result
        state.response = result
        return state

    if getattr(state, "is_greeting", False) or intent == "greeting":
        msg = "👋 Xin chào! Tôi có thể giúp gì cho bạn?"
        state.set_final_answer(msg, route="Greeting")
        return state

    if not state.prompt:
        msg = "Không thể tạo prompt — thiếu dữ liệu RAG."
        state.set_final_answer(msg, route="RAG")
        return state

    try:
        if not hasattr(llm_services, "model"):
            msg = "LLM chưa được khởi tạo."
            state.set_final_answer(msg, route="RAG")
            add_debug_info(state, "llm_status", "model_not_initialized")
            return state

        outputs = llm_services.model.invoke(
            [
                {"role": "user", "content": state.prompt},
                {"role": "assistant", "content": ""}
            ],
            temperature=0.7,
            max_tokens=2048
        )

        # Chuẩn hóa kết quả trả về
        if isinstance(outputs, str):
            text = outputs.strip()
        elif isinstance(outputs, list):
            if len(outputs) > 0 and isinstance(outputs[0], dict):
                text = outputs[0].get("content") or outputs[0].get("text") or str(outputs[0])
            else:
                text = " ".join(map(str, outputs))
        elif hasattr(outputs, "content"):
            text = outputs.content.strip()
        else:
            text = str(outputs)

        # Làm sạch markdown, code block, URL, từ khóa không cần thiết
        assistant_msg = re.sub(r"```[\s\S]*?```", "", text)
        assistant_msg = re.sub(r"http\S+", "(link)", assistant_msg)
        assistant_msg = re.sub(r"^(Assistant:|User:|Trợ lý:|Người dùng:)\s*", "", assistant_msg, flags=re.I).strip()

        # Nếu LLM trả về quá ngắn hoặc trống, fallback tự tóm tắt context
        if len(assistant_msg.split()) < 10:
            if state.intent == "market" and getattr(state, "api_response", None):
                assistant_msg = f"📊 Dữ liệu thị trường:\n{state.api_response}\n(Tin tức gần đây chưa khả dụng để phân tích chi tiết.)"
            else:
                assistant_msg = "Hiện chưa có đủ dữ liệu để phân tích chi tiết."
        # Cập nhật hội thoại
        if state.user_query:
            state.conversation_history.append({"role": "user", "content": state.user_query})
        entry = {"role": "assistant", "content": assistant_msg}

        # Nếu có nguồn, thêm metadata
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
        state.final_answer = assistant_msg
        state.response = assistant_msg
        add_debug_info(state, "llm_status", "response_generated")
        add_debug_info(state, "route", route)
        add_debug_info(state, "intent", intent)
        add_debug_info(state, "timestamp", datetime.now().isoformat())

    except Exception as e:
        add_debug_info(state, "llm_error", str(e))
        err_msg = "Đã xảy ra lỗi khi gọi LLM, vui lòng thử lại sau."
        state.set_final_answer(err_msg, route="RAG")
        state.response = err_msg
        return state

    return state

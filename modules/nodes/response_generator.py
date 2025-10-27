from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info
from modules.utils.services import llm_services
from datetime import datetime
import re
from modules.nodes.prompt_builder import SYSTEM_INSTRUCTION, CONSTRAINTS

def _summarize_docs_for_user(docs, limit=5):
    if not docs:
        return ""
    lines = ["Một số diễn biến đáng chú ý gần đây:"]
    for i, d in enumerate(docs[:limit], start=1):
        title = (d.get("title") or "").strip()
        body = (d.get("content") or "").strip().replace("\n", " ")
        if len(body) > 1000:
            body = body[:1000].strip()
        if title or body:
            lines.append(f"{i}. {title}\n   {body}")
    lines.append(
        "\nLưu ý: Đây là tóm tắt thông tin thị trường/chứng khoán, không phải khuyến nghị đầu tư."
    )
    return "\n".join(lines).strip()

def response_node(state: GlobalState, max_history: int = 50) -> GlobalState:
    route = getattr(state, "route_to", "")
    intent = getattr(state, "intent", "rag")

    if route not in ["rag", "hybrid"]:
        result = state.api_response or "Không có phản hồi API"
        state.mark_api_response(
            api_type=getattr(state, "api_type", None) or "api",
            result=result,
            text=result
        )
        if state.user_query:
            state.conversation_history.append({"role": "user", "content": state.user_query})
        state.conversation_history.append({"role": "assistant", "content": result})

        state.final_answer = result
        state.response = result
        state.llm_status = "response_api_done"
        return state

    if getattr(state, "is_greeting", False) or intent == "greeting":
        msg = "Xin chào! Tôi có thể giúp gì cho bạn?"
        state.set_final_answer(msg, route="Greeting")
        state.llm_status = "response_greeting"
        return state

    if not state.prompt:
        msg = "Không thể tạo prompt — thiếu dữ liệu RAG."
        state.set_final_answer(msg, route="RAG")
        state.llm_status = "response_missing_prompt"
        return state

    fallback_summary = _summarize_docs_for_user(getattr(state, "retrieved_docs", []) or [])

    try:
        messages = [
            {
                "role": "system",
                "content": (SYSTEM_INSTRUCTION.strip() + "\n" + CONSTRAINTS.strip())
            },
            {
                "role": "user",
                "content": state.prompt
            },
        ]

        outputs = llm_services.model.invoke(
            messages,
            temperature=0.7,
            max_tokens=2048
        )

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

        assistant_msg = re.sub(r"```[\s\S]*?```", "", text)
        assistant_msg = re.sub(r"http\S+", "(link)", assistant_msg)
        assistant_msg = re.sub(
            r"^(Assistant:|User:|Trợ lý:|Người dùng:)\s*",
            "",
            assistant_msg,
            flags=re.I
        ).strip()

        if len(assistant_msg.split()) < 10:
            if intent == "market" and getattr(state, "api_response", None):
                assistant_msg = (
                    f"Dữ liệu thị trường:\n{state.api_response}\n"
                    "Diễn biến chung: xu hướng thị trường chịu ảnh hưởng bởi các thông tin gần đây.\n"
                    "Lưu ý: đây chỉ là mô tả tình hình, không phải khuyến nghị đầu tư."
                )
            elif fallback_summary:
                assistant_msg = fallback_summary
            else:
                assistant_msg = "Hiện chưa có đủ dữ liệu để phân tích chi tiết."

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

        state.final_answer = assistant_msg
        state.response = assistant_msg
        state.llm_status = "response_generated"

        add_debug_info(state, "llm_status", "response_generated")
        add_debug_info(state, "route", route)
        add_debug_info(state, "intent", intent)
        add_debug_info(state, "timestamp", datetime.now().isoformat())

    except Exception as e:
        add_debug_info(state, "llm_error", str(e))

        if fallback_summary:
            err_msg = fallback_summary
        elif intent == "market" and getattr(state, "api_response", None):
            err_msg = (
                f"Dữ liệu thị trường:\n{state.api_response}\n"
                "Đây là mô tả lại thông tin thị trường, không phải khuyến nghị đầu tư."
            )
        else:
            err_msg = "Đã xảy ra lỗi khi gọi LLM, vui lòng thử lại sau."

        state.set_final_answer(err_msg, route="RAG")
        state.response = err_msg
        state.llm_status = "response_llm_error"
        return state

    return state

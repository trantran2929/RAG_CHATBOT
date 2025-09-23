from modules.core.state import GlobalState
from modules.utils.debug import add_debug_info


def chatbot_response(state: GlobalState) -> GlobalState:
    """
    Node cuối cùng trong graph:
    - Trả về response cuối cùng
    - Cập nhật conversation_history
    """
    user_msg = state.user_query or state.processed_query
    assistant_msg = state.response or state.final_answer

    # Cập nhật conversation history
    if user_msg:
        state.conversation_history.append({"role": "user", "content": user_msg})
    if assistant_msg:
        state.conversation_history.append({"role": "assistant", "content": assistant_msg})

    # Debug info
    add_debug_info(state, "response_node", "done")
    add_debug_info(state, "response_len", len(assistant_msg) if assistant_msg else 0)

    return state

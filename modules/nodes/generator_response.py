from .services import llm_services
from modules.utils.debug import add_debug_info

def generate_response(state): 
    """
    Sinh câu trả lời dựa trên state.prompt
    và cập nhật state.final_answer (chỉ text).
    """
    if not state.prompt:
        raise ValueError("Prompt chưa được tạo, chạy build_prompt trước.")

    if not getattr(llm_services, "generator", None):
        state.final_answer = "[LLM chưa được khởi tạo]"
        add_debug_info(state, "llm_status", "generator_not_found")
        return state
    
    pipe = llm_services.generator
    outputs = pipe( 
        state.prompt, 
        max_new_tokens=512, 
        do_sample=True, 
        temperature=0.7,  
        repetition_penalty=1.2,
        return_full_text=False
    )

    if not outputs:
        state.final_answer = ""
        add_debug_info(state, "llm_status", "empty_output")
        return state
    
    response = outputs[0].get("generated_text", "").strip()

    stop_words = ["User:", "Người dùng hỏi:", "Trợ lý AI trả lời:"]
    for stop_word in stop_words:
        if stop_word in response:
            response = response.split(stop_word)[0].strip()

    # Chỉ giữ lại câu trả lời, không lưu meta
    state.final_answer = response

    add_debug_info(state, "llm_status", "success")
    return state
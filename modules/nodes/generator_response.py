from .services import llm_services
from modules.utils.debug import add_debug_info
from modules.core.state import GlobalState

def generate_response(state: GlobalState)->GlobalState: 
    """
    Sinh câu trả lời dựa trên state.prompt
    """
    if not state.prompt:
        add_debug_info(state,"response_generator","không có prompt để sinh output")
        state.llm_output = ""
        state.final_answer = ""
        state.response = ""
        return state

    try:
    
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
            state.llm_output = ""
            state.final_answer = ""
            state.response = ""
            add_debug_info(state, "llm_status", "empty_output")
            return state
    
        response = outputs[0].get("generated_text", "").strip()

        stop_words = ["User:", "Người dùng hỏi:", "Trợ lý AI trả lời:"]
        for stop_word in stop_words:
            if stop_word in response:
                response = response.split(stop_word)[0].strip()

        state.llm_output = outputs
        state.final_answer = response
        state.response = response

        add_debug_info(state,"llm_status", "success")
        add_debug_info(state,"llm_output_len", len(response))
        add_debug_info(state, "llm_preview", response[:200])
    except Exception as e:
        state.llm_output = ""
        state.final_answer = ""
        state.response = ""
        add_debug_info(state,"llm_status","error")
        add_debug_info(state,"llm_error",str(e))

    return state
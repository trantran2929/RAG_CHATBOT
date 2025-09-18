import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# Tải biến môi trường từ file .env 
load_dotenv() 

# Kiểm tra và đăng nhập HuggingFace 
hf_token = os.getenv("HF_TOKEN") 
login(token=hf_token) 

# Tắt tối ưu hóa Dynamo để giảm sử dụng bộ nhớ 
os.environ['TORCH_COMPILE_DEBUG'] = '0' 
os.environ['TORCHDYNAMO_DISABLE'] = '1' 
torch._dynamo.config.suppress_errors = True 


# Khởi tạo mô hình Gemma-2-2B-IT 
model_id = "google/gemma-3-1b-it"
model = AutoModelForCausalLM.from_pretrained( 
    model_id, 
    device_map="auto", 
    torch_dtype=torch.float32,  
    trust_remote_code=True
) 

tokenizer = AutoTokenizer.from_pretrained(model_id) 

# Tạo pipeline 
pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=256, 
    do_sample=True, 
    temperature=0, 
    return_full_text=False, 
    repetition_penalty=1.2 
)

# Hàm xử lý tin nhắn cho Gradio 
def generate_response(prompt: str) -> str: 
    outputs = pipe( 
            prompt, 
            max_new_tokens=512, 
            do_sample=True, 
            temperature=0.7,  
            repetition_penalty=1.2,
            return_full_text=False
        ) 
    response = outputs[0]['generated_text'].strip() 

    # Cắt bỏ nếu model sinh thừa "User:" hoặc "Người dùng hỏi:"
    for stop_word in ["User:", "Người dùng hỏi:", "Trợ lý AI trả lời:"]:
        if stop_word in response:
            response = response.split(stop_word)[0].strip()
    return response
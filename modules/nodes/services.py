from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
import redis
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import os

# Tải biến môi trường từ file .env 
load_dotenv() 

# Kiểm tra và đăng nhập HuggingFace 
hf_token = os.getenv("HF_TOKEN") 
login(token=hf_token) 

# Tắt tối ưu hóa Dynamo để giảm sử dụng bộ nhớ 
os.environ['TORCH_COMPILE_DEBUG'] = '0' 
os.environ['TORCHDYNAMO_DISABLE'] = '1' 
torch._dynamo.config.suppress_errors = True 

class QdrantServices:
    def __init__(self, collection_name="collection", vector_size=384):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))

        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

        # Kiểm tra collection
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            print(f"Collection `{self.collection_name}` chưa tồn tại, tạo mới...")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance="Cosine")
            )
        else:
            print(f"Collection `{self.collection_name}` đã tồn tại.")


qdrant_services = QdrantServices()


class RedisCacheServices:
    def __init__(self, db=0):
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))

        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)


redis_services = RedisCacheServices()


class LLMServices:
    def __init__(self, model_id = "google/gemma-3-1b-it"):
        self.model = AutoModelForCausalLM.from_pretrained( 
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float32,  
        trust_remote_code=True
        ) 
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generator = pipeline( 
            "text-generation", 
            model=model_id, 
            tokenizer=self.tokenizer, 
            max_new_tokens=256, 
            do_sample=True, 
            temperature=0, 
            return_full_text=False, 
            repetition_penalty=1.2 
        )
llm_services = LLMServices()
class EmbedderServices:
    def __init__(self, model_id = "sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
embedder_services = EmbedderServices()
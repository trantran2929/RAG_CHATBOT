from qdrant_client import QdrantClient
from qdrant_client.http import models
import redis
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import os
import numpy as np
from rank_bm25 import BM25Okapi

# Load biến môi trường
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Disable Torch Dynamo
os.environ['TORCH_COMPILE_DEBUG'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
torch._dynamo.config.suppress_errors = True


class QdrantServices:
    def __init__(self, collection_name="cafef_articles", vector_size=384):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            print(f"Tạo mới collection `{self.collection_name}` (hybrid dense+sparse)")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                    "binary": models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.DOT,
                        on_disk=True
                    )
                },
                sparse_vectors_config={
                    "sparse_vector": models.SparseVectorParams()
                }
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
    def __init__(self, model_id="google/gemma-3-1b-it"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0,
            return_full_text=False,
            repetition_penalty=1.2
        )


# try:
#     llm_services = LLMServices()
# except Exception as e:
#     print("Skip LLMServices init:", e)
#     llm_services = None
llm_services = LLMServices()


class EmbedderServices:
    def __init__(
            self, 
            dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
            device=None
        ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Dense encoder
        self.dense_tokenizer = AutoTokenizer.from_pretrained(dense_model_name)
        self.dense_model = AutoModel.from_pretrained(dense_model_name).to(self.device)

        #BM25
        self.corpus_tokens = None
        self.bm25 = None

    def fit_bm25(self,corpus):
        self.corpus_tokens = [doc.split(" ")  for doc in corpus]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def encode_dense(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.dense_tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.dense_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
    
    def encode_sparse(self, query):
        if not self.bm25:
            raise ValueError("BM25 chưa được fit")
        
        tokenizer_query = query.split(" ")
        scores = self.bm25.get_scores(tokenizer_query)
        # vị trí từ trong vocab
        indices = [i for i,s in enumerate(scores) if s>0]
        #score tương ứng
        values = [s for s in scores if s>0]
        return {"indices":indices, "values":values}
    
    def encode_binary(self, dense_vec, threshold=0.0):
        arr = np.array(dense_vec)
        binary = (arr>threshold).astype(int).tolist()
        return binary

embedder_services = EmbedderServices()

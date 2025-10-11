from qdrant_client import QdrantClient
from qdrant_client.http import models
import redis
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.pipelines import pipeline
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
            print(f"Tạo mới collection `{self.collection_name}`")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
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
            torch_dtype=torch.float16,
            trust_remote_code=True,
            # low_cpu_mem_usage=True,
            offload_folder="offload",
            offload_buffers=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            return_full_text=True,
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
            device=None,
            auto_fit=True,
            collection_name="cafef_articles",
            max_docs=5000
        ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("EmbedderServices device:", self.device)
        self.dense_tokenizer = AutoTokenizer.from_pretrained(dense_model_name)
        self.dense_model = AutoModel.from_pretrained(dense_model_name).to(self.device)

        self.corpus_tokens = None
        self.bm25 = None
        self.vocab= {}

        if auto_fit:
            self.auto_fit_bm25(collection_name, max_docs)

    def fit_bm25(self,corpus):
        self.corpus_tokens = [doc.split(" ")  for doc in corpus]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        unique_tokens = sorted(set(token for doc in self.corpus_tokens for token in doc))
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens)}

    def auto_fit_bm25(self, collection_name, max_docs=5000):
        """
        Tự động fit BM25 từ Qdrant nếu chưa được fit
        """
        try:
            docs, _ = qdrant_services.client.scroll(
                collection_name=collection_name,
                limit=max_docs,
                with_payload=True
            )
            corpus = []
            for d in docs:
                payload = d.payload or {}
                text = payload.get("content") or payload.get("summary")
                if text and len(text.split()) > 5:
                    corpus.append(text)
            
            if corpus:
                self.fit_bm25(corpus)
            else:
                print("[BM25] Không tìm thấy dữ liệu để fit BM25")
        
        except Exception as e:
            print("[BM25] Lỗi auto_fit BM25: {e}")

    def encode_dense(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.dense_tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.dense_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
    
    def encode_sparse(self, texts):
        if self.bm25 is None or not hasattr(self, "vocab") or len(self.vocab) == 0:
            raise ValueError("BM25 chưa được fit")

        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts:
            tokens = text.split(" ")
            index_map = {}
            for token in tokens:
                if token in self.vocab:
                    idx = self.vocab[token]
                    val = self.bm25.idf.get(token, 0.0)
                    if val > 0:
                        index_map[idx] = index_map.get(idx, 0.0) + val
            indices = sorted(list(index_map.keys()))
            values = list(index_map.values())
            results.append({"indices": indices, "values": values})
        return results
    
embedder_services = EmbedderServices(auto_fit=True)

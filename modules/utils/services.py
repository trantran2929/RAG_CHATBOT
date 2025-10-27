from qdrant_client import QdrantClient
from qdrant_client.http import models
import redis
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.pipelines import pipeline
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import os, time
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import requests
import numpy as np
from langchain.chat_models import init_chat_model

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

# class LLMServices:
#     def __init__(self, model_id="google/gemma-3-1b-it"):
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#             # low_cpu_mem_usage=True,
#             offload_folder="offload",
#             offload_buffers=True
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.generator = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             max_new_tokens=256,
#             do_sample=True,
#             temperature=0.6,
#             return_full_text=True,
#             repetition_penalty=1.2
#         )

#     def generate(self, prompt: str):
#         return self.generator(prompt)[0]["generated_text"]

#     # try:
#     #     llm_services = LLMServices()
#     # except Exception as e:
#     #     print("Skip LLMServices init:", e)
#     #     llm_services = None
# llm_services = LLMServices()

class LLMServices:
    def __init__(self,
                 base_url="https://nonomissible-winfred-doggedly.ngrok-free.dev/v1",
                 model_name="qwen3-30b-thinking"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = os.getenv("LLM_API_KEY", "dummy") 

        self.model = init_chat_model(
            model=self.model_name,
            model_provider="openai",       
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048):
        """
        Gọi model để sinh phản hồi (chat completion).
        """
        try:
            res = self.model.invoke(
                [
                    {"role": "system", "content": "Bạn là trợ lý AI tài chính thông minh, trả lời bằng tiếng Việt."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Kết quả LangChain có thể là string hoặc object
            return res if isinstance(res, str) else getattr(res, "content", str(res))
        except Exception as e:
            print(f"[LLM] Lỗi khi gọi model qua LangChain: {e}")
            return f"[Lỗi LLM từ server] {e}"

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
            print(f"[BM25] Lỗi auto_fit BM25: {e}")

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

class RerankerServices:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("RerankerServices device:", self.device)
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, docs: list):
        """
            Sắp xếp lại danh sách tài liệu theo độ liên quan ngữ nghĩa (semantic relevance).
            Args:
                query: câu hỏi hoặc truy vấn của người dùng
                docs: list chứa text hoặc dict có key 'content'
            Returns:
                list đã rerank theo điểm số giảm dần
        """
        if not docs:
            return []
            
        # Nếu input là list[str]
        if isinstance(docs[0], str):
            pairs = [(query,d) for d in docs]
            scores = self.model.predict(pairs)
            return list(zip(docs, scores))
        # Nếu input là list[dict]
        pairs = [(query, d.get("content", "")) for d in docs]
        scores = self.model.predict(pairs)

        for i, s in enumerate(scores):
            docs[i]["rerank_score"] = float(s)
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return docs
reranker_services =RerankerServices()

class SentimentServices:
    """
    Chuẩn hoá output:
      - sentiment ∈ [-1..+1] = P(pos) - P(neg)
      - label ∈ {'neg','neu','pos'} theo argmax
    Model mặc định: cardiffnlp/twitter-xlm-roberta-base-sentiment (NEG/NEU/POS).
    Tự động fallback -> 'neu', 0.0 nếu lỗi hoặc input quá ngắn.
    """
    def __init__(self,
                 model_id: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                 device: str | None = None,
                 max_len: int = 1000):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

        self.max_len = int(max_len)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ready = False

        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
            self.pipe = pipeline(
                "text-classification",
                model=mdl,
                tokenizer=tok,
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                return_all_scores=True
            )
            # Chuẩn hoá nhãn từ config (LABEL_0/1/2 hoặc negative/neutral/positive)
            id2label = getattr(mdl.config, "id2label", None)
            if isinstance(id2label, dict) and len(id2label) >= 2:
                labset = {str(v).lower() for v in id2label.values()}
            else:
                labset = {"label_0", "label_1", "label_2"}

            self.has_neutral = any("neu" in x or "neutral" in x for x in labset)
            self.ready = True
        except Exception as e:
            print(f"[Sentiment] init failed: {e}")
            self.pipe = None

    @staticmethod
    def _pack_from_scores(scores: list[dict]) -> dict:
        """
        scores: list[{'label': 'NEGATIVE/NEUTRAL/POSITIVE' (hoặc LABEL_0/1/2), 'score': float}, ...]
        Trả về {'label','sentiment'} với label ∈ {'neg','neu','pos'}.
        """
        p = {"neg": 0.0, "neu": 0.0, "pos": 0.0}
        for s in scores:
            lab = s.get("label", "").lower()
            sc = float(s.get("score", 0.0))

            if any(k in lab for k in ("neg", "label_0", "negative")):
                p["neg"] = max(p["neg"], sc)
            elif any(k in lab for k in ("neu", "label_1", "neutral")):
                p["neu"] = max(p["neu"], sc)
            elif any(k in lab for k in ("pos", "label_2", "positive")):
                p["pos"] = max(p["pos"], sc)

        # nếu model nhị phân (không có neutral), dồn phần còn lại cho 'neu'=0
        # sentiment = P(pos) - P(neg)
        sentiment = float(p["pos"] - p["neg"])

        # chọn nhãn theo xác suất lớn nhất
        label = max(p.items(), key=lambda kv: kv[1])[0]
        return {"label": label, "sentiment": float(max(-1.0, min(1.0, sentiment)))}

    def analyze(self, title: str = "", summary: str = "", content: str = "") -> dict:
        """
        Dùng cho từng bài/chunk. Ưu tiên title+summary; rỗng thì fallback content.
        """
        if not self.ready:
            return {"label": "neu", "sentiment": 0.0}

        base = " ".join(t for t in [title or "", summary or ""] if t).strip()
        if not base:
            base = (content or "").strip()
        if len(base) < 10:
            return {"label": "neu", "sentiment": 0.0}

        text = base[: self.max_len]
        try:
            out = self.pipe(text)
            scores = out[0] if isinstance(out, list) and out and isinstance(out[0], list) else out
            return self._pack_from_scores(scores)
        except Exception as e:
            print(f"[Sentiment] predict error: {e}")
            return {"label": "neu", "sentiment": 0.0}

    def analyze_text(self, text: str) -> dict:
        """Phân tích trực tiếp 1 string."""
        return self.analyze("", "", text)

    def analyze_batch(self, items: list[dict]) -> list[dict]:
        """
        Batch API (nhanh hơn gọi từng cái).
        items: [{'title':..., 'summary':..., 'content':...}, ...]
        Trả về list [{'label','sentiment'}...], giữ nguyên thứ tự.
        """
        if not self.ready or not items:
            return [{"label": "neu", "sentiment": 0.0} for _ in (items or [])]

        texts = []
        for it in items:
            title = it.get("title", "") or ""
            summary = it.get("summary", "") or ""
            content = it.get("content", "") or ""
            base = " ".join(t for t in [title, summary] if t).strip() or content.strip()
            if len(base) < 10:
                base = ""
            texts.append(base[: self.max_len] if base else "")

        results = []
        try:
            outs = self.pipe(texts, truncation=True)
            for out in outs:
                if not out:
                    results.append({"label": "neu", "sentiment": 0.0})
                else:
                    results.append(self._pack_from_scores(out))
        except Exception as e:
            print(f"[Sentiment] batch error: {e}")
            results = [{"label": "neu", "sentiment": 0.0} for _ in texts]

        return results

try:
    sentiment_services = SentimentServices()
except Exception as _e:
    print("[Sentiment] global init failed:", _e)
    sentiment_services = None

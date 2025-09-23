import torch
from .services import embedder_services
from modules.utils.debug import add_debug_info
from modules.core.state import GlobalState

class EmbedderNode:

    @staticmethod
    def get_embedding(state: GlobalState)->GlobalState:
        """
        Tính embedding cho processed_query và lưu vào state.query_embedding
        """
        texts = state.processed_query or state.user_query or ""
        if isinstance(texts, str):
            text = [texts]
        else:
            text = texts

        inputs = embedder_services.tokenizer(
            text,
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(embedder_services.model.device)

        with torch.no_grad():
            outputs = embedder_services.model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # Nếu chỉ có 1 câu → lấy vector 1D
        if embedding.shape[0] == 1:
            embedding = embedding[0]

        state.query_embedding = embedding.tolist()
        add_debug_info(state, "embedding_shape", str(embedding.shape))
        return state
    
embedder_instance = EmbedderNode()

def embed_query(state):
    state = embedder_instance.get_embedding(state)
    return state
import torch
from .services import embedder_services
from modules.utils.debug import add_debug_info

class EmbedderNode:

    @staticmethod
    def get_embedding(state):
        """
        Tính embedding cho state.clean_query
        """
        texts = state.clean_query or state.raw_query or ""
        if isinstance(texts, str):
            texts = [texts]

        inputs = embedder_services.tokenizer(
            texts,
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

        state.query_embedding = embedding
        add_debug_info(state, "embedding_shape", embedding.shape)
        return state
    
embedder_instance = EmbedderNode()

def embed_query(state):
    state = embedder_instance.get_embedding(state)
    return state
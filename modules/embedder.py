import torch
from transformers import AutoTokenizer, AutoModel

# Embedding model cho RAG
embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
embed_model = AutoModel.from_pretrained(embed_model_id)

def get_embedding(text: str):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

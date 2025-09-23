import re
import unicodedata
from typing import List, Dict

def normalize_unicode(text: str)->str:
    return unicodedata.normalize("NFC",text)


def clean_text(text: str) -> str:
    """Làm sạch nội dung văn bản"""
    text = normalize_unicode(text)
    # Bỏ HTML entities & ký tự đặc biệt
    text = re.sub(r"&[a-z]+;", " ", text)
    # Giữ lại chữ, số, dấu chấm câu cơ bản
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s\.,!?]", " ", text)
    # Bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, max_words: int = 200) -> List[str]:
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            chunks.append("".join(cur))
            cur = []
    if cur:
        chunks.append("".join(cur))
    return chunks

def preprocess_articles(articles: List[Dict], max_words: int = 200) -> List[Dict]:
    """
    Nhận danh sách articles từ crawler
    return danh sách chunks
    """
    docs = []
    for art in articles:
        cleaned = clean_text(art["content"])
        chunks = chunk_text(cleaned, max_words=max_words)
        for idx, chunk in enumerate(chunks):
            docs.append({
                "id": f"{art['id']}_{idx}",
                "title": art["title"],
                "time": art["time"],
                "summary": art["summary"],
                "link": art["link"],
                "content": chunk
            })
    return docs
import re
import unicodedata
import hashlib
from typing import List, Dict


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa unicode (NFC)."""
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """Làm sạch nội dung bài viết."""
    text = normalize_unicode(text)
    text = re.sub(r"&[a-z]+;", " ", text)
    # Giữ thêm một số ký tự phổ biến: … –
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s\.,!?\-:;/()\"'%…–]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, max_words: int = 200, overlap: float = 0.2) -> List[str]:
    """
    chia nhỏ văn bản thành các chunk bằng sliding window.
    max_words: số từ tối đa mỗi chunk
    overlap: số từ overlap giữa các chunk liên tiếp (0.2)
    """
    words = text.split()
    if not words:
        return []
    step = int(max_words * (1 - overlap))
    if step <= 0:
        step = max_words
    chunks = []
    for i in range(0, len(words), step):
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))
        if i + max_words >= len(words):
            break
    return chunks


def preprocess_articles(articles: List[Dict], max_words: int = 200) -> List[Dict]:
    """
    Nhận danh sách bài viết crawl từ CafeF, làm sạch + chunk text.
    Trả về danh sách docs để upsert vào Vector DB.
    """
    docs = []
    for art in articles:
        raw_id = art.get("id")
        title = art.get("title", "").strip()
        content = art.get("content", "").strip()
        cleaned = clean_text(content)
        if not cleaned:
            continue
        
        # fallback ID
        if not raw_id:
            hash_id = hashlib.md5((title + content).encode("utf-8")).hexdigest()[:8]
            raw_id = f"cafef_{hash_id}"
        chunks = chunk_text(cleaned, max_words=max_words)
        for idx, chunk in enumerate(chunks):
            docs.append({
                "id": f"cafef_{raw_id}_{idx}",   # id unique cho từng chunk
                "title": title,
                "time": art.get("time", ""),
                "summary": art.get("summary", ""),
                "url": art.get("url", ""),
                "content": chunk,
                "source": "cafef_stock"
            })
    return docs

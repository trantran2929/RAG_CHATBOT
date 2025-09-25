import re
import unicodedata
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


def chunk_text(text: str, max_words: int = 200) -> List[str]:
    """Chia nhỏ văn bản thành các chunk có tối đa max_words từ."""
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def preprocess_articles(articles: List[Dict], max_words: int = 200) -> List[Dict]:
    """
    Nhận danh sách bài viết crawl từ CafeF, làm sạch + chunk text.
    Trả về danh sách docs để upsert vào Vector DB.
    """
    docs = []
    for art in articles:
        cleaned = clean_text(art.get("content", ""))
        if not cleaned:
            continue

        chunks = chunk_text(cleaned, max_words=max_words)
        for idx, chunk in enumerate(chunks):
            docs.append({
                "id": f"cafef_{art['id']}_{idx}",   # id unique cho từng chunk
                "title": art.get("title", ""),
                "time": art.get("time", ""),
                "summary": art.get("summary", ""),
                "url": art.get("url", ""),
                "content": chunk,
                "source": "cafef_stock"
            })
    return docs

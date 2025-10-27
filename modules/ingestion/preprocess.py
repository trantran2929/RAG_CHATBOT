import os, re, json, time, hashlib, unicodedata
from pathlib import Path
from typing import List, Dict, Set
import pandas as pd
from vnstock import Listing

CACHE_PATH = Path(os.getenv("TICKER_CACHE_PATH", "data/symbols.json"))
CACHE_TTL  = int(os.getenv("TICKER_CACHE_TTL", 24 * 3600))  # 1 ngày

INDEX_KEYWORDS = {
    "VNINDEX": [r"\bvn[- ]?index\b", r"\bvnindex\b"],
    "VN30":    [r"\bvn[- ]?30\b", r"\bvn30\b"],
    "HNX30":   [r"\bhnx[- ]?30\b", r"\bhnx30\b"],
    "HNX":     [r"\bhnx[- ]?index\b", r"\bhnxindex\b"],
    "UPCOM":   [r"\bupcom[- ]?index\b", r"\bupcomindex\b"],
}

SYMBOL_BLACKLIST: Set[str] = set([
    "USD", "GDP", "VND", "VNĐ", "USDOLLAR", "USDT", "BTC", "ETH",  
])

TOKEN_RE = re.compile(r"\b[A-Z]{2,5}\b")

def _load_cached_symbols() -> List[str] | None:
    try:
        if CACHE_PATH.exists() and time.time() - CACHE_PATH.stat().st_mtime <= CACHE_TTL:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _save_cached_symbols(symbols: List[str]):
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(symbols, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _load_all_tickers_from_vnstock() -> List[str]:
    listing = Listing()
    syms = listing.all_symbols()
    if isinstance(syms, list):
        out = syms
    else:
        out = None
        if hasattr(syms, "columns"):
            for col in ("symbol", "ticker", "code"):
                if col in syms.columns:
                    out = (syms[col].dropna().astype(str).str.upper().str.strip().unique().tolist())
                    break
        if out is None:
            out = list({str(x).strip().upper() for x in syms})
    out = [s for s in (x.strip().upper() for x in out) if 2 <= len(s) <= 5]
    out = [s for s in out if s not in SYMBOL_BLACKLIST]
    return sorted(list(set(out)))

def get_all_tickers(force_refresh: bool = False) -> Set[str]:
    if not force_refresh:
        cached = _load_cached_symbols()
        if cached:
            return set(cached)
    syms = _load_all_tickers_from_vnstock()
    if syms:
        _save_cached_symbols(syms)
    return set(syms)

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")

def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s\.,!?\-:;/()\"'%…–]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, max_words: int = 220, overlap: float = 0.01) -> List[str]:
    """
    Sliding window theo từ: max_words, overlap rất nhỏ.
    """
    words = (text or "").split()
    if not words:
        return []
    step = max(1, int(max_words * (1 - overlap)))
    chunks = []
    for i in range(0, len(words), step):
        w = words[i:i + max_words]
        if not w:
            break
        chunk = " ".join(w).strip()
        if chunk and (not chunks or chunk != chunks[-1]):
            chunks.append(chunk)
        if i + max_words >= len(words):
            break
    return chunks

def _extract_index_codes(text: str) -> List[str]:
    t = (text or "").lower()
    codes = []
    for code, pats in INDEX_KEYWORDS.items():
        if any(re.search(p, t) for p in pats):
            codes.append(code)
    seen = set()
    out = []
    for c in codes:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def _extract_symbols(title: str, content: str, universe: Set[str]) -> List[str]:
    """Lọc token A-Z 2–5 ký tự, thuộc universe (vnstock), bỏ blacklist."""
    text = f"{title or ''} {content or ''}"
    cands = set(TOKEN_RE.findall(text.upper()))
    syms = [s for s in cands if s in universe and s not in SYMBOL_BLACKLIST]
    found = []
    for token in TOKEN_RE.findall(text.upper()):
        if token in syms and token not in found:
            found.append(token)
    return found

def _to_time_ts(time_str: str) -> int:
    """
    Nhận time dạng 'dd-MM-YYYY HH:MM:SS' -> epoch (giây) ICT.
    Nếu parse fail -> dùng now (UTC) (an toàn cho demo).
    """
    try:
        ts = pd.to_datetime(time_str, format="%d-%m-%Y %H:%M:%S", errors="raise")
        ts = ts.tz_localize("Asia/Ho_Chi_Minh").tz_convert("UTC")
        return int(ts.timestamp())
    except Exception:
        return int(pd.Timestamp.utcnow().timestamp())

def preprocess_articles(articles: List[Dict], max_words: int = 220) -> List[Dict]:
    """
    Trả về list docs cho loader:
      id, title, url, content, summary, source, symbols(List[str]), index_codes(List[str]),
      time_ts(int), sentiment(None), label(None)
    """
    universe = get_all_tickers()
    out = []
    for a in articles:
        title = a.get("title", "").strip()
        content = a.get("content", "").strip()
        summary = a.get("summary", "").strip()
        clean = clean_text(content) or clean_text(summary) or clean_text(title)
        if not clean:
            continue
        time_str = a.get("time", "")
        time_ts = _to_time_ts(time_str)

        symbols = _extract_symbols(title, clean, universe)
        index_codes = _extract_index_codes(" ".join([title, summary, content]))

        chunks = chunk_text(clean, max_words=max_words, overlap=0.001)
        if not chunks:
            continue
        for idx, ch in enumerate(chunks):
            base = f"{a.get('url','')}_{title}_{time_ts}_{idx}"
            pid = hashlib.md5(base.encode("utf-8")).hexdigest()

            out.append({
                "id": pid,
                "title": title,
                "url": a.get("url",""),
                "time": time_str,
                "time_ts": int(time_ts),  
                "summary": summary,
                "content": ch,
                "symbols": symbols,           
                "index_codes": index_codes,       
                "sentiment": None,
                "label": None,
                "source": a.get("source","cafef"),
            })
    return out

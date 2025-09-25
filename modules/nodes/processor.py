import re
import unicodedata
from langdetect import detect, DetectorFactory
from difflib import get_close_matches

DetectorFactory.seed = 0  # langdetect ổn định kết quả

class Processor:
    def __init__(self, target_lang="vi", synonyms=None, stopwords=None):
        """
        target_lang: ngôn ngữ chính của dữ liệu (vd: 'vi' cho tiếng Việt)
        synonyms: dict ánh xạ từ đồng nghĩa/alias
        stopwords: list stopwords cần loại bỏ
        """
        self.target_lang = target_lang
        self.synonyms = synonyms or {}
        self.stopwords = stopwords or []

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", text)  # giữ full chữ có dấu
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            return lang
        except:
            return self.target_lang


    # Dịch (placeholder)
    def translate_if_needed(self, text: str, lang: str) -> str:
        if lang != self.target_lang and lang != "unknown":
            # TODO: gọi Google Translate hoặc LLM
            translated_text = text  
            return translated_text
        return text

    # Ánh xạ alias / synonym 
    def map_synonyms(self, text: str) -> str:
        words = text.split()
        mapped = [self.synonyms.get(w, w) for w in words]
        return " ".join(mapped)

    # Loại bỏ stopwords 
    def remove_stopwords(self, text: str) -> str:
        words = text.split()
        filtered = [w for w in words if w not in self.stopwords]
        return " ".join(filtered)

    # Sửa chính tả đơn giản (fuzzy match) 
    def correct_typo(self, text: str, vocab: list) -> str:
        words = text.split()
        corrected = []
        for w in words:
            match = get_close_matches(w, vocab, n=1, cutoff=0.8)
            corrected.append(match[0] if match else w)
        return " ".join(corrected)

    # Chuẩn hóa query tổng hợp
    def process_query(self, state, vocab: list = None):
        """
        update state.clean_query và state.lang
        """
        query = state.user_query
        query = self.normalize(query)
        lang = self.detect_language(query)
        query = self.translate_if_needed(query, lang)
        query = self.map_synonyms(query)
        query = self.remove_stopwords(query)
        if vocab:
            query = self.correct_typo(query, vocab)
        
        state.processed_query = query
        state.lang = lang

        return state
    
processor_instance = Processor()
def processor_query(state, vocab: list = None):
    return processor_instance.process_query(state,vocab)

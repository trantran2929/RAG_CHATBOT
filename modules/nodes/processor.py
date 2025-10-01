import re
import unicodedata
from langdetect import detect, DetectorFactory
from difflib import get_close_matches

DetectorFactory.seed = 0  # langdetect ổn định kết quả

class Processor:
    def __init__(self, target_lang="vi", synonyms=None, stopwords=None, greetings=None):
        """
        target_lang: ngôn ngữ chính của dữ liệu (vd: 'vi' cho tiếng Việt)
        synonyms: dict ánh xạ từ đồng nghĩa/alias
        stopwords: list stopwords cần loại bỏ
        """
        self.target_lang = target_lang
        self.synonyms = {k.lower(): v.lower() for k, v in (synonyms or {}).items()}
        self.stopwords = [w.lower() for w in (stopwords or [])]
        self.greetings = [g.lower() for g in (greetings or ["hi", "hello", "chào", "chao", "chào bạn", "xin chào"])]
        self.finance_keywords = [
            "cổ phiếu", "chứng khoán", "thị trường", "vốn hóa", "lợi nhuận", "doanh thu",
            "tăng trưởng", "đầu tư", "quỹ", "trái phiếu", "lãi suất", "ngân hàng",
            "tài chính", "báo cáo tài chính", "cổ tức", "thị trường chứng khoán",
            "chỉ số", "vĩ mô", "phân tích kỹ thuật",
            "phân tích cơ bản", "bảng cân đối kế toán", "dòng tiền", "báo cáo thu nhập",
            "thị trường tiền tệ", "thị trường vốn", "chính sách tiền tệ",
            "rủi ro", "lợi nhuận ròng", "tỷ suất sinh lời", "đầu cơ", "bong bóng",
            "thị trường phái sinh", "hợp đồng tương lai", "quyền chọn",
            "ngoại hối", "tiền tệ", "vàng", "bạc", "hàng hóa",
            "chính sách tài khóa", "lạm phát", "thất nghiệp", "GDP",
            "ngân hàng trung ương", "FED", "ECB", "NHTW", "thị trường bất động sản",
            "bất động sản", "cung tiền", "cầu tiền", "thị trường lao động",
            "tỷ giá", "tỷ giá hối đoái", "thị trường trái phiếu", "chứng chỉ quỹ",
            "phân tích định lượng", "phân tích định tính", "chiến lược đầu tư",
            "quản lý rủi ro", "đòn bẩy tài chính", "thị trường sơ cấp",
            "thị trường thứ cấp", "mua bán cổ phiếu", "sở giao dịch",
            "công ty chứng khoán", "tư vấn đầu tư", "phân tích thị trường",
            "báo cáo phân tích", "tín dụng", "nợ xấu", "tài sản",
            "danh mục đầu tư", "quản lý quỹ", "tài chính cá nhân",
            "ngân hàng số", "fintech", "blockchain", "tiền điện tử",
            "bitcoin", "ethereum", "altcoin", "ICO", "DeFi"
        ]

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
    # def translate_if_needed(self, text: str, lang: str) -> str:
    #     if lang != self.target_lang and lang != "unknown":
    #         # TODO: gọi Google Translate hoặc LLM
    #         translated_text = text  
    #         return translated_text
    #     return text

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

    def is_greeting(self, text: str) -> bool:
        norm_text = self.normalize(text)
        if len(norm_text.split())<=3 and norm_text in self.greetings:
            return True
        for greet in self.greetings:
            if re.search(rf"\b{re.escape(greet)}\b", norm_text):
                return True
        if len(norm_text.split())<=3:
            match = get_close_matches(norm_text, self.greetings, n=1, cutoff=0.8)
            return bool(match)
        return False
    # Chuẩn hóa query tổng hợp
    def process_query(self, state, vocab: list = None):
        """
        update state.clean_query và state.lang
        """
        user_query = state.user_query
        processed_query = self.normalize(user_query)
        lang = self.detect_language(processed_query)
        # processed_query = self.translate_if_needed(processed_query, lang)
        processed_query = self.map_synonyms(processed_query)
        processed_query = self.remove_stopwords(processed_query)
        if vocab:
            processed_query = self.correct_typo(processed_query, vocab)

        # Nếu query chứa từ khóa tài chính → không coi là greeting
        is_greeting = self.is_greeting(user_query)
        for kw in self.finance_keywords:
            if kw in processed_query:
                is_greeting = False
                break
        state.processed_query = processed_query
        state.lang = lang
        state.is_greeting = is_greeting
        state.extract_tickers()

        return state
    
processor_instance = Processor()
def processor_query(state, vocab: list = None):
    return processor_instance.process_query(state,vocab)

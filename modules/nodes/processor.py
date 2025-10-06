import re
import unicodedata
from langdetect import detect, DetectorFactory
from difflib import get_close_matches
from modules.api.weather_api import get_weather,normalize_city_name
from modules.api.stock_api import get_stock_price
from modules.api.time_api import format_full, get_now
import pytz
from datetime import timedelta, datetime
DetectorFactory.seed = 0  # langdetect ổn định kết quả

class Processor:
    def __init__(self, target_lang="vi", synonyms=None, stopwords=None, greetings=None):

        self.target_lang = target_lang
        self.synonyms = {k.lower(): v.lower() for k, v in (synonyms or {}).items()}
        self.stopwords = [w.lower() for w in (stopwords or [])]
        self.greetings = [g.lower() for g in (greetings or ["hi", "hello", "chào", "chao", "chào bạn", "xin chào"])]
        self.finance_keywords = [
            "cổ phiếu", "chứng khoán", "thị trường", "vốn hóa", "lợi nhuận",
            "doanh thu", "tăng trưởng", "đầu tư", "trái phiếu", "lãi suất",
            "ngân hàng", "cổ tức", "GDP", "bitcoin", "ethereum", "DeFi"
        ]
        self.weather_keywords = ["thời tiết", "nhiệt độ", "mưa", "nắng"]
        self.time_keywords = ["mấy giờ", "bây giờ", "hôm nay", "ngày mấy", "thứ mấy"]

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
    def detect_intent(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["tin", "tin tức", "cập nhập", "thị trường"]):
            return "news"
        if any(k in q for k in self.finance_keywords):
            return "stock"
        if any(k in q for k in self.weather_keywords):
            return "weather"
        if any(k in q for k in self.time_keywords):
            return "time"
        return "rag"
    def detect_time_filter(self,query: str):
        """
        Trả về tuple (start_ts, end_ts) dạng epoch timestamp nếu query có từ khóa thời gian.
        Nếu không match thì trả về None.
        """
        vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
        now = datetime.now(vn_tz)

        today_start = vn_tz.localize(datetime(now.year, now.month, now.day, 0, 0, 0))
        today_end   = vn_tz.localize(datetime(now.year, now.month, now.day, 23, 59, 59))

        yesterday_start = today_start - timedelta(days=1)
        yesterday_end   = today_end - timedelta(days=1)

        tomorrow_start = today_start + timedelta(days=1)
        tomorrow_end   = today_end + timedelta(days=1)

        last_week_start = today_start - timedelta(days=7)
        next_week_end   = today_end + timedelta(days=7)

        q = query.lower()

        if "hôm nay" in q:
            return (int(today_start.timestamp()), int(today_end.timestamp()))
        elif "hôm qua" in q:
            return (int(yesterday_start.timestamp()), int(yesterday_end.timestamp()))
        elif "ngày mai" in q or q.startswith("mai "):
            return (int(tomorrow_start.timestamp()), int(tomorrow_end.timestamp()))
        elif "tuần trước" in q:
            return (int(last_week_start.timestamp()), int(today_end.timestamp()))
        elif "tuần sau" in q:
            return (int(today_start.timestamp()), int(next_week_end.timestamp()))
        
        return None

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
        state.intent = self.detect_intent
        state.time_filter = self.detect_time_filter(user_query)

        return state
    
processor_instance = Processor()
def processor_query(state, vocab: list = None):
    return processor_instance.process_query(state,vocab)

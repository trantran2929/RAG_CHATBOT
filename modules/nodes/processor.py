import re
import unicodedata
from langdetect import detect, DetectorFactory
from difflib import get_close_matches
import pytz
from datetime import timedelta, datetime
from vnstock import Listing
DetectorFactory.seed = 0  # langdetect ổn định kết quả

class Processor:
    def __init__(self, target_lang="vi", synonyms=None, stopwords=None, greetings=None):

        self.target_lang = target_lang
        self.synonyms = {k.lower(): v.lower() for k, v in (synonyms or {}).items()}
        self.stopwords = [w.lower() for w in (stopwords or [])]
        self.greetings = [g.lower() for g in (greetings or ["hi", "hello", "chào", "chao", "chào bạn", "xin chào"])]
        self.finance_keywords = [
            "cổ phiếu", "chứng khoán", "thị trường", "vnindex", "vn-index",
            "vni", "vn30", "hnx", "upcom", "vốn hóa", "doanh thu", "tăng trưởng",
            "đầu tư", "trái phiếu", "lãi suất", "cổ tức", "bitcoin", "crypto"
        ]
        self.weather_keywords = ["thời tiết", "nhiệt độ", "mưa", "nắng"]
        self.time_keywords = ["mấy giờ", "bây giờ", "hôm nay", "ngày mấy", "thứ mấy"]
        self.market_indices = {"VNINDEX", "VN-INDEX", "VNI", "VN30", "HNX", "UPCOM"}

        try:
            listing = Listing(source="VCI")
            df = listing.all_symbols()
            self.valid_tickers = set(df["symbol"].dropna().str.upper().tolist())
        except Exception as e:
            print(f"[Processor] ⚠️ Không thể tải danh sách mã chứng khoán ({e})")
            self.valid_tickers = set()

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", text)  # giữ full chữ có dấu
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def detect_language(self, text: str) -> str:
        try:
            if len(text.split()) < 3:
                return self.target_lang
            return detect(text)
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
    
    def detect_tickers(self, text: str) -> str:
        """
        Tìm mã chứng khoán hoặc chỉ số thị trường trong câu hỏi.
        Luôn lọc ticker rỗng và chỉ giữ lại mã hợp lệ.
        """
        text_upper = text.upper()

        # Tìm chuỗi viết hoa từ 2-6 kí tự
        potential = re.findall(r"\b[A-Z]{2,6}\b", text_upper)
        aliases = {"VNI": "VNINDEX", "VN-INDEX": "VNINDEX"}

        # Danh sách loại trừ các từ dễ nhầm là tiếng Việt
        invalid_tickers = {"TIN", "MUA", "BAN", "SON", "TOI", "CON", "AN", "DEP", "DO", "XANH"}

        clean_list = []
        for t in potential:
            t = aliases.get(t, t.strip().upper())
            if (
                3 <= len(t) <= 10
                and t.isalpha()
                and (t in self.valid_tickers or t in self.market_indices)
                and t not in invalid_tickers
            ):
                clean_list.append(t)

        # Nếu chưa có, thử tìm theo pattern “cổ phiếu XYZ”
        if not clean_list:
            match = re.findall(r"(?:cổ phiếu|mã)\s+([A-Z]{2,6})", text_upper)
            for t in match:
                t = t.strip().upper()
                if (
                    3 <= len(t) <= 10
                    and (t in self.valid_tickers or t in self.market_indices)
                    and t not in invalid_tickers
                ):
                    clean_list.append(t)

        return sorted(set(clean_list))

    def detect_intent(self, query: str) -> str:
        q = query.lower()
        tickers = self.detect_tickers(query)
        if any(k in q for k in ["phân tích", "xu hướng", "thị trường", "nhận định", "biến động"]):
            return "market"
        if tickers or any(k in q for k in self.finance_keywords):
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
        
        match = re.search(r"ngày\s*(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?", q)
        if match:
            day, month, year = int(match.group(1)), int(match.group(2)), match.group(3)
            year = int(year) if year else now.year
            try:
                dt_start = vn_tz.localize(datetime(year, month, day, 0, 0, 0))
                dt_end = vn_tz.localize(datetime(year, month, day, 23, 59, 59))
                return (int(dt_start.timestamp()), int(dt_end.timestamp()))
            except Exception:
                pass

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
        state.intent = self.detect_intent(user_query)
        state.time_filter = self.detect_time_filter(user_query)
        state.tickers = self.detect_tickers(user_query)

        return state
    
processor_instance = Processor()
def processor_query(state, vocab: list = None):
    return processor_instance.process_query(state,vocab)

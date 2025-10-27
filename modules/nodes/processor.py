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
        self.greetings = [
            g.lower()
            for g in (greetings or ["hi", "hello", "chào", "chao", "chào bạn", "xin chào", "alo"])
        ]

        self.finance_keywords = [
            "cổ phiếu", "chứng khoán", "thị trường", "vnindex", "vn-index",
            "vni", "vn30", "hnx", "upcom", "vốn hóa", "doanh thu", "tăng trưởng",
            "đầu tư", "trái phiếu", "lãi suất", "cổ tức", "bitcoin", "crypto"
        ]
        self.news = ["tin", "tin tức", "điểm nóng", "điểm nong", "đáng chú ý", "cập nhật", "diễn biến"]
        self.weather_keywords = ["thời tiết", "nhiệt độ", "mưa", "nắng"]
        self.time_keywords = ["mấy giờ", "bây giờ", "hôm nay", "ngày mấy", "thứ mấy"]
        self.forecast_keywords = [
            "dự báo", "forecast", "ước tính", "dự đoán", "tiên lượng",
            "kịch bản", "phiên tới"
        ]
        self.market_indices = {"VNINDEX", "VN-INDEX", "VNI", "VN30", "HNX", "UPCOM"}
        self.HISTORY_KEYWORDS = [
            "lịch sử", "quá khứ", "hôm qua", "trước đó", "giai đoạn", "trong",
            "5 ngày", "7 ngày", "10 ngày", "30 ngày", "60 ngày", "90 ngày",
            "1 tháng", "3 tháng", "6 tháng", "9 tháng", "12 tháng", "1 năm",
            "ytd", "y-t-d"
        ]

        self.advice_keywords = [
            "có nên mua", "mua được không", "nên mua", "có nên bán", "bán được không", "nên bán",
            "mua hay bán", "khuyến nghị mua", "khuyến nghị bán", "lời khuyên đầu tư",
            "gợi ý đầu tư", "mua lúc nào", "còn tăng", "còn giảm",
            "còn lên", "còn xuống",
        ]


        try:
            listing = Listing(source="VCI")
            df = listing.all_symbols()
            self.valid_tickers = set(df["symbol"].dropna().str.upper().tolist())
        except Exception as e:
            print(f"[Processor] Không thể tải danh sách mã chứng khoán ({e})")
            self.valid_tickers = set()

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def detect_language(self, text: str) -> str:
        try:
            if len(text.split()) < 3:
                return self.target_lang
            return detect(text)
        except:
            return self.target_lang

    def map_synonyms(self, text: str) -> str:
        words = text.split()
        mapped = [self.synonyms.get(w, w) for w in words]
        return " ".join(mapped)

    def remove_stopwords(self, text: str) -> str:
        words = text.split()
        filtered = [w for w in words if w not in self.stopwords]
        return " ".join(filtered)

    def correct_typo(self, text: str, vocab: list) -> str:
        words = text.split()
        corrected = []
        for w in words:
            match = get_close_matches(w, vocab, n=1, cutoff=0.8)
            corrected.append(match[0] if match else w)
        return " ".join(corrected)

    def is_greeting(self, text: str) -> bool:
        norm_text = self.normalize(text)
        if len(norm_text.split()) <= 3 and norm_text in self.greetings:
            return True
        for greet in self.greetings:
            if re.search(rf"\b{re.escape(greet)}\b", norm_text):
                return True
        if len(norm_text.split()) <= 3:
            match = get_close_matches(norm_text, self.greetings, n=1, cutoff=0.8)
            return bool(match)
        return False

    def detect_tickers(self, text: str) -> list[str]:
        text_upper = text.upper()
        potential = re.findall(r"\b[A-Z]{2,6}\b", text_upper)
        aliases = {"VNI": "VNINDEX", "VN-INDEX": "VNINDEX"}

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
        asking_price_keywords = [
            "giá", "bao nhiêu", "mấy nghìn", "tăng hay giảm",
            "biến động thế nào trong phiên", "phần trăm", "%"
        ]

        # 1. Hỏi tin tức / diễn biến / cập nhật về 1 mã cụ thể
        #    ví dụ: "Hôm nay có tin tức gì mới về VCB?",
        #           "VCB có diễn biến gì đáng chú ý?"
        # → Đây không chỉ là hỏi giá, mà là hỏi bối cảnh + news.
        # → Trả về "market" để router dùng nhánh HYBRID (api + rag).
        if tickers and any(k in q for k in self.news):
            return "market"

        # 2. Hỏi tin tức chung chung (không ticker)
        #    ví dụ: "có tin gì nóng không", "cập nhật thị trường hôm nay"
        # → Thuần RAG.
        if any(k in q for k in self.news):
            return "rag"

        # 3. Hỏi lời khuyên mua/bán, khuyến nghị, còn lên/giảm,...
        #    (có ticker) → market (để phân tích broader context)
        if tickers and any(k in q for k in self.advice_keywords):
            return "market"

        # 4. Dự báo / dự đoán / forecast
        if any(k in q for k in self.forecast_keywords):
            return "forecast"

        # 5. Các câu phân tích xu hướng thị trường, dòng tiền, khối ngoại...
        #    Nếu có ticker là index (VNINDEX, VN30...) và họ hỏi về giá %, thì coi như "stock"
        #    Nếu không, thì "market".
        if any(k in q for k in ["phân tích", "xu hướng", "thị trường", "nhận định", "biến động", "dòng tiền", "khối ngoại"]):
            if tickers and any(t in self.market_indices for t in tickers):
                if any(k in q for k in asking_price_keywords):
                    return "stock"
            return "market"

        # 6. User nhắc tới ticker cụ thể
        if tickers:
            # Nếu ticker là index như VNINDEX/VN30/HNX...
            if any(t in self.market_indices for t in tickers):
                # hỏi giá/phần trăm? -> stock
                if any(k in q for k in asking_price_keywords):
                    return "stock"
                # nếu không hỏi giá cụ thể -> market (bối cảnh vĩ mô)
                return "market"

            # ticker là mã cổ phiếu (VCB, FPT,...)
            # Nếu câu hỏi là giá / biến động / phần trăm -> stock (lấy quote trực tiếp)
            if any(k in q for k in asking_price_keywords):
                return "stock"

            # còn lại (ví dụ: "VCB dạo này sao", "đánh giá VCB")
            # -> market để cho phép hybrid phân tích bối cảnh thay vì chỉ trả giá
            return "market"

        # 7. Nếu không có ticker nhưng có từ khóa tài chính chung
        if any(k in q for k in self.finance_keywords):
            return "market"

        # 8. Weather/time
        if any(k in q for k in self.weather_keywords):
            return "weather"
        if any(k in q for k in self.time_keywords):
            return "time"

        # 9. fallback
        return "rag"


    def detect_time_filter(self, query: str):
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

    def days_from_time_filter(self, tf) -> int:
        if not tf or not isinstance(tf, tuple) or len(tf) != 2:
            return 30
        start_ts, end_ts = tf
        try:
            secs = max(0, int(end_ts) - int(start_ts))
            days = int(round(secs / 86400.0)) or 1
            return max(1, min(days, 365))
        except Exception:
            return 30

    def parse_history_window(self, text: str, default_days: int = 30) -> int:
        q = (text or "").lower().strip()

        m = re.search(r"(\d+)\s*ngày", q)
        if m:
            d = int(m.group(1))
            return max(3, min(d, 365))

        m = re.search(r"(\d+)\s*tháng", q)
        if m:
            months = int(m.group(1))
            days_map = {1: 30, 3: 90, 6: 180, 9: 270, 12: 360}
            d = days_map.get(months, months * 30)
            return max(7, min(int(d), 365))

        if re.search(r"\b(1\s*năm|một\s*năm)\b", q):
            return 360

        if "ytd" in q or "y-t-d" in q:
            try:
                tz = pytz.timezone("Asia/Ho_Chi_Minh")
                now = datetime.now(tz)
                start = tz.localize(datetime(now.year, 1, 1))
                days = max(7, min(int((now - start).days), 365))
                return days
            except Exception:
                return 120

        return default_days

    def resolve_history_request(self, text: str, time_filter: tuple | None, default_days: int = 30) -> tuple[bool, int | None]:
        q = (text or "").lower().strip()

        if "hôm nay" in q:
            return (False, None)

        if "hôm qua" in q:
            return (True, 1)

        if time_filter is not None:
            days = self.days_from_time_filter(time_filter)
            return (True, max(1, days))

        has_kw = any(k in q for k in self.HISTORY_KEYWORDS)
        has_num = (
            bool(re.search(r"(\d+)\s*(ngày|tháng)", q))
            or ("ytd" in q)
            or ("y-t-d" in q)
            or bool(re.search(r"\b(1\s*năm|một\s*năm)\b", q))
        )

        if has_kw or has_num:
            days = self.parse_history_window(q, default_days=default_days)
            return (True, max(1, days))

        return (False, None)

    def process_query(self, state, vocab: list = None):
        user_query = state.user_query
        processed_query = self.normalize(user_query)
        lang = self.detect_language(processed_query)

        processed_query = self.map_synonyms(processed_query)
        processed_query = self.remove_stopwords(processed_query)

        if vocab:
            processed_query = self.correct_typo(processed_query, vocab)

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

        # cache_key gợi ý: bản query chuẩn hóa
        state.cache_key = f"qa::{processed_query[:100]}"

        # thêm debug tiện theo dõi
        if getattr(state, "add_debug", None):
            state.add_debug("processor_intent", state.intent)
            state.add_debug("processor_tickers", state.tickers)
            state.add_debug("processor_lang", state.lang)
            state.add_debug("processor_cache_key", state.cache_key)

        return state


processor_instance = Processor()

def processor_query(state, vocab: list = None):
    return processor_instance.process_query(state, vocab)

import re
import unicodedata
import json
import os
from pathlib import Path
from langdetect import detect, DetectorFactory
from difflib import get_close_matches
import pytz
from datetime import timedelta, datetime
from vnstock import Listing
from unidecode import unidecode
from modules.api.time_api import normalize_time_query

from collections import defaultdict
from typing import List, Tuple

DetectorFactory.seed = 0  # langdetect ổn định kết quả


# Alias thủ công cho một số mã phổ biến
MANUAL_ALIASES = {
    "VCB": ["vietcombank", "ngan hang ngoai thuong", "ngoai thuong viet nam"],
    "CTG": ["vietinbank", "ngan hang cong thuong"],
    "BID": ["bidv", "ngan hang dau tu va phat trien"],
    "TCB": ["techcombank"],
    "MBB": ["mbbank", "quan doi"],
    "VPB": ["vpbank"],
    "FPT": ["fpt"],
    "HPG": ["hoa phat"],
    # có thể bổ sung dần
}

# Conservative phrase-level corrections. These target domain phrases rather
# than arbitrary words, avoiding accidental changes to names and stock symbols.
TYPO_PATTERNS = [
    (r"\bth[ơo]i\s+ti[eế]t\b", "thời tiết"),
    (r"\bch[uứ]ng\s+kho[aá]n\b", "chứng khoán"),
    (r"\bc[oổ]\s+phi[eế]u\b", "cổ phiếu"),
    (r"\bthi\s+tru[oờ]ng\b", "thị trường"),
    (r"\bdu\s+b[aá]o\b", "dự báo"),
    (r"\bng[aà]i\s+mai\b", "ngày mai"),
    (r"\bh[oô]m\s+n[aà]y\b", "hôm nay"),
    (r"\bh[oô]m\s+qua\b", "hôm qua"),
    (r"\bm[aấ]y\s+gi[oờ]\b", "mấy giờ"),
    (r"\bb[aâ]y\s+gi[oờ]\b", "bây giờ"),
    (r"\bth[uứ]\s+m[aấ]y\b", "thứ mấy"),
    (r"\bng[aà]y\s+bao\s+nhi[eê]u\b", "ngày bao nhiêu"),
]


class Processor:
    def __init__(self, target_lang="vi", synonyms=None, stopwords=None, greetings=None):
        self.target_lang = target_lang
        self.synonyms = {k.lower(): v.lower() for k, v in (synonyms or {}).items()}
        self.stopwords = [w.lower() for w in (stopwords or [])]
        self.greetings = [
            g.lower()
            for g in (
                greetings
                or ["hi", "hello", "chào", "chao", "chào bạn", "xin chào", "alo"]
            )
        ]

        # ====== Từ khóa nhận diện intent ======
        self.finance_keywords = [
            "cổ phiếu", "chứng khoán", "thị trường", "vnindex", "vn-index",
            "vni", "vn30", "hnx", "upcom", "vốn hóa", "doanh thu", "tăng trưởng",
            "đầu tư", "trái phiếu", "lãi suất", "cổ tức", "bitcoin", "crypto"
        ]
        self.news = [
            "tin", "tin tức", "điểm nóng", "điểm nong",
            "đáng chú ý", "cập nhật", "diễn biến"
        ]
        self.weather_keywords = ["thời tiết", "nhiệt độ", "mưa", "nắng"]
        self.time_keywords = [
            "mấy giờ", "bây giờ", "giờ hiện tại", "hiện tại là mấy giờ",
            "ngày mấy", "ngày bao nhiêu", "thứ mấy", "thứ gì", "ngày nào",
            "hôm nay là ngày", "hôm nay là thứ", "ngày mai", "ngày kia",
            "hôm qua", "còn bao lâu", "bao nhiêu ngày nữa", "cuối năm",
            "hết năm",
        ]
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
            "có nên mua", "mua được không", "nên mua", "có nên bán", "bán được không",
            "nên bán", "mua hay bán", "khuyến nghị mua", "khuyến nghị bán",
            "lời khuyên đầu tư", "gợi ý đầu tư", "mua lúc nào", "còn tăng",
            "còn giảm", "còn lên", "còn xuống",
        ]

        # ====== Master ticker + alias index ======
        self.valid_tickers = set()
        self.symbol_alias_index: dict[str, set[str]] = {}
        # alias_stop_tokens: các alias quá chung, không dùng để map mã
        self.alias_stop_tokens: set[str] = set()

        # Prefer the repository cache so startup does not depend on Vnstock/VCI.
        # Set REFRESH_TICKERS=1 to bypass the cache and refresh from the API.
        cache_path = Path(os.getenv("TICKER_CACHE_PATH", "data/symbols.json"))
        if os.getenv("REFRESH_TICKERS", "0") != "1":
            try:
                cached_symbols = json.loads(cache_path.read_text(encoding="utf-8"))
                self.valid_tickers = {
                    str(symbol).strip().upper()
                    for symbol in cached_symbols
                    if str(symbol).strip()
                }

                if self.valid_tickers:
                    alias_index: dict[str, set[str]] = {}
                    for symbol in self.valid_tickers:
                        aliases = {self._normalize_alias(symbol)}
                        aliases.update(
                            self._normalize_alias(alias)
                            for alias in MANUAL_ALIASES.get(symbol, [])
                        )
                        for alias in aliases:
                            if alias:
                                alias_index.setdefault(alias, set()).add(symbol)

                    self.symbol_alias_index = alias_index
                    self.alias_stop_tokens = {
                        "hom", "nay", "mai", "qua", "toi", "sang", "chieu",
                        "dem", "trua", "co", "phieu", "gia", "mua", "ban",
                        "ngay", "thang", "nam",
                    }
                    print(
                        f"[Processor] Loaded {len(self.valid_tickers)} tickers "
                        f"from {cache_path}"
                    )
                    return
            except (OSError, ValueError, TypeError) as e:
                print(f"[Processor] Cannot load ticker cache {cache_path}: {e}")

        try:
            listing = Listing(source="VCI")
            df = listing.all_symbols()

            if df is None or df.empty:
                raise ValueError("Listing all_symbols() empty")

            # Chuẩn hóa tên cột
            df = df.rename(
                columns={
                    "symbol": "symbol",
                    "stock_code": "symbol",
                    "ticker": "symbol",
                    "stock_name": "stock_name",
                    "organName": "stock_name",
                    "organ_name": "stock_name",
                    "company_name": "stock_name",
                    "companyName": "stock_name",
                }
            )

            df["symbol"] = df["symbol"].astype(str).str.upper()
            if "stock_name" not in df.columns:
                df["stock_name"] = ""

            self.valid_tickers = set(df["symbol"].dropna().str.upper().tolist())

            alias_index: dict[str, set[str]] = {}

            for _, row in df.iterrows():
                sym = str(row["symbol"]).upper()
                if not sym:
                    continue

                raw_name = str(row.get("stock_name") or "")
                aliases = set()

                # alias từ chính mã
                aliases.add(self._normalize_alias(sym))

                # alias từ tên công ty
                if raw_name:
                    aliases.add(self._normalize_alias(raw_name))

                # alias thủ công (nếu có)
                if sym in MANUAL_ALIASES:
                    for a in MANUAL_ALIASES[sym]:
                        aliases.add(self._normalize_alias(a))

                for a in aliases:
                    if not a:
                        continue
                    alias_index.setdefault(a, set()).add(sym)

            self.symbol_alias_index = alias_index

            # Các alias quá chung / mang nghĩa thời gian -> không dùng để map mã
            self.alias_stop_tokens = {
                "hom", "nay", "mai", "qua", "toi", "sang", "chieu", "dem",
                "trua", "co", "phieu", "gia", "mua", "ban", "ngay", "thang", "nam"
            }

        except Exception as e:
            print(f"[Processor] Không thể tải danh sách mã chứng khoán / alias ({e})")
            self.valid_tickers = set()
            self.symbol_alias_index = {}
            self.alias_stop_tokens = set()

    # ====== Helper cho alias ======
    def _normalize_alias(self, s: str) -> str:
        """
        Dùng cho so khớp alias:
        - lower
        - bỏ dấu tiếng Việt
        - bỏ khoảng trắng + ký tự không alnum
        """
        s = s or ""
        s = s.lower()
        s = unidecode(s)
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s

    # ====== Core text processing ======
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

    def correct_domain_typos(self, text: str) -> tuple[str, list[dict[str, str]]]:
        """Correct known domain phrases while preserving numbers/proper nouns."""
        corrected = re.sub(r"\s+", " ", (text or "")).strip()
        changes: list[dict[str, str]] = []

        for pattern, replacement in TYPO_PATTERNS:
            updated, count = re.subn(pattern, replacement, corrected, flags=re.I)
            if count and updated != corrected:
                changes.append({"from": corrected, "to": updated})
                corrected = updated

        updated = normalize_time_query(corrected)
        if updated != corrected:
            changes.append({"from": corrected, "to": updated})
            corrected = updated

        # "háng" is treated as "tháng" only in an explicit date context.
        if re.search(r"\bngày\s+\d{1,2}\b", corrected, flags=re.I):
            updated, count = re.subn(r"\bháng\b", "tháng", corrected, flags=re.I)
            if count and updated != corrected:
                changes.append({"from": corrected, "to": updated})
                corrected = updated

            updated, count = re.subn(
                r"(\bngày\s+\d{1,2}\s+)(?:t|thg)(?=\s+\d{1,2}\b)",
                r"\1tháng",
                corrected,
                flags=re.I,
            )
            if count and updated != corrected:
                changes.append({"from": corrected, "to": updated})
                corrected = updated

        # Sửa "lag" thành "là" chỉ khi đứng ngay trước cụm hỏi thời gian.
        updated, count = re.subn(
            r"\blag(?=\s+(?:thứ\s+(?:mấy|gì)|ngày\s+(?:nào|mấy|bao nhiêu)))",
            "là",
            corrected,
            flags=re.I,
        )
        if count and updated != corrected:
            changes.append({"from": corrected, "to": updated})
            corrected = updated

        return corrected, changes

    @staticmethod
    def _contains_any(text: str, keywords) -> bool:
        """Match domain keywords with or without Vietnamese diacritics."""
        plain_text = unidecode((text or "").lower())
        return any(unidecode(str(keyword).lower()) in plain_text for keyword in keywords)

    # ====== Resolver alias: query -> [(ticker, score)] ======
    def resolve_tickers_with_score(
        self, query: str, max_results: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Dùng alias (tên công ty, alias thủ công) để map query -> ticker.
        Trả về list (ticker, score) đã sort theo score giảm dần.
        """
        if not self.symbol_alias_index:
            return []

        q = query or ""
        q_lower = q.lower()

        # tách token chữ
        word_tokens = re.findall(r"[a-zA-ZÀ-ỹ0-9]+", q_lower)
        n = len(word_tokens)
        scores = defaultdict(float)

        # duyệt n-gram dài trước (3 từ -> 2 từ -> 1 từ)
        for length in [3, 2, 1]:
            if n < length:
                continue
            for i in range(0, n - length + 1):
                span = " ".join(word_tokens[i : i + length])
                norm = self._normalize_alias(span)
                if not norm:
                    continue

                # BỎ QUA CÁC ALIAS STOP (ví dụ: hom, nay, mai, ngay, ...)
                if norm in self.alias_stop_tokens:
                    continue

                if norm in self.symbol_alias_index:
                    tickers = self.symbol_alias_index[norm]
                    weight = float(length)  # n-gram dài hơn -> trọng số cao hơn
                    for tic in tickers:
                        scores[tic] += weight

        # build list (ticker, score) sort theo score giảm dần
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return items[:max_results]

    # ====== detect_tickers: regex + alias resolver ======
    def detect_tickers(self, text: str) -> list[str]:
        text_upper = text.upper()
        aliases = {"VNI": "VNINDEX", "VN-INDEX": "VNINDEX"}

        invalid_tickers = {
            "TIN", "MUA", "BAN", "SON", "TOI", "CON", "AN", "DEP", "DO", "XANH",
            "THU", "MAI", "NAY", "QUA", "NGAY", "NAM", "GIO"
        }

        found = set()
        explicit_tickers = {
            t.strip().upper()
            for t in re.findall(r"(?:CỔ PHIẾU|MÃ)\s+([A-Z]{2,6})", text_upper)
        }

        # 1) Regex bắt mã in hoa (VCB, FPT,...)
        potential = re.findall(r"\b[A-Z]{2,6}\b", text_upper)
        for t in potential:
            t = aliases.get(t, t.strip().upper())
            if (
                3 <= len(t) <= 10
                and t.isalpha()
                and (t in self.valid_tickers or t in self.market_indices)
                and (t not in invalid_tickers or t in explicit_tickers)
            ):
                found.add(t)

        # 2) Trường hợp 'cổ phiếu VCB', 'mã FPT'
        if not found:
            match = re.findall(r"(?:cổ phiếu|mã)\s+([A-Z]{2,6})", text_upper)
            for t in match:
                t = t.strip().upper()
                if (
                    3 <= len(t) <= 10
                    and (t in self.valid_tickers or t in self.market_indices)
                    and (t not in invalid_tickers or t in explicit_tickers)
                ):
                    found.add(t)

        # 3) Dùng alias resolver: 'vietcombank', 'hoa phat', 'mbbank'...
        alias_results = self.resolve_tickers_with_score(text, max_results=5)
        for tic, score in alias_results:
            if tic not in invalid_tickers:
                found.add(tic)

        return sorted(found)

    # ====== Detect intent ======
    def detect_intent(self, query: str) -> str:
        q = re.sub(r"\s+", " ", (query or "").lower()).strip()

        # Weather must be resolved before the generic forecast keyword.
        if self._contains_any(q, self.weather_keywords):
            return "weather"

        tickers = self.detect_tickers(query)
        asking_price_keywords = [
            "giá", "bao nhiêu", "mấy nghìn", "tăng hay giảm",
            "biến động thế nào trong phiên", "phần trăm", "%"
        ]

        # 1. Hỏi tin tức / diễn biến / cập nhật về 1 mã cụ thể
        if tickers and self._contains_any(q, self.news):
            return "market"

        # 2. Hỏi tin tức chung chung (không ticker)
        if self._contains_any(q, self.news):
            return "rag"

        # 3. Hỏi lời khuyên mua/bán
        if tickers and self._contains_any(q, self.advice_keywords):
            return "market"

        # 4. Dự báo / forecast
        if self._contains_any(q, self.forecast_keywords):
            return "forecast"

        # 5. Phân tích xu hướng thị trường / dòng tiền...
        if self._contains_any(
            q,
            [
                "phân tích", "xu hướng", "thị trường", "nhận định",
                "biến động", "dòng tiền", "khối ngoại"
            ],
        ):
            if tickers and any(t in self.market_indices for t in tickers):
                if self._contains_any(q, asking_price_keywords):
                    return "stock"
            return "market"

        # 6. Có ticker cụ thể
        if tickers:
            # ticker là index
            if any(t in self.market_indices for t in tickers):
                if self._contains_any(q, asking_price_keywords):
                    return "stock"
                return "market"

            # ticker là cổ phiếu: hỏi giá / % -> stock, còn lại -> market
            if self._contains_any(q, asking_price_keywords):
                return "stock"
            return "market"

        # 7. Không ticker nhưng có từ khóa tài chính
        if self._contains_any(q, self.finance_keywords):
            return "market"

        # 8. Time (after finance/news so "giá FPT ngày mai" is not a clock query)
        if self._contains_any(q, self.time_keywords):
            return "time"

        # 9. fallback
        return "rag"

    # ====== Time filter & history ======
    def detect_time_filter(self, query: str):
        vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
        now = datetime.now(vn_tz)

        today_start = vn_tz.localize(
            datetime(now.year, now.month, now.day, 0, 0, 0)
        )
        today_end = vn_tz.localize(
            datetime(now.year, now.month, now.day, 23, 59, 59)
        )

        yesterday_start = today_start - timedelta(days=1)
        yesterday_end = today_end - timedelta(days=1)

        tomorrow_start = today_start + timedelta(days=1)
        tomorrow_end = today_end + timedelta(days=1)

        last_week_start = today_start - timedelta(days=7)
        next_week_end = today_end + timedelta(days=7)

        q = query.lower()

        if "hôm nay" in q or "trong ngày" in q or "trong phiên" in q:
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
                dt_start = vn_tz.localize(
                    datetime(year, month, day, 0, 0, 0)
                )
                dt_end = vn_tz.localize(
                    datetime(year, month, day, 23, 59, 59)
                )
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

    def resolve_history_request(
        self, text: str, time_filter: tuple | None, default_days: int = 30
    ) -> tuple[bool, int | None]:
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

    # ====== Entry point ======
    def process_query(self, state, vocab: list = None):
        user_query = state.user_query
        corrected_query, corrections = self.correct_domain_typos(user_query)
        processed_query = self.normalize(corrected_query)
        lang = self.detect_language(processed_query)

        processed_query = self.map_synonyms(processed_query)
        processed_query = self.remove_stopwords(processed_query)

        if vocab:
            processed_query = self.correct_typo(processed_query, vocab)

        is_greeting = self.is_greeting(corrected_query)
        for kw in self.finance_keywords:
            if kw in processed_query:
                is_greeting = False
                break

        state.corrected_query = corrected_query
        state.processed_query = processed_query
        state.lang = lang
        state.is_greeting = is_greeting
        state.intent = self.detect_intent(corrected_query)
        if state.intent != "rag" or self._contains_any(corrected_query, self.news):
            state.intent_confidence = 0.95
        elif state.is_greeting or len(corrected_query.split()) < 3:
            state.intent_confidence = 0.85
        else:
            state.intent_confidence = 0.4
        state.time_filter = self.detect_time_filter(corrected_query)
        state.tickers = self.detect_tickers(corrected_query)

        # cache_key gợi ý: bản query chuẩn hóa
        state.cache_key = f"qa::{processed_query[:100]}"

        # debug
        if getattr(state, "add_debug", None):
            state.add_debug("processor_intent", state.intent)
            state.add_debug("processor_tickers", state.tickers)
            state.add_debug("processor_lang", state.lang)
            state.add_debug("processor_cache_key", state.cache_key)
            state.add_debug("processor_corrected_query", corrected_query)
            state.add_debug("processor_corrections", corrections)
            state.add_debug("processor_intent_confidence", state.intent_confidence)

        return state


processor_instance = Processor()


def processor_query(state, vocab: list = None):
    return processor_instance.process_query(state, vocab)

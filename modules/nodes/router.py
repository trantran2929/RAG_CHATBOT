from modules.api.time_api import format_full, get_now
from modules.api.weather_api import get_weather, normalize_city_name
from modules.api.stock_api import (
    format_market_summary,
    get_stock_quote,
    format_stock_info,
    get_history_prices,
    format_history_text,
    get_price_at_date,
    format_price_at_date,
)
from modules.api.forecast_api import (
    get_forecast_brief_for_symbol,
    get_full_forecast_answer,
    get_intraday_step_forecast_answer,
    get_next_session_forecast_answer,
)
from modules.api.news_api import format_today_news_brief

from modules.core.state import GlobalState
from modules.nodes.processor import processor_instance
import re
import pytz
from datetime import datetime, timedelta, date


def _extract_point_date_from_query(q: str) -> date | None:
    """
    Cố gắng suy ra 1 NGÀY CỤ THỂ từ câu hỏi:
      - 'ngày 2 tháng 12 năm 2025'
      - 'ngày 02/12/2025', 'ngày 2-12-2025'
      - 'hôm qua'
      - 'hôm kia'
      - '3 ngày trước'

    Trả về datetime.date hoặc None nếu không bắt được.
    """
    q = (q or "").lower()

    now_dt = get_now()

    # 1) dạng "ngày dd/mm/yyyy" hoặc "ngày dd-mm-yyyy"
    m_abs1 = re.search(r"ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})", q)
    if m_abs1:
        d, m_, y = map(int, m_abs1.groups())
        try:
            return now_dt.replace(year=y, month=m_, day=d).date()
        except ValueError:
            return None

    # 2) dạng "ngày d tháng m năm yyyy"
    m_abs2 = re.search(
        r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", q
    )
    if m_abs2:
        d, m_, y = map(int, m_abs2.groups())
        try:
            return now_dt.replace(year=y, month=m_, day=d).date()
        except ValueError:
            return None

    # 3) 'hôm qua'
    if "hôm qua" in q:
        return (now_dt - timedelta(days=1)).date()

    # 4) 'hôm kia'
    if "hôm kia" in q:
        return (now_dt - timedelta(days=2)).date()

    # 5) 'X ngày trước'
    m_rel = re.search(r"(\d+)\s+ngày\s+trước", q)
    if m_rel:
        n = int(m_rel.group(1))
        return (now_dt - timedelta(days=n)).date()

    return None


def _detect_forecast_mode(q: str) -> str:
    """
    Phân loại kiểu dự báo mà user hỏi:
    - 'step'    : bước tiếp theo trong phiên (intraday)
    - 'session' : phiên giao dịch kế tiếp (AM/PM / ngày mai)

    Nếu không match gì rõ ràng -> mặc định 'session'
    (user phổ thông thường muốn biết phiên tới hơn là vài tick kế tiếp).
    """
    q = (q or "").lower().strip()

    # ---- RULE 0: rõ ràng hỏi "bước" -> intraday step ----
    step_phrases_strong = [
        "bước tiếp theo",
        "bước kế tiếp",
        "bước tiếp",
        "next step",
    ]
    if any(kw in q for kw in step_phrases_strong):
        return "step"

    # ---- RULE 1: cụm chứa 'phiên ... kế tiếp/tiếp theo/sau/tới' -> phiên kế tiếp ----
    # bắt trước cả khi có 'trong phiên ...'
    if (
        re.search(r"\btrong\s+phiên\s+(kế tiếp|tiếp theo|sau|tới)\b", q)
        or re.search(r"\bphiên\s+(kế tiếp|tiếp theo|sau|tới)\b", q)
    ):
        return "session"

    # ---- RULE 2: keyword gợi ý nội phiên / intraday (nhưng không dùng 'trong phiên' chung chung) ----
    step_keywords = [
        "trong phiên",
        "nội phiên",
        "intraday",
        "rất ngắn hạn",
        "ngắn hạn",
        "vài phút nữa",
        "ít phút nữa",
        "ngay bây giờ",
    ]
    if any(kw in q for kw in step_keywords):
        return "step"

    # ---- RULE 3: keyword gợi ý phiên kế tiếp / ngày mai ----
    session_keywords = [
        "phiên tới",
        "phiên sau",
        "phiên kế tiếp",
        "phiên tiếp theo",
        "phiên sáng mai",
        "phiên chiều mai",
        "phiên sáng",
        "phiên chiều",
        "ngày mai",
        "mai",
        "mở cửa mai",
        "mở cửa phiên tới",
    ]
    if any(kw in q for kw in session_keywords):
        return "session"

    # ---- DEFAULT: hiểu là phiên tới ----
    return "session"

def _extract_news_keyword_from_query(
    q: str,
    tickers: list[str] | None = None,
) -> str | None:
    """
    Lấy keyword để lọc tin tức hôm nay:
    - Nếu có ticker -> ưu tiên dùng ticker (VD: SBT, VCB).
    - Nếu không có ticker -> cố gắng bắt cụm sau 'liên quan đến' hoặc 'về'.
    """
    tickers = tickers or []
    if tickers:
        return tickers[0]  # VD: 'SBT'

    if not q:
        return None

    # Dùng raw câu hỏi để giữ nguyên tên như 'Agris'
    m = re.search(r"liên quan đến\s+([A-Za-z0-9\-_\.]+)", q, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"về\s+([A-Za-z0-9\-_\.]+)", q, flags=re.IGNORECASE)

    if m:
        return m.group(1).strip()

    return None


def route_intent(state: GlobalState) -> GlobalState:
    user_query = (state.user_query or "").lower()
    intent = getattr(state, "intent", "rag")
    tickers = getattr(state, "tickers", [])
    symbol = tickers[0] if tickers else None

    # Greeting
    if getattr(state, "is_greeting", False):
        state.intent = "greeting"
        state.route_to = "api"
        state.api_type = "greeting"
        state.api_response = "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"
        state.llm_status = "greeting_handled"
        state.add_debug("route", "greeting")
        return state

        # Pure RAG / tin tức
    if intent == "rag":
        tf = getattr(state, "time_filter", None)

        if tf is not None:
            vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
            start_ts, end_ts = tf

            start_day = datetime.fromtimestamp(int(start_ts), tz=vn_tz).date()
            today = datetime.now(vn_tz).date()

            # Nếu time_filter trỏ vào hôm nay -> ưu tiên news API "tin hôm nay"
            if start_day is not None and start_day == today:
                # 🔹 Lấy keyword: SBT, Agris, ...
                news_keyword = _extract_news_keyword_from_query(
                    state.user_query or "",
                    tickers=getattr(state, "tickers", []),
                )

                state.route_to = "api"
                state.api_type = "news_today"
                try:
                    state.api_response = format_today_news_brief(
                        limit=10,
                        max_pages=1,
                        keyword=news_keyword,
                    )
                    state.llm_status = "route_news_today_api"
                    state.add_debug("route", "news_today_api")
                    state.add_debug("time_filter", tf)
                    state.add_debug("news_keyword", news_keyword)
                    return state
                except Exception as e:
                    state.add_debug("news_api_error", str(e))
                    state.add_debug("route", "rag_fallback")

        # Nếu không phải hôm nay hoặc news_api lỗi -> fallback RAG
        state.route_to = "rag"
        state.api_type = None
        state.api_response = None
        state.llm_status = "route_rag"
        state.add_debug("route", "rag_news")
        return state


    # Stock info / history
    if intent == "stock":
        if not symbol:
            # Không có mã -> trả tổng quan thị trường
            state.route_to = "api"
            state.api_type = "market_summary"
            try:
                state.api_response = format_market_summary()
            except Exception:
                state.api_response = "Tổng quan thị trường tạm thời không khả dụng"
            state.llm_status = "route_market_summary"
            state.add_debug("route", "market_summary_no_symbol")
            return state

        # 1) Ưu tiên: câu hỏi về 1 NGÀY CỤ THỂ (absolute / relative)
        target_date = _extract_point_date_from_query(user_query)
        if target_date is not None:
            data = get_price_at_date(symbol, target_date)
            state.route_to = "api"
            state.api_type = "stock_price_at_date"
            state.api_response = format_price_at_date(symbol, target_date, data)
            state.llm_status = "route_stock_price_at_date"
            state.add_debug(
                "route",
                f"stock_price_at_date_{symbol}_{target_date.strftime('%d-%m-%Y')}",
            )
            return state

        # 2) Ngược lại: lịch sử N ngày gần đây
        need_hist, days = processor_instance.resolve_history_request(
            text=state.user_query,
            time_filter=getattr(state, "time_filter", None),
            default_days=30,
        )

        if need_hist:
            d = days or 30
            data = get_history_prices(symbol, days=d)
            state.route_to = "api"
            state.api_type = "stock_history"
            state.api_response = format_history_text(symbol, d, data)
            state.llm_status = "route_stock_history"
            state.add_debug("route", f"stock_history_{symbol}_{d}d")
            return state

        # 3) Không phải hỏi lịch sử -> trả giá hiện tại
        data = get_stock_quote(symbol)
        if "error" not in data:
            state.api_response = format_stock_info(symbol)
        else:
            state.api_response = f"Không thể lấy dữ liệu cho mã {symbol.upper()}."
        state.api_type = "stock_quote"
        state.route_to = "api"
        state.llm_status = "route_stock_quote"
        state.add_debug("route", f"stock_quote_{symbol}")
        return state

    # Market view (không có ticker nhưng có từ khóa tài chính)
    if intent == "market":
        # Nếu câu hỏi mang tính "lịch sử" (ngày trước / hôm qua / ...) nhưng KHÔNG có mã
        # -> hỏi lại để làm rõ mã, không nên tóm tắt thị trường / RAG.
        need_hist, _ = processor_instance.resolve_history_request(
            text=state.user_query,
            time_filter=getattr(state, "time_filter", None),
            default_days=30,
        )
        if need_hist and not symbol:
            state.route_to = "api"
            state.api_type = "clarify_symbol_for_history"
            state.api_response = (
                "Bạn muốn xem **giá cổ phiếu nào** vào thời điểm đó?\n"
                "Ví dụ: `Giá VCB 3 ngày trước?` hoặc `Giá FPT ngày 2/12/2025`."
            )
            state.llm_status = "route_market_need_symbol_for_history"
            state.add_debug("route", "market_need_symbol_for_history")
            return state

        # Còn lại: market analysis + hybrid (API + RAG)
        state.route_to = "hybrid"
        state.api_type = "market_analysis"

        base_text = ""
        if symbol:
            data = get_stock_quote(symbol)
            try:
                base_text = format_stock_info(symbol)
            except Exception:
                base_text = (
                    f"{symbol}: giá ~ {data.get('price')}, "
                    f"thay đổi {data.get('percent_change')}%"
                )
        else:
            try:
                base_text = format_market_summary()
            except Exception:
                base_text = "Tổng quan thị trường"

        brief_forecast = ""
        if symbol:
            brief_forecast = get_forecast_brief_for_symbol(symbol)

        state.api_response = base_text + (brief_forecast if brief_forecast else "")
        state.llm_status = "route_market_hybrid"
        state.add_debug("route", "hybrid_with_brief_forecast")
        return state

    # Forecast explicit
    if intent == "forecast":
        state.route_to = "api"

        if not symbol:
            state.api_type = "forecast_need_symbol"
            state.api_response = (
                "Bạn muốn dự đoán mã nào? Ví dụ: `Dự đoán phiên tới của VCB?` "
                "hoặc `Dự đoán bước tiếp theo trong phiên của FPT?`"
            )
            state.llm_status = "route_forecast_need_symbol"
            state.add_debug("route", "forecast_need_symbol")
            return state

        # Phân loại kiểu dự báo theo câu hỏi user
        fmode = _detect_forecast_mode(state.user_query or "")

        if fmode == "step":
            # Dự đoán bước tiếp theo trong phiên
            state.api_type = "forecast_intraday_step"
            state.api_response = get_intraday_step_forecast_answer(symbol)
            state.llm_status = "route_forecast_intraday_step"
            state.add_debug("route", "forecast_intraday_step")
            return state

        # Mặc định / còn lại: dự đoán phiên giao dịch kế tiếp
        state.api_type = "forecast_next_session"
        state.api_response = get_next_session_forecast_answer(symbol)
        state.llm_status = "route_forecast_next_session"
        state.add_debug("route", "forecast_next_session")
        return state

    # Weather
    if intent == "weather":
        match = re.search(r"thời tiết\s+(?:ở\s+)?(.+)", user_query)
        city_raw = match.group(1).strip() if match else "Hà Nội"
        city = normalize_city_name(city_raw)
        weather = get_weather(city, "Việt Nam")

        state.route_to = "api"
        state.api_type = "weather"
        if "error" not in weather:
            state.api_response = (
                f"🌤️ Thời tiết tại **{weather['location']}**: "
                f"{weather['temp']}°C, {weather['desc']}."
            )
        else:
            state.api_response = "⚠️ Không lấy được dữ liệu thời tiết."
        state.llm_status = "route_weather"
        state.add_debug("route", "weather_api")
        return state

    # Time
    if intent == "time":
        q = (state.user_query or "").lower().strip()
        now_dt = get_now()
        hhmmss = now_dt.strftime("%H:%M:%S")
        date_text = now_dt.strftime("%d/%m/%Y")

        if "còn bao lâu" in q and ("cuối năm" in q or "hết năm" in q):
            end_dt = now_dt.replace(
                month=12, day=31, hour=23, minute=59, second=59
            )
            diff = end_dt - now_dt
            days_left = diff.days
            hours_left = diff.seconds // 3600
            mins_left = (diff.seconds % 3600) // 60
            reply = (
                f"Còn khoảng {days_left} ngày "
                f"{hours_left} giờ {mins_left} phút nữa là hết năm."
            )
        elif (
            "ngày bao nhiêu" in q
            or "hôm nay là ngày" in q
            or "hôm nay ngày" in q
        ):
            reply = f"Hôm nay là {date_text}."
        else:
            reply = f"Hiện tại là {hhmmss}, {date_text}."

        state.route_to = "api"
        state.api_type = "time"
        state.api_response = reply
        state.llm_status = "route_time"
        state.add_debug("route", "time_api")
        state.add_debug("time_reply", reply)
        return state

    # Mặc định: RAG
    state.route_to = "rag"
    state.api_type = None
    state.api_response = None
    state.llm_status = "route_default_rag"
    state.add_debug("route", "rag_default")
    return state

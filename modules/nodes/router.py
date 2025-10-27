from modules.api.time_api import format_full, get_now
from modules.api.weather_api import get_weather, normalize_city_name
from modules.api.stock_api import (
    format_market_summary,
    get_stock_quote,
    format_stock_info,
    get_history_prices,
    format_history_text,
)
from modules.api.forecast_api import (
    get_forecast_brief_for_symbol,
    get_full_forecast_answer,
)

from modules.core.state import GlobalState
from modules.nodes.processor import processor_instance
import re


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
        state.route_to = "rag"
        state.api_type = None
        state.api_response = None
        state.llm_status = "route_rag"
        state.add_debug("route", "rag_news")
        return state

    # Stock info / history
    if intent == "stock":
        if not symbol:
            state.route_to = "api"
            state.api_type = "market_summary"
            try:
                state.api_response = format_market_summary()
            except Exception:
                state.api_response = "Tổng quan thị trường tạm thời không khả dụng"
            state.llm_status = "route_market_summary"
            state.add_debug("route", "market_summary_no_symbol")
            return state

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

    # Market view
    if intent == "market":
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
        state.api_type = "forecast"
        if not symbol:
            state.api_response = (
                "Bạn muốn dự đoán mã nào? Ví dụ: 'Dự đoán phiên tới của VCB?'"
            )
            state.llm_status = "route_forecast_need_symbol"
            state.add_debug("route", "forecast_need_symbol")
            return state

        state.api_response = get_full_forecast_answer(symbol)
        state.llm_status = "route_forecast_api"
        state.add_debug("route", "forecast_api_only")
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
            end_dt = now_dt.replace(month=12, day=31, hour=23, minute=59, second=59)
            diff = end_dt - now_dt
            days_left = diff.days
            hours_left = diff.seconds // 3600
            mins_left = (diff.seconds % 3600) // 60
            reply = (
                f"Còn khoảng {days_left} ngày "
                f"{hours_left} giờ {mins_left} phút nữa là hết năm."
            )
        elif "ngày bao nhiêu" in q or "hôm nay là ngày" in q or "hôm nay ngày" in q:
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

    # fallback -> rag
    state.route_to = "rag"
    state.api_type = None
    state.api_response = None
    state.llm_status = "route_default_rag"
    state.add_debug("route", "rag_default")
    return state

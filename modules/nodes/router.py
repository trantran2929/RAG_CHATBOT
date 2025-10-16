from modules.api.time_api import format_full, get_now
from modules.api.weather_api import get_weather, normalize_city_name
from modules.api.stock_api import (
    format_market_summary,
    get_stock_quote,
    format_stock_info,
    get_history_prices
)
from modules.core.state import GlobalState
import re, pytz
from datetime import datetime


def route_intent(state: GlobalState) -> GlobalState:
    """
    Định tuyến theo intent:
    - time → API thời gian
    - weather → API thời tiết
    - stock → API cổ phiếu (hiện tại / lịch sử / tổng quan)
    - market → lai (API + tin tức)
    """
    user_query = (state.user_query or "").lower()
    intent = getattr(state, "intent", "rag")
    tickers = getattr(state, "tickers", [])
    symbol = tickers[0] if tickers else None

    if getattr(state, "is_greeting", False):
        state.route_to = "api"
        state.api_type = "greeting"
        state.api_response = "👋 Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"
        state.add_debug("route", "greeting")
        return state
    
    if intent == "stock":
        if not symbol:
            state.route_to = "api"
            state.api_type = "market_summary"
            state.api_response = format_market_summary()
            state.add_debug("route", "market_summary")
            return state

        data = get_stock_quote(symbol)
        if "error" not in data:
            state.api_response = format_stock_info(symbol)
        else:
            state.api_response = f"Không thể lấy dữ liệu cho mã {symbol.upper()}."
        state.api_type = "stock"
        state.route_to = "api"
        state.add_debug("route", f"stock_api_{symbol}")
        return state
    
    if intent == "market":
        state.route_to = "hybrid"
        state.api_type = "market_analysis"

        # Nếu có mã → lấy giá cổ phiếu hiện tại
        if symbol:
            data = get_stock_quote(symbol)
            api_text = (
                format_stock_info(symbol)
                if "error" not in data
                else f"Không thể lấy dữ liệu cho {symbol.upper()}."
            )
            state.api_response = api_text
            state.add_debug("route", f"hybrid_stock_market_{symbol}")
        else:
            state.api_response = format_market_summary()
            state.add_debug("route", "hybrid_market_summary")

        return state

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
        state.add_debug("route", "weather_api")
        return state
    
    if intent == "time":
        state.route_to = "api"
        state.api_type = "time"
        state.api_response = f"Hôm nay là {format_full(get_now())}."
        state.add_debug("route", "time_api")
        return state
    
    state.route_to = "rag"
    state.api_type = None
    state.api_response = None
    state.add_debug("route", "rag_default")
    return state

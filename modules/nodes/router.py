from modules.api.time_api import format_full, get_now
from modules.api.weather_api import get_weather, normalize_city_name
from modules.api.stock_api import format_market_summary, get_stock_quote, format_stock_info
import re
from modules.core.state import GlobalState


def route_intent(state: GlobalState) -> GlobalState:
    """
    Xác định tuyến xử lý dựa trên nội dung câu hỏi người dùng.
    """
    user_query = (state.user_query or "").lower()
    query = (state.processed_query or "").lower()

    if getattr(state, "is_greeting", False):
        state.route_to = "api"
        state.api_type = "greeting"
        state.api_response = "👋 Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"
        state.add_debug("route", "greeting")
        return state

    stock_keywords = [
        "cổ phiếu", "chứng khoán", "thị trường", "lợi nhuận", "doanh thu",
        "tăng trưởng", "đầu tư", "trái phiếu", "lãi suất", "ngân hàng",
        "cổ tức", "gdp", "bitcoin", "ethereum", "defi", "vnindex",
        "vn-index", "vn30", "hnx", "upcom"
    ]

    news_keywords = ["tin tức", "tin mới", "cập nhật", "bản tin", "thời sự"]
    if any(kw in user_query for kw in news_keywords):
        if any(k in user_query for k in stock_keywords):
            state.route_to = "hybrid"
            state.intent = "news"
            state.api_type = "market_news"
            state.api_response = format_market_summary()
            state.add_debug("route", "hybrid_news_stock")
        else:
            state.route_to = "rag"
            state.intent = "news"
            state.api_type = None
            state.api_response = None
            state.add_debug("route", "rag_new_general")
        return state

    if any(kw in user_query for kw in stock_keywords):

        symbol = None
        if getattr(state, "tickers", None):
            valid_tickers = [t.strip().upper() for t in state.tickers if 3<= len(t.strip())]
            if valid_tickers:
                symbol = valid_tickers[0]
        if symbol:
            data = get_stock_quote(symbol)
            state.api_type = "stock"
            state.intent = "stock"
            if "error" not in data:
                state.api_response = format_stock_info(symbol)
            else:
                state.api_response = f"Không thể lấy dữ liệu cho mã {symbol.upper()}."

            if any(x in user_query for x in ["tin", "phân tích", "đánh giá", "biến động", "xu hướng"]):
                state.route_to = "hybrid"
                state.add_debug("route", "hybrid_stock")
            else:
                state.route_to = "api"
                state.add_debug("route", "stock_api")
        else:
            state.api_type = "market"
            state.intent = "market"
            state.api_response = format_market_summary()
            state.route_to = "api"
            state.add_debug("route", "market_summary")
        
        return state

    weather_keywords = ["thời tiết", "nhiệt độ", "mưa", "nắng"]
    if any(kw in user_query for kw in weather_keywords):
        match = re.search(r"thời tiết\s+(?:ở\s+)?(.+)", user_query, re.IGNORECASE)
        city_raw = match.group(1).strip() if match else "Hà Nội"
        city = normalize_city_name(city_raw)
        weather = get_weather(city, "Việt Nam")

        state.route_to = "api"
        state.api_type = "weather"
        state.intent = "weather"

        if "error" not in weather:
            state.api_response = (
                f"🌤️ Thời tiết tại **{weather['location']}**: "
                f"{weather['temp']}°C, {weather['desc']}."
            )
        else:
            state.api_response = "⚠️ Xin lỗi, tôi không lấy được dữ liệu thời tiết."
        state.add_debug("route", "weather_api")
        return state

    time_keywords = ["mấy giờ", "bây giờ", "hôm nay", "ngày mấy", "thứ mấy"]
    if any(kw in user_query for kw in time_keywords):
        state.route_to = "api"
        state.api_type = "time"
        state.intent = "time"
        state.api_response = f"🕒 Hôm nay là {format_full(get_now())}."
        state.add_debug("route", "time_api")
        return state

    state.route_to = "rag"
    state.api_type = None
    state.intent = "general"
    state.api_response = None
    state.add_debug("route", "rag_default")
    return state

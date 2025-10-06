from modules.api.time_api import format_full, get_now
from modules.api.weather_api import get_weather, normalize_city_name
from modules.api.stock_api import format_market_summary, get_stock_quote
import re
from modules.core.state import GlobalState

def route_intent(state: GlobalState) -> GlobalState:
    """
    Dựa trên processed_query/ intent từ processor để chọn hướng xử lý
    """
    query = state.processed_query.lower()
    user_query = state.user_query.lower()

    if getattr(state, "is_greeting", False):
        state.route_to = "api"
        state.api_type = "greeting"
        state.api_response = "Xin chào! Tôi có thể giúp gì có bạn?"
        state.add_debug("route", "greeting")
        return state
    
    if "tin tức" in user_query and "chứng khoán" in user_query:
        state.route_to = "hybrid"
        state.api_type = "hybrid_market"
        state.api_response = format_market_summary()
        state.add_debug("route", "hybrid_news_stock")
        return state
    
    stock_keys = [
            "cổ phiếu", "chứng khoán", "thị trường", "lợi nhuận", "doanh thu", "tăng trưởng", 
            "đầu tư", "trái phiếu", "lãi suất", "ngân hàng", "cổ tức", "GDP", "bitcoin", 
            "ethereum", "DeFi", "vnindex", "vn-index", "vn30", "hnx", "upcom",
        ]
    if any(kw in user_query for kw in stock_keys):
        if state.tickers:
            symbol = state.tickers[0]
            data = get_stock_quote(symbol)
            state.api_type = "stock"
            
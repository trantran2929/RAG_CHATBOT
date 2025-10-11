"""
STOCK API — VNStock Only (Final Optimized + TTL Cache)
Cung cấp API / hàm tiện ích cho chatbot:
- Lấy dữ liệu chỉ số (VNINDEX, VN30...)
- Lấy giá cổ phiếu hiện tại (real-time + fallback)
- Lấy top tăng / giảm
- Cache TTL decorator
- Hiển thị ngày + giờ thực tế
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="vnai.scope.profile")

from vnstock import Quote, Trading, Screener
from datetime import datetime, timedelta
import pytz, time
from functools import wraps

def TTLCache(ttl_seconds: int = 300, verbose: bool = False):
    """Cache kết quả của hàm trong TTL (giây)."""
    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            now = time.time()
            if key in cache:
                value, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    if verbose:
                        print(f"[Cache Hit] {func.__name__}")
                    return value
                else:
                    cache.pop(key, None)
            if verbose:
                print(f"[Cache Miss] {func.__name__}")
            value = func(*args, **kwargs)
            cache[key] = (value, now)
            return value
        return wrapper
    return decorator

def get_time_vn() -> str:
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    return datetime.now(tz).strftime("%d-%m-%Y %H:%M:%S")

@TTLCache(ttl_seconds=300)
def get_index_detail(index_code: str = "VNINDEX") -> dict:
    try:
        quote = Quote(symbol=index_code, source="VCI")
        df = quote.history(
            start=(datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
            end=datetime.now().strftime("%Y-%m-%d")
        )
        if df.empty:
            raise ValueError("Không có dữ liệu chỉ số.")
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        change = last["close"] - prev["close"]
        percent = (change / prev["close"]) * 100 if prev["close"] else 0
        return {
            "ticker": index_code,
            "price": round(last["close"], 2),
            "change": round(change, 2),
            "percent_change": round(percent, 2),
            "open": round(last["open"], 2),
            "high": round(last["high"], 2),
            "low": round(last["low"], 2),
            "volume": int(last.get("volume", 0)),
            "timestamp": get_time_vn(),
            "source": "VNStock"
        }
    except Exception as e:
        print(f"[VNStock] Lỗi lấy dữ liệu chỉ số {index_code}: {e}")
        return {"ticker": index_code, "error": str(e)}

@TTLCache(ttl_seconds=60)
def get_stock_quote(symbol: str) -> dict:
    """Lấy giá cổ phiếu real-time từ VNStock. Fallback sang lịch sử nếu match_price=0."""
    if not symbol or len(symbol.strip()) < 3:
        return {"symbol": symbol, "error": "Symbol phải có độ dài từ 3 đến 10 ký tự."}

    try:
        trading = Trading()
        df = trading.price_board(symbols_list=[symbol])
        if df.empty:
            raise ValueError("Không có dữ liệu cổ phiếu.")
        df.columns = [f"{a}_{b}" for a, b in df.columns]
        data = df.iloc[0].to_dict()

        price = data.get("match_match_price")
        ref_price = data.get("match_reference_price") or data.get("listing_ref_price")
        ceil_price = data.get("match_ceiling_price") or data.get("listing_ceiling")
        floor_price = data.get("match_floor_price") or data.get("listing_floor")
        volume_val = data.get("match_accumulated_volume") or data.get("match_match_vol") or 0

        if not price or price == 0:
            raise ValueError("Không có giá khớp lệnh hợp lệ (match_price = 0).")

        change_val = float(price) - float(ref_price)
        change_pct = (change_val / float(ref_price) * 100) if ref_price else 0

        return {
            "symbol": symbol.upper(),
            "price": float(price),
            "open": float(ref_price),
            "high": float(ceil_price),
            "low": float(floor_price),
            "change": round(change_val, 2),
            "percent_change": round(change_pct, 2),
            "volume": int(volume_val),
            "timestamp": get_time_vn(),
            "source": "VNStock.Trading"
        }

    except Exception as e:
        print(f"[VNStock] Lỗi lấy dữ liệu cổ phiếu {symbol}: {e}")
        try:
            # fallback sang lịch sử
            quote = Quote(symbol=symbol, source="VCI")
            df = quote.history(
                start=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                end=datetime.now().strftime("%Y-%m-%d")
            )
            if not df.empty:
                last = df.iloc[-1]
                trade_date = (
                    datetime.strptime(str(last.name)[:10], "%Y-%m-%d").strftime("%d-%m-%Y")
                    if last.name else "phiên gần nhất"
                )
                open_price = float(last.get("open", 0))
                close_price = float(last.get("close", 0))
                change_val = close_price - open_price
                change_pct = (change_val / open_price * 100) if open_price else 0
                return {
                    "symbol": symbol.upper(),
                    "price": close_price,
                    "open": open_price,
                    "high": float(last.get("high", 0)),
                    "low": float(last.get("low", 0)),
                    "change": round(change_val, 2),
                    "percent_change": round(change_pct, 2),
                    "volume": int(last.get("volume", 0)),
                    "timestamp": get_time_vn(),
                    "fallback": True,
                    "fallback_date": trade_date,
                    "source": "VNStock.Quote"
                }
        except Exception as ex:
            print(f"[VNStock] Fallback Quote lỗi: {ex}")
        return {"symbol": symbol.upper(), "error": str(e)}
    
@TTLCache(ttl_seconds=300)
def get_top_stocks(limit: int = 10, direction: str = "up") -> list:
    try:
        screener_df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=3000)
        if screener_df.empty:
            raise ValueError("Không có dữ liệu sàng lọc.")
        cols = [c.lower() for c in screener_df.columns]
        growth_col = next((c for c in screener_df.columns if "growth" in c.lower()), None)
        if not growth_col:
            raise ValueError("Không tìm thấy cột tăng trưởng giá hợp lệ.")
        screener_df[growth_col] = screener_df[growth_col].astype(float)
        ascending = True if direction == "down" else False
        top_df = screener_df.sort_values(growth_col, ascending=ascending).head(limit)
        results = []
        for _, row in top_df.iterrows():
            results.append({
                "symbol": row.get("ticker") or row.get("symbol"),
                "exchange": row.get("exchange", ""),
                "price": row.get("price_near_realtime") or row.get("close_price"),
                "pct_change": round(float(row[growth_col]), 2),
                "volume": int(row.get("avg_trading_value_10d", 0)),
                "timestamp": get_time_vn()
            })
        return results
    except Exception as e:
        print(f"[VNStock] Lỗi lấy top cổ phiếu: {e}")
        return []

@TTLCache(ttl_seconds=300)
def get_history_prices(symbol: str, days: int = 7) -> dict:
    """
    Lấy lịch sử giá cổ phiếu trong N ngày gần nhất từ VNStock.
    Trả về danh sách dict: [{date, open, high, low, close, volume, change, pct_change}]
    """
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(tz)
    start_date = (now - timedelta(days=days + 2)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    if not symbol or len(symbol.strip()) < 3:
        return {"symbol": symbol, "error": "Symbol phải có độ dài từ 3 đến 10 ký tự."}

    try:
        quote = Quote(symbol=symbol.upper(), source="VCI")
        df = quote.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError("Không có dữ liệu lịch sử.")

        # Tính phần trăm thay đổi so với phiên trước
        df["change"] = df["close"].diff()
        df["pct_change"] = df["close"].pct_change() * 100
        df = df.dropna()

        records = []
        for idx, row in df.tail(days).iterrows():
            trade_date = idx.strftime("%d-%m-%Y") if isinstance(idx, datetime) else str(idx)
            records.append({
                "date": trade_date,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row.get("volume", 0)),
                "change": round(float(row["change"]), 2),
                "pct_change": round(float(row["pct_change"]), 2)
            })

        return {
            "symbol": symbol.upper(),
            "history": records,
            "days": days,
            "timestamp": get_time_vn(),
            "source": "VNStock.Quote"
        }

    except Exception as e:
        print(f"[VNStock] Lỗi lấy lịch sử giá {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}
    
def format_stock_info(symbol: str) -> str:
    data = get_stock_quote(symbol)
    if "error" in data or not data.get("price"):
        return f"⚠️ Không thể lấy dữ liệu cho mã **{symbol.upper()}**."
    note = ""
    if data.get("fallback"):
        note = f" _(giá đóng cửa của {data.get('fallback_date', 'phiên gần nhất')})_"
    return (
        f"📊 **{data['symbol']}** — Giá hiện tại: {data['price']:,} VNĐ  \n"
        f"• Mở cửa: {data['open']:,} | Cao nhất: {data['high']:,} | Thấp nhất: {data['low']:,}  \n"
        f"• Thay đổi: {data['change']:+.2f} ({data['percent_change']:+.2f}%)  \n"
        f"• Khối lượng: {data['volume']:,}  \n"
        f"🕒 Cập nhật: {data['timestamp']}{note}"
    )

def format_top_stocks(direction="up", limit=5) -> str:
    tops = get_top_stocks(limit=limit, direction=direction)
    arrow = "🔺" if direction == "up" else "🔻"
    title = "Top cổ phiếu tăng mạnh" if direction == "up" else "Top cổ phiếu giảm mạnh"
    if not tops:
        return f"⚠️ Không có dữ liệu {title.lower()}."
    lines = [f"{arrow} {t['symbol']} ({t['pct_change']}%)" for t in tops]
    return f"📊 **{title}**  \n" + "  \n".join(lines)

@TTLCache(ttl_seconds=300)
def format_market_summary(indices: list[str] = None) -> str:
    try:
        if indices is None:
            indices = ["VNINDEX", "VN30", "HNX", "UPCOM"]
        
        summaries = []
        for idx in indices:
            data = get_index_detail(idx)
            if "error" in data:
                continue
            emoji = "📈" if data["change"] > 0 else "📉" if data["change"] < 0 else "⏸️"
            summaries.append(
                f"{emoji} **{data['ticker']}**: {data['price']:,} điểm "
                f"({data['change']:+.2f}, {data['percent_change']:+.2f}%)"
            )

            if not summaries:
                return "⚠️ Không thể lấy dữ liệu thị trường."
            
        return (
            f"📊 **TỔNG QUAN THỊ TRƯỜNG VIỆT NAM**  \n" +
            "  \n".join(summaries) + "  \n\n" +
            f"{format_top_stocks('up', 3)}  \n\n" +
            f"{format_top_stocks('down', 3)}  \n" +
            f"🕒 Cập nhật: {get_time_vn()}"
        )
    except Exception as e:
        print(f"[VNStock] Lỗi tóm tắt thị trường: {e}")
        return "⚠️ Không thể lấy dữ liệu thị trường."


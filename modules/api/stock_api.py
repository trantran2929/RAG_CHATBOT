from vnstock3 import Vnstock
from datetime import datetime, timedelta
import pytz
import pandas as pd

vn = Vnstock()

def get_time_vn():
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    return datetime.now(tz).strftime("%d-%m-%Y, %H:%M:%S")

def get_index_detail(index_code: str) -> dict:
    """
    Lấy dữ liệu chi tiết của 1 chỉ số:
    - price, open, high/low, change, volume, value
    """
    try:
        df = vn.stock(index_code).quote().head(1)
        row = df.iloc[0]
        return {
            "ticker": index_code,
            "price": float(row["price"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "change": float(row["change"]),
            "percent_change": float(row["pct_change"]),
            "volume": int(row["volume"]),
            "value": round(float(row["value"]), 2),
            "timestamp": get_time_vn()
        }
    except Exception as e:
        print(f"[StockAPI] Lỗi lấy dữ liệu chỉ số {index_code}: {e}")
        return {"ticker": index_code, "error": str(e)}
    
def get_all_indices(limit: int = None) -> list:
    """
    Lấy danh sách tất cả các chỉ số thị trường Việt Nam
    """

    try:
        df = vn.stock().index_list()
        if limit:
            df = df.head(limit)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"[StockAPI] Lỗi lấy danh sách chỉ số: {e}")
        return []
    
def get_stock_quote(symbol: str) -> dict:
    """
    Lấy thông tin price, open, high/low, percent_change.
    """
    try:
        df = vn.stock(symbol).quote().head(1)
        row = df.iloc[0]
        return {
            "symbol": symbol.upper(),
            "price": float(row["price"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "change": float(row["change"]),
            "percent_change": float(row["pct_change"]),
            "volume": int(row["volume"]),
            "value": round(float(row["value"]), 2),
            "timestamp": get_time_vn()
        }
    except Exception as e:
        print(f"[StockAPI] Lỗi lấy dữ liệu cổ phiếu {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}
    
def get_historical_prices(symbol: str, days: int = 30) -> str:
    """
    Lấy lịch sử giá cổ phiếu X trong ngày gần nhất
    """
    end_date = datetime.now(pytz.timezone("Asia/Ho_Chi_Minh"))
    start_date = end_date - timedelta(days=days)
    try:
        df = vn.stock(symbol).history(
            start = start_date.strftime("%Y-%m-%d"),
            end = end_date.strftime("%Y-%m-%d")
        )
        df["date"] = pd.to_datetime(df["time"]).dt.strftime("%d-%m-%Y")
        return df[["date", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
    except Exception as e:
        print(f"[StockAPI] Lỗi lấy lịch sử {symbol}: {e}")
        return []
    
def format_market_summary(show_limit: int = None, group_by_market: bool = False) -> str:
    """
    Hiển thị toàn cảnh tất cả các chỉ số Việt Nam (VNINDEX, VN30,...)
    show_limit: số lượng chỉ số hiển thị (Nếu none thì hiển thị hết)
    group_by_market: nhóm theo sàn HOSE / HNX / UPCOM
    """
    try:
        df = vn.stock().index_list()
    except Exception as e:
        print(f"[StockAPI] Lỗi lấy danh sách chỉ số: {e}")
        return "Không thể lấy được dữ liệu các chỉ số thị trường"
    
    if show_limit:
        df = df.head(show_limit)
    
    lines = ["🌏 **TOÀN CẢNH THỊ TRƯỜNG CHỨNG KHOÁN VIỆT NAM**\n"]
    if group_by_market:
        for market, group in df.group("market"):
            lines.append(f"\n **Sàn {market}**")
            for _, row in group.iterrows():
                name = row.get("index_name", row["ticker"])
                change = row.get("change", 0)
                pct = row.get("pct_change", 0)
                emoji = "📈" if change > 0 else ("📉" if change < 0 else "⏸")
                lines.append(
                    f"{emoji} **{name}** ({row['ticker']}) → {row['last']:.2f} điểm "
                    f"({change:+.2f} / {pct:+.2f}%)"
                )
    else:
        for _, row in df.iterrows():
            name = row.get("index_name", row["ticker"])
            change = row.get("change", 0)
            pct = row.get("pct_change", 0)
            emoji = "📈" if change > 0 else ("📉" if change < 0 else "⏸")
            lines.append(
                f"{emoji} **{name}** ({row['ticker']}) → {row['last']:.2f} điểm "
                f"({change:+.2f} / {pct:+.2f}%)"
            )
    lines.append(f"\n Cập nhập: {get_time_vn()}")
    return "\n".join(lines)

def format_stock_info(symbol: str) -> str:
    """
    Mô tả nhanh 1 cổ phiếu (để chatbot đọc).
    """
    data = get_stock_quote(symbol)
    if "error" in data:
        return f"❌ Không thể lấy dữ liệu cho mã {symbol.upper()}."

    return (
        f"📊 **{data['symbol']}** — giá hiện tại: {data['price']} VNĐ\n"
        f"• Mở cửa: {data['open']} | Cao nhất: {data['high']} | Thấp nhất: {data['low']}\n"
        f"• Thay đổi: {data['change']} ({data['percent_change']}%)\n"
        f"• Khối lượng: {data['volume']:,} | Giá trị GD: {data['value']:,} VNĐ\n"
        f"🕒 Cập nhật: {data['timestamp']}"
    )
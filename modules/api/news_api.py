from datetime import datetime
from typing import List, Dict
import pytz
from modules.ingestion.crawler import crawl_cafef_stock

ICT = pytz.timezone("Asia/Ho_Chi_Minh")

def _parse_time_str(t_str: str) -> datetime:
    """Chuyển chuỗi thời gian từ Cafef sang datetime có timezone ICT."""
    if not t_str:
        return None
    t_str = t_str.strip()
    formats = [
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(t_str, fmt)
            return ICT.localize(dt)
        except ValueError:
            continue
    return None

def get_today_cafef_stock(
    limit: int = 30,
    max_pages: int = 1,
    keyword: str | None = None,
) -> List[Dict]:
    """
    Lấy danh sách bài CafeF 'thị trường chứng khoán' trong ngày hôm nay (ICT).
    - Crawl từ CafeF.
    - Lọc theo date == hôm nay.
    - Nếu có keyword -> chỉ giữ bài có keyword trong title/summary.
    - Sort theo thời gian giảm dần.
    """
    articles = crawl_cafef_stock(max_pages=max_pages)
    if not articles:
        return []
    
    today = datetime.now(ICT).date()
    filtered = []
    keyword_l = keyword.lower() if keyword else None

    for a in articles:
        t_str = a.get("time", "")
        dt = _parse_time_str(t_str)
        if not dt:
            continue
        if dt.date() != today:
            continue

        # Nếu có keyword -> lọc theo title + summary
        if keyword_l:
            title = (a.get("title") or "").lower()
            summary = (a.get("summary") or "").lower()
            big_text = f"{title} {summary}"
            if keyword_l not in big_text:
                continue

        a = dict(a)  # copy
        a["time_dt"] = dt
        filtered.append(a)
    
    # Sort giảm dần theo thời gian
    filtered.sort(key=lambda x: x["time_dt"], reverse=True)
    for a in filtered:
        a.pop("time_dt", None)

    return filtered[:limit]

def format_today_news_brief(
    limit: int = 10,
    max_pages: int = 1,
    keyword: str | None = None,
) -> str:
    items = get_today_cafef_stock(
        limit=limit,
        max_pages=max_pages,
        keyword=keyword,
    )
    if not items:
        # Không có bài nào sau khi lọc theo keyword
        if keyword:
            return (
                f"Hiện chưa tìm được tin chứng khoán trong ngày hôm nay "
                f"liên quan đến **{keyword}** từ CafeF.\n"
                "Bạn vui lòng thử lại sau."
            )
        # Không có dữ liệu chung
        return (
            "Hiện chưa lấy được tin tức chứng khoán trong ngày từ CafeF.\n"
            "Bạn vui lòng thử lại sau."
        )

    if keyword:
        header = (
            f"**Một số tin tức chứng khoán trong ngày hôm nay "
            f"liên quan đến {keyword}:**\n"
        )
    else:
        header = "**Một số tin tức chứng khoán trong ngày hôm nay:**\n"

    lines = [header]

    for i, a in enumerate(items[:limit], start=1):
        title = (a.get("title") or "").strip()
        t = (a.get("time") or "").strip()
        url = a.get("url") or ""
        summary = (a.get("summary") or "").strip()

        if len(summary) > 220:
            summary = summary[:220].rsplit(" ", 1)[0] + "..."

        line = f"{i}. {title}\n"
        if summary:
            line += f"   {summary}\n"
        if url:
            line += f"   [Nguồn]({url})\n"

        lines.append(line)

    lines.append(
        "Lưu ý: Đây chỉ là mô tả tin tức thị trường, **không phải khuyến nghị mua/bán**."
    )
    return "\n".join(lines).strip()

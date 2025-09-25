import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re

BASE_URL = "https://cafef.vn"
TIMEZONE_OFFSET = 7 # UTC+7


def normalize_time(time_tag) -> str:
    """Chuẩn hóa thời gian về format dd-mm-YYYY HH:MM:SS (UTC+7)."""
    if not time_tag:
        return ""

    # Nếu có attribute title sẵn ISO (ví dụ: 2025-09-25T08:27:00)
    if time_tag.has_attr("title"):
        try:
            dt = datetime.fromisoformat(time_tag["title"])
            return dt.strftime("%d-%m-%Y %H:%M:%S")
        except Exception:
            return time_tag["title"]

    # Nếu chỉ có text: "1 giờ trước", "5 phút trước", "2 ngày trước"
    text = time_tag.get_text(strip=True).lower()
    now = datetime.utcnow() + timedelta(hours=TIMEZONE_OFFSET)

    num = int(re.search(r"\d+", text).group()) if re.search(r"\d+", text) else 0

    if "phút" in text:
        dt = now - timedelta(minutes=num)
    elif "giờ" in text:
        dt = now - timedelta(hours=num)
    elif "ngày" in text:
        dt = now - timedelta(days=num)
    else:
        dt = now

    return dt.strftime("%d-%m-%Y %H:%M:%S")


def get_article_content(link: str) -> str:
    """Lấy nội dung chi tiết của 1 bài viết."""
    try:
        resp = requests.get(link, timeout=10)
        resp.encoding = "utf-8"
    except Exception as e:
        print(f"[Crawler] Lỗi khi request {link}: {e}")
        return ""

    if resp.status_code != 200:
        print(f"[Crawler] Lỗi {resp.status_code} khi truy cập {link}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.select_one("div.detail-content")
    if not content_div:
        return ""

    paragraphs = [
        p.get_text(strip=True)
        for p in content_div.find_all("p")
        if p.get_text(strip=True)
    ]
    return "\n".join(paragraphs)


def crawl_cafef_stock(max_pages: int = 1):
    """Crawl tin tức Thị trường chứng khoán trên CafeF."""
    articles = []
    for page in range(1, max_pages + 1):
        url = (
            f"{BASE_URL}/thi-truong-chung-khoan.chn"
            if page == 1
            else f"{BASE_URL}/thi-truong-chung-khoan/trang-{page}.chn"
        )

        try:
            resp = requests.get(url, timeout=10)
            resp.encoding = "utf-8"
        except Exception as e:
            print(f"[Crawler] Lỗi khi request {url}: {e}")
            continue

        if resp.status_code != 200:
            print(f"[Crawler] Lỗi {resp.status_code} khi truy cập {url}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select("div.tlitem.box-category-item")

        for item in items:
            link_tag = item.select_one("h3 > a")
            if not link_tag:
                continue

            href = link_tag.get("href", "")
            title = link_tag.get("title") or link_tag.text.strip()
            link = BASE_URL + href if href.startswith("/") else href
            article_id = item.get("data-id") or href.split("-")[-1].replace(".chn", "")

            summary_tag = item.select_one("p.sapo") or item.select_one("p.box-category-sapo")
            summary = summary_tag.get_text(strip=True) if summary_tag else ""

            time_tag = item.select_one("span.time")
            time_text = normalize_time(time_tag)

            # Nội dung chi tiết
            full_content = get_article_content(link)

            articles.append({
                "id": article_id,
                "title": title,
                "time": time_text,
                "summary": summary,
                "url": link,
                "content": full_content,
                "source": "cafef"
            })

    print(f"[Crawler] Crawled {len(articles)} articles")
    return articles


if __name__ == "__main__":
    data = crawl_cafef_stock(max_pages=1)
    for item in data[:3]:
        print(item)

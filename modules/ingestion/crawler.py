import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re

BASE_URL = "https://cafef.vn"
TIMEZONE_OFFSET = 7 # UTC+7
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG_CHATBOT/1.0; +https://cafef.vn)",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
}


def _http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(REQUEST_HEADERS)
    return session


def normalize_time(time_tag) -> str:
    """Chuẩn hóa thời gian về format dd-mm-YYYY HH:MM:SS (UTC+7)."""
    now = datetime.utcnow() + timedelta(hours=TIMEZONE_OFFSET)
    if not time_tag:
        return now.strftime("%d-%m-%Y %H:%M:%S")

    # Nếu có attribute title sẵn ISO (2025-09-25T08:27:00)
    if time_tag.has_attr("title"):
        try:
            dt = datetime.fromisoformat(time_tag["title"])
            return dt.strftime("%d-%m-%Y %H:%M:%S")
        except Exception:
            pass

    # Nếu chỉ có text: "1 giờ trước", "5 phút trước", "2 ngày trước"
    text = time_tag.get_text(strip=True).lower()

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


def get_article_content(link: str, session: requests.Session | None = None) -> str:
    """Lấy nội dung chi tiết của 1 bài viết."""
    try:
        resp = (session or _http_session()).get(link, timeout=15)
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
    seen_urls = set()
    session = _http_session()
    # CafeF no longer serves /trang-N.chn (HTTP 404). The scheduler polls the
    # live category page, and Qdrant deduplication keeps each poll idempotent.
    for page in range(1, min(max_pages, 1) + 1):
        url = (
            f"{BASE_URL}/thi-truong-chung-khoan.chn"
            if page == 1
            else f"{BASE_URL}/thi-truong-chung-khoan/trang-{page}.chn"
        )

        try:
            resp = session.get(url, timeout=15)
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
            if not link or link in seen_urls:
                continue
            seen_urls.add(link)
            article_id = item.get("data-id") or href.split("-")[-1].replace(".chn", "")

            summary_tag = item.select_one("p.sapo") or item.select_one("p.box-category-sapo")
            summary = summary_tag.get_text(strip=True) if summary_tag else ""

            time_tag = item.select_one("span.time")
            time_text = normalize_time(time_tag)

            # Nội dung chi tiết
            full_content = get_article_content(link, session=session)

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

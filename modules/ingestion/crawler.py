import requests
from bs4 import BeautifulSoup

BASE_URL = "https://cafef.vn"


def get_url(page: int) -> str:
    """Sinh URL cho từng trang tin nhanh."""
    return f"{BASE_URL}/doc-nhanh.chn" if page == 1 else f"{BASE_URL}/doc-nhanh/trang-{page}.chn"


def get_article_content(link: str) -> str:
    """Vào trang chi tiết và lấy toàn bộ nội dung bài viết."""
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

    # Ghép các đoạn văn lại
    paragraphs = [
        p.get_text(strip=True)
        for p in content_div.find_all("p")
        if p.get_text(strip=True)
    ]
    return "\n".join(paragraphs)


def crawl_cafef(max_pages: int = 1):
    """Crawl tin nhanh trên CafeF, trả về list dict bài viết."""
    articles = []
    for page in range(1, max_pages + 1):
        url = get_url(page)
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

        # Mỗi bài trong block fast-news
        items = soup.select("div.foreverblock.list-fast-news div.item")
        for item in items:
            nv_text = item.select_one("div.nv-text-cont")
            if not nv_text:
                continue

            link_tag = nv_text.select_one("a.news-title")
            if not link_tag:
                continue

            href = link_tag["href"]
            article_id = item.get("data-id") or href.split("-")[-1].replace(".chn", "")
            title = link_tag.get("title") or link_tag.text.strip()
            link = BASE_URL + href if href.startswith("/") else href

            details = item.select_one("div.nv-details")
            summary = (
                details.select_one("div.abs").get_text(strip=True)
                if details and details.select_one("div.abs")
                else ""
            )
            time_text = (
                details.select_one("span.time").get_text(strip=True)
                if details and details.select_one("span.time")
                else ""
            )

            # Crawl nội dung chi tiết
            full_content = get_article_content(link)

            articles.append(
                {
                    "id": article_id,
                    "title": title,
                    "time": time_text,
                    "summary": summary,
                    "link": link,
                    "content": full_content,
                }
            )

    print(f"[Crawler] Crawled {len(articles)} articles")
    return articles


if __name__ == "__main__":
    data = crawl_cafef(max_pages=2)
    for item in data[:2]:
        print(item)

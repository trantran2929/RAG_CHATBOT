import requests
from bs4 import BeautifulSoup

BASE_URL = "https://cafef.vn"

def get_url(page: int) ->str:
    if page == 1:
        return f"{BASE_URL}/doc-nhanh.chn"
    else:
        return f"{BASE_URL}/doc-nhanh/trang-{page}.chn"

def get_article_content(link: str) ->str:
    """Vào trang chi tiết lấy toàn bộ nội dung"""
    response = requests.get(link, timeout=20)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    content_div = soup.select_one("div.detail-content")
    if not content_div:
        return ""
    
    paragraphs = [
        p.get_text(strip=True)
        for p in content_div.find_all("p")
        if p.get_text(strip=True)
    ]
    return "\n".join(paragraphs)

def crawl_cafef(max_page: int = 10):
    """Crawl dữ liệu từ cafef"""
    articles = []
    for page in range(1, max_page + 1):
        url = get_url(page)
        response = requests.get(url, timeout=20)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")

        blocks = soup.select("div.nv-text-cont")
        for block in blocks:
            link_tag = block.select_one("a.new-title")
            if not link_tag:
                continue

            href = link_tag["href"]
            article_id = href.split("-")[-1].replace(".chn","")
            title = link_tag.get("title") or link_tag.text.strip()
            link = BASE_URL + href

            details = block.find_next_sibling("div", class_="nv-details")
            short_content = details.select_one("div.abs").text.strip() if details else ""
            time_tag = details.select_one("span.time")
            time_text = time_tag.get_text(strip=True) if time_tag else ""

            full_content = get_article_content(link)

            articles.append({
                "id": article_id,
                "title": title,
                "time": time_text,
                "summary": short_content,
                "link": link,
                "content": full_content
            })
    return articles
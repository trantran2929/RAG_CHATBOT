import time
from modules.ingestion.crawler import crawl_cafef
from modules.ingestion.preprocess import preprocess_articles
from modules.ingestion.loader import load_to_vector_db, delete_by_collection

def run_scheduler(
        interval: int = 3600, 
        max_page: int = 10,
        collection_name: str = "cafef_articles",
        reset_collection: bool = False
    ):
    """Scheduler định kì"""
    while True:
        print(f"\n Update news:")

        raw_articles = crawl_cafef(max_page=max_page)
        print(f"Crawl xong {len(raw_articles)} bài báo")

        docs = preprocess_articles(raw_articles, max_words=200)
        print(f"Sau preprocess còn {len(docs)} bài báo")

        # if reset_collection:
        #     delete_by_collection(collection_name)

        count = load_to_vector_db(docs,collection_name=collection_name)
        print(f"Đã nạp {count} documents vào {collection_name}")

        print(f"Chờ {interval} giây trước lần crawl tiếp theo \n")
        time.sleep(interval)
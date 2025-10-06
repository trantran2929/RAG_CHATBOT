# modules/ingestion/scheduler.py
import time
import traceback
from modules.ingestion.crawler import crawl_cafef_stock
from modules.ingestion.preprocess import preprocess_articles
from modules.ingestion.loader import load_to_vector_db

def run_scheduler(
    interval: int = 3600,
    collection_name: str = "cafef_articles",
):
    while True:
        try:
            print("\n[Ingestion] Update news...")
            raw_articles = crawl_cafef_stock()
            print(f"[Ingestion] Crawled {len(raw_articles)} raw articles")

            if not raw_articles:
                print("[Ingestion] Không lấy được bài nào, bỏ qua batch.")
            else:
                docs = preprocess_articles(raw_articles)
                print(f"[Ingestion] Prepared {len(docs)} chunks")

                if docs:
                    count = load_to_vector_db(docs, collection_name=collection_name)
                    print(f"[Ingestion] Upserted {count} chunks into `{collection_name}`")
                else:
                    print("[Ingestion] Không có docs sau preprocess, skip.")

        except Exception as e:
            print(f"[Ingestion] ERROR: {e}")
            traceback.print_exc()

        print(f"[Ingestion] Sleep {interval} seconds...\n")
        time.sleep(interval)

if __name__ == "__main__":
    run_scheduler()

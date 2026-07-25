import sys
import types
import unittest
from unittest.mock import patch

from modules.ingestion.preprocess import _to_time_ts, preprocess_articles
from modules.ingestion.scheduler import _doc_fingerprint, _filter_new_docs, run_ingestion_cycle


class IngestionTests(unittest.TestCase):
    def test_chunk_ids_are_stable_when_relative_time_changes(self):
        base = {
            "id": "article-1",
            "url": "https://cafef.vn/example.chn",
            "title": "FPT công bố kết quả kinh doanh",
            "summary": "FPT ghi nhận tăng trưởng trong quý gần nhất.",
            "content": "FPT ghi nhận tăng trưởng doanh thu và lợi nhuận trong quý gần nhất.",
            "source": "cafef",
        }
        with patch("modules.ingestion.preprocess.get_all_tickers", return_value={"FPT"}):
            first = preprocess_articles([{**base, "time": "21-07-2026 10:00:00"}])
            second = preprocess_articles([{**base, "time": "21-07-2026 11:00:00"}])
        self.assertEqual([d["id"] for d in first], [d["id"] for d in second])
        self.assertNotEqual(first[0]["time_ts"], second[0]["time_ts"])

    def test_time_parser_accepts_slash_and_dash_formats(self):
        self.assertEqual(
            _to_time_ts("21/07/2026, 10:00:00"),
            _to_time_ts("21-07-2026 10:00:00"),
        )

    def test_filter_skips_existing_old_and_invalid_docs(self):
        docs = [
            {"id": "existing", "time_ts": 1000},
            {"id": "fresh", "time_ts": 950},
            {"id": "old", "time_ts": 100},
            {"id": "bad", "time_ts": None},
        ]
        result = _filter_new_docs(docs, {"existing"}, min_time_ts=900)
        self.assertEqual([d["id"] for d in result], ["fresh"])

    def test_content_fingerprint_prevents_duplicate_after_id_migration(self):
        old = {"id": "old-id", "url": "https://cafef.vn/a.chn", "content": "same", "time_ts": 900}
        recrawled = {**old, "id": "new-stable-id", "time_ts": 1000}
        result = _filter_new_docs(
            [recrawled],
            {"old-id", _doc_fingerprint(old)},
            min_time_ts=800,
        )
        self.assertEqual(result, [])

    def test_one_cycle_uses_current_cutoff_and_upserts_only_new_docs(self):
        now = 1_000_000
        chunks = [
            {"id": "existing", "time_ts": now - 10, "content": "a"},
            {"id": "new", "time_ts": now - 20, "content": "b"},
            {"id": "old", "time_ts": now - 4 * 86400, "content": "c"},
        ]
        captured = []

        loader_module = types.ModuleType("modules.ingestion.loader")

        def fake_load(docs, collection_name=None):
            captured.extend(docs)
            return len(docs)

        loader_module.load_to_vector_db = fake_load
        with patch("modules.ingestion.scheduler.crawl_cafef_stock", return_value=[{"id": "a"}]), patch(
            "modules.ingestion.scheduler.preprocess_articles", return_value=chunks
        ), patch(
            "modules.ingestion.scheduler._get_existing_ids_from_qdrant", return_value={"existing"}
        ), patch.dict(sys.modules, {"modules.ingestion.loader": loader_module}):
            stats = run_ingestion_cycle("cafef_articles", max_pages=3, max_age_days=3, now_ts=now)

        self.assertEqual([d["id"] for d in captured], ["new"])
        self.assertEqual(stats["upserted"], 1)
        self.assertEqual(stats["new_docs"], 1)


if __name__ == "__main__":
    unittest.main()

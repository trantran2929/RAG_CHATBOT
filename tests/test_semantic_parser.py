import json
import sys
import types
import unittest
from unittest.mock import patch

from modules.core.state import GlobalState
from modules.nodes.semantic_parser import (
    _extract_json,
    _preserves_protected_tokens,
    semantic_parse_node,
)


class FakeRedisClient:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        self.values[key] = value
        return True


class FakeModel:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, *_args, **_kwargs):
        self.calls += 1
        return types.SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


class SemanticParserTests(unittest.TestCase):
    def test_extracts_json_from_code_fence(self):
        result = _extract_json('```json\n{"intent":"time"}\n```')
        self.assertEqual(result["intent"], "time")

    def test_protected_numbers_and_tickers_must_survive(self):
        original = "giá FPT ngày 25/7/2026"
        self.assertTrue(_preserves_protected_tokens(original, "giá FPT ngày 25/7/2026"))
        self.assertFalse(_preserves_protected_tokens(original, "giá VCB ngày 24/7/2026"))

    def test_low_confidence_rule_result_uses_llm_parser(self):
        payload = {
            "intent": "forecast",
            "corrected_query": "dự báo giá FPT ngày mai",
            "tickers": ["FPT"],
            "location": None,
            "time_expression": "ngày mai",
            "confidence": 0.96,
        }
        model = FakeModel(payload)
        services = types.ModuleType("modules.utils.services")
        services.llm_services = types.SimpleNamespace(model=model)
        services.redis_services = types.SimpleNamespace(client=FakeRedisClient())
        state = GlobalState(
            user_query="dư bao gia FPT ngài mai",
            corrected_query="dư bao gia FPT ngài mai",
            intent="rag",
            intent_confidence=0.4,
        )

        with patch.dict(sys.modules, {"modules.utils.services": services}):
            result = semantic_parse_node(state)

        self.assertTrue(result.semantic_parser_used)
        self.assertEqual(result.intent, "forecast")
        self.assertEqual(result.tickers, ["FPT"])
        self.assertEqual(model.calls, 1)

    def test_clear_rule_intent_skips_llm(self):
        state = GlobalState(
            user_query="mấy giờ rồi",
            corrected_query="mấy giờ rồi",
            intent="time",
            intent_confidence=0.95,
        )
        result = semantic_parse_node(state)
        self.assertFalse(result.semantic_parser_used)


if __name__ == "__main__":
    unittest.main()

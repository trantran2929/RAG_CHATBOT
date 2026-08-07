from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any


ALLOWED_INTENTS = {
    "time", "weather", "stock", "market", "forecast", "rag", "greeting"
}
PARSER_VERSION = "v1"


def _extract_json(value: Any) -> dict:
    if hasattr(value, "content"):
        value = value.content
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("LLM did not return a JSON object")
    result = json.loads(match.group(0))
    if not isinstance(result, dict):
        raise ValueError("Semantic parse result is not an object")
    return result


def _protected_tokens(text: str) -> set[str]:
    numbers = re.findall(r"\b\d+(?:[./-]\d+)*\b", text or "")
    tickers = re.findall(r"\b[A-Z]{2,6}\b", text or "")
    return set(numbers + tickers)


def _preserves_protected_tokens(original: str, corrected: str) -> bool:
    return _protected_tokens(original).issubset(_protected_tokens(corrected))


def _should_parse(state) -> bool:
    if os.getenv("SEMANTIC_PARSER_ENABLED", "1") != "1":
        return False
    if getattr(state, "is_greeting", False):
        return False
    if getattr(state, "intent", "rag") != "rag":
        return False
    threshold = float(os.getenv("SEMANTIC_PARSER_TRIGGER_CONFIDENCE", "0.7"))
    return float(getattr(state, "intent_confidence", 0.0)) < threshold


def _cache_key(query: str) -> str:
    digest = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
    return f"semantic:{PARSER_VERSION}:{digest}"


def _semantic_prompt(query: str) -> list[dict[str, str]]:
    system = """Bạn là bộ phân tích ngữ nghĩa cho chatbot tài chính Việt Nam.
Không trả lời câu hỏi. Chỉ trả về đúng một JSON object, không Markdown.
Chỉ sửa lỗi chính tả khi chắc chắn; không thay đổi số, ngày, mã cổ phiếu,
tên người, công ty hoặc địa danh. Nếu không chắc, giữ nguyên câu.
intent chỉ được là: time, weather, stock, market, forecast, rag, greeting.
Schema:
{"intent":"...","corrected_query":"...","tickers":[],"location":null,
 "time_expression":null,"confidence":0.0}
stock = giá/lịch sử một mã; market = tin/phân tích thị trường hoặc mã;
forecast = dự báo cổ phiếu; rag = câu hỏi chung/không thuộc nhóm trên."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


def semantic_parse_fallback(state) -> dict | None:
    """Return a validated LLM parse when Processor rules are uncertain."""
    state.semantic_parser_used = False
    if not _should_parse(state):
        return None

    query = getattr(state, "corrected_query", "") or state.user_query
    key = _cache_key(query)
    parsed = None

    try:
        from modules.utils.services import llm_services, redis_services

        try:
            cached = redis_services.client.get(key)
            if cached:
                parsed = json.loads(cached)
        except Exception:
            parsed = None

        if parsed is None:
            output = llm_services.model.invoke(
                _semantic_prompt(query),
                temperature=0,
                max_tokens=250,
                timeout=float(os.getenv("SEMANTIC_PARSER_TIMEOUT", "4")),
            )
            parsed = _extract_json(output)

        intent = str(parsed.get("intent", "rag")).lower().strip()
        confidence = float(parsed.get("confidence", 0.0))
        corrected = str(parsed.get("corrected_query") or query).strip()
        minimum = float(os.getenv("SEMANTIC_PARSER_MIN_CONFIDENCE", "0.75"))

        if intent not in ALLOWED_INTENTS:
            raise ValueError(f"Unsupported semantic intent: {intent}")
        if confidence < minimum:
            raise ValueError(f"Semantic confidence too low: {confidence}")
        if not _preserves_protected_tokens(state.user_query, corrected):
            raise ValueError("Semantic correction changed a protected token")

        parsed["intent"] = intent
        parsed["confidence"] = confidence
        parsed["corrected_query"] = corrected

        try:
            redis_services.client.set(
                key,
                json.dumps(parsed, ensure_ascii=False),
                ex=int(os.getenv("SEMANTIC_PARSER_CACHE_TTL", str(7 * 86400))),
            )
        except Exception:
            pass

        return parsed
    except Exception as exc:
        state.add_debug("semantic_parser", "fallback")
        state.add_debug("semantic_parser_error", str(exc))
        return None

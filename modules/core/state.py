from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from langgraph.graph.message import add_messages
import uuid, time, datetime
import re


@dataclass
class GlobalState:
    """
    Global state quản lý toàn bộ dữ liệu trong RAG pipeline.
    """

    # Session ID để phân biệt hội thoại khác nhau
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Thời điểm query (timestamp)
    timestamp: float = field(default_factory=time.time)
    user_query: str = ""
    processed_query: str = ""
    lang: str = "unknown"
    # Chế độ trả lời (general, math_step_by_step, translation, code_generation)
    mode: str = "general"
    role: str = ""
    intent: str = "rag"     # “time”, “weather”, “stock”, “news”, “rag”, …
    route_to: Optional[str] = None  # “api”, “rag”, “multi_agent”...

    # khóa để check Redis cache
    cache_key: str = ""
    # Đánh dấu nếu lấy từ redis
    from_cache: bool = False
    cached_response: Optional[str] = None

    # Vector embedding cho processed_query
    query_embedding: Optional[Dict[str, Any]] = None
    # Kết quả tìm kiếm trong vector DB (ID + score)
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    # Docs thực tế được lấy từ retriever
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    # Context được tổng hợp (snippets gộp lại)
    context: str = ""
    # Tickers được trích xuất từ user_query
    tickers: List[str] = field(default_factory=list)
    time_filter: Optional[tuple] = None

    # Few-shot
    examples: List[Dict[str,str]] = field(default_factory=list)
    # Prompt đã được build cho LLM
    prompt: str = ""
    # Final Answer từ LLM
    raw_response: str = ""  # Output gốc của LLM
    llm_output: str = ""    # (tuỳ chọn: lưu log đầy đủ)
    llm_status: str = ""       # “success”, “error”, “empty_prompt”, “api_response”, ...
    final_answer: str = ""  # câu trả lời đã chuẩn hóa
    response: str = ""      # phản hồi gửi UI

    # Resonse cuối cùng trước khi thêm vào cached
    api_type: Optional[str] = None
    api_response: Optional[str] = None

    # Thông tin debug/tracking
    route: str = "RAG"                     # “RAG”, “API”, hoặc “Greeting”
    debug: bool = False                    # in log nếu True
    debug_info: Dict[str, Any] = field(default_factory=dict)

    # Toàn bộ messages trong hội thoại hiện tại
    messages: List[Dict[str, str]] = field(default_factory=list, metadata={"reducer": add_messages})
    # Lịch sử hội thoại
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    is_greeting: bool = False   # Chỉ chào hỏi
    is_confirmation: bool = False  # Các câu "có", "ok", "yes"
    is_smalltalk: bool = False  # Tán gẫu, không cần gọi LLM

    
    def add_debug(self, key: str, value: Any):
        """Thêm thông tin debug vào state."""
        self.debug_info[key] = value

    def add_message(self, role: str, content: str):
        """Thêm 1 message vào hội thoại hiện tại."""
        self.messages.append({"role": role, "content": content})

    def extract_tickers(self):
        """Trích xuất tickers"""
        if self.user_query:
            self.tickers = re.findall(r"\b[A-Z]{2,5}\b", self.user_query)
        return self.tickers
    def formatted_time(self):
        return datetime.datetime.fromtimestamp(self.timestamp).strftime("%d/%m/%Y %H:%M:%S")
    
    def set_final_answer(self, text: str, route: str = "API"):
        """Gán câu trả lời cuối cùng + đánh dấu route."""
        self.final_answer = text
        self.response = text
        self.route = route
        self.add_debug("final_answer", text)

    def mark_api_response(self, api_type: str, result: Any, text: str):
        """Ghi nhận phản hồi từ API (time/weather/stock...)."""
        self.api_type = api_type
        self.api_result = result
        self.api_response = text
        self.set_final_answer(text, route="API")
        self.llm_status = "api_response"
        self.add_debug("api_type", api_type)
        self.add_debug("api_result", result)
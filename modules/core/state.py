from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from langgraph.graph.message import add_messages
import uuid, time


@dataclass
class GlobalState:
    """
    Global state quản lý toàn bộ dữ liệu trong RAG pipeline.
    """

    # Session ID để phân biệt hội thoại khác nhau
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Input gốc từ người dùng
    user_query: str = ""

    # Query sau khi xử lý (clean, detect ngôn ngữ... )
    processed_query: str = ""

    # Ngôn ngữ detect
    lang: str = "unknown"

    role: str = ""

    # Chế độ trả lời (general, math_step_by_step, translation, code_generation)
    mode: str = "general"

    # Thời điểm query (timestamp)
    timestamp: float = field(default_factory=time.time)

    # khóa để check Redis cache
    cache_key: str = ""

    # Đánh dấu nếu lấy từ redis
    from_cache: bool = False

    # Vector embedding cho processed_query
    query_embedding: Optional[Dict[str, Any]] = None

    # Kết quả tìm kiếm trong vector DB (ID + score)
    search_results: List[Dict[str, Any]] = field(default_factory=list)

    # Docs thực tế được lấy từ retriever
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)

    # Context được tổng hợp (snippets gộp lại)
    context: str = ""

    # Few-shot
    examples: List[Dict[str,str]] = field(default_factory=list)
    
    # Prompt đã được build cho LLM
    prompt: str = ""

    # Toàn bộ messages trong hội thoại hiện tại
    # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    messages: List[Dict[str, str]] = field(default_factory=list, metadata={"reducer": add_messages})

    # Lịch sử hội thoại
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    # Final Answer từ LLM
    raw_response: str = ""  # Output gốc của LLM
    llm_output: str = ""
    final_answer: str = ""

    # Resonse cuối cùng trước khi thêm vào cached
    response: str = ""
    
    # Thông tin debug/tracking
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def add_debug(self, key: str, value: Any):
        """Thêm thông tin debug vào state."""
        self.debug_info[key] = value

    def add_message(self, role: str, content: str):
        """Thêm 1 message vào hội thoại hiện tại."""
        self.messages.append({"role": role, "content": content})
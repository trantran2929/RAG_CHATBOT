from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from langgraph.graph.message import add_messages
import uuid


@dataclass
class GlobalState:
    """
    Global state quản lý toàn bộ dữ liệu trong RAG pipeline.
    """

    # Session ID để phân biệt hội thoại khác nhau
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Input gốc từ người dùng
    raw_query: str = ""

    # Query sau khi xử lý (clean, detect ngôn ngữ... )
    clean_query: str = ""

    # Ngôn ngữ detect
    lang: str = "unknown"

    # Vector embedding cho processed_query
    query_embedding: Optional[List[float]] = None

    # Kết quả tìm kiếm trong vector DB (chỉ ID + score)
    search_results: List[Dict[str, Any]] = field(default_factory=list)

    # Tài liệu thực tế được lấy từ retriever
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)

    # Context được tổng hợp (snippets gộp lại)
    context: str = ""
    
    # Prompt đã được build cho LLM
    prompt: str = ""

    # Final Answer từ LLM
    final_answer: str = ""

    # Toàn bộ messages trong hội thoại hiện tại
    # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    messages: List[Dict[str, str]] = field(default_factory=list, metadata={"reducer": add_messages})

    # Hội thoại trước đó (lấy từ Redis memory)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    # Thông tin debug/tracking
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def add_debug(self, key: str, value: Any):
        """Thêm thông tin debug vào state."""
        self.debug_info[key] = value

    def add_message(self, role: str, content: str):
        """Thêm 1 message vào hội thoại hiện tại."""
        self.messages.append({"role": role, "content": content})
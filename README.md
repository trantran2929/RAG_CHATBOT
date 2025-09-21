# RAG_CHATBOT

rag_project/
│── main.py
│── requirements.txt
│
├── ui/
│   └── app.py                  # Streamlit UI
│
├── modules/
│   ├── core/               # Điều phối (graph + state)
│   │   ├── graph.py
│   │   └── state.py
│   │
│   ├── nodes/                  # Các bước xử lý RAG
│   │   ├── cache.py
│   │   ├── processor.py        # Xử lý query (detect ngôn ngữ, clean)
│   │   ├── embedder.py         # Embedding
│   │   ├── vector_db.py        # Knowledge base + similarity search
│   │   ├── retriever.py        # Lấy snippets từ vector DB
│   │   ├── prompt_builder.py   # Xây prompt động
│   │   ├── response_generator.py   # Wrapper LLM backend
│   │   ├── sevices.py
│   │   └── response.py         # Hàm chatbot_response (tích hợp RAG)
│   │
│   └── utils/                  # Tiện ích chung
│       ├── debug.py
│       └── logger.py (tùy chọn)
│
└── data/
    └── documents.json          # Knowledge base

# WORKFLOW

[User Input]
     │
     ▼
[Processor] ──► Clean & Normalize ──► Detect Language
     │
     ▼
[Check Short Greeting?] ──► Yes ──► Return canned response ("Xin chào") ──┐
     │                                                                   │
     ▼                                                                   │
     No                                                                  │
     │                                                                   │
[Embedder] ──► Convert query to vector                                   │
     │                                                                   │
     ▼                                                                   │
[Vector DB: Qdrant Search]                                               │
     │
     ├─► Match found (cosine similarity > threshold)? ──► Yes ──► Use existing snippets ─┐
     │                                                                                   │
     ▼                                                                                   │
     No                                                                                   │
     │                                                                                   │
[Retriever] ──► Select top-k relevant snippets                                         │
     │                                                                                   │
     ▼                                                                                   │
[Prompt Builder] ──► Build final prompt including:                                      │
     │                  - User query                                                   │
     │                  - Retrieved snippets                                           │
     │                  - Conversation context (Redis)                                 │
     ▼                                                                                   │
[Redis: Load Session History] ──► Merge with current context                            │
     │                                                                                   │
     ▼                                                                                   │
[LLM Engine] ──► Generate natural language answer                                      │
     │
     ▼
[Response] ──► Return to user & append to Redis conversation history
     │
     ▼
[Vector DB: Store if new question] ──► Only if:
     │    - Not a short greeting
     │    - Not duplicate (cosine similarity below threshold)
     │    - Assign version + timestamp
     │    - Convert vector → list
     │    - Type = "knowledge"
     ▼
[End]


# Giải thích:
1. Processor:
- Làm sạch câu hỏi, chuẩn hóa chữ hoa/thường, loại bỏ ký tự thừa.
- Detect ngôn ngữ để build prompt phù hợp.

2. Check Short Greeting:
- Các câu như "hi", "hello", "xin chào" được lọc ra, không lưu Qdrant, trả câu chào sẵn.

3. Embedder:
- Chuyển query thành vector embedding.
- Chuẩn bị cho bước search Qdrant.

4. Vector DB (Qdrant):
- Tìm các câu hỏi tương tự.
- Nếu trùng (cosine similarity cao) → sử dụng snippets cũ, không lưu lại.
- Nếu chưa có → lưu vector + payload mới (có version, timestamp).

5. Retriever:
- Lấy top-k snippets liên quan để đưa vào prompt.

6. Prompt Builder:
- Tích hợp: user query + snippets + lịch sử hội thoại từ Redis.
- Tạo prompt tối ưu cho LLM.
7. Redis:
- Lưu context hội thoại theo session, không lưu lâu dài.
- Hỗ trợ ngữ cảnh hội thoại liên tục.

8. LLM Engine:
- Sinh câu trả lời tự nhiên dựa trên prompt cuối cùng.

9. Response
- Trả kết quả cho user.
- Append conversation vào Redis để duy trì lịch sử.

10. Vector DB (lưu mới):
- Chỉ lưu câu hỏi mới, không trùng, không phải greeting.
- Chuẩn hóa vector → list, gắn version, timestamp.
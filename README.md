# RAG_CHATBOT

rag_project/ <br>
│── docker-compose.yml <br>
│── requirements.txt <br>
│    <br>
├── ui/      <br>              
│   └── app.py <br>
│    <br>
├── modules/ <br>
│   ├── core/ <br>
│   │   ├── graph.py <br>
│   │   └── state.py     <br>
│   │     <br>
│   ├── nodes/ <br>
│   │   ├── cache.py     <br>
│   │   ├── processor.py      <br>
│   │   ├── embedder.py       <br>
│   │   ├── vector_db.py      <br>
│   │   ├── retriever.py      <br>
│   │   ├── prompt_builder.py      <br>
│   │   ├── response_generator.py       <br>
│   │   ├── response.py       <br>
│   │   └── services.py       <br>
│   │     <br>
│   ├── ingestion/              # service ingestion chạy trong container   <br>
│   │   ├── Dockerfile        <br>
│   │   ├── crawler.py        <br>
│   │   ├── preprocess.py          <br>
│   │   ├── loader.py         <br>
│   │   └── scheduler.py        # optional: chạy loop/celery/cron     <br>
│   │     <br>
│   └── utils/      <br>
│       ├── debug.py          <br>
│       └── logger.py         <br>
│         <br>
└── data/      <br>
    └── documents.json   <br>
       
# WORKFLOW

[User Input]
     │
     ▼
[Processor] ──► Clean & Normalize ──► Detect Language
     │
     ▼

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
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
│   ├── api/ <br>
│   │   ├── stock_api.py <br>
│   │   ├── time_api.py <br>
│   │   └── weather_api.py     <br>
│   │     <br>
│   ├── nodes/ <br>
│   │   ├── cache.py          # Lưu lịch sử hội thoại  <br> 
│   │   ├── processor.py      # Tiền xử lý <br>
│   │   ├── embedder.py       <br>
│   │   ├── vector_db.py      <br>
│   │   ├── retriever.py      <br>
│   │   ├── prompt_builder.py      <br>
│   │   ├── response_generator.py       <br>
│   │   ├── router.py       <br>
│   │     <br>
│   ├── ingestion/              # crawl và xử lý dữ liệu   <br>
│   │   ├── Dockerfile        <br>
│   │   ├── crawler.py        <br>
│   │   ├── preprocess.py          <br>
│   │   ├── loader.py         <br>
│   │   └── scheduler.py        # optional: chạy loop/celery/cron     <br>
│   │     <br>
│   └── utils/      <br>
│       ├── __init__.py          <br>
│       ├── debug.py          <br>
│       ├── logger.py         <br>
|       ├── qdrant_ultils.py  <br>
│       └── service.py       <br>
│         <br>
└── data/      <br>
    └── documents.json   <br>


rag_project/    <br>
│── docker-compose.yml  <br>
│── requirements.txt    <br>
│   <br>
├── ui/ <br>
│   └── app.py                      # Streamlit/FastAPI UI hoặc REST interface  <br>
│   <br>
├── modules/    <br>
│   ├── core/   <br>
│   │   ├── graph.py                # LangGraph: định nghĩa các node và edges   <br>
│   │   ├── state.py                # GlobalState + cấu trúc dữ liệu hội thoại  <br>
│   │   └── router.py (optional)    # có thể để ở đây nếu router liên quan trực tiếp đến graph  <br>
│   │   <br>
│   ├── api/    <br>
│   │   ├── stock_api.py            # REST + WebSocket chứng khoán (Vnstock + VNDirect) <br>
│   │   ├── time_api.py             # API thời gian hiện tại (UTC+7)    <br>
│   │   ├── weather_api.py          # API thời tiết (OpenWeather)   <br>
│   │   └── news_api.py (optional)  # API tin tức nhanh (VD: RSS, Vietstock)    <br>
│   │   <br>
│   ├── nodes/  <br>
│   │   ├── cache.py                # Redis cache (conversation, vector results)    <br>
│   │   ├── processor.py            # Tiền xử lý câu hỏi người dùng (detect intent, clean)  <br>
│   │   ├── router.py               # Phân luồng: gọi API hay RAG pipeline  <br>
│   │   ├── embedder.py             # Sinh embedding từ text    <br>
│   │   ├── vector_db.py            # CRUD Qdrant: upsert, search, filter   <br>
│   │   ├── retriever.py            # Lấy top-k docs từ Qdrant  <br>
│   │   ├── prompt_builder.py       # Xây prompt kết hợp Context + API data <br>
│   │   ├── response_generator.py   # Gọi LLM (Gemma/Ollama/vLLM)   <br>
│   │   ├── response.py             # Chuẩn hóa kết quả (format, append history)    <br>
│   │   └── evaluator.py (optional) # Đánh giá chất lượng phản hồi (RAG, factuality)    <br>
│   │   <br>
│   ├── ingestion/  <br>
│   │   ├── Dockerfile  <br>
│   │   ├── crawler.py              # Crawl CafeF, Vietstock, ...   <br>
│   │   ├── preprocess.py           # Làm sạch + tách đoạn text <br>
│   │   ├── loader.py               # Tạo embedding, upsert vào Qdrant  <br>
│   │   ├── scheduler.py            # Cron job tự crawl định kỳ <br>
│   │   └── validator.py (optional) # Kiểm tra dữ liệu trước khi insert (duplicate, null)   <br>
│   │   <br>
│   ├── ml/ (mới thêm)              # phục vụ huấn luyện dự đoán / phân tích nâng cao    <br>
│   │   ├── predictor.py            # Mô hình dự đoán thị trường (ARIMA, LSTM, Prophet) <br>
│   │   ├── sentiment_model.py      # Mô hình phân tích cảm xúc bài báo <br>
│   │   ├── train_utils.py          # Tiện ích huấn luyện, chia tập, scaler <br>
│   │   └── model_store/            # Lưu checkpoint / model weights    <br>
│   │   <br>
│   └── utils/  <br>
│       ├── debug.py                # Hàm log + thêm debug_info vào state   <br>
│       ├── logger.py               # Logging thống nhất    <br>
│       ├── qdrant_utils.py         # Đổi tên chuẩn (fix lỗi "qdrant_ultils")   <br>
│       ├── redis_utils.py (optional) # Các hàm Redis helper    <br>
│       ├── api_utils.py (optional) # Hàm chung cho gọi API (retry, timeout)    <br>
│       └── services.py             # Qdrant, Redis, LLM, Embedding service wrapper <br>
│   <br>
└── data/   <br>
    ├── documents.json              # Lưu tài liệu đã crawl <br>
    ├── raw/                        # Dữ liệu gốc (HTML, JSON)  <br>
    ├── processed/                  # Sau preprocess    <br>
    ├── embeddings/                 # Cache embedding (nếu cần) <br>
    └── models/                     # Model đã huấn luyện (ARIMA, Prophet, v.v.)    <br>

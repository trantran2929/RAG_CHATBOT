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


rag_project/ <br>
│── README.md   <br>
│── .env                      <br>
│── requirements.txt    <br>
│── docker-compose.yml                  <br>
│
├── ui/ <br>
│   └── app.py                          <br>
│   <br>
├── modules/    <br>
│   ├── core/   <br>
│   │   ├── graph.py     <br>
│   │   └──state.py                    <br>
│   │   <br>
│   ├── nodes/  <br>
│   │   ├── cache.py                    <br>
│   │   ├── processor.py                <br>
│   │   ├── embedder.py <br>
│   │   ├── vector_db.py    <br>
│   │   ├── retriever.py    <br>
│   │   ├── prompt_builder.py   <br>
│   │   ├── response_generator.py   <br>
│   │   └── router.py                   <br>
│   │   <br>
│   ├── api/    <br>
│   │   ├── stock_api.py               <br>
│   │   ├── time_api.py <br>
│   │   └── weather_api.py  <br>
│   │   <br>
│   ├── ML/ <br>
│   │   ├── predictors/ <br>
│   │   │   └── sarimax_exog.py         # SARIMAX + exog tin tức (shift T-1)    <br>
│   │   ├── features.py                 # build_news_features() lấy CafeF từ Qdrant → news_count/sent_* <br>
│   │   ├── pipeline.py                 # train()/forecast(): lấy giá từ stock_api + exog từ features   <br>
│   │   ├── registry.py                 # (optional) lưu/đọc model/meta (nếu cần persist)   <br>
│   │   └── metrics.py                  # (optional) RMSE/MAPE utils    <br>
│   │   <br>
│   ├── ingestion/  <br> 
│   │   ├── Dockerfile  <br>
│   │   ├── crawler.py         <br>
│   │   ├── preprocess.py               <br>
│   │   ├── loader.py                   <br>
│   │   └── scheduler.py                <br>
│   │   <br>
│   └── utils/  <br>
│       ├── logger.py   <br>
│       ├── debug.py    <br>
│       ├── qdrant_utils.py <br>
│       └── services.py          <br>

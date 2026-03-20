# 🚀 AI Financial Assistant & Stock Forecasting System

<p align="center">
  <b>Trợ lý tài chính AI kết hợp RAG + Machine Learning + LLM</b><br/>
  Phân tích – Trả lời – Dự báo thị trường chứng khoán 🇻🇳
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/LLM-Llama--3-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/VectorDB-Qdrant-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/RAG-Hybrid-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square"/>
</p>

---

## 📌 Giới thiệu

Đây là hệ thống **AI Financial Assistant** có khả năng:

- 🧠 Hiểu câu hỏi tài chính bằng tiếng Việt  
- 🔎 Truy xuất thông tin từ dữ liệu thực (RAG)  
- 📰 Phân tích tin tức + sentiment thị trường  
- 📈 Dự báo giá cổ phiếu bằng mô hình Machine Learning  
- 🤖 Sinh câu trả lời thông minh bằng LLM  

👉 Mục tiêu: xây dựng một **AI Assistant chuyên sâu cho chứng khoán Việt Nam**

---

## ✨ Demo Use Cases

```text

## 🔥 Tính năng nổi bật

### 🧠 1. Hiểu ngữ nghĩa & Intent Detection

- Tự động nhận diện:
  - Mã cổ phiếu (VCB, FPT, HPG…)
  - Ý định người dùng (giá, tin tức, dự báo…)
- Hỗ trợ tiếng Việt tự nhiên
- Chuẩn hóa và xử lý query trước khi đưa vào pipeline

---

### 🔍 2. Hybrid Search (RAG nâng cao)

Kết hợp nhiều kỹ thuật:

- Dense Embedding (Semantic Search)
- BM25 (Keyword Search)
- RRF (Reciprocal Rank Fusion)
- Cross-Encoder Reranking

👉 Giúp tăng độ chính xác truy xuất thông tin đáng kể so với chỉ dùng 1 phương pháp

---

### 📰 3. Pipeline dữ liệu tài chính

- Crawl dữ liệu từ **CafeF**
- Làm sạch và chuẩn hóa văn bản
- Chunk document để tối ưu retrieval
- Trích xuất thông tin:
  - Mã cổ phiếu
  - Chỉ số thị trường (VNIndex, VN30…)
  - Sentiment (tích cực / tiêu cực / trung lập)

---

### 🤖 4. LLM Integration

- Model: **Llama-3 8B Instruct (vLLM)**
- Trả lời dựa trên context từ RAG
- Output:
  - Tự nhiên
  - Ngắn gọn
  - Dễ hiểu với người dùng

---

### 📈 5. Dự báo cổ phiếu (Machine Learning)

- Model: **SARIMAX (Time Series Forecasting)**

Kết hợp nhiều nguồn dữ liệu:

- Giá cổ phiếu lịch sử  
- Sentiment từ tin tức  
- Chỉ số thị trường  

👉 Dự báo:

- Giá phiên tiếp theo  
- Xu hướng tăng / giảm  
- Mức độ tin cậy (confidence)  

---

### ⚡ 6. Cache thông minh

- Redis (real-time caching)
- Fallback: lưu file local
- Lưu lịch sử hội thoại
- Giảm latency và chi phí gọi LLM

---

### 🔄 7. Auto Data Pipeline

- Tự động crawl dữ liệu mới theo lịch
- Loại bỏ dữ liệu trùng lặp
- Cập nhật liên tục vào Vector DB
- Đảm bảo dữ liệu luôn fresh

---

## 🏗️ Tech Stack

### 🤖 AI / NLP

- Sentence Transformers  
- BM25 (rank_bm25)  
- Cross-Encoder (reranking)  
- Llama-3 (vLLM)  

---

### 📊 Machine Learning

- SARIMAX (statsmodels)  
- Pandas  
- NumPy  

---

### 🗄️ Data

- Qdrant (Vector Database)  
- Redis (Cache)  

---

### 🌐 Data Sources

- CafeF (Financial News)  
- VNStock API  

---

### ⚙️ Backend

- Python  
- LangGraph (Pipeline orchestration)  
• Giá cổ phiếu FPT hôm nay thế nào?
• Tin tức đáng chú ý về VNIndex
• Có nên mua HPG không?
• Dự báo VCB phiên tới

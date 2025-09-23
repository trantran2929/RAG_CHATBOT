# modules/ingestion/Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Cài đặt các package cần thiết (system deps) 
# -> để tránh lỗi khi build một số lib Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để cache layer pip
COPY requirements.txt .

# Cài Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir accelerate

# Copy toàn bộ mã nguồn ingestion
COPY . .

# Default command khi container chạy
CMD ["python", "-m", "modules.ingestion.scheduler"]



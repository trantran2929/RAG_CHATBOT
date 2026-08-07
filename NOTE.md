## MOMENTUM

Momentum là quy tắc dự đoán rằng xu hướng gần đây sẽ tiếp tục:
- Phiên trước tăng → dự đoán phiên sau tăng.
- Phiên trước giảm → dự đoán phiên sau giảm.

Ví dụ: FPT hôm qua tăng 2%, momentum dự đoán hôm nay tiếp tục tăng.

Trong dự án, quy tắc này chỉ là mốc so sánh. Model SARIMAX phải dự báo tốt hơn quy tắc đơn giản đó thì mới được xem là có giá trị.

Momentum báo tăng
      +
Thị trường ổn
      +
Tin tức không xấu
      +
Độ tin cậy đủ cao
      ↓
     BUY

Nếu các tín hiệu mâu thuẫn
      ↓
  NO_TRADE hoặc thoát lệnh

## Update data to qdrant

```powershell
docker compose exec -T ingestion python -c "from modules.ingestion.scheduler import run_ingestion_cycle; print(run_ingestion_cycle(collection_name='cafef_articles', max_pages=1, max_age_days=3))"
```

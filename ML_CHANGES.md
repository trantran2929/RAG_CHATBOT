# Ghi chú thay đổi hệ thống dự báo ML

Ngày cập nhật: 20/07/2026

## Mục tiêu

Đợt sửa này loại bỏ model quá hạn, phiên giao dịch giả, sai đơn vị giá,
dự báo PM cố định và backtest rò rỉ dữ liệu. Kết quả dự báo vẫn chỉ mang tính
tham khảo; metrics phải được theo dõi trước khi sử dụng cho quyết định đầu tư.

## Thay đổi đã thực hiện

### 1. Đơn vị giá

- Giá lịch sử cổ phiếu từ VNStock được đổi từ nghìn đồng sang VNĐ ở tầng API.
- Giá chỉ số thị trường không bị nhân 1.000 và tiếp tục dùng đơn vị điểm.
- Dữ liệu train, giá tại ngày cụ thể, fallback realtime và giao diện dự báo dùng
  cùng thang VNĐ.
- Model cũ có thang giá nghìn đồng hoặc index Statsmodels không tương thích bị
  vô hiệu qua `schema_version=3`.

### 2. Phiên giao dịch và ngày nghỉ

- Bỏ `asfreq("D").ffill()`: không còn tạo dòng Thứ Bảy/Chủ Nhật và không còn
  sao chép volume sang ngày không giao dịch.
- Thêm package `holidays` để nhận lịch nghỉ Việt Nam, bao gồm ngày nghỉ bù và
  Tết theo dữ liệu của package.
- Có fallback cho các ngày lễ cố định nếu package tạm thời chưa khả dụng.

### 3. Freshness và tự retrain

Metadata model mới có:

- `schema_version`
- `last_train_date`
- `holdout.rmse`
- `holdout.mae`
- `holdout.directional_accuracy`
- `holdout.test_size`

Mặc định `MODEL_MAX_STALENESS_DAYS=0`: model phải học tới đúng phiên thị trường
gần nhất. Model thiếu metadata, sai schema hoặc cũ hơn phiên mới nhất sẽ tự train.

Ingestion kiểm tra và refresh model sau mỗi vòng crawl. Danh sách mặc định:

```text
FPT,HPG,VCB,SHS,THU
```

Đổi danh sách bằng `ML_RETRAIN_SYMBOLS`. Tắt background retrain bằng
`ML_RETRAIN_AFTER_INGEST=0`.

### 4. Lưu model bền vững

Docker mount:

```yaml
./models:/app/models
```

Model không mất khi recreate container. File model và metadata được ghi vào file
tạm rồi `os.replace`, tránh đọc phải file ghi dở khi process bị dừng.

### 5. Dự báo phiên sáng

- SARIMAX vẫn dự báo log-return mở cửa phiên kế tiếp.
- Nếu Qdrant khả dụng, model dùng news/sentiment và price-lag exogenous features.
- Nếu Qdrant lỗi, hệ thống train SARIMAX price-only thay vì làm hỏng toàn bộ API.
- Qdrant dùng client payload nhẹ; không còn nạp embedder/reranker/LLM khi import ML.
- Grid mặc định giảm xuống `p,q <= 2`, cấu hình qua `SARIMAX_MAX_P/Q`.

### 6. Dự báo PM và intraday

- Bỏ dự báo PM cố định `0% ±0.5%`.
- Bỏ trung bình momentum 5 bước.
- Fit `AutoReg` trên log-return thực tế, dự báo bước tiếp theo và lấy độ lệch chuẩn
  residual để tạo biên bất định.
- Intraday ưu tiên OHLC intraday. Nếu provider không trả dữ liệu, dùng AutoReg
  daily và đánh dấu `source_used=daily_fallback`, độ tin cậy thấp.

### 7. Backtest

`backtest_gap_model()` hiện dùng expanding-window:

1. Train chỉ bằng dữ liệu trước ngày cần dự báo.
2. Dự báo đúng một phiên.
3. Mở rộng cửa sổ thêm một phiên.
4. Tính RMSE, MAE và directional accuracy out-of-sample.

Backtest này đánh giá thành phần price-return và cố ý không dùng news exog vì hệ
thống chưa lưu snapshot tin tức point-in-time. Điều này tránh dùng tin được ingest
sau thời điểm dự báo.

### 8. Tích hợp `forecast_api`

- `Router → forecast_api → pipeline` dùng chung schema AM/PM/intraday.
- Formatter kiểm tra đủ `px_mean/px_lo/px_hi`; pack thiếu trường trả thông báo an
  toàn thay vì lỗi format `None`.
- Giá hiển thị theo VNĐ và làm tròn hợp lý cho intraday.
- Confidence được dịch sang tiếng Việt.
- Đầu ra AM hiển thị ngày model đã học tới.
- Đầu ra PM ghi rõ AutoReg và có dùng daily fallback hay không.
- Daily fallback nội phiên không còn được trình bày như dữ liệu intraday thật.
- Pack có `error` hoặc thiếu hướng dự báo trả lỗi rõ ràng.
- Ngoài phiên không còn gọi `predict_next_session()` lặp hai lần.
- Có regression test cho formatter AM, PM, intraday và kết nối Router.
- SARIMAX fit trên `RangeIndex`; ngày giao dịch thật được giữ trong metadata và
  news features, tránh cảnh báo/index error ở các bản Statsmodels tương lai.

Chạy backtest:

```powershell
docker compose exec ingestion python -c "from modules.ML.backtest import backtest_gap_model,print_backtest_report; r=backtest_gap_model('FPT',60); print_backtest_report(r)"
```

## Kết quả smoke test FPT

Cold-start ngày 20/07/2026:

```text
last_train_date: 2026-07-20
last_close: 67,100 VNĐ
next_price_est: 66,963.95 VNĐ
holdout RMSE: 0.01677 log-return
holdout MAE: 0.01039 log-return
holdout directional accuracy: 55% (20 phiên)
confidence: uncertain
```

Cold-start đầy đủ mất khoảng 114 giây với grid cũ `p,q <= 3`. Grid mới mặc định
`p,q <= 2` giảm số cấu hình cần fit. Background retrain giúp thời gian này không
chặn request người dùng.

## Lệnh triển khai

Vì có dependency `holidays` và thay đổi Docker volume, cần build/recreate:

```powershell
docker compose down
docker compose build ingestion
docker compose up -d
docker compose logs -f ingestion
```

Kiểm tra metadata:

```powershell
docker compose exec ingestion python -c "import json; print(json.load(open('models/FPT_gap.json',encoding='utf-8')))"
```

Dự báo phiên kế tiếp:

```powershell
docker compose exec ingestion python -c "from modules.ML.pipeline import predict_next_session; print(predict_next_session('FPT'))"
```

## Giới hạn còn lại

- SARIMAX/AutoReg của Statsmodels chạy CPU; GPU không làm các model này nhanh hơn.
- Intraday phụ thuộc provider VNStock. Khi provider không có dữ liệu, kết quả là
  daily fallback chứ không phải dự báo vi mô nội phiên.
- Directional accuracy 55% trên 20 phiên là thấp và mẫu nhỏ; chưa đủ cơ sở dùng
  làm tín hiệu giao dịch.
- News exog còn thưa. Cần tích lũy thêm dữ liệu và xây snapshot point-in-time
  trước khi backtest được tác động thực của sentiment.
- Không nên diễn giải khoảng tin cậy thống kê thành xác suất chắc chắn tăng/giảm.

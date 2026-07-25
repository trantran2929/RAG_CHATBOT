# Ghi chú thay đổi ingestion

Ngày cập nhật: 21/07/2026.

## Hành vi sau khi sửa

- Khi container `ingestion` khởi động, một chu kỳ crawl chạy ngay lập tức.
- Sau đó hệ thống kiểm tra trang `https://cafef.vn/thi-truong-chung-khoan.chn` mỗi 900 giây (15 phút).
- Chỉ các bài trong 3 ngày gần nhất và chưa có trong Qdrant mới được nạp.
- ID chunk không còn phụ thuộc thời gian tương đối như "5 phút trước", nên cùng một bài không sinh ID mới ở lần crawl sau.
- Ngoài ID, hệ thống so sánh dấu vân tay URL + nội dung để tránh trùng với dữ liệu đã nạp bằng ID cũ.
- Qdrant dùng `wait=True`, nên log hoàn tất chỉ xuất hiện sau khi dữ liệu đã được ghi.
- Refresh model ML chạy nền; việc train lâu không còn chặn chu kỳ crawl tiếp theo.
- Mã nguồn `./modules` được mount vào `/app/modules`, nên có thể áp dụng sửa đổi bằng cách recreate container mà không phải build lại image CUDA lớn.
- Mốc lọc 3 ngày được tính lại ở từng chu kỳ, không bị cố định từ lúc container khởi động.
- HTTP có retry/backoff và User-Agent rõ ràng.
- Python chạy unbuffered để log từng chu kỳ xuất hiện ngay trong `docker compose logs -f ingestion`.
- Bước kiểm tra ID Qdrant dùng client nhẹ, không nạp embedder/reranker khi không có tin mới.

## Lưu ý về CafeF

URL phân trang cũ `/thi-truong-chung-khoan/trang-N.chn` hiện trả HTTP 404. Crawler chỉ quét trang danh mục mới nhất. Việc quét định kỳ cộng với chống trùng bảo đảm bài mới xuất hiện trên trang chính sẽ được phát hiện mà không gọi URL đã hỏng.

## Cấu hình

- `INGEST_INTERVAL=900`: số giây giữa hai lần kiểm tra.
- `CRAWL_MAX_PAGES=1`: giữ ở 1 vì phân trang cũ không còn hoạt động.
- `MAX_NEWS_AGE_DAYS=3`: chỉ nạp tin trong số ngày gần nhất.

## Lệnh triển khai và kiểm tra

```powershell
docker compose up -d --no-build --force-recreate ingestion
docker compose logs -f ingestion
```

Log một chu kỳ bình thường:

```text
[Crawler] Crawled ... articles
[Ingestion] Cycle stats: {'articles': ..., 'chunks': ..., 'new_docs': ..., 'upserted': ...}
```

Kiểm tra tổng số point:

```powershell
(Invoke-RestMethod http://localhost:6333/collections/cafef_articles).result.points_count
```

Khi chưa có tin mới, `new_docs=0, upserted=0` là đúng. Khi CafeF đăng bài mới, chu kỳ gần nhất phải có `new_docs` và `upserted` lớn hơn 0.

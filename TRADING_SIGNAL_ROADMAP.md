# Lộ trình nâng cấp mô hình thành tín hiệu giao dịch

> **Trạng thái:** Pha 1 đã triển khai và kiểm thử; Pha 2–5 chưa triển khai.
>
> Mục tiêu không phải tối đa hóa accuracy trên dữ liệu quá khứ, mà xây dựng tín
> hiệu có khả năng hoạt động trên dữ liệu ngoài mẫu và còn hiệu quả sau phí giao
> dịch. Kết quả dự báo chỉ mang tính tham khảo, không phải khuyến nghị đầu tư.

## 1. Các vấn đề cần giải quyết trước

### Accuracy hiện tại chưa đo đúng model production

Pipeline dự báo chính dùng SARIMAX và có thể sử dụng biến ngoại sinh từ tin tức.
Tuy nhiên, `_holdout_metrics()` và `backtest_gap_model()` trước đây đánh giá
AutoReg price-only. Vì vậy, directional accuracy cũ chưa phản ánh đúng hiệu quả
của model đang phục vụ người dùng.

### Tập kiểm tra còn quá nhỏ

Tập 20 phiên chỉ tương đương khoảng một tháng giao dịch. Với accuracy 55%, model
chỉ đúng 11 phiên và sai 9 phiên; chưa đủ bằng chứng để phân biệt năng lực dự báo
với kết quả ngẫu nhiên.

Đề xuất:

- Dùng tối thiểu 250–500 phiên out-of-sample.
- Bao phủ giai đoạn thị trường tăng, giảm, đi ngang và biến động mạnh.
- Báo cáo kết quả theo từng mã, từng market regime và toàn bộ danh mục.

### Cần kiểm tra alignment feature và target

Phải xác minh target phiên `t+1` chỉ sử dụng price lag và tin tức đã biết đến
phiên `t`. Đặc biệt cần kiểm tra hàng feature được tạo tại `last_train_date` để
loại bỏ lỗi off-by-one và look-ahead leakage.

### Mục tiêu hiện tại chưa phù hợp trực tiếp với giao dịch

Dự báo đúng dấu của một biến động rất nhỏ chưa chắc tạo ra lợi nhuận sau phí,
thuế, spread và slippage. Hệ thống cần tối ưu tín hiệu có thể giao dịch thay vì
buộc phải đoán tăng hoặc giảm ở mọi phiên.

## 2. Thiết kế target giao dịch

Target đề xuất gồm ba trạng thái:

```text
UP       nếu future_return > chi phí giao dịch + biên an toàn
DOWN     nếu future_return < -(chi phí giao dịch + biên an toàn)
NO_TRADE nếu return nằm giữa hai ngưỡng
```

Ngưỡng phải được cấu hình theo phí, thuế, spread, thanh khoản và slippage thực
tế. Model chỉ phát tín hiệu khi expected return đủ bù chi phí và xác suất vượt
ngưỡng được chọn trên validation set.

Việc có trạng thái `NO_TRADE` giúp tập trung vào precision của các tín hiệu đáng
tin cậy, thay vì cố dự báo mọi phiên.

## 3. Bộ đặc trưng point-in-time cần bổ sung

### Giá và thanh khoản

- Return 1, 2, 5, 10 và 20 phiên.
- Overnight gap và intraday return.
- High-low range và realized volatility 5/10/20 phiên.
- Relative volume, turnover và value traded.
- Khoảng cách tới MA/EMA.
- Drawdown và khoảng cách tới đỉnh 20/60 phiên.
- Một số ít indicator như RSI, ATR và MACD.

Không nên đưa hàng trăm indicator tương quan cao vào ngay từ đầu.

### Thị trường và ngành

- Return và volatility của VNINDEX/VN30.
- Market breadth nếu nguồn dữ liệu hỗ trợ.
- Return của ngành hoặc nhóm cổ phiếu liên quan.
- Relative strength so với VNINDEX.
- Rolling beta và rolling correlation.

### Market regime

Nhận diện các trạng thái:

- Bullish.
- Bearish.
- Sideways.
- High volatility.
- Low volatility.

Regime có thể được dùng như một feature hoặc dùng để áp dụng model/ngưỡng tín
hiệu riêng.

### Tin tức

- Sentiment có trọng số theo độ mới.
- Tách tin trước giờ mở cửa và trong phiên.
- Số nguồn độc lập nói về cùng sự kiện.
- Độ bất ngờ so với sentiment trung bình gần đây.
- Loại sự kiện: kết quả kinh doanh, cổ tức, phát hành, pháp lý hoặc M&A.
- Topic/embedding và mức liên quan trực tiếp đến mã cổ phiếu.

Tin tức phải được lưu thành snapshot point-in-time: tại thời điểm dự báo, model
chỉ được sử dụng dữ liệu đã thực sự xuất hiện và có thể thu thập trước thời điểm
đó.

## 4. Model đề xuất: XGBoost kết hợp SARIMAX

Giữ SARIMAX làm baseline dự báo return và thêm **XGBoost classifier** để dự báo
xác suất `UP / DOWN / NO_TRADE`.

XGBoost phù hợp với dataset hiện tại hơn deep learning vì:

- Học tốt quan hệ phi tuyến giữa dữ liệu dạng bảng.
- Hoạt động được với lượng dữ liệu vừa phải.
- Có regularization và kiểm soát độ phức tạp.
- Cung cấp xác suất để xây dựng ngưỡng `NO_TRADE`.
- Có thể phân tích feature importance và ablation.

Luồng ensemble dự kiến:

```text
OHLCV + thị trường + news
            |
Feature pipeline point-in-time
            |
     +------+------+
     |             |
 SARIMAX       XGBoost
  return     P(up/down/flat)
     |             |
     +------+------+
            |
 Probability calibration
            |
 Threshold + cost filter
            |
   BUY / SELL / NO_TRADE
```

Chỉ phát tín hiệu khi:

- Xác suất XGBoost vượt ngưỡng tin cậy.
- Expected return vượt tổng chi phí và biên an toàn.
- SARIMAX và XGBoost không mâu thuẫn mạnh.

Threshold phải được chọn trên validation data, không được tối ưu trên test cuối
cùng.

## 5. Thiết kế backtest mới

Mỗi bước walk-forward:

1. Train hoặc cập nhật model bằng dữ liệu trước ngày dự báo.
2. Tạo feature bằng đúng pipeline production.
3. Dự báo xác suất cho phiên kế tiếp.
4. Áp dụng threshold và cost filter.
5. Ghi nhận `BUY / SELL / NO_TRADE`.
6. Ghi kết quả thực tế và chuyển sang phiên kế tiếp.

Không dùng random K-fold cho dữ liệu chuỗi thời gian. Cần time-series split và
có thể dùng một khoảng `gap` giữa train và validation khi feature có cửa sổ
chồng lấn.

Các baseline cần so sánh:

- Luôn dự báo tăng.
- Dự báo theo hướng phiên trước.
- Buy-and-hold.
- Logistic regression đơn giản.
- SARIMAX hiện tại.

## 6. Metric đánh giá

### Metric dự báo

- Accuracy và balanced accuracy.
- Precision, recall và F1 riêng cho `UP` và `DOWN`.
- ROC-AUC và PR-AUC.
- Brier score và probability calibration.
- Coverage: tỷ lệ phiên model thực sự phát tín hiệu.

### Metric giao dịch sau chi phí

- Cumulative return và annualized return.
- Sharpe ratio.
- Maximum drawdown.
- Profit factor.
- Win rate và average win/loss.
- Turnover.
- Hiệu quả theo mã, theo thời gian và theo market regime.

Model mới chỉ được chấp nhận nếu vượt baseline ổn định trên dữ liệu hoàn toàn
ngoài mẫu, thay vì chỉ tăng một chỉ số accuracy.

## 7. Thứ tự triển khai

### Pha 1 — Sửa nền đánh giá

**Trạng thái: đã triển khai và kiểm thử.**

Pha 1 tập trung làm cho kết quả đánh giá đáng tin cậy và chuyển dự báo thống kê
thành tín hiệu có xét chi phí. Pha này chưa thêm XGBoost, feature kỹ thuật mới
hoặc deep learning.

### 1. Chuẩn hóa target và thời điểm phát tín hiệu

Target thực tế của SARIMAX được định nghĩa rõ là:

```text
next_session_close_to_close_log_return
= log(close[t+1] / close[t])
```

Tín hiệu được tạo sau khi phiên `t` đóng cửa để dự báo giá đóng cửa phiên
`t+1`. Giao diện đã đổi cách trình bày từ “giá mở cửa dự kiến” sang “giá đóng
cửa phiên kế tiếp dự kiến”, vì model không được huấn luyện để dự báo giá mở cửa.

Metadata model hiện lưu thêm:

- `target`: định nghĩa target chính xác.
- `signal_timing`: thời điểm tạo tín hiệu.
- `forecast_target_date`: ngày giao dịch được dự báo.

### 2. Sửa alignment của price-lag và tin tức

Trước khi sửa, hàng feature dùng cho dự báo phiên kế tiếp có thể bỏ qua return
mới nhất và dùng `ret_lag1` của phiên `t-1`.

Sau khi sửa:

```text
Dự báo t+1 ← price return đã quan sát đến hết t
Dự báo t+1 ← tin tức đã xuất hiện đến hết t
```

Các feature `ret_lag1`, `ret_lag2` và `ret_lag5` của hàng forecast được tính
trực tiếp từ các return mới nhất đã quan sát. Target date được đặt thành ngày
giao dịch kế tiếp, có loại cuối tuần và ngày nghỉ.

Regression test đã được thêm để xác nhận `ret_lag1` của target `t+1` đúng bằng
return mới nhất của phiên `t`.

### 3. Holdout dùng cùng họ model với production

Trước đây:

```text
Production: SARIMAX, có thể có exogenous features
Holdout:    AutoReg price-only
```

Directional accuracy trong metadata vì vậy không đo đúng họ model đang phục vụ
người dùng.

Sau khi sửa, holdout dùng expanding-window SARIMAX. Tại mỗi fold:

1. Chỉ dùng return trước target.
2. Fit scaler bằng tập train của fold.
3. Chọn và fit SARIMAX theo cùng cấu hình production.
4. Dự báo đúng một phiên kế tiếp.
5. So sánh với return thực tế.

Nếu có biến ngoại sinh point-in-time thì holdout dùng biến ngoại sinh đã align;
nếu không có thì chuyển sang SARIMAX price-only.

### 4. Thêm `BUY / SELL / NO_TRADE` sau chi phí

Model không còn bị buộc phát tín hiệu giao dịch cho mọi dự báo nhỏ. Quy tắc hiện
tại:

```text
BUY       nếu predicted_return > chi phí + biên an toàn
SELL      nếu predicted_return < -(chi phí + biên an toàn)
NO_TRADE  nếu predicted_return nằm trong vùng trung tính
```

Cấu hình mặc định:

```text
TRADING_ROUND_TRIP_COST_BPS=35
TRADING_SIGNAL_BUFFER_BPS=15
Tổng ngưỡng tín hiệu = 50 bps = 0,50%
```

Ví dụ:

| Return dự báo | Tín hiệu |
|---:|---|
| `+0,80%` | `BUY` |
| `+0,20%` | `NO_TRADE` |
| `-0,30%` | `NO_TRADE` |
| `-0,90%` | `SELL` |

Ngưỡng này mới là giả định cấu hình ban đầu, chưa phải ngưỡng đã được tối ưu và
chứng minh trên tập validation dài hạn.

### 5. Backtest khớp với pipeline SARIMAX production

`backtest_gap_model()` đã được chuyển từ AutoReg price-only sang walk-forward
SARIMAX. Mỗi target chỉ sử dụng dữ liệu có trước target:

```text
Dữ liệu đến t-1
      ↓
Fit scaler và SARIMAX
      ↓
Dự báo return tại t
      ↓
BUY / SELL / NO_TRADE
      ↓
Áp dụng chi phí
      ↓
Ghi nhận kết quả thực tế
```

Mỗi dòng backtest hiện có:

- `pred`: return dự báo.
- `actual`: return thực tế.
- `train_end`: phiên cuối model được nhìn thấy.
- `signal`: `BUY`, `SELL` hoặc `NO_TRADE`.
- `position`: vị thế được mô phỏng.
- `gross_return`: return trước chi phí.
- `net_return`: return sau chi phí.
- `correct_direction`: model có dự báo đúng hướng hay không.

### 6. Mặc định backtest long-only

Vì cổ phiếu cơ sở Việt Nam không nên được giả định có thể bán khống tự do,
backtest mặc định sử dụng:

- `BUY`: mở vị thế mua.
- `SELL`: thoát/né vị thế, không tự động mở short.
- `NO_TRADE`: không mở vị thế.

Chỉ khi chủ động gọi `allow_short=True`, tín hiệu `SELL` mới được mô phỏng thành
vị thế short.

### 7. Metric và baseline mới

Metric dự báo:

- RMSE.
- MAE.
- Directional accuracy.
- Số phiên kiểm thử.

Metric giao dịch sau chi phí:

- Coverage.
- Trade count.
- Win rate.
- Cumulative net return.
- Sharpe ratio.
- Maximum drawdown.
- Profit factor.

Baseline:

- Luôn dự báo tăng: kiểm tra model có vượt qua được tỷ lệ tăng tự nhiên của thị trường không.
- Dùng hướng của phiên trước: kiểm tra model có tốt hơn quy tắc momentum đơn giản không.
- Buy-and-hold: kiểm tra chiến lược giao dịch có hiệu quả hơn việc chỉ mua và giữ hay không.

Việc này tách hai câu hỏi khác nhau: model có đoán đúng hướng không, và tín hiệu
của model có tạo hiệu quả kinh tế sau chi phí không.

### 8. Nâng model schema

`MODEL_SCHEMA_VERSION` được nâng từ `3` lên `4`. Model schema cũ sẽ được coi là
stale và được retrain, vì model cũ chưa có target/timing rõ ràng và sử dụng cách
tạo forecast price-lag trước khi sửa.

### 9. Kết quả kiểm thử Pha 1

Bộ regression test kiểm tra:

- `BUY`, `SELL` và `NO_TRADE` tuân thủ cost threshold.
- Forecast `t+1` sử dụng return mới nhất của `t`.
- `train_end` luôn trước ngày target.
- Backtest có target definition và baseline.
- Giao diện hiển thị đúng “giá đóng cửa”.
- Giao diện hiển thị tín hiệu sau ngưỡng chi phí.

Kết quả:

```text
Ran 36 tests in 2.262s
OK
```

Backtest smoke FPT trên 5 phiên:

```text
Directional accuracy: 20.00%
Coverage: 0.00%
Trades: 0
Net return: 0.00%
Always-up accuracy: 40.00%
Previous-session accuracy: 20.00%
Buy-and-hold return: -6.12%
```

Đây chỉ là smoke test để xác nhận pipeline hoạt động, không phải kết quả đủ để
đánh giá chất lượng model. `Coverage=0%` cho thấy cost filter đã chọn
`NO_TRADE` cho các dự báo yếu thay vì ép hệ thống giao dịch. Cần backtest tối
thiểu 250–500 phiên trước khi kết luận về accuracy hoặc lợi nhuận.

#### Việc còn lại trước khi kết luận về hiệu quả

- [ ] Chạy đánh giá dài 250–500 phiên trên nhiều mã và nhiều market regime.
- [ ] Hiệu chỉnh chi phí/slippage bằng dữ liệu giao dịch thực tế.

### Pha 2 — Xây feature dataset point-in-time

1. Bổ sung OHLCV, technical, market, sector và regime.
2. Chuẩn hóa timestamp tin tức.
3. Lưu snapshot news theo thời điểm.
4. Kiểm tra missing data và corporate actions.
5. Cache dataset để backtest có thể tái lập.

### Pha 3 — Thêm model phân loại

1. Thêm logistic regression làm baseline.
2. Thêm XGBoost classifier.
3. Dùng time-series cross-validation.
4. Giới hạn hyperparameter search để giảm backtest overfitting.
5. Calibration xác suất, feature importance và ablation test.

### Pha 4 — Ensemble và signal policy

1. Kết hợp xác suất XGBoost với dự báo SARIMAX.
2. Chọn threshold trên validation.
3. Chỉ phát tín hiệu khi đủ độ tin cậy.
4. Trả ra xác suất, expected return, độ rủi ro và lý do chính.

### Pha 5 — Paper trading

1. Chạy forward/paper test tối thiểu 2–3 tháng.
2. Theo dõi drift, coverage, PnL sau phí và drawdown.
3. Không dùng tiền thật nếu kết quả ngoài mẫu không ổn định.

## 8. Deep learning có nên dùng ngay không?

Chưa nên ưu tiên LSTM hoặc Temporal Fusion Transformer với dataset hiện tại.
Deep learning chỉ nên được thử sau khi:

- Có dữ liệu nhiều năm.
- Huấn luyện pooled model trên nhiều mã.
- Có snapshot point-in-time sạch.
- Có baseline XGBoost đáng tin cậy.
- Có quy trình tuning không sử dụng test set.

Thứ tự model nên thử:

1. Logistic regression.
2. XGBoost.
3. Ensemble XGBoost + SARIMAX.
4. LightGBM hoặc CatBoost nếu cần so sánh.
5. LSTM/TFT khi dataset đã đủ lớn.

## 9. Tiêu chí chấp nhận

Một phiên bản mới chỉ nên được đưa vào paper trading khi:

- Không có look-ahead leakage trong kiểm thử.
- Vượt baseline trên nhiều fold thời gian và nhiều mã.
- Xác suất được calibration hợp lý.
- Lợi nhuận kỳ vọng vẫn dương sau toàn bộ chi phí.
- Drawdown nằm trong giới hạn đã định trước.
- Hiệu quả không phụ thuộc vào một giai đoạn hoặc một mã duy nhất.

Không có model nào đảm bảo accuracy hoặc lợi nhuận cố định. Mục tiêu của lộ
trình này là xây dựng một quy trình kiểm chứng đáng tin cậy trước khi coi đầu ra
của model là tín hiệu giao dịch.


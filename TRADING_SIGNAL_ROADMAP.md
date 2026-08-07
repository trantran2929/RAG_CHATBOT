# Lộ trình nâng cấp mô hình thành tín hiệu giao dịch

> **Trạng thái:** Pha 1 đã hoàn tất quy trình kỹ thuật nhưng không vượt validation;
> Pha 2 đã triển khai dataset và kiểm thử, còn chờ xác minh thủ công các phiên nghi ngờ
> corporate action; Pha 3–5 chưa triển khai.
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

#### 10. Chạy đánh giá dài 250–500 phiên trên nhiều mã và market regime

Mục tiêu là kiểm tra model có hoạt động ổn định trên nhiều cổ phiếu và nhiều
điều kiện thị trường hay chỉ tình cờ tốt ở một giai đoạn.

Danh sách mã khởi đầu đề xuất:

```text
FPT: công nghệ
VCB: ngân hàng
HPG: thép
PNJ: bán lẻ
SHS: chứng khoán
```

Chạy thử 250 phiên với grid SARIMAX nhỏ để kiểm tra thời gian và pipeline:

```powershell
docker compose exec -T -e SARIMAX_MAX_P=1 -e SARIMAX_MAX_Q=1 ingestion python -c "from modules.ML.backtest import backtest_gap_model,print_backtest_report; r=backtest_gap_model('SHS',test_days=250,use_exog=False); print_backtest_report(r)"
```

Thay `FPT` lần lượt bằng các mã còn lại. Chỉ tăng lên 500 phiên hoặc dùng
`p,q <= 2` sau khi lần chạy 250 phiên ổn định, vì SARIMAX được fit lại ở từng
bước walk-forward và có thể chạy lâu.

Tạm thời dùng `use_exog=False` cho backtest dài vì Qdrant chưa có snapshot tin
tức lịch sử point-in-time đầy đủ. Không nên coi backtest exogenous là đáng tin
cậy cho tới khi Pha 2 hoàn thành dữ liệu lịch sử này.

Với mỗi mã cần lưu:

- Directional accuracy.
- Always-up accuracy và previous-session accuracy.
- Coverage và trade count.
- Win rate sau chi phí.
- Cumulative net return.
- Sharpe ratio và maximum drawdown.
- Buy-and-hold return.

Phân loại regime bằng VNINDEX:

```text
Bull:
- VNINDEX trên MA200
- Return 60 phiên dương

Bear:
- VNINDEX dưới MA200
- Return 60 phiên âm

Sideways:
- Các trường hợp còn lại

High volatility:
- Volatility 20 phiên thuộc nhóm 25% cao nhất
```

Sau khi ghép regime theo ngày, tạo báo cáo:

| Regime | Số phiên | Accuracy | Coverage | Net return | Drawdown |
|---|---:|---:|---:|---:|---:|
| Bull | | | | | |
| Bear | | | | | |
| Sideways | | | | | |
| High volatility | | | | | |

Điều kiện đánh giá tối thiểu:

- Ít nhất 5 mã và 250 phiên mỗi mã.
- Mỗi regime nên có ít nhất 40–50 quan sát.
- Model vượt baseline trên đa số mã, không chỉ một mã.
- Net return sau phí dương.
- Hiệu quả không phụ thuộc vào một regime duy nhất.
- Maximum drawdown nằm trong giới hạn chấp nhận trước.

Backtest hiện chưa tự gắn market regime. Cần bổ sung bước lấy lịch sử VNINDEX,
tính MA200/return 60 phiên/volatility 20 phiên và ghép kết quả theo ngày trước
khi đánh dấu mục này hoàn thành.

#### 11. Hiệu chỉnh chi phí và slippage bằng dữ liệu thực tế

Các giá trị hiện tại:

```text
TRADING_ROUND_TRIP_COST_BPS=35
TRADING_SIGNAL_BUFFER_BPS=15
Tổng ngưỡng tín hiệu=50 bps
```

chỉ là giả định ban đầu, chưa được hiệu chỉnh bằng lịch sử khớp lệnh.

Thu thập file lịch sử giao dịch hoặc paper trading với các trường:

```text
symbol
signal_time
order_time
side
quantity
decision_price
average_fill_price
broker_fee
tax
```

Không lưu số tài khoản hoặc thông tin cá nhân vào dataset model.

Quy đổi đơn vị:

```text
1%    = 100 bps
0,1%  = 10 bps
0,15% = 15 bps
```

Với nhà đầu tư cá nhân, cần đưa thuế bán chứng khoán vào chi phí theo quy định
hiện hành. Phí môi giới phải lấy từ biểu phí hoặc sao kê thực tế của chính tài
khoản sử dụng, không dùng một con số chung cho mọi công ty chứng khoán.

Tính slippage từng lệnh:

```text
Slippage mua
= (average_fill_price - decision_price) / decision_price

Slippage bán
= (decision_price - average_fill_price) / decision_price
```

Tổng chi phí khứ hồi:

```text
total_cost_bps
= phí mua
+ phí bán
+ thuế bán
+ slippage mua
+ slippage bán
```

Nên chia kết quả theo nhóm thanh khoản vì cổ phiếu thanh khoản thấp thường có
slippage lớn hơn:

| Nhóm | Phí + thuế | Slippage | Tổng chi phí |
|---|---:|---:|---:|
| Thanh khoản cao | | | |
| Thanh khoản trung bình | | | |
| Thanh khoản thấp | | | |

Với mỗi nhóm, tính:

- Median: kịch bản optimistic.
- P75: kịch bản base/thận trọng.
- P90: stress test.

Ví dụ chạy lại với tổng chi phí 50 bps và buffer 15 bps:

```powershell
docker compose exec -T ingestion python -c "from modules.ML.backtest import backtest_gap_model,print_backtest_report; r=backtest_gap_model('FPT',test_days=250,use_exog=False,round_trip_cost_bps=50,signal_buffer_bps=15); print_backtest_report(r)"
```

Sau đó chạy lại bằng mức P75 và P90. Nếu chiến lược chỉ có lãi với median nhưng
lỗ ở P75, tín hiệu chưa đủ an toàn.

Chỉ đánh dấu mục này hoàn thành khi:

- Có đủ mẫu khớp lệnh thật hoặc paper trade cho từng nhóm thanh khoản.
- Phí, thuế và slippage được tách riêng, không ước lượng gộp tùy ý.
- Backtest được chạy lại với median, P75 và P90.
- Kết quả vẫn chấp nhận được ở kịch bản P75.
- Các biến môi trường chi phí được cập nhật theo số liệu đã đo.

#### 12. Quy trình kiểm tra sau backtest 250 phiên

Kết quả 250 phiên hiện tại cho thấy SARIMAX chưa vượt baseline ổn định và các
mã có giao dịch đều có net return âm. Trước khi tối ưu model hoặc chuyển sang
Pha 2, cần thực hiện lần lượt các bước dưới đây.

##### Bước 1 — Kiểm tra giá bất thường và corporate action

Pipeline hiện lấy OHLCV từ VNStock và đổi giá cổ phiếu sang VNĐ, nhưng chưa có
bước điều chỉnh riêng cho chia tách, cổ tức cổ phiếu, thưởng cổ phiếu hoặc quyền
mua.

Tìm các phiên biến động tuyệt đối trên 6,5%:

```powershell
docker compose exec -T ingestion python -c "from modules.api.stock_api import get_prices_df; df=get_prices_df('FPT',days=730); df['return_pct']=df['close'].pct_change()*100; print(df.loc[df['return_pct'].abs()>6.5,['close','volume','return_pct']].to_string())"
```

Thay `FPT` bằng `PNJ`, `VCB`, `HPG` hoặc `SHS`. Với cổ phiếu HOSE, các return
nằm sát `±7%` thường là phiên trần/sàn chứ chưa đủ để kết luận dữ liệu lỗi.

Một điểm cần kiểm tra corporate action khi:

- Giá nhảy một lần rất lớn rồi duy trì mặt bằng mới.
- Mức thay đổi không giống biên độ giao dịch thông thường.
- Có ngày giao dịch không hưởng quyền tương ứng.
- Giá không khớp với dữ liệu từ Sở hoặc nguồn đối chiếu khác.

Đối với PNJ, chuỗi nhiều phiên giảm sát sàn liên tiếp kèm volume lớn có thể là
biến động thực, không giống một lần điều chỉnh giá do chia tách. Tuy nhiên vẫn
cần đối chiếu ngày sự kiện doanh nghiệp trước khi kết luận.

##### Bước 2 — Kiểm tra buy-and-hold từ đầu đến cuối kỳ

```powershell
docker compose exec -T ingestion python -c "from modules.api.stock_api import get_close_series; s=get_close_series('FPT',days=730).tail(251); print('Start:',s.index[0],s.iloc[0]); print('End:',s.index[-1],s.iloc[-1]); print('Return:',s.iloc[-1]/s.iloc[0]-1)"
```

Thực hiện cho từng mã. Nếu mức giảm khớp với giá thị trường thực tế thì
buy-and-hold âm là hợp lệ. Nếu không khớp, dừng phân tích model và sửa dữ liệu
trước.

##### Bước 3 — Xuất chi tiết giao dịch ra CSV

Ví dụ với PNJ:

```powershell
docker compose exec -T -e SARIMAX_MAX_P=1 -e SARIMAX_MAX_Q=1 ingestion python -c "from modules.ML.backtest import backtest_gap_model; r=backtest_gap_model('PNJ',test_days=250,use_exog=False); t=r[r['position']!=0].sort_values('net_return'); t.to_csv('/app/models/PNJ_backtest_trades.csv'); print(t.to_string()); print('Saved:',len(t),'trades')"
```

File được lưu tại `models/PNJ_backtest_trades.csv`. Xuất tương tự cho các mã
còn lại.

Mỗi giao dịch cần kiểm tra:

- `date`: ngày target.
- `train_end`: phải trước `date`.
- `pred`: return dự báo.
- `actual`: return thực tế.
- `signal` và `position`.
- `gross_return`: kết quả trước chi phí.
- `net_return`: kết quả sau chi phí.

Xem các giao dịch lỗ lớn nhất trong PowerShell:

```powershell
Import-Csv .\models\PNJ_backtest_trades.csv |
    Sort-Object {[double]$_.net_return} |
    Select-Object -First 10 |
    Format-Table
```

Đặc biệt kiểm tra model có phát `BUY` ngay trước chuỗi giảm sàn hoặc khi thị
trường chuyển sang bear/high-volatility hay không.

##### Bước 4 — Phân biệt lỗi model với ảnh hưởng của chi phí

```powershell
docker compose exec -T ingestion python -c "import pandas as pd, numpy as np; df=pd.read_csv('/app/models/PNJ_backtest_trades.csv'); print('Gross:',np.exp(df['gross_return'].sum())-1); print('Net:',np.exp(df['net_return'].sum())-1); print('Cost impact:',np.exp(df['gross_return'].sum())-np.exp(df['net_return'].sum()))"
```

Diễn giải:

```text
Gross âm, net âm:
- Model chọn sai giao dịch; giảm phí không giải quyết được.

Gross dương, net âm:
- Tín hiệu có thông tin nhưng không đủ bù chi phí/slippage.

Gross dương, net dương:
- Tiếp tục kiểm tra ngoài mẫu và theo market regime.
```

Không nên giảm cost threshold chỉ để làm coverage tăng nếu gross return của các
giao dịch hiện tại vẫn âm.

##### Bước 5 — Kiểm tra SARIMAX có chỉ lặp lại momentum không

FPT và PNJ có directional accuracy bằng baseline dùng hướng phiên trước. Cần
chia kết quả thành:

```text
SARIMAX đồng thuận với momentum
SARIMAX mâu thuẫn với momentum
```

Với mỗi nhóm, báo cáo:

| Nhóm | Số phiên | Accuracy | Coverage | Net return |
|---|---:|---:|---:|---:|
| Đồng thuận | | | | |
| Mâu thuẫn | | | | |

Nếu hầu hết dự báo đồng thuận và hiệu quả bằng baseline, SARIMAX chưa tạo thêm
thông tin đáng kể ngoài price lag. Backtest hiện chưa xuất bảng này tự động nên
cần bổ sung báo cáo trước khi kết luận.

##### Bước 6 — Ghép market regime

Lấy lịch sử VNINDEX cùng khoảng ngày với backtest, tính MA200, return 60 phiên
và volatility 20 phiên, sau đó ghép nhãn regime theo ngày:

```text
Bull: VNINDEX > MA200 và return 60 phiên > 0
Bear: VNINDEX < MA200 và return 60 phiên < 0
Sideways: các trường hợp còn lại
High volatility: volatility 20 phiên thuộc top 25%
```

Kiểm tra riêng accuracy, coverage, trade count, net return và drawdown của từng
regime. Nếu các lệnh thua tập trung trong bear/high-volatility, có thể thử bộ
lọc `NO_TRADE` hoặc tăng threshold trong regime đó trên tập validation.

##### Bước 7 — Chia train, validation và test

Không được thử nhiều threshold trực tiếp trên 250 phiên vừa dùng để đánh giá.
Cần chia theo đúng thứ tự thời gian:

```text
Train:
- Fit model.

Validation:
- Chọn p/q, cost buffer, signal threshold và regime rule.

Test:
- Chạy một lần cuối để xác nhận.
```

Không dùng test cuối để lựa chọn cấu hình. Nếu liên tục sửa threshold cho tới
khi 250 phiên test có lãi, kết quả sẽ bị backtest overfitting.

##### Thứ tự thực hiện bắt buộc

1. Kiểm tra giá bất thường và corporate action.
2. Xác minh buy-and-hold từ giá đầu/cuối kỳ.
3. Xuất toàn bộ giao dịch của từng mã.
4. Kiểm tra gross return trước phí.
5. Tìm các giao dịch gây lỗ lớn nhất.
6. So sánh SARIMAX với momentum theo từng phiên.
7. Ghép market regime từ VNINDEX.
8. Chia lại train/validation/test.
9. Chỉ sau đó mới hiệu chỉnh threshold hoặc chuyển sang Pha 2.

Nếu phát hiện dữ liệu corporate action chưa sạch thì dừng ở bước 1–2, vì mọi
kết luận về accuracy, return và drawdown sau đó đều có thể sai.

#### 13. Hoàn thiện Pha 1 bằng risk filter và final test

Sau khi audit dữ liệu và giao dịch, bước tiếp theo là xây dựng bộ lọc rủi ro để
kiểm tra liệu các lệnh BUY sai trong bear/high-volatility có thể được loại bỏ
một cách hợp lệ hay không. Chưa thay đổi SARIMAX ở bước này.

##### Bước 1 — Khóa tập final test

Không tiếp tục chọn quy tắc trực tiếp trên 250 phiên đã xem kết quả. Với khoảng
500 phiên gần nhất, chia theo đúng thứ tự thời gian:

```text
Train:
- Các phiên đầu tiên.
- Dùng để fit SARIMAX tại mỗi bước.

Validation:
- 125 phiên tiếp theo.
- Dùng để chọn filter và threshold.

Final test:
- 125 phiên cuối.
- Chỉ chạy sau khi đã khóa cấu hình.
```

Không dùng random split. Sau khi xem final test, không được quay lại thay đổi
threshold dựa trên kết quả của chính tập này.

##### Bước 2 — Bổ sung feature kiểm soát rủi ro

Tại mỗi ngày target `t`, backtest cần lưu thêm:

```text
ret_1
ret_5
distance_ma20 = close / MA20 - 1
volatility_5
volatility_20
previous_shock = abs(return phiên trước)
vnindex_ret_5
vnindex_distance_ma200
market_regime
```

Tất cả feature của target `t` chỉ được tính bằng dữ liệu có đến hết `t-1`.
Rolling mean, volatility và percentile không được nhìn thấy target hoặc các
phiên tương lai.

##### Bước 3 — Xác định market regime

Quy tắc khởi đầu:

```text
BULL:
- VNINDEX > MA200
- Return 60 phiên > 0

BEAR:
- VNINDEX < MA200
- Return 60 phiên < 0

SIDEWAYS:
- Các trường hợp còn lại

HIGH_VOLATILITY:
- Volatility 20 phiên vượt percentile 75
  được tính từ dữ liệu quá khứ
```

Không tính percentile bằng toàn bộ final test. Ngưỡng volatility phải được fit
từ train/validation rồi giữ cố định khi chạy test.

##### Bước 4 — Tách tín hiệu gốc và tín hiệu sau lọc

Mỗi dòng backtest cần có:

```text
raw_signal
final_signal
filter_reason
```

Ví dụ:

```text
raw_signal   = BUY
final_signal = NO_TRADE
filter_reason = bear_regime|below_ma20|high_volatility
```

Giữ cả hai tín hiệu giúp đo chính xác bộ lọc đã chặn bao nhiêu lệnh thắng và bao
nhiêu lệnh thua.

##### Bước 5 — Thử risk filter trên validation

Quy tắc khởi đầu:

```text
Nếu raw_signal khác BUY:
    giữ nguyên

Nếu raw_signal là BUY:
    chuyển thành NO_TRADE khi:
    - market_regime == BEAR
    hoặc
    - close < MA20 và ret_5 < 0
    hoặc
    - previous_shock vượt ngưỡng
    hoặc
    - volatility thuộc nhóm cao
```

Chỉ thử một grid nhỏ:

```text
previous_shock: 5%, 6%, 6,5%
volatility percentile: 75, 90
cooldown sau shock: 1 hoặc 2 phiên
```

Không chọn `ret_5 < -5%` hoặc một ngưỡng khác chỉ vì nó chặn đúng hai lệnh PNJ
đã biết. Ngưỡng phải cải thiện validation trên nhiều mã.

##### Bước 6 — So sánh trước và sau filter

Tạo báo cáo:

| Metric | SARIMAX gốc | Sau risk filter |
|---|---:|---:|
| Directional accuracy | | |
| Coverage | | |
| Trade count | | |
| Win rate | | |
| Gross return | | |
| Net return | | |
| Sharpe | | |
| Maximum drawdown | | |

Thêm các thống kê:

- Số BUY bị chặn.
- PnL của các BUY bị chặn.
- PnL của các BUY được giữ lại.
- Số lệnh thắng bị chặn nhầm.
- Kết quả theo từng `filter_reason`.
- Kết quả theo từng mã và từng regime.

Risk filter tốt khi giảm tail loss và drawdown nhưng không xóa gần hết lệnh
thắng hoặc làm coverage về 0%.

##### Bước 7 — Đo phần thông tin khác momentum

Chia dự báo thành:

```text
SARIMAX đồng thuận với hướng phiên trước
SARIMAX mâu thuẫn với hướng phiên trước
```

Báo cáo:

| Nhóm | Số phiên | Accuracy | Coverage | Net return |
|---|---:|---:|---:|---:|
| Đồng thuận | | | | |
| Mâu thuẫn | | | | |

Nếu nhóm đồng thuận chỉ bằng momentum baseline và nhóm mâu thuẫn không có hiệu
quả, SARIMAX chưa tạo thêm thông tin đáng kể ngoài price lag.

##### Bước 8 — Chọn và khóa cấu hình

Chọn một cấu hình duy nhất trên validation theo thứ tự ưu tiên:

1. Maximum drawdown và tail loss giảm.
2. Net return sau phí cải thiện.
3. Kết quả không phụ thuộc vào một mã.
4. Coverage không về gần 0%.
5. Hoạt động hợp lý trên nhiều regime.
6. Accuracy không thấp hơn baseline một cách rõ rệt.

Không chọn cấu hình chỉ vì có accuracy cao nhất.

Sau khi chọn, lưu đầy đủ:

```text
validation date range
regime definition
shock threshold
volatility percentile
cooldown sessions
round-trip cost
signal buffer
SARIMAX p/q grid
```

##### Bước 9 — Chạy final test đúng một lần

Chạy cấu hình đã khóa trên 125 phiên cuối của:

```text
FPT
VCB
HPG
PNJ
SHS
```

Báo cáo riêng từng mã và một danh mục equal-weight. Nếu final test thất bại,
ghi nhận SARIMAX không đủ mạnh; không tối ưu tiếp bằng chính final test đó.

##### Bước 10 — Stress test chi phí

Với cấu hình tín hiệu đã khóa, chạy:

```text
Optimistic: chi phí median
Base: chi phí P75
Stress: chi phí P90
```

Đánh giá chính dùng P75. Kết quả chỉ có lãi ở median nhưng lỗ lớn ở P75 được xem
là chưa đủ ổn định.

##### Điều kiện kết thúc Pha 1

Có thể chuyển sang Pha 2 khi:

- [ ] Dữ liệu giá và corporate action đã được xác minh.
- [ ] Không có look-ahead leakage trong risk feature và regime.
- [ ] Đã tách train, validation và final test theo thời gian.
- [ ] Có `raw_signal`, `final_signal` và `filter_reason`.
- [ ] Đã so sánh SARIMAX với momentum theo từng phiên.
- [ ] Đã báo cáo theo mã và market regime.
- [ ] Đã chạy final test bằng cấu hình khóa trước.
- [ ] Đã stress test chi phí median/P75/P90.
- [ ] Đã kết luận rõ SARIMAX có hoặc không tạo alpha.

Pha 1 không bắt buộc SARIMAX phải có lãi. Nếu quy trình đánh giá đã đầy đủ nhưng
SARIMAX vẫn thua baseline, đó là kết quả hợp lệ để chuyển sang Pha 2 và thử
feature/model mới.

### Pha 2 — Xây feature dataset point-in-time

**Trạng thái: đã triển khai dataset; chưa thay đổi pipeline production và chưa huấn luyện
model Pha 3. Còn chờ audit thủ công các phiên sát biên độ.**

#### Kết luận đầu vào từ Pha 1

Validation walk-forward gồm 125 phiên cho mỗi mã `FPT`, `VCB`, `HPG`, `PNJ`, `SHS`.
Tổng cộng có 625 phiên validation. SARIMAX chỉ tạo 16 tín hiệu BUY, tương đương raw
coverage 2,56%. Baseline không dùng risk filter đạt:

| Mã | Coverage | Net return | Sharpe |
|---|---:|---:|---:|
| FPT | 0,0% | 0,00% | 0,000 |
| VCB | 4,0% | +7,84% | 0,995 |
| HPG | 2,4% | -7,73% | -2,350 |
| PNJ | 5,6% | -1,56% | -0,261 |
| SHS | 0,8% | -2,40% | -1,420 |

Không cấu hình risk filter nào đồng thời đạt `mean_coverage >= 2%` và
`mean_score > 0`. Validation đã trả về `VALIDATION_REJECTED`, không tạo
`phase1_locked_config.json`, và final test 125 phiên cuối vẫn được giữ kín.

Kết luận: quy trình Pha 1 hoạt động đúng, nhưng SARIMAX price-only chưa tạo tín hiệu có
lợi thế ổn định trên nhiều mã. Pha 2 không tiếp tục nới filter để làm đẹp kết quả mà xây
dataset point-in-time tốt hơn cho các model phân loại ở Pha 3.

#### Nguyên tắc bắt buộc của Pha 2

- Final test 125 phiên cuối không được dùng để thiết kế feature, chọn target hoặc sửa
  threshold.
- Mỗi hàng target tại phiên `t` chỉ được dùng dữ liệu đã biết chậm nhất tại cuối phiên
  `t-1`.
- Feature rolling phải `shift(1)` trước khi ghép với target.
- Không backfill dữ liệu thị trường hoặc tin tức từ tương lai về quá khứ.
- Dataset phải tái lập được từ cấu hình, mã nguồn và dữ liệu đầu vào đã ghi nhận.
- Pha 2 chỉ xây dataset và kiểm tra dữ liệu; chưa tuning XGBoost trên final test.

#### Bước 1 — Khóa đặc tả dataset và target

Tạo một cấu hình riêng cho dataset, dự kiến tại
`modules/ML/phase2_config.py`, chứa tối thiểu:

```text
symbols = FPT, VCB, HPG, PNJ, SHS
validation_days = 125
final_test_days = 125
round_trip_cost_bps = 35
signal_buffer_bps = 15
feature_version = 1
```

Target phân loại khởi đầu:

```text
future_return[t] = log(close[t] / close[t-1])
UP       nếu future_return[t] >  0,50%
DOWN     nếu future_return[t] < -0,50%
NO_TRADE trong các trường hợp còn lại
```

Ngưỡng 0,50% gồm 35 bps chi phí và 15 bps biên an toàn. Đây là đặc tả ban đầu;
nếu so sánh threshold thì chỉ được làm trên train/validation và phải dùng một grid nhỏ
được khai báo trước.

**Điều kiện hoàn thành:** có test chứng minh target tại `t` không được dùng làm feature
của chính hàng `t`.

#### Bước 2 — Xây feature OHLCV point-in-time tối thiểu

Tạo `modules/ML/phase2_features.py`. Phiên bản đầu chỉ dùng một bộ feature nhỏ:

```text
ret_1, ret_2, ret_5, ret_10, ret_20
range_1 = log(high / low)
intraday_return = log(close / open)
overnight_gap = log(open / close_previous)
volatility_5, volatility_10, volatility_20
distance_ma20, distance_ma50, distance_ma200
relative_volume_5, relative_volume_20
drawdown_20, drawdown_60
RSI_14, ATR_14
```

Tất cả feature dùng để dự báo target `t` phải được tính từ dữ liệu đến hết `t-1`.
Không thêm hàng trăm indicator ở phiên bản đầu.

**Điều kiện hoàn thành:** thay đổi OHLCV tại hoặc sau ngày target không làm thay đổi
feature của các target trước đó.

#### Bước 3 — Ghép feature thị trường và regime

Tái sử dụng logic leakage-safe hiện có trong `risk_features.py`, sau đó bổ sung:

```text
vnindex_ret_1, vnindex_ret_5, vnindex_ret_20, vnindex_ret_60
vnindex_volatility_5, vnindex_volatility_20
vnindex_distance_ma20, vnindex_distance_ma200
relative_strength_5, relative_strength_20
rolling_beta_60, rolling_correlation_60
market_regime
```

VNINDEX chỉ được forward-fill từ một quan sát đã xuất hiện. Tuyệt đối không backfill.
Ngưỡng high-volatility phải được fit từ phần train đứng trước validation.

**Điều kiện hoàn thành:** test tạo một quan sát VNINDEX tương lai và chứng minh nó không
thay đổi feature thị trường của ngày trước đó.

#### Bước 4 — Audit missing data và corporate action

Tạo báo cáo cho từng mã:

```text
first_date, last_date, row_count
duplicate_dates
missing OHLCV
non-positive price/volume
return tuyệt đối vượt 10%
chuỗi phiên sát trần/sàn
ngày cổ phiếu và VNINDEX không khớp
```

Không tự động xóa outlier sát biên độ vì đó có thể là biến động thật. Các điểm nghi ngờ
corporate action phải được ghi vào báo cáo và có quyết định giữ/sửa rõ ràng.

**Điều kiện hoàn thành:** cả 5 mã có báo cáo audit, không còn duplicate và không còn
giá không hợp lệ trước khi tạo dataset chính thức.

#### Bước 5 — Chuẩn hóa news snapshot point-in-time

Chưa dùng embedding trực tiếp trong bước đầu. Trước tiên chuẩn hóa mỗi bài viết:

```text
published_at
collected_at
symbol
source
sentiment_score
article_count_24h
source_count_24h
```

Với target `t`, chỉ dùng bài có `published_at` và `collected_at` trước cutoff của target.
Nếu không thể chứng minh timestamp point-in-time, feature news phải bị tắt thay vì điền
dữ liệu tương lai.

**Điều kiện hoàn thành:** thêm một bài báo sau cutoff không làm thay đổi feature news
của target trước cutoff.

#### Bước 6 — Tạo dataset builder và cache tái lập

Tạo `modules/ML/phase2_dataset.py` để:

1. Đọc OHLCV cổ phiếu và VNINDEX.
2. Tạo feature point-in-time.
3. Ghép news snapshot nếu hợp lệ.
4. Tạo target nhưng không đưa future return vào feature.
5. Gắn nhãn `train`, `validation`, `final_test` theo thời gian.
6. Lưu dataset và metadata vào `models/phase2/`.

Các file dự kiến:

```text
models/phase2/phase2_dataset.csv hoặc parquet
models/phase2/phase2_dataset_metadata.json
models/phase2/phase2_data_audit.csv
```

Metadata phải lưu feature version, symbols, date range, target threshold, chi phí,
feature columns và hash cấu hình.

**Điều kiện hoàn thành:** chạy builder hai lần với cùng đầu vào tạo cùng số hàng, cùng
schema và cùng hash nội dung.

#### Bước 7 — Kiểm thử leakage và schema

Thêm tối thiểu các nhóm test:

- Target alignment và shift một phiên.
- Không backfill VNINDEX/news.
- Rolling feature không nhìn target hiện tại.
- Train/validation/final test không chồng lấn.
- Final test không được truy cập trong quá trình fit hoặc chọn feature.
- Dataset không có `NaN`/`inf` ngoài warm-up đã khai báo.
- Thứ tự cột và kiểu dữ liệu ổn định.
- Cache có thể tái lập.

Chạy test bằng:

```powershell
docker compose exec -T ingestion python -W error::FutureWarning -m unittest discover -s /app/tests -v
```

**Điều kiện hoàn thành:** toàn bộ test hiện tại và test Pha 2 đều `OK`.

#### Bước 8 — Báo cáo dataset trước khi sang Pha 3

Báo cáo riêng cho train và validation, không đọc thống kê target của final test:

```text
số hàng theo mã
tỷ lệ UP / DOWN / NO_TRADE
tỷ lệ missing theo feature
phân phối và outlier chính
tương quan feature quá cao
số hàng theo market regime
date range của từng split
```

Không loại feature chỉ vì tương quan với target trên một mã. Quyết định feature phải dựa
trên tính hợp lệ point-in-time, độ ổn định và kết quả validation sau này.

#### Điều kiện kết thúc Pha 2

- [x] Có cấu hình dataset và target được version hóa.
- [x] Có feature OHLCV/technical point-in-time tối thiểu.
- [x] Có feature VNINDEX và regime không leakage.
- [ ] Có audit tự động cho 5 mã; còn xác minh thủ công corporate action của các phiên sát biên độ.
- [x] News được tắt có chủ đích vì chưa có snapshot point-in-time được chứng minh.
- [x] Có dataset cache và metadata tái lập.
- [x] Có test leakage, split, schema và reproducibility.
- [x] Có báo cáo train/validation mà chưa đọc final-test target.
- [x] Final test vẫn chưa được mở; 625 target final đã được mask.

#### Nhật ký hành động Pha 2

| Ngày | Hành động | Kết quả | Trạng thái |
|---|---|---|---|
| 2026-08-02 | Ghi nhận Pha 1 bị validation từ chối; không tạo locked config và không mở final test | SARIMAX không đạt `coverage >= 2%` đồng thời `score > 0` | Hoàn thành |
| 2026-08-02 | Lập kế hoạch Pha 2, đặc tả thứ tự triển khai và tiêu chí hoàn thành | Chưa sửa code Pha 2 | Hoàn thành |
| 2026-08-02 | Khóa cấu hình dataset/target trong `phase2_config.py` | Threshold 0,50%; validation/final mỗi tập 125 phiên; tối thiểu 250 hàng train mỗi mã | Hoàn thành |
| 2026-08-02 | Xây feature OHLCV point-in-time trong `phase2_features.py` | 20 feature giá, volume, volatility, MA, drawdown, RSI và ATR; tất cả shift một phiên | Hoàn thành |
| 2026-08-02 | Ghép feature VNINDEX và market regime | 13 feature thị trường; chỉ forward-fill dữ liệu đã quan sát, không backfill | Hoàn thành |
| 2026-08-02 | Tạo audit, split và cache trong `phase2_dataset.py` | Train/validation/final tách theo thời gian; CSV, metadata, audit và report được lưu | Hoàn thành |
| 2026-08-02 | Khóa news feature | Tắt có chủ đích (`news_enabled=false`) cho tới khi có timestamp snapshot hợp lệ | Hoàn thành |
| 2026-08-02 | Đánh giá feature ngành | Chưa có chuỗi chỉ số ngành point-in-time đáng tin cậy trong data provider hiện tại; hoãn thay vì ghép dữ liệu không kiểm chứng | Hoãn có chủ đích |
| 2026-08-02 | Thêm CLI `run_phase2_dataset.py` | Builder mặc định lấy 1100 ngày lịch và không xuất final target | Hoàn thành |
| 2026-08-02 | Thêm 7 test Pha 2 và chạy toàn bộ regression test | `Ran 68 tests in 3.500s — OK` | Hoàn thành |
| 2026-08-02 | Chạy builder dữ liệu thật cho FPT, VCB, HPG, PNJ, SHS | 2.985 hàng: train 1.735, validation 625, final 625; final target bị mask | Hoàn thành |
| 2026-08-02 | Ghi nhận hash dataset Pha 2 | `bd1883efbc5981386d6eb4ce11bb0ab25e19092ad754efd5e589060d286c0c0a` | Hoàn thành |
| 2026-08-02 | Audit dữ liệu thật | Không duplicate/missing/giá không hợp lệ/return vượt 10%; PNJ có 26 và SHS có 43 phiên sát biên độ cần xem thủ công | Đang chờ |
| 2026-08-02 | Audit thủ công sơ bộ 26 phiên PNJ sát ±7% | Các phiên nằm trong biên độ HOSE; nhiều chuỗi trần/sàn liên tiếp và volume tăng mạnh, đặc biệt tháng 7/2026; không có bước nhảy đơn lẻ vượt 10% | `PROVISIONAL_KEEP` |
| 2026-08-02 | Audit thủ công sơ bộ 43 phiên SHS sát biên độ | Các phiên nằm trong biên độ giao dịch, phần lớn có volume khoảng 14–72 triệu và nhiều chuỗi đảo chiều trần/sàn; không có dấu hiệu điều chỉnh giá đơn lẻ vượt biên độ | `PROVISIONAL_KEEP` |
| 2026-08-02 | Kiểm tra khả năng corporate action PNJ/SHS | Chưa có tài liệu công bố doanh nghiệp hoặc Sở được gắn với từng ngày nghi ngờ; chưa được đổi trạng thái thành `KEEP` chính thức | Chờ nguồn xác nhận |
| 2026-08-02 | Tạo mẫu audit thủ công | `PNJ_manual_audit.csv` có 26 dòng và `SHS_manual_audit.csv` có 43 dòng; đủ cột nguồn, sự kiện, quyết định và ghi chú | Hoàn thành |
| 2026-08-02 | Đối chiếu 4 thông báo PNJ năm 2026 do người dùng cung cấp | Quyền dự ĐHĐCĐ có ex-date 17/03 không điều chỉnh giá; đợt nhận cổ phiếu chốt 24/04 không trùng phiên 08/04 và chuỗi giá 23–24/04 không có bước nhảy; lấy ý kiến chốt 17/08 nằm sau dataset 31/07 | Xác minh một phần |
| 2026-08-02 | Đối chiếu công bố PNJ ngày 28/01 và 09/02/2026 | Thay đổi nhân sự, bảo lãnh công ty con và thay đổi sở hữu nước ngoài không phải quyền kinh tế làm điều chỉnh giá; hai phiên 29/01 và 02/02 chuyển sang `KEEP` | Hoàn thành |
| 2026-08-02 | Hoàn tất các dòng PNJ năm 2026 trong file audit | 17/17 phiên năm 2026 có `event_type`, nguồn, quyết định `KEEP` và ghi chú; không có bước điều chỉnh giá kỹ thuật bị dùng làm biến động thị trường | Hoàn thành |
| 2026-08-02 | Đối chiếu thông báo PNJ năm 2025 | Ex-date cổ tức là 10/02 và 26/09, không trùng các phiên audit; ESOP hoàn tất 23/09 và giao dịch người liên quan 09/10–07/11 là tác động cung cầu, không điều chỉnh giá | Hoàn thành |
| 2026-08-02 | Cập nhật `PNJ_manual_audit.csv` cho năm 2025 | 8/8 phiên năm 2025 chuyển sang `KEEP`, có nguồn và ghi chú | Hoàn thành |
| 2026-08-02 | Đối chiếu ESOP PNJ năm 2024 | Thông báo ngày 05/09, nhận tiền 12–23/09, tỷ lệ 1%; phiên tăng trần 19/08 xảy ra trước sự kiện và không phải điều chỉnh giá | Hoàn thành |
| 2026-08-02 | Đóng audit thủ công PNJ | 26/26 phiên `KEEP`; không còn `PROVISIONAL_KEEP`, `REVIEW_REQUIRED` hoặc trường nguồn/ghi chú bị trống | Hoàn thành |
| 2026-08-02 | Kiểm tra chéo audit PNJ bằng trang Thông báo chính thức | Phát hiện có CBTT Quy chế ESOP và danh sách người tham gia đúng ngày 19/08/2024; sửa `event_type` từ `NONE` thành `OTHER` và hiệu đính lý do. Quyết định vẫn `KEEP` vì đây không phải điều chỉnh giá cơ học | Hoàn thành |
| 2026-08-02 | Kiểm tra lại riêng thông báo PNJ năm 2025 | Đối chiếu 8 phiên audit năm 2025; phát hiện phiên 11/04 trùng CBTT bổ sung phương án mua lại cổ phiếu để giảm vốn điều lệ, sửa `event_type` từ `NONE` thành `OTHER`. Không phiên nào trùng ngày GDKHQ 10/02 hoặc 26/09; toàn bộ quyết định vẫn `KEEP` | Hoàn thành |
| 2026-08-02 | Kiểm tra lại riêng thông báo PNJ năm 2026 | Đối chiếu 17 phiên audit năm 2026 với nguồn PNJ; sửa 7 phiên `NONE` thành `OTHER` do trùng hoặc ngay sau CBTT: 10/03, 03/07, 06/07, 07/07, 10/07, 21/07, 22/07. Không phiên nào là điều chỉnh giá cơ học; toàn bộ vẫn `KEEP` | Hoàn thành |
| 2026-08-02 | Bắt đầu audit thủ công SHS từ nguồn chính thức | Đã rà 591 công bố bất thường và 140 công bố định kỳ; xác nhận ngày ĐKCC 25/04/2025 cho cổ tức tiền 10%, cổ tức cổ phiếu 5% và cổ phiếu thưởng 5%, không trùng 43 phiên audit. Phiên 20/07/2026 trùng công bố BCTC quý II và được đề xuất `OTHER/KEEP`; tệp audit đang bị khóa nên chưa lưu được quyết định | Đang thực hiện |
| 2026-08-02 | Ghi nhận tài liệu SHS do người dùng cung cấp vào mẫu audit | Đã cập nhật 11 phiên trùng ngày công bố chính thức thành `OTHER/KEEP`: 27/09/2023, 02/11/2023, 03/04/2025, 11/04/2025, 30/07/2025, 01/08/2025, 29/08/2025, 03/11/2025, 23/03/2026, 08/04/2026 và 20/07/2026. Đây là thông tin pháp lý, quản trị, ESOP, cổ đông lớn hoặc báo cáo tài chính; không có bằng chứng là điều chỉnh giá cơ học. Các phiên còn lại vẫn giữ `PROVISIONAL_KEEP` chờ đối chiếu. | Hoàn thành một phần |
| 2026-08-02 | Hoàn tất audit thủ công 43 phiên SHS từ toàn bộ tài liệu người dùng giao | Đã khóa cả 43 phiên thành `KEEP`: 27 phiên `OTHER` có công bố đúng ngày hoặc ngay trước phiên và 16 phiên `NONE` không có sự kiện liên quan. Khi tin xuất hiện sau ngày giao dịch, không dùng ngược thời gian. Sự kiện quyền ĐKCC 25/04/2025 không trùng mẫu; chưa có bằng chứng phiên nào là điều chỉnh giá cơ học nên không loại dữ liệu. | Hoàn thành |
| 2026-08-02 | Chuẩn bị kiểm tra audit bằng chương trình và tái tạo dataset Pha 2 | Quy trình gồm kiểm tra schema/số dòng/ô trống/trùng khóa của PNJ và SHS, chạy unit test Pha 2, rebuild vào thư mục mới với final target tiếp tục bị che, rồi kiểm tra audit OHLCV, split, khóa trùng và metadata. Hai audit đều 100% `KEEP`, vì vậy không có giá nào bị xóa hoặc sửa trước khi rebuild. | Chờ người dùng chạy và gửi kết quả |
| 2026-08-02 | Xác nhận metadata dataset Pha 2 sau audit | Bản `models/phase2_audited` có 2.985 dòng, SHA256 `bd1883efbc5981386d6eb4ce11bb0ab25e19092ad754efd5e589060d286c0c0a`, `final_targets_masked=true` và `include_final_targets=false`. Hash trùng bản trước audit, phù hợp với việc toàn bộ quyết định PNJ/SHS đều là `KEEP`. | Đạt |

### Pha 3 — Thêm model phân loại

**Trạng thái: đang triển khai; final test tiếp tục bị khóa.**

#### Bước 1 — Khóa cấu hình và hàng rào chống leakage

- Tạo `phase3_config.py` với số fold, seed và hyperparameter giới hạn.
- Tạo loader chỉ trả về `train` và `validation`.
- Từ chối chạy nếu phát hiện target của `final_test` không còn bị che.

#### Bước 2 — Logistic regression baseline

- Chuẩn hóa feature số, one-hot `market_regime`, cân bằng class.
- Dùng expanding-window cross-validation theo ngày.
- Báo cáo accuracy, balanced accuracy, macro F1 và log loss.

#### Bước 3 — XGBoost classifier

- Dùng một cấu hình nhỏ đã khóa trước, chưa mở hyperparameter search rộng.
- Chỉ chạy sau khi rebuild container có dependency `xgboost`.
- Kết quả chỉ được đánh giá trên validation; chưa đọc final test.

#### Bước 4 — Calibration xác suất

- Fit calibrator từ xác suất out-of-fold của riêng tập train.
- So sánh log loss, multiclass Brier score và ECE trước/sau calibration trên validation.
- Không giữ calibration nếu các metric xác suất xấu đi rõ rệt.

#### Bước 5 — Feature importance

- Logistic dùng trị tuyệt đối trung bình của hệ số sau chuẩn hóa.
- XGBoost dùng importance nội tại của cây, model chỉ fit trên train.
- Importance dùng để giải thích và thiết kế ablation, không chứng minh quan hệ nhân quả.

#### Bước 6 — Ablation test giới hạn

- Khóa trước 5 nhóm: stock returns, stock risk, stock trend, volume và market.
- Mỗi lượt chỉ bỏ một nhóm và so với full model trên cùng validation.
- Không thử mọi tổ hợp feature và không đọc final test.

#### Bước 7 — So sánh và khóa full/compact XGBoost

- Compact được định nghĩa trước là full model bỏ đúng nhóm `stock_returns`.
- So sánh cặp full/compact trên cùng 4 expanding folds và cùng validation.
- Chỉ chọn compact nếu mean CV balanced accuracy và macro F1 không thấp hơn full,
  validation balanced accuracy giảm không quá 0,5 điểm % và log loss tăng không quá 0,01.
- Nếu bất kỳ điều kiện nào thất bại thì khóa full model; không thay đổi tiêu chí sau khi xem kết quả.
- Lưu quyết định tại `models/phase3/phase3_locked_config.json`; không đọc final test.

1. Thêm logistic regression làm baseline.
2. Thêm XGBoost classifier.
3. Dùng time-series cross-validation.
4. Giới hạn hyperparameter search để giảm backtest overfitting.
5. Calibration xác suất, feature importance và ablation test.

#### Nhật ký triển khai Pha 3

| Ngày | Hành động | Kết quả | Trạng thái |
|---|---|---|---|
| 2026-08-02 | Thêm cấu hình, loader và test chống leakage | Loader chỉ dùng `train/validation`, từ chối target final bị lộ; expanding folds luôn train trước validate; 3/3 test đạt | Hoàn thành |
| 2026-08-02 | Chạy logistic regression baseline | 4 fold có balanced accuracy lần lượt 36,67%, 38,99%, 40,20%, 32,68%; validation accuracy 39,52%, balanced accuracy 35,70%, macro F1 0,317; majority accuracy 26,24%; final test không được đọc | Hoàn thành |
| 2026-08-02 | Tích hợp XGBoost classifier có cấu hình giới hạn | Đã thêm code và dependency `xgboost>=2.1,<3`; hai lần rebuild toàn image vượt giới hạn 2 và 5 phút nên đã cài wheel XGBoost 2.1.4 trực tiếp để kiểm chứng. Dependency vẫn được khóa trong `requirements.txt` cho lần build chính thức | Hoàn thành có lưu ý môi trường |
| 2026-08-02 | Chạy kiểm thử hồi quy sau khi thêm Pha 3 | Toàn bộ 71 test đạt `OK`, gồm 3 test mới về che final target, loader train/validation và expanding-window không dùng tương lai | Hoàn thành |
| 2026-08-02 | Chạy XGBoost validation với cấu hình giới hạn | 4 fold có balanced accuracy 41,17%, 37,92%, 33,17%, 36,24%; validation accuracy 36,48%, balanced accuracy 36,10%, macro F1 0,360; majority accuracy 26,24%; final test không được đọc | Hoàn thành |
| 2026-08-02 | Calibration xác suất XGBoost bằng train OOF | Fit calibrator trên 985 dự báo out-of-fold của train. Trên validation: log loss giảm 1,1260→1,0959, multiclass Brier giảm 0,6847→0,6651, ECE giảm 9,60%→2,27%; accuracy argmax giảm nhẹ 36,48%→35,52%. Giữ calibration cho chính sách xác suất, không coi là cải thiện accuracy | Đạt về xác suất |
| 2026-08-02 | Feature importance XGBoost | Top feature gồm ATR14, volatility 5/10, distance MA20 và nhóm biến động/xu hướng VNINDEX. Importance chỉ mang ý nghĩa giải thích model, không phải quan hệ nhân quả | Hoàn thành |
| 2026-08-02 | Ablation XGBoost giới hạn theo 5 nhóm | Bỏ stock returns cho balanced accuracy tốt nhất 37,85% (+1,75 điểm %) và macro F1 0,380; bỏ market làm giảm balanced accuracy xuống 34,60%, cho thấy market features có ích. Chưa khóa compact model vì mới là một validation window | Hoàn thành |
| 2026-08-02 | Calibration và ablation Logistic đối chứng | Calibration giảm log loss 1,2234→1,0874 và ECE 15,01%→2,84%. Ablation cho kết quả không đồng nhất với XGBoost; nhóm market giúp accuracy nhưng bỏ market tăng macro F1, xác nhận cần thận trọng và chưa loại feature chỉ từ một cửa sổ | Hoàn thành |
| 2026-08-02 | Chạy toàn bộ kiểm thử sau khi hoàn thiện phân tích Pha 3 | 73 test đạt `OK`; final test vẫn bị che và tất cả báo cáo calibration/importance/ablation ghi `Final test read: NO` | Hoàn thành |
| 2026-08-02 | So sánh ghép cặp full và compact XGBoost trên từng fold | Compact bỏ nhóm `stock_returns` đạt mean CV balanced accuracy 38,87% so với full 37,13%; mean macro F1 0,3570 so với 0,3437; mean log loss 1,1428 so với 1,1513. Compact tốt hơn ở fold 2–3, thua fold 1 và gần ngang fold 4 | Hoàn thành |
| 2026-08-02 | Khóa lựa chọn feature set Pha 3 | Validation compact đạt balanced accuracy 37,85%, macro F1 0,3796, log loss 1,1245; full lần lượt 36,10%, 0,3597, 1,1260. Tất cả tiêu chí định trước đều đạt nên khóa `compact`, bỏ đúng nhóm `stock_returns`, còn 26 feature. File `models/phase3/phase3_locked_config.json`; SHA256 dataset khớp bản audited và `final_test_read=false` | Đã khóa |
| 2026-08-02 | Kiểm thử hồi quy sau khi khóa compact | Toàn bộ 75 test đạt `OK`, gồm 2 test mới bảo đảm compact chỉ được chọn khi mọi tiêu chí định trước đều đạt | Hoàn thành |

### Pha 4 — Ensemble và signal policy

**Trạng thái: bắt đầu triển khai; final test tiếp tục bị khóa.**

#### Bước 1 — Khóa calibrator compact XGBoost

- Kiểm tra SHA256 dataset khớp `phase3_locked_config.json`.
- Fit XGBoost compact trên train và calibrator multinomial sigmoid từ 4 fold OOF train.
- Lưu model + calibrator bằng `joblib` kèm SHA256; không fit calibrator trên validation.

#### Bước 2 — Chuyển xác suất thành BUY/SELL/NO_TRADE

- Grid khóa trước: threshold 0,36/0,40/0,44/0,48 và margin 0/0,03/0,06.
- BUY khi xác suất UP vượt threshold và thắng xác suất còn lại đủ margin; SELL tương tự với DOWN.
- SELL mang nghĩa dự báo giảm/thoát vị thế, không giả định bán khống khi tính PnL.
- Policy phải có ít nhất 30 hành động, tối thiểu 10 BUY và 10 SELL.
- Chọn bằng cận dưới Wilson của directional accuracy; PnL sau phí chỉ dùng để báo cáo phụ.
- Lưu khóa tại `models/phase4/phase4_locked_policy.json`; không đọc final test.
- Khóa để tái lập không đồng nghĩa được dùng giao dịch: chỉ `ACCEPTED` khi directional
  accuracy và BUY win rate sau phí đều đạt tối thiểu 50%; nếu không, ghi `REJECTED`
  và `approved_for_signal_use=false`.

#### Nhật ký triển khai Pha 4

| Ngày | Hành động | Kết quả | Trạng thái |
|---|---|---|---|
| 2026-08-02 | Khóa calibrator compact XGBoost | Fit calibrator multinomial sigmoid từ 985 dự báo OOF train, lưu cùng base model 26 feature; dataset hash khớp Pha 3 và final test không được đọc | Hoàn thành |
| 2026-08-02 | Chạy grid policy validation đã định trước | Chỉ threshold 0,36/margin 0 đủ quy mô: 212 hành động (91 BUY, 121 SELL), coverage 33,92%; các threshold cao hơn gần như không phát tín hiệu | Hoàn thành |
| 2026-08-02 | Áp dụng cổng chấp nhận signal policy | Policy có directional accuracy 40,57% và BUY win rate sau phí 47,25%, đều dưới 50%; dù long-only report +2,65%, Sharpe 0,933 và max drawdown -2,88%, khóa được ghi `REJECTED`, `approved_for_signal_use=false`. Không mở final test | Bị từ chối |
| 2026-08-02 | Kiểm thử hồi quy sau bước đầu Pha 4 | Toàn bộ 77 test đạt `OK`, gồm test threshold/margin, Wilson score và toàn bộ hàng rào Pha 2–3 | Hoàn thành |

#### Bước 3 — Candidate ensemble compact XGBoost + SARIMAX

- Giữ nguyên `phase4_locked_policy.json` làm baseline `REJECTED`; không ghi đè.
- Ghép raw `pred` SARIMAX với xác suất compact đã calibration theo đúng `symbol/date`;
  từ chối nếu không phủ đủ 625 dòng hoặc actual return không khớp.
- Candidate khóa trước: blend trọng số SARIMAX 15%/25%/35% và agreement gate.
- Tái sử dụng threshold 0,36/margin 0 của baseline, không mở threshold grid mới.
- Candidate chỉ qua cổng khi đủ mẫu, directional accuracy và BUY win rate sau phí
  đều >=50%, đồng thời Wilson lower bound vượt baseline.
- Nếu không candidate nào đạt, chỉ lưu report `REJECTED`; không tạo ensemble lock.

| 2026-08-03 | Kiểm tra candidate ensemble compact XGBoost + SARIMAX | Đã ghép đủ 625 dòng validation, kiểm tra khớp symbol/date, actual return, dataset hash và calibrator hash. Blend 15%/25%/35% có directional accuracy 37,33–37,70%, BUY win rate 39,29–39,58% và long-only return khoảng -10,84% đến -10,92%. Agreement gate có 102 hành động, accuracy 38,24%, BUY win rate 40%, long-only +0,61%. Tất cả dưới cổng 50% và Wilson không vượt baseline | `REJECTED` |
| 2026-08-03 | Quyết định khóa ensemble | Giữ nguyên baseline policy Pha 4 ở trạng thái `REJECTED`; chỉ lưu `phase4_ensemble_candidate_report.json` và grid. Không tạo `phase4_ensemble_candidate_lock.json`, không mở final test | Không khóa candidate |
| 2026-08-03 | Kiểm thử hồi quy sau ensemble candidate | Toàn bộ 79 test đạt `OK`; xác nhận `phase4_ensemble_candidate_lock.json` không tồn tại khi validation bị từ chối | Hoàn thành |

#### Bước 4 — Chẩn đoán cố định policy bị từ chối

- Chỉ đọc 625 dòng validation và signal baseline Pha 4; không thay threshold/model/feature.
- Báo cáo riêng theo symbol, `market_regime` và phía BUY/SELL.
- Gắn cờ nhóm có ít nhất 10 hành động và accuracy thấp hơn overall >=5 điểm %.
- Báo cáo incorrect actions, BUY win rate sau phí và tỷ lệ bỏ lỡ biến động khi NO_TRADE.
- Kết quả chỉ dùng để quyết định hướng cải thiện dữ liệu/feature hoặc chờ validation mới;
  không dùng để tuning lại cùng cửa sổ và không mở final test.

| 2026-08-03 | Chẩn đoán policy theo từng mã | HPG bị gắn cờ: 38 hành động, accuracy 34,21%, thấp hơn overall 6,36 điểm % và có 25 lỗi. SHS có coverage 86,4%, tạo 108/212 hành động và 66 lỗi — nguồn lỗi tuyệt đối lớn nhất; BUY mean return sau phí của SHS -0,68%. FPT/PNJ/VCB có BUY win rate trên 50% nhưng mẫu nhỏ | Hoàn thành |
| 2026-08-03 | Chẩn đoán theo market regime | BULL accuracy 40,63%, SIDEWAYS 40,00%; không nhóm nào thấp hơn overall đủ 5 điểm %. Validation không có BEAR nên chưa thể kết luận hiệu quả trong bear regime | Hoàn thành |
| 2026-08-03 | Chẩn đoán riêng BUY/SELL | BUY accuracy 43,96%, win rate sau phí 47,25%; SELL accuracy 38,02% và tạo 75 lỗi. SELL yếu hơn BUY nhưng cả hai đều dưới cổng chấp nhận | Hoàn thành |
| 2026-08-03 | Quyết định sau chẩn đoán | Không chỉnh threshold theo mã/regime trên cùng validation. Ưu tiên nghiên cứu bằng train OOF: symbol calibration/imbalance, feature thanh khoản–biên độ cho SHS/HPG và policy SELL; sau đó đánh giá trên validation window mới. Final test tiếp tục khóa | Chờ cải thiện dữ liệu/model |
| 2026-08-03 | Kiểm thử hồi quy sau chẩn đoán cố định | Toàn bộ 81 test đạt `OK`, gồm 2 test mới bảo đảm accuracy chỉ tính trên hành động và không gắn cờ nhóm mẫu nhỏ | Hoàn thành |

#### Bước 5 — Chẩn đoán compact XGBoost trên train OOF

- Tạo dự báo trên 4 expanding folds; mọi `train_max_date` phải nhỏ hơn ngày OOF.
- Dùng xác suất thô và argmax cố định: UP→BUY, DOWN→SELL, NO_TRADE→NO_TRADE.
- Không dùng calibrator hiện tại vì calibrator được fit từ chính toàn bộ OOF; dùng lại để
  chẩn đoán cùng OOF sẽ gây leakage.
- Báo cáo theo symbol, BUY/SELL và fold; gắn cờ symbol có >=20 hành động và accuracy
  thấp hơn overall >=5 điểm %.
- Không thay policy, không tuning threshold và không đọc validation/final target ngoài
  các nhãn train OOF được phép dùng cho chẩn đoán.

| 2026-08-03 | Tạo 985 dự báo compact XGBoost train OOF | 4 expanding folds, mọi train cutoff đứng trước ngày dự báo. Dùng raw argmax, không dùng calibrator, không thay policy và không đọc final test | Hoàn thành |
| 2026-08-03 | Chẩn đoán OOF theo symbol | VCB bị gắn cờ: 65 hành động, accuracy 18,46%, thấp hơn overall 15,70 điểm % và 53 lỗi. SHS phát hành động trên 98,48% dòng, accuracy 34,54% và 127 lỗi, xác nhận overtrade nhất quán. HPG accuracy 31,25%; PNJ/FPT tốt hơn tương đối nhưng vẫn thấp | Hoàn thành |
| 2026-08-03 | Chẩn đoán OOF BUY/SELL | BUY accuracy 34,18%, SELL 34,15%; gần như ngang nhau. BUY win rate sau phí 37,82% và mean return sau phí -0,39%, nên train OOF chưa có edge giao dịch | Hoàn thành |
| 2026-08-03 | Chẩn đoán OOF theo fold | Fold 1/2/3 accuracy 32,54%/39,81%/39,21%; fold 4 giảm còn 25,68%, có 171 SELL và 136 lỗi. Kết quả cho thấy drift/không ổn định theo thời gian, không chỉ lỗi một phía signal | Hoàn thành |
| 2026-08-03 | Quyết định sau OOF diagnostics | Không áp threshold riêng cho VCB/SHS/HPG và không tắt SELL dựa trên validation cũ. Ưu tiên đặc tả mới trên train CV: thêm symbol identity one-hot, chuẩn hóa feature theo symbol, feature thanh khoản/biên độ và kiểm tra drift; chỉ đánh giá trên validation window mới | Chờ phiên bản dataset/model mới |
| 2026-08-03 | Kiểm thử hồi quy sau train OOF diagnostics | Toàn bộ 82 test đạt `OK`; có test cutoff thời gian và hàng rào Pha 2–4 vẫn đạt | Hoàn thành |

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


# CRC theo người trên điểm SSL-MD của EpaDB — khoá 08/11/2026

**Thăm dò.** Điểm held của v3 đã được xem khi báo cáo pAUC. Mục đích là đo xem với 10 người, CRC theo người cho ngưỡng và recall như thế nào. Không dùng study này để chọn arm, epoch hay nhãn.

## Vì sao không dùng split CRC trong fold
Mỗi fold chỉ có n = 2 người calibration. Bound CRC là (ΣLᵢ + 1)/(n + 1) ≤ α. Với n = 2, vế trái luôn ≥ 1/3 > α ∈ {0,10; 0,15; 0,20}, nên tập khả thi rỗng: không phát cảnh báo nào. Đây là kết quả giải tích, được báo như một phát hiện, không cần chạy.

## Thiết kế cross-fit leave-one-speaker-out (LOSO)
- **Điểm:** `mixed60` của v3 (ensemble 5 seeds) trên 10.013 known positions. Mỗi người có điểm out-of-fold từ đúng một model fold, model đó không fit trên người này. priority = sigmoid(logit).
- Với từng người s, calibrate một ngưỡng toàn cục trên 9 người còn lại bằng `public_risk.calibrate_crc`: loss theo người = FP / số position acceptable; bound (ΣLᵢ + 1)/(9 + 1) ≤ α; cảnh báo khi priority > t. Sau đó áp ngưỡng cho s.
- **α:** 0,10 (chính), 0,15, 0,20.
- **So sánh:**
  - (a) ngưỡng empirical không hiệu chỉnh: macro loss trên 9 người ≤ α;
  - (b) ngưỡng per-phone ActCost của v3 (đã có);
  - (c) cùng quy trình CRC trên `uniform60` và `prior`.
- **Báo cáo theo α:**
  - FPR có điều kiện trung bình trên 10 người, so với α;
  - số người có FPR > α;
  - recall trung bình theo người;
  - tỷ lệ position bị cảnh báo;
  - ngưỡng của từng lần LOSO;
  - CI bootstrap theo người (5.000 lần, seed 3191) cho FPR trung bình và recall.

## Giới hạn khai báo trước
- Điểm out-of-fold đến từ 5 model khác nhau, nên tính trao đổi được (exchangeability) chỉ xấp xỉ. Đảm bảo CRC là đảm bảo biên (marginal) trên người mới, xấp xỉ kiểu cross-conformal; không phải cho từng người.
- 10 người cho ước lượng rất nhiễu.
- Nhãn là phiên bản EpaDB đã phát hành; hàng đợi A03 chưa giải quyết.

# L2-ARCTIC: CRC theo người với head SSL-MD huấn luyện ngoài L2-ARCTIC — contract 08/11/2026

Phương án A, người dùng chọn ngày 08/11/2026. Contract này được viết **trước** khi chạy bất kỳ model SSL-MD nào trên L2-ARCTIC. Phần protocol sẽ khoá bằng hash ngay sau khi v4 chạy xong (để biết arm theo quy tắc đã khoá) và trước khi chạy encoder.

## Dữ liệu và nhãn
- L2-ARCTIC v5.0, phần có annotation thủ công: 3.599 câu, 24 người, 6 L1 (Arabic, Hindi, Korean, Mandarin, Spanish, Vietnamese), mỗi L1 có 4 người. Dùng manifest đã có (`data/derived/public_corpora/l2arctic_manifest.jsonl`), không tải dữ liệu mới.
- **Position** là các interval có canonical phone theo thứ tự: `acceptable`, `substitution` và `deletion`. Interval `insertion` không phải canonical position; chỉ đếm và báo, không chấm điểm.
- **Nhãn:** `acceptable` = 0. `substitution` và `deletion` = 1, kể cả khi phone quan sát là `err` (258 trường hợp, vẫn là lỗi). Phát hiện lỗi (detection), không chẩn đoán (diagnosis).
- **Loại cả câu**, khai báo trước:
  - 10 câu có interval `unparsed` hoặc `unscorable`, vì không xác định được chuỗi canonical;
  - câu chứa canonical `AX` (1 trường hợp), vì head 39 âm vị không có AX. Không map AX sang AH.
- Ranh giới thủ công **không** được dùng. Ranh giới position lấy từ alignment tự động `posterior_align` trên posterior của chính COML HuBERT với chuỗi canonical, giống pipeline EpaDB.

## Model (cố định, không huấn luyện trên L2-ARCTIC)
- Frozen COML HuBERT, cùng revision, cùng tiền xử lý: mono PCM16/32768, `resample_poly` 44,1 → 16 kHz, batch 1, CPU 4 luồng.
- **SSL-MD:** trung bình logit của 25 checkpoint `conv` của v4 (5 fold × 5 seed), lấy theo arm do quy tắc đã khoá của v4 chọn (`gate_B_baseline_by_locked_rule`). Mỗi checkpoint chỉ thấy 6 người EpaDB; 25 checkpoint cùng nhau phủ 10 người. Không dùng bất kỳ nhãn L2-ARCTIC nào để huấn luyện, chọn epoch hay chọn arm.
- Điểm position là trung bình logit của canonical phone trên khoảng frame được align (`pool_logits`). priority = sigmoid.
- **So sánh:**
  - **GOP posterior:** priority = 1 − trung bình posterior của canonical phone trên cùng khoảng frame, lấy từ classifier gốc của COML.
  - **Phone prior:** priority = sigmoid của trung bình hệ số prior trên 5 fold v2. Đây là baseline chỉ dùng loại âm vị.

## Phân tích
**P1 — LOSO CRC (chính).** Với từng người s trong 24: calibrate một ngưỡng toàn cục trên 23 người còn lại bằng `public_risk.calibrate_crc` (loss theo người = FP / số position acceptable; bound (ΣL+1)/24 ≤ α; cảnh báo khi priority > t), rồi áp lên s.
- α ∈ {0,05; **0,10**; 0,15}.
- Báo cáo: FPR trung bình theo người, số người có FPR > α, recall trung bình theo người, tỷ lệ position bị cảnh báo, precision gộp.
- So sánh với ngưỡng empirical (macro loss trên 23 người ≤ α, không hiệu chỉnh).

**Endpoint chính** (paired speaker bootstrap, 24 người, 5.000 lần, seed 3191; Holm trên 2 contrast):
- **U1:** recall SSL-MD − recall GOP, ở CRC α = 0,10.
- **U2:** recall SSL-MD − recall prior, ở CRC α = 0,10.

**Hợp lệ (mô tả):** FPR trung bình của SSL-MD ở α = 0,10 so với 0,10, kèm CI, và số người vượt α. Lặp lại với GOP và prior.

**P2 — Đổi L1 (độ bền, không phải bảo đảm).** Với từng L1: calibrate CRC trên 20 người thuộc 5 L1 kia, áp lên 4 người của L1 bị để riêng. Báo FPR và recall theo L1 so với α. Không có bảo đảm vì L1 bị để riêng không trao đổi được với tập calibration.

**P3 — Xếp hạng.** Speaker-macro standardized pAUC (max FPR 0,1) và AUROC theo người, của SSL-MD, GOP và prior; contrast SSL-MD − GOP có CI (mô tả). Báo thêm theo L1.

**P4 — Bối cảnh tài liệu (mô tả, không so sánh trực tiếp).**
- Trên 6 người test thường dùng trong các bài (NJS, TLV, TNI, TXHC, YKWK, ZHAA), báo precision, recall và F1 gộp ở hai ngưỡng:
  - (a) ngưỡng CRC α = 0,10 calibrate trên 18 người còn lại;
  - (b) ngưỡng tối đa F1 chọn trên 18 người còn lại.
- Ghi rõ các khác biệt: head huấn luyện trên EpaDB, chỉ phát hiện lỗi, bỏ insertion, alignment tự động.

**Nhạy cảm về exposure.** 6 người pilot (ABA, ASI, BWC, EBVS, HJK, HQTV) đã chạy HuBERT thô trên 72 câu ngày 03/11 (không SSL-MD). Lặp lại P1 chỉ trên 18 người test chưa exposure; tập calibration vẫn gồm 23 người.

## Giới hạn khai báo trước
- Bảo đảm CRC là biên (marginal) trên một người mới trao đổi được với 23 người calibration, không phải cho từng người. LOSO là 24 lần áp dụng đảm bảo đó, không độc lập.
- 4 người mỗi L1 cho P2 rất nhiễu.
- Head được huấn luyện trên EpaDB (L1 tiếng Tây Ban Nha, nhãn và quy ước khác). Hiệu năng có thể giảm vì domain shift; CRC vẫn hợp lệ dưới giả định trao đổi được giữa người L2-ARCTIC.
- Nhãn L2-ARCTIC chỉ do một annotator mỗi câu.
- Không có kết quả nào ở đây là fresh confirmation theo nghĩa cohort mới hoàn toàn chưa từng chạm.

## Tài nguyên
Chỉ lưu điểm theo position và receipt theo câu, không lưu đặc trưng 12 layer. RAM gate 2 GiB trước mỗi câu; khi gate dừng thì chạy tiếp từ câu đã xong. Không chạy song song với v4.

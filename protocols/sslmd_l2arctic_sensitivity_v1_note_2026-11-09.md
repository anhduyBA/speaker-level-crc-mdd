# Phân tích độ nhạy cho bài ICCIES (L2-ARCTIC) — ghi chú khai báo trước, 09/11/2026

Ghi chú này được viết **trước** khi chạy, để trả lời các nhận xét phản biện số 1, 3 và 5.

**Phạm vi.** Chỉ dùng lại các điểm đã khóa của `sslmd_l2arctic_crc_v1` (kiểm hash theo `scores_locked.json`), với SSL-MD và GOP. Không huấn luyện lại. Không đổi endpoint chính U1/U2. Mọi kết quả dưới đây là **phân tích độ nhạy**, ghi nhãn như vậy trong bài.

**Kiểm tra trước khi phân tích.** Bộ hiệu chỉnh vector hóa mới phải tái tạo đúng các ngưỡng LOSO của CRC và empirical đã lưu trong `thresholds.json` ở cả ba mức α (chênh lệch ≤ 1e-12), cũng như FPR và recall trung bình trong `summary.json`. Nếu không khớp thì dừng.

## S1. Ngưỡng gộp theo âm vị (đối chứng cho nhận xét 1)
- Với mỗi người được để riêng: chọn trên 23 người còn lại t nhỏ nhất sao cho FPR gộp mọi vị trí đúng ≤ α, không phân biệt người nói.
- Áp ngưỡng đó cho người được để riêng.
- Ở α ∈ {0,05; 0,10; 0,15}, báo cáo: FPR trung bình theo người, số người có FPR > α, FPR lớn nhất, recall trung bình.
- So sánh với CRC theo người và ngưỡng empirical theo người đã có.
- Hiệu chỉnh hữu hạn ở mức vị trí, kiểu (k+1)/(N+1) với N khoảng 10^5, gần như không đổi kết quả, nên không báo riêng.

## S2. Số người dùng để hiệu chỉnh (nhận xét 3)
- n ∈ {5, 9, 14, 19, 23}, α = 0,10.
- Với mỗi người được để riêng: lấy ngẫu nhiên 50 tập con gồm n người từ 23 người còn lại (seed 4242). Riêng n = 23 chỉ có 1 tập.
- Tính ngưỡng CRC và ngưỡng empirical rồi áp cho người được để riêng.
- Báo cáo trên mọi cặp (người, tập con):
  - FPR trung bình;
  - xác suất FPR > α;
  - recall trung bình;
  - tỷ lệ ngưỡng t = 1 (không phát cảnh báo nào).
- Đây là mô phỏng lấy mẫu con, không phải dữ liệu mới.

## S3. Kiểm định hoán vị cho U1 và U2 (nhận xét 5)
- Dùng đúng chênh lệch recall theo người của endpoint chính (24 người, CRC α = 0,10).
- Kiểm định hoán vị dấu ghép cặp chính xác: xét đủ 2^24 tổ hợp dấu, p hai phía = tỷ lệ tổ hợp có |trung bình| ≥ |trung bình quan sát|.
- Hiệu chỉnh Holm trên U1 và U2.
- Phép kiểm này bổ sung cho bootstrap đã khóa, không thay thế nó.

## Ghi chú về bootstrap đã khóa
Bootstrap đã khóa lấy mẫu lại 24 người được để riêng, với ngưỡng LOSO của từng người giữ nguyên. Ngưỡng **không** được hiệu chỉnh lại trong mỗi lần lấy mẫu, nên khoảng tin cậy không bao gồm bất định do bước chọn ngưỡng. Điều này phải được ghi rõ trong bài.

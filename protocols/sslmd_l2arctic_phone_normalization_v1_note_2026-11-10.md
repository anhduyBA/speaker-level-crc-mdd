# Chuẩn hóa thứ hạng theo âm vị trước CRC (thăm dò, post-hoc) — ghi chú viết trước khi chạy, 10/11/2026

## Lý do
Phản biện yêu cầu kiểm tra liệu việc báo nhầm của SSL-MD dồn vào một âm vị (AH chiếm 52,7% số báo nhầm) có giảm được bằng cách chuẩn hóa điểm theo từng âm vị hay không.

Phân tích này được đề xuất **sau khi** đã xem Bảng 8, nên là thăm dò post-hoc. Kết quả không thay endpoint chính và không dùng để chọn mô hình cho bài.

## Phép biến đổi
Với mỗi âm vị chuẩn c, điểm ưu tiên p của một vị trí được thay bằng thứ hạng chuẩn hóa:
- r = (số vị trí của âm vị c có điểm ≤ p) / (tổng số vị trí của âm vị c);
- tập so sánh là mọi vị trí của âm vị c trên cả 24 người, **không dùng nhãn**.

Hàm biến đổi đối xứng theo người nói, vì nó phụ thuộc vào tập không nhãn của tất cả người và không phân biệt người calibration hay người kiểm tra. Do đó tính hoán vị giữa các người nói vẫn được giữ, và bảo đảm CRC áp dụng như trước, theo lập luận chuẩn của conformal transductive.

Hàm biến đổi cố định cho cả 24 lượt LOSO. Không có tham số nào được chỉnh theo kết quả.

## Phân tích
- Dùng LOSO CRC và ngưỡng empirical như bản khóa (cùng bộ hiệu chỉnh đã kiểm tra trong `sslmd_l2arctic_sensitivity_v1`), α ∈ {0,05; 0,10; 0,15}.
- Áp dụng cho SSL-MD và GOP, mỗi điểm ở hai dạng: điểm gốc và điểm đã chuẩn hóa theo âm vị.
- Báo cáo:
  - FPR trung bình, số người có FPR > α, recall trung bình;
  - tỷ lệ báo nhầm rơi vào AH, số âm vị chiếm 50% số báo nhầm, số âm vị có ít nhất một cảnh báo;
  - recall của T, D, L, DH trên toàn bộ người nói và riêng người nói tiếng Việt.
- Khoảng tin cậy cho chênh lệch recall (đã chuẩn hóa trừ gốc) ở α = 0,10: bootstrap ghép cặp theo người, 5.000 lần, seed 3191, ngưỡng giữ cố định.

## Giới hạn khai báo
- Đây là phân tích post-hoc.
- Chuẩn hóa theo thứ hạng làm tỷ lệ cảnh báo của các âm vị gần nhau hơn. Nó có thể làm giảm recall ở những âm vị thực sự có nhiều lỗi.
- Không có kiểm định nào được dùng làm kết luận chính.

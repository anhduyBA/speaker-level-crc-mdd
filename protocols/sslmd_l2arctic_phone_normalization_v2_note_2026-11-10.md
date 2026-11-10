# Chuẩn hóa thứ hạng theo âm vị: đối chứng quy nạp với tham chiếu EpaDB cố định — ghi chú viết trước khi chạy, 10/11/2026

## Lý do
Vòng phản biện thứ ba chỉ ra rằng v1 (`sslmd_l2arctic_phone_normalization_v1`) là **transductive**: bảng thứ hạng được dựng từ các vị trí không nhãn của cả 24 người L2-ARCTIC, trong đó có người đang được để riêng. Phản biện yêu cầu một đối chứng dùng **bảng ánh xạ cố định**, dựng trước trên một tập tham chiếu độc lập, rồi áp như nhau cho người calibration và người kiểm tra. Phân tích vẫn là thăm dò post-hoc.

## Phép biến đổi quy nạp
- **Tập tham chiếu:** mọi vị trí chuẩn của 634 câu EpaDB development (10 người), không dùng nhãn. Không dùng dữ liệu L2-ARCTIC nào.
- **Điểm tham chiếu:** điểm SSL-MD của chính bộ 25 checkpoint `conv` đã khóa (arm mixed), lấy trung bình logit theo khoảng frame đã căn chỉnh. Tính từ đặc trưng 12 layer đã cache ở v1/v2 và khoảng frame trong `frame_mapping.json`. Bỏ các vị trí có 0 frame.
- **Ánh xạ:** với mỗi âm vị c, r(p) = tỷ lệ vị trí tham chiếu của c có logit ≤ logit của p. Hàm này cố định trước khi chạm vào L2-ARCTIC. Áp như nhau cho mọi người L2-ARCTIC, và với một người học mới cũng áp đúng hàm này mà không cần dữ liệu của họ.
- **Hệ quả:** vì hàm cố định và không phụ thuộc L2-ARCTIC, bảo đảm CRC giữ nguyên dưới giả định exchangeability như với điểm gốc.

## Lưu ý đã biết trước
- Các head đã được huấn luyện trên chính các người EpaDB này, nên phân bố điểm tham chiếu là in-sample: lệch so với điểm out-of-sample.
- Điều này không ảnh hưởng tính hợp lệ của CRC trên L2-ARCTIC, chỉ ảnh hưởng chất lượng ánh xạ.
- Âm vị có ít hơn 20 vị trí tham chiếu dùng bảng thứ hạng gộp mọi âm vị. Báo số âm vị rơi vào trường hợp này.

## Báo cáo
Chỉ cho SSL-MD, LOSO CRC và empirical ở α ∈ {0,05; 0,10; 0,15}. So sánh ba dạng điểm: gốc, transductive (v1), quy nạp (v2).
- **Các chỉ số như v1:**
  - FPR trung bình, số người > α, recall;
  - tỷ lệ báo nhầm ở AH, số âm vị chiếm 50% báo nhầm;
  - recall T/D/L/DH, cho toàn bộ và riêng người Việt.
- **Thêm (vấn đề 2 của phản biện):** với 10 âm vị có nhiều lỗi nhất, báo:
  - FPR theo âm vị (báo nhầm / số vị trí đúng);
  - precision theo âm vị (cảnh báo đúng / tổng cảnh báo);
  - recall theo âm vị;
  ở dạng gốc và dạng quy nạp, CRC α = 0,10.
- **Chênh lệch recall (quy nạp trừ gốc):** bootstrap theo người, 5.000 lần, seed 3191, ngưỡng giữ cố định.
- **Kiểm tra trước khi chạy:** điểm gốc phải tái tạo đúng các ngưỡng đã khóa.

# Phân tích theo âm vị trên L2-ARCTIC (mô tả) — ghi chú khai báo trước, 09/11/2026

Ghi chú này được viết **trước** khi chạy phân tích. Phân tích chỉ dùng lại các file đã khóa:
- điểm của `sslmd_l2arctic_crc_v1`, được kiểm hash theo `scores_locked.json`;
- các ngưỡng LOSO CRC đã lưu trong `thresholds.json`.

Không huấn luyện lại mô hình, không đổi ngưỡng, không thêm endpoint hay kiểm định mới. Mọi kết quả chỉ mang tính **mô tả**, phục vụ phần thảo luận của bài ICCIES.

## Thiết lập
- **Điểm:** SSL-MD và GOP. Prior không đưa vào phân tích vì mọi vị trí cùng một âm vị có cùng điểm.
- **Ngưỡng:** CRC LOSO ở α = 0,10. Mỗi người được giữ lại dùng đúng ngưỡng của mình trong `thresholds.json`.
- **Kiểm tra trước khi dùng:** tính lại FPR và recall trung bình theo người, phải khớp `summary.json` sai số không quá 1e-9. Nếu không khớp thì dừng.

## Bảng sẽ báo cáo
1. **Mức tập trung của báo nhầm.** Với mỗi điểm: tổng số báo nhầm, và số âm vị ít nhất chiếm 50% số báo nhầm.
2. **Âm vị gây nhiều báo nhầm nhất**, 10 âm vị đầu theo tỷ lệ báo nhầm của SSL-MD. Với mỗi âm vị: số vị trí đúng (acceptable), FPR theo âm vị và tỷ lệ trong tổng báo nhầm, cho cả SSL-MD và GOP.
3. **Recall theo âm vị trên toàn bộ 24 người**, cho các âm vị có ít nhất 200 lỗi, xếp theo số lỗi. Báo recall SSL-MD, recall GOP và hiệu số.
4. **Người nói tiếng Việt** (HQTV, PNV, THV, TLV). Dùng cùng ngưỡng LOSO, lấy 10 âm vị có nhiều lỗi nhất (tối thiểu 50 lỗi). Báo số lỗi, recall SSL-MD, recall GOP và FPR của từng điểm.

## Giới hạn khai báo
- Các số được gộp theo vị trí, không phải trung bình theo người; người có nhiều vị trí sẽ nặng ký hơn.
- Không có khoảng tin cậy và không kiểm định. Không được viết "SSL-MD tốt hơn hoặc kém hơn ở âm X" như một kết luận thống kê.
- Nhóm tiếng Việt chỉ có 4 người, nhãn do một annotator gán và căn chỉnh là tự động.
- Kết quả có thể gợi ý giả thuyết cho bước tiếp theo (huấn luyện trong L2-ARCTIC), không dùng để chọn mô hình.

## Bổ sung sau khi đã xem bảng 1–4 (post-hoc, thăm dò)
Bảng 1–4 cho thấy cảnh báo của SSL-MD tập trung vào vài âm vị: riêng AH đã chiếm 53% số báo nhầm, và đó là các âm có tỷ lệ lỗi cao trong EpaDB. Phép đo dưới đây được thêm **sau khi** đã xem các bảng, nên chỉ mang tính thăm dò:
- Tính tương quan Spearman, qua các âm vị, giữa tỷ lệ lỗi của từng âm vị trong 10 người EpaDB (đếm good/bad từ `sslmd_convergence_v4/summary.json`, mixedConv) và tỷ lệ cảnh báo của âm vị đó trên L2-ARCTIC (mọi vị trí), tính riêng cho SSL-MD và GOP.
- Để so sánh, tính thêm tương quan giữa tỷ lệ lỗi thật của từng âm vị trên L2-ARCTIC và tỷ lệ cảnh báo.
- Chỉ dùng các âm vị có ít nhất 50 vị trí ở cả hai corpus. Không có kiểm định nào được dùng làm kết luận.

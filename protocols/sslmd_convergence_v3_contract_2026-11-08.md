# SSL-MD v3: hội tụ và đối chứng trộn đều layer — 08/11/2026

Khoá trước mọi fit của v3. Trả lời hai câu hỏi còn mở ở [SSL-MD v2](../06_experiment_results/sslmd_training_v2/findings_2026-11-08.md): 45/50 fit chọn epoch 20 (trần), và chưa có đối chứng trộn đều 12 layer. Đây vẫn là retrospective development trên 10 người đã exposure, không phải fresh validation.

## Giữ nguyên từ v2
Cùng 634 feature caches đã seal của v2 (đọc theo hash, không trích xuất lại), 10.013 known positions, 453 unknown giữ riêng; cùng 5 folds 6 fit/2 cal/2 evaluation, 5 seeds, AdamW lr 1e-4 weight decay 0,01, batch 1 với gradient accumulation 4, frame-BCE cân bằng theo người fit, khởi tạo theo `torch.manual_seed(seed)`, thứ tự utterance `default_rng(seed+10000*fold)`. Cùng chọn epoch (pAUC cal lớn nhất, epoch sớm nhất khi hoà), cùng ngưỡng per-phone ActCost trên cal, cùng ensemble trung bình logit 5 seeds, cùng bootstrap 5.000 lần seed 3191 theo 10 người. Phone prior dùng lại 5 fit tất định của v2 theo hash.

## Thay đổi đã khai báo
1. **Arm mới `uniform`:** đặc trưng là trung bình cộng 12 layer (trọng số cố định 1/12), cùng linear head 768→39, cùng số tham số với `last` (29.991). Không có tham số trộn.
2. **Trần epoch 60** cho cả ba arm `last`, `mixed`, `uniform`. Không dừng sớm, không đổi trần theo kết quả. Mỗi fit lưu hai checkpoint: tốt nhất trong epoch 1–20 (`cap20`) và tốt nhất trong epoch 1–60 (`cap60`).
3. Tổng 75 fit neural (3 arm × 5 fold × 5 seed). Ngân sách wall-clock 16.200 giây; hết ngân sách thì dừng, giữ các fit đã xong, báo thiếu, không rút gọn arm/seed/fold.

## Kiểm tra tái lập
Vì epoch 1–20 dùng cùng khởi tạo, cùng thứ tự và cùng thuật toán tất định, `last`/`mixed` ở `cap20` phải tái lập v2. Báo độ lệch tối đa của lịch sử pAUC cal và epoch được chọn so với v2. Lệch được báo, không dùng để chọn kết quả.

## Endpoint khai báo trước
Đơn vị: speaker-macro standardized pAUC (max FPR 0,1) trên cùng 10.013 held positions; CI bootstrap ghép cặp theo người, mô tả, chưa hiệu chỉnh đa so sánh.

- **E1 (trọng số học được):** `mixed60 − uniform60`. Chỉ nói learned mixing có ích khi CI nằm hoàn toàn trên 0.
- **E2 (dùng nhiều layer):** `uniform60 − last60`.
- **E3 (hội tụ):** với từng arm, `cap60 − cap20`; và tỷ lệ fit chọn epoch ≥ 55. Nếu tỷ lệ này lớn, báo là chưa hội tụ trong 60 epoch, không kéo dài thêm trong study này.
- Phụ: `mixed60 − last60`, `mixed60 − prior`, conditional FPR, recall, ActCost ở ngưỡng empirical (không phải CRC).

Mọi arm và mọi kết quả âm đều được báo. Không dùng calibration20/final20 cũ, không sửa nhãn, không chọn arm/epoch theo held values.

## Tài nguyên
Như v2: 4 luồng, tất định, kiểm RAM khả dụng ≥ 2 GiB trước mỗi nhóm utterance, không đóng ứng dụng của người dùng.

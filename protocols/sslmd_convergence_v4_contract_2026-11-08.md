# SSL-MD v4: huấn luyện đến hội tụ theo patience — 08/11/2026

Khoá trước mọi fit của v4. v3 cho thấy 70/75 fit chọn epoch ≥ 55 ở trần 60, nghĩa là chưa hội tụ. Contract v3 không cho kéo dài trong chính study đó, nên v4 là một study mới. Đây vẫn là retrospective development trên 10 người EpaDB, không phải fresh validation.

## Giữ nguyên từ v3/v2
- Cùng 634 feature caches theo hash, 10.013 known positions, 453 unknown giữ riêng.
- Cùng 5 folds 6/2/2 và 5 seeds.
- Cùng thuật toán huấn luyện: AdamW lr 1e-4 wd 0,01, batch 1 tích luỹ 4, frame-BCE cân bằng theo người, `torch.manual_seed(seed)`, thứ tự `default_rng(seed+10000*fold)`.
- Cùng ngưỡng per-phone ActCost trên calibration, cùng ensemble 5 seeds, cùng bootstrap 5.000 lần seed 3191.

## Thay đổi khai báo
1. **Arm:** chỉ `mixed` và `uniform`. Arm `last` đã thua uniform ở v3 (E2) nên không chạy lại. Tổng 50 fit.
2. **Dừng theo patience:** trần 200 epoch. Dừng khi 20 epoch liên tiếp không có pAUC calibration **lớn hơn hẳn** giá trị tốt nhất. Chọn epoch tốt nhất (sớm nhất khi hoà). Fit chạm trần 200 mà chưa dừng được ghi là "chưa hội tụ trong 200".
3. Mỗi fit lưu thêm checkpoint tốt nhất trong epoch 1–60 để đối chiếu với v3.
4. Ngân sách wall-clock 28.800 giây. Hết ngân sách thì dừng, giữ các fit đã xong, báo thiếu; không rút gọn arm, seed hay fold. RAM gate 2 GiB giữ nguyên; khi gate dừng thì chạy tiếp từ fit đã xong (kiểm hash).

## Kiểm tra tái lập
Lịch sử pAUC calibration epoch 1–60 phải trùng v3, vì cùng khởi tạo, cùng thứ tự và cùng thuật toán tất định. Báo độ lệch tối đa và số fit lệch epoch được chọn ở cap60.

## Endpoint khai báo trước
Đơn vị: speaker-macro standardized pAUC (max FPR 0,1) trên 10.013 held positions. CI bootstrap ghép cặp theo người. p hai phía từ bootstrap. **Hiệu chỉnh Holm** trên họ 3 contrast suy luận E1, E3a, E3b.

- **E1 (chính):** `mixedConv − uniformConv`. Learned mixing chỉ được coi là có ích khi E1 dương và có ý nghĩa sau Holm.
- **E3a, E3b:** `mixedConv − mixed60` và `uniformConv − uniform60`, tức lợi ích thêm khi vượt 60 epoch. mixed60/uniform60 lấy từ checkpoint cap60 của chính v4 (trùng v3 nếu tái lập đạt).
- **E2 (mô tả hội tụ):** theo arm, gồm số fit dừng bằng patience, số fit chạm trần 200, median epoch được chọn và median epoch dừng.
- Phụ, không suy luận: `mixedConv − prior`, conditional FPR, recall, ActCost ở ngưỡng empirical (không phải CRC).

Baseline acoustic mạnh dùng cho gate B sẽ là arm tốt hơn ở hội tụ. Nếu E1 không có ý nghĩa sau Holm thì chọn `uniform`, vì nó đơn giản hơn và ít tham số hơn. Quy tắc này được khoá trước khi xem kết quả.

Mọi kết quả âm đều được báo. Không chọn arm hay epoch theo held values, không sửa nhãn.

## Tài nguyên
4 luồng, tất định, kiểm RAM khả dụng ≥ 2 GiB trước mỗi nhóm utterance, không đóng ứng dụng của người dùng.

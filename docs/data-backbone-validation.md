# Kiểm chứng data/backbone

Ngày thực hiện: 01/11/2026. Nhánh: `feat/data-backbone`. Code chưa commit khi chạy; manifest của từng run lưu Git status và SHA-256 của source để truy vết đúng trạng thái.

## Môi trường

Windows, Python 3.12.10; NumPy 2.2.6, Pillow 11.2.1, datasets 3.6.0, PyYAML 6.0.3. Bộ test và smoke CPU đã chạy với PyTorch 2.7.1+cpu / torchvision 0.22.1+cpu. Môi trường cuối đã chuyển sang PyTorch 2.7.1+cu128 / torchvision 0.22.1+cu128, CUDA runtime 12.8; GPU NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB. `torch.cuda.is_available()` trả True; `pip check` không có dependency lỗi. Bản `pip freeze` đầy đủ lưu tại `outputs/backbone-smoke-cuda/environment.txt`.

## Test tự động

Lệnh: `.venv\Scripts\python -m unittest discover -s tests -v`.

**9 test đạt** trên CPU, và chạy lại vẫn đạt sau khi cài bản PyTorch CUDA (unit test dùng CPU; GPU được kiểm tra bằng smoke thật bên dưới). Bao phủ:

- Class order MT19937 seed 1993, lịch `[50,10,10,10,10,10]`, mapping nhãn và từ chối lịch/task/label không hợp lệ.
- Train chỉ lớp mới, test chỉ lớp đã thấy, giới hạn mẫu theo từng lớp, từ chối seen-train và dữ liệu thiếu mẫu/lớp.
- Feature `[N,512]` không âm/hữu hạn; evaluation transform xác định; batch cuối không bị bỏ.
- Cosine head, backward cập nhật trọng số, lưu/nạp checkpoint cho output tương đương; từ chối ghi đè checkpoint.
- Đóng băng cả tham số và BatchNorm dù caller gọi `train()`; extraction không tạo gradient.
- Hai config chính dùng chung data/backbone/training; không cho gắn nhãn reproduction với giới hạn mẫu.
- CLI train → checkpoint → extraction task kế tiếp bằng dữ liệu in-memory, chỉ mock ranh giới tải mạng; từ chối ghi đè output.

## CIFAR-100 thật

Đã chạy:

```powershell
.venv\Scripts\python src/scripts/run_experiment.py prepare --config configs/reproduction/cifar100_fecam.yaml --output outputs/data-prepare
```

Nguồn/revision theo [protocol](protocol.md). Kiểm tra đạt 100 lớp, 500 train và 100 test mẫu mỗi lớp. Thời gian tải/chuẩn bị trong lần đo: khoảng **61.0 giây**, gồm mạng/cache, không dùng làm benchmark training.

| Task | Lớp train mới | Mẫu train | Mẫu test trên tất cả lớp đã thấy |
|---|---|---|---|
| 0 | 0–49 | 25,000 | 5,000 |
| 1 | 50–59 | 5,000 | 6,000 |
| 2 | 60–69 | 5,000 | 7,000 |
| 3 | 70–79 | 5,000 | 8,000 |
| 4 | 80–89 | 5,000 | 9,000 |
| 5 | 90–99 | 5,000 | 10,000 |

Đây là nhãn sau remap. Mười lớp gốc đầu của class order: `[68,56,78,8,23,84,90,65,74,76]`.

Prepared Arrow cache nằm dưới `data/huggingface/`; thư viện Hub có thể lưu thêm metadata/source cache ở thư mục Hugging Face mặc định của người dùng. Windows cảnh báo không hỗ trợ symlink và Hub cảnh báo request không đăng nhập; lần kiểm tra vẫn tải/đọc dữ liệu thành công, không cần token hoặc quyền admin.

## Smoke CPU với ảnh thật

Đã chạy `train-base --config configs/smoke.yaml --output outputs/backbone-smoke-cpu --device cpu`, sau đó `extract-features` từ checkpoint đó với `--task 1 --output outputs/backbone-smoke-cpu-task1 --device cpu`.

| Chặng | Kết quả | Thời gian stage |
|---|---|---|
| Train task đầu | 40 mẫu, 1 epoch, loss hữu hạn 1.6239 | 1.387 s |
| Extract train task 0 | `[40,512]`, nhãn 0–4 | 0.457 s |
| Extract test task 0 | `[20,512]`, nhãn 0–4 | 0.153 s |
| Nạp checkpoint, extract train task 1 | `[40,512]`, nhãn 5–9 | 0.421 s |
| Extract test seen task 1 | `[40,512]`, nhãn 0–9 | 0.277 s |

SHA-256 checkpoint CPU: `6ce078b7afb241866f67f2c2e994aa66555ae2c5e2545cef3bf0c807ee611275`. Hai run dùng cùng checksum. Không lưu feature train từng mẫu ra đĩa.

Các thư mục output trên là log/checkpoint cục bộ được Git ignore. Số đo này chỉ xác nhận pipeline; subset rất nhỏ và chưa warm-up nên không suy ra tốc độ trên toàn bộ dữ liệu hoặc ETA run 200 epoch.

## Smoke CUDA với ảnh thật

Đã chạy thành công cùng hai lệnh train/extract ở trên với `--device cuda`, output lần lượt là `outputs/backbone-smoke-cuda` và `outputs/backbone-smoke-cuda-task1`.

| Chặng | Kết quả | Thời gian stage |
|---|---|---|
| Train task đầu | 40 mẫu, 1 epoch, loss hữu hạn 1.6242 | 1.374 s |
| Extract train task 0 | `[40,512]`, nhãn 0–4 | 0.061 s |
| Extract test task 0 | `[20,512]`, nhãn 0–4 | 0.039 s |
| Nạp checkpoint, extract train task 1 | `[40,512]`, nhãn 5–9 | 0.288 s |
| Extract test seen task 1 | `[40,512]`, nhãn 0–9 | 0.032 s |

Đỉnh bộ nhớ do PyTorch **allocated**: 223.2 MiB cho smoke train, 67.0 MiB cho tiến trình nạp/extract task 1. Đây không phải tổng VRAM dùng bởi driver/hệ thống và không đại diện cho batch 128/full run. Stage đầu có khởi động kernel; không suy ra speedup CPU/GPU từ subset này.

SHA-256 checkpoint CUDA: `7c721c1cac074c52f8cc2d982349f91a967a6f1b48d40ed222b69a6f4a6228ce`. Hai run CUDA dùng cùng checksum. Smoke CPU/CUDA có sai khác số học nhỏ, không so sánh checkpoint byte-for-byte giữa hai thiết bị.

`git diff --check` đạt theo cấu hình Git hiện tại. Git chỉ cảnh báo chuyển LF sang CRLF trên Windows; không thay cấu hình Git. Các source/test mới cũng đã được kiểm tra import/runtime qua test suite. Code, tài liệu và kiểm tra được xây dựng với trợ giúp Codex; nguồn đối chiếu được ghi trong protocol.

## Phần chưa kiểm tra/chưa triển khai

Chưa chạy 200 epoch, chưa có accuracy NCM/FeCAM, full incremental evaluation, pilot chi phí đầy đủ, W&B public run hoặc resume training. Chưa đo RAM hệ thống. Chưa thử loader nhiều worker trên Windows hoặc môi trường Colab/Kaggle. Các sai khác với recipe tác giả (AutoAugment thư viện, RNG/thứ tự dữ liệu/phiên bản thư viện) và paper (weight decay) được ghi trong protocol; chưa khẳng định khớp số paper.

# Chạy pipeline dữ liệu/backbone

Đọc [protocol đã chốt](protocol.md) trước khi đổi cấu hình. Cần Python 3.10+; môi trường kiểm chứng được ghi trong [validation](data-backbone-validation.md). Chạy các lệnh bên dưới từ thư mục gốc `fecam-study`.

## Cài đặt

Windows PowerShell, NVIDIA GPU:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

Máy chỉ có CPU: thay index PyTorch bằng `https://download.pytorch.org/whl/cpu`. Linux dùng `.venv/bin/python`. Không cần tải pretrained weights. Cặp phiên bản và index theo [hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/#v271).

## Kiểm tra và chuẩn bị dữ liệu

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python src/scripts/run_experiment.py prepare --config configs/reproduction/cifar100_fecam.yaml --output outputs/data-prepare
```

`prepare` tải CIFAR-100 từ revision Hugging Face cố định, kiểm tra số mẫu/lớp và tạo manifest/order/task counts. Chạy lại được bằng output mới; thư mục output cũ sẽ bị từ chối để tránh ghi đè. Prepared Arrow cache nằm trong `data/`, được tái sử dụng khi tải lại; Hub có thể lưu thêm metadata/source cache trong thư mục Hugging Face mặc định của người dùng. Nếu cache hỏng hoặc tải thất bại, lệnh trả lỗi thay vì chuyển nguồn âm thầm.

## Smoke backbone và feature task kế tiếp

```powershell
.venv\Scripts\python scripts/run_experiment.py train-base --config configs/smoke.yaml --output outputs/backbone-smoke
.venv\Scripts\python scripts/run_experiment.py extract-features --config configs/smoke.yaml --checkpoint outputs/backbone-smoke/base.pt --task 1 --output outputs/backbone-smoke-task1
```

Các lệnh `train-base` và `extract-features` kiểm tra riêng backbone/checkpoint. Để đánh giá end-to-end gồm classifier và accuracy incremental, chạy `python scripts/run_experiment.py --config configs/smoke.yaml --device cpu` cho NCM hoặc `python scripts/run_experiment.py --config configs/smoke_fecam.yaml --device cpu` cho FeCAM. Mỗi YAML chạy một classifier; smoke dùng ít mẫu và không đại diện cho tái hiện đầy đủ. Summary end-to-end ghi accuracy theo session cùng các metrics CL; feature train không được dump ra đĩa.

## Huấn luyện backbone chính khi đã sẵn sàng

```powershell
.venv\Scripts\python scripts/run_experiment.py train-base --config configs/reproduction/cifar100_fecam.yaml --output outputs/cifar100-b50-seed1
```

Lệnh này chạy **200 epoch**, chưa được tự khởi chạy bởi thay đổi data/backbone. Chỉ chạy sau khi có tài nguyên/thời gian phù hợp và đã kiểm tra pilot; checkpoint hiện chưa hỗ trợ resume training giữa chừng. `base.pt` lưu sau epoch cuối, `epochs.jsonl` lưu tiến độ từng epoch. Nếu cần tái sử dụng cho NCM, dùng cùng checkpoint và config NCM có cùng trường data/split/backbone/training.

```powershell
.venv\Scripts\python scripts/run_experiment.py extract-features --config configs/reproduction/cifar100_ncm.yaml --checkpoint outputs/cifar100-b50-seed1/base.pt --task 1 --output outputs/cifar100-b50-task1
```

CLI `extract-features` chỉ kiểm tra luồng extraction và không lưu feature ra đĩa. Đánh giá NCM/FeCAM hoàn chỉnh hiện được thực hiện bởi `scripts/run_experiment.py` với một YAML tương ứng.

## Artifact và lỗi

- `manifest.json`: config đầy đủ, class order/mapping, code hash, môi trường và thiết bị lúc bắt đầu.
- `summary.json`: trạng thái hoàn tất, thống kê dữ liệu/task, dataset fingerprint, số đo và checksum checkpoint nếu có.
- `epochs.jsonl` / `base.pt`: tiến độ train / checkpoint backbone task đầu.
- `failure.json`: lỗi gặp khi tải/train/extract; traceback vẫn hiện ở terminal.

Data/checkpoint/output được Git ignore. Đưa kết quả kiểm chứng ngắn vào docs khi cần báo cáo. Không có W&B upload hay public result trong luồng hiện tại.

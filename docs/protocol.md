# Protocol CIFAR-100 — dữ liệu và backbone

Đã chốt ngày 01/11/2026, cập nhật weight decay theo paper ngày 02/11/2026 trên `feat/data-backbone`. ID hiện tại: `cifar100-b50-i10-paper-wd-v2`.
Phạm vi hiện tại: tải dữ liệu, chia task, train task đầu có checkpoint, trích feature và đánh giá smoke cho NCM/FeCAM. Reproduction đầy đủ theo CIFAR-100 100 lớp và 200 epoch chưa được xác nhận.

## Setting và số liệu đối chiếu

Chọn many-shot exemplar-free class-incremental learning, CIFAR-100 **B=50, increment=10**: một task đầu và năm incremental task, tổng cộng sáu lần đánh giá. T=5 trong bảng paper chỉ số incremental task; không hiểu thành tổng năm task. Backbone chỉ học task đầu; sau đó đóng băng. Đánh giá không cung cấp task ID.

Mục tiêu so sánh sau khi tích hợp classifier: Table 1, cột CIFAR-100 T=5; ablation tiếp theo: Table 4, cùng setting. Mốc paper: FeCAM per-class covariance Avg=70.9%, Last=62.1%; Euclidean-NCM Avg=64.8%, Last=51.6%. NCM là đối chứng đơn giản; không thay thế toàn bộ baseline của Table 1. Nguồn: [paper, §4.1, Tables 1/4](https://papers.neurips.cc/paper_files/paper/2023/file/15294ba2dcfb4521274f7aa1c26f4dd4-Paper-Conference.pdf).

Average incremental accuracy là trung bình top-1 accuracy trên tất cả lớp đã thấy sau **cả sáu task, gồm task đầu**; final accuracy là accuracy sau task cuối. Không lấy trung bình accuracy từng lớp/task con thay cho accuracy trên toàn bộ test seen-classes. Nguồn: [supplement §2](https://papers.nips.cc/paper_files/paper/2023/file/15294ba2dcfb4521274f7aa1c26f4dd4-Supplemental-Conference.pdf).

## Nguồn xác minh

Code tác giả chỉ được đọc để đối chiếu cấu hình/kiến trúc; không đưa file implementation của tác giả vào repo. Các liên kết dưới cố định commit `e33f39d112ff2d2a2df2e68c490af579a50edd31`:

- [Config CIFAR-100](https://github.com/dipamgoswami/FeCAM/blob/e33f39d112ff2d2a2df2e68c490af579a50edd31/exps/FeCAM_cifar100.json): lịch task, optimizer settings và tham số classifier.
- [Data manager](https://github.com/dipamgoswami/FeCAM/blob/e33f39d112ff2d2a2df2e68c490af579a50edd31/utils/data_manager.py): permutation lớp và ánh xạ nhãn.
- [Data transforms](https://github.com/dipamgoswami/FeCAM/blob/e33f39d112ff2d2a2df2e68c490af579a50edd31/utils/data.py), [setup training](https://github.com/dipamgoswami/FeCAM/blob/e33f39d112ff2d2a2df2e68c490af579a50edd31/models/fecam.py), [trainer](https://github.com/dipamgoswami/FeCAM/blob/e33f39d112ff2d2a2df2e68c490af579a50edd31/trainer.py): augmentation, cosine schedule, momentum và training seed.
- [ResNet](https://github.com/dipamgoswami/FeCAM/blob/e33f39d112ff2d2a2df2e68c490af579a50edd31/convs/resnet.py), [network/head](https://github.com/dipamgoswami/FeCAM/blob/e33f39d112ff2d2a2df2e68c490af579a50edd31/utils/inc_net.py): CIFAR stem và cosine head khi train task đầu.

## Dữ liệu và chia task

| Mục | Quyết định |
|---|---|
| Nguồn | [uoft-cs/cifar100 trên Hugging Face](https://huggingface.co/datasets/uoft-cs/cifar100), revision `aadb3af77e9048adbea6b47c21a81e47dd092ae5` |
| Split/label | Giữ split gốc; chỉ dùng `fine_label`, bỏ `coarse_label` |
| Kiểm tra | Train 50,000 ảnh, test 10,000 ảnh; 100 lớp với 500 train/100 test mỗi lớp; ảnh RGB 32×32 |
| Class order | `numpy.random.RandomState(1993).permutation(100)`, dùng MT19937 legacy; không dùng `default_rng` |
| Nhãn | Nhãn liên tục là vị trí trong class order; lưu order, mapping và lịch task trong manifest |
| Train task t | Chỉ lớp mới của task t; không cung cấp tùy chọn seen-train/replay |
| Test task t | Toàn bộ test của lớp đã thấy từ task 0 đến t |
| Dữ liệu cục bộ | Prepared Arrow cache dưới `data/huggingface/`, được Git ignore; Hub có thể lưu thêm metadata/source cache trong thư mục người dùng; revision và fingerprint ghi trong summary |

Class-order seed `1993` và training seed `1` là hai giá trị riêng. Config hiện tại tái lập setting phát hành một order/một training seed. Nếu khảo sát thêm seed, giữ nguyên recipe và ghi cả hai seed; không tự gọi một run là kết quả nhiều seed.

Smoke dùng **10 lớp đầu của cùng permutation 100 lớp**, B=5, increment=5; dùng toàn bộ 500 ảnh train và 100 ảnh test cho mỗi lớp đã chọn; 3 epoch, batch 16. Giới hạn 10 lớp và backbone 3 epoch có chủ đích để kiểm tra đường dữ liệu và không thuộc kết quả tái hiện đầy đủ.

Smoke CPU ghép cặp mới nhất (07/11/2026), cùng seed và backbone: NCM với L2 normalization đạt Avg=58.40%, Final=42.80%, BWT=−10.20 điểm %; FeCAM đạt Avg=64.25%, Final=51.70%, BWT=−14.00 điểm %. Task 0 accuracy lần lượt 74.0% và 76.8%; sau task 1, accuracy trên lớp cũ/mới lần lượt là NCM 63.8%/21.8%, FeCAM 62.8%/40.6%. Mỗi backbone train 3 epoch trên 2,500 ảnh của base task. Đây là smoke 10 lớp, không so trực tiếp với reproduction full CIFAR-100. Summary: `outputs/smoke-ncm-20261107-l2-normalized/summary.json` và `outputs/smoke-fecam-20261107-confirmed/summary.json`.

## Backbone và training đã chốt

| Mục | Quyết định |
|---|---|
| Kiến trúc | ResNet-18, blocks [2,2,2,2], widths [64,128,256,512] |
| Stem CIFAR | Conv 3×3, stride 1, padding 1, BatchNorm + ReLU; bỏ maxpool đầu |
| Feature | Adaptive average pooling cuối mạng, flatten thành `[N,512]`; sau ReLU, không âm |
| Khởi tạo | Random; không nạp pretrained ImageNet |
| Head task đầu | Cosine giữa feature và weight chuẩn hóa L2; scale trainable khởi tạo 1, không bias; chỉ dùng để train task đầu |
| Loss/optimizer | Cross entropy; SGD lr=0.1, momentum=0.9, weight decay=0.0001 |
| Schedule | CosineAnnealingLR, T_max=200, eta_min=0; step sau mỗi epoch |
| Thời lượng | 200 epoch task đầu; batch 128; FP32 |
| Loader | Shuffle train, không shuffle extraction/test; không drop batch cuối; mặc định workers=0 cho Windows |
| Train transforms | RandomCrop(32,padding=4) → HorizontalFlip(p=0.5) → ColorJitter(brightness=63/255) → torchvision AutoAugment(CIFAR10) → ToTensor → Normalize |
| Eval/extract | ToTensor → Normalize, không resize, không augmentation |
| Normalize | mean=(0.5071,0.4867,0.4408), std=(0.2675,0.2565,0.2761) |
| Chọn checkpoint | Trọng số sau epoch cuối cố định; không chọn bằng test accuracy |
| Incremental | `freeze()` toàn bộ tham số; BatchNorm giữ eval kể cả khi caller gọi train() |

Weight decay của cả hai cấu hình tái hiện và cấu hình smoke là `1e-4`, khớp paper §4.1. Cosine head được dùng để train backbone task đầu; khi trích feature cho NCM/FeCAM, runner gọi `Backbone.extract_features()` trực tiếp và bỏ qua head. Khi thay đổi cấu hình training, cần huấn luyện lại để tạo checkpoint tương ứng; không dùng checkpoint của cấu hình khác.

Implementation tái sử dụng [torchvision ResNet-18](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.resnet18.html), thay stem/pooling và tự viết cosine head. [AutoAugment CIFAR10 của torchvision](https://docs.pytorch.org/vision/0.22/generated/torchvision.transforms.AutoAugment.html) thay cho custom `CIFAR10Policy` của tác giả; cách lấy mức biến đổi, RNG/fill có thể khác. Ngoài ra, trainer hiện gọi `CosineAnnealingLR.step()` mỗi epoch, trong khi trainer tác giả gọi scheduler sau mỗi minibatch. Nguồn Hugging Face, thứ tự dữ liệu và phiên bản thư viện cũng có thể khác môi trường gốc. Đây là các sai khác cần ghi nhận khi đối chiếu accuracy; chưa cam kết tái hiện từng bit hoặc khớp chính xác số paper.

## Giao diện bàn giao classifier

- `TaskSplit.classes(task, seen=False)` trả nhãn đã remap của lớp mới; `seen=True` trả tất cả lớp đã thấy.
- `TaskDataset(..., source='train', seen=False, augment=False)` dùng trích xuất feature train task hiện tại; `source='test', seen=True` dùng đánh giá seen-classes. DataLoader trả `(images, labels)`.
- `Backbone.extract_features(images)` trả **raw features** 512 chiều trước cosine head, chưa Tukey/chưa L2 normalize. Hàm dùng bởi runner là `src/engine/extract_features.py::extract_features`, trả tensor CPU `(features[N,512], labels[N])` đúng thứ tự loader, không gradient. `IncrementalNCM._forward_features()` cũng gọi trực tiếp `self.backbone.extract_features(imgs)`; tránh gọi `self.backbone(imgs)`, vì hàm `forward()` trả logits cosine-head chứ không phải embedding.
- FeCAM thực hiện Tukey beta=0.5, covariance đầy đủ theo lớp, shrinkage gamma1=gamma2=1 và correlation normalization. NCM dùng Euclidean với L2-normalize feature kiểm tra và prototype (`normalize_features: true`, `normalize_prototypes: true`), như code tác giả trong `BaseLearner._mahalanobis`; prototype được tính bằng mean của raw features trước khi chuẩn hóa ở bước dự đoán. FeCAM implementation hiện tại chưa L2-normalize feature/prototype như code tác giả; ghi nhận đây là sai khác khi đối chiếu. Avg Acc được tính từ accuracy của NCM/FeCAM ở từng phiên; cosine head không tham gia dự đoán incremental.
- NCM/FeCAM dùng cùng checkpoint task đầu. Tính prototype/covariance từ train task hiện tại, giữ thống kê cần thiết; không lưu ảnh/feature mẫu cũ cho task sau. Runner chỉ lưu summary feature, không dump feature train ra đĩa.
- Checkpoint gồm weights, số lớp head và metadata plain gồm config/order/seed/phiên bản/hash code; nạp với `weights_only=True`, kiểm tra config/order trước extract. Không tương thích trực tiếp checkpoint của tác giả do tên module khác.

## Truy vết và giới hạn vận hành

Lệnh mới tạo thư mục output; `--resume` chỉ dùng thư mục đang có checkpoint và chưa có `summary.json`. Runner lưu `base_training_checkpoint.pt` sau mỗi epoch bằng ghi tạm rồi thay thế nguyên tử, nên checkpoint trước đó vẫn còn nếu lần ghi mới bị gián đoạn. Summary ghi thời gian train tổng cộng qua các lần resume. Manifest có task counts để kiểm tra rò rỉ lớp.

Checkpoint weights-only `base.pt` dùng để nạp cho extraction. Checkpoint `base_training_checkpoint.pt` giữ thêm optimizer, scheduler, RNG và trạng thái DataLoader để tiếp tục từ đầu epoch kế tiếp. Nếu Kaggle khởi tạo working directory mới, cần tải/đính kèm checkpoint và chép lại vào thư mục output trước khi dùng `--resume`. Full reproduction và tracking W&B tự động trong runner vẫn là các bước cần hoàn thiện; smoke summaries đã có thể upload bằng script riêng.

Không dùng test để chọn siêu tham số. Smoke dùng đủ ảnh của 10 lớp nhưng chỉ train 3 epoch và không đại diện cho thời gian/độ chính xác của reproduction đầy đủ; cần pilot full dữ liệu task đầu và stage classifier trước khi bắt đầu run chính.

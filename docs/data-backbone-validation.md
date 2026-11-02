# Kiểm chứng chạy pipeline dữ liệu/backbone

Pipeline đã được chạy kiểm tra trên CPU và CUDA trong môi trường Windows. Tài liệu ghi lại các bước đã chạy thành công và dấu hiệu nhận biết khi chạy lại. Cách cài đặt và lệnh thực thi nằm trong [hướng dẫn pipeline](data-backbone.md); cấu hình thực nghiệm nằm trong [protocol](protocol.md).

## Môi trường và test tự động

Dependencies được cài theo `requirements.txt`. PyTorch nhận GPU khi dùng CUDA và `pip check` không phát hiện dependency lỗi.

Chạy bộ test từ thư mục gốc repository:

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

Bộ test đã chạy thành công. Các kiểm tra bao phủ chia task, ánh xạ nhãn, tách lớp train/test, trích xuất feature, đóng băng backbone, lưu/nạp checkpoint và luồng CLI. Chạy thành công khi lệnh kết thúc với `OK`, không có test lỗi hoặc thất bại.

## Chuẩn bị dữ liệu

Đã chạy lệnh `prepare` với cấu hình tái hiện. Pipeline tải và đọc được CIFAR từ nguồn đã chốt, xác nhận dữ liệu đúng protocol và tạo được lịch task.

Dấu hiệu thành công:

- Không có lỗi tải, đọc ảnh hoặc kiểm tra dữ liệu.
- Mỗi task chỉ có các lớp train mới; tập test bao gồm các lớp đã thấy.
- Terminal báo `Completed prepare`.

## Smoke backbone và trích xuất feature

Đã chạy cấu hình smoke trên CPU và CUDA. Sau khi cập nhật weight decay theo paper, đã chạy lại smoke trên CUDA và kiểm tra checkpoint mới.

Luồng kiểm tra gồm:

1. Chạy `train-base` với cấu hình smoke để huấn luyện backbone task đầu.
2. Xác nhận quá trình train hoàn tất, loss hữu hạn và checkpoint được ghi thành công.
3. Chạy `extract-features` để nạp lại checkpoint và trích xuất feature của task kế tiếp.
4. Xác nhận feature có dạng hợp lệ, đủ mẫu và đúng nhãn; train chỉ chứa lớp mới, test chứa tất cả lớp đã thấy.

Terminal đã báo `Completed train-base` và `Completed extract-features`. Metadata checkpoint khớp cấu hình chạy; việc nạp lại và trích xuất feature hoàn tất không lỗi. Các kiểm tra tự động cũng xác nhận backbone đã đóng băng không cập nhật tham số hoặc thống kê BatchNorm.

Lệnh chạy lại từng bước được ghi trong [hướng dẫn smoke](data-backbone.md#smoke-backbone-và-feature-task-kế-tiếp).

## Phạm vi đã xác nhận

Luồng dữ liệu → chia task → train backbone → lưu/nạp checkpoint → trích xuất feature đã chạy thành công. Đây là kiểm chứng vận hành của phần data/backbone, chưa xác nhận kết quả tái hiện FeCAM/NCM.

Chưa chạy cấu hình huấn luyện chính đầy đủ, đánh giá incremental, W&B public run, resume training hoặc môi trường Colab/Kaggle. Những lựa chọn triển khai chưa được xác nhận từ paper chính vẫn được ghi rõ trong protocol.

Outputs và checkpoint kiểm tra cục bộ đã được xóa. Repository giữ code, cấu hình, test và tài liệu; khi chạy lại, chương trình tạo artifact mới tại thư mục output của lần chạy.

Code, tài liệu và kiểm tra được xây dựng với trợ giúp Codex; nguồn đối chiếu được ghi trong protocol.

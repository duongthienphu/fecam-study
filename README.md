FeCAM Study
Tái hiện độc lập, phân tích và đánh giá FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning (NeurIPS 2023).

Đồ án môn Nhập môn Học máy. Repository lưu mã nguồn, cấu hình thực nghiệm, tài liệu đọc hiểu và các kết quả phục vụ slide giữa kỳ, poster cuối kỳ.
Trạng thái: Đang xây dựng cấu trúc và triển khai ban đầu. Phạm vi thực nghiệm dưới đây là kế hoạch; chưa có kết quả tái hiện được xác nhận. Các mục TODO cần cập nhật trước khi nộp.

1. Liên kết
Nội dung	Liên kết
Bài báo gốc	NeurIPS 2023
Mã nguồn tác giả	dipamgoswami/FeCAM
Repository nhóm	TODO: thêm URL repository
W&B công khai	TODO: thêm URL project
Slide giữa kỳ	TODO: thêm đường dẫn bản nộp
Poster cuối kỳ	TODO: thêm đường dẫn bản nộp


2. Mục tiêu đồ án
1. Đọc hiểu và trình bày lại vấn đề, đóng góp, phương pháp và protocol của FeCAM bằng lời giải thích và hình minh họa của nhóm.
2. Tự cài đặt phần cốt lõi từ mô tả trong bài báo và supplementary materials.
3. Tái hiện các thí nghiệm đã chọn; đối chiếu với bài báo và implementation tác giả, giải thích sai khác.
4. Phân tích điểm mạnh, giả định, hạn chế và độ phức tạp của phương pháp.
5. Thiết kế thí nghiệm để kiểm tra các hạn chế được đề xuất.
6. Công khai cấu hình, mã nguồn và W&B run để truy xuất nguồn gốc các kết quả.
3. Phạm vi thực nghiệm dự kiến
Nhóm chọn core scope: tối thiểu hai thí nghiệm quan trọng, bao gồm so sánh chính, và một phân tích bổ sung do nhóm thiết kế.
Hạng mục	Kế hoạch
Bài toán chính	Exemplar-free class-incremental learning
Dataset chính	CIFAR-100
Backbone chính	ResNet-18; chốt cấu hình cụ thể theo bài gốc
Phương pháp chính	FeCAM tự cài đặt
Baseline đơn giản	Nearest Class Mean (NCM)
Tái hiện thứ nhất	So sánh chính theo protocol được chọn từ bài gốc
Tái hiện thứ hai	Ablation quan trọng về covariance hoặc thành phần xử lý
Phân tích bổ sung	Độ nhạy theo số mẫu mỗi lớp
Khảo sát chi phí	Bộ nhớ và thời gian theo số lớp/số chiều đặc trưng


TODO trước các run chính: ghi rõ bảng/hình được tái hiện, baseline cần thiết, số lớp task đầu, increment, class order, seed, preprocessing và cấu hình training trong docs/protocol.md.

Các run rút gọn dùng để kiểm tra pipeline phải được gắn nhãn smoke-test, không trình bày như kết quả tái hiện đầy đủ. NCM không thay thế toàn bộ các baseline cần thiết cho so sánh chính.
4. Thành viên và phân công
Thành viên	Trách nhiệm chính
Dương Thiên Phú	Phương pháp, FeCAM classifier, ablation, lập luận phản biện
TV2 — TODO: tên, MSSV	Dataset, chia task, backbone, training, tích hợp pipeline và tái hiện
TV3 — TODO: tên, MSSV	NCM, metrics, W&B, đo chi phí, tổng hợp kết quả


Phần đồ án được cân bằng theo tổng khối lượng code, thực nghiệm, tài liệu và review. Mỗi thành viên chịu trách nhiệm nội dung, hình và speaker notes của phần mình, đồng thời phải hiểu toàn bộ pipeline.

Review chéo: Phú review TV2; TV2 review TV3; TV3 review Phú.
5. Cấu trúc repository
Đường dẫn	Công dụng	Theo dõi bằng Git
configs/	Cấu hình chạy thử, tái hiện, phân tích và nghiên cứu	Có
src/	Implementation, pipeline dữ liệu, backbone và đánh giá	Có
scripts/	Chạy thí nghiệm, tổng hợp bảng và vẽ đồ thị	Có
tests/	Kiểm tra tính đúng của chia task, classifier và metrics	Có
docs/	Method, protocol, evaluation, critique và related work	Có
notebooks/	Notebook chạy Colab/Kaggle và khảo sát	Có; tránh output lớn
results/	CSV tổng hợp, danh mục run, hình đã chọn cho báo cáo	Có
data/	Dữ liệu tải về, dữ liệu xử lý và cache cục bộ	Không
checkpoints/	Trọng số và trạng thái training	Không
outputs/	Log, dự đoán chi tiết và đầu ra tạm của từng run	Không
README.md	Tổng quan và hướng dẫn sử dụng	Có
requirements.txt	Dependencies được nhóm kiểm chứng	Có


Các tài liệu dự kiến trong docs/: method.md, protocol.md, evaluation.md, reproduction.md, critique.md, related_work.md, research_plan.md, contributions.md. Tạo dần khi có nội dung.
6. Môi trường và cách chạy
Chuẩn bị môi trường
Yêu cầu môn học: Python 3.9 trở lên. Nhóm cần thống nhất và ghi rõ phiên bản Python, PyTorch, CUDA, hệ điều hành và GPU đã kiểm chứng.

Sau khi clone repository, mở terminal tại thư mục gốc:
python -m venv .venv
Kích hoạt trên Windows PowerShell:
.venv\Scripts\Activate.ps1
Hoặc Linux/macOS:
source .venv/bin/activate
Sau khi nhóm điền dependencies đã kiểm chứng vào requirements.txt:
python -m pip install -r requirements.txt
Dữ liệu, checkpoint và lệnh thực nghiệm
- Dữ liệu phải được tải và chuẩn bị bằng code từ nguồn Hugging Face/Kaggle phù hợp; ghi rõ nguồn, phiên bản và cách xử lý trong tài liệu protocol.
- Checkpoint dùng chung cần có URL tải, nguồn gốc, cấu hình huấn luyện và mã kiểm tra khi có. Không commit file trọng số vào Git.
- Cache đặc trưng không được dùng để truy cập lại mẫu lớp cũ hoặc lớp tương lai trái với protocol exemplar-free đã chọn.
- Không đưa token hoặc API key vào notebook, config hay repository.
TODO: bổ sung lệnh thực tế sau khi entry point được triển khai và kiểm tra. Hiện README chưa cam kết có lệnh chạy end-to-end hoạt động.
Tác vụ	Lệnh đã kiểm chứng
Tải và chuẩn bị dữ liệu	TODO
Chạy smoke test	TODO
Huấn luyện backbone task đầu	TODO
Tái hiện FeCAM và baseline	TODO
Chạy phân tích/stress-test	TODO
Tổng hợp bảng và vẽ hình	TODO


7. Theo dõi và báo cáo thực nghiệm
Mỗi run ghi nhận: experiment ID, config, seed, class order, dataset/protocol, checkpoint backbone, phiên bản môi trường, thiết bị và Git commit. Các thay đổi chưa commit phải được ghi rõ; run dùng cho báo cáo cần gắn với một phiên bản code xác định.

W&B dùng các tag phân biệt: smoke-test, reproduction, analysis, research.

Sau mỗi task, đánh giá trên tất cả các lớp đã gặp. Average incremental accuracy và final accuracy phải tuân theo định nghĩa và các bước được tính trong protocol đã chốt.
Experiment	Script/notebook	Config	W&B run	Điều run cho thấy
Smoke test	TODO	TODO	TODO	Kiểm tra pipeline; không dùng làm kết quả tái hiện
Tái hiện so sánh chính	TODO	TODO	TODO	Chưa chạy/xác nhận
Tái hiện ablation	TODO	TODO	TODO	Chưa chạy/xác nhận
Phân tích số mẫu	TODO	TODO	TODO	Chưa chạy/xác nhận
Khảo sát chi phí	TODO	TODO	TODO	Chưa chạy/xác nhận


Kết quả tổng hợp lưu trong results/. Mỗi số liệu trên slide/poster phải truy được đến run tương ứng. Khi so sánh với bài gốc, báo cáo chênh lệch điểm phần trăm và sai lệch tương đối khi phù hợp; nêu rõ những khác biệt về cấu hình. Không chọn siêu tham số bằng test set.
8. Quy trình Git
Branch	Vai trò
main	Bản ổn định dùng cho các mốc nộp
dev	Tích hợp và kiểm tra code của nhóm
feat/fecam-classifier	Phú triển khai FeCAM
feat/data-backbone	TV2 triển khai dữ liệu và backbone
feat/evaluation-tracking	TV3 triển khai baseline, metrics và tracking


1. Tạo Issue với người phụ trách, đầu ra, tiêu chí hoàn thành và tuần hạn.
2. Tạo branch công việc từ dev mới nhất.
3. Commit theo từng thay đổi có ý nghĩa, mở pull request vào dev.
4. Thành viên khác review theo vòng phân công, kiểm tra trước khi merge.
5. Khi pipeline ổn định, mở pull request từ dev vào main.
Xóa branch công việc sau khi merge; tạo branch mới cho đầu việc tiếp theo. Không cần một branch cho mỗi lần chạy hoặc mỗi seed.
9. Các mốc nộp
Hạn dự kiến	Sản phẩm	Tag phiên bản
07/11/2026	Repository công khai, hướng dẫn và tiến độ triển khai	project-github
21/11/2026	Slide đọc hiểu và tái hiện; link repository	project-midterm
05/12/2026	Poster và code cuối kỳ; chuẩn bị bảo vệ theo lịch môn học	project-final


Deadline chính thức theo Moodle được ưu tiên nếu có thay đổi.
- Slide: 20–40 slide gồm appendix, mỗi slide có speaker notes tối đa 500 từ, hình tự tạo, citation IEEE tại slide liên quan; PPTX dưới 20 MB.
- Poster: một slide PPTX khổ A2 dọc; kết nối lập luận phản biện với bằng chứng thực nghiệm và nguồn W&B.
- Phản biện: ít nhất ba điểm mạnh, hai hạn chế có lập luận và hai hướng phát triển gắn với hạn chế. Chỉ kết luận điều dữ liệu thực nghiệm thực sự hỗ trợ.
10. Hướng nghiên cứu mở rộng
Sau các mục tiêu bắt buộc của đồ án, nhóm dự kiến khảo sát phân bổ mức độ chi tiết của covariance theo từng lớp dưới ngân sách bộ nhớ cố định.

Câu hỏi nghiên cứu: Trong exemplar-free class-incremental learning với backbone cố định, phân bổ độ chi tiết theo độ nhạy của khoảng cách phân loại đối với phép nén có cải thiện accuracy so với covariance chung, đường chéo và nén đồng đều dưới cùng giới hạn bộ nhớ không?

Đây là giả thuyết đang khảo sát, chưa phải đóng góp mới hoặc kết quả đã được chứng minh. Ngân sách cần tính đầy đủ trạng thái classifier lưu giữa các task, gồm prototype, biểu diễn covariance, thống kê phân bổ và metadata; báo cáo riêng chi phí cập nhật và bộ nhớ đỉnh.

Phú dự kiến dẫn dắt phương pháp và bản thảo; TV2 phụ trách benchmark/protocol; TV3 phụ trách đối chứng, độ bền và chi phí. Sáu tuần đầu ưu tiên đồ án; nhánh nghiên cứu có lịch riêng khoảng 20–22 tuần và được điều chỉnh theo bằng chứng thực tế. Việc ghi nhận tác giả được cả nhóm thống nhất theo đóng góp thực tế.
11. Tính độc lập và ghi nhận nguồn
- Core method được tự cài đặt từ mô tả trong paper và supplement.
- Code tác giả được chạy riêng để kiểm chứng và so sánh; không sao chép vào implementation nộp của nhóm.
- Thư viện và thành phần phụ trợ tái sử dụng phải được ghi nhận nguồn cùng giấy phép liên quan.
- Ghi lại việc sử dụng công cụ AI hỗ trợ; mỗi thành viên chịu trách nhiệm kiểm tra, hiểu và giải thích phần đã dùng.
- Repository này là công trình học tập/nghiên cứu độc lập của nhóm, không phải implementation chính thức của tác giả FeCAM.
12. Trích dẫn bài gốc
@inproceedings{goswami2023fecam,
  title={FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning},
  author={Goswami, Dipam and Liu, Yuyang and Twardowski, Bartlomiej and van de Weijer, Joost},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
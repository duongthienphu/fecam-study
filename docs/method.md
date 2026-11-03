# FeCAM: ghi chú phương pháp và đặc tả classifier

**Người phụ trách:** Phú. **Mốc:** bàn giao tuần 1.

**Nguồn chính:** Goswami et al., *FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning*, NeurIPS 2023.

## 1. Đọc hiểu phương pháp

### 1.1. Bài toán và ý tưởng

Trong class-incremental learning (CIL), mô hình nhận dữ liệu các lớp mới theo từng task. Khi dự đoán, mô hình phải chọn trong **tất cả lớp đã học**, không được biết task của ảnh kiểm tra. Exemplar-free nghĩa là không giữ ảnh/mẫu đặc trưng lớp cũ để replay ở các task sau, vẫn được lưu thống kê lớp như prototype và covariance.

FeCAM đóng băng backbone sau khi học task đầu, hoặc dùng backbone pretrained theo protocol tương ứng. Backbone biến ảnh thành vector đặc trưng. Classifier không học bằng gradient ở các task tiếp theo mà bổ sung thống kê của lớp mới.

NCM Euclidean đại diện mỗi lớp bằng một điểm trung bình và đo khoảng cách đến điểm đó. Cách này không mô tả được lớp phân tán theo hướng nào. FeCAM thêm covariance để dùng khoảng cách Mahalanobis: một độ lệch theo hướng vốn biến thiên nhiều của lớp sẽ được đánh giá khác với độ lệch theo hướng ít biến thiên. Sau correlation normalization, metric tập trung vào cấu trúc tương quan, không còn giữ nguyên toàn bộ độ lớn phương sai ban đầu.

**Pipeline bản per-class:**

1. Khang trích đặc trưng bằng backbone cố định.
2. Phú tính prototype từ đặc trưng gốc của từng lớp mới.
3. Biến đổi đặc trưng bằng Tukey nếu protocol cho phép.
4. Tính covariance của đặc trưng đã biến đổi.
5. Shrinkage covariance, rồi correlation normalization.
6. Lưu prototype và biểu diễn metric cho từng lớp, bỏ đặc trưng mẫu sau cập nhật.
7. Với ảnh kiểm tra, trích đặc trưng bằng cùng backbone, áp dụng cùng biến đổi rồi chọn lớp có bình phương khoảng cách Mahalanobis nhỏ nhất.

### 1.2. Prototype: tâm đại diện của lớp

Với đặc trưng gốc $h_i=\phi(x_i)$ và $n_y$ là số mẫu train thuộc lớp $y$:

$$\mu_y=\frac{1}{n_y}\sum_{i:y_i=y}h_i.$$

Trong đó $\mu_y$ là trung bình lớp $y$. Prototype có $d$ phần tử, cùng chiều với đặc trưng.
### 1.3. Tukey: biến đổi từng tọa độ đặc trưng

Theo Eq. (9):

$$\tilde{\phi}(x) = \begin{cases} \phi(x)^\lambda & \text{if } \lambda \neq 0 \\  \log(\phi(x)) & \text{if } \lambda = 0 \end{cases}$$

Đặt $T$ là phép biến đổi được chọn theo protocol, với $\tilde\phi(x)=T(\phi(x))$. Khi bật Tukey, $T$ được định nghĩa bởi Eq. (9); khi tắt Tukey, $T(h)=h$.

Phép toán thực hiện trên từng phần tử. Mục đích là giảm độ lệch phân phối, làm dữ liệu gần Gaussian hơn.

- ResNet-18 trong paper có đặc trưng không âm sau ReLU: dùng $\lambda=0.5$, tức căn bậc hai.
- Supplement mục 7 nêu **không dùng Tukey cho ViT pretrained** vì đặc trưng có giá trị âm. Khi đó $T(h)=h$.
- Bản đầu chỉ hỗ trợ `none` và `sqrt`, không lấy trị tuyệt đối hay cắt toàn bộ số âm về 0 để chạy được. Giá trị âm ở chế độ `sqrt` phải báo lỗi để kiểm tra pipeline.


### 1.4. Covariance: độ phân tán và quan hệ giữa các chiều

Đặt $z_i=T(h_i)=\tilde\phi(x_i)$, $\bar z_y=\frac1{n_y}\sum_{i:y_i=y}z_i$. Quy ước cài đặt đề xuất là sample covariance với mẫu số $n_y-1$:

$$S_y=\frac{1}{n_y-1}\sum_{i:y_i=y}(z_i-\bar z_y)(z_i-\bar z_y)^\top.$$

Đường chéo $S_y[j,j]$ là phương sai chiều $j$. Phần tử ngoài đường chéo cho biết hai chiều có xu hướng cùng tăng/giảm hay biến thiên ngược nhau. Ma trận có kích thước $d\times d$, đối xứng.

Paper không chỉ rõ mẫu số của estimator trong các công thức đang dùng, `ddof=1` là quy ước của bản đặc tả, cần kiểm tra khi tái lập. 

Nếu $n_y\le d$, covariance thường suy biến vì rank không vượt quá $n_y-1$.

### 1.5. Shrinkage: điều chỉnh covariance trước khi giải hệ

Eq. (8):

$$S_{y,s}=S_y+\gamma_1 V_{1,y} I+\gamma_2 V_{2,y}(J-I),$$

$$V_{1,y}=\frac1d\sum_j S_y[j,j],\qquad V_{2,y}=\frac1{d(d-1)}\sum_{j\ne k}S_y[j,k].$$

$I\in\mathbb R^{d\times d}$ là ma trận đơn vị; $J\in\mathbb R^{d\times d}$ là ma trận toàn số 1. Các chỉ số $j,k$ chạy từ $1$ đến $d$. Ký hiệu $(1-I)$ trong paper được viết thành $(J-I)$ để tránh hiểu nhầm trong code. $V_{2,y}$ ở đây là trung bình **có dấu**, với $d=1$, đặt $V_{2,y}=0$.

Thành phần đường chéo tăng mức regularization, thành phần ngoài đường chéo điều chỉnh tương quan. Đây là đúng dạng Eq. (8), không tự thay bằng một công thức shrinkage phổ biến khác.

| Setting trong paper | $\gamma_1$ | $\gamma_2$ |
| --- | ---: | ---: |
| Many-shot trên CIFAR100/TinyImageNet/ImageNet-Subset | 1 | 1 |
| Few-shot CIL | 100 | 100 |
| Split-CIFAR100/Core50 | 1 | 0 |
| Split-ImageNet-R | 10 | 10 |

**Tukey transformation:** Xét riêng theo đặc trưng của backbone. Với ResNet-18, đặc trưng không âm nhờ ReLU nên có thể áp dụng Tukey, phép biến đổi căn bậc hai tương ứng với $\lambda=0.5$. Với ViT pretrained trên ImageNet-21K, paper không áp dụng Tukey vì đặc trưng có thể âm.

**Nguồn:** mục 4.1 và supplement mục 7. Các giá trị $\gamma_1,\gamma_2$ trên được dùng cho những dataset/setting tương ứng, không phải quy tắc suy ra chỉ từ backbone.

**Lưu ý số học:** công thức có thành phần ngoài đường chéo nên không bảo đảm dương xác định với mọi đầu vào/tham số tùy ý. Nếu covariance toàn 0 thì $V_{1,y}=V_{2,y}=0$, phép cộng trên cũng không sửa được suy biến. Vì vậy cần kiểm tra, không chỉ gọi inverse và hy vọng chạy được.

### 1.6. Correlation normalization: đưa đường chéo về 1

Áp dụng Eq. (7) **sau shrinkage**:

$$R_y[j,k]=\frac{S_{y,s}[j,k]}{\sqrt{S_{y,s}[j,j]}\sqrt{S_{y,s}[k,k]}}.$$
Chuẩn hóa ma trận hiệp phương sai thành ma trận tương quan: đưa các phần tử đường chéo về 1, còn các phần tử ngoài đường chéo biểu diễn hệ số tương quan trong $[-1,1]$.

Ví dụ $S_s=\begin{bmatrix}4&2\\2&9\end{bmatrix}$ cho $R=\begin{bmatrix}1&1/3\\1/3&1\end{bmatrix}$. Hai chiều có cùng phương sai chuẩn hóa bằng 1 nhưng vẫn có tương quan dương.

Đây là chuẩn hóa **ma trận**, không phải chuẩn hóa L2 từng vector. Đường chéo bằng 1 cũng không có nghĩa determinant của các lớp bằng nhau.

### 1.7. Score và dự đoán

Với ảnh query $x$, đặt $h=\phi(x)$ và $\tilde\mu_y=T(\mu_y)$. Lưu ý $\tilde\mu_y$ khác $\bar z_y$ nói chung: $T(\mathrm{mean}(h_i)) \ne \mathrm{mean}(T(h_i))$.

$$v_y=T(h)-\tilde\mu_y,\qquad D_y(h)=v_y^\top R_y^{-1}v_y,\qquad\hat y=\arg\min_{y\in\mathcal C_t}D_y(h).$$

Đây là **bình phương** khoảng cách Mahalanobis, không cần căn vì căn không thay đổi thứ tự. Không có jitter, $R_y^{-1}$ đóng vai trò ma trận metric.

Để thống nhất với cài đặt có jitter ở §3.2, định nghĩa:

$$A_y=R_y+\eta_y I=L_yL_y^\top,\qquad M_y=A_y^{-1},\qquad D_y(h)=v_y^\top M_yv_y=\|L_y^{-1}v_y\|_2^2.$$

Ở đây $\eta_y$ là jitter thực dùng cho lớp $y$, $L_y$ là factor Cholesky tam giác dưới của $A_y$, không phải của $M_y$. Khi $\eta_y=0$, công thức trở về score dùng $R_y^{-1}$ ở trên.

### 1.8. Hai cách lưu covariance

| Biến thể | Thống kê lưu | Bộ nhớ thống kê theo bậc lớn |
| --- | --- | --- |
| Per-class, phạm vi bản đầu | Một prototype và metric mỗi lớp | $O(Cd+Cd^2)$ |
| Common covariance | Prototype mỗi lớp và một covariance chung | $O(Cd+d^2)$ |

Trong bảng bộ nhớ và các shape/pseudocode, $C=C_t$ là số lớp đã học tại thời điểm đang xét. Công thức toán học dùng task $t=1,\ldots,T_{\mathrm{sess}}$; nếu repo đánh số từ 0 thì `task_idx=t-1`.

Eq. (6) cập nhật covariance chung theo số lớp, với $t\ge2$ và khởi tạo $S_{1:1}=S_1$:

$$S_{1:t}=\frac{C_{t-1}}{C_t}S_{1:t-1}+\frac{K_t}{C_t}S_t,$$

trong đó $S_t$ là covariance chung của task hiện tại, $S_{1:t}$ là covariance chung tích lũy, $K_t$ là số lớp mới và $C_t=C_{t-1}+K_t$. Quy ước $C_0=0$. Đây không phải mặc nhiên là cập nhật theo số mẫu. Cần đối chiếu cách tác giả xây dựng $S_t$ (trung bình covariance từng lớp hay covariance gộp đặc trưng task). Hai cách không tương đương vì covariance gộp có biến thiên giữa các tâm lớp. Bản đầu **chỉ triển khai per-class**, nên điểm này không chặn việc cài đặt.

## 2. Bảng ký hiệu và giao diện đề xuất

### 2.1. Ký hiệu dùng thống nhất trong repo

| Ký hiệu | Ý nghĩa | Shape/kiểu |
| --- | --- | --- |
| $t$, $T_{\mathrm{sess}}$ | Chỉ số task trong toán học từ 1 và tổng số session (`num_sessions`); `task_idx=t-1` nếu code đánh số từ 0 | int |
| $C$ | Viết gọn cho $C_t$ trong bộ nhớ, shape và pseudocode | int |
| $d$, $N$ | Số chiều feature và số mẫu train trong một lần `update` | int |
| $i$, $j$, $k$ | Chỉ số mẫu và hai chỉ số chiều feature ($1\le j,k\le d$) | int |
| $y$, $\hat y$ | ID lớp đang xét và ID lớp dự đoán | scalar |
| $\mathcal C_t$, $C_t$ | Tập lớp đã học và số lớp đã học | set, int |
| $K_t$ | Số lớp mới ở task hiện tại | int |
| $x_i$ | Ảnh đầu vào | Do pipeline quy định |
| $\phi$ | Backbone cố định khi cập nhật classifier | Model |
| $h_i=\phi(x_i)$ | Đặc trưng gốc, chưa Tukey | `[d]` |
| $H$ | Đặc trưng train của task hiện tại | `[N, d]` |
| $y_i$, `labels` | Nhãn toàn cục của mẫu | scalar, `[N]` |
| $n_y$ | Số mẫu train của lớp $y$ | int |
| $\mu_y$ | Trung bình đặc trưng gốc | `[d]` |
| $T$, $\lambda$ | Phép biến đổi feature theo protocol và số mũ Tukey khi bật biến đổi | function, scalar |
| $\tilde\phi(x)=T(\phi(x))$ | Backbone kết hợp phép biến đổi feature | `[d]` |
| $z_i=T(h_i)$, $\bar z_y$ | Feature sau biến đổi và trung bình của chúng trong lớp $y$ | `[d]` |
| $\tilde\mu_y$ | Prototype dùng khi chấm điểm: $T(\mu_y)$ | `[d]` |
| $S_y$, $S_{y,s}$, $R_y$ | Covariance của feature sau biến đổi (trước shrinkage), sau shrinkage, sau correlation normalization | `[d, d]` |
| $S_t$, $S_{1:t}$ | Covariance chung của task $t$ và covariance chung tích lũy đến task $t$ | `[d, d]` |
| $I$, $J$ | Ma trận đơn vị và ma trận toàn số 1 | `[d, d]` |
| $V_{1,y}$, $V_{2,y}$ | Trung bình đường chéo và ngoài đường chéo của $S_y$ | scalar |
| $\gamma_1$, $\gamma_2$ | Hệ số shrinkage | scalar |
| $\eta_y$ | Jitter thực dùng cho lớp $y$, lưu trong `jitter_used` | scalar |
| $A_y$, $M_y$ | Ma trận sau jitter $A_y=R_y+\eta_yI$ và metric $M_y=A_y^{-1}$ | `[d, d]` |
| $L_y$ | Factor Cholesky tam giác dưới: $L_yL_y^\top=A_y$ | `[d, d]` |
| $v_y$, $D_y(h)$ | Độ lệch $T(h)-\tilde\mu_y$ và bình phương khoảng cách đến lớp $y$ | `[d]`, scalar |
| $B$ | Số query trong một batch dự đoán | int |
| $D$ | Ma trận bình phương khoảng cách; cột ứng với `class_ids`, càng nhỏ càng tốt | `[B, C]` |

Paper dùng $D$ cho chiều đặc trưng và $Y$ trong một số công thức cho số lớp; repo dùng `d` và `C` để tránh nhầm nhãn với số lớp. Trong công thức Gaussian của paper, $x$ còn được dùng cho feature, tài liệu này dùng $h$ để phân biệt với ảnh.

### 2.2. Khang: đầu vào

```python
update(features: Tensor[N, d], labels: LongTensor[N]) -> None
```

- `features` là feature gốc do backbone trả ra, chưa Tukey, chưa L2-normalize ở phía classifier. TV2 ghi rõ layer lấy feature, checkpoint và preprocessing. Không thay đổi backbone, layer hay preprocessing giữa các task.
- Bản đầu nhận tensor thực hữu hạn; dùng `float64` nội bộ cho thống kê/giải hệ để dễ kiểm tra số học. Tối ưu `float32` là thay đổi có kiểm chứng sau.
- `labels` dùng ID lớp toàn cục kiểu `int64`, không reset về `0..K_t-1` ở mỗi task.
- Một lần `update` nhận **toàn bộ feature train của các lớp mới trong task**, không phải một minibatch bất kỳ. TV2 có thể trích theo minibatch rồi ghép tạm trong task; classifier không lưu feature mẫu sau cập nhật.
- Mỗi lớp có ít nhất hai mẫu trong bản đầu. Lớp chỉ một mẫu phải báo lỗi; hỗ trợ one-shot cần quy tắc ước lượng covariance riêng.
- Gọi lại với một lớp đã có phải báo lỗi. Bản đầu không hỗ trợ cập nhật lại lớp cũ hoặc domain-incremental learning.
- Kiểm tra shape, `N > 0`, `d > 0`, số nhãn khớp số dòng, feature hữu hạn và cùng `d` với trạng thái hiện tại.

**Ánh xạ nhãn:** TV2 quản lý `original_dataset_id -> global_id` trong metadata. Classifier quản lý `global_id -> column_index`. Ví dụ task đầu có global ID `[42, 7]`, sắp lớp mới tăng dần thành `[7, 42]`; task sau thêm `19` thì `class_ids=[7,42,19]`. Cột 2 của score ứng với nhãn 19, không phải nhãn 2. Giữ nguyên vị trí lớp cũ.

### 2.3. Nhi: đầu ra

```python
distances(features: Tensor[B, d]) -> Tensor[B, C]
predict(features: Tensor[B, d]) -> LongTensor[B]
class_ids: LongTensor[C]
```

- `distances` nhận feature gốc cùng định nghĩa như `update`; classifier tự thực hiện biến đổi.
- `predict` trả **global ID**, bằng `class_ids[argmin(distances, dim=1)]`.
- Khoảng cách không phải xác suất. Nếu TV3 cần score càng lớn càng tốt, dùng `-distances` và đặt tên rõ.
- Không nhận `task_id` hoặc nhãn thật trong `predict`; luôn so sánh tất cả lớp đã học.
- Dự đoán trước `update` đầu tiên phải báo lỗi. Batch query rỗng được trả tensor rỗng đúng shape. Nếu bằng điểm, chọn cột đầu theo `class_ids` để tái lập được.
- TV3 tính accuracy trên test của toàn bộ lớp đã học. Average incremental accuracy là trung bình accuracy qua các session, **bao gồm session đầu** (supplement Eq. 13). Báo cáo cả final accuracy.
- Dùng `num_sessions` cho tổng số session; đừng nhầm setting “50 lớp đầu + 5 IL tasks” thành tổng 5 session: trường hợp đó có 6 session.

### 2.4. Cấu hình tối thiểu đề xuất

```yaml
classifier:
  name: fecam
  covariance_mode: per_class
  transform: sqrt           # none cho protocol ViT tương ứng
  prototype_mode: transform_of_raw_mean
  gamma1: 1.0              # ví dụ many-shot ResNet, không dùng chung mọi setting
  gamma2: 1.0
  covariance_ddof: 1
  stats_dtype: float64
  jitter_schedule: [0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2]
```

Danh sách jitter là **biện pháp ổn định số học do nhóm đề xuất**, không phải hyperparameter được trích từ paper. Không dùng validation/test để tự chọn jitter; thử lần lượt khi factorization thất bại và ghi giá trị thực dùng.

## 3. Đặc tả chức năng và pseudocode

### 3.1. Trạng thái cần lưu

| Trạng thái | Mục đích |
| --- | --- |
| `class_ids[C]`, `id_to_column` | Ánh xạ lớp toàn cục với cột score |
| `raw_prototypes[C,d]` | $\mu_y$, tiện đối chiếu/debug |
| `score_prototypes[C,d]` | $T(\mu_y)$, dùng dự đoán |
| `factors[C,d,d]` | Cholesky factor, đủ để tính khoảng cách |
| `counts[C]` | Số mẫu từng lớp, kiểm tra protocol |
| `d`, cấu hình, `jitter_used[C]` | Kiểm tra tương thích và tái lập |

Không bắt buộc lưu đồng thời covariance, inverse và factor. Bản đầu lưu factor; có thể xuất covariance tạm cho debug nhưng phải tính vào bộ nhớ nếu giữ lâu dài. Không lưu ảnh hoặc feature từng mẫu lớp cũ trong trạng thái classifier. Cache feature cũ chỉ dùng cho kiểm tra offline phải tách rõ khỏi learner và không được đọc lại khi cập nhật task mới.

### 3.2. Hàm hỗ trợ ổn định số học

Trong hàm xử lý một lớp, các biến cục bộ `S`, `V1`, `V2`, `S_shrunk`, `R`, `L`, `eta` lần lượt ứng với $S_y,V_{1,y},V_{2,y},S_{y,s},R_y,L_y,\eta_y$. `transform` trong pseudocode chính là $T$. Các ví dụ tính tay cũng lược chỉ số lớp khi không gây nhầm lẫn.

```text
build_factor(S, gamma1, gamma2, class_id):
    S = (S + transpose(S)) / 2
    V1 = mean(diagonal(S))
    V2 = mean(off_diagonal_entries(S)) if d > 1 else 0
    S_shrunk = S + gamma1 * V1 * I + gamma2 * V2 * (J - I)
    if any diagonal(S_shrunk) <= 0 or any value is nonfinite:
        raise NumericalError with class_id and diagnostic statistics

    s = sqrt(diagonal(S_shrunk))
    R = S_shrunk / outer(s, s)
    R = (R + transpose(R)) / 2
    for eta in configured_jitter_schedule:
        try Cholesky(R + eta * I):
            return factor, eta
    raise NumericalError; do not silently switch to pseudoinverse
```

Jitter làm metric thực tế khác Eq. (10) một lượng đã biết, nên phải log. Nếu gặp lớp có feature hằng hoặc phương sai bằng 0, bản đầu báo lỗi rõ thay vì âm thầm tạo covariance giả. Xử lý trường hợp đó sẽ cần một chính sách được ghi trong cấu hình sau.

### 3.3. Cập nhật lớp mới

```text
update(H, labels):
    validate shape, finite values, feature dimension and transform domain
    new_ids = sorted(unique(labels))
    require intersection(new_ids, existing_class_ids) is empty
    temporary_state = []

    for class_id in new_ids:
        H_y = H[labels == class_id]
        require number_of_rows(H_y) >= 2
        mu = mean(H_y, axis=0)
        Z_y = transform(H_y)
        score_mu = transform(mu)
        centered = Z_y - mean(Z_y, axis=0)
        S = transpose(centered) @ centered / (number_of_rows(H_y) - 1)
        L, eta = build_factor(S, gamma1, gamma2, class_id)
        append (class_id, mu, score_mu, L, number_of_rows(H_y), eta) to temporary_state

    append temporary_state to persistent state only after all classes succeed
    discard per-sample features and temporary covariance matrices
    return None
```

Cập nhật theo kiểu “tất cả thành công mới ghi” tránh trạng thái nửa chừng nếu một lớp lỗi. Thống kê lớp cũ giữ nguyên; không tính lại từ dữ liệu cũ và không cần lưu exemplar.

### 3.4. Tính khoảng cách và dự đoán

```text
distances(H_query):
    require at least one known class
    validate H_query and transform domain
    Z_query = transform(H_query)
    allocate D with shape [B, C]
    for column in range(C):
        V = Z_query - score_prototypes[column]       # [B, d]
        U = solve_lower_triangular(factors[column], transpose(V))
        D[:, column] = sum(U * U, axis=0)
    require all scores are finite
    return D

predict(H_query):
    D = distances(H_query)
    columns = argmin(D, axis=1)
    return class_ids[columns]
```

Không tạo tensor `[B,C,d,d]`; tính theo lớp/batch để giảm bộ nhớ tạm. Dense covariance cần khoảng $O(n_y d^2)$ để ước lượng mỗi lớp, Cholesky khoảng $O(d^3)$ mỗi lớp và dự đoán dense khoảng $O(BCd^2)$. Đây là lý do nghiên cứu nén covariance có ý nghĩa, nhưng chưa thuộc baseline tuần 1.

## 4. Toy example để tự giải thích và kiểm tra

Tắt Tukey (`transform=none`), chọn $\gamma_1=1,\gamma_2=0$ để dễ tính tay. Đây là cấu hình minh họa, không phải kết quả benchmark.

- Lớp global ID 7: $(1,1),(1,2),(2,1)$.
- Lớp global ID 42: $(5,5),(5,6),(6,5)$.

Prototype lần lượt là $(4/3,4/3)$ và $(16/3,16/3)$. Cả hai có sample covariance:

$$S=\begin{bmatrix}1/3&-1/6\\-1/6&1/3\end{bmatrix},\quad
S_s=\begin{bmatrix}2/3&-1/6\\-1/6&2/3\end{bmatrix},\quad
R=\begin{bmatrix}1&-1/4\\-1/4&1\end{bmatrix}.$$

Với query $(1,1)$, khoảng cách bình phương đến lớp 7 là $8/27\approx0.2963$, đến lớp 42 là $1352/27\approx50.0741$. Vì thế dự đoán global ID 7. Ví dụ này kiểm tra được prototype, mẫu số covariance, shrinkage, correlation normalization và ánh xạ nhãn trong cùng một bài tính nhỏ.

Khi cài đặt, cần thêm trường hợp Tukey để phát hiện lỗi thứ tự biến đổi; không chỉ chạy toy `none` rồi kết luận toàn bộ pipeline đúng.

## 5. Điểm chưa rõ và việc cần xác nhận

| Vấn đề | Hiện đã biết / lựa chọn đề xuất | Cách xử lý trước khi tuyên bố tái lập |
| --- | --- | --- |
| Prototype qua Tukey | Algorithm 1 tính mean gốc; Eq. 10 dùng prototype đã biến đổi | Đối chiếu mã tác giả; giữ `prototype_mode` rõ ràng |
| Feature/prototype L2 normalization | Phần method được đọc mô tả rõ Tukey và correlation normalization, chưa đủ để kết luận mọi chi tiết L2 trong code | Kiểm tra nguồn tham khảo và ghi thứ tự nếu có; không nhầm ba phép biến đổi |
| Mẫu số covariance | Đề xuất $n_y-1$ | Đối chiếu estimator của implementation |
| Covariance chung $S_t$ | Eq. 6 cho trọng số theo số lớp, mô tả cách tính $S_t$ cần làm rõ | Xác nhận trước khi thêm common baseline |
| Suy biến/không dương xác định | Có thể xảy ra dù đã shrinkage | Log chẩn đoán/jitter; không đổi metric âm thầm |
| Setting chạy chính | Tham số khác nhau theo backbone/dataset | TV2 chốt protocol và checkpoint, ghi vào config |
| Nhãn và cột score | Đã có quy ước đề xuất | TV2/TV3 review ví dụ `[7,42,19]` |
| Diễn giải Bayes | Khoảng cách đơn thuần không phải Gaussian Bayes đầy đủ trong mọi trường hợp | Viết đúng điều kiện trong báo cáo; baseline vẫn theo Eq. 10 |

## 6. Checklist bàn giao ba công việc tuần 1

- [x] Giải thích pipeline, prototype, covariance, Tukey, shrinkage, normalization và Mahalanobis bằng lời của nhóm.
- [x] Bảng ký hiệu, shape, quy ước nhãn và đầu vào–đầu ra đề xuất.
- [x] Đặc tả cập nhật/dự đoán, thống kê lưu và pseudocode.
- [x] Ghi riêng điểm chưa rõ và phân biệt nội dung paper với lựa chọn cài đặt.
- [ ] TV2 xác nhận feature layer, preprocessing, backbone, nhãn và cách gom feature theo task.
- [ ] TV3 xác nhận `class_ids`, chiều score, accuracy trên toàn bộ lớp đã học và cách đếm session.
- [ ] Phú đối chiếu các điểm chưa rõ với mã tham khảo rồi cập nhật tài liệu có dẫn vị trí nguồn.

Ba công việc trong tài liệu đã được soạn; các checkbox chưa đánh dấu là xác nhận/tích hợp thực tế còn cần làm, không phải kết quả đã có. Việc viết `src/models/fecam.py` và chạy kiểm thử là bước tiếp theo, ngoài phạm vi ba đầu việc tài liệu này.

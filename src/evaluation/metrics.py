"""
metrics.py — Continual Learning Evaluation Metrics
====================================================
Triển khai đầy đủ các metrics chuẩn cho Class-Incremental Learning (CIL):

  - Accuracy tại task t : A_t
  - Average Incremental Accuracy : Avg_Acc  (metrics chính của bài báo)
  - Backward Transfer (BWT)      : đo mức độ quên
  - Forward Transfer (FWT)       : NaN nếu protocol không đo baseline tương lai
  - Last Accuracy                : A_T (accuracy sau task cuối cùng)
  - Plasticity / Stability       : dẫn xuất từ BWT/FWT

Avg Acc is the primary reported metric. FWT is left undefined without an
explicit pre-learning baseline for future tasks.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Hàm tiện ích cơ bản
# ---------------------------------------------------------------------------

def accuracy_at_task(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Top-1 accuracy (%) tại một task/session cụ thể.

    Args:
        y_pred: mảng nhãn dự đoán, shape (N,).
        y_true: mảng nhãn thật,    shape (N,).

    Returns:
        Độ chính xác (%) dạng float.
    """
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_pred == y_true) * 100.0)


def topk_accuracy(scores: np.ndarray, y_true: np.ndarray, k: int = 5) -> float:
    """Top-k accuracy (%).

    Args:
        scores: ma trận logit/điểm số, shape (N, C).
        y_true: nhãn thật, shape (N,).
        k     : số lượng dự đoán hàng đầu.

    Returns:
        Top-k accuracy (%).
    """
    topk_preds = np.argsort(scores, axis=1)[:, -k:]          # (N, k)
    correct = np.any(topk_preds == y_true[:, None], axis=1)  # (N,)
    return float(np.mean(correct) * 100.0)


# ---------------------------------------------------------------------------
# Continual Learning Metrics chính
# ---------------------------------------------------------------------------

class CLMetrics:
    """Tính toán và lưu trữ toàn bộ metrics CIL qua nhiều tasks.

    Cách dùng:
        m = CLMetrics(n_tasks=10)
        for task_id in range(10):
            ...train/eval...
            m.update(task_id, acc_matrix_row)   # sau mỗi task
        summary = m.compute()
    """

    def __init__(self, n_tasks: int, task_sizes=None):
        """
        Args:
            n_tasks: Tổng số tasks (sessions) trong kịch bản CIL.
        """
        self.n_tasks = n_tasks
        self.task_sizes = np.asarray(
            task_sizes if task_sizes is not None else np.ones(n_tasks), dtype=float
        )
        if self.task_sizes.shape != (n_tasks,) or np.any(self.task_sizes <= 0):
            raise ValueError("task_sizes must contain one positive size per task")
        # R[i][j] = accuracy trên test-set của task j sau khi học xong task i
        # i >= j  (không đánh giá task tương lai)
        self.acc_matrix = np.full((n_tasks, n_tasks), fill_value=np.nan)

    # ------------------------------------------------------------------
    def update(self, current_task: int, accs_per_seen_task: list):
        """Cập nhật ma trận accuracy sau khi học xong task `current_task`.

        Args:
            current_task       : chỉ số task vừa học (0-indexed).
            accs_per_seen_task : list/array độ chính xác [A(t,0), A(t,1), ..., A(t,t)]
                                 trên từng task đã học tính đến hiện tại.
        """
        if not 0 <= current_task < self.n_tasks:
            raise IndexError("current_task is outside the configured task range")
        if len(accs_per_seen_task) != current_task + 1:
            raise ValueError("provide one accuracy for each task seen so far")
        for j, acc in enumerate(accs_per_seen_task):
            self.acc_matrix[current_task][j] = acc

    # ------------------------------------------------------------------
    def compute(self) -> dict:
        """Tính toán tất cả metrics sau khi hoàn thành toàn bộ T tasks.

        Returns:
            dict chứa:
              - avg_acc   : Average Incremental Accuracy (%)  ← metrics chính
              - last_acc  : Accuracy sau task cuối (%)
              - bwt       : Backward Transfer (%)
              - fwt       : Forward Transfer (%) — NaN nếu thiếu zero-shot acc
              - acc_per_task : list accuracy trên từng task sau task cuối
              - plasticity: mean accuracy trên task mới nhất tại thời điểm học
              - stability : mean accuracy giữ lại các task cũ
        """
        T = self.n_tasks
        matrix = self.acc_matrix

        # ── Average Incremental Accuracy ──────────────────────────────
        # Avg_Acc = (1/T) * Σ_{t=1}^{T} A_t   với A_t = average acc trên t tasks
        incremental_accs = []
        for t in range(T):
            row = matrix[t, : t + 1]
            valid = row[~np.isnan(row)]
            if len(valid) > 0:
                weights = self.task_sizes[: t + 1][~np.isnan(row)]
                incremental_accs.append(float(np.average(valid, weights=weights)))
        avg_acc = float(np.mean(incremental_accs)) if incremental_accs else np.nan

        # ── Last Accuracy ─────────────────────────────────────────────
        last_row = matrix[T - 1, :]
        valid_last = last_row[~np.isnan(last_row)]
        last_weights = self.task_sizes[~np.isnan(last_row)]
        last_acc = (float(np.average(valid_last, weights=last_weights))
                    if len(valid_last) > 0 else np.nan)

        # ── Backward Transfer (BWT) ───────────────────────────────────
        # BWT = (1/(T-1)) * Σ_{j=0}^{T-2} [R(T,j) - R(j,j)]
        # Âm → quên; dương → backward positive transfer
        if T >= 2:
            bwt_terms = []
            for j in range(T - 1):
                r_T_j = matrix[T - 1, j]
                r_j_j = matrix[j, j]
                if not (np.isnan(r_T_j) or np.isnan(r_j_j)):
                    bwt_terms.append(r_T_j - r_j_j)
            bwt = float(np.mean(bwt_terms)) if bwt_terms else np.nan
        else:
            bwt = np.nan

        # FWT requires an explicit pre-learning baseline for each future task.
        # This protocol does not measure those baselines, so leave it undefined.
        fwt = np.nan

        # ── Per-task accuracy ─────────────────────────────────────────
        acc_per_task = [
            float(matrix[T - 1, j]) if not np.isnan(matrix[T - 1, j]) else np.nan
            for j in range(T)
        ]

        # ── Plasticity (accuracy trên task vừa học) ───────────────────
        plasticity_list = [float(matrix[t, t]) for t in range(T) if not np.isnan(matrix[t, t])]
        plasticity = float(np.mean(plasticity_list)) if plasticity_list else np.nan

        # ── Stability (avg acc giữ lại old tasks) ─────────────────────
        stability_list = []
        for t in range(1, T):
            for j in range(t):
                val = matrix[t, j]
                if not np.isnan(val):
                    stability_list.append(val)
        stability = float(np.mean(stability_list)) if stability_list else np.nan

        return {
            "avg_acc"     : avg_acc,
            "last_acc"    : last_acc,
            "bwt"         : bwt,
            "fwt"         : fwt,
            "acc_per_task": acc_per_task,
            "plasticity"  : plasticity,
            "stability"   : stability,
        }

    # ------------------------------------------------------------------
    def get_acc_matrix(self) -> np.ndarray:
        """Trả về ma trận accuracy đầy đủ (T x T)."""
        return self.acc_matrix.copy()


# ---------------------------------------------------------------------------
# Hàm helper độc lập (tương thích ngược với code cũ)
# ---------------------------------------------------------------------------

def compute_avg_acc(acc_list: list) -> float:
    """Average Incremental Accuracy từ list accuracy theo từng session.

    Args:
        acc_list: [A_0, A_1, ..., A_{T-1}]  — accuracy tổng thể mỗi session.

    Returns:
        Avg_Acc (%).
    """
    return float(np.mean(acc_list)) if acc_list else 0.0


def compute_bwt(acc_matrix: np.ndarray) -> float:
    """Backward Transfer từ ma trận accuracy (T x T).

    Args:
        acc_matrix: R[i][j] = accuracy task j sau khi học xong task i.

    Returns:
        BWT (%). Âm → mô hình bị quên.
    """
    T = acc_matrix.shape[0]
    if T < 2:
        return 0.0
    terms = []
    for j in range(T - 1):
        r_Tj = acc_matrix[T - 1, j]
        r_jj = acc_matrix[j, j]
        if not (np.isnan(r_Tj) or np.isnan(r_jj)):
            terms.append(r_Tj - r_jj)
    return float(np.mean(terms)) if terms else 0.0


def compute_forgetting(acc_matrix: np.ndarray) -> float:
    """Average Forgetting (dương → quên nhiều hơn BWT âm).

    Args:
        acc_matrix: R[i][j] = accuracy task j sau khi học xong task i.

    Returns:
        Forgetting (%).
    """
    return -compute_bwt(acc_matrix)


def calculate_accuracy(predictions, true_labels) -> float:
    """Tương thích ngược: tính accuracy (%).

    Args:
        predictions : Tensor hoặc ndarray shape (N,).
        true_labels : Tensor hoặc ndarray shape (N,).

    Returns:
        Accuracy (%).
    """
    import torch
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if len(true_labels) == 0:
        return 0.0
    return float(np.mean(predictions == true_labels) * 100.0)

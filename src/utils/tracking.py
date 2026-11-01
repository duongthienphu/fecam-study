"""
tracking.py — Weights & Biases Integration cho FeCAM Study
===========================================================
Module này đóng gói toàn bộ logic W&B:
  - Khởi tạo run với config chuẩn
  - Log metrics mỗi task (Acc_t, Avg_Acc, BWT, FWT, ...)
  - Log bảng kết quả cuối (summary table)
  - Log confusion matrix và accuracy heatmap
  - Graceful fallback khi W&B không khả dụng (offline / dry-run)

Cách dùng:
    tracker = WandbTracker(config, project="fecam-study", run_name="ncm-cifar100")
    tracker.start()
    for task_id in range(n_tasks):
        ...train/eval...
        tracker.log_task(task_id, acc=82.3, avg_acc=85.1, bwt=-1.2)
    tracker.log_summary(metrics_dict)
    tracker.finish()
"""

import os
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Thử import wandb; nếu chưa cài → chạy ở chế độ offline
try:
    import wandb

    # Nếu chưa login (ví dụ server timeout), thử dùng key từ env
    # Chạy: $env:WANDB_API_KEY = "your_key"  hoặc ghi vào .env
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    logger.warning(
        "⚠️  wandb chưa được cài đặt. Chạy ở chế độ offline (chỉ log ra console).\n"
        "    Cài đặt bằng: pip install wandb"
    )


# ---------------------------------------------------------------------------
# WandbTracker
# ---------------------------------------------------------------------------

class WandbTracker:
    """Wrapper W&B cho thực nghiệm FeCAM / NCM Continual Learning.

    Attributes:
        run    : wandb.Run object (None nếu offline).
        config : dict cấu hình thực nghiệm.
    """

    def __init__(
        self,
        config: dict,
        project: str = "fecam-study",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        mode: str = "online",          # "online" | "offline" | "disabled"
        dry_run: bool = False,
    ):
        """
        Args:
            config   : dict cấu hình (dataset, model, hyperparams, ...).
            project  : tên project trên W&B (phải là Public).
            entity   : W&B username hoặc team (None → dùng default).
            run_name : tên run trên W&B. Nếu None → W&B tự sinh.
            tags     : list tags để filter trên W&B UI.
            notes    : mô tả ngắn về run.
            mode     : "online" để đẩy lên cloud, "offline" để lưu local,
                       "disabled" để tắt hoàn toàn (không log gì).
            dry_run  : True → chạy như offline dù mode=online (test nhanh).
        """
        self.config   = config
        self.project  = project
        self.entity   = entity
        self.run_name = run_name
        self.tags     = tags or []
        self.notes    = notes
        self.mode     = "disabled" if dry_run else mode
        self.run      = None
        self._offline = not _WANDB_AVAILABLE or dry_run or (mode == "disabled")

        # Buffer để log ngay cả khi W&B không khả dụng
        self._log_buffer: list[dict] = []

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Khởi tạo W&B run. Gọi trước vòng lặp training."""
        if self._offline or not _WANDB_AVAILABLE:
            logger.info("📋 [W&B Offline] Run bắt đầu (không kết nối cloud).")
            return

        try:
            self.run = wandb.init(
                project  = self.project,
                entity   = self.entity,
                name     = self.run_name,
                config   = self.config,
                tags     = self.tags,
                notes    = self.notes,
                mode     = self.mode,
                settings = wandb.Settings(
                    init_timeout = 30,   # giảm timeout để fail nhanh
                ),
            )
            logger.info(f"🚀 [W&B] Run khởi tạo: {self.run.url}")

        except Exception as e:
            # Server unreachable → tự động chuyển sang offline mode
            logger.warning(f"⚠️  [W&B] Không kết nối được server ({type(e).__name__}): {e}")
            logger.warning("📋 [W&B] Tự động chuyển sang OFFLINE mode.")
            logger.warning("   Sau khi có mạng, sync bằng: wandb sync wandb/")
            try:
                self.run = wandb.init(
                    project = self.project,
                    name    = self.run_name,
                    config  = self.config,
                    tags    = self.tags,
                    mode    = "offline",   # lưu local, sync sau
                )
                logger.info("✅ [W&B] Offline run khởi tạo thành công.")
            except Exception as e2:
                logger.warning(f"⚠️  [W&B] Offline init cũng thất bại: {e2}. Tiếp tục không có W&B.")
                self.run = None
                self._offline = True

    # ------------------------------------------------------------------
    def log_task(
        self,
        task_id: int,
        acc: float,
        avg_acc: Optional[float] = None,
        bwt: Optional[float] = None,
        fwt: Optional[float] = None,
        plasticity: Optional[float] = None,
        stability: Optional[float] = None,
        n_classes_seen: Optional[int] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Log metrics sau mỗi task/session.

        Args:
            task_id        : chỉ số task (0-indexed).
            acc            : accuracy (%) tổng thể tại task này.
            avg_acc        : Average Incremental Accuracy tính đến task này.
            bwt            : Backward Transfer (%).
            fwt            : Forward Transfer (%).
            plasticity     : accuracy trên task mới nhất.
            stability      : accuracy trung bình giữ lại task cũ.
            n_classes_seen : số class đã thấy tính đến task này.
            extra          : dict các metric phụ bổ sung.
        """
        log_dict = {
            "task"              : task_id,
            "acc_all"           : acc,
        }
        if avg_acc       is not None: log_dict["avg_acc"]       = avg_acc
        if bwt           is not None: log_dict["bwt"]           = bwt
        if fwt           is not None: log_dict["fwt"]           = fwt
        if plasticity    is not None: log_dict["plasticity"]    = plasticity
        if stability     is not None: log_dict["stability"]     = stability
        if n_classes_seen is not None: log_dict["n_classes_seen"] = n_classes_seen
        if extra:
            log_dict.update(extra)

        self._log_buffer.append(log_dict)

        # Console log
        parts = [f"[Task {task_id:02d}] acc={acc:.2f}%"]
        if avg_acc is not None: parts.append(f"avg_acc={avg_acc:.2f}%")
        if bwt     is not None: parts.append(f"bwt={bwt:+.2f}%")
        logger.info("📊 " + "  ".join(parts))

        if self.run is not None:
            wandb.log(log_dict, step=task_id)

    # ------------------------------------------------------------------
    def log_acc_matrix(self, acc_matrix: np.ndarray) -> None:
        """Log ma trận accuracy (T x T) lên W&B dưới dạng heatmap.

        Args:
            acc_matrix: shape (T, T) — R[i][j] = acc task j sau task i.
        """
        if self.run is None:
            return
        try:
            import matplotlib.pyplot as plt
            T = acc_matrix.shape[0]
            fig, ax = plt.subplots(figsize=(max(6, T), max(5, T - 1)))
            im = ax.imshow(acc_matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
            ax.set_xlabel("Task j (được đánh giá)")
            ax.set_ylabel("Task i (đã học xong)")
            ax.set_title("Accuracy Matrix R[i][j]")
            plt.colorbar(im, ax=ax, label="Accuracy (%)")
            for i in range(T):
                for j in range(T):
                    val = acc_matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                                fontsize=7, color="black" if val > 50 else "white")
            plt.tight_layout()
            wandb.log({"acc_matrix_heatmap": wandb.Image(fig)})
            plt.close(fig)
            logger.info("📈 [W&B] Đã log accuracy matrix heatmap.")
        except Exception as e:
            logger.warning(f"⚠️  Không thể log acc matrix: {e}")

    # ------------------------------------------------------------------
    def log_summary(self, metrics: dict) -> None:
        """Log tổng kết cuối run (ghi vào W&B summary).

        Args:
            metrics: dict chứa avg_acc, last_acc, bwt, fwt, ...
        """
        summary_log = {
            f"final/{k}": v for k, v in metrics.items()
            if not isinstance(v, (list, np.ndarray))
            and not (isinstance(v, (float, np.floating)) and np.isnan(v))
        }

        logger.info("=" * 50)
        logger.info("🏆 KẾT QUẢ TỔNG KẾT:")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"   {k:<20} = {v:+.2f}%")
        logger.info("=" * 50)

        if self.run is not None:
            wandb.log(summary_log)
            # Ghi vào W&B run summary (hiện trên dashboard)
            for k, v in summary_log.items():
                wandb.run.summary[k] = v
            logger.info("✅ [W&B] Đã sync kết quả tổng kết.")

    # ------------------------------------------------------------------
    def log_table(self, data: list[dict], table_name: str = "results") -> None:
        """Log bảng kết quả chi tiết lên W&B.

        Args:
            data      : list các dict, mỗi dict là 1 dòng.
            table_name: tên artifact table trên W&B.
        """
        if not data:
            return
        if self.run is not None:
            columns = list(data[0].keys())
            rows    = [[row.get(c, None) for c in columns] for row in data]
            table   = wandb.Table(columns=columns, data=rows)
            wandb.log({table_name: table})
            logger.info(f"📋 [W&B] Đã log bảng '{table_name}' ({len(data)} dòng).")

    # ------------------------------------------------------------------
    def finish(self) -> None:
        """Kết thúc run W&B."""
        if self.run is not None:
            self.run.finish()
            logger.info("✅ [W&B] Run đã kết thúc.")
        else:
            logger.info("📋 [W&B Offline] Run kết thúc (offline mode).")

    # ------------------------------------------------------------------
    def get_log_buffer(self) -> list:
        """Trả về toàn bộ metrics đã log (hữu ích khi offline)."""
        return self._log_buffer.copy()

    # ------------------------------------------------------------------
    @property
    def run_url(self) -> Optional[str]:
        """URL của W&B run (None nếu offline)."""
        return self.run.url if self.run is not None else None


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_tracker(
    config: dict,
    project: str = "fecam-study",
    run_name: Optional[str] = None,
    tags: Optional[list] = None,
    offline: bool = False,
) -> WandbTracker:
    """Tạo WandbTracker từ config dict.

    Args:
        config   : dict cấu hình (được ghi vào W&B config tab).
        project  : tên W&B project (nên là Public để khớp yêu cầu đồ án).
        run_name : tên run.
        tags     : danh sách tags.
        offline  : True → lưu run local để có thể sync lên cloud sau.

    Returns:
        WandbTracker instance (chưa start).
    """
    return WandbTracker(
        config   = config,
        project  = project,
        run_name = run_name,
        tags     = tags,
        mode     = "offline" if offline else "online",
    )

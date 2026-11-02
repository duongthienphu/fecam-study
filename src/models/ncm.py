"""
ncm.py — Nearest Class Mean (NCM) Baseline cho Class-Incremental Learning
==========================================================================
Cài đặt NCM classifier đầy đủ theo đúng pipeline của bài báo FeCAM:

  1. Trích xuất features từ backbone đóng băng (frozen pretrained ViT/ResNet).
  2. Tính prototype (mean vector) của mỗi class từ training set.
  3. Phân loại bằng khoảng cách: Euclidean hoặc Cosine.
  4. Tích hợp CL loop (incremental) để đánh giá qua nhiều tasks.

NCM là baseline quan trọng nhất:
  - Nó là "cận dưới" (lower bound) mà FeCAM phải vượt qua.
  - FeCAM mở rộng NCM bằng cách thay identity covariance → per-class covariance.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List


# ---------------------------------------------------------------------------
# Lớp NCMClassifier độc lập (không phụ thuộc BaseLearner)
# ---------------------------------------------------------------------------

class NCMClassifier:
    """Nearest Class Mean Classifier thuần túy.

    Hỗ trợ cả khoảng cách Euclidean và Cosine như bài báo.
    Class index được duy trì tự động khi thêm class mới (incremental).
    """

    def __init__(self, metric: str = "euclidean", normalize: bool = True):
        """
        Args:
            metric    : 'euclidean' | 'cosine' — khoảng cách phân loại.
            normalize : True → chuẩn hóa L2 features trước khi tính khoảng cách
                        (khuyến nghị khi dùng Cosine, và cũng cải thiện Euclidean).
        """
        assert metric in ("euclidean", "cosine"), \
            "metric phải là 'euclidean' hoặc 'cosine'"
        self.metric = metric
        self.normalize = normalize

        # prototypes[c] = vector trung bình của class c, shape (D,)
        self.prototypes: dict = {}        # {class_idx: np.ndarray}
        self.class_order: list = []       # thứ tự class đã thấy

    # ------------------------------------------------------------------
    # Thêm / cập nhật prototypes
    # ------------------------------------------------------------------

    def update_prototypes(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        new_classes: Optional[List[int]] = None,
    ) -> None:
        """Tính và lưu prototype cho các class trong `new_classes`.

        Args:
            features   : feature matrix, shape (N, D).
            labels     : nhãn tương ứng,  shape (N,).
            new_classes: danh sách class cần tính prototype.
                         Nếu None → tính cho tất cả class trong `labels`.
        """
        if new_classes is None:
            new_classes = sorted(set(labels.tolist()))

        for c in new_classes:
            mask = labels == c
            if mask.sum() == 0:
                continue
            proto = features[mask].mean(axis=0)          # (D,)
            self.prototypes[c] = proto
            if c not in self.class_order:
                self.class_order.append(c)

    def reset(self) -> None:
        """Xoá toàn bộ prototypes (dùng khi bắt đầu lại từ đầu)."""
        self.prototypes.clear()
        self.class_order.clear()

    # ------------------------------------------------------------------
    # Dự đoán
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn cho batch features.

        Args:
            features: shape (N, D).

        Returns:
            y_pred: shape (N,) — class index thực (không phải position trong list).
        """
        if not self.prototypes:
            raise RuntimeError("Chưa có prototype nào! Gọi update_prototypes() trước.")

        classes = sorted(self.prototypes.keys())          # [c0, c1, ...]
        proto_mat = np.stack([self.prototypes[c] for c in classes])  # (C, D)

        if self.normalize:
            norm_f = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            norm_p = proto_mat / (np.linalg.norm(proto_mat, axis=1, keepdims=True) + 1e-8)
        else:
            norm_f = features
            norm_p = proto_mat

        if self.metric == "euclidean":
            # khoảng cách L2 bình phương
            # (N, D) → (N, 1, D) - (1, C, D) → (N, C)
            diff = norm_f[:, None, :] - norm_p[None, :, :]
            dists = (diff ** 2).sum(axis=-1)              # (N, C)
        else:
            # cosine similarity → -similarity = khoảng cách
            sim = norm_f @ norm_p.T                       # (N, C)
            dists = -sim

        pred_indices = np.argmin(dists, axis=1)           # (N,)
        return np.array([classes[i] for i in pred_indices])

    def predict_with_scores(self, features: np.ndarray):
        """Trả về cả nhãn và ma trận khoảng cách (dùng cho Top-k eval).

        Returns:
            (y_pred, dist_matrix):
              y_pred      : shape (N,)
              dist_matrix : shape (N, C) — khoảng cách tới mỗi prototype
        """
        if not self.prototypes:
            raise RuntimeError("Chưa có prototype nào!")

        classes = sorted(self.prototypes.keys())
        proto_mat = np.stack([self.prototypes[c] for c in classes])

        if self.normalize:
            norm_f = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            norm_p = proto_mat / (np.linalg.norm(proto_mat, axis=1, keepdims=True) + 1e-8)
        else:
            norm_f, norm_p = features, proto_mat

        if self.metric == "euclidean":
            diff = norm_f[:, None, :] - norm_p[None, :, :]
            dists = (diff ** 2).sum(axis=-1)
        else:
            dists = -(norm_f @ norm_p.T)

        pred_indices = np.argmin(dists, axis=1)
        y_pred = np.array([classes[i] for i in pred_indices])
        return y_pred, dists


# ---------------------------------------------------------------------------
# Hàm helper tương thích ngược với code cũ (ncm.py gốc)
# ---------------------------------------------------------------------------

def compute_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_list: list,
) -> torch.Tensor:
    """Tính prototype (mean vector) cho từng class trong class_list.

    Args:
        features   : Tensor shape (N, D).
        labels     : Tensor shape (N,) — nhãn nguyên.
        class_list : list[int] — các class cần tính.

    Returns:
        Tensor shape (len(class_list), D).
    """
    prototypes = []
    for c in class_list:
        mask = labels == c
        if mask.sum() == 0:
            # Trả về zero vector nếu không có mẫu (tránh crash)
            prototypes.append(torch.zeros(features.shape[1]))
        else:
            prototypes.append(features[mask].mean(dim=0))
    return torch.stack(prototypes)


def eucl_ncm_predict(
    test_features: torch.Tensor,
    all_prototypes: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Dự đoán bằng khoảng cách Euclidean bình phương (NCM cơ bản).

    Args:
        test_features  : Tensor (N, D).
        all_prototypes : Tensor (C, D).
        normalize      : chuẩn hóa L2 trước khi tính khoảng cách.

    Returns:
        Tensor shape (N,) — class index dự đoán.
    """
    if normalize:
        test_features  = F.normalize(test_features,  p=2, dim=-1)
        all_prototypes = F.normalize(all_prototypes, p=2, dim=-1)
    distances = torch.cdist(test_features, all_prototypes, p=2) ** 2
    return torch.argmin(distances, dim=1)


def cosine_ncm_predict(
    test_features: torch.Tensor,
    all_prototypes: torch.Tensor,
) -> torch.Tensor:
    """Dự đoán bằng Cosine Similarity lớn nhất (tương đương Euclidean với L2-norm).

    Args:
        test_features  : Tensor (N, D).
        all_prototypes : Tensor (C, D).

    Returns:
        Tensor shape (N,) — class index dự đoán.
    """
    test_norm  = F.normalize(test_features,  p=2, dim=-1)   # (N, D)
    proto_norm = F.normalize(all_prototypes, p=2, dim=-1)   # (C, D)
    sim = test_norm @ proto_norm.T                           # (N, C)
    return torch.argmax(sim, dim=1)


# ---------------------------------------------------------------------------
# NCM Incremental Loop (tích hợp với extract_features.py)
# ---------------------------------------------------------------------------

class IncrementalNCM:
    """NCM Baseline runner theo đúng kịch bản CIL trong bài báo FeCAM.

    Pipeline:
      1. Backbone (ViT / ResNet) đóng băng, pretrained.
      2. Mỗi task → extract features → tính prototype → eval all seen classes.
      3. Tích hợp CLMetrics để log Avg_Acc, BWT, v.v.

    Dùng kèm với WandbTracker trong tracking.py để log lên W&B.
    """

    def __init__(self, backbone, device, metric: str = "euclidean", normalize: bool = True):
        """
        Args:
            backbone : mô hình đã load (output là feature vector, không có fc head).
            device   : torch.device.
            metric   : 'euclidean' | 'cosine'.
            normalize: chuẩn hóa L2 trước khi tính khoảng cách.
        """
        self.backbone  = backbone
        self.device    = device
        self.clf       = NCMClassifier(metric=metric, normalize=normalize)
        self.metric    = metric

    # ------------------------------------------------------------------
    def extract_features(self, loader) -> tuple:
        """Trích xuất toàn bộ features từ DataLoader.

        Returns:
            (features, labels) — np.ndarray.
        """
        self.backbone.eval()
        all_feats, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                # Hỗ trợ 2 format: (imgs, labels) hoặc (_, imgs, labels)
                if len(batch) == 3:
                    _, imgs, labels = batch
                else:
                    imgs, labels = batch

                imgs = imgs.to(self.device)
                feats = self._forward_features(imgs)
                all_feats.append(feats.cpu().numpy())
                all_labels.append(labels.numpy())

        return np.concatenate(all_feats), np.concatenate(all_labels)

    def _forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward qua backbone, trả về feature vector.
        Hỗ trợ ResNet (output trực tiếp), ViT (dùng forward_features), và dict output.
        """
        out = self.backbone(imgs)
        if isinstance(out, dict):
            # BaseLearner style: {"logits": ..., "features": ...}
            return out.get("features", out.get("logits"))
        if isinstance(out, torch.Tensor):
            return out
        # timm ViT: dùng forward_features → lấy CLS token
        feats = self.backbone.forward_features(imgs)
        return feats[:, 0] if feats.ndim == 3 else feats

    # ------------------------------------------------------------------
    def run_task(
        self,
        task_id: int,
        new_classes: list,
        train_loader,
        test_loader,
    ) -> dict:
        """Thực hiện 1 task: cập nhật prototypes và eval.

        Args:
            task_id     : chỉ số task (0-indexed).
            new_classes : list class id mới trong task này.
            train_loader: DataLoader cho training set của task hiện tại.
            test_loader : DataLoader cho test set của tất cả class đã thấy.

        Returns:
            dict {"acc": float, "y_pred": ndarray, "y_true": ndarray}
        """
        # 1. Trích xuất features tập train của task mới
        train_feats, train_labels = self.extract_features(train_loader)

        # 2. Cập nhật prototype cho các class mới
        self.clf.update_prototypes(train_feats, train_labels, new_classes=new_classes)

        # 3. Đánh giá trên test set của tất cả class đã thấy
        test_feats, test_labels = self.extract_features(test_loader)
        y_pred = self.clf.predict(test_feats)

        acc = float(np.mean(y_pred == test_labels) * 100.0)
        return {"acc": acc, "y_pred": y_pred, "y_true": test_labels}

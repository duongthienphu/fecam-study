"""
test_classifiers.py — Unit Tests cho src/models/ncm.py
=======================================================
Chạy: python -m pytest tests/test_classifiers.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import torch

from models.ncm import (
    NCMClassifier,
    compute_prototypes,
    eucl_ncm_predict,
    cosine_ncm_predict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_simple_data(n_classes=3, n_per_class=20, dim=10, seed=42):
    """Tạo dữ liệu linearly separable đơn giản cho unit test."""
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for c in range(n_classes):
        center = rng.standard_normal(dim) * 5  # các cluster cách xa nhau
        feats  = center + rng.standard_normal((n_per_class, dim)) * 0.1
        features.append(feats)
        labels.extend([c] * n_per_class)
    return np.concatenate(features), np.array(labels)


# ---------------------------------------------------------------------------
# NCMClassifier — Euclidean
# ---------------------------------------------------------------------------

class TestNCMClassifierEuclidean:
    def test_perfect_on_simple_data(self):
        feats, labels = make_simple_data()
        clf = NCMClassifier(metric="euclidean")
        clf.update_prototypes(feats, labels)
        y_pred = clf.predict(feats)
        acc = np.mean(y_pred == labels) * 100
        assert acc > 95.0, f"Accuracy {acc:.1f}% quá thấp trên dữ liệu đơn giản"

    def test_incremental_update(self):
        """Thêm class theo từng bước, mỗi lần không ảnh hưởng prototype cũ."""
        feats, labels = make_simple_data(n_classes=4)
        clf = NCMClassifier(metric="euclidean")

        # Task 0: class 0, 1
        mask_01 = labels < 2
        clf.update_prototypes(feats[mask_01], labels[mask_01], new_classes=[0, 1])
        assert set(clf.class_order) == {0, 1}

        # Task 1: class 2, 3
        mask_23 = labels >= 2
        clf.update_prototypes(feats[mask_23], labels[mask_23], new_classes=[2, 3])
        assert set(clf.class_order) == {0, 1, 2, 3}

        # Dự đoán toàn bộ
        y_pred = clf.predict(feats)
        acc = np.mean(y_pred == labels) * 100
        assert acc > 95.0

    def test_predict_returns_class_indices(self):
        """y_pred phải là class index thực (không phải vị trí trong list)."""
        feats, labels = make_simple_data(n_classes=3)
        clf = NCMClassifier(metric="euclidean")
        clf.update_prototypes(feats, labels)
        y_pred = clf.predict(feats)
        assert set(y_pred).issubset({0, 1, 2})

    def test_empty_class_no_crash(self):
        """update_prototypes với class không có mẫu → không crash."""
        feats  = np.random.randn(10, 5)
        labels = np.zeros(10, dtype=int)
        clf = NCMClassifier()
        clf.update_prototypes(feats, labels, new_classes=[0, 99])  # 99 không có mẫu
        assert 0 in clf.prototypes
        assert 99 not in clf.prototypes  # bỏ qua

    def test_no_prototype_raises(self):
        clf = NCMClassifier()
        with pytest.raises(RuntimeError, match="prototype"):
            clf.predict(np.random.randn(5, 10))

    def test_reset(self):
        feats, labels = make_simple_data(n_classes=2)
        clf = NCMClassifier()
        clf.update_prototypes(feats, labels)
        clf.reset()
        assert len(clf.prototypes) == 0

    def test_predict_with_scores_shape(self):
        feats, labels = make_simple_data(n_classes=3)
        clf = NCMClassifier(metric="euclidean")
        clf.update_prototypes(feats, labels)
        y_pred, dist_mat = clf.predict_with_scores(feats)
        assert y_pred.shape == (len(labels),)
        assert dist_mat.shape == (len(labels), 3)


# ---------------------------------------------------------------------------
# NCMClassifier — Cosine
# ---------------------------------------------------------------------------

class TestNCMClassifierCosine:
    def test_high_acc_on_simple_data(self):
        feats, labels = make_simple_data()
        clf = NCMClassifier(metric="cosine")
        clf.update_prototypes(feats, labels)
        y_pred = clf.predict(feats)
        acc = np.mean(y_pred == labels) * 100
        assert acc > 90.0


# ---------------------------------------------------------------------------
# compute_prototypes (helper Tensor)
# ---------------------------------------------------------------------------

class TestComputePrototypes:
    def test_correct_mean(self):
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels   = torch.tensor([0, 0, 1])
        protos   = compute_prototypes(features, labels, [0, 1])
        expected_0 = torch.tensor([2.0, 3.0])   # mean([1,2],[3,4])
        expected_1 = torch.tensor([5.0, 6.0])
        assert torch.allclose(protos[0], expected_0)
        assert torch.allclose(protos[1], expected_1)

    def test_empty_class_zero_proto(self):
        features = torch.randn(5, 4)
        labels   = torch.zeros(5, dtype=torch.long)
        protos   = compute_prototypes(features, labels, [0, 99])
        # class 99 không có mẫu → zero vector
        assert torch.allclose(protos[1], torch.zeros(4))


# ---------------------------------------------------------------------------
# eucl_ncm_predict
# ---------------------------------------------------------------------------

class TestEuclNCMPredict:
    def test_predicts_nearest(self):
        # 3 prototypes cách nhau xa
        protos = torch.eye(3).float() * 10   # [[10,0,0],[0,10,0],[0,0,10]]
        test   = torch.eye(3).float() * 9    # gần với prototype tương ứng
        preds  = eucl_ncm_predict(test, protos)
        assert preds.tolist() == [0, 1, 2]

    def test_output_shape(self):
        protos = torch.randn(5, 128)
        test   = torch.randn(20, 128)
        preds  = eucl_ncm_predict(test, protos)
        assert preds.shape == (20,)

    def test_index_in_range(self):
        protos = torch.randn(4, 64)
        test   = torch.randn(10, 64)
        preds  = eucl_ncm_predict(test, protos)
        assert all(0 <= p < 4 for p in preds.tolist())


# ---------------------------------------------------------------------------
# cosine_ncm_predict
# ---------------------------------------------------------------------------

class TestCosineNCMPredict:
    def test_predicts_identical(self):
        protos = torch.randn(3, 10)
        test   = protos.clone()  # test = prototype → phải predict đúng 100%
        preds  = cosine_ncm_predict(test, protos)
        assert preds.tolist() == [0, 1, 2]

    def test_output_shape(self):
        protos = torch.randn(6, 32)
        test   = torch.randn(15, 32)
        preds  = cosine_ncm_predict(test, protos)
        assert preds.shape == (15,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
test_metrics.py — Unit Tests cho src/evaluation/metrics.py
===========================================================
Chạy: python -m pytest tests/test_metrics.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from evaluation.metrics import (
    accuracy_at_task,
    topk_accuracy,
    CLMetrics,
    compute_avg_acc,
    compute_bwt,
    compute_forgetting,
    calculate_accuracy,
)


# ---------------------------------------------------------------------------
# accuracy_at_task
# ---------------------------------------------------------------------------

class TestAccuracyAtTask:
    def test_all_correct(self):
        y_pred = np.array([0, 1, 2, 3])
        y_true = np.array([0, 1, 2, 3])
        assert accuracy_at_task(y_pred, y_true) == pytest.approx(100.0)

    def test_all_wrong(self):
        y_pred = np.array([1, 2, 3, 0])
        y_true = np.array([0, 1, 2, 3])
        assert accuracy_at_task(y_pred, y_true) == pytest.approx(0.0)

    def test_half_correct(self):
        y_pred = np.array([0, 1, 5, 5])
        y_true = np.array([0, 1, 2, 3])
        assert accuracy_at_task(y_pred, y_true) == pytest.approx(50.0)

    def test_empty_input(self):
        assert accuracy_at_task(np.array([]), np.array([])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# topk_accuracy
# ---------------------------------------------------------------------------

class TestTopkAccuracy:
    def test_top1_perfect(self):
        scores = np.array([[3.0, 1.0, 0.0],
                           [0.0, 5.0, 1.0]])
        y_true = np.array([0, 1])
        assert topk_accuracy(scores, y_true, k=1) == pytest.approx(100.0)

    def test_top2_recovers(self):
        # top-1 sai nhưng top-2 đúng
        scores = np.array([[1.0, 3.0, 0.0],   # pred=1, true=0  →  top2: [1,0] ✓
                           [0.0, 5.0, 1.0]])   # pred=1, true=1  →  top2: [1,2] ✓
        y_true = np.array([0, 1])
        assert topk_accuracy(scores, y_true, k=2) == pytest.approx(100.0)

    def test_top1_half(self):
        scores = np.array([[3.0, 0.0], [0.0, 3.0]])
        y_true = np.array([0, 0])
        assert topk_accuracy(scores, y_true, k=1) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# CLMetrics
# ---------------------------------------------------------------------------

class TestCLMetrics:
    def _make_matrix(self):
        """3 tasks, ma trận giả định:
           R = [[80,  nan, nan],
                [78,  75,  nan],
                [70,  72,  85]]
        """
        m = CLMetrics(n_tasks=3)
        m.update(0, [80.0])
        m.update(1, [78.0, 75.0])
        m.update(2, [70.0, 72.0, 85.0])
        return m

    def test_avg_acc(self):
        m = self._make_matrix()
        res = m.compute()
        # A_0 = 80, A_1 = (78+75)/2 = 76.5, A_2 = (70+72+85)/3 = 75.67
        expected = np.mean([80.0, 76.5, (70 + 72 + 85) / 3])
        assert res["avg_acc"] == pytest.approx(expected, abs=0.1)

    def test_last_acc(self):
        m = self._make_matrix()
        res = m.compute()
        expected = np.mean([70.0, 72.0, 85.0])
        assert res["last_acc"] == pytest.approx(expected, abs=0.1)

    def test_bwt_negative_means_forgetting(self):
        m = self._make_matrix()
        res = m.compute()
        # BWT = [(70-80) + (72-75)] / 2 = [-10 + -3] / 2 = -6.5
        expected_bwt = ((70 - 80) + (72 - 75)) / 2
        assert res["bwt"] == pytest.approx(expected_bwt, abs=0.1)
        assert res["bwt"] < 0, "BWT âm → model bị quên"

    def test_plasticity(self):
        m = self._make_matrix()
        res = m.compute()
        expected = np.mean([80.0, 75.0, 85.0])
        assert res["plasticity"] == pytest.approx(expected, abs=0.1)

    def test_single_task(self):
        m = CLMetrics(n_tasks=1)
        m.update(0, [90.0])
        res = m.compute()
        assert res["avg_acc"] == pytest.approx(90.0)
        assert np.isnan(res["bwt"]), "Không đủ tasks để tính BWT"


# ---------------------------------------------------------------------------
# compute_avg_acc (helper function)
# ---------------------------------------------------------------------------

class TestComputeAvgAcc:
    def test_basic(self):
        assert compute_avg_acc([80.0, 75.0, 70.0]) == pytest.approx(75.0)

    def test_empty(self):
        assert compute_avg_acc([]) == pytest.approx(0.0)

    def test_single(self):
        assert compute_avg_acc([88.5]) == pytest.approx(88.5)


# ---------------------------------------------------------------------------
# compute_bwt
# ---------------------------------------------------------------------------

class TestComputeBWT:
    def test_no_forgetting(self):
        mat = np.array([[80., np.nan],
                        [80., 75.]])
        assert compute_bwt(mat) == pytest.approx(0.0)

    def test_forgetting(self):
        mat = np.array([[90., np.nan],
                        [70., 80.]])
        assert compute_bwt(mat) == pytest.approx(-20.0)

    def test_single_task(self):
        mat = np.array([[80.]])
        assert compute_bwt(mat) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# calculate_accuracy (tương thích ngược, nhận Tensor)
# ---------------------------------------------------------------------------

class TestCalculateAccuracy:
    def test_with_torch_tensor(self):
        import torch
        preds  = torch.tensor([0, 1, 2, 1])
        labels = torch.tensor([0, 1, 2, 0])
        assert calculate_accuracy(preds, labels) == pytest.approx(75.0)

    def test_with_numpy(self):
        preds  = np.array([0, 1, 2])
        labels = np.array([0, 1, 2])
        assert calculate_accuracy(preds, labels) == pytest.approx(100.0)

    def test_empty(self):
        assert calculate_accuracy(np.array([]), np.array([])) == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

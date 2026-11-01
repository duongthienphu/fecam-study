"""
run_experiment.py — Entry point chạy thực nghiệm FeCAM / NCM với W&B tracking
==============================================================================
Cách dùng:

  # Chạy FeCAM với W&B online:
  python scripts/run_experiment.py --config configs/reproduction/cifar100_fecam.yaml

  # Chạy NCM baseline, offline (không cần login W&B):
  python scripts/run_experiment.py --config configs/reproduction/cifar100_ncm.yaml --offline

  # Pipeline smoke test (10 classes, 1 epoch):
  python scripts/run_experiment.py --config configs/smoke.yaml --offline
"""

import os
import sys
import json
import yaml
import argparse

# ── Load .env nếu có (để dùng WANDB_API_KEY mà không cần wandb login) ──────
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
import logging
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Đảm bảo import từ src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.tracking import create_tracker, WandbTracker
from evaluation.metrics import CLMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader (hỗ trợ cả YAML và JSON)
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)

    # Normalize the documented experiment schema to the legacy FeCAM model args.
    if isinstance(raw.get("dataset"), dict):
        dataset = raw["dataset"]
        split = raw["split"]
        training = raw["training"]
        loader = raw["loader"]
        classifier = raw["classifier"]
        is_ncm = classifier["name"].lower() == "ncm"
        return {
            **raw,
            "dataset_name": dataset["name"],
            "dataset_source": dataset["name"],
            "dataset_revision": dataset.get("revision"),
            "data_root": dataset.get("root", "data"),
            "dataset": "cifar100",
            "shuffle": True,
            "seed": [split.get("class_order_seed", 1993)],
            "training_seed": raw.get("seed", 1),
            "cpu_threads": raw.get("cpu_threads", 4),
            "total_classes": split.get("total_classes", 100),
            "init_cls": split["initial_classes"],
            "increment": split["increment"],
            "model_name": "fecam",
            "convnet_type": "resnet18",
            "ncm": is_ncm,
            "full_cov": bool(classifier.get("full_covariance", False)),
            "diagonal": False,
            "per_class": bool(classifier.get("per_class_covariance", False)),
            "shrink": not is_ncm,
            "norm_cov": bool(classifier.get("correlation_normalization", False)),
            "tukey": classifier.get("tukey_beta") is not None,
            "beta": classifier.get("tukey_beta") or 0.5,
            "alpha1": classifier.get("shrinkage_gamma1", 1.0),
            "alpha2": classifier.get("shrinkage_gamma2", 1.0),
            "init_epochs": training["epochs"],
            "init_lr": training["lr"],
            "init_weight_decay": training["weight_decay"],
            "momentum": training.get("momentum", 0.9),
            "batch_size": loader["batch_size"],
            "num_workers": loader.get("num_workers", 0),
            "train_samples_per_class": loader.get("train_samples_per_class"),
            "test_samples_per_class": loader.get("test_samples_per_class"),
            "memory_size": 0,
            "device": ["cpu"],
            "T": 2,
            "prefix": raw.get("experiment", "reproduction"),
        }
    return raw


def seed_everything(seed: int, cpu_threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(cpu_threads)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracies_by_task(model, task_sizes, batch_size, num_workers):
    """Evaluate the active classifier separately on each seen task's test data."""
    test_dataset = model.test_loader.dataset
    labels = np.asarray(test_dataset.labels)
    result = []
    start = 0
    for size in task_sizes[: model._cur_task + 1]:
        end = start + size
        indices = np.flatnonzero((labels >= start) & (labels < end)).tolist()
        task_loader = DataLoader(
            Subset(test_dataset, indices), batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
        )
        y_pred, y_true = model._eval_maha(
            task_loader, model._init_protos, model._protos
        )
        result.append(float(np.mean(y_pred[:, 0] == y_true) * 100.0))
        start = end
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm FeCAM / NCM với W&B")
    parser.add_argument("--config",  type=str, required=True,
                        help="Đường dẫn tới file config (.yaml hoặc .json)")
    parser.add_argument("--offline", action="store_true",
                        help="Chạy W&B ở chế độ offline (không cần login)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Tên run trên W&B (ghi đè config)")
    parser.add_argument("--tags",    type=str, nargs="*", default=None,
                        help="Tags cho W&B run, cách nhau bằng dấu cách")
    args = parser.parse_args()

    # 1. Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Không tìm thấy config: {args.config}")
    config = load_config(args.config)
    seed_everything(
        config.get("training_seed", config.get("seed", [1])[0]
                   if isinstance(config.get("seed"), list)
                   else config.get("seed", 1)),
        config.get("cpu_threads", 4),
    )
    logger.info(f"✅ Đã nạp config: {args.config}")
    logger.info(f"   Config: {config}")

    # 2. Khởi tạo W&B tracker
    model_name = config.get("model_name", "unknown")
    classifier_name = "ncm" if config.get("ncm", False) else model_name
    dataset    = config.get("dataset",    "unknown")
    is_smoke = (
        config.get("total_classes", 100) < 100
        or config.get("init_epochs", 200) < 200
        or config.get("train_samples_per_class") is not None
        or config.get("test_samples_per_class") is not None
    )
    run_type = "smoke-test" if is_smoke else config.get("experiment", "reproduction")
    config["run_type"] = run_type
    config.setdefault("protocol_id", config.get("protocol", "cifar100-b50-i10-release-v1"))
    run_name   = args.run_name or f"{run_type}-{classifier_name}-{dataset}"
    tags       = args.tags or [classifier_name, dataset, run_type]

    tracker = create_tracker(
        config   = config,
        project  = "fecam-study",           # Public project cho yêu cầu đồ án
        run_name = run_name,
        tags     = tags,
        offline  = args.offline,
    )
    tracker.start()

    # 3. Khởi tạo DataManager và Model theo code gốc FeCAM
    #    Import lazy để tránh crash khi chạy unit test
    try:
        from utils.data_manager import DataManager
        from utils.factory import get_model

        data_manager = DataManager(
            dataset_name = config["dataset"],
            shuffle      = config.get("shuffle", True),
            seed         = config.get("seed", [1993])[0],
            init_cls     = config.get("init_cls", 10),
            increment    = config.get("increment", 10),
            data_root    = config.get("data_root", "data"),
            dataset_source = config.get("dataset_source"),
            dataset_revision = config.get("dataset_revision"),
            total_classes = config.get("total_classes"),
        )
        model = get_model(model_name, config)
        n_tasks = data_manager.nb_tasks

    except Exception as e:
        logger.exception(f"❌ Không thể khởi tạo DataManager/Model: {e}")
        logger.error("   Đảm bảo bạn đang chạy từ thư mục gốc của project.")
        tracker.finish()
        return

    # 4. Khởi tạo CLMetrics
    cl_metrics = CLMetrics(
        n_tasks=n_tasks, task_sizes=data_manager._increments
    )
    csv_path   = "results/tables/experiments.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    session_rows = []   # buffer cho W&B table

    # 5. Vòng lặp qua các Tasks
    for task_id in range(n_tasks):
        logger.info(f"\n{'='*60}")
        logger.info(f"  BẮT ĐẦU TASK {task_id}/{n_tasks - 1}")
        logger.info(f"{'='*60}")

        # ── Huấn luyện tăng dần ──────────────────────────────────────
        model.incremental_train(data_manager)

        # ── Đánh giá ─────────────────────────────────────────────────
        cnn_accy, nme_accy, maha_accy = model.eval_task()

        if config.get("ncm", False) and maha_accy is not None:
            acc_t    = maha_accy["top1"]
            acc_type = "Euclidean NCM"
        elif maha_accy is not None:
            acc_t    = maha_accy["top1"]
            acc_type = "Mahalanobis (FeCAM)"
        else:
            acc_t    = cnn_accy["top1"]
            acc_type = "CNN"
        classifier_accy = maha_accy if maha_accy is not None else cnn_accy

        # ── Cập nhật accuracy matrix bằng accuracy riêng từng task ────
        accs_per_seen = accuracies_by_task(
            model, data_manager._increments,
            config["batch_size"], config["num_workers"],
        )
        cl_metrics.update(task_id, accs_per_seen)

        # ── Tính metrics tạm thời ─────────────────────────────────────
        partial = cl_metrics.compute()
        avg_acc = partial["avg_acc"]
        bwt     = partial["bwt"] if not (isinstance(partial["bwt"], float)
                                         and np.isnan(partial["bwt"])) else None

        # ── Log W&B ──────────────────────────────────────────────────
        tracker.log_task(
            task_id        = task_id,
            acc            = acc_t,
            avg_acc        = avg_acc,
            bwt            = bwt,
            n_classes_seen = model._total_classes,
            extra          = {
                "acc_type": acc_type,
                "top5"    : classifier_accy.get("top5", None),
            },
        )

        # ── Ghi CSV ──────────────────────────────────────────────────
        row = {
            "run_type"     : run_type,
            "protocol"     : config["protocol_id"],
            "seed"         : config.get("training_seed", config.get("seed", [1])[0]),
            "class_order_seed": config.get("seed", [1993])[0],
            "model"        : classifier_name,
            "dataset"      : dataset,
            "session"      : task_id,
            "classes_seen" : model._total_classes,
            "acc_all"      : acc_t,
            "avg_acc"      : avg_acc,
            "acc_type"     : acc_type,
        }
        session_rows.append(row)
        df_row = pd.DataFrame([row])
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            df_row.to_csv(csv_path, index=False)
        else:
            df_row.to_csv(csv_path, mode="a", header=False, index=False)

        model.after_task()

    # 6. Tổng kết
    final_metrics = cl_metrics.compute()
    tracker.log_summary(final_metrics)

    # Log acc matrix heatmap
    tracker.log_acc_matrix(cl_metrics.get_acc_matrix())

    # Log bảng chi tiết từng session
    tracker.log_table(session_rows, table_name="session_results")

    logger.info(f"\n📁 Kết quả đã lưu tại: {csv_path}")
    if tracker.run_url:
        logger.info(f"🔗 W&B run: {tracker.run_url}")

    tracker.finish()


if __name__ == "__main__":
    main()

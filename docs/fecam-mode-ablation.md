# FeCAM full vs identity ablation

This experiment changes only the FeCAM covariance mode. Both evaluations must load the same trained base-task backbone checkpoint. The `full` mode uses class-wise covariance, shrinkage, correlation normalization, and Cholesky factors. The `identity` mode skips covariance and uses squared Euclidean distance on the same transformed features/prototypes; it is an ablation and is not the NCM baseline.

## Configs

- `configs/reproduction/cifar100_fecam.yaml`: `classifier.mode: full`.
- `configs/reproduction/cifar100_fecam_identity.yaml`: same data, class order, seed, backbone, and training settings; only `classifier.mode: identity`.

The two configs intentionally share protocol `cifar100-b50-i10-paper-wd-v2`, so checkpoint compatibility checks and the comparison script can verify the pair.

## Run on Kaggle T4 x2

Run from the repository root. Prepare the pinned dataset first if it is not cached.

```bash
python scripts/run_experiment.py train-base \
  --config configs/reproduction/cifar100_fecam.yaml \
  --device cuda:0 \
  --output outputs/fecam-mode-shared-base
```

This trains the base backbone once and writes `outputs/fecam-mode-shared-base/base.pt`. Then run both evaluations against that exact file; the two evaluations can run concurrently on separate GPUs:

```bash
python scripts/run_experiment.py evaluate \
  --config configs/reproduction/cifar100_fecam.yaml \
  --checkpoint outputs/fecam-mode-shared-base/base.pt \
  --device cuda:0 --output outputs/fecam-mode-full

python scripts/run_experiment.py evaluate \
  --config configs/reproduction/cifar100_fecam_identity.yaml \
  --checkpoint outputs/fecam-mode-shared-base/base.pt \
  --device cuda:1 --output outputs/fecam-mode-identity
```

Both summaries record the checkpoint SHA-256. The comparison script refuses to compare runs with different checkpoints or any config difference beyond `classifier.mode`.

```bash
python scripts/compare_fecam_modes.py \
  --full-summary outputs/fecam-mode-full/summary.json \
  --identity-summary outputs/fecam-mode-identity/summary.json \
  --output-dir outputs/fecam-mode-comparison
```

## Outputs and interpretation

- Each evaluation writes `per_class_accuracy.csv`: one row per seen class after each task, with base/incremental group, class name, correct count, total count, and accuracy.
- `fecam_mode_per_class_comparison.csv` reports `identity - full` for each class and task in percentage points.
- `fecam_mode_group_comparison.csv` reports base-class, incremental-class, and all-seen accuracy by task, also as `identity - full` percentage points.

“Base” means the 50 classes in the initial task. “Incremental” means all classes added after task 0 that have appeared by the evaluated task. Per-class accuracy is correct test images divided by test images for that class. Group accuracy is pooled over the test images in that group. Smoke configs validate the pipeline only; use the full reproduction configs for reported results.

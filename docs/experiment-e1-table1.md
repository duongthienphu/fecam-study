# E1 — CIFAR-100 Table 1 comparison

E1 evaluates three classifiers on the CIFAR-100 B=50, five-increment setting
from Table 1 of the FeCAM paper. All three runs must load the **same frozen
base checkpoint**, use training seed 1 and class-order seed 1993, and evaluate
all six sessions (base session 0 plus incremental sessions 1–5).

The current E1 configs use weight decay `0.0001`, matching the supplied seed-1
checkpoint and the paper's ResNet-18 training setting. The 64.8/68.8/70.9 values
below are the paper reference points for the three methods.
Euclidean NCM L2-normalizes test features and prototypes before computing
Euclidean distance, matching the authors' released implementation. The paper's
compact Eq. (2) does not spell out that implementation detail.
FeCAM shared and classwise modes now also normalize query features and
transformed prototypes at scoring time, following the authors' released code;
covariance estimates continue to use transformed training features.

| Group method | Config | Covariance | Paper Avg Acc. |
|---|---|---|---:|
| Euclidean NCM | `configs/experiments/e1_ncm_euclidean.yaml` | None; L2-normalized features and prototypes | 64.8% |
| FeCAM shared | `configs/experiments/e1_fecam_shared.yaml` | One class-averaged within-class covariance for all seen classes | 68.8% |
| FeCAM classwise | `configs/experiments/e1_fecam_classwise.yaml` | One covariance per class | 70.9% |

The shared mode follows the paper's Σ¹:ᵗ update: average the covariance
estimated from the classes in the current task with the previously stored
shared covariance, weighted by the number of classes. Each class covariance is
computed from Tukey-transformed features. Shrinkage is applied before
factorization. Correlation normalization is enabled for classwise matrices, as
described for the multiple-matrix case in the paper; it is disabled for the
single shared matrix. The classwise mode applies shrinkage and correlation
normalization to each class matrix. Both FeCAM configs use Tukey β=0.5 and
γ₁=γ₂=1.

## Run with one provided checkpoint

The configs reject evaluation without `--checkpoint`; they cannot silently
train three different backbones. Run these commands from the repository root,
substituting the path to the supplied seed-1 `base.pt` checkpoint:

```powershell
python scripts/run_experiment.py evaluate --config configs/experiments/e1_ncm_euclidean.yaml --checkpoint PATH_TO_SEED1_BASE_PT --device cuda:0 --output outputs/e1/seed1/ncm
python scripts/run_experiment.py evaluate --config configs/experiments/e1_fecam_shared.yaml --checkpoint PATH_TO_SEED1_BASE_PT --device cuda:0 --output outputs/e1/seed1/fecam_shared
python scripts/run_experiment.py evaluate --config configs/experiments/e1_fecam_classwise.yaml --checkpoint PATH_TO_SEED1_BASE_PT --device cuda:0 --output outputs/e1/seed1/fecam_classwise
```

`--output` must be new for each run. On CPU, replace `cuda:0` with `cpu`.
The checkpoint must have been trained with weight decay `0.0001` and the other
training settings in the E1 YAMLs.
Each summary records the checkpoint SHA-256, class-order list and hash, full
config, per-session seen accuracy, Avg/Final/BWT and (for FeCAM) matrix storage
by task.

## Paired comparison and W&B

After the three runs finish, create the two CSV tables and the three-line plot:

```powershell
python scripts/report_e1.py outputs/e1/seed1/ncm/summary.json outputs/e1/seed1/fecam_shared/summary.json outputs/e1/seed1/fecam_classwise/summary.json
```

The report stops if checkpoint hashes, class order, seed, protocol or training
configuration differ. It writes `results/e1/e1_table1_comparison.csv`,
`results/e1/e1_seen_accuracy_by_task.csv` and `results/e1/e1_seen_accuracy.png`.
The comparison CSV includes measured Avg/Final, the paper Avg reference and
the measured-minus-paper difference in percentage points. The paper reference
is not a pass/fail threshold: the group's independent reproduction may differ.

To upload one W&B run per classifier under the same E1 group, configure
`WANDB_API_KEY` in the environment or the local ignored `.env` file, then run:

```powershell
python scripts/upload_e1_to_wandb.py outputs/e1/seed1/ncm/summary.json outputs/e1/seed1/fecam_shared/summary.json outputs/e1/seed1/fecam_classwise/summary.json --plot results/e1/e1_seen_accuracy.png
```

The script prints the three run URLs. Set the W&B project visibility to Public
in W&B before sharing those links.

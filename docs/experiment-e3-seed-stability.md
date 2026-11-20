# E3 — seed stability

E3 checks whether the E1 comparison depends on the base-backbone training
seed. It keeps the CIFAR-100 class order fixed at 1993 and changes only the
backbone training seed across 1, 2, and 3. The three classifier runs within a
seed must use that seed's same checkpoint. Seed-1 evaluations are reused from
E1; E3 adds six classifier evaluations for seeds 2 and 3.

The three E1 configs are reused with their training seed changed. All use
weight decay 0.0001, matching the paper and the supplied seed-1 checkpoint.
Each missing seed-2/3 backbone is trained
once with the E1 backbone recipe. Training checkpoints are saved each epoch and
`run_e3.py` resumes interrupted training from that checkpoint.

## Run

First complete E1 and keep its three seed-1 `summary.json` files. Then on a GPU
machine or Kaggle, run from the repository root:

```powershell
python scripts/run_e3.py `
  --seed1-summaries outputs/e1/seed1/ncm/summary.json outputs/e1/seed1/fecam_shared/summary.json outputs/e1/seed1/fecam_classwise/summary.json `
  --checkpoint-template "outputs/e3/backbones/seed{seed}/base.pt" `
  --train-missing-checkpoints `
  --device cuda:0
```

The command reuses completed seed-1 E1 results, trains seed-2/3 backbones only
when their final checkpoints are missing, and evaluates the three methods on
each checkpoint. If a training session stops, run the same command again in
the same working directory; the backbone trainer resumes from its saved epoch.
If a final checkpoint already exists, it is reused. Do not change the
class-order seed between the three training seeds.

Required input/output checks are enforced: each seed's three classifier
summaries must have one checkpoint hash; all nine must have one class-order
hash; the three seeds must have distinct checkpoint hashes; each run must
contain sessions 0–5 with 50, 60, 70, 80, 90, and 100 classes seen.

## Results

The runner writes six new evaluation summaries under
`outputs/e3/evaluations/seed2/` and `seed3/`; seed-2/3 backbone checkpoints and
their resumable training states live under `outputs/e3/backbones/`.
`scripts/report_e3.py` creates `outputs/e3/report/e3_seed_stability.csv` and
`outputs/e3/report/e3_seed_stability.png`. The CSV has one row per method and seed,
including FeCAM-minus-NCM Avg/Final deltas in percentage points, plus one
aggregate row per method with mean and **sample** standard deviation across the
three seeds. The plot shows mean ± sample SD and each seed's point for Avg and
Final accuracy.

To upload nine W&B runs grouped by seed, set `WANDB_API_KEY` in the environment
or ignored `.env`, then pass the nine `summary.json` files to
`scripts/upload_e3_to_wandb.py`. Make the W&B project Public before sharing its
links.

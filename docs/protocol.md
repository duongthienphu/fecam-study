# CIFAR-100 reproduction protocol

Protocol ID: `cifar100-b50-i10-release-v1`

The checked-in run configurations are `configs/reproduction/cifar100_fecam.yaml`
and `configs/reproduction/cifar100_ncm.yaml`. Both use the same dataset revision,
class order, initial training and loader settings. They differ only in the
classifier being compared.

## Data and task sequence

- Dataset: `uoft-cs/cifar100`, Hugging Face revision
  `aadb3af77e9048adbea6b47c21a81e47dd092ae5`; use the `fine_label` target.
- Local cache: `data/hf-cache` (ignored by Git).
- Class order: shuffled with seed 1993. The first 50 mapped classes form the
  initial task; five subsequent tasks add 10 classes each.
- All available training and test examples are used (`null` samples-per-class).
- The test set is evaluated task by task after each session. The reported
  session accuracy is weighted by the number of classes in each task, so the
  initial 50-class task is not treated as the same size as a 10-class task.

## Model and classifier

- Backbone: CIFAR ResNet-18 with cosine classifier. Train on the initial task
  for 200 epochs using SGD, learning rate 0.1, momentum 0.9, weight decay
  0.0005, and batch size 128. Freeze the feature extractor after that task.
- FeCAM: per-class full covariance, Tukey transform with beta 0.5, shrinkage
  gamma1=gamma2=1, and correlation normalization.
- NCM: Euclidean nearest class mean using L2-normalized feature vectors and
  prototypes. It uses the same trained backbone and task sequence.
- Random training seed: 1. Data order seed: 1993. DataLoader workers: 0.

The paper text reports weight decay 0.0001 for its ResNet-18 training. This
project's supplied release config retains 0.0005; report that as a protocol
deviation unless the group verifies a different value in the exact author
configuration being reproduced.

## Metrics

Let `R[i,j]` be top-1 accuracy on task `j`'s test examples after finishing task
`i`. Average incremental accuracy is the mean of the overall seen-class
accuracy at each session; each session's accuracy is weighted by task class
counts. Final accuracy is the overall seen-class accuracy after the final
session. BWT is the mean change in accuracy on old tasks from the session when
they were first learned to the final session. FWT is omitted because this
protocol does not measure a pre-learning baseline on future tasks.

## Run commands

From the repository root, after installing `requirements.txt`:

```powershell
python scripts/run_experiment.py --config configs/smoke.yaml --offline
python scripts/run_experiment.py --config configs/smoke_fecam.yaml --offline
python scripts/run_experiment.py --config configs/reproduction/cifar100_ncm.yaml --offline
python scripts/run_experiment.py --config configs/reproduction/cifar100_fecam.yaml --offline
```

The smoke configs use 10 classes (5+5), 10 train/test examples per class and 1
epoch to check the NCM and FeCAM paths; neither is a reproduction result. Full runs can be long on CPU. For cloud runs,
remove `--offline` after creating a public W&B project and configuring the
account credentials locally. Do not upload `.env` or API keys to GitHub.

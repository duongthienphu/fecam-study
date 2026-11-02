"""Run a short, task-incremental NCM or FeCAM evaluation from a YAML config."""

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.datasets import DATASET_ID, DATASET_REVISION, TaskDataset, load_cifar100
from src.data.task_split import TaskSplit
from src.engine.extract_features import extract_features
from src.engine.train_base import train_base
from src.evaluation.metrics import CLMetrics
from src.models.backbone import Backbone
from src.models.fecam import FeCAM
from src.models.ncm import NCMClassifier
from src.utils.seed import seed_everything, seed_worker


def load_config(path):
    with Path(path).open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    required = ('experiment', 'protocol', 'seed', 'dataset', 'split', 'backbone',
                'training', 'loader', 'classifier')
    if not isinstance(config, dict) or any(key not in config for key in required):
        raise ValueError(f'Config must include these fields: {required}')
    if config['dataset'].get('name') != DATASET_ID or config['dataset'].get('revision') != DATASET_REVISION:
        raise ValueError('Only the pinned uoft-cs/cifar100 revision is supported')
    if config['backbone'] != 'resnet18-cifar-cosine':
        raise ValueError('Unsupported backbone')
    classifier = config['classifier']
    if classifier.get('name') == 'ncm':
        if classifier.get('metric', 'euclidean') not in ('euclidean', 'cosine'):
            raise ValueError('NCM metric must be euclidean or cosine')
        if classifier.get('normalize_features', True) != classifier.get('normalize_prototypes', True):
            raise ValueError('NCM normalizes features and prototypes together')
    elif classifier.get('name') == 'fecam':
        if not classifier.get('per_class_covariance') or not classifier.get('full_covariance'):
            raise ValueError('This FeCAM implementation requires per-class full covariance')
        if not classifier.get('correlation_normalization'):
            raise ValueError('This FeCAM implementation requires correlation normalization')
        if classifier.get('tukey_beta') not in (None, 0.5):
            raise ValueError('This implementation supports Tukey beta=0.5 or no transform')
    else:
        raise ValueError('classifier.name must be ncm or fecam')
    TaskSplit(**config['split'])
    loader = config['loader']
    if loader.get('batch_size', 0) < 2 or loader.get('num_workers', -1) < 0:
        raise ValueError('Require batch_size >= 2 and num_workers >= 0')
    if config['experiment'] != 'smoke-test':
        raise ValueError('This compact runner is intended for smoke-test configs only')
    return config


def make_loader(data, split, config, task, source, device, augment=False):
    options = config['loader']
    rows = data[source]
    dataset = TaskDataset(
        rows, rows['fine_label'], split, task, source, augment=augment,
        samples_per_class=options[f'{source}_samples_per_class'],
    )
    return DataLoader(
        dataset, batch_size=options['batch_size'], shuffle=augment,
        num_workers=options['num_workers'], pin_memory=str(device).startswith('cuda'),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config['seed']), drop_last=False,
    )


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def provenance():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True)
    package_names = ('torch', 'torchvision', 'numpy', 'Pillow', 'datasets', 'PyYAML')
    return {
        'git_commit': result.stdout.strip() if result.returncode == 0 else None,
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: version(name) for name in package_names},
        'cuda_runtime': torch.version.cuda,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', help='New output directory; it cannot already exist')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or cuda:0')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    split = TaskSplit(**config['split'])
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    seed_everything(config['seed'])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    started = datetime.now(timezone.utc)
    output = Path(args.output) if args.output else ROOT / 'outputs' / f"smoke-{config['classifier']['name']}-{started:%Y%m%dT%H%M%SZ}"
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'action': 'evaluate', 'started_utc': started.isoformat(), 'status': 'running',
        'config': config, 'class_order': split.class_order, 'tasks': split.tasks,
        'label_map': split.label_map, 'device': device, 'provenance': provenance(),
    }
    try:
        data_start = perf_counter()
        data_root = Path(config['dataset'].get('root', 'data'))
        if not data_root.is_absolute():
            data_root = ROOT / data_root
        data = load_cifar100(data_root)
        manifest['data_seconds'] = perf_counter() - data_start
        manifest['dataset_source_mode'] = (
            'local-cifar100-python-pickle' if str(data['train']._fingerprint).startswith('cifar100-python-local')
            else 'huggingface-pinned-revision'
        )
        manifest['dataset'] = {
            name: {'samples': len(rows), 'fingerprint': rows._fingerprint}
            for name, rows in data.items()
        }
        model = Backbone(config['split']['initial_classes'])
        base_loader = make_loader(data, split, config, 0, 'train', device, augment=True)
        history = train_base(model, base_loader, device=device, **config['training'])
        model.freeze()
        manifest['base_training'] = history

        classifier_cfg = config['classifier']
        if classifier_cfg['name'] == 'ncm':
            classifier = NCMClassifier(
                metric=classifier_cfg.get('metric', 'euclidean'),
                normalize=classifier_cfg.get('normalize_features', True),
            )
            update = lambda features, labels: classifier.update_prototypes(features.numpy(), labels.numpy())
            predict = lambda features: classifier.predict(features.numpy())
        else:
            beta = classifier_cfg.get('tukey_beta')
            classifier = FeCAM(
                transform='sqrt' if beta is not None else 'none',
                gamma1=classifier_cfg.get('shrinkage_gamma1', 1.0),
                gamma2=classifier_cfg.get('shrinkage_gamma2', 1.0), device='cpu',
            )
            update = classifier.update
            predict = lambda features: classifier.predict(features).cpu().numpy()

        task_sizes = [len(task) for task in split.tasks]
        metrics = CLMetrics(len(split.tasks), task_sizes=task_sizes)
        sessions = []
        for task_id in range(len(split.tasks)):
            train_loader = make_loader(data, split, config, task_id, 'train', device)
            train_features, train_labels = extract_features(model, train_loader, device)
            update(train_features, train_labels)
            task_accuracies = []
            for eval_task in range(task_id + 1):
                test_loader = make_loader(data, split, config, eval_task, 'test', device)
                test_features, test_labels = extract_features(model, test_loader, device)
                prediction = predict(test_features)
                task_accuracies.append(float(np.mean(prediction == test_labels.numpy()) * 100))
            metrics.update(task_id, task_accuracies)
            session = {
                'task': task_id, 'classes_seen': sum(task_sizes[:task_id + 1]),
                'task_accuracies': task_accuracies,
                'seen_accuracy': float(np.average(task_accuracies, weights=task_sizes[:task_id + 1])),
            }
            sessions.append(session)
            print(json.dumps(session), flush=True)

        final_metrics = metrics.compute()
        final_metrics = {
            key: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value)
            for key, value in final_metrics.items()
        }
        matrix = metrics.get_acc_matrix()
        manifest['evaluation'] = {
            'classifier': classifier_cfg, 'sessions': sessions, 'metrics': final_metrics,
            'accuracy_matrix': [[None if not np.isfinite(v) else float(v) for v in row] for row in matrix],
        }
        manifest['status'] = 'completed'
        write_json(output / 'summary.json', manifest)
        print(f'Completed smoke evaluation: {output}', flush=True)
    except Exception as exc:
        write_json(output / 'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()

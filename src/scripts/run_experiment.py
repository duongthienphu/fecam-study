"""Data/backbone entry point. Incremental classifiers are integrated separately."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.datasets import DATASET_ID, DATASET_REVISION, TaskDataset, load_cifar100
from src.data.task_split import TaskSplit
from src.engine.extract_features import extract_features
from src.engine.train_base import load_checkpoint, save_checkpoint, train_base
from src.models.backbone import Backbone
from src.utils.seed import seed_everything, seed_worker


def load_config(path):
    with Path(path).open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError('Config must be a YAML mapping')
    for key in ('experiment', 'protocol', 'seed', 'dataset', 'split', 'backbone', 'training', 'loader'):
        if key not in config:
            raise ValueError(f'Missing config field: {key}')
    if config['dataset']['name'] != DATASET_ID or config['dataset']['revision'] != DATASET_REVISION:
        raise ValueError('This pipeline supports only the pinned CIFAR-100 source')
    if config['backbone'] != 'resnet18-cifar-cosine':
        raise ValueError('Unsupported backbone')
    TaskSplit(**config['split'])
    if config['loader']['batch_size'] < 2 or config['loader']['num_workers'] < 0:
        raise ValueError('Require batch_size >= 2 and num_workers >= 0')
    if config['experiment'] == 'reproduction':
        if (config['split']['total_classes'] != 100
                or any(config['loader'][key] is not None
                       for key in ('train_samples_per_class', 'test_samples_per_class'))):
            raise ValueError('Reproduction must use all classes and all samples')
    return config


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def make_loader(data, split, config, task, source, device, augment=False, seen=False):
    settings = config['loader']
    rows = data[source]
    dataset = TaskDataset(rows, rows['fine_label'], split, task, source,
                          augment=augment, seen=seen,
                          samples_per_class=settings[f'{source}_samples_per_class'])
    return DataLoader(dataset, batch_size=settings['batch_size'], shuffle=augment,
                      num_workers=settings['num_workers'], pin_memory=str(device).startswith('cuda'),
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config['seed']), drop_last=False)


def provenance():
    def git(*args):
        result = subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    paths = sorted((ROOT / 'src').rglob('*.py')) + [ROOT / 'requirements.txt']
    return {'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
            'code_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in paths},
            'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'datasets', 'PyYAML')},
            'cuda_runtime': torch.version.cuda}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train-base', 'extract-features'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', help='New directory; existing output is never overwritten')
    parser.add_argument('--checkpoint')
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or cuda:0')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    split = TaskSplit(**config['split'])
    split.classes(args.task)
    if args.action == 'extract-features' and not args.checkpoint:
        parser.error('extract-features requires --checkpoint')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    seed_everything(config['seed'])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    started = datetime.now(timezone.utc)
    output = Path(args.output) if args.output else ROOT / 'outputs' / f'backbone-{started:%Y%m%dT%H%M%S%fZ}'
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'action': args.action, 'started_utc': started.isoformat(), 'config': config,
                'class_order': split.class_order, 'tasks': split.tasks,
                'label_map': split.label_map, 'device': device, 'provenance': provenance()}
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
        manifest['device_name'] = torch.cuda.get_device_name(device)
    write_json(output / 'manifest.json', manifest)
    try:
        start = perf_counter()
        root = Path(config['dataset']['root'])
        data = load_cifar100(root if root.is_absolute() else ROOT / root)
        manifest['data_seconds'] = perf_counter() - start
        manifest['dataset'] = {name: {'samples': len(rows), 'fingerprint': rows._fingerprint,
                                     'fine_label_names': rows.features['fine_label'].names}
                               for name, rows in data.items()}
        manifest['task_counts'] = []
        for task in range(len(split.tasks)):
            train = make_loader(data, split, config, task, 'train', device)
            test = make_loader(data, split, config, task, 'test', device, seen=True)
            manifest['task_counts'].append({'task': task, 'train': len(train.dataset),
                                            'test_seen': len(test.dataset),
                                            'train_per_class': dict(Counter(train.dataset.targets))})
        if args.action == 'train-base':
            model = Backbone(config['split']['initial_classes'])
            train = make_loader(data, split, config, 0, 'train', device, augment=True)
            def log_epoch(record):
                with (output / 'epochs.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, allow_nan=False) + '\n')
                print(json.dumps(record), flush=True)
            history = train_base(model, train, device=device, on_epoch=log_epoch, **config['training'])
            metadata = {'config': config, 'class_order': split.class_order,
                        'provenance': manifest['provenance'], 'epochs_completed': len(history)}
            save_checkpoint(output / 'base.pt', model, metadata)
            manifest['checkpoint_sha256'] = hashlib.sha256((output / 'base.pt').read_bytes()).hexdigest()
            manifest['history'] = history
            model.freeze()
        elif args.action == 'extract-features':
            model, metadata = load_checkpoint(args.checkpoint, device)
            for key in ('protocol', 'seed', 'dataset', 'split', 'backbone', 'training'):
                if metadata['config'][key] != config[key]:
                    raise ValueError(f'Checkpoint/config mismatch: {key}')
            if (metadata['class_order'] != split.class_order
                    or metadata['config']['loader']['train_samples_per_class'] != config['loader']['train_samples_per_class']):
                raise ValueError('Checkpoint class order or training subset mismatch')
            manifest['checkpoint'] = str(Path(args.checkpoint).resolve())
            manifest['checkpoint_sha256'] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        if args.action != 'prepare':
            manifest['features'] = {}
            for source in ('train', 'test'):
                loader = make_loader(data, split, config, args.task, source, device, seen=source == 'test')
                start = perf_counter()
                vectors, labels = extract_features(model, loader, device)
                manifest['features'][source] = {'shape': list(vectors.shape),
                                                'labels': sorted(labels.unique().tolist()),
                                                'seconds': perf_counter() - start}
                # Keep only summaries: feature caches must not become replay data.
                del vectors, labels
        if str(device).startswith('cuda'):
            manifest['peak_vram_bytes'] = torch.cuda.max_memory_allocated(device)
        manifest['status'] = 'completed'
        write_json(output / 'summary.json', manifest)
        print(f'Completed {args.action}: {output}', flush=True)
    except Exception as exc:
        write_json(output / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()

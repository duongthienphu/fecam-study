"""Run a short, task-incremental NCM or FeCAM evaluation from a YAML config."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
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

try:
    import psutil
except ImportError:  # Local CPU smoke runs can proceed without optional telemetry.
    psutil = None

from src.data.datasets import DATASET_ID, DATASET_REVISION, TaskDataset, load_cifar100
from src.data.task_split import TaskSplit
from src.engine.extract_features import extract_features
from src.engine.train_base import load_checkpoint, save_checkpoint, train_base
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
        mode = classifier.get('mode', 'full')
        if mode not in ('full', 'identity'):
            raise ValueError('FeCAM mode must be full or identity')
        if mode == 'full' and (
                not classifier.get('per_class_covariance')
                or not classifier.get('full_covariance')
                or not classifier.get('correlation_normalization')):
            raise ValueError('FeCAM full mode requires per-class full covariance and correlation normalization')
        if classifier.get('tukey_beta') not in (None, 0.5):
            raise ValueError('This implementation supports Tukey beta=0.5 or no transform')
    else:
        raise ValueError('classifier.name must be ncm or fecam')
    TaskSplit(**config['split'])
    loader = config['loader']
    if loader.get('batch_size', 0) < 2 or loader.get('num_workers', -1) < 0:
        raise ValueError('Require batch_size >= 2 and num_workers >= 0')
    if config['experiment'] == 'reproduction':
        if (config['split']['total_classes'] != 100
                or config['loader']['train_samples_per_class'] is not None
                or config['loader']['test_samples_per_class'] is not None):
            raise ValueError('Reproduction configs must use all classes and all samples')
    elif config['experiment'] != 'smoke-test':
        raise ValueError('Supported experiment values are smoke-test and reproduction')
    return config


def make_loader(data, split, config, task, source, device, augment=False, seen=False):
    options = config['loader']
    rows = data[source]
    dataset = TaskDataset(
        rows, rows['fine_label'], split, task, source, augment=augment, seen=seen,
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
    if psutil is not None:
        package_names += ('psutil',)
    return {
        'git_commit': result.stdout.strip() if result.returncode == 0 else None,
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: version(name) for name in package_names},
        'cuda_runtime': torch.version.cuda,
    }


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_storage_record(tensor):
    """Measure bytes in a tensor's backing storage (not allocator/process peak)."""
    nbytes = tensor.untyped_storage().nbytes() if tensor.numel() else 0
    device_type = tensor.device.type
    return {
        'device': str(tensor.device),
        'dtype': str(tensor.dtype),
        'shape': list(tensor.shape),
        'storage_bytes': nbytes,
        'storage_mib': round(nbytes / (1024 ** 2), 6),
        'storage_ram_bytes': nbytes if device_type == 'cpu' else 0,
        'storage_vram_bytes': nbytes if device_type == 'cuda' else 0,
        'storage_ram_mib': round(nbytes / (1024 ** 2), 6) if device_type == 'cpu' else 0.0,
        'storage_vram_mib': round(nbytes / (1024 ** 2), 6) if device_type == 'cuda' else 0.0,
    }


def validate_checkpoint(metadata, config, split):
    checkpoint_config = metadata['config']
    for key in ('protocol', 'seed', 'dataset', 'split', 'backbone', 'training'):
        if checkpoint_config[key] != config[key]:
            raise ValueError(f'Checkpoint/config mismatch: {key}')
    if metadata['class_order'] != split.class_order:
        raise ValueError('Checkpoint class order mismatch')


class PeakMemoryMonitor:
    """Sample process RSS; CUDA peak stats are collected from torch at stop()."""

    def __init__(self, interval_seconds=0.1):
        self.process = psutil.Process() if psutil is not None else None
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = self.process.memory_info().rss if self.process else None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        if self.process is None:
            return
        while not self._stop.wait(self.interval_seconds):
            try:
                self.peak_rss_bytes = max(self.peak_rss_bytes, self.process.memory_info().rss)
            except psutil.Error:
                return

    def start(self):
        if self.process is not None:
            self._thread.start()

    def stop(self, cuda_device_index=None, measure_cuda=True):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join()
        if self.process is not None:
            try:
                self.peak_rss_bytes = max(self.peak_rss_bytes, self.process.memory_info().rss)
            except psutil.Error:
                pass
        result = {
            'peak_ram_rss_bytes': self.peak_rss_bytes,
            'peak_ram_rss_mb': (round(self.peak_rss_bytes / (1024 ** 2), 3)
                                if self.peak_rss_bytes is not None else None),
            'ram_sample_interval_seconds': self.interval_seconds,
            'peak_vram_allocated_bytes': None,
            'peak_vram_allocated_mb': None,
            'peak_vram_reserved_bytes': None,
            'peak_vram_reserved_mb': None,
        }
        if measure_cuda and cuda_device_index is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(cuda_device_index)
                allocated = torch.cuda.max_memory_allocated(cuda_device_index)
                reserved = torch.cuda.max_memory_reserved(cuda_device_index)
                result.update({
                    'peak_vram_allocated_bytes': allocated,
                    'peak_vram_allocated_mb': round(allocated / (1024 ** 2), 3),
                    'peak_vram_reserved_bytes': reserved,
                    'peak_vram_reserved_mb': round(reserved / (1024 ** 2), 3),
                    'cuda_device': f'cuda:{cuda_device_index}',
                })
            except RuntimeError as exc:
                result['vram_measurement_warning'] = str(exc)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('prepare', 'train-base', 'extract-features', 'evaluate'),
                        default='evaluate')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--output', help='New output directory; it cannot already exist')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or cuda:0')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    split = TaskSplit(**config['split'])
    split.classes(args.task)
    if args.action == 'extract-features' and not args.checkpoint:
        parser.error('extract-features requires --checkpoint')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    cuda_device_index = None
    if str(device).startswith('cuda'):
        requested_device = torch.device(device)
        cuda_device_index = (torch.cuda.current_device() if requested_device.index is None
                             else requested_device.index)
        visible_gpu_count = torch.cuda.device_count()
        if not 0 <= cuda_device_index < visible_gpu_count:
            raise RuntimeError(
                f'Requested {device}, but this process sees {visible_gpu_count} CUDA GPU(s). '
                'Check CUDA_VISIBLE_DEVICES and the assigned GPU index.'
            )
        torch.cuda.set_device(cuda_device_index)
        device = f'cuda:{cuda_device_index}'
    seed_everything(config['seed'])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    started = datetime.now(timezone.utc)
    output = Path(args.output) if args.output else ROOT / 'outputs' / f"smoke-{config['classifier']['name']}-{started:%Y%m%dT%H%M%SZ}"
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'action': args.action, 'started_utc': started.isoformat(), 'status': 'running',
        'config': config, 'class_order': split.class_order, 'tasks': split.tasks,
        'label_map': split.label_map, 'device': device,
        'assigned_gpu': os.environ.get('FECAM_ASSIGNED_GPU', device),
        'provenance': provenance(),
    }
    memory_monitor = PeakMemoryMonitor() if args.action == 'evaluate' else None
    measure_cuda_memory = cuda_device_index is not None
    if memory_monitor:
        if measure_cuda_memory:
            try:
                # PyTorch expects a device object or index; use the validated index.
                torch.cuda.reset_peak_memory_stats(cuda_device_index)
            except RuntimeError as exc:
                measure_cuda_memory = False
                manifest['vram_measurement_warning'] = str(exc)
                print(f'Warning: VRAM peak tracking disabled: {exc}', flush=True)
        memory_monitor.start()
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
        manifest['task_counts'] = []
        for task_id in range(len(split.tasks)):
            train_task = make_loader(data, split, config, task_id, 'train', device)
            test_seen = make_loader(data, split, config, task_id, 'test', device, seen=True)
            manifest['task_counts'].append({
                'task': task_id, 'train': len(train_task.dataset), 'test_seen': len(test_seen.dataset),
                'train_per_class': dict(Counter(train_task.dataset.targets)),
            })

        if args.action == 'prepare':
            manifest['status'] = 'completed'
            write_json(output / 'summary.json', manifest)
            print(f'Completed prepare: {output}', flush=True)
            return
        elif args.action == 'extract-features':
            model, metadata = load_checkpoint(args.checkpoint, device)
            validate_checkpoint(metadata, config, split)
            manifest['checkpoint'] = str(Path(args.checkpoint).resolve())
            manifest['checkpoint_sha256'] = checkpoint_sha256(args.checkpoint)
            manifest['features'] = {}
            for source in ('train', 'test'):
                loader = make_loader(
                    data, split, config, args.task, source, device,
                    seen=(source == 'test'),
                )
                vectors, labels = extract_features(model, loader, device)
                manifest['features'][source] = {
                    'shape': list(vectors.shape), 'labels': sorted(labels.unique().tolist()),
                }
            manifest['status'] = 'completed'
            write_json(output / 'summary.json', manifest)
            print(f'Completed extract-features: {output}', flush=True)
            return
        else:
            if args.action == 'evaluate' and args.checkpoint:
                model, metadata = load_checkpoint(args.checkpoint, device)
                validate_checkpoint(metadata, config, split)
                manifest['checkpoint'] = str(Path(args.checkpoint).resolve())
                manifest['checkpoint_sha256'] = checkpoint_sha256(args.checkpoint)
                manifest['base_training'] = 'loaded_shared_checkpoint'
            else:
                model = Backbone(config['split']['initial_classes'])
                base_loader = make_loader(data, split, config, 0, 'train', device, augment=True)
                history = train_base(model, base_loader, device=device, **config['training'])
                manifest['base_training'] = history
                if args.action == 'train-base':
                    metadata = {'config': config, 'class_order': split.class_order,
                                'provenance': manifest['provenance'], 'epochs_completed': len(history)}
                    save_checkpoint(output / 'base.pt', model, metadata)
                    manifest['checkpoint'] = str((output / 'base.pt').resolve())
                    manifest['checkpoint_sha256'] = checkpoint_sha256(output / 'base.pt')
                    manifest['status'] = 'completed'
                    write_json(output / 'summary.json', manifest)
                    print(f'Completed train-base: {output}', flush=True)
                    return
                model.freeze()

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
                mode=classifier_cfg.get('mode', 'full'),
                transform='sqrt' if beta is not None else 'none',
                gamma1=classifier_cfg.get('shrinkage_gamma1', 1.0),
                gamma2=classifier_cfg.get('shrinkage_gamma2', 1.0), device='cpu',
            )
            update = classifier.update
            predict = lambda features: classifier.predict(features).cpu().numpy()

        task_sizes = [len(task) for task in split.tasks]
        metrics = CLMetrics(len(split.tasks), task_sizes=task_sizes)
        sessions = []
        matrix_storage_by_task = []
        class_names = data['test'].features['fine_label'].names
        for task_id in range(len(split.tasks)):
            train_loader = make_loader(data, split, config, task_id, 'train', device)
            train_features, train_labels = extract_features(model, train_loader, device)
            update(train_features, train_labels)
            if classifier_cfg['name'] == 'fecam':
                # FeCAM persists Cholesky factors as its per-class matrix state.
                # Measure the actual backing storage after each incremental update.
                matrix_storage_by_task.append({
                    'task': task_id,
                    'mode': classifier.mode,
                    'classes_stored': classifier.num_classes,
                    'matrix_state': tensor_storage_record(classifier.factors),
                })
            task_accuracies = []
            class_counts = {}
            for eval_task in range(task_id + 1):
                test_loader = make_loader(data, split, config, eval_task, 'test', device)
                test_features, test_labels = extract_features(model, test_loader, device)
                prediction = predict(test_features)
                truth = test_labels.numpy()
                correct_mask = prediction == truth
                task_accuracies.append(float(np.mean(correct_mask) * 100))
                for class_id in np.unique(truth):
                    mask = truth == class_id
                    counts = class_counts.setdefault(int(class_id), [0, 0])
                    counts[0] += int(np.count_nonzero(correct_mask & mask))
                    counts[1] += int(np.count_nonzero(mask))
            metrics.update(task_id, task_accuracies)
            class_rows = []
            base_classes = set(split.tasks[0])
            for class_id, (n_correct, n_total) in sorted(class_counts.items()):
                original_id = split.class_order[class_id]
                group = 'base' if class_id in base_classes else 'incremental'
                introduced_task = (0 if group == 'base' else
                                   1 + (class_id - config['split']['initial_classes']) // config['split']['increment'])
                row = {
                    'task': task_id,
                    'class_id': class_id,
                    'original_class_id': original_id,
                    'class_name': class_names[original_id],
                    'group': group,
                    'class_role': 'backbone_train' if group == 'base' else 'incremental_new',
                    'introduced_task': introduced_task,
                    'correct': n_correct,
                    'total': n_total,
                    'accuracy_percent': 100.0 * n_correct / n_total,
                }
                class_rows.append(row)
            group_accuracy = {}
            for group in ('base', 'incremental'):
                group_counts = [row for row in class_rows if row['group'] == group]
                n_total = sum(row['total'] for row in group_counts)
                group_accuracy[group] = (
                    100.0 * sum(row['correct'] for row in group_counts) / n_total
                    if n_total else None
                )
            session = {
                'task': task_id, 'classes_seen': sum(task_sizes[:task_id + 1]),
                'task_accuracies': task_accuracies,
                'seen_accuracy': float(np.average(task_accuracies, weights=task_sizes[:task_id + 1])),
                'base_accuracy_percent': group_accuracy['base'],
                'incremental_accuracy_percent': group_accuracy['incremental'],
                'per_class_accuracy': class_rows,
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
        if classifier_cfg['name'] == 'fecam':
            manifest['evaluation']['matrix_storage_by_task'] = matrix_storage_by_task
        manifest['memory'] = (memory_monitor.stop(cuda_device_index, measure_cuda_memory)
                              if memory_monitor else None)
        manifest['status'] = 'completed'
        write_json(output / 'summary.json', manifest)
        print(f'Completed smoke evaluation: {output}', flush=True)
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'error': str(exc)}
        if memory_monitor and memory_monitor._thread.is_alive():
            failure['memory'] = memory_monitor.stop(cuda_device_index, measure_cuda_memory)
        write_json(output / 'failure.json', failure)
        raise


if __name__ == '__main__':
    main()

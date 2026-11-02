"""CIFAR-100 from a pinned Hugging Face revision, without changing its splits."""

from collections import Counter
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision import transforms

DATASET_ID = 'uoft-cs/cifar100'
DATASET_REVISION = 'aadb3af77e9048adbea6b47c21a81e47dd092ae5'
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def cifar_transforms(augment=False):
    operations = []
    if augment:
        operations = [transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.ColorJitter(brightness=63 / 255),
                      transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def validate_cifar100(data):
    counts = {}
    for name, per_class in [('train', 500), ('test', 100)]:
        labels = list(data[name]['fine_label'])
        counter = Counter(labels)
        if counter != Counter({label: per_class for label in range(100)}):
            raise ValueError(f'Invalid CIFAR-100 {name}: expected 100 classes x {per_class} samples')
        counts[name] = len(labels)
    return counts


class _LocalCifarRows:
    """Small lazy view over the canonical CIFAR-100 Python pickle split."""

    def __init__(self, pixels, labels, names, fingerprint):
        self.pixels = pixels
        self.labels = labels
        self.features = {'fine_label': SimpleNamespace(names=names)}
        self._fingerprint = fingerprint

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if index == 'fine_label':
            return self.labels.tolist()
        return {'img': PILImage.fromarray(self.pixels[index]), 'fine_label': int(self.labels[index])}


def load_cifar100(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    local = root / 'cifar-100-python'
    if (local / 'train').is_file() and (local / 'test').is_file():
        with (local / 'meta').open('rb') as handle:
            names = pickle.load(handle, encoding='bytes')[b'fine_label_names']
        names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) for name in names]
        splits = {}
        for source, filename in (('train', 'train'), ('test', 'test')):
            with (local / filename).open('rb') as handle:
                batch = pickle.load(handle, encoding='bytes')
            pixels = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            fingerprint = f'cifar100-python-local-{source}'
            splits[source] = _LocalCifarRows(pixels, np.asarray(batch[b'fine_labels']), names, fingerprint)
        data = splits
        validate_cifar100(data)
        return data
    from datasets import load_dataset
    from huggingface_hub.constants import HF_HUB_CACHE

    # Reuse a complete, revision-addressed Hub snapshot without contacting HF.
    cached_split = (
        Path(HF_HUB_CACHE) / 'datasets--uoft-cs--cifar100' / 'snapshots'
        / DATASET_REVISION / 'cifar100'
    )
    train_parquet = cached_split / 'train-00000-of-00001.parquet'
    test_parquet = cached_split / 'test-00000-of-00001.parquet'
    if train_parquet.is_file() and test_parquet.is_file():
        data = load_dataset(
            'parquet',
            data_files={'train': str(train_parquet), 'test': str(test_parquet)},
            cache_dir=str(root / 'hf-cache'),
        )
        validate_cifar100(data)
        return data

    # revision/cache_dir are documented at https://huggingface.co/docs/datasets/loading
    try:
        data = load_dataset(DATASET_ID, revision=DATASET_REVISION, cache_dir=str(root / 'huggingface'))
    except (OSError, ConnectionError) as exc:
        raise RuntimeError(f'Cannot load CIFAR-100 into {root}; check network, disk and cached files') from exc
    validate_cifar100(data)
    return data


class TaskDataset(Dataset):
    def __init__(self, rows, labels, split, task, source, augment=False,
                 seen=False, samples_per_class=None):
        if source not in ('train', 'test'):
            raise ValueError('source must be train or test')
        if (source == 'train' and seen) or (source == 'test' and augment):
            raise ValueError('No replay of old train classes or augmentation of test images')
        if samples_per_class is not None and samples_per_class <= 0:
            raise ValueError('samples_per_class must be positive')
        if len(rows) != len(labels):
            raise ValueError('Image and label counts differ')
        self.rows = rows
        self.indices = split.indices(labels, task, seen=seen)
        if samples_per_class is not None:
            counts = Counter()
            selected = []
            for index in self.indices:
                label = int(labels[index])
                if counts[label] < samples_per_class:
                    selected.append(index)
                    counts[label] += 1
            self.indices = selected
        self.targets = split.remap([labels[i] for i in self.indices])
        expected = set(split.classes(task, seen=seen))
        if set(self.targets) != expected:
            raise ValueError('Selected task has missing classes')
        self.transform = cifar_transforms(augment)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image = self.rows[self.indices[index]]['img']
        if image.size != (32, 32) or image.mode != 'RGB':
            raise ValueError('Expected a 32x32 RGB CIFAR image')
        return self.transform(image), self.targets[index]

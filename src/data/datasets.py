"""CIFAR-100 from a pinned Hugging Face revision, without changing its splits."""

from collections import Counter
from pathlib import Path

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


def load_cifar100(root):
    from datasets import load_dataset

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
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

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.data.datasets import TaskDataset, cifar_transforms, validate_cifar100
from src.data.task_split import TaskSplit
from src.engine.extract_features import extract_features
from src.engine.train_base import load_checkpoint, save_checkpoint, train_base
from src.models.backbone import Backbone
from src.utils.seed import seed_everything


class DataTests(unittest.TestCase):
    def test_dataset_remaps_and_caps_each_class_without_changing_source(self):
        split = TaskSplit(total_classes=10, initial_classes=5, increment=5)
        labels = split.class_order * 3
        image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        rows = [{'img': image, 'fine_label': label} for label in labels]
        dataset = TaskDataset(rows, labels, split, 1, source='train', augment=False,
                              samples_per_class=2)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(set(dataset.targets), set(range(5, 10)))
        self.assertEqual(dataset[0][0].shape, (3, 32, 32))
        self.assertEqual(len(rows), 30)
        with self.assertRaises(ValueError):
            TaskDataset(rows, labels, split, 1, source='train', augment=True, seen=True)

    def test_eval_transform_is_deterministic_and_bad_counts_fail(self):
        image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
        transform = cifar_transforms(augment=False)
        self.assertTrue(torch.equal(transform(image), transform(image)))
        with self.assertRaises(ValueError):
            validate_cifar100({'train': {'fine_label': [0]}, 'test': {'fine_label': [0]}})


class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shapes_cosine_head_and_frozen_batchnorm(self):
        seed_everything(1)
        model = Backbone(5)
        images = torch.randn(3, 3, 32, 32)
        model.eval()
        features = model.extract_features(images)
        self.assertEqual(features.shape, (3, 512))
        self.assertTrue((features >= 0).all())
        self.assertEqual(model(images).shape, (3, 5))
        self.assertTrue(torch.allclose(model.head(features), model.head(features * 2), atol=1e-6))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        model.freeze()
        model.train()
        loader = DataLoader(TensorDataset(images, torch.tensor([0, 1, 2])), batch_size=2)
        vectors, labels = extract_features(model, loader, 'cpu')
        self.assertEqual(vectors.shape, (3, 512))
        self.assertEqual(labels.tolist(), [0, 1, 2])
        self.assertFalse(vectors.requires_grad)
        self.assertTrue(torch.isfinite(vectors).all())
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        with self.assertRaises(ValueError):
            extract_features(model, DataLoader(TensorDataset(images[:0], torch.tensor([], dtype=torch.long))), 'cpu')

    def test_training_updates_weights_and_checkpoint_roundtrip(self):
        seed_everything(1)
        model = Backbone(2)
        loader = DataLoader(TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 0, 1])), batch_size=2)
        before = model.encoder.conv1.weight.detach().clone()
        history = train_base(model, loader, epochs=1, lr=0.01, momentum=0.9,
                             weight_decay=0.0005, device='cpu')
        self.assertFalse(torch.equal(before, model.encoder.conv1.weight))
        self.assertEqual(history[0]['samples'], 4)
        self.assertTrue(np.isfinite(history[0]['loss']))
        model.eval()
        probe = torch.randn(2, 3, 32, 32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'base.pt'
            metadata = {'class_order': [7, 3], 'config': {'seed': 1}}
            save_checkpoint(path, model, metadata)
            restored, saved = load_checkpoint(path)
            self.assertEqual(saved, metadata)
            with torch.no_grad():
                torch.testing.assert_close(model(probe), restored(probe))
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, metadata)


if __name__ == '__main__':
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image
from PIL import Image as PILImage

from src.data.task_split import TaskSplit
from scripts.run_experiment import ROOT, load_config, main


class RunnerTests(unittest.TestCase):
    def test_reproduction_configs_share_training_and_cannot_be_truncated(self):
        fecam = load_config(ROOT / 'configs/reproduction/cifar100_fecam.yaml')
        ncm = load_config(ROOT / 'configs/reproduction/cifar100_ncm.yaml')
        for key in ('seed', 'dataset', 'split', 'backbone', 'training', 'loader'):
            self.assertEqual(fecam[key], ncm[key])
        self.assertEqual(fecam['training']['weight_decay'], 0.0001)
        smoke = load_config(ROOT / 'configs/smoke.yaml')
        self.assertEqual(smoke['training']['weight_decay'], 0.0001)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.yaml'
            fecam['loader']['train_samples_per_class'] = 1
            path.write_text(yaml.safe_dump(fecam), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_config(path)

    def test_train_save_reload_and_extract_next_task_without_network(self):
        torch.set_num_threads(2)
        labels = TaskSplit(total_classes=10, initial_classes=5, increment=5).class_order * 2
        images = [PILImage.fromarray(np.full((32, 32, 3), i % 255, dtype=np.uint8))
                  for i in range(len(labels))]
        rows = Dataset.from_dict({'img': images, 'fine_label': labels}, features=Features({
            'img': Image(), 'fine_label': ClassLabel(names=[str(i) for i in range(100)])}))
        data = DatasetDict(train=rows, test=rows)
        with tempfile.TemporaryDirectory() as folder, patch('scripts.run_experiment.load_cifar100', return_value=data):
            base = Path(folder) / 'base'
            features = Path(folder) / 'features'
            config_path = str(ROOT / 'configs/smoke.yaml')
            main(['train-base', '--config', config_path, '--output', str(base), '--device', 'cpu'])
            main(['extract-features', '--config', config_path, '--checkpoint', str(base / 'base.pt'),
                  '--task', '1', '--output', str(features), '--device', 'cpu'])
            summary = json.loads((features / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['features']['train']['labels'], [5, 6, 7, 8, 9])
            self.assertEqual(summary['features']['test']['labels'], list(range(10)))
            self.assertEqual(summary['features']['train']['shape'], [10, 512])
            self.assertEqual(summary['features']['test']['shape'], [20, 512])
            self.assertFalse(list(features.glob('*.pt')))
            with self.assertRaises(FileExistsError):
                main(['prepare', '--config', config_path, '--output', str(base), '--device', 'cpu'])


if __name__ == '__main__':
    unittest.main()

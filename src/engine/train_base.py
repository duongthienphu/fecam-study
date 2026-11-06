import math
import os
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from src.models.backbone import Backbone


def _rng_state(device, loader):
    cuda_device = torch.device(device)
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state(cuda_device) if cuda_device.type == 'cuda' else None,
        'loader_generator': loader_generator_state(loader),
    }


def loader_generator_state(loader):
    generator = getattr(loader, 'generator', None)
    return generator.get_state() if generator is not None else None


def _restore_rng_state(state, loader, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    cuda_device = torch.device(device)
    if cuda_device.type == 'cuda' and state['torch_cuda'] is not None:
        torch.cuda.set_rng_state(state['torch_cuda'], cuda_device)
    generator = getattr(loader, 'generator', None)
    if generator is not None and state['loader_generator'] is not None:
        generator.set_state(state['loader_generator'])


def _save_training_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def train_base(model, loader, epochs, lr, momentum, weight_decay, device, on_epoch=None,
               checkpoint_path=None, resume=False, checkpoint_metadata=None):
    if epochs <= 0 or lr <= 0 or not 0 <= momentum < 1 or weight_decay < 0:
        raise ValueError('Invalid training hyperparameters')
    if model.frozen:
        raise ValueError('Cannot train a frozen backbone')
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    start_epoch = 0
    prior_training_wall_seconds = 0.0
    training_started = perf_counter()
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and checkpoint_path is None:
        raise ValueError('resume requires checkpoint_path')
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Training checkpoint not found: {checkpoint_path}')
        # This checkpoint is created locally by this training code and includes RNG state.
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if state.get('checkpoint_version') != 1:
            raise ValueError('Unsupported training checkpoint version')
        if state.get('metadata') != checkpoint_metadata:
            raise ValueError('Training checkpoint metadata does not match this config/code run')
        if state['total_epochs'] != epochs:
            raise ValueError('Training checkpoint total_epochs differs from config')
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        start_epoch = int(state['completed_epochs'])
        history = state['history']
        prior_training_wall_seconds = float(state.get('cumulative_training_wall_seconds', 0.0))
        if not 0 <= start_epoch <= epochs or len(history) != start_epoch:
            raise ValueError('Invalid completed epoch count in training checkpoint')
        _restore_rng_state(state['rng_state'], loader, device)
        print(f'Resuming backbone training at epoch {start_epoch + 1}/{epochs}', flush=True)
    elif checkpoint_path is not None and checkpoint_path.exists():
        raise FileExistsError(f'Training checkpoint already exists: {checkpoint_path}; use resume=True')

    for epoch in range(start_epoch, epochs):
        model.train()
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        start = perf_counter()
        total_loss, correct, count = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if (labels < 0).any() or (labels >= model.num_classes).any():
                raise ValueError('Base training contains labels outside the initial task')
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite base training loss')
            loss.backward()
            optimizer.step()
            count += labels.numel()
            total_loss += loss.item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
        if not count:
            raise ValueError('Empty base training loader')
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        record = dict(epoch=epoch + 1, samples=count, loss=total_loss / count,
                      train_accuracy=100 * correct / count,
                      lr=optimizer.param_groups[0]['lr'], seconds=perf_counter() - start)
        if not math.isfinite(record['loss']):
            raise ValueError('Non-finite epoch loss')
        history.append(record)
        scheduler.step()
        if checkpoint_path is not None:
            record['cumulative_training_wall_seconds'] = (
                prior_training_wall_seconds + perf_counter() - training_started
            )
            _save_training_checkpoint(checkpoint_path, {
                'checkpoint_version': 1,
                'completed_epochs': epoch + 1,
                'total_epochs': epochs,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'history': history,
                'cumulative_training_wall_seconds': record['cumulative_training_wall_seconds'],
                'rng_state': _rng_state(device, loader),
                'metadata': checkpoint_metadata,
            })
        if on_epoch:
            on_epoch(record)
    model.eval()
    return history


def save_checkpoint(path, model, metadata):
    """Save final model weights and plain metadata atomically; never overwrite a run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    try:
        torch.save({'num_classes': model.num_classes, 'model_state': model.state_dict(),
                    'metadata': metadata}, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path, device='cpu'):
    # https://docs.pytorch.org/docs/2.7/notes/serialization.html
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = Backbone(checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state'], strict=True)
    return model.to(device).freeze(), checkpoint['metadata']

import math
import os
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from src.models.backbone import Backbone


def train_base(model, loader, epochs, lr, momentum, weight_decay, device, on_epoch=None):
    if epochs <= 0 or lr <= 0 or not 0 <= momentum < 1 or weight_decay < 0:
        raise ValueError('Invalid training hyperparameters')
    if model.frozen:
        raise ValueError('Cannot train a frozen backbone')
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for epoch in range(epochs):
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
        if on_epoch:
            on_epoch(record)
    model.eval()
    return history


def save_checkpoint(path, model, metadata):
    """Save weights and plain metadata atomically; never overwrite an existing run."""
    # shortcut: final weights only; add optimizer/RNG state before interruption-prone long runs.
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
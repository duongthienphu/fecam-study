import torch


@torch.inference_mode()
def extract_features(model, loader, device):
    """Return raw CPU features [N, 512] and remapped labels [N], without a disk cache."""
    model.eval()
    features, labels = [], []
    for images, targets in loader:
        vectors = model.extract_features(images.to(device))
        if vectors.ndim != 2 or vectors.shape[1] != model.feature_dim or not torch.isfinite(vectors).all():
            raise ValueError('Invalid backbone features')
        features.append(vectors.cpu())
        labels.append(targets.cpu().long())
    if not features:
        raise ValueError('Cannot extract features from an empty loader')
    return torch.cat(features), torch.cat(labels)
"""Library ResNet-18 with the CIFAR stem and a trainable cosine head."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class CosineHead(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, 512))
        self.scale = nn.Parameter(torch.ones(1))
        nn.init.uniform_(self.weight, -1 / math.sqrt(512), 1 / math.sqrt(512))

    def forward(self, features):
        return self.scale * F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1))


class Backbone(nn.Module):
    feature_dim = 512

    def __init__(self, num_classes):
        super().__init__()
        if not 1 <= num_classes <= 100:
            raise ValueError('num_classes must be between 1 and 100')
        self.num_classes = num_classes
        # Reuse torchvision's implementation; CIFAR uses 3x3/stride 1 and no maxpool.
        self.encoder = resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.encoder.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        self.head = CosineHead(num_classes)
        self.frozen = False

    def extract_features(self, images):
        return self.encoder(images)

    def forward(self, images):
        return self.head(self.extract_features(images))

    def freeze(self):
        self.frozen = True
        self.requires_grad_(False)
        return self.eval()

    def train(self, mode=True):
        # Frozen BatchNorm must remain in eval mode even if a caller calls train().
        return super().train(False if self.frozen else mode)
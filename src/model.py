from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import ArcFaceLoss
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGE_SIZE = 112
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ARCFACE_MARGIN_DEGREES = 28.6
ARCFACE_SCALE = 64


@dataclass(frozen=True, slots=True)
class EmbedderConfig:
    embedding_dim: int = 128
    pretrained: bool = True


def face_to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image with shape (112, 112, 3)")

    return torch.from_numpy(image).permute(2, 0, 1).float().div(255)


class FaceEmbedder(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        super().__init__()
        config = config or EmbedderConfig()

        weights = ResNet18_Weights.IMAGENET1K_V1 if config.pretrained else None
        self.embedding_dim = config.embedding_dim
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, config.embedding_dim)

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean) / self.std
        embeddings = self.backbone(images)
        return F.normalize(embeddings, dim=1)


def create_arcface_loss(number_of_classes: int, embedding_dim: int) -> ArcFaceLoss:
    return ArcFaceLoss(
        num_classes=number_of_classes,
        embedding_size=embedding_dim,
        margin=ARCFACE_MARGIN_DEGREES,
        scale=ARCFACE_SCALE,
    )

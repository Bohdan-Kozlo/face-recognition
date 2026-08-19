from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from config import FACE_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

EMBEDDING_DIM = 512
BACKBONE_NAME = "resnet50_imagenet1k_v2"
CHECKPOINT_FORMAT_VERSION = 1

FineTuningMode = Literal["last-layer", "all"]


class FaceEmbedder(nn.Module):
    def __init__(
        self, *, weights: ResNet50_Weights | None = ResNet50_Weights.IMAGENET1K_V2
    ) -> None:
        super().__init__()
        backbone = resnet50(weights=weights)
        input_features = backbone.fc.in_features
        backbone.fc = nn.Linear(input_features, EMBEDDING_DIM, bias=False)
        self.backbone = backbone
        self._fine_tuning_mode: FineTuningMode = "last-layer"
        self.set_fine_tuning_mode(self._fine_tuning_mode)

    def set_fine_tuning_mode(self, mode: FineTuningMode) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = mode == "all"
        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True
        self._fine_tuning_mode = mode

    def optimizer_parameter_groups(
        self,
        *,
        backbone_learning_rate: float,
        embedding_learning_rate: float,
    ) -> list[dict[str, object]]:
        groups: list[dict[str, object]] = [
            {
                "name": "embedding",
                "params": self.backbone.fc.parameters(),
                "lr": embedding_learning_rate,
            }
        ]
        if self._fine_tuning_mode == "all":
            groups.insert(
                0,
                {
                    "name": "backbone",
                    "params": (
                        parameter
                        for name, parameter in self.backbone.named_parameters()
                        if name != "fc.weight"
                    ),
                    "lr": backbone_learning_rate,
                },
            )
        return groups

    def train(self, mode: bool = True) -> FaceEmbedder:
        super().train(mode)
        if mode and self._fine_tuning_mode == "last-layer":
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.backbone(images)
        return F.normalize(embeddings, dim=1)


def checkpoint_metadata(*, fine_tuning: FineTuningMode) -> dict[str, str | int]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "backbone": BACKBONE_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "face_image_size": FACE_IMAGE_SIZE,
        "normalization": f"imagenet:{IMAGENET_MEAN}:{IMAGENET_STD}",
        "fine_tuning": fine_tuning,
    }


def validate_checkpoint_metadata(checkpoint: Mapping[str, Any]) -> FineTuningMode:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint has no ResNet50 metadata and cannot be loaded")

    expected = checkpoint_metadata(fine_tuning="last-layer")
    for field in (
        "format_version",
        "backbone",
        "embedding_dim",
        "face_image_size",
        "normalization",
    ):
        if metadata.get(field) != expected[field]:
            raise ValueError(f"Checkpoint is incompatible: expected {field}={expected[field]!r}")

    fine_tuning = metadata.get("fine_tuning")
    if fine_tuning not in {"last-layer", "all"}:
        raise ValueError("Checkpoint has an unknown fine-tuning mode")
    return fine_tuning

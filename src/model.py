from __future__ import annotations

import torch
from facenet_pytorch import InceptionResnetV1
from torch import nn

from config import FACE_IMAGE_SIZE

EMBEDDING_DIM = 512
BACKBONE_NAME = "inception_resnet_v1_vggface2"


class FaceEmbedder(nn.Module):
    """Create face embeddings with a VGGFace2-pretrained InceptionResnetV1."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, *, pretrained: bool = True) -> None:
        super().__init__()

        if embedding_dim != EMBEDDING_DIM:
            raise ValueError(f"InceptionResnetV1 requires embedding_dim={EMBEDDING_DIM}")
        self.embedding_dim = embedding_dim
        backbone = InceptionResnetV1(
            pretrained="vggface2" if pretrained else None,
            classify=False,
        )
        if hasattr(backbone, "logits"):
            del backbone.logits
        self.backbone: nn.Module = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (FACE_IMAGE_SIZE, FACE_IMAGE_SIZE):
            raise ValueError(f"Expected {FACE_IMAGE_SIZE}x{FACE_IMAGE_SIZE} face images")
        images = images * 2.0 - 1.0
        return self.backbone(images)

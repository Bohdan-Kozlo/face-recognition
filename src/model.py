from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from config import FACE_IMAGE_SIZE

EMBEDDING_DIM = 512
BACKBONE_NAME = "edgeface_s_gamma_05"
EDGEFACE_REVISION = "ce86851cfc37979a9cd2558598d0e9bc592cbba3"
EDGEFACE_REPOSITORY = f"otroshi/edgeface:{EDGEFACE_REVISION}"
EDGEFACE_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/otroshi/edgeface/"
    f"{EDGEFACE_REVISION}/checkpoints/edgeface_s_gamma_05.pt"
)

FineTuningStage = Literal["frozen", "last_stage", "all"]


class FaceEmbedder(nn.Module):
    """Create normalized face embeddings with pretrained EdgeFace-S."""

    def __init__(self) -> None:
        super().__init__()
        backbone = torch.hub.load(
            EDGEFACE_REPOSITORY,
            BACKBONE_NAME,
            source="github",
            # Build the architecture; pinned pretrained weights are loaded below.
            pretrained=False,
            trust_repo=True,  # pyright: ignore[reportArgumentType]
            skip_validation=True,
        )
        if not isinstance(backbone, nn.Module):
            raise TypeError("EdgeFace repository did not return a PyTorch module")
        state_dict = torch.hub.load_state_dict_from_url(
            EDGEFACE_WEIGHTS_URL,
            map_location="cpu",
            file_name=f"{BACKBONE_NAME}_{EDGEFACE_REVISION[:8]}.pt",
            weights_only=True,
        )
        backbone.load_state_dict(state_dict)
        self.backbone = backbone

    def set_fine_tuning_stage(self, stage: FineTuningStage) -> None:
        """Select how much of the pretrained backbone can be updated."""

        if stage not in {"frozen", "last_stage", "all"}:
            raise ValueError(f"Unknown fine-tuning stage: {stage}")

        for parameter in self.backbone.parameters():
            parameter.requires_grad = stage == "all"

        if stage == "last_stage":
            for module_name in ("model.stages.3", "model.head"):
                for parameter in self.backbone.get_submodule(module_name).parameters():
                    parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (FACE_IMAGE_SIZE, FACE_IMAGE_SIZE):
            raise ValueError(f"Expected {FACE_IMAGE_SIZE}x{FACE_IMAGE_SIZE} face images")
        embeddings = self.backbone(images * 2.0 - 1.0)
        return F.normalize(embeddings, dim=1)

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IMAGE_SIZE = 112
BACKBONE_NAME = "iresnet34"


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_channels: int, out_channels: int, stride: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class _IBasicBlock(nn.Module):
    """Pre-activation residual block used by InsightFace IResNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = _conv3x3(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = _conv3x3(out_channels, out_channels, stride)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(
                _conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs

        outputs = self.bn1(inputs)
        outputs = self.conv1(outputs)
        outputs = self.bn2(outputs)
        outputs = self.prelu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn3(outputs)

        if self.downsample is not None:
            identity = self.downsample(identity)
        return outputs + identity


class _IResNet34(nn.Module):
    """InsightFace IResNet34 backbone for aligned 112x112 faces."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.in_channels = 64
        self.conv1 = _conv3x3(3, self.in_channels)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.prelu = nn.PReLU(self.in_channels)
        self.layer1 = self._make_layer(64, blocks=3)
        self.layer2 = self._make_layer(128, blocks=4)
        self.layer3 = self._make_layer(256, blocks=6)
        self.layer4 = self._make_layer(512, blocks=3)
        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim)
        self.features = nn.BatchNorm1d(embedding_dim)
        self.features.weight.requires_grad_(False)

        self._initialize_weights()

    def _make_layer(self, out_channels: int, blocks: int) -> nn.Sequential:
        layers: list[nn.Module] = [_IBasicBlock(self.in_channels, out_channels, stride=2)]
        self.in_channels = out_channels
        layers.extend(_IBasicBlock(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = self.conv1(images.float())
            features = self.bn1(features)
            features = self.prelu(features)
            features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.bn2(features)
        features = torch.flatten(features, 1)
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = self.fc(features.float())
            return self.features(features)


class FaceEmbedder(nn.Module):
    """Convert aligned face images into L2-normalized IResNet34 embeddings."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.backbone = _IResNet34(embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images * 2.0 - 1.0
        embeddings = self.backbone(images)
        return F.normalize(embeddings, dim=1)

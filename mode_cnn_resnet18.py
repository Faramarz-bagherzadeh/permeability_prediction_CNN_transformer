"""A compact ResNet-18-style regressor built from 3D convolutions."""

import torch
from torch import nn


class ResidualBlock3D(nn.Module):
    """Two-convolution residual block for volumetric inputs."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.activation(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        return self.activation(outputs + residual)


class ResNet18CNN3D(nn.Module):
    """Map one-channel 3D images to three directional permeability values."""

    def __init__(self, output_features: int = 3) -> None:
        super().__init__()
        if output_features != 3:
            raise ValueError("The permeability model must predict exactly three values.")

        self.in_channels = 16
        self.stem = nn.Sequential(
            nn.Conv3d(1, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(16, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, blocks=2, stride=2)
        self.layer4 = self._make_layer(128, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.regressor = nn.Linear(128, output_features)

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock3D(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(ResidualBlock3D(channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError("Input must have shape (batch, 1, depth, height, width).")
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        return self.regressor(self.pool(outputs).flatten(1))
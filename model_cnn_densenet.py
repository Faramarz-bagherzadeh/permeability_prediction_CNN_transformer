"""A compact DenseNet-style regressor built from 3D convolutions."""

import torch
from torch import nn


class DenseLayer3D(nn.Module):
    """Append newly learned feature maps to the dense block input."""

    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        bottleneck_channels = 4 * growth_rate
        self.layers = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(inputs)
        return torch.cat((inputs, new_features), dim=1)


class DenseBlock3D(nn.Module):
    def __init__(self, in_channels: int, layers: int, growth_rate: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *[
                DenseLayer3D(in_channels + index * growth_rate, growth_rate)
                for index in range(layers)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DenseTransition3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )


class DenseNetCNN3D(nn.Module):
    """Map one-channel 3D images to three directional permeability values."""

    def __init__(self, output_features: int = 3, growth_rate: int = 8) -> None:
        super().__init__()
        if output_features != 3:
            raise ValueError("The permeability model must predict exactly three values.")
        if growth_rate < 1:
            raise ValueError("growth_rate must be positive.")

        channels = 16
        self.stem = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.block1 = DenseBlock3D(channels, layers=4, growth_rate=growth_rate)
        channels += 4 * growth_rate
        self.transition1 = DenseTransition3D(channels, channels // 2)
        channels //= 2

        self.block2 = DenseBlock3D(channels, layers=6, growth_rate=growth_rate)
        channels += 6 * growth_rate
        self.transition2 = DenseTransition3D(channels, channels // 2)
        channels //= 2

        self.block3 = DenseBlock3D(channels, layers=8, growth_rate=growth_rate)
        channels += 8 * growth_rate
        self.final_norm = nn.BatchNorm3d(channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.regressor = nn.Linear(channels, output_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError("Input must have shape (batch, 1, depth, height, width).")
        outputs = self.stem(inputs)
        outputs = self.block1(outputs)
        outputs = self.transition1(outputs)
        outputs = self.block2(outputs)
        outputs = self.transition2(outputs)
        outputs = self.block3(outputs)
        outputs = torch.relu(self.final_norm(outputs))
        return self.regressor(self.pool(outputs).flatten(1))
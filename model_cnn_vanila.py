"""A compact 3D CNN regressor for directional permeability."""

from torch import nn


class VanillaCNN3D(nn.Module):
    """Map a one-channel 3D image to (kx, ky, kz)."""

    def __init__(self, output_features: int = 3) -> None:
        super().__init__()
        if output_features != 3:
            raise ValueError("The permeability model must predict exactly three values.")

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, output_features),
        )

    def forward(self, inputs):
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError("Input must have shape (batch, 1, depth, height, width).")
        return self.regressor(self.features(inputs))
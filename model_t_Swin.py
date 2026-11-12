"""MONAI 3D Swin Transformer adapted for permeability regression."""

import importlib.util

import torch
from torch import nn


class MONAISwin3DRegressor(nn.Module):
    """Pool hierarchical Swin features to predict three permeability values."""

    def __init__(self, image_size: int = 64, output_features: int = 3) -> None:
        super().__init__()
        if image_size < 8:
            raise ValueError("image_size must be at least 8.")
        if output_features != 3:
            raise ValueError("The permeability model must predict exactly three values.")

        try:
            from monai.networks.nets.swin_unetr import SwinTransformer
        except ImportError as error:
            raise ImportError(
                "The monai_swin3d model requires MONAI. Install it in the training "
                "environment with: pip install monai"
            ) from error
        if importlib.util.find_spec("einops") is None:
            raise ImportError(
                "The monai_swin3d model requires einops for MONAI's Swin implementation. "
                "Install it in the training environment with: pip install einops"
            )

        self.image_size = image_size
        self.backbone = SwinTransformer(
            in_chans=1,
            embed_dim=24,
            window_size=(4, 4, 4),
            patch_size=(2, 2, 2),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            spatial_dims=3,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.regressor = nn.Linear(self.backbone.num_features, output_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError("Input must have shape (batch, 1, depth, height, width).")
        if tuple(inputs.shape[2:]) != (self.image_size,) * 3:
            raise ValueError(
                f"Input spatial dimensions must be {(self.image_size,) * 3}, "
                f"got {tuple(inputs.shape[2:])}."
            )
        features = self.backbone(inputs)[-1]
        pooled = self.pool(features).flatten(1)
        return self.regressor(pooled)
"""MONAI 3D Vision Transformer adapted for permeability regression."""

import torch
from torch import nn


class MONAIViT3DRegressor(nn.Module):
    """Predict three directional permeability values from a 3D image."""

    def __init__(
        self,
        image_size: int | tuple[int, int, int] = 64,
        output_features: int = 3,
    ) -> None:
        super().__init__()
        spatial_size = (image_size,) * 3 if isinstance(image_size, int) else tuple(image_size)
        if len(spatial_size) != 3 or any(size < 8 for size in spatial_size):
            raise ValueError("image_size must be an integer or three dimensions of at least 8.")
        if output_features != 3:
            raise ValueError("The permeability model must predict exactly three values.")

        try:
            from monai.networks.nets import ViT
        except ImportError as error:
            raise ImportError(
                "The monai_vit3d model requires MONAI. Install it in the training "
                "environment with: pip install monai"
            ) from error

        patch_size = tuple(max(1, size // 4) for size in spatial_size)
        self.image_size = spatial_size
        self.backbone = ViT(
            in_channels=1,
            img_size=spatial_size,
            patch_size=patch_size,
            hidden_size=96,
            mlp_dim=384,
            num_layers=4,
            num_heads=4,
            classification=True,
            num_classes=output_features,
            spatial_dims=3,
            post_activation=None,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError("Input must have shape (batch, 1, depth, height, width).")
        if tuple(inputs.shape[2:]) != self.image_size:
            raise ValueError(
                f"Input spatial dimensions must be {self.image_size}, "
                f"got {tuple(inputs.shape[2:])}."
            )
        predictions, _ = self.backbone(inputs)
        return predictions

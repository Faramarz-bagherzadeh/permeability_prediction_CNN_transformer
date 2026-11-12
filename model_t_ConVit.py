"""A compact 3D ConViT regressor with gated positional self-attention."""

import torch
from torch import nn
from torch.nn import functional as torch_functional


class GatedPositionalSelfAttention3D(nn.Module):
    """Blend content attention with a learned 3D positional attention prior."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Parameter(torch.full((num_heads,), -2.0))
        self.positional_weights = nn.Parameter(torch.zeros(num_heads, 6))
        with torch.no_grad():
            self.positional_weights[:, 3:].fill_(-2.0)

    def _positional_attention(
        self,
        grid_size: tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        axes = [
            torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=dtype)
            for size in grid_size
        ]
        coordinates = torch.stack(
            torch.meshgrid(*axes, indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)
        offsets = coordinates[:, None, :] - coordinates[None, :, :]
        positional_features = torch.cat((offsets, offsets.square()), dim=-1)
        positional_logits = torch.einsum(
            "ijc,hc->hij",
            positional_features,
            self.positional_weights.to(dtype=dtype),
        )
        return positional_logits.softmax(dim=-1)

    def forward(
        self,
        inputs: torch.Tensor,
        grid_size: tuple[int, int, int],
    ) -> torch.Tensor:
        batch_size, token_count, _ = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        content_attention = torch.matmul(
            query * self.scale,
            key.transpose(-2, -1),
        ).softmax(dim=-1)
        positional_attention = self._positional_attention(
            grid_size,
            inputs.device,
            inputs.dtype,
        ).unsqueeze(0)
        content_weight = self.gate.sigmoid().view(1, self.num_heads, 1, 1)
        attention = (
            (1.0 - content_weight) * positional_attention
            + content_weight * content_attention
        )
        outputs = torch.matmul(attention, value)
        outputs = outputs.transpose(1, 2).reshape(batch_size, token_count, self.embed_dim)
        return self.projection(outputs)


class ConViTBlock3D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_gated_positional_attention: bool,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if use_gated_positional_attention:
            self.attention = GatedPositionalSelfAttention3D(embed_dim, num_heads)
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                batch_first=True,
            )
        self.use_gated_positional_attention = use_gated_positional_attention
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        grid_size: tuple[int, int, int],
    ) -> torch.Tensor:
        normalized = self.norm1(inputs)
        if self.use_gated_positional_attention:
            attended = self.attention(normalized, grid_size)
        else:
            attended = self.attention(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )[0]
        outputs = inputs + attended
        return outputs + self.mlp(self.norm2(outputs))


class ConViT3D(nn.Module):
    """Map a one-channel 3D image to three directional permeability values."""

    def __init__(
        self,
        output_features: int = 3,
        embed_dim: int = 96,
        num_heads: int = 4,
        gpsa_layers: int = 2,
        attention_layers: int = 2,
    ) -> None:
        super().__init__()
        if output_features != 3:
            raise ValueError("The permeability model must predict exactly three values.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if gpsa_layers < 1 or attention_layers < 1:
            raise ValueError("The model must have at least one layer of each attention type.")

        self.patch_embed = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=8,
            stride=8,
        )
        self.max_grid_size = 4
        self.position_embedding = nn.Parameter(
            torch.zeros(1, embed_dim, self.max_grid_size, self.max_grid_size, self.max_grid_size)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            [
                *[
                    ConViTBlock3D(embed_dim, num_heads, use_gated_positional_attention=True)
                    for _ in range(gpsa_layers)
                ],
                *[
                    ConViTBlock3D(embed_dim, num_heads, use_gated_positional_attention=False)
                    for _ in range(attention_layers)
                ],
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.regressor = nn.Linear(embed_dim, output_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError("Input must have shape (batch, 1, depth, height, width).")
        if any(size < 8 for size in inputs.shape[2:]):
            raise ValueError("Each image dimension must be at least 8 for patch embedding.")

        features = self.patch_embed(inputs)
        grid_size = tuple(min(size, self.max_grid_size) for size in features.shape[2:])
        if features.shape[2:] != grid_size:
            features = torch_functional.adaptive_avg_pool3d(features, grid_size)
        position_embedding = self.position_embedding
        if position_embedding.shape[2:] != grid_size:
            position_embedding = torch_functional.interpolate(
                position_embedding,
                size=grid_size,
                mode="trilinear",
                align_corners=False,
            )
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + position_embedding.flatten(2).transpose(1, 2)

        for block in self.blocks:
            tokens = block(tokens, grid_size)
        pooled = self.norm(tokens).mean(dim=1)
        return self.regressor(pooled)
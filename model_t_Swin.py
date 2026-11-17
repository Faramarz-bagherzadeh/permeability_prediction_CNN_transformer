"""A pure-PyTorch 3D Swin Transformer for permeability regression."""

from itertools import product

import torch
from torch import nn
from torch.nn import functional as torch_functional


def _window_partition(
    inputs: torch.Tensor,
    window_size: tuple[int, int, int],
) -> torch.Tensor:
    batch_size, depth, height, width, channels = inputs.shape
    window_depth, window_height, window_width = window_size
    inputs = inputs.reshape(
        batch_size,
        depth // window_depth,
        window_depth,
        height // window_height,
        window_height,
        width // window_width,
        window_width,
        channels,
    )
    inputs = inputs.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return inputs.reshape(-1, window_depth * window_height * window_width, channels)


def _window_reverse(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    batch_size: int,
    spatial_size: tuple[int, int, int],
) -> torch.Tensor:
    depth, height, width = spatial_size
    window_depth, window_height, window_width = window_size
    channels = windows.shape[-1]
    inputs = windows.reshape(
        batch_size,
        depth // window_depth,
        height // window_height,
        width // window_width,
        window_depth,
        window_height,
        window_width,
        channels,
    )
    inputs = inputs.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return inputs.reshape(batch_size, depth, height, width, channels)


def _shifted_window_attention_mask(
    spatial_size: tuple[int, int, int],
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor | None:
    if not any(shift_size):
        return None

    slices = []
    for window, shift in zip(window_size, shift_size):
        if shift:
            slices.append((slice(0, -window), slice(-window, -shift), slice(-shift, None)))
        else:
            slices.append((slice(None),))

    region_ids = torch.zeros((1, *spatial_size, 1), device=device)
    for region_id, region in enumerate(product(*slices)):
        region_ids[(slice(None), *region, slice(None))] = region_id
    window_ids = _window_partition(region_ids, window_size).squeeze(-1)
    mask = window_ids.unsqueeze(1) - window_ids.unsqueeze(2)
    return mask.masked_fill(mask != 0, -100.0)


class WindowAttention3D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = window_size
        self.qkv = nn.Linear(channels, channels * 3)
        self.projection = nn.Linear(channels, channels)
        relative_bias_size = tuple(2 * size - 1 for size in window_size)
        self.relative_position_bias = nn.Parameter(
            torch.zeros(
                relative_bias_size[0] * relative_bias_size[1] * relative_bias_size[2],
                num_heads,
            )
        )
        coordinates = torch.stack(
            torch.meshgrid(
                *(torch.arange(size) for size in window_size),
                indexing="ij",
            )
        ).flatten(1)
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).contiguous()
        relative_coordinates[..., 0] += window_size[0] - 1
        relative_coordinates[..., 1] += window_size[1] - 1
        relative_coordinates[..., 2] += window_size[2] - 1
        relative_coordinates[..., 0] *= relative_bias_size[1] * relative_bias_size[2]
        relative_coordinates[..., 1] *= relative_bias_size[2]
        self.register_buffer(
            "relative_position_index",
            relative_coordinates.sum(dim=-1),
            persistent=False,
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(
        self,
        windows: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        window_batches, token_count, channels = windows.shape
        qkv = self.qkv(windows).reshape(
            window_batches,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        relative_bias = self.relative_position_bias[
            self.relative_position_index.reshape(-1)
        ].reshape(token_count, token_count, self.num_heads)
        attention = attention + relative_bias.permute(2, 0, 1).to(
            dtype=attention.dtype
        ).unsqueeze(0)

        if attention_mask is not None:
            window_count = attention_mask.shape[0]
            batch_size = window_batches // window_count
            attention = attention.reshape(
                batch_size,
                window_count,
                self.num_heads,
                token_count,
                token_count,
            )
            attention = attention + attention_mask.to(attention.dtype)[
                None, :, None, :, :
            ]
            attention = attention.reshape(
                window_batches,
                self.num_heads,
                token_count,
                token_count,
            )

        attention = attention.softmax(dim=-1)
        outputs = (attention @ value).transpose(1, 2).reshape(
            window_batches, token_count, channels
        )
        return self.projection(outputs)


class SwinTransformerBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: tuple[int, int, int],
        shift_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(channels)
        self.attention = WindowAttention3D(channels, num_heads, window_size)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, depth, height, width, channels = inputs.shape
        residual = inputs
        outputs = self.norm1(inputs)
        pad_depth = (self.window_size[0] - depth % self.window_size[0]) % self.window_size[0]
        pad_height = (self.window_size[1] - height % self.window_size[1]) % self.window_size[1]
        pad_width = (self.window_size[2] - width % self.window_size[2]) % self.window_size[2]
        outputs = torch_functional.pad(
            outputs,
            (0, 0, 0, pad_width, 0, pad_height, 0, pad_depth),
        )
        padded_size = (
            depth + pad_depth,
            height + pad_height,
            width + pad_width,
        )
        if any(self.shift_size):
            outputs = torch.roll(
                outputs,
                shifts=tuple(-shift for shift in self.shift_size),
                dims=(1, 2, 3),
            )

        attention_mask = _shifted_window_attention_mask(
            padded_size,
            self.window_size,
            self.shift_size,
            inputs.device,
        )
        windows = _window_partition(outputs, self.window_size)
        windows = self.attention(windows, attention_mask)
        outputs = _window_reverse(windows, self.window_size, batch_size, padded_size)
        if any(self.shift_size):
            outputs = torch.roll(outputs, shifts=self.shift_size, dims=(1, 2, 3))
        outputs = outputs[:, :depth, :height, :width, :].contiguous()
        outputs = residual + outputs
        return outputs + self.mlp(self.norm2(outputs))


class PatchMerging3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels * 8)
        self.reduction = nn.Linear(channels * 8, channels * 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, depth, height, width, _ = inputs.shape
        inputs = torch_functional.pad(
            inputs,
            (0, 0, 0, width % 2, 0, height % 2, 0, depth % 2),
        )
        features = (
            inputs[:, 0::2, 0::2, 0::2, :],
            inputs[:, 1::2, 0::2, 0::2, :],
            inputs[:, 0::2, 1::2, 0::2, :],
            inputs[:, 0::2, 0::2, 1::2, :],
            inputs[:, 1::2, 1::2, 0::2, :],
            inputs[:, 1::2, 0::2, 1::2, :],
            inputs[:, 0::2, 1::2, 1::2, :],
            inputs[:, 1::2, 1::2, 1::2, :],
        )
        return self.reduction(self.norm(torch.cat(features, dim=-1)))


class Swin3DRegressor(nn.Module):
    """Pool hierarchical 3D Swin features to predict three permeability values."""

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

        self.image_size = spatial_size
        self.patch_embed = nn.Conv3d(1, 32, kernel_size=4, stride=4)
        #self.patch_embed = nn.Conv3d(1, 24, kernel_size=2, stride=2) out of memory issue
        self.stages = nn.ModuleList()
        self.patch_merging = nn.ModuleList()
        stage_channels = (32, 64, 128, 256)
        stage_heads = (4, 8, 16, 32)
        window_size = (4, 4, 4)
        for stage_index, (channels, heads) in enumerate(
            zip(stage_channels, stage_heads)
        ):
            self.stages.append(
                nn.Sequential(
                    *[
                        SwinTransformerBlock3D(
                            channels,
                            heads,
                            window_size,
                            (0, 0, 0) if block_index % 2 == 0 else (2, 2, 2),
                        )
                        for block_index in range(2)
                    ]
                )
            )
            if stage_index < len(stage_channels) - 1:
                self.patch_merging.append(PatchMerging3D(channels))

        self.norm = nn.LayerNorm(stage_channels[-1])
        self.regressor = nn.Linear(stage_channels[-1], output_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError("Input must have shape (batch, 1, depth, height, width).")
        if tuple(inputs.shape[2:]) != self.image_size:
            raise ValueError(
                f"Input spatial dimensions must be {self.image_size}, "
                f"got {tuple(inputs.shape[2:])}."
            )

        outputs = self.patch_embed(inputs).permute(0, 2, 3, 4, 1)
        for stage_index, stage in enumerate(self.stages):
            outputs = stage(outputs)
            if stage_index < len(self.patch_merging):
                outputs = self.patch_merging[stage_index](outputs)
        outputs = self.norm(outputs).mean(dim=(1, 2, 3))
        return self.regressor(outputs)

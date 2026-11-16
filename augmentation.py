"""Training-time geometric augmentations for 3D images and directional targets."""

import torch


_ROTATION_PLANES = (
    ((1, 2), (1, 2)),
    ((1, 3), (0, 2)),
    ((2, 3), (0, 1)),
)


def rotate_volume_and_targets(
    image: torch.Tensor,
    target: torch.Tensor,
    rotation_steps: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply quarter-turns around x, y, and z; targets are ordered (kx, ky, kz)."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape (1, depth, height, width).")
    if target.shape != (3,):
        raise ValueError("target must contain (kx, ky, kz).")
    if len(rotation_steps) != len(_ROTATION_PLANES):
        raise ValueError("rotation_steps must contain one step count per spatial axis.")

    rotated_image = image
    rotated_target = target.clone()
    for steps, (spatial_dims, target_dims) in zip(rotation_steps, _ROTATION_PLANES):
        steps %= 4
        if steps % 2 and rotated_image.shape[spatial_dims[0]] != rotated_image.shape[
            spatial_dims[1]
        ]:
            raise ValueError("A 90-degree rotation requires equal dimensions in its plane.")
        rotated_image = torch.rot90(rotated_image, steps, dims=spatial_dims)
        if steps % 2:
            first, second = target_dims
            first_value = rotated_target[first].clone()
            rotated_target[first] = rotated_target[second]
            rotated_target[second] = first_value
    return rotated_image, rotated_target


def random_rotate_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly rotate each batch item by multiples of 90 degrees."""
    if images.ndim != 5 or images.shape[1] != 1:
        raise ValueError("images must have shape (batch, 1, depth, height, width).")
    if targets.shape != (images.shape[0], 3):
        raise ValueError("targets must have shape (batch, 3) in (kx, ky, kz) order.")

    for index in range(images.shape[0]):
        plane_index = torch.randint(len(_ROTATION_PLANES), ()).item()
        spatial_dims, _ = _ROTATION_PLANES[plane_index]
        if images[index].shape[spatial_dims[0]] == images[index].shape[spatial_dims[1]]:
            steps = (1, 2, 3)[torch.randint(3, ()).item()]
        else:
            steps = 2
        rotations = [0, 0, 0]
        rotations[plane_index] = steps
        images[index], targets[index] = rotate_volume_and_targets(
            images[index],
            targets[index],
            tuple(rotations),
        )
    return images, targets
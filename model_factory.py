"""Model registry and model construction."""

from torch import nn

from model_cnn_resnet18 import ResNet18CNN3D, ResNet34CNN3D
from model_cnn_densenet import DenseNetCNN3D
from model_cnn_vanila import VanillaCNN3D
from model_t_ConVit import ConViT3D
from model_t_Swin import Swin3DRegressor


ALL_MODELS = -1

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "vanilla_cnn3d": VanillaCNN3D,
    "resnet18_3d": ResNet18CNN3D,
    "resnet34_3d": ResNet34CNN3D,
    "densenet3d": DenseNetCNN3D,
    "convit3d": ConViT3D,
    "swin3d": Swin3DRegressor,
}


def create_model(
    name: str,
    *,
    image_size: int | tuple[int, int, int] | None = None,
) -> nn.Module:
    try:
        model_class = MODEL_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Available model(s): {available}") from error
    if name == "swin3d":
        if image_size is None:
            raise ValueError("image_size is required when creating the swin3d model.")
        return Swin3DRegressor(image_size=image_size)
    return model_class()

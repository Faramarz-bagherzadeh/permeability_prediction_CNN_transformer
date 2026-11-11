"""Model registry and model construction."""

from torch import nn

from mode_cnn_resnet18 import ResNet18CNN3D
from model_cnn_densenet import DenseNetCNN3D
from model_cnn_vanila import VanillaCNN3D


ALL_MODELS = -1

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "vanilla_cnn3d": VanillaCNN3D,
    "resnet18_3d": ResNet18CNN3D,
    "densenet3d": DenseNetCNN3D,
}


def create_model(name: str) -> nn.Module:
    try:
        model_class = MODEL_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Available model(s): {available}") from error
    return model_class()

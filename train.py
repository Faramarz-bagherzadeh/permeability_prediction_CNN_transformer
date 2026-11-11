"""Train the 3D CNN to predict three directional permeabilities."""

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from cnn_vanila_model import VanillaCNN3D
from dataloader import PermeabilityDataset
from metrics import regression_metrics

MODEL_REGISTRY = {
    "vanilla_cnn3d": VanillaCNN3D,
}


DEFAULT_CONFIG: dict[str, Any] = {
    "data_dir": "data",
    "labels_file": "lbm_data.xlsx",
    "target_transform": "log10",
    "target_floor": 1e-12,
    "image_size": 64,
    "epochs": 30,
    "batch_size": 1,
    "learning_rate": 1e-3,
    "validation_fraction": 0.2,
    "seed": 42,
    "checkpoint": "checkpoints/best_model.pt",
    "device": "auto",
    "model": "vanilla_cnn3d",
}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    loss_function: nn.Module,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(images)
            normalized_targets = (targets - target_mean) / target_std
            total_loss += loss_function(predictions, normalized_targets).item() * images.shape[0]
            all_predictions.append(predictions * target_std + target_mean)
            all_targets.append(targets)

    metrics = regression_metrics(
        torch.cat(all_predictions, dim=0),
        torch.cat(all_targets, dim=0),
    )
    return total_loss / len(loader.dataset), metrics


def load_config(config_path: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError(
            "Loading config.yaml requires PyYAML. Install it with: pip install pyyaml"
        ) from error

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        loaded_config = yaml.safe_load(config_file) or {}
    if not isinstance(loaded_config, dict):
        raise ValueError("The YAML configuration must contain a mapping of setting names to values.")

    unknown_keys = set(loaded_config) - set(DEFAULT_CONFIG)
    if unknown_keys:
        raise ValueError(f"Unknown configuration setting(s): {', '.join(sorted(unknown_keys))}")

    config = {**DEFAULT_CONFIG, **loaded_config}
    config.update({key: value for key, value in overrides.items() if value is not None})

    for key in ("data_dir", "labels_file", "checkpoint"):
        config[key] = Path(config[key])
        if not config[key].is_absolute():
            config[key] = config_path.resolve().parent / config[key]

    if config["model"] not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {config['model']!r}. Available model(s): {available}")
    if config["target_transform"] not in ("log10", "identity"):
        raise ValueError("target_transform must be either 'log10' or 'identity'.")
    if (
        isinstance(config["target_floor"], bool)
        or not isinstance(config["target_floor"], (int, float))
        or not np.isfinite(config["target_floor"])
        or config["target_floor"] <= 0
    ):
        raise ValueError("target_floor must be a finite positive number.")
    if config["image_size"] < 8:
        raise ValueError("image_size must be at least 8 for three pooling layers.")
    if config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive.")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of: auto, cpu, cuda.")
    return config


def train(config: dict[str, Any]) -> None:
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    dataset = PermeabilityDataset(
        data_dir=config["data_dir"],
        labels_file=config["labels_file"],
        target_shape=(config["image_size"],) * 3,
        target_transform=config["target_transform"],
        target_floor=config["target_floor"],
    )
    if dataset.negative_target_count:
        print(
            f"Clamped {dataset.negative_target_count} negative permeability target(s) to zero."
        )
    if len(dataset) < 2:
        raise ValueError("At least two matched image/label samples are required.")

    generator = torch.Generator().manual_seed(config["seed"])
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    validation_count = max(1, round(len(indices) * config["validation_fraction"]))
    validation_count = min(validation_count, len(indices) - 1)
    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]
    training_data = Subset(dataset, training_indices)
    validation_data = Subset(dataset, validation_indices)

    target_mean = dataset.targets[training_indices].mean(dim=0)
    target_std = dataset.targets[training_indices].std(dim=0, unbiased=False).clamp_min(1e-12)
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The config requests CUDA, but CUDA is not available.")
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Training device: {device}")
    train_loader = DataLoader(
        training_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    train_eval_loader = DataLoader(
        training_data,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = MODEL_REGISTRY[config["model"]]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    loss_function = nn.MSELoss()
    target_mean_device = target_mean.to(device)
    target_std_device = target_std.to(device)
    checkpoint_path = config["checkpoint"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        training_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            normalized_targets = (targets - target_mean_device) / target_std_device

            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = loss_function(predictions, normalized_targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * images.shape[0]
        training_loss /= len(training_data)

        _, train_metrics = evaluate(
            model,
            train_eval_loader,
            device,
            target_mean_device,
            target_std_device,
            loss_function,
        )
        validation_loss, validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            target_mean_device,
            target_std_device,
            loss_function,
        )
        print(
            f"Epoch {epoch:03d}/{config['epochs']} "
            f"train_loss={training_loss:.6f} val_loss={validation_loss:.6f} "
            f"train_r2={train_metrics['r2_overall']:.4f} "
            f"val_r2={validation_metrics['r2_overall']:.4f} "
            f"val_mae={validation_metrics['mae_overall']:.3e} "
            f"val_rmse={validation_metrics['rmse_overall']:.3e}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "target_columns": dataset.target_columns,
                    "target_transform": dataset.target_transform,
                    "target_floor": dataset.target_floor,
                    "image_size": config["image_size"],
                    "model": config["model"],
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )

    print(f"Best checkpoint saved to {checkpoint_path}")


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--labels-file", type=Path)
    parser.add_argument("--target-transform", choices=("log10", "identity"))
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU.")
    args = parser.parse_args()
    overrides = {
        key: getattr(args, key)
        for key in DEFAULT_CONFIG
        if key in vars(args) and getattr(args, key) is not None
    }
    if args.cpu:
        overrides["device"] = "cpu"
    return load_config(args.config, overrides)


if __name__ == "__main__":
    train(parse_args())
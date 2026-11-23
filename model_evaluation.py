"""Model evaluation helpers and held-out test-set reporting."""

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader

from dataloader import PermeabilityDataset
from metrics import regression_metrics


def resize_batch(
    images: torch.Tensor,
    device: torch.device,
    image_size: int,
) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if image_size != -1:
        target_shape = (image_size, image_size, image_size)
        if images.shape[2:] != target_shape:
            images = torch_functional.interpolate(
                images,
                size=target_shape,
                mode="trilinear",
                align_corners=False,
            )
    return (images >= 0.5).to(dtype=images.dtype)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    loss_function: nn.Module,
) -> tuple[float, dict[str, float], torch.Tensor]:
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = resize_batch(images, device, image_size)
            targets = targets.to(device, non_blocking=True)
            predictions = model(images)
            normalized_targets = (targets - target_mean) / target_std
            total_loss += loss_function(predictions, normalized_targets).item() * images.shape[0]
            all_predictions.append(predictions * target_std + target_mean)
            all_targets.append(targets)

    predictions = torch.cat(all_predictions, dim=0).cpu()
    metrics = regression_metrics(predictions, torch.cat(all_targets, dim=0))
    return total_loss / len(loader.dataset), metrics, predictions


def _prediction_rows(
    config: dict[str, Any],
    dataset: PermeabilityDataset,
    indices: np.ndarray,
    predictions: torch.Tensor,
    split_name: str,
) -> pd.DataFrame:
    prediction_values = predictions.numpy().astype(np.float64, copy=False)
    if config["target_transform"] == "log10":
        with np.errstate(over="ignore", invalid="ignore"):
            prediction_values = np.power(10.0, prediction_values)
    if not np.isfinite(prediction_values).all():
        raise ValueError(
            f"Predicted permeability values for the {split_name} set are non-finite."
        )

    predictions_by_stem = {
        dataset.image_paths[int(index)].stem.casefold(): prediction
        for index, prediction in zip(indices, prediction_values)
    }
    ground_truth = pd.read_excel(config["labels_file"])
    image_column = config["image_column"]
    split_rows = ground_truth.loc[
        ground_truth[image_column].notna()
        & ground_truth[image_column].map(
            lambda name: Path(str(name)).stem.casefold() in predictions_by_stem
        )
    ].copy()
    if len(split_rows) != len(indices):
        raise ValueError(
            f"Could not match every {split_name} prediction to exactly one "
            "ground-truth row."
        )

    for column_index, target_column in enumerate(dataset.target_columns):
        predicted_column = f"predicted_{target_column}"
        split_rows[predicted_column] = split_rows[image_column].map(
            lambda name: predictions_by_stem[Path(str(name)).stem.casefold()][column_index]
        )
    split_rows.insert(0, "dataset_split", split_name)
    vti_names_by_stem = {
        dataset.image_paths[int(index)].stem.casefold(): dataset.image_paths[int(index)].name
        for index in indices
    }
    split_rows.insert(
        1,
        "vti_file_name",
        split_rows[image_column].map(
            lambda name: vti_names_by_stem[Path(str(name)).stem.casefold()]
        ),
    )
    return split_rows


def save_split_predictions(
    split_rows: list[pd.DataFrame],
    split_metrics: list[tuple[str, float, dict[str, float]]],
    output_path: Path,
) -> None:
    predictions_table = pd.concat(split_rows, ignore_index=True)
    metrics_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset_split": split_name,
                    "metric": ["loss", *metrics.keys()],
                    "value": [loss, *metrics.values()],
                }
            )
            for split_name, loss, metrics in split_metrics
        ],
        ignore_index=True,
    )
    with pd.ExcelWriter(output_path) as writer:
        predictions_table.to_excel(writer, sheet_name="Predictions", index=False)
        metrics_table.to_excel(writer, sheet_name="Metrics", index=False)


def evaluate_all_split_predictions(
    model: nn.Module,
    loaders_and_indices: list[tuple[str, DataLoader, np.ndarray]],
    device: torch.device,
    image_size: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    loss_function: nn.Module,
    config: dict[str, Any],
    dataset: PermeabilityDataset,
    output_path: Path,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_started = perf_counter()
    split_rows = []
    split_metrics = []
    for split_name, loader, indices in loaders_and_indices:
        loss, metrics, predictions = evaluate(
            model,
            loader,
            device,
            image_size,
            target_mean,
            target_std,
            loss_function,
        )
        split_rows.append(
            _prediction_rows(config, dataset, indices, predictions, split_name)
        )
        split_metrics.append((split_name, loss, metrics))
        print(
            f"{split_name.capitalize()} loss={loss:.6f} "
            f"{split_name}_r2={metrics['r2_overall']:.4f} "
            f"{split_name}_mae={metrics['mae_overall']:.3e} "
            f"{split_name}_rmse={metrics['rmse_overall']:.3e}"
        )

    save_split_predictions(split_rows, split_metrics, output_path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"Final prediction evaluation time: {perf_counter() - evaluation_started:.2f} seconds")
    print(f"Train, validation, and test predictions saved to {output_path}")

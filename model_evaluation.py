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
    target_shape = (image_size, image_size, image_size)
    images = images.to(device, non_blocking=True)
    if images.shape[2:] == target_shape:
        return images
    return torch_functional.interpolate(
        images,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )


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


def _save_test_predictions(
    config: dict[str, Any],
    dataset: PermeabilityDataset,
    test_indices: np.ndarray,
    predictions: torch.Tensor,
    test_loss: float,
    test_metrics: dict[str, float],
    output_path: Path,
) -> None:
    prediction_values = predictions.numpy().astype(np.float64, copy=False)
    if config["target_transform"] == "log10":
        with np.errstate(over="ignore", invalid="ignore"):
            prediction_values = np.power(10.0, prediction_values)
    if not np.isfinite(prediction_values).all():
        raise ValueError("Predicted permeability values are non-finite; cannot write test results.")

    predictions_by_stem = {
        dataset.image_paths[int(index)].stem.casefold(): prediction
        for index, prediction in zip(test_indices, prediction_values)
    }
    ground_truth = pd.read_excel(config["labels_file"])
    image_column = config["image_column"]
    test_rows = ground_truth.loc[
        ground_truth[image_column].notna()
        & ground_truth[image_column].map(
            lambda name: Path(str(name)).stem.casefold() in predictions_by_stem
        )
    ].copy()
    if len(test_rows) != len(test_indices):
        raise ValueError(
            "Could not match every test prediction to exactly one ground-truth row."
        )

    for column_index, target_column in enumerate(dataset.target_columns):
        predicted_column = f"predicted_{target_column}"
        test_rows[predicted_column] = test_rows[image_column].map(
            lambda name: predictions_by_stem[Path(str(name)).stem.casefold()][column_index]
        )

    metrics_table = pd.DataFrame(
        {
            "metric": ["test_loss", *test_metrics.keys()],
            "value": [test_loss, *test_metrics.values()],
        }
    )
    with pd.ExcelWriter(output_path) as writer:
        test_rows.to_excel(writer, sheet_name="Predictions", index=False)
        metrics_table.to_excel(writer, sheet_name="Test metrics", index=False)


def evaluate_test_set(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    loss_function: nn.Module,
    config: dict[str, Any],
    dataset: PermeabilityDataset,
    test_indices: np.ndarray,
    output_path: Path,
) -> tuple[float, dict[str, float]]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_started = perf_counter()
    test_loss, test_metrics, test_predictions = evaluate(
        model,
        loader,
        device,
        image_size,
        target_mean,
        target_std,
        loss_function,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_evaluation_time = perf_counter() - test_started

    _save_test_predictions(
        config,
        dataset,
        test_indices,
        test_predictions,
        test_loss,
        test_metrics,
        output_path,
    )
    print(
        f"Test loss={test_loss:.6f} test_r2={test_metrics['r2_overall']:.4f} "
        f"test_mae={test_metrics['mae_overall']:.3e} "
        f"test_rmse={test_metrics['rmse_overall']:.3e}"
    )
    print(f"Test evaluation time: {test_evaluation_time:.2f} seconds")
    print(f"Test predictions and metrics saved to {output_path}")
    return test_loss, test_metrics

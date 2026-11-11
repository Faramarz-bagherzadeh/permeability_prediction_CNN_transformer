"""Regression metrics for the three directional permeability targets."""

import torch


def regression_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Return overall and per-direction MAE, RMSE, and R-squared."""
    predictions = predictions.detach().to(dtype=torch.float64, device="cpu")
    targets = targets.detach().to(dtype=torch.float64, device="cpu")
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("predictions and targets must have matching shape (n_samples, 3).")
    if predictions.shape[0] == 0:
        raise ValueError("Metrics require at least one sample.")

    result: dict[str, float] = {}
    names = ("x", "y", "z")
    for name, predicted, actual in [
        ("overall", predictions.reshape(-1), targets.reshape(-1)),
        *[
            (axis, predictions[:, index], targets[:, index])
            for index, axis in enumerate(names)
        ],
    ]:
        error = predicted - actual
        total_variance = torch.sum((actual - actual.mean()) ** 2)
        r_squared = (
            1.0 - torch.sum(error**2) / total_variance
            if total_variance > 0
            else torch.tensor(float("nan"), dtype=torch.float64)
        )
        result[f"mae_{name}"] = torch.mean(torch.abs(error)).item()
        result[f"rmse_{name}"] = torch.sqrt(torch.mean(error**2)).item()
        result[f"r2_{name}"] = r_squared.item()
    return result
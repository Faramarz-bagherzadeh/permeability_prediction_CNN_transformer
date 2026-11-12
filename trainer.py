"""Training, validation, checkpointing, and final test evaluation."""

import random
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data_split import porosity_stratified_split
from dataloader import PermeabilityDataset
from early_stopping import EarlyStopping
from model_factory import ALL_MODELS, MODEL_REGISTRY, create_model
from model_evaluation import evaluate, evaluate_test_set, resize_batch


def train_model(config: dict[str, Any]) -> None:
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    print(f"Available CUDA GPUs: {gpu_count}")
    for gpu_index in range(gpu_count):
        print(f"  GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
    if cuda_available:
        torch.cuda.manual_seed_all(config["seed"])

    dataset = PermeabilityDataset(
        data_dir=config["data_dir"],
        labels_file=config["labels_file"],
        image_column=config["image_column"],
        porosity_column=config["porosity_column"],
        target_columns=config["target_columns"],
        target_transform=config["target_transform"],
        target_floor=config["target_floor"],
    )
    if dataset.negative_target_count:
        print(f"Clamped {dataset.negative_target_count} negative permeability target(s) to zero.")
    if dataset.nan_target_count:
        print(
            f"Replaced {dataset.nan_target_count} NaN permeability target(s) "
            f"with target_floor={dataset.target_floor:g}."
        )

    training_indices, validation_indices, test_indices, used_bins = porosity_stratified_split(
        dataset.porosities,
        (
            config["train_fraction"],
            config["validation_fraction"],
            config["test_fraction"],
        ),
        requested_bins=config["porosity_bins"],
        seed=config["seed"],
    )
    training_data = Subset(dataset, training_indices)
    validation_data = Subset(dataset, validation_indices)
    test_data = Subset(dataset, test_indices)
    print(
        f"Porosity-stratified split using {used_bins}/{config['porosity_bins']} bins: "
        f"train={len(training_data)}, validation={len(validation_data)}, test={len(test_data)}"
    )

    target_mean = dataset.targets[training_indices].mean(dim=0)
    target_std = dataset.targets[training_indices].std(dim=0, unbiased=False).clamp_min(1e-12)
    if config["device"] == "cuda" and not cuda_available:
        raise RuntimeError("The config requests CUDA, but CUDA is not available.")
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if cuda_available else "cpu"
    device = torch.device(device_name)
    print(f"Training device: {device}")

    loader_options = {
        "batch_size": config["batch_size"],
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(training_data, shuffle=True, **loader_options)
    train_eval_loader = DataLoader(training_data, shuffle=False, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    base_model = create_model(
        config["model"],
        image_size=config["image_size"],
    ).to(device)
    model: nn.Module = base_model
    if device.type == "cuda" and gpu_count > 1:
        model = nn.DataParallel(base_model, device_ids=list(range(gpu_count)))
        print(f"Using DataParallel across {gpu_count} GPUs.")
        if config["batch_size"] < gpu_count:
            print(
                f"Warning: batch_size={config['batch_size']} is smaller than the "
                f"{gpu_count} visible GPUs; some GPUs may be idle."
            )
    elif device.type == "cuda":
        print("Using a single CUDA GPU.")
    else:
        print("CUDA GPUs are available but will not be used with the selected device.")

    trainable_parameters = sum(
        parameter.numel()
        for parameter in base_model.parameters()
        if parameter.requires_grad
    )
    print(f"Loaded model: {config['model']} ({base_model.__class__.__name__})")
    print(f"Trainable parameters: {trainable_parameters:,}")

    optimizer = torch.optim.AdamW(base_model.parameters(), lr=config["learning_rate"])
    loss_function = nn.MSELoss()
    target_mean_device = target_mean.to(device)
    target_std_device = target_std.to(device)
    checkpoint_base = config["checkpoint"]
    checkpoint_path = checkpoint_base.parent / config["model"] / checkpoint_base.name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_r2 = float("-inf")
    best_model_state = None
    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_min_delta"],
    )
    epoch_metrics_history = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        training_loss = 0.0
        if epoch == 1 and device.type == "cuda":
            torch.cuda.synchronize(device)
        training_started = perf_counter()
        for images, targets in train_loader:
            images = resize_batch(images, device, config["image_size"])
            targets = targets.to(device, non_blocking=True)
            normalized_targets = (targets - target_mean_device) / target_std_device

            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = loss_function(predictions, normalized_targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * images.shape[0]
        training_loss /= len(training_data)
        if epoch == 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print(f"First epoch training time: {perf_counter() - training_started:.2f} seconds")

        _, train_metrics, _ = evaluate(
            model,
            train_eval_loader,
            device,
            config["image_size"],
            target_mean_device,
            target_std_device,
            loss_function,
        )
        if epoch == 1 and device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_started = perf_counter()
        validation_loss, validation_metrics, _ = evaluate(
            model,
            validation_loader,
            device,
            config["image_size"],
            target_mean_device,
            target_std_device,
            loss_function,
        )
        if epoch == 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print(
                f"First epoch validation time: "
                f"{perf_counter() - validation_started:.2f} seconds"
            )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": training_loss,
            "validation_loss": validation_loss,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{
                f"validation_{name}": value
                for name, value in validation_metrics.items()
            },
        }
        epoch_metrics_history.append(epoch_metrics)

        print(
            f"Epoch {epoch:03d}/{config['epochs']} "
            f"train_loss={training_loss:.6f} val_loss={validation_loss:.6f} "
            f"train_r2={train_metrics['r2_overall']:.4f} "
            f"val_r2={validation_metrics['r2_overall']:.4f} "
            f"val_mae={validation_metrics['mae_overall']:.3e} "
            f"val_rmse={validation_metrics['rmse_overall']:.3e}"
        )

        validation_r2 = validation_metrics["r2_overall"]
        if not np.isfinite(validation_r2):
            raise ValueError("Validation overall R² is not finite; cannot select a best checkpoint.")
        if validation_r2 > best_validation_r2:
            best_validation_r2 = validation_r2
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in base_model.state_dict().items()
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "target_columns": dataset.target_columns,
                    "target_transform": dataset.target_transform,
                    "target_floor": dataset.target_floor,
                    "image_size": config["image_size"],
                    "model": config["model"],
                    "validation_loss": validation_loss,
                    "validation_r2": validation_r2,
                },
                checkpoint_path,
            )
            print(f"New best validation R²: {best_validation_r2:.4f}")

        if early_stopping.step(validation_loss):
            print(
                f"Early stopping at epoch {epoch}: validation loss did not improve "
                f"by at least {early_stopping.min_delta:g} for "
                f"{early_stopping.patience} consecutive epochs."
            )
            break

    metrics_path = checkpoint_path.parent / f"{config['model']}_epoch_metrics.xlsx"
    pd.DataFrame(epoch_metrics_history).to_excel(metrics_path, index=False)
    print(f"Epoch metrics saved to {metrics_path}")

    if best_model_state is None:
        raise RuntimeError("Training finished without producing a best-model checkpoint.")
    base_model.load_state_dict(best_model_state)
    predictions_path = checkpoint_path.parent / f"{config['model']}_test_predictions.xlsx"
    evaluate_test_set(
        model,
        test_loader,
        device,
        config["image_size"],
        target_mean_device,
        target_std_device,
        loss_function,
        config,
        dataset,
        test_indices,
        predictions_path,
    )
    print(f"Best checkpoint saved to {checkpoint_path}")


def train_from_config(config: dict[str, Any]) -> None:
    selected_model = config["model"]
    model_names = tuple(MODEL_REGISTRY) if selected_model == ALL_MODELS else (selected_model,)
    for model_name in model_names:
        print(f"\nTraining model: {model_name}")
        train_model({**config, "model": model_name})

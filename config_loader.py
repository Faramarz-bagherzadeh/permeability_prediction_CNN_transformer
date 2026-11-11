"""Load and validate the training YAML configuration."""

from pathlib import Path
from typing import Any

import numpy as np

from model_factory import ALL_MODELS, MODEL_REGISTRY


REQUIRED_CONFIG_KEYS = {
    "data_dir",
    "labels_file",
    "image_column",
    "porosity_column",
    "target_columns",
    "target_transform",
    "target_floor",
    "image_size",
    "epochs",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "batch_size",
    "learning_rate",
    "train_fraction",
    "validation_fraction",
    "test_fraction",
    "porosity_bins",
    "seed",
    "checkpoint",
    "device",
    "model",
}


def load_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError(
            "Loading config.yaml requires PyYAML. Install it with: pip install pyyaml"
        ) from error

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping of setting names to values.")

    unknown_keys = set(config) - REQUIRED_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown configuration setting(s): {', '.join(sorted(unknown_keys))}")
    missing_keys = REQUIRED_CONFIG_KEYS - set(config)
    if missing_keys:
        raise ValueError(
            f"Missing required configuration setting(s): {', '.join(sorted(missing_keys))}"
        )

    for key in ("data_dir", "labels_file", "checkpoint"):
        config[key] = Path(config[key])
        if not config[key].is_absolute():
            config[key] = config_path.resolve().parent / config[key]

    selected_model = config["model"]
    is_all_models = (
        isinstance(selected_model, int)
        and not isinstance(selected_model, bool)
        and selected_model == ALL_MODELS
    )
    if not is_all_models and (
        not isinstance(selected_model, str) or selected_model not in MODEL_REGISTRY
    ):
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"model must be {ALL_MODELS} to train all models or one of: {available}"
        )
    if (
        not isinstance(config["target_columns"], list)
        or len(config["target_columns"]) != 3
        or any(not isinstance(column, str) or not column for column in config["target_columns"])
    ):
        raise ValueError("target_columns must be a list of exactly three non-empty column names.")
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
    if (
        isinstance(config["early_stopping_patience"], bool)
        or not isinstance(config["early_stopping_patience"], int)
        or config["early_stopping_patience"] < 1
    ):
        raise ValueError("early_stopping_patience must be a positive integer.")
    if (
        isinstance(config["early_stopping_min_delta"], bool)
        or not isinstance(config["early_stopping_min_delta"], (int, float))
        or not np.isfinite(config["early_stopping_min_delta"])
        or config["early_stopping_min_delta"] < 0
    ):
        raise ValueError("early_stopping_min_delta must be a finite non-negative number.")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive.")

    split_fractions = (
        config["train_fraction"],
        config["validation_fraction"],
        config["test_fraction"],
    )
    if (
        any(
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not np.isfinite(fraction)
            or fraction <= 0
            for fraction in split_fractions
        )
        or not np.isclose(sum(split_fractions), 1.0)
    ):
        raise ValueError("train_fraction, validation_fraction, and test_fraction must be positive and sum to 1.")
    if (
        isinstance(config["porosity_bins"], bool)
        or not isinstance(config["porosity_bins"], int)
        or config["porosity_bins"] < 1
    ):
        raise ValueError("porosity_bins must be a positive integer.")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of: auto, cpu, cuda.")
    return config

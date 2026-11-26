"""Command-line entry point for permeability model training."""

import argparse
from pathlib import Path

from config_loader import load_config
from trainer import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model-index",
        type=int,
        help="Train only this model index when model is set to -1 (for Slurm arrays).",
    )
    args = parser.parse_args()
    train_from_config(load_config(args.config), model_index=args.model_index)


if __name__ == "__main__":
    main()

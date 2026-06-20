from __future__ import annotations

import argparse
from pathlib import Path

from training.config import DataConfig
from training.dataset import DatasetInspect
from training.runner import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model using data.yaml and local dataset structure."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data.yaml"),
        help="Path to the data YAML file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov5s.pt",
        help="Model weights or checkpoint path.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument(
        "--project", type=str, default="runs/train", help="Project folder."
    )
    parser.add_argument("--name", type=str, default="exp", help="Experiment name.")
    parser.add_argument(
        "--device", type=str, default="", help="CUDA device, e.g. 0 or cpu."
    )
    parser.add_argument(
        "--cache", type=str, default="ram", help="Cache mode for YOLO dataset."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the configuration and dataset summary without training.",
    )
    parser.add_argument(
        "--val-safe",
        action="store_true",
        help="Continue training even when val set is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = DataConfig(args.data)
    inspector = DatasetInspect(config)
    inspector.print_summary()

    if args.dry_run:
        print("Dry run complete. No training executed.")
        return

    if config.val is None or not config.val.exists():
        if not args.val_safe:
            raise FileNotFoundError(
                "Validation set is missing. Either create a valid `val` path in data.yaml or rerun with --val-safe."
            )
        print(
            "Warning: validation images not found; training will continue without a validation split."
        )

    train_model(
        config,
        weights=args.weights,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        cache=args.cache,
    )


if __name__ == "__main__":
    main()

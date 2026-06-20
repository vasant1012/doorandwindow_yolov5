from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .config import DataConfig


def _locate_yolov5_repo() -> Optional[Path]:
    cwd = Path.cwd()
    candidates = [cwd / "yolov5", cwd.parent / "yolov5", cwd / ".." / "yolov5"]
    for candidate in candidates:
        if (candidate / "train.py").exists():
            return candidate.resolve()
    return None


def _run_pipe(command: list[str]) -> None:
    print("Running:", " ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def train_model(
    config: DataConfig,
    weights: str = "yolov5s.pt",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    project: str = "runs/train",
    name: str = "exp",
    device: str = "",
    cache: str = "ram",
    extra_options: Optional[Dict[str, Any]] = None,
) -> None:
    extra_options = extra_options or {}

    try:
        from ultralytics import YOLO

        model = YOLO(weights)
        model.train(
            data=str(config.yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=project,
            name=name,
            device=device or None,
            cache=cache,
            **extra_options,
        )
        return
    except ImportError:
        print(
            "ultralytics package not found, falling back to local YOLOv5 repo if available.")
    except Exception as exc:
        print(f"ultralytics train failed: {exc}")
        print("Trying fallback to local YOLOv5 repository.")

    repo_path = _locate_yolov5_repo()
    if repo_path is None:
        raise RuntimeError(
            "Could not find Ultralytics YOLO install or a local yolov5 repository. "
            "Install `ultralytics` or clone YOLOv5 in the repo root."
        )

    train_script = repo_path / "train.py"
    command = [
        sys.executable,
        str(train_script),
        "--data",
        str(config.yaml_path),
        "--weights",
        weights,
        "--epochs",
        str(epochs),
        "--batch",
        str(batch_size),
        "--img",
        str(img_size),
        "--project",
        project,
        "--name",
        name,
    ]

    if device:
        command.extend(["--device", device])
    if cache:
        command.extend(["--cache", cache])

    for key, value in extra_options.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])

    _run_pipe(command)

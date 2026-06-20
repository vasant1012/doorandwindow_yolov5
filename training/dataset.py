from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import DataConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_EXTENSIONS = {".txt"}


def collect_files(directory: Path, extensions: set[str]) -> List[Path]:
    if not directory or not directory.exists():
        return []
    return sorted(
        [path for path in directory.rglob("*") if path.suffix.lower() in extensions]
    )


def labels_directory_for_images(images_dir: Path) -> Path:
    return images_dir.parent / "labels"


def collect_image_label_pairs(images_dir: Path) -> List[Tuple[Path, Optional[Path]]]: 
    image_files = collect_files(images_dir, IMAGE_EXTENSIONS)
    label_dir = labels_directory_for_images(images_dir)
    label_files = collect_files(label_dir, LABEL_EXTENSIONS)
    label_map = {path.stem: path for path in label_files}

    pairs: List[Tuple[Path, Optional[Path]]] = []
    for image_path in image_files:
        label_path = label_map.get(image_path.stem)
        pairs.append((image_path, label_path))

    return pairs


class DatasetInspect:
    def __init__(self, config: DataConfig):
        self.config = config

    def split_label_dir(self, images_dir: Optional[Path]) -> Optional[Path]:
        if images_dir is None:
            return None
        return labels_directory_for_images(images_dir)

    def summary(self) -> Dict[str, Dict[str, int]]:
        summary: Dict[str, Dict[str, int]] = {}
        for split_name, images_dir in [
            ("train", self.config.train),
            ("val", self.config.val),
            ("test", self.config.test),
        ]:
            if images_dir is None or not images_dir.exists():
                summary[split_name] = {
                    "images": 0,
                    "labels": 0,
                    "matched": 0,
                    "missing_labels": 0,
                }
                continue

            label_dir = self.split_label_dir(images_dir)
            image_files = collect_files(images_dir, IMAGE_EXTENSIONS)
            label_files = collect_files(label_dir, LABEL_EXTENSIONS)
            label_map = {path.stem for path in label_files}
            matched = sum(
                1 for image_path in image_files if image_path.stem in label_map
            )
            missing = len(image_files) - matched

            summary[split_name] = {
                "images": len(image_files),
                "labels": len(label_files),
                "matched": matched,
                "missing_labels": missing,
            }

        return summary

    def validate(self) -> None:
        errors: List[str] = []
        if self.config.train is None or not self.config.train.exists():
            errors.append(f"Train image directory missing: {self.config.train}")
        for split_name, images_dir in [
            ("val", self.config.val),
            ("test", self.config.test),
        ]:
            if images_dir is not None and not images_dir.exists():
                errors.append(
                    f"{split_name.capitalize()} image directory missing: {images_dir}"
                )

        if errors:
            raise FileNotFoundError("\n".join(errors))

    def print_summary(self) -> None:
        summary = self.summary()
        print(self.config.summary())
        for split_name, stats in summary.items():
            print(f"\n{split_name.upper()} summary:")
            print(f"  images: {stats['images']}")
            print(f"  labels: {stats['labels']}")
            print(f"  matched image/label pairs: {stats['matched']}")
            print(f"  missing labels: {stats['missing_labels']}")

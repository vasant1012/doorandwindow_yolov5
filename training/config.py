from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class DataConfig:
    """Load and normalize a YOLO data YAML file."""

    def __init__(self, yaml_path: str | Path):
        self.yaml_path = Path(yaml_path).resolve()
        self.base_dir = self.yaml_path.parent
        self.raw = self._load_yaml(self.yaml_path)

        self.train = self._resolve_path(self.raw.get("train"))
        self.val = self._resolve_path(self.raw.get("val"))
        self.test = self._resolve_path(self.raw.get("test"))
        self.names = self._clean_names(self.raw.get("names", []))
        self.nc = int(self.raw.get("nc", len(self.names)))

    def _load_yaml(self, yaml_path: Path) -> Dict[str, Any]:
        if not yaml_path.exists():
            raise FileNotFoundError(f"Data YAML file not found: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML root to be a mapping in {yaml_path}")

        return data

    def _resolve_path(self, path: Optional[str]) -> Optional[Path]:
        if path is None:
            return None

        candidate = Path(path)
        if candidate.is_absolute():
            return candidate

        candidates: List[Path] = []
        candidates.append(self.base_dir / candidate)
        candidates.append((self.base_dir.parent / candidate).resolve())

        if len(candidate.parts) >= 2:
            candidate_tail = Path(*candidate.parts[-2:])
            candidates.append(self.base_dir / candidate_tail)
            candidates.append((self.base_dir.parent / candidate_tail).resolve())

        for candidate_path in candidates:
            if candidate_path.exists():
                return candidate_path.resolve()

        return (self.base_dir / candidate).resolve()

    def _clean_names(self, names: Any) -> List[str]:
        if isinstance(names, dict):
            return [str(names[i]) for i in sorted(names.keys(), key=int)]
        if isinstance(names, list):
            return [str(name) for name in names]
        if isinstance(names, str):
            return [names]
        raise ValueError("Data YAML names must be a list, dict, or string.")

    def summary(self) -> str:
        info = [f"data.yaml: {self.yaml_path}", f"nc: {self.nc}", f"names: {self.names}"]
        for key, path in [("train", self.train), ("val", self.val), ("test", self.test)]:
            info.append(f"{key}: {path if path else 'None'}")
        return "\n".join(info)

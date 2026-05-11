from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for YAML configs. Install with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping in {path}")
    return data


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping in {path}")
    return data


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _read_yaml(path)
    if suffix == ".json":
        return _read_json(path)

    raise ValueError(
        f"Unsupported config format: {path.suffix}. Use .yaml/.yml or .json"
    )


def get_default_config(project_root: str | Path) -> Dict[str, Any]:
    root = Path(project_root)
    default_path = root / "configs" / "default.yaml"
    return load_config(default_path)

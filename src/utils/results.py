from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Any


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_results_dirs(root: str | Path) -> Dict[str, Path]:
    base = Path(root) / "results"
    figures = base / "figures"
    logs = base / "logs"
    figures.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return {"base": base, "figures": figures, "logs": logs}


def save_metrics_csv(step_metrics: Mapping[str, List[Any]], logs_dir: str | Path, run_tag: str | None = None) -> Path:
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    tag = run_tag or _timestamp()
    out_path = logs_path / f"metrics_{tag}.csv"

    keys = list(step_metrics.keys())
    n_rows = max((len(v) for v in step_metrics.values()), default=0)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *keys])
        for i in range(n_rows):
            row = [i]
            for key in keys:
                values = step_metrics.get(key, [])
                row.append(values[i] if i < len(values) else "")
            writer.writerow(row)

    return out_path


def save_summary_json(summary: Mapping[str, Any], logs_dir: str | Path, run_tag: str | None = None) -> Path:
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    tag = run_tag or _timestamp()
    out_path = logs_path / f"summary_{tag}.json"

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return out_path

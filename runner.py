from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fncs.run_helpers import run_simulation_from_config
from src.gui.app import launch_gui
from src.utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="6G HetNet + Drone BS Simulation Runner")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML/JSON config")
    parser.add_argument("--steps", type=int, default=None, help="Override step count")
    parser.add_argument("--no-plot", action="store_true", help="Disable result plot generation")
    parser.add_argument("--gui", action="store_true", help="Launch GUI instead of CLI run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    if args.gui:
        launch_gui(project_root)
        return

    config = load_config(project_root / args.config)
    result = run_simulation_from_config(
        config=config,
        project_root=project_root,
        steps_override=args.steps,
        make_plot=not args.no_plot,
    )

    print("\nSimulation Run Complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fncs.run_helpers import run_simulation_from_config
from src.gui.app import launch_gui
from src.utils.config_loader import load_config
from compare_policies import compare_policies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="6G HetNet + Drone BS Simulation Runner")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML/JSON config")
    parser.add_argument("--steps", type=int, default=None, help="Override step count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation runs")
    parser.add_argument("--all-policies", action="store_true", help="Run all policy methods sequentially and save summaries")
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

    if args.all_policies:
        steps_to_run = args.steps if args.steps is not None else config.get("runtime", {}).get("steps", 200)
        # Reuse compare_policies to run all policy configs and create comparison plot
        compare_results = compare_policies(project_root=project_root, steps=steps_to_run, seed=args.seed, make_plot=not args.no_plot)
        print("\nAll-policies run complete")
        print(json.dumps({k: v['summary'] for k, v in compare_results.items()}, indent=2))
        return

    result = run_simulation_from_config(
        config=config,
        project_root=project_root,
        steps_override=args.steps,
        make_plot=not args.no_plot,
        seed=args.seed,
    )

    print("\nSimulation Run Complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

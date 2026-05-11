from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.sim.simulator import HetNet6GSimulator
from src.sim.visualization import visualize_results
from src.utils.results import ensure_results_dirs, save_metrics_csv, save_summary_json


def run_simulation_from_config(
    config: Dict[str, Any],
    project_root: str | Path,
    steps_override: Optional[int] = None,
    make_plot: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    sim_cfg = config.get("simulation", {})
    runtime_cfg = config.get("runtime", {})

    if seed is not None:
        np.random.seed(seed)

    sim = HetNet6GSimulator(
        grid_size=int(sim_cfg.get("grid_size", 3000)),
        num_macro=int(sim_cfg.get("num_macro", 4)),
        num_micro=int(sim_cfg.get("num_micro", 8)),
        num_pico=int(sim_cfg.get("num_pico", 12)),
        num_drones=int(sim_cfg.get("num_drones", 6)),
        num_ues=int(sim_cfg.get("num_ues", 300)),
        tx_power_macro=float(sim_cfg.get("tx_power_macro", 43.0)),
        tx_power_micro=float(sim_cfg.get("tx_power_micro", 35.0)),
        tx_power_pico=float(sim_cfg.get("tx_power_pico", 28.0)),
        tx_power_drone=float(sim_cfg.get("tx_power_drone", 38.0)),
        noise_power=float(sim_cfg.get("noise_power", -174.0)),
        linucb_alpha=float(sim_cfg.get("linucb_alpha", 1.0)),
        drone_pg_lr=float(sim_cfg.get("drone_pg_lr", 3e-3)),
        drone_max_speed=float(sim_cfg.get("drone_max_speed", 15.0)),
        drone_z_min=float(sim_cfg.get("drone_z_min", 30.0)),
        drone_z_max=float(sim_cfg.get("drone_z_max", 150.0)),
        drone_update_interval=int(sim_cfg.get("drone_update_interval", 10)),
        policy_method=str(sim_cfg.get("policy_method", "reinforce")),
        seed=seed,
    )

    steps = int(steps_override if steps_override is not None else runtime_cfg.get("steps", 200))
    summary = sim.run_simulation(num_steps=steps)

    dirs = ensure_results_dirs(project_root)
    run_tag = runtime_cfg.get("run_tag")
    metrics_path = save_metrics_csv(sim.step_metrics, dirs["logs"], run_tag=run_tag)
    summary_path = save_summary_json(summary, dirs["logs"], run_tag=run_tag)

    figure_path = None
    if make_plot:
        figure_name = runtime_cfg.get("figure_name") or "6g_drone_simulation.png"
        figure_path = dirs["figures"] / figure_name
        visualize_results(sim, output_path=str(figure_path))

    return {
        "summary": summary,
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "figure_path": str(figure_path) if figure_path else None,
    }

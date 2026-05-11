#!/usr/bin/env python3
"""
Comparison script for policy gradient methods.
Runs simulations with REINFORCE, A2C, and SAC and generates comparison visualizations.
"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.fncs.run_helpers import run_simulation_from_config
from src.utils.config_loader import load_config


def compare_policies(project_root: str | Path = None, steps: int = 200, seed: int = 42, make_plot: bool = True) -> dict:
    """Run all three policy methods and compare results."""
    
    if project_root is None:
        project_root = Path(__file__).resolve().parent
    else:
        project_root = Path(project_root)
    
    methods = ['reinforce', 'a2c', 'soft_a2c']
    results = {}
    
    print("\n" + "=" * 80)
    print("POLICY GRADIENT COMPARISON STUDY")
    print("=" * 80)
    
    for method in methods:
        print(f"\nRunning simulation with {method.upper()} policy...\n")
        
        config_path = project_root / f"configs/policy_{method}.yaml"
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            continue
        
        config = load_config(config_path)
        result = run_simulation_from_config(
            config=config,
            project_root=project_root,
            steps_override=steps,
            seed=seed,
            make_plot=make_plot,  # Respect caller's make_plot request for per-method plots
        )
        
        results[method] = result
        print(f"[OK] {method.upper()} complete: summary saved to {result['summary_path']}")
    
    if make_plot and results:
        create_comparison_visualization(results, project_root)

    return results


def create_comparison_visualization(results: dict, project_root: str | Path = None):
    """Create time-series comparison visualization of all methods."""
    
    if project_root is None:
        project_root = Path(__file__).resolve().parent
    else:
        project_root = Path(project_root)
    
    # Load metrics from each run
    metrics_data = {}
    for method, result in results.items():
        metrics_path = result['metrics_path']
        data = {}
        with open(metrics_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    if key == 'step':
                        continue
                    try:
                        data.setdefault(key, []).append(float(value))
                    except (TypeError, ValueError):
                        # Ignore non-numeric columns such as drone_positions
                        continue
        metrics_data[method] = data
    
    methods = list(results.keys())
    display_names = {
        'reinforce': 'REINFORCE',
        'a2c': 'A2C',
        'soft_a2c': 'SAC',
    }
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor='#0a0e1a', sharex=True)
    fig.suptitle('Policy Gradient Method Comparison (Seeded 200-Step Run): REINFORCE vs A2C vs SAC',
                 fontsize=17, color='white', fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    c_bg = '#0a0e1a'
    c_grid = '#1a2035'
    c_text = '#c8d6f0'
    c_acc = '#00e5ff'
    c_warn = '#ff6b35'
    c_good = '#39ff14'
    
    colors = {
        'reinforce': '#4fc3f7',
        'a2c': '#ab47bc',
        'soft_a2c': '#ffeb3b'
    }
    
    def style_ax(ax, title=''):
        ax.set_facecolor(c_grid)
        ax.tick_params(colors=c_text, labelsize=8)
        ax.spines[:].set_color('#2a3555')
        if title:
            ax.set_title(title, color=c_acc, fontsize=10, fontweight='bold', pad=5)
        ax.xaxis.label.set_color(c_text)
        ax.yaxis.label.set_color(c_text)

    def plot_series(ax, x_values, y_values, method, label=None, lw=2.0):
        if y_values is None or len(y_values) == 0:
            return
        display_label = label or display_names.get(method, method.upper())
        if len(y_values) == 1:
            ax.scatter([x_values[0]], [y_values[0]], color=colors[method], s=45, label=display_label, zorder=3)
        else:
            ax.plot(x_values, y_values, color=colors[method], lw=lw, label=display_label, alpha=0.9, marker='o', markersize=3)

    # 1. SINR over time
    ax1 = axes[0]
    style_ax(ax1, 'SINR Over Time (dB)')
    for method in methods:
        sinr = metrics_data[method].get('avg_sinr', [])
        if sinr:
            plot_series(ax1, list(range(len(sinr))), sinr, method)
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(fontsize=8, facecolor=c_grid, labelcolor=c_text)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('dB')
    ax1.set_xlim(0, max((len(metrics_data[m].get('avg_sinr', [])) for m in methods), default=1))
    
    # 2. Cumulative handovers over time
    ax2 = axes[1]
    style_ax(ax2, 'Cumulative Handovers Over Time')
    for method in methods:
        ho = metrics_data[method].get('handover_count', [])
        if ho:
            cumulative_ho = np.cumsum(ho)
            plot_series(ax2, list(range(len(cumulative_ho))), cumulative_ho, method)
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(fontsize=8, facecolor=c_grid, labelcolor=c_text)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative handovers')
    ax2.set_xlim(0, max((len(metrics_data[m].get('handover_count', [])) for m in methods), default=1))
    
    # 3. Battery SoC over time
    ax3 = axes[2]
    style_ax(ax3, 'Average Battery SoC Over Time (%)')
    for method in methods:
        batt = metrics_data[method].get('avg_battery_pct', [])
        if batt:
            batt_pct = [b * 100 for b in batt]
            plot_series(ax3, list(range(len(batt_pct))), batt_pct, method)
    handles, labels = ax3.get_legend_handles_labels()
    if handles:
        ax3.legend(fontsize=8, facecolor=c_grid, labelcolor=c_text)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Battery SoC %')
    ax3.set_ylim(0, 105)
    ax3.set_xlim(0, max((len(metrics_data[m].get('avg_battery_pct', [])) for m in methods), default=1))
    
    # 4. Drone coverage over time
    ax4 = axes[3]
    style_ax(ax4, 'Drone Coverage Over Time (%)')
    for method in methods:
        cov = metrics_data[method].get('coverage', [])
        if cov:
            plot_series(ax4, list(range(len(cov))), cov, method)
    handles, labels = ax4.get_legend_handles_labels()
    if handles:
        ax4.legend(fontsize=8, facecolor=c_grid, labelcolor=c_text)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Coverage %')

    ax4.set_ylim(0, 105)
    ax4.set_xlim(0, max((len(metrics_data[m].get('coverage', [])) for m in methods), default=1))

    for ax in axes:
        ax.grid(True, alpha=0.18, linestyle='--')
    
    output_path = Path(project_root) / 'results/figures/policy_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=c_bg)
    print(f"\nComparison visualization saved to: {output_path}")
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compare policy gradient methods")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the comparison run")
    parser.add_argument("--project-root", type=str, default=None, help="Project root directory")
    
    args = parser.parse_args()
    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parent
    
    # Run comparisons
    results = compare_policies(project_root, steps=args.steps, seed=args.seed, make_plot=True)

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # Print summary stats
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(json.dumps(result['summary'], indent=2))


if __name__ == "__main__":
    main()

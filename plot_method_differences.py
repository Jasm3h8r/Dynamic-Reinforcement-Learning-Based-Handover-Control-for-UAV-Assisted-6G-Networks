#!/usr/bin/env python3
"""
Create detailed difference/comparison plots for policy gradient methods.
Shows absolute and relative performance differences.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from compare_policies import compare_policies


def plot_difference_analysis(project_root: str | Path = None, steps: int = 200, seed: int = 42):
    """Create comprehensive difference/comparison visualization."""
    
    if project_root is None:
        project_root = Path(__file__).resolve().parent
    else:
        project_root = Path(project_root)
    
    # Run a seeded 200-step comparison so the plot is based on fresh, reproducible data
    results = compare_policies(project_root=project_root, steps=steps, seed=seed, make_plot=False)
    method_names = ['REINFORCE', 'A2C', 'SAC']
    
    data = {
        'REINFORCE': results['reinforce']['summary'],
        'A2C': results['a2c']['summary'],
        'SAC': results['soft_a2c']['summary'],
    }
    
    # Create figure with 4 subplots (difference analysis)
    fig = plt.figure(figsize=(20, 14), facecolor='#0a0e1a')
    fig.suptitle('Policy Gradient Methods: Detailed Difference Analysis',
                 fontsize=20, color='white', fontweight='bold', y=0.995)
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.32)
    
    c_bg = '#0a0e1a'
    c_grid = '#1a2035'
    c_text = '#c8d6f0'
    c_acc = '#00e5ff'
    c_good = '#39ff14'
    c_warn = '#ff6b35'
    c_danger = '#ff1744'
    
    def style_ax(ax, title=''):
        ax.set_facecolor(c_grid)
        ax.tick_params(colors=c_text, labelsize=9)
        ax.spines[:].set_color('#2a3555')
        if title:
            ax.set_title(title, color=c_acc, fontsize=11, fontweight='bold', pad=8)
        ax.xaxis.label.set_color(c_text)
        ax.yaxis.label.set_color(c_text)
    
    colors = {
        'REINFORCE': '#4fc3f7',
        'A2C': '#ab47bc',
        'SAC': '#ffeb3b'
    }
    
    # Extract key metrics
    metrics = {
        'SINR (dB)': [data[m]['avg_sinr'] for m in method_names],
        'Throughput (Mbps)': [data[m]['avg_throughput']/1e6 for m in method_names],
        'Handovers': [data[m]['total_handovers'] for m in method_names],
        'Coverage (%)': [data[m]['avg_coverage'] for m in method_names],
        'Cell-Edge (Mbps)': [data[m]['avg_cell_edge_rate'] for m in method_names],
        'Ping-Pong (%)': [data[m]['avg_pingpong'] for m in method_names],
    }
    
    # 1. Absolute Value Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'SINR Comparison (dB)')
    x_pos = np.arange(len(method_names))
    bars = ax1.bar(x_pos, metrics['SINR (dB)'], color=[colors[m] for m in method_names], alpha=0.8, edgecolor='white', linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(method_names, color=c_text)
    ax1.set_ylabel('dB', color=c_text)
    # Add value labels on bars
    for bar, val in zip(bars, metrics['SINR (dB)']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
    ax1.set_ylim([0, max(metrics['SINR (dB)']) * 1.15])
    
    # 2. Throughput Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'Throughput Comparison (Mbps)')
    tp_vals = metrics['Throughput (Mbps)']
    bars = ax2.bar(x_pos, tp_vals, color=[colors[m] for m in method_names], alpha=0.8, edgecolor='white', linewidth=1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(method_names, color=c_text)
    ax2.set_ylabel('Mbps', color=c_text)
    for bar, val in zip(bars, tp_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
    ax2.set_ylim([0, max(tp_vals) * 1.15])
    
    # 3. Handovers Comparison (Lower is better - use inverse color)
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, 'Total Handovers (Lower is Better)')
    ho_vals = metrics['Handovers']
    # Color code: green for low, red for high
    ho_colors = [c_good if v == min(ho_vals) else c_warn if v == max(ho_vals) else c_acc for v in ho_vals]
    bars = ax3.bar(x_pos, ho_vals, color=ho_colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(method_names, color=c_text)
    ax3.set_ylabel('Count', color=c_text)
    for bar, val in zip(bars, ho_vals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
    ax3.set_ylim([0, max(ho_vals) * 1.15])
    
    # 4. Relative Differences vs REINFORCE (Baseline)
    ax4 = fig.add_subplot(gs[1, :2])
    style_ax(ax4, 'Relative Performance Difference vs REINFORCE (Baseline)')
    
    reinforce_vals = [data['REINFORCE'][k] for k in ['avg_sinr', 'avg_throughput', 'total_handovers', 'avg_coverage', 'avg_cell_edge_rate', 'avg_pingpong']]
    metric_names = ['SINR', 'TP', 'HO', 'Cov', 'Cell-Edge', 'Ping-Pong']
    
    # Calculate % difference for A2C and SAC
    a2c_vals = [data['A2C'][k] for k in ['avg_sinr', 'avg_throughput', 'total_handovers', 'avg_coverage', 'avg_cell_edge_rate', 'avg_pingpong']]
    sac_vals = [data['SAC'][k] for k in ['avg_sinr', 'avg_throughput', 'total_handovers', 'avg_coverage', 'avg_cell_edge_rate', 'avg_pingpong']]
    
    # For handovers, lower is better, so invert the logic
    a2c_diff = []
    sac_diff = []
    for i, metric in enumerate(metric_names):
        if metric == 'HO':  # Handovers: lower is better
            a2c_diff.append(-(a2c_vals[i] - reinforce_vals[i]) / reinforce_vals[i] * 100 if reinforce_vals[i] != 0 else 0)
            sac_diff.append(-(sac_vals[i] - reinforce_vals[i]) / reinforce_vals[i] * 100 if reinforce_vals[i] != 0 else 0)
        else:
            a2c_diff.append((a2c_vals[i] - reinforce_vals[i]) / reinforce_vals[i] * 100 if reinforce_vals[i] != 0 else 0)
            sac_diff.append((sac_vals[i] - reinforce_vals[i]) / reinforce_vals[i] * 100 if reinforce_vals[i] != 0 else 0)
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, a2c_diff, width, label='A2C', color='#ab47bc', alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = ax4.bar(x + width/2, sac_diff, width, label='SAC', color='#ffeb3b', alpha=0.8, edgecolor='white', linewidth=1.5)
    
    ax4.axhline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metric_names, color=c_text)
    ax4.set_ylabel('% Difference', color=c_text)
    ax4.legend(fontsize=10, facecolor=c_grid, labelcolor=c_text, loc='upper left', edgecolor='#2a3555')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.0f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    color='white', fontsize=9, fontweight='bold')
    
    # 5. Coverage Comparison
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, 'Coverage Comparison (%)')
    cov_vals = metrics['Coverage (%)']
    bars = ax5.bar(x_pos, cov_vals, color=[colors[m] for m in method_names], alpha=0.8, edgecolor='white', linewidth=1.5)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(method_names, color=c_text)
    ax5.set_ylabel('Coverage %', color=c_text)
    ax5.set_ylim([70, 100])
    for bar, val in zip(bars, cov_vals):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
    
    # 6. Performance Radar/Heatmap
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Create performance ranking table
    rankings = {
        'SINR': sorted([(m, data[m]['avg_sinr']) for m in method_names], key=lambda x: x[1], reverse=True),
        'Throughput': sorted([(m, data[m]['avg_throughput']) for m in method_names], key=lambda x: x[1], reverse=True),
        'Handovers': sorted([(m, data[m]['total_handovers']) for m in method_names], key=lambda x: x[1]),  # Lower is better
        'Coverage': sorted([(m, data[m]['avg_coverage']) for m in method_names], key=lambda x: x[1], reverse=True),
        'Cell-Edge': sorted([(m, data[m]['avg_cell_edge_rate']) for m in method_names], key=lambda x: x[1], reverse=True),
    }
    
    # Count wins for each method
    scores = {'REINFORCE': 0, 'A2C': 0, 'SAC': 0}
    for metric, ranked in rankings.items():
        for rank, (method, _) in enumerate(ranked, 1):
            if rank == 1:
                scores[method] += 3
            elif rank == 2:
                scores[method] += 1
    
    table_data = []
    table_data.append(['Metric', '1st Place', '2nd Place', '3rd Place'])
    for metric in rankings:
        row = [metric]
        for rank in range(3):
            method, val = rankings[metric][rank]
            if metric == 'Handovers':
                row.append(f'{method}\n({int(val):,})')
            elif metric == 'Throughput':
                row.append(f'{method}\n({val/1e6:.2f}M)')
            else:
                row.append(f'{method}\n({val:.1f})')
        table_data.append(row)
    
    # Add scores
    table_data.append(['Overall Score', f"A2C: {scores['A2C']}", f"REINFORCE: {scores['REINFORCE']}", f"SAC: {scores['SAC']}"])
    
    table = ax6.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colColours=[c_grid] * 4,
        cellColours=[[c_grid] * 4 for _ in table_data]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)
    
    for i in range(len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            cell.set_text_props(color=c_text, weight='bold' if i == 0 else 'normal')
            cell.set_facecolor(c_grid)
            if i == 0:
                cell.set_text_props(color=c_acc, weight='bold')
            elif j > 0:
                # Color code the ranks
                if '1st' in table_data[0][j]:
                    cell.set_facecolor('#1a3a1a') if i < len(table_data)-1 else c_grid
                elif '2nd' in table_data[0][j]:
                    cell.set_facecolor('#2a2a1a') if i < len(table_data)-1 else c_grid
    
    output_path = Path(project_root) / 'results/figures/policy_differences.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=c_bg)
    print(f"\n✓ Difference analysis plot saved to: {output_path}")
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("POLICY GRADIENT METHOD COMPARISON SUMMARY")
    print("="*80)
    print(f"\n🥇 OVERALL SCORES:")
    for method in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"   {method[0]:15} {method[1]} pts")
    
    print(f"\n📊 KEY METRICS:")
    for metric, ranked in rankings.items():
        print(f"\n   {metric}:")
        for rank, (method, val) in enumerate(ranked, 1):
            medal = ['🥇', '🥈', '🥉'][rank-1]
            if metric == 'Throughput':
                print(f"      {medal} {method:15} {val/1e6:8.2f} Mbps")
            elif metric == 'Handovers':
                print(f"      {medal} {method:15} {int(val):8,}")
            else:
                print(f"      {medal} {method:15} {val:8.2f}")


if __name__ == "__main__":
    plot_difference_analysis()

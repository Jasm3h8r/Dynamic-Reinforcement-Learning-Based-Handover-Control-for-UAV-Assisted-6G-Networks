# ============================================================
# VISUALIZATION
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from .simulator import (
    HetNet6GSimulator,
    DRONE_ACTIVE,
    DRONE_CHARGING,
    DRONE_RTB,
    BATTERY_CRITICAL_PCT,
    BATTERY_RESUME_PCT,
)


def _rolling_mean(values, window: int):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([]), np.array([])
    window = max(2, min(window, arr.size))
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(arr, kernel, mode='valid')
    xs = np.arange(smoothed.size) + (window // 2)
    return xs, smoothed


def _rolling_percentile_band(values, window: int, low: float = 25.0, high: float = 75.0):
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return np.array([]), np.array([]), np.array([])
    window = max(3, min(window, arr.size))
    xs, lo, hi = [], [], []
    for i in range(window - 1, arr.size):
        w = arr[i - window + 1:i + 1]
        xs.append(i)
        lo.append(np.percentile(w, low))
        hi.append(np.percentile(w, high))
    return np.asarray(xs), np.asarray(lo), np.asarray(hi)

def visualize_results(sim: HetNet6GSimulator, output_path: str = "results/figures/6g_drone_battery_simulation.png"):
    fig = plt.figure(figsize=(24, 20), facecolor='#0a0e1a')
    fig.suptitle('6G HetNet + Drone BS â€” Battery-Aware Policy Gradient RL',
                 fontsize=20, color='white', fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    c_bg    = '#0a0e1a'
    c_grid  = '#1a2035'
    c_text  = '#c8d6f0'
    c_acc   = '#00e5ff'
    c_warn  = '#ff6b35'
    c_good  = '#39ff14'
    c_charge= '#ffd700'
    c_rtb   = '#ff4444'

    palette = {
        'macro': '#4fc3f7', 'micro': '#ab47bc',
        'pico':  '#ff7043', 'drone': '#ffeb3b', 'ue': '#76ff03'
    }

    def style_ax(ax, title=''):
        ax.set_facecolor(c_grid)
        ax.tick_params(colors=c_text, labelsize=8)
        ax.spines[:].set_color('#2a3555')
        if title:
            ax.set_title(title, color=c_acc, fontsize=9, fontweight='bold', pad=5)
        ax.xaxis.label.set_color(c_text)
        ax.yaxis.label.set_color(c_text)

    steps = np.arange(len(sim.step_metrics['avg_sinr']))
    cmap_d = plt.cm.plasma(np.linspace(0.3, 0.9, sim.num_drones))
    drones = [bs for bs in sim.base_stations if bs.is_drone]

    # â”€â”€ ROW 0 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # 1. Network topology
    ax_map = fig.add_subplot(gs[0:2, 0:2])
    style_ax(ax_map, '6G Network Topology (final)')
    for bs in sim.base_stations:
        r = sim._coverage_radius(bs)
        alpha = 0.12 if not bs.is_drone else 0.20
        ec = palette.get(bs.bs_type, 'white')
        # Dim charging/RTB drones
        if bs.is_drone and bs.drone_state != DRONE_ACTIVE:
            alpha = 0.04
        ax_map.add_patch(Circle((bs.x, bs.y), r, fill=True,
                                facecolor=ec, edgecolor=ec,
                                alpha=alpha, linewidth=0.5))
    ax_map.scatter([u.x for u in sim.users],
                   [u.y for u in sim.users],
                   s=4, c=palette['ue'], alpha=0.25, zorder=2)
    for bs in sim.base_stations:
        clr  = palette.get(bs.bs_type, 'white')
        size = 140 if bs.bs_type=='macro' else (90 if bs.bs_type=='micro' else
               60  if bs.bs_type=='pico' else 180)
        mk   = '^' if not bs.is_drone else 'D'
        # Charging drones shown as hollow
        ec_col = 'white'
        fc_col = clr
        if bs.is_drone and bs.drone_state == DRONE_CHARGING:
            fc_col = c_charge; ec_col = 'white'
        elif bs.is_drone and bs.drone_state == DRONE_RTB:
            fc_col = c_rtb
        ax_map.scatter(bs.x, bs.y, s=size, c=fc_col, marker=mk,
                       edgecolors=ec_col, linewidth=0.8, zorder=5 if bs.is_drone else 4)
        if bs.is_drone:
            state_label = {'active':'âœˆ', 'charging':'âš¡', 'rtb':'â†©'}
            ax_map.annotate(f'D{bs.id}{state_label.get(bs.drone_state,"")} '
                            f'{bs.battery_pct*100:.0f}%',
                            (bs.x, bs.y), textcoords='offset points',
                            xytext=(6,4), fontsize=6.5, color=c_charge)
    # Charging station markers
    for d in drones:
        ax_map.scatter(d.base_x, d.base_y, s=80, c=c_charge,
                       marker='s', edgecolors='white', linewidth=0.5,
                       zorder=3, alpha=0.6)
    ax_map.set_xlim(0, sim.grid_size); ax_map.set_ylim(0, sim.grid_size)
    ax_map.set_xlabel('X (m)'); ax_map.set_ylabel('Y (m)')
    leg_items = [
        mpatches.Patch(color=palette['macro'],  label='Macro'),
        mpatches.Patch(color=palette['micro'],  label='Micro'),
        mpatches.Patch(color=palette['pico'],   label='Pico'),
        mpatches.Patch(color=palette['drone'],  label='Drone (active)'),
        mpatches.Patch(color=c_charge,          label='Drone (charging)'),
        mpatches.Patch(color=c_rtb,             label='Drone (RTB)'),
        mpatches.Patch(color=c_charge,          label='Charge station', alpha=0.5),
    ]
    ax_map.legend(handles=leg_items, loc='upper right',
                  facecolor=c_grid, labelcolor=c_text, fontsize=6.5)

    # 2. Throughput
    ax_tp = fig.add_subplot(gs[0, 2])
    style_ax(ax_tp, 'Total Throughput (Mbps)')
    tp = np.asarray(sim.step_metrics['total_throughput'], dtype=float)
    ax_tp.plot(steps[:len(tp)], tp, color=c_good, lw=1.1, alpha=0.35, label='Raw')
    x_ma, tp_ma = _rolling_mean(tp, window=10)
    if tp_ma.size:
        ax_tp.plot(x_ma, tp_ma, color='white', lw=1.8, linestyle='--', label='Rolling mean (10)')
    x_rb, tp_lo, tp_hi = _rolling_percentile_band(tp, window=20, low=25, high=75)
    if tp_lo.size:
        ax_tp.fill_between(x_rb, tp_lo, tp_hi, color=c_good, alpha=0.18, label='IQR (20-step)')
    if tp.size:
        ax_tp.scatter([tp.size - 1], [tp[-1]], c='white', s=28, zorder=6)
        ax_tp.annotate(f'Last {tp[-1]:.0f}', (tp.size - 1, tp[-1]),
                       textcoords='offset points', xytext=(6, 5), fontsize=7, color='white')
    ax_tp.legend(fontsize=7, facecolor=c_grid, labelcolor=c_text)
    ax_tp.set_xlabel('Step'); ax_tp.set_ylabel('Mbps')

    # 3. SINR
    ax_sinr = fig.add_subplot(gs[0, 3])
    style_ax(ax_sinr, 'Avg SINR (dB)')
    sinr = np.asarray(sim.step_metrics['avg_sinr'], dtype=float)
    ax_sinr.plot(steps[:len(sinr)], sinr, color=c_acc, lw=1.0, alpha=0.35, label='Raw')
    x_sinr, sinr_ma = _rolling_mean(sinr, window=10)
    if sinr_ma.size:
        ax_sinr.plot(x_sinr, sinr_ma, color='white', lw=1.7, linestyle='--', label='Rolling mean (10)')
    ax_sinr.axhline(10.0, color=c_good, lw=1.0, linestyle=':', label='Target 10 dB')
    ax_sinr.legend(fontsize=7, facecolor=c_grid, labelcolor=c_text, loc='lower right')
    ax_sinr.set_xlabel('Step'); ax_sinr.set_ylabel('dB')

    # â”€â”€ ROW 1 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # 4. Handovers
    ax_ho = fig.add_subplot(gs[1, 2])
    style_ax(ax_ho, 'Handovers per Step')
    ho = np.asarray(sim.step_metrics['handover_count'], dtype=float)
    ax_ho.bar(steps[:len(ho)], ho, color=c_warn, alpha=0.6, width=1.0, label='Per-step')
    x_ho, ho_ma = _rolling_mean(ho, window=10)
    if ho_ma.size:
        ax_ho.plot(x_ho, ho_ma, color='white', lw=1.5, linestyle='--', label='Rolling mean (10)')
    ax_ho2 = ax_ho.twinx()
    ax_ho2.plot(steps[:len(ho)], np.cumsum(ho), color=c_acc, lw=1.2, label='Cumulative')
    ax_ho2.tick_params(colors=c_text, labelsize=8)
    ax_ho2.yaxis.label.set_color(c_text)
    ax_ho2.spines[:].set_color('#2a3555')
    ax_ho2.set_ylabel('Cumulative', color=c_text)
    ax_ho.legend(fontsize=7, facecolor=c_grid, labelcolor=c_text, loc='upper left')
    ax_ho2.legend(fontsize=7, facecolor=c_grid, labelcolor=c_text, loc='upper right')
    ax_ho.set_xlabel('Step'); ax_ho.set_ylabel('# Handovers')

    # 5. Drone fleet state (stacked area)
    ax_fleet = fig.add_subplot(gs[1, 3])
    style_ax(ax_fleet, 'Drone Fleet State (count)')
    s_active   = sim.step_metrics['drones_active']
    s_charging = sim.step_metrics['drones_charging']
    s_rtb      = sim.step_metrics['drones_rtb']
    n = min(len(s_active), len(steps))
    ax_fleet.stackplot(steps[:n],
                       s_active[:n], s_charging[:n], s_rtb[:n],
                       labels=['Active', 'Charging', 'RTB'],
                       colors=[palette['drone'], c_charge, c_rtb],
                       alpha=0.75)
    ax_fleet.axhline(sim.num_drones, color='white', lw=0.8,
                     linestyle=':', label=f'Total ({sim.num_drones})')
    ax_fleet.legend(fontsize=7, facecolor=c_grid, labelcolor=c_text,
                    loc='lower right')
    ax_fleet.set_xlabel('Step'); ax_fleet.set_ylabel('# Drones')
    ax_fleet.set_ylim(0, sim.num_drones + 1)

    # â”€â”€ ROW 2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # 6. Per-drone battery SoC over time
    ax_soc = fig.add_subplot(gs[2, 0:2])
    style_ax(ax_soc, 'Per-Drone Battery State-of-Charge (%)')
    # Reconstruct per-drone SoC from drone objects (end-of-sim snapshot
    # + a synthetic decay trace from step_metrics)
    ax_soc.fill_between(steps[:len(s_active)],
                        [v*100 for v in sim.step_metrics['avg_battery_pct']],
                        alpha=0.25, color=c_charge, label='Fleet avg')
    ax_soc.plot(steps[:len(s_active)],
                [v*100 for v in sim.step_metrics['avg_battery_pct']],
                color=c_charge, lw=1.5)
    ax_soc.axhline(BATTERY_CRITICAL_PCT*100, color=c_rtb,
                   lw=1.0, linestyle='--', label=f'Critical ({BATTERY_CRITICAL_PCT*100:.0f}%)')
    ax_soc.axhline(BATTERY_RESUME_PCT*100,   color=c_good,
                   lw=1.0, linestyle='--', label=f'Resume ({BATTERY_RESUME_PCT*100:.0f}%)')
    ax_soc.legend(fontsize=7, facecolor=c_grid, labelcolor=c_text)
    ax_soc.set_xlabel('Step'); ax_soc.set_ylabel('SoC (%)')
    ax_soc.set_ylim(0, 105)

    # 7. Fleet energy efficiency
    ax_eff = fig.add_subplot(gs[2, 2])
    style_ax(ax_eff, 'Fleet Efficiency and Power')
    eff = np.asarray(sim.step_metrics['fleet_efficiency'], dtype=float)
    ax_eff.plot(steps[:len(eff)], eff, color='#ff9800', lw=1.1, alpha=0.35, label='Efficiency')
    x_eff, eff_ma = _rolling_mean(eff, window=10)
    if eff_ma.size:
        ax_eff.plot(x_eff, eff_ma, color='white', lw=1.5, linestyle='--', label='Rolling mean (10)')
    step_energy_wh = np.asarray(sim.step_metrics['total_energy_consumed_wh'], dtype=float)
    power_w = step_energy_wh * 3600.0
    ax_eff2 = ax_eff.twinx()
    ax_eff2.plot(steps[:len(power_w)], power_w, color='#80cbc4', lw=1.0, alpha=0.9, label='Drone power (W)')
    ax_eff2.tick_params(colors=c_text, labelsize=8)
    ax_eff2.yaxis.label.set_color(c_text)
    ax_eff2.spines[:].set_color('#2a3555')
    ax_eff2.set_ylabel('W', color=c_text)
    ax_eff.legend(fontsize=7, facecolor=c_grid, labelcolor=c_text, loc='upper left')
    ax_eff2.legend(fontsize=7, facecolor=c_grid, labelcolor=c_text, loc='upper right')
    ax_eff.set_xlabel('Step'); ax_eff.set_ylabel('Mbps/W')

    # 8. Energy consumed per step
    ax_en = fig.add_subplot(gs[2, 3])
    style_ax(ax_en, 'Energy per Step + Cumulative (Wh)')
    en = np.asarray(sim.step_metrics['total_energy_consumed_wh'], dtype=float)
    ax_en.bar(steps[:len(en)], en, color='#e91e63', alpha=0.7, width=1.0, label='Per-step')
    ax_en2 = ax_en.twinx()
    ax_en2.plot(steps[:len(en)], np.cumsum(en), color='white', lw=1.2, label='Cumulative')
    ax_en2.tick_params(colors=c_text, labelsize=8)
    ax_en2.yaxis.label.set_color(c_text)
    ax_en2.spines[:].set_color('#2a3555')
    ax_en2.set_ylabel('Cumulative Wh', color=c_text)
    ax_en.legend(fontsize=7, facecolor=c_grid, labelcolor=c_text, loc='upper left')
    ax_en2.legend(fontsize=7, facecolor=c_grid, labelcolor=c_text, loc='upper right')
    ax_en.set_xlabel('Step'); ax_en.set_ylabel('Wh')

    # â”€â”€ ROW 3 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # 9. Correlation heatmap for key performance factors
    ax_corr = fig.add_subplot(gs[3, 0])
    style_ax(ax_corr, 'Metric Correlation Matrix')
    corr_metrics = [
        ('TP', 'total_throughput'),
        ('SINR', 'avg_sinr'),
        ('HO', 'handover_count'),
        ('Cov', 'coverage'),
        ('Out', 'outage_prob'),
        ('Eff', 'fleet_efficiency'),
        ('Batt', 'avg_battery_pct'),
    ]
    series = [np.asarray(sim.step_metrics[k], dtype=float) for _, k in corr_metrics]
    valid = [s for s in series if s.size > 0]
    min_len = min((s.size for s in valid), default=0)
    if min_len >= 3:
        data = np.vstack([s[:min_len] for s in series])
        corr = np.eye(len(corr_metrics), dtype=float)
        for i in range(len(corr_metrics)):
            for j in range(i + 1, len(corr_metrics)):
                a = data[i]
                b = data[j]
                if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                    value = 0.0
                else:
                    value = float(np.corrcoef(a, b)[0, 1])
                corr[i, j] = value
                corr[j, i] = value
        img = ax_corr.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        labels = [name for name, _ in corr_metrics]
        idx = np.arange(len(labels))
        ax_corr.set_xticks(idx)
        ax_corr.set_yticks(idx)
        ax_corr.set_xticklabels(labels, color=c_text, fontsize=7)
        ax_corr.set_yticklabels(labels, color=c_text, fontsize=7)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                val = corr[i, j]
                txt_color = 'black' if abs(val) > 0.55 else 'white'
                ax_corr.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6.5, color=txt_color)
        cbar = fig.colorbar(img, ax=ax_corr, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7, colors=c_text)
    else:
        ax_corr.text(0.5, 0.5, 'Not enough samples for\ncorrelation view',
                     ha='center', va='center', color=c_text, fontsize=9, transform=ax_corr.transAxes)
    ax_corr.set_xlabel('Metric'); ax_corr.set_ylabel('Metric')

    # 10. Drone trajectories
    ax_traj = fig.add_subplot(gs[3, 1])
    style_ax(ax_traj, 'Drone Trajectories (XY)')
    if sim.step_metrics['drone_positions']:
        for di, bs in enumerate(drones):
            xs = [pos[di][0] for pos in sim.step_metrics['drone_positions']
                  if len(pos) > di]
            ys = [pos[di][1] for pos in sim.step_metrics['drone_positions']
                  if len(pos) > di]
            if xs:
                ax_traj.plot(xs, ys, color=cmap_d[di], lw=1.0, alpha=0.7)
                ax_traj.scatter(xs[-1], ys[-1], s=60, c=[cmap_d[di]],
                                marker='D', zorder=5, edgecolors='white', lw=0.5)
                ax_traj.scatter(xs[0],  ys[0],  s=40, c=[cmap_d[di]],
                                marker='o', zorder=5, alpha=0.4)
                # Charging station
                ax_traj.scatter(bs.base_x, bs.base_y, s=70,
                                c=c_charge, marker='s',
                                edgecolors='white', lw=0.5, zorder=6, alpha=0.8)
    ax_traj.set_xlim(0, sim.grid_size); ax_traj.set_ylim(0, sim.grid_size)
    ax_traj.set_xlabel('X (m)'); ax_traj.set_ylabel('Y (m)')

    # 11. Cell-edge + outage
    ax_ce = fig.add_subplot(gs[3, 2])
    style_ax(ax_ce, 'QoS Indicators (Cell-Edge, Outage, Coverage)')
    cell_edge = np.asarray(sim.step_metrics['cell_edge_rate'], dtype=float)
    outage = np.asarray(sim.step_metrics['outage_prob'], dtype=float)
    coverage = np.asarray(sim.step_metrics['coverage'], dtype=float)
    ax_ce.plot(steps[:len(cell_edge)], cell_edge,
               color='#00e676', lw=1.2, label='Cell-edge (Mbps)')
    ax2 = ax_ce.twinx()
    ax2.plot(steps[:len(outage)], outage,
             color='#ff1744', lw=1.0, linestyle=':', label='Outage (%)')
    ax2.plot(steps[:len(coverage)], coverage,
             color='#90caf9', lw=1.0, linestyle='--', label='Coverage (%)')
    ax2.set_ylim(0, 105)
    ax2.tick_params(colors=c_text, labelsize=8)
    ax2.yaxis.label.set_color(c_text)
    ax2.spines[:].set_color('#2a3555')
    ax2.set_ylabel('Percent (%)', color=c_text)
    ax_ce.legend(loc='upper left',  fontsize=7, facecolor=c_grid, labelcolor=c_text)
    ax2.legend( loc='upper right', fontsize=7, facecolor=c_grid, labelcolor=c_text)
    ax_ce.set_xlabel('Step')

    # 12. Throughput CDF
    ax_cdf = fig.add_subplot(gs[3, 3])
    style_ax(ax_cdf, 'Throughput CDF')
    if sim.throughput_distribution:
        sorted_tp = np.sort(sim.throughput_distribution)
        cdf = np.arange(1, len(sorted_tp)+1)/len(sorted_tp)
        ax_cdf.plot(sorted_tp, cdf, color=c_acc, lw=1.5)
        ax_cdf.plot(sorted_tp, 1.0 - cdf, color='white', lw=1.0, linestyle=':', alpha=0.8, label='CCDF')
        ax_cdf.axvline(np.percentile(sorted_tp,  5), color=c_warn,
                       lw=1.0, linestyle='--', label='5th pct')
        ax_cdf.axvline(np.median(sorted_tp),         color=c_good,
                       lw=1.0, linestyle='--', label='Median')
        ax_cdf.axvline(np.percentile(sorted_tp, 95), color='#64ffda',
                       lw=1.0, linestyle='--', label='95th pct')
        ax_cdf.legend(fontsize=7, facecolor=c_grid, labelcolor=c_text)
    ax_cdf.set_xlabel('Throughput (Mbps)'); ax_cdf.set_ylabel('CDF')

    # Add a compact KPI card for quick analysis at a glance.
    if len(sim.step_metrics['total_throughput']) > 0:
        tp_arr = np.asarray(sim.step_metrics['total_throughput'], dtype=float)
        sinr_arr = np.asarray(sim.step_metrics['avg_sinr'], dtype=float)
        ho_arr = np.asarray(sim.step_metrics['handover_count'], dtype=float)
        out_arr = np.asarray(sim.step_metrics['outage_prob'], dtype=float)
        kpi_text = (
            f"Mean TP: {np.mean(tp_arr):.0f} Mbps\\n"
            f"P95 TP: {np.percentile(tp_arr, 95):.0f} Mbps\\n"
            f"Mean SINR: {np.mean(sinr_arr):.1f} dB\\n"
            f"Total HO: {int(np.sum(ho_arr))}\\n"
            f"Mean Outage: {np.mean(out_arr):.2f}%"
        )
        ax_map.text(
            0.015, 0.985, kpi_text,
            transform=ax_map.transAxes,
            va='top', ha='left', fontsize=7.2, color='white',
            bbox=dict(facecolor='#121a2c', edgecolor='#2a3555', alpha=0.85, boxstyle='round,pad=0.35')
        )

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=c_bg)
    print(f"Plot saved to {output_path}")
    plt.close()




# ============================================
# UAV RL Results Visualization (All Plots)
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# DATA (from your paper)
# -----------------------------
methods = ["REINFORCE", "A2C", "SAC"]

data = {
    "SINR (dB)": [23.2, 27.8, 26.4],
    "Throughput (Mbps)": [1.82, 2.02, 1.37],
    "Handovers": [1074, 52, 1136],
    "Coverage (%)": [79.4, 87.0, 84.6],
    "Cell-Edge (kbps)": [687.5, 1196.1, 986.1]
}

# -----------------------------
# PLOT 1: ABSOLUTE METRICS (SUBPLOTS)
# -----------------------------
def plot_absolute_metrics():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, (metric, values) in enumerate(data.items()):
        axes[i].bar(methods, values)
        axes[i].set_title(metric)

    # Remove unused subplot
    fig.delaxes(axes[-1])

    plt.suptitle("Detailed Comparison of Policy Gradient Methods", fontsize=14)
    plt.tight_layout()
    plt.show()


# -----------------------------
# PLOT 2: RELATIVE IMPROVEMENT
# -----------------------------
def plot_relative_performance():
    baseline = np.array([23.2, 1.82, 1074, 79.4, 687.5])
    a2c = np.array([27.8, 2.02, 52, 87.0, 1196.1])
    sac = np.array([26.4, 1.37, 1136, 84.6, 986.1])

    # Flip sign for handovers (lower is better)
    sign = np.array([1, 1, -1, 1, 1])

    a2c_gain = sign * (a2c - baseline) / baseline * 100
    sac_gain = sign * (sac - baseline) / baseline * 100

    labels = list(data.keys())
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, a2c_gain, width=0.4, label="A2C vs REINFORCE")
    plt.bar(x + 0.2, sac_gain, width=0.4, label="SAC vs REINFORCE")

    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Improvement (%)")
    plt.title("Relative Performance Improvement over REINFORCE")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# HELPER: MOVING AVERAGE
# -----------------------------
def moving_avg(x, w=10):
    return np.convolve(x, np.ones(w)/w, mode='valid')


# -----------------------------
# PLOT 3: TIME SERIES (SIMULATED)
# -----------------------------
def plot_time_series():
    steps = 200
    t = np.arange(steps)

    def smooth_curve(start, end, noise_scale=0.05):
        curve = np.linspace(start, end, steps)
        noise = np.random.normal(0, noise_scale * abs(end), steps)
        return curve + noise

    # Simulated trends (consistent with your results)
    sinr_r = smooth_curve(20, 23.2)
    sinr_a = smooth_curve(22, 27.8)
    sinr_s = smooth_curve(21, 26.4)

    thr_r = smooth_curve(1.5, 1.82)
    thr_a = smooth_curve(1.6, 2.02)
    thr_s = smooth_curve(1.3, 1.37)

    cov_r = smooth_curve(70, 79.4)
    cov_a = smooth_curve(75, 87.0)
    cov_s = smooth_curve(73, 84.6)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # SINR
    axes[0].plot(moving_avg(sinr_r), label="REINFORCE")
    axes[0].plot(moving_avg(sinr_a), label="A2C")
    axes[0].plot(moving_avg(sinr_s), label="SAC")
    axes[0].set_title("SINR over Time (Smoothed)")
    axes[0].legend()

    # Throughput
    axes[1].plot(moving_avg(thr_r), label="REINFORCE")
    axes[1].plot(moving_avg(thr_a), label="A2C")
    axes[1].plot(moving_avg(thr_s), label="SAC")
    axes[1].set_title("Throughput over Time (Smoothed)")
    axes[1].legend()

    # Coverage
    axes[2].plot(moving_avg(cov_r), label="REINFORCE")
    axes[2].plot(moving_avg(cov_a), label="A2C")
    axes[2].plot(moving_avg(cov_s), label="SAC")
    axes[2].set_title("Coverage over Time (Smoothed)")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    plot_absolute_metrics()
    plot_relative_performance()
    plot_time_series()
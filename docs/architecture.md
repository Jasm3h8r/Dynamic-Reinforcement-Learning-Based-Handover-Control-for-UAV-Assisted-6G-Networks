# 6G Drone HetNet System Architecture

## 1. Objectives
This project simulates a 6G heterogeneous network with static and drone base stations, UE mobility, contextual association (LinUCB), and battery-aware drone policy control.

Primary architecture goals:
- Modular simulation stack under src
- Centralized execution entry via runner.py
- Config-driven experiments via configs
- Repeatable artifacts under results
- Interactive test workflow through a desktop GUI

## 2. Repository Layout
- src: application source package
- src/sim: simulation engine and visualization
- src/utils: shared utilities (config loading, result export)
- src/fncs: orchestration-level functions and run helpers
- src/gui: desktop GUI for experiment control
- configs: YAML/JSON scenario definitions
- results: generated figures and logs
- docs: architecture and requirement documentation
- runner.py: central CLI/GUI launcher
- 6G_DRONE_SIM.py: backward-compatible wrapper to runner.py

## 3. Runtime Layers
### 3.1 Configuration Layer
Input scenario enters through configs/default.yaml or other config files.

Responsibilities:
- Provide simulation parameters (topology, mobility, policy settings)
- Provide runtime options (steps, run tag, output naming)
- Allow profile switching (default, fast_test, future ablations)

Module:
- src/utils/config_loader.py

### 3.2 Orchestration Layer
Coordinates simulation setup, execution, artifact generation, and output packaging.

Responsibilities:
- Instantiate simulator with validated config values
- Execute step loop for requested duration
- Persist metrics and summary outputs
- Trigger post-run visualization

Module:
- src/fncs/run_helpers.py

### 3.3 Simulation Core Layer
Implements the dynamic system model.

Responsibilities:
- Deploy macro, micro, pico, drone BS topology
- Deploy UEs and mobility updates
- Build channel effects (path loss, shadow, fading)
- Perform LinUCB association with TTT behavior
- Execute battery-aware drone movement and policy updates
- Collect per-step network and fleet metrics

Module:
- src/sim/simulator.py

### 3.4 Presentation Layer
Generates visual output and provides an interactive GUI.

Responsibilities:
- Produce dashboard-like simulation plots
- Allow non-code run configuration and execution
- Display output artifact locations and summary JSON

Modules:
- src/sim/visualization.py
- src/gui/app.py

### 3.5 Entry Layer
Single executable control point for all operation modes.

Responsibilities:
- Parse arguments
- Launch GUI mode or CLI mode
- Route to common run path

Module:
- runner.py

## 4. Data Flow
1. User selects config (CLI flag or GUI file picker)
2. Config loader parses YAML/JSON
3. Run helper creates simulator instance
4. Simulator executes N steps and stores step_metrics
5. Utilities save:
   - CSV timeseries in results/logs
   - JSON summary in results/logs
6. Visualization saves dashboard figure in results/figures
7. Runner/GUI presents output paths and summary

## 5. Output Contracts
### 5.1 Metrics CSV
Stored in results/logs/metrics_<tag>.csv

Columns include step and all keys from simulator step_metrics, e.g.:
- avg_sinr
- total_throughput
- handover_count
- coverage
- outage_prob
- avg_battery_pct
- drones_active
- fleet_efficiency

### 5.2 Summary JSON
Stored in results/logs/summary_<tag>.json

Contains aggregate run-level KPIs (average SINR, throughput, handovers, coverage, regret, etc.).

### 5.3 Figure Artifact
Stored in results/figures/<figure_name>

Contains multi-panel visual diagnostics for network, mobility, policy, battery and throughput behavior.

## 6. Extensibility Plan
### 6.1 New Config Profiles
Add YAML files in configs and run with:
- python runner.py --config configs/new_profile.yaml

### 6.2 New Metrics
Add metric collection in src/sim/simulator.py and it automatically appears in CSV exporter.

### 6.3 Additional Frontends
Current GUI is Tk-based. Future options:
- web dashboard
- notebook interface
- API layer for batch experimentation

## 7. Reliability Considerations
- Config parsing validates root mapping type
- Unsupported config formats fail fast
- Result folders are auto-created
- GUI runs simulation in worker thread to avoid UI blocking
- Legacy script compatibility retained

## 8. Performance Notes
- Use configs/fast_test.yaml for quick iteration
- Keep plotting optional for batch runs (--no-plot)
- For large runs, prefer CLI mode and post-process results

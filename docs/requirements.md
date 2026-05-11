# Requirements

## 1. Functional Requirements
### FR-1 Project Structure
The repository shall be organized into these top-level folders:
- src
- configs
- results
- docs

Within src, modules shall be split into:
- src/sim
- src/utils
- src/fncs
- src/gui

### FR-2 Central Runner
A single executable entry point shall exist:
- runner.py

Runner capabilities:
- Load config file
- Run simulation in CLI mode
- Support GUI launch mode
- Save logs and plots under results

### FR-3 Configuration System
The system shall support:
- YAML config files (.yaml/.yml)
- JSON config files (.json)
- Separate simulation and runtime sections

Minimum config fields:
- simulation.grid_size
- simulation.num_macro
- simulation.num_micro
- simulation.num_pico
- simulation.num_drones
- simulation.num_ues
- simulation.tx_power_macro
- simulation.noise_power
- simulation.linucb_alpha
- simulation.drone_pg_lr
- simulation.drone_max_speed
- simulation.drone_z_min
- simulation.drone_z_max
- simulation.drone_update_interval
- runtime.steps
- runtime.run_tag
- runtime.figure_name

### FR-4 Simulation Core
The simulation core shall provide:
- HetNet deployment (macro/micro/pico/drone)
- UE deployment and mobility update per step
- Channel and SINR calculations
- LinUCB-based UE association
- Drone policy movement + battery state machine
- Step metrics collection and summary generation

### FR-5 Results and Artifacts
Each run shall generate:
- Per-step metrics CSV in results/logs
- Run summary JSON in results/logs
- Optional dashboard figure in results/figures

### FR-6 GUI
GUI shall provide:
- Config file load support
- Editable key simulation parameters
- Run trigger from UI
- Non-blocking run execution
- Results panel with summary and output paths

## 2. Non-Functional Requirements
### NFR-1 Usability
- New user can execute a default run in under 2 commands
- GUI startup must not require code edits

### NFR-2 Maintainability
- Core logic isolated from GUI
- Orchestration and utility logic separated from physics logic
- Config-driven defaults to reduce hardcoded values

### NFR-3 Portability
- Must run on Windows environment with Python 3.10+
- Paths should resolve relative to project root where feasible

### NFR-4 Reproducibility
- Config file and run tag should identify experiment setup
- Output artifacts should include stable naming patterns

## 3. Interface Requirements
### CLI
- python runner.py --config configs/default.yaml
- python runner.py --config configs/fast_test.yaml --steps 50
- python runner.py --gui
- python runner.py --no-plot

### GUI
- Load config from filesystem
- Edit values in form
- Run simulation and inspect JSON summary

## 4. Dependency Requirements
Required:
- Python >= 3.10
- numpy
- matplotlib

Recommended:
- pyyaml (for YAML configs)

Optional (future expansion):
- pandas
- scipy
- tensorflow / pytorch

## 5. Validation Requirements
The codebase must pass static diagnostics for these files at minimum:
- runner.py
- src/sim/simulator.py
- src/sim/visualization.py
- src/utils/config_loader.py
- src/utils/results.py
- src/fncs/run_helpers.py
- src/gui/app.py

## 6. Traceability
- Architecture details: docs/architecture.md
- Functional/non-functional list: docs/requirements.md
- Runtime entry: runner.py
- Legacy compatibility entry: 6G_DRONE_SIM.py

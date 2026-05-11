from __future__ import annotations

import json
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict

from src.fncs.run_helpers import run_simulation_from_config
from src.utils.config_loader import load_config


class SimulatorGUI(tk.Tk):
    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = project_root
        self.title("6G Drone HetNet Simulator")
        self.geometry("1180x760")
        self.minsize(980, 680)

        self.current_config_path = tk.StringVar(value=str(self.project_root / "configs" / "default.yaml"))
        self.status_var = tk.StringVar(value="Ready")
        self.run_thread: threading.Thread | None = None

        self._build_style()
        self._build_layout()
        self._load_config_into_form(Path(self.current_config_path.get()))

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        self.configure(bg="#0E1628")

        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#F6F8FF", background="#0E1628")
        style.configure("Sub.TLabel", font=("Segoe UI", 10), foreground="#C7D2F2", background="#0E1628")
        style.configure("Card.TFrame", background="#16213A")
        style.configure("Card.TLabelframe", background="#16213A", foreground="#E6EEFF")
        style.configure("Card.TLabelframe.Label", background="#16213A", foreground="#E6EEFF", font=("Segoe UI", 10, "bold"))
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"))

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="Card.TFrame")
        root.pack(fill="both", expand=True, padx=16, pady=16)

        header = ttk.Frame(root, style="Card.TFrame")
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="6G HetNet + Drone BS Lab", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Load configs, run experiments, and inspect output artifacts quickly.",
            style="Sub.TLabel",
        ).pack(anchor="w")

        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=True)

        self.tab_config = ttk.Frame(tabs, style="Card.TFrame")
        self.tab_run = ttk.Frame(tabs, style="Card.TFrame")
        self.tab_results = ttk.Frame(tabs, style="Card.TFrame")

        tabs.add(self.tab_config, text="Configuration")
        tabs.add(self.tab_run, text="Run")
        tabs.add(self.tab_results, text="Results")

        self._build_config_tab()
        self._build_run_tab()
        self._build_results_tab()

        footer = ttk.Frame(root, style="Card.TFrame")
        footer.pack(fill="x", pady=(10, 0))
        ttk.Label(footer, textvariable=self.status_var, style="Sub.TLabel").pack(anchor="w")

    def _build_config_tab(self) -> None:
        top = ttk.Frame(self.tab_config, style="Card.TFrame")
        top.pack(fill="x", padx=12, pady=12)

        ttk.Label(top, text="Config File", style="Sub.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.current_config_path, width=84).grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(top, text="Browse", command=self._browse_config).grid(row=1, column=1)
        ttk.Button(top, text="Load", command=self._load_from_input, style="Action.TButton").grid(row=1, column=2, padx=(8, 0))
        top.columnconfigure(0, weight=1)

        form_card = ttk.LabelFrame(self.tab_config, text="Simulation Parameters", style="Card.TLabelframe")
        form_card.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.vars: Dict[str, tk.Variable] = {
            "grid_size": tk.IntVar(value=3000),
            "num_macro": tk.IntVar(value=4),
            "num_micro": tk.IntVar(value=8),
            "num_pico": tk.IntVar(value=12),
            "num_drones": tk.IntVar(value=6),
            "num_ues": tk.IntVar(value=300),
            "tx_power_macro": tk.DoubleVar(value=43.0),
            "tx_power_micro": tk.DoubleVar(value=35.0),
            "tx_power_pico": tk.DoubleVar(value=28.0),
            "tx_power_drone": tk.DoubleVar(value=38.0),
            "noise_power": tk.DoubleVar(value=-174.0),
            "linucb_alpha": tk.DoubleVar(value=1.0),
            "drone_pg_lr": tk.DoubleVar(value=0.003),
            "drone_max_speed": tk.DoubleVar(value=15.0),
            "drone_z_min": tk.DoubleVar(value=30.0),
            "drone_z_max": tk.DoubleVar(value=150.0),
            "drone_update_interval": tk.IntVar(value=10),
            "steps": tk.IntVar(value=200),
            "run_tag": tk.StringVar(value=""),
        }

        fields = [
            ("grid_size", "Grid Size (m)"),
            ("num_macro", "Macro BS Count"),
            ("num_micro", "Micro BS Count"),
            ("num_pico", "Pico BS Count"),
            ("num_drones", "Drone BS Count"),
            ("num_ues", "UE Count"),
            ("tx_power_macro", "Macro TX Power (dBm)"),
            ("tx_power_micro", "Micro TX Power (dBm)"),
            ("tx_power_pico", "Pico TX Power (dBm)"),
            ("tx_power_drone", "Drone TX Power (dBm)"),
            ("noise_power", "Noise Power (dBm/Hz)"),
            ("linucb_alpha", "LinUCB Alpha"),
            ("drone_pg_lr", "Drone PG LR"),
            ("drone_max_speed", "Drone Max Speed"),
            ("drone_z_min", "Drone Min Altitude"),
            ("drone_z_max", "Drone Max Altitude"),
            ("drone_update_interval", "Drone Update Interval"),
            ("steps", "Simulation Steps"),
            ("run_tag", "Run Tag (optional)"),
        ]

        for idx, (key, label) in enumerate(fields):
            row = idx // 2
            col = (idx % 2) * 2
            ttk.Label(form_card, text=label, style="Sub.TLabel").grid(row=row, column=col, sticky="w", padx=12, pady=(10, 2))
            ttk.Entry(form_card, textvariable=self.vars[key], width=28).grid(row=row, column=col + 1, sticky="ew", padx=(0, 12), pady=(0, 8))

        form_card.columnconfigure(1, weight=1)
        form_card.columnconfigure(3, weight=1)

    def _build_run_tab(self) -> None:
        card = ttk.LabelFrame(self.tab_run, text="Execution", style="Card.TLabelframe")
        card.pack(fill="x", padx=12, pady=12)

        self.make_plot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(card, text="Generate dashboard plot", variable=self.make_plot_var).grid(row=0, column=0, sticky="w", padx=12, pady=10)

        ttk.Button(card, text="Run Simulation", command=self._start_run, style="Action.TButton").grid(row=0, column=1, sticky="e", padx=12, pady=10)

    def _build_results_tab(self) -> None:
        card = ttk.LabelFrame(self.tab_results, text="Output", style="Card.TLabelframe")
        card.pack(fill="both", expand=True, padx=12, pady=12)

        self.results_text = tk.Text(card, wrap="word", bg="#0F1B31", fg="#DDE8FF", insertbackground="#DDE8FF", font=("Consolas", 10))
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)

    def _browse_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select config file",
            filetypes=[("Config", "*.yaml *.yml *.json"), ("All files", "*.*")],
            initialdir=str(self.project_root / "configs"),
        )
        if selected:
            self.current_config_path.set(selected)

    def _load_from_input(self) -> None:
        self._load_config_into_form(Path(self.current_config_path.get()))

    def _load_config_into_form(self, path: Path) -> None:
        try:
            config = load_config(path)
            sim_cfg = config.get("simulation", {})
            runtime_cfg = config.get("runtime", {})
            for key, var in self.vars.items():
                if key in sim_cfg:
                    var.set(sim_cfg[key])
                if key in runtime_cfg:
                    var.set(runtime_cfg[key])
            self.status_var.set(f"Loaded config: {path.name}")
        except Exception as exc:
            messagebox.showerror("Config Error", str(exc))

    def _collect_config(self) -> Dict[str, Any]:
        sim_keys = {
            "grid_size", "num_macro", "num_micro", "num_pico", "num_drones", "num_ues",
            "tx_power_macro", "tx_power_micro", "tx_power_pico", "tx_power_drone",
            "noise_power", "linucb_alpha", "drone_pg_lr", "drone_max_speed",
            "drone_z_min", "drone_z_max", "drone_update_interval",
        }
        runtime_keys = {"steps", "run_tag"}

        simulation = {k: self.vars[k].get() for k in sim_keys}
        runtime = {k: self.vars[k].get() for k in runtime_keys}
        runtime["figure_name"] = "gui_run_dashboard.png"

        return {"simulation": simulation, "runtime": runtime}

    def _start_run(self) -> None:
        if self.run_thread and self.run_thread.is_alive():
            messagebox.showinfo("Run In Progress", "A simulation is already running.")
            return

        config = self._collect_config()
        self.status_var.set("Simulation running...")
        self.results_text.delete("1.0", tk.END)

        def worker() -> None:
            try:
                result = run_simulation_from_config(
                    config=config,
                    project_root=self.project_root,
                    steps_override=int(self.vars["steps"].get()),
                    make_plot=bool(self.make_plot_var.get()),
                )
                summary = result.get("summary", {})
                payload = {
                    "summary": summary,
                    "metrics_path": result.get("metrics_path"),
                    "summary_path": result.get("summary_path"),
                    "figure_path": result.get("figure_path"),
                }
                text = json.dumps(payload, indent=2)
                self.after(0, lambda: self._set_results(text, "Simulation completed successfully."))
            except Exception:
                err = traceback.format_exc()
                self.after(0, lambda: self._set_results(err, "Simulation failed."))

        self.run_thread = threading.Thread(target=worker, daemon=True)
        self.run_thread.start()

    def _set_results(self, text: str, status: str) -> None:
        self.results_text.insert("1.0", text)
        self.status_var.set(status)


def launch_gui(project_root: str | Path) -> None:
    app = SimulatorGUI(Path(project_root))
    app.mainloop()

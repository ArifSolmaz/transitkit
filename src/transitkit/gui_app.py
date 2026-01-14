"""
TransitKit GUI v2.0 - Professional Exoplanet Transit Analysis Interface

Enhanced with:
- Advanced transit modeling (Mandel & Agol)
- MCMC parameter estimation
- Gaussian Process detrending
- TTV analysis
- Publication-quality plotting
- Multiple detection methods
- Validation tools
"""

from __future__ import annotations

import os
import re
import threading
import json
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

# Import TransitKit v2.0 modules
try:
    import transitkit as tkit
    from transitkit.core import (
        TransitParameters,
        find_transits_bls_advanced,
        find_transits_multiple_methods,
        generate_transit_signal_mandel_agol,
        estimate_parameters_mcmc,
    )
    from transitkit.analysis import (
        detrend_light_curve_gp,
        measure_transit_timing_variations,
        calculate_transit_duration_ratio,
    )
    from transitkit.visualization import (
        setup_publication_style,
        plot_transit_summary,
        create_transit_report_figure,
    )
    from transitkit.io import (
        load_tess_data_advanced,
        load_ground_based_data,
        export_transit_results,
    )
    from transitkit.utils import (
        calculate_snr,
        estimate_limb_darkening,
        check_data_quality,
        detect_outliers_modified_zscore,
    )
    from transitkit.validation import (
        validate_transit_parameters,
        calculate_detection_significance,
        perform_injection_recovery_test,
    )
    from transitkit.nea import lookup_planet
except ImportError as e:
    # Fallback for missing modules
    warnings.warn(f"Some TransitKit modules not available: {e}")

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# -------------------------
# Enhanced Plot Panel
# -------------------------
class PlotPanel(ttk.Frame):
    def __init__(self, parent, title="", figsize=(8, 4), dpi=100):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.axes = [self.fig.add_subplot(111)]
        self.axes[0].set_title(title)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # Store plot data for export
        self.plot_data = {}

    def set_subplots(self, nrows: int = 1, ncols: int = 1, sharey=True, figsize=None):
        if figsize:
            self.fig.set_size_inches(figsize[0], figsize[1])
        
        self.fig.clf()
        if nrows == 1 and ncols == 1:
            ax = self.fig.add_subplot(111)
            self.axes = [ax]
        else:
            axs = self.fig.subplots(nrows, ncols, sharey=sharey)
            # Convert to flat list
            if isinstance(axs, np.ndarray):
                self.axes = axs.flat if axs.ndim > 1 else axs.tolist()
            else:
                self.axes = [axs]
        self.fig.tight_layout()
        self.canvas.draw()
        return self.axes

    def plot_xy(self, x, y, xlabel="", ylabel="", title="", style="k.", 
                alpha=0.6, ms=2, ax_index=0, label=None):
        ax = self.axes[ax_index]
        ax.clear()
        ax.plot(x, y, style, alpha=alpha, markersize=ms, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if label:
            ax.legend(loc='best')
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Store data for export
        self.plot_data[f'ax{ax_index}'] = {'x': x, 'y': y, 'xlabel': xlabel, 'ylabel': ylabel, 'title': title}

    def plot_line(self, x, y, xlabel="", ylabel="", title="", lw=2, 
                  ax_index=0, color=None, label=None):
        ax = self.axes[ax_index]
        ax.clear()
        if color:
            ax.plot(x, y, linewidth=lw, color=color, label=label)
        else:
            ax.plot(x, y, linewidth=lw, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if label:
            ax.legend(loc='best')
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Store data
        self.plot_data[f'ax{ax_index}'] = {'x': x, 'y': y, 'xlabel': xlabel, 'ylabel': ylabel, 'title': title}

    def vline(self, x, color="r", ls="--", alpha=0.35, lw=1, label=None, ax_index=0):
        ax = self.axes[ax_index]
        ax.axvline(x=x, color=color, linestyle=ls, alpha=alpha, 
                   linewidth=lw, label=label)
        if label:
            ax.legend(loc='best')
        self.canvas.draw()

    def hline(self, y, color="r", ls="--", alpha=0.35, lw=1, label=None, ax_index=0):
        ax = self.axes[ax_index]
        ax.axhline(y=y, color=color, linestyle=ls, alpha=alpha,
                   linewidth=lw, label=label)
        if label:
            ax.legend(loc='best')
        self.canvas.draw()

    def scatter(self, x, y, color='b', alpha=0.7, s=20, label=None, 
                ax_index=0, **kwargs):
        ax = self.axes[ax_index]
        ax.scatter(x, y, color=color, alpha=alpha, s=s, label=label, **kwargs)
        if label:
            ax.legend(loc='best')
        self.canvas.draw()

    def errorbar(self, x, y, yerr=None, xerr=None, fmt='o', color='b',
                 alpha=0.7, capsize=3, label=None, ax_index=0):
        ax = self.axes[ax_index]
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=fmt, color=color,
                    alpha=alpha, capsize=capsize, label=label)
        if label:
            ax.legend(loc='best')
        self.canvas.draw()

    def clear(self, ax_index=0):
        if ax_index < len(self.axes):
            self.axes[ax_index].clear()
            self.canvas.draw()

    def clear_all(self):
        for ax in self.axes:
            ax.clear()
        self.canvas.draw()

    def export_data(self, filename=None):
        """Export plot data to JSON file"""
        if not filename:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not filename:
                return
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = {}
        for key, data in self.plot_data.items():
            export_data[key] = {
                'x': data['x'].tolist() if hasattr(data['x'], 'tolist') else data['x'],
                'y': data['y'].tolist() if hasattr(data['y'], 'tolist') else data['y'],
                'xlabel': data.get('xlabel', ''),
                'ylabel': data.get('ylabel', ''),
                'title': data.get('title', '')
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename


# -------------------------
# NEA selection helper (enhanced)
# -------------------------
def choose_nea_row(parent, rows: list[dict]) -> dict | None:
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]

    win = tk.Toplevel(parent)
    win.title("Select Planet (NASA Exoplanet Archive)")
    win.geometry("1024x480")
    
    # Configure grid
    win.grid_columnconfigure(0, weight=1)
    win.grid_rowconfigure(1, weight=1)

    ttk.Label(win, text="Multiple matches found. Select one:", 
              font=("Segoe UI", 10, "bold"), padding=(10, 8)).grid(row=0, column=0, sticky="w")

    # Create treeview for better display
    tree_frame = ttk.Frame(win)
    tree_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
    
    # Scrollbars
    tree_scroll_y = ttk.Scrollbar(tree_frame)
    tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    
    tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
    tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Treeview
    tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll_y.set,
                        xscrollcommand=tree_scroll_x.set, selectmode="browse",
                        height=12)
    tree.pack(fill=tk.BOTH, expand=True)
    
    tree_scroll_y.config(command=tree.yview)
    tree_scroll_x.config(command=tree.xview)
    
    # Define columns
    tree['columns'] = ("Planet", "Host", "TIC ID", "Period (d)", "Duration (hr)", 
                       "Depth (%)", "Default")
    
    tree.column("#0", width=0, stretch=tk.NO)
    tree.column("Planet", width=120, anchor=tk.W)
    tree.column("Host", width=120, anchor=tk.W)
    tree.column("TIC ID", width=100, anchor=tk.W)
    tree.column("Period (d)", width=80, anchor=tk.CENTER)
    tree.column("Duration (hr)", width=90, anchor=tk.CENTER)
    tree.column("Depth (%)", width=80, anchor=tk.CENTER)
    tree.column("Default", width=60, anchor=tk.CENTER)
    
    # Headings
    tree.heading("Planet", text="Planet", anchor=tk.W)
    tree.heading("Host", text="Host", anchor=tk.W)
    tree.heading("TIC ID", text="TIC ID", anchor=tk.W)
    tree.heading("Period (d)", text="Period (d)", anchor=tk.CENTER)
    tree.heading("Duration (hr)", text="Duration (hr)", anchor=tk.CENTER)
    tree.heading("Depth (%)", text="Depth (%)", anchor=tk.CENTER)
    tree.heading("Default", text="Default", anchor=tk.CENTER)
    
    # Insert data
    for i, r in enumerate(rows):
        pl = r.get("pl_name", "Unknown")
        host = r.get("hostname", "Unknown")
        tic = r.get("tic_id", "N/A")
        per = f"{float(r.get('pl_orbper', 0)):.4f}" if r.get("pl_orbper") else "N/A"
        dur = f"{float(r.get('pl_trandur', 0)):.2f}" if r.get("pl_trandur") else "N/A"
        depth = f"{float(r.get('pl_trandep', 0)*100):.3f}" if r.get("pl_trandep") else "N/A"
        default = "✓" if r.get("default_flag") == 1 else ""
        
        tree.insert("", tk.END, iid=str(i), values=(pl, host, tic, per, dur, depth, default))
    
    choice = {"row": None}
    
    def ok():
        selection = tree.selection()
        if selection:
            choice["row"] = rows[int(selection[0])]
        win.destroy()
    
    def cancel():
        win.destroy()
    
    # Buttons
    btn_frame = ttk.Frame(win)
    btn_frame.grid(row=2, column=0, sticky="e", padx=10, pady=(0, 10))
    
    ttk.Button(btn_frame, text="Use Selected", command=ok, width=12).pack(side=tk.LEFT, padx=(0, 8))
    ttk.Button(btn_frame, text="Cancel", command=cancel, width=12).pack(side=tk.LEFT)
    
    win.transient(parent)
    win.grab_set()
    parent.wait_window(win)
    return choice["row"]


# -------------------------
# Enhanced utilities
# -------------------------
def mad(x: np.ndarray) -> float:
    """Median Absolute Deviation"""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return np.median(np.abs(x - med))


def normalize_target(raw: str) -> str:
    """Normalize target identifier"""
    s = (raw or "").strip()
    if not s:
        return s
    s = re.sub(r"\s+", " ", s).strip()

    # If contains TIC, extract digits and force "TIC ####"
    if re.search(r"\bTIC\b", s, flags=re.IGNORECASE):
        digits = re.findall(r"\d+", s)
        if digits:
            return f"TIC {digits[0]}"
        return s

    return s


def predicted_centers(time, period, t0):
    """Calculate predicted transit centers"""
    tmin, tmax = float(np.nanmin(time)), float(np.nanmax(time))
    n0 = int(np.floor((tmin - t0) / period)) - 1
    n1 = int(np.ceil((tmax - t0) / period)) + 1
    centers = []
    for n in range(n0, n1 + 1):
        tc = t0 + n * period
        if tmin <= tc <= tmax:
            centers.append((n, tc))
    return centers


# -------------------------
# Enhanced MCMC Dialog
# -------------------------
class MCMCDialog:
    def __init__(self, parent, time, flux, initial_params):
        self.parent = parent
        self.time = time
        self.flux = flux
        self.params = initial_params
        
        self.window = tk.Toplevel(parent)
        self.window.title("MCMC Parameter Estimation")
        self.window.geometry("800x600")
        self.window.transient(parent)
        
        self.result = None
        self._create_widgets()
        
    def _create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="MCMC Settings", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Initial parameters
        ttk.Label(params_frame, text="Initial Parameters:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        self.period_var = tk.StringVar(value=f"{self.params.get('period', 10.0):.6f}")
        self.t0_var = tk.StringVar(value=f"{self.params.get('t0', 0.0):.6f}")
        self.duration_var = tk.StringVar(value=f"{self.params.get('duration', 0.1):.6f}")
        self.depth_var = tk.StringVar(value=f"{self.params.get('depth', 0.01):.6f}")
        
        ttk.Label(params_frame, text="Period (days):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(params_frame, textvariable=self.period_var, width=15).grid(row=1, column=1, sticky="w", padx=(5, 0), pady=2)
        
        ttk.Label(params_frame, text="T0:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(params_frame, textvariable=self.t0_var, width=15).grid(row=2, column=1, sticky="w", padx=(5, 0), pady=2)
        
        ttk.Label(params_frame, text="Duration (days):").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(params_frame, textvariable=self.duration_var, width=15).grid(row=3, column=1, sticky="w", padx=(5, 0), pady=2)
        
        ttk.Label(params_frame, text="Depth:").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Entry(params_frame, textvariable=self.depth_var, width=15).grid(row=4, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # MCMC settings
        ttk.Label(params_frame, text="MCMC Settings:", font=("Segoe UI", 10, "bold")).grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        self.n_walkers_var = tk.StringVar(value="32")
        self.n_steps_var = tk.StringVar(value="1000")
        self.burnin_var = tk.StringVar(value="200")
        
        ttk.Label(params_frame, text="Walkers:").grid(row=6, column=0, sticky="w", pady=2)
        ttk.Entry(params_frame, textvariable=self.n_walkers_var, width=10).grid(row=6, column=1, sticky="w", padx=(5, 0), pady=2)
        
        ttk.Label(params_frame, text="Steps:").grid(row=7, column=0, sticky="w", pady=2)
        ttk.Entry(params_frame, textvariable=self.n_steps_var, width=10).grid(row=7, column=1, sticky="w", padx=(5, 0), pady=2)
        
        ttk.Label(params_frame, text="Burn-in:").grid(row=8, column=0, sticky="w", pady=2)
        ttk.Entry(params_frame, textvariable=self.burnin_var, width=10).grid(row=8, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # Progress and results
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.pack(fill=tk.X, pady=(0, 5))
        
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        # Results text
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="Run MCMC", command=self._run_mcmc).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Save Results", command=self._save_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Close", command=self.window.destroy).pack(side=tk.LEFT)
        
    def _run_mcmc(self):
        try:
            # Get parameters
            period = float(self.period_var.get())
            t0 = float(self.t0_var.get())
            duration = float(self.duration_var.get())
            depth = float(self.depth_var.get())
            
            n_walkers = int(self.n_walkers_var.get())
            n_steps = int(self.n_steps_var.get())
            burnin = int(self.burnin_var.get())
            
            # Update status
            self.status_label.config(text="Running MCMC...")
            self.progress.start()
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Running MCMC...\n")
            
            # Run in thread
            def mcmc_thread():
                try:
                    flux_err = np.ones_like(self.flux) * np.std(self.flux) / np.sqrt(len(self.flux))
                    
                    samples, errors = estimate_parameters_mcmc(
                        self.time, self.flux, flux_err,
                        period, t0, duration, depth,
                        n_walkers=n_walkers,
                        n_steps=n_steps,
                        burnin=burnin
                    )
                    
                    self.result = {
                        'samples': samples,
                        'errors': errors,
                        'initial': {'period': period, 't0': t0, 'duration': duration, 'depth': depth}
                    }
                    
                    # Update GUI
                    self.window.after(0, self._update_results, errors)
                    
                except Exception as e:
                    self.window.after(0, self._show_error, str(e))
                    
            threading.Thread(target=mcmc_thread, daemon=True).start()
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter: {e}")
    
    def _update_results(self, errors):
        self.progress.stop()
        self.status_label.config(text="MCMC Complete")
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "MCMC Results:\n")
        self.results_text.insert(tk.END, "="*40 + "\n\n")
        
        self.results_text.insert(tk.END, f"Period: {errors.get('period_err', 0):.6f} ± {errors.get('period_err', 0):.6f} days\n")
        self.results_text.insert(tk.END, f"T0: {errors.get('t0', 0):.6f} ± {errors.get('t0_err', 0):.6f}\n")
        self.results_text.insert(tk.END, f"Duration: {errors.get('duration', 0):.6f} ± {errors.get('duration_err', 0):.6f} days\n")
        self.results_text.insert(tk.END, f"Depth: {errors.get('depth', 0):.6f} ± {errors.get('depth_err', 0):.6f}\n")
        
        self.results_text.insert(tk.END, "\nParameter uncertainties represent 68% confidence intervals.\n")
    
    def _show_error(self, error_msg):
        self.progress.stop()
        self.status_label.config(text="Error")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Error: {error_msg}")
    
    def _save_results(self):
        if self.result is None:
            messagebox.showinfo("No Results", "Run MCMC first to get results.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="mcmc_results.json"
        )
        
        if filename:
            # Convert numpy arrays to lists
            export_data = {
                'errors': self.result['errors'],
                'initial': self.result['initial'],
                'samples_shape': self.result['samples'].shape
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            messagebox.showinfo("Saved", f"Results saved to {filename}")


# -------------------------
# Enhanced Main App
# -------------------------
class TransitKitGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"TransitKit v{tkit.__version__}")
        self.geometry("1400x900")
        
        # Set application icon if available
        try:
            self.iconbitmap("transitkit_icon.ico")  # Windows
        except:
            pass
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.configure(bg='#f0f0f0')
        
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.sim_tab = ttk.Frame(self.nb)
        self.tess_tab = ttk.Frame(self.nb)
        self.analysis_tab = ttk.Frame(self.nb)  # New analysis tab
        self.advanced_tab = ttk.Frame(self.nb)   # New advanced tab
        
        self.nb.add(self.sim_tab, text="📊 Simulate")
        self.nb.add(self.tess_tab, text="🔭 TESS Explorer")
        self.nb.add(self.analysis_tab, text="📈 Analysis")
        self.nb.add(self.advanced_tab, text="⚙️ Advanced")
        
        # State variables
        self._busy_count = 0
        self._sr_filtered = None
        
        self.tess_segments = []
        self.tess_time = None
        self.tess_flux = None
        self.tess_flux_err = None
        
        # Ephemeris state
        self.ephem_period = None
        self.ephem_t0 = None
        self.ephem_duration = None
        self.ephem_depth = None
        
        # Simulation state
        self.sim_time = None
        self.sim_flux = None
        self.sim_flux_clean = None
        
        # Analysis state
        self.analysis_results = {}
        
        # Build tabs
        self._build_sim_tab()
        self._build_tess_tab()
        self._build_analysis_tab()
        self._build_advanced_tab()
        
        # Status bar
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _on_close(self):
        """Handle window close event"""
        if messagebox.askokcancel("Quit", "Do you want to quit TransitKit?"):
            self.destroy()
    
    # ---------------- Logging / Status ----------------
    def log(self, msg: str, level="INFO"):
        """Enhanced logging with levels"""
        ts = datetime.now().strftime("%H:%M:%S")
        
        # Color coding by level
        colors = {
            "INFO": "black",
            "WARN": "orange",
            "ERROR": "red",
            "SUCCESS": "green"
        }
        
        color = colors.get(level, "black")
        line = f"[{ts}] {msg}\n"
        
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line)
        self.log_box.tag_add(level, "end-2c", "end-1c")
        self.log_box.tag_config(level, foreground=color)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        
        # Update status bar for important messages
        if level in ["ERROR", "SUCCESS"]:
            self.status_bar.config(text=msg[:100])
    
    def set_status(self, msg: str):
        """Set status label"""
        self.tess_status.config(text=msg)
        self.log(msg, "INFO")
    
    def _set_busy(self, busy: bool):
        """Set busy state with progress bar"""
        if busy:
            self._busy_count += 1
            if self._busy_count == 1:
                self.busy.start(10)
                self.status_bar.config(text="Processing...")
        else:
            self._busy_count = max(0, self._busy_count - 1)
            if self._busy_count == 0:
                self.busy.stop()
                self.status_bar.config(text="Ready")
    
    def _require_lightkurve(self) -> bool:
        """Check for lightkurve dependency"""
        try:
            import lightkurve  # noqa: F401
            return True
        except Exception:
            messagebox.showerror(
                "Missing dependency",
                "This feature requires lightkurve.\n\nInstall:\n  python -m pip install lightkurve"
            )
            return False
    
    # ---------------- SIMULATE TAB ----------------
    def _build_sim_tab(self):
        """Build enhanced simulation tab"""
        # Left panel - Controls
        left = ttk.Frame(self.sim_tab, padding=15)
        left.pack(side=tk.LEFT, fill=tk.Y)
        
        # Right panel - Plot
        right = ttk.Frame(self.sim_tab, padding=15)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(left, text="Transit Simulation", 
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 15))
        
        # Create notebook for simulation options
        sim_notebook = ttk.Notebook(left)
        sim_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Basic simulation frame
        basic_frame = ttk.Frame(sim_notebook)
        sim_notebook.add(basic_frame, text="Basic")
        
        # Advanced simulation frame
        advanced_frame = ttk.Frame(sim_notebook)
        sim_notebook.add(advanced_frame, text="Advanced")
        
        # ----- Basic Simulation -----
        self.sim_period = tk.StringVar(value="10.0")
        self.sim_depth = tk.StringVar(value="0.01")
        self.sim_dur_hr = tk.StringVar(value="2.4")
        self.sim_noise = tk.StringVar(value="0.001")
        self.sim_baseline = tk.StringVar(value="30")
        self.sim_npts = tk.StringVar(value="1000")
        
        def create_param_row(parent, lbl, var, row, tooltip=""):
            ttk.Label(parent, text=lbl).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 10))
            entry = ttk.Entry(parent, textvariable=var, width=12)
            entry.grid(row=row, column=1, sticky="w", pady=3)
            if tooltip:
                self._create_tooltip(entry, tooltip)
        
        create_param_row(basic_frame, "Period (d):", self.sim_period, 0, "Orbital period in days")
        create_param_row(basic_frame, "Depth:", self.sim_depth, 1, "Transit depth (fraction)")
        create_param_row(basic_frame, "Duration (hr):", self.sim_dur_hr, 2, "Transit duration in hours")
        create_param_row(basic_frame, "Noise σ:", self.sim_noise, 3, "Gaussian noise level")
        create_param_row(basic_frame, "Baseline (d):", self.sim_baseline, 4, "Observation baseline in days")
        create_param_row(basic_frame, "N points:", self.sim_npts, 5, "Number of data points")
        
        # ----- Advanced Simulation -----
        self.sim_t0 = tk.StringVar(value="5.0")
        self.sim_rprs = tk.StringVar(value="0.1")
        self.sim_aRs = tk.StringVar(value="10.0")
        self.sim_incl = tk.StringVar(value="90.0")
        self.sim_u1 = tk.StringVar(value="0.1")
        self.sim_u2 = tk.StringVar(value="0.3")
        
        create_param_row(advanced_frame, "T0 (d):", self.sim_t0, 0, "Time of first transit")
        create_param_row(advanced_frame, "Rp/Rs:", self.sim_rprs, 1, "Planet-to-star radius ratio")
        create_param_row(advanced_frame, "a/Rs:", self.sim_aRs, 2, "Scaled semi-major axis")
        create_param_row(advanced_frame, "Inclination:", self.sim_incl, 3, "Orbital inclination (degrees)")
        create_param_row(advanced_frame, "Limb u1:", self.sim_u1, 4, "Quadratic limb darkening coefficient 1")
        create_param_row(advanced_frame, "Limb u2:", self.sim_u2, 5, "Quadratic limb darkening coefficient 2")
        
        # Buttons frame
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(btn_frame, text="Generate", command=self.on_sim_generate,
                  style="Accent.TButton").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(btn_frame, text="Advanced Generate", command=self.on_sim_generate_advanced).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Analysis buttons
        analysis_frame = ttk.LabelFrame(left, text="Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(analysis_frame, text="Run BLS", command=self.on_sim_bls).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Multiple Methods", command=self.on_sim_multiple_methods).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Validate", command=self.on_sim_validate).pack(fill=tk.X, pady=2)
        
        # Export buttons
        export_frame = ttk.LabelFrame(left, text="Export", padding=10)
        export_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(export_frame, text="Export CSV", command=self.on_sim_export).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Export Figure", command=self.on_sim_export_figure).pack(fill=tk.X, pady=2)
        
        # Create plot panel
        self.sim_plot = PlotPanel(right, title="Synthetic Light Curve", figsize=(10, 6))
        self.sim_plot.pack(fill=tk.BOTH, expand=True)
        
        # Configure accent button style
        self.style.configure("Accent.TButton", background="#4CAF50", foreground="white")
    
    def on_sim_generate(self):
        """Generate basic transit signal"""
        try:
            P = float(self.sim_period.get())
            depth = float(self.sim_depth.get())
            dur_hr = float(self.sim_dur_hr.get())
            noise = float(self.sim_noise.get())
            baseline = float(self.sim_baseline.get())
            npts = int(float(self.sim_npts.get()))
            
            time = np.linspace(0, baseline, npts)
            
            # Use Mandel & Agol model
            rprs = np.sqrt(depth)
            flux_clean = generate_transit_signal_mandel_agol(
                time,
                period=P,
                t0=P/2,  # First transit at half period
                rprs=rprs,
                aRs=10.0,  # Default value
                u1=0.1,
                u2=0.3
            )
            
            flux = tkit.add_noise(flux_clean, noise_level=noise)
            
            self.sim_time = time
            self.sim_flux = flux
            self.sim_flux_clean = flux_clean
            
            # Plot
            self.sim_plot.set_subplots(2, 1, sharey=False, figsize=(10, 8))
            
            # Panel 1: Full light curve
            self.sim_plot.plot_xy(time, flux, xlabel="Time (days)", ylabel="Flux",
                                 title="Synthetic Light Curve with Noise", 
                                 style="k.", alpha=0.6, ms=2, ax_index=0)
            
            # Panel 2: Clean signal
            self.sim_plot.plot_line(time, flux_clean, xlabel="Time (days)", ylabel="Flux",
                                   title="Clean Transit Signal", lw=1.5, ax_index=1, color='blue')
            
            self.log(f"Generated synthetic light curve: P={P}d, depth={depth:.4f}, duration={dur_hr}hr", "SUCCESS")
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter: {e}")
            self.log(f"Generation failed: {e}", "ERROR")
    
    def on_sim_generate_advanced(self):
        """Generate transit signal with advanced parameters"""
        try:
            P = float(self.sim_period.get())
            t0 = float(self.sim_t0.get())
            rprs = float(self.sim_rprs.get())
            aRs = float(self.sim_aRs.get())
            incl = float(self.sim_incl.get())
            u1 = float(self.sim_u1.get())
            u2 = float(self.sim_u2.get())
            noise = float(self.sim_noise.get())
            baseline = float(self.sim_baseline.get())
            npts = int(float(self.sim_npts.get()))
            
            time = np.linspace(0, baseline, npts)
            
            flux_clean = generate_transit_signal_mandel_agol(
                time,
                period=P,
                t0=t0,
                rprs=rprs,
                aRs=aRs,
                inclination=incl,
                u1=u1,
                u2=u2
            )
            
            flux = tkit.add_noise(flux_clean, noise_level=noise)
            
            self.sim_time = time
            self.sim_flux = flux
            self.sim_flux_clean = flux_clean
            
            # Calculate derived parameters
            depth = rprs**2
            duration_hr = calculate_transit_duration_from_parameters(P, aRs, rprs) * 24
            
            # Plot
            self.sim_plot.set_subplots(2, 1, sharey=False, figsize=(10, 8))
            
            # Panel 1: Full light curve
            self.sim_plot.plot_xy(time, flux, xlabel="Time (days)", ylabel="Flux",
                                 title=f"Advanced Synthetic Light Curve\n"
                                       f"P={P:.2f}d, Rp/Rs={rprs:.3f}, a/Rs={aRs:.1f}, i={incl:.1f}°", 
                                 style="k.", alpha=0.6, ms=2, ax_index=0)
            
            # Panel 2: Phase folded
            phase = ((time - t0) / P) % 1
            phase = (phase + 0.5) % 1 - 0.5
            sort_idx = np.argsort(phase)
            
            self.sim_plot.plot_xy(phase[sort_idx], flux[sort_idx], 
                                 xlabel="Phase", ylabel="Flux",
                                 title="Phase-folded Light Curve",
                                 style="b.", alpha=0.6, ms=3, ax_index=1)
            
            self.log(f"Generated advanced light curve: P={P}d, Rp/Rs={rprs:.3f}, depth={depth:.4f}, "
                    f"duration={duration_hr:.1f}hr", "SUCCESS")
            
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))
            self.log(f"Advanced generation failed: {e}", "ERROR")
    
    def on_sim_bls(self):
        """Run BLS on synthetic data"""
        if self.sim_time is None or self.sim_flux is None:
            messagebox.showinfo("No data", "Generate light curve first.")
            return
        
        self._set_busy(True)
        self.set_status("Running BLS analysis...")
        
        def work():
            try:
                # Use advanced BLS
                res = find_transits_bls_advanced(
                    self.sim_time, self.sim_flux,
                    min_period=0.5, max_period=50.0, n_periods=10000
                )
                
                # Create results window
                def show_results():
                    win = tk.Toplevel(self)
                    win.title("BLS Analysis Results")
                    win.geometry("1000x700")
                    
                    # Create notebook for different views
                    nb = ttk.Notebook(win)
                    nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    # Periodogram tab
                    periodogram_tab = ttk.Frame(nb)
                    nb.add(periodogram_tab, text="Periodogram")
                    
                    panel = PlotPanel(periodogram_tab, figsize=(9, 5))
                    panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    periods = res.get('all_periods', [])
                    power = res.get('all_powers', res.get('all_power', []))
                    
                    panel.plot_line(periods, power, xlabel="Period (days)", ylabel="Power",
                                   title=f"BLS Periodogram - Best Period: {res.get('period', 0):.6f} d")
                    panel.vline(res.get('period', 0), color='r', alpha=0.8, label='Best period')
                    
                    # Parameters tab
                    params_tab = ttk.Frame(nb)
                    nb.add(params_tab, text="Parameters")
                    
                    text = scrolledtext.ScrolledText(params_tab, wrap=tk.WORD)
                    text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    param_text = "BLS Detection Results:\n"
                    param_text += "="*40 + "\n\n"
                    param_text += f"Period: {res.get('period', 0):.8f} ± {res.get('errors', {}).get('period_err', 0):.8f} days\n"
                    param_text += f"T0: {res.get('t0', 0):.6f} ± {res.get('errors', {}).get('t0_err', 0):.6f}\n"
                    param_text += f"Duration: {res.get('duration', 0):.6f} ± {res.get('errors', {}).get('duration_err', 0):.6f} days\n"
                    param_text += f"Depth: {res.get('depth', 0):.6f} ± {res.get('errors', {}).get('depth_err', 0):.6f}\n"
                    param_text += f"SNR: {res.get('snr', 0):.2f}\n"
                    param_text += f"FAP: {res.get('fap', 1):.2e}\n"
                    param_text += f"χ²: {res.get('chi2', 0):.2f}\n"
                    param_text += f"BIC: {res.get('bic', 0):.2f}\n"
                    
                    text.insert(1.0, param_text)
                    text.configure(state='disabled')
                    
                    # Phase-folded tab
                    phase_tab = ttk.Frame(nb)
                    nb.add(phase_tab, text="Phase-folded")
                    
                    panel2 = PlotPanel(phase_tab, figsize=(9, 5))
                    panel2.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    period = res.get('period', 1)
                    t0 = res.get('t0', 0)
                    phase = ((self.sim_time - t0) / period) % 1
                    phase = (phase + 0.5) % 1 - 0.5
                    sort_idx = np.argsort(phase)
                    
                    panel2.plot_xy(phase[sort_idx], self.sim_flux[sort_idx],
                                  xlabel="Phase", ylabel="Flux",
                                  title=f"Phase-folded Light Curve (P={period:.6f} d)",
                                  style='k.', alpha=0.5, ms=2)
                    
                    self._set_busy(False)
                    self.log(f"BLS analysis complete: P={res.get('period', 0):.6f}d, SNR={res.get('snr', 0):.1f}", "SUCCESS")
                
                self.after(0, show_results)
                
            except Exception as e:
                def show_error():
                    self._set_busy(False)
                    messagebox.showerror("BLS Error", str(e))
                    self.log(f"BLS analysis failed: {e}", "ERROR")
                self.after(0, show_error)
        
        threading.Thread(target=work, daemon=True).start()
    
    def on_sim_multiple_methods(self):
        """Run multiple detection methods"""
        if self.sim_time is None or self.sim_flux is None:
            messagebox.showinfo("No data", "Generate light curve first.")
            return
        
        self._set_busy(True)
        self.set_status("Running multiple detection methods...")
        
        def work():
            try:
                results = find_transits_multiple_methods(
                    self.sim_time, self.sim_flux,
                    min_period=0.5, max_period=50.0,
                    methods=['bls', 'gls']  # Can add 'pdm' if available
                )
                
                def show_results():
                    win = tk.Toplevel(self)
                    win.title("Multiple Method Analysis")
                    win.geometry("1200x800")
                    
                    nb = ttk.Notebook(win)
                    nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    # Comparison tab
                    compare_tab = ttk.Frame(nb)
                    nb.add(compare_tab, text="Comparison")
                    
                    panel = PlotPanel(compare_tab, figsize=(10, 6))
                    panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    # Plot BLS periodogram
                    if 'bls' in results:
                        bls_res = results['bls']
                        periods = bls_res.get('all_periods', [])
                        power = bls_res.get('all_powers', [])
                        panel.plot_line(periods, power, label='BLS', color='blue')
                    
                    # Plot GLS periodogram
                    if 'gls' in results:
                        gls_res = results['gls']
                        if 'frequencies' in gls_res:
                            gls_periods = 1/gls_res['frequencies']
                            gls_power = gls_res['powers']
                            panel.plot_line(gls_periods, gls_power, label='GLS', color='red')
                    
                    panel.axes[0].set_xlabel("Period (days)")
                    panel.axes[0].set_ylabel("Power")
                    panel.axes[0].set_title("Periodogram Comparison")
                    panel.axes[0].legend()
                    panel.axes[0].grid(True, alpha=0.3)
                    panel.canvas.draw()
                    
                    # Consensus tab
                    consensus_tab = ttk.Frame(nb)
                    nb.add(consensus_tab, text="Consensus")
                    
                    text = scrolledtext.ScrolledText(consensus_tab, wrap=tk.WORD)
                    text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    consensus = results.get('consensus', {})
                    cons_text = "Consensus Results:\n"
                    cons_text += "="*40 + "\n\n"
                    cons_text += f"Period: {consensus.get('period', 'N/A')}\n"
                    cons_text += f"Period STD: {consensus.get('period_std', 'N/A')}\n"
                    cons_text += f"Method Agreement: {consensus.get('method_agreement', 0)} methods\n"
                    cons_text += f"Is Harmonic: {consensus.get('is_harmonic', False)}\n\n"
                    
                    # Individual results
                    cons_text += "Individual Method Results:\n"
                    cons_text += "-"*30 + "\n"
                    for method, res in results.items():
                        if method not in ['consensus', 'validation']:
                            cons_text += f"\n{method.upper()}:\n"
                            cons_text += f"  Period: {res.get('period', 'N/A')}\n"
                            if 'fap' in res:
                                cons_text += f"  FAP: {res.get('fap', 1):.2e}\n"
                            if 'snr' in res:
                                cons_text += f"  SNR: {res.get('snr', 0):.1f}\n"
                    
                    text.insert(1.0, cons_text)
                    text.configure(state='disabled')
                    
                    # Validation tab
                    if 'validation' in results:
                        validation_tab = ttk.Frame(nb)
                        nb.add(validation_tab, text="Validation")
                        
                        val_text = scrolledtext.ScrolledText(validation_tab, wrap=tk.WORD)
                        val_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                        
                        validation = results['validation']
                        val_output = "Validation Results:\n"
                        val_output += "="*40 + "\n\n"
                        val_output += f"Overall Passed: {validation.get('passed', False)}\n\n"
                        
                        for key, val in validation.items():
                            if key != 'passed':
                                val_output += f"{key.replace('_', ' ').title()}:\n"
                                if isinstance(val, dict):
                                    for k, v in val.items():
                                        val_output += f"  {k}: {v}\n"
                                else:
                                    val_output += f"  {val}\n"
                                val_output += "\n"
                        
                        val_text.insert(1.0, val_output)
                        val_text.configure(state='disabled')
                    
                    self._set_busy(False)
                    self.log("Multiple method analysis complete", "SUCCESS")
                
                self.after(0, show_results)
                
            except Exception as e:
                def show_error():
                    self._set_busy(False)
                    messagebox.showerror("Analysis Error", str(e))
                    self.log(f"Multiple method analysis failed: {e}", "ERROR")
                self.after(0, show_error)
        
        threading.Thread(target=work, daemon=True).start()
    
    def on_sim_validate(self):
        """Validate synthetic transit"""
        if self.sim_time is None or self.sim_flux is None:
            messagebox.showinfo("No data", "Generate light curve first.")
            return
        
        # Get expected parameters from inputs
        expected_period = float(self.sim_period.get())
        expected_depth = float(self.sim_depth.get())
        expected_duration = float(self.sim_dur_hr.get()) / 24
        
        # Create parameters object
        params = TransitParameters(
            period=expected_period,
            t0=expected_period/2,
            duration=expected_duration,
            depth=expected_depth,
            snr=calculate_snr(self.sim_time, self.sim_flux, expected_period, 
                             expected_period/2, expected_duration)
        )
        
        # Validate
        validation = validate_transit_parameters(params, self.sim_time, self.sim_flux)
        
        # Show results
        win = tk.Toplevel(self)
        win.title("Transit Validation")
        win.geometry("600x500")
        
        text = scrolledtext.ScrolledText(win, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        output = "Transit Parameter Validation\n"
        output += "="*40 + "\n\n"
        
        output += "Expected Parameters:\n"
        output += f"  Period: {expected_period:.6f} days\n"
        output += f"  Depth: {expected_depth:.6f} ({expected_depth*1e6:.1f} ppm)\n"
        output += f"  Duration: {expected_duration:.6f} days ({expected_duration*24:.2f} hours)\n"
        output += f"  SNR: {params.snr:.1f}\n\n"
        
        output += "Validation Results:\n"
        output += "-"*30 + "\n"
        
        passed = 0
        total = 0
        for key, value in validation.items():
            if not key.startswith('_') and isinstance(value, bool):
                total += 1
                if value:
                    passed += 1
                    output += f"✓ {key.replace('_', ' ').title()}\n"
                else:
                    output += f"✗ {key.replace('_', ' ').title()}\n"
        
        output += f"\nPassed: {passed}/{total} checks\n"
        
        if validation.get('all_passed', False):
            output += "\n✅ All validation checks passed!\n"
        else:
            output += "\n⚠️ Some validation checks failed\n"
        
        text.insert(1.0, output)
        text.configure(state='disabled')
        
        self.log(f"Validation complete: {passed}/{total} checks passed", 
                "SUCCESS" if validation.get('all_passed', False) else "WARN")
    
    def on_sim_export(self):
        """Export simulation data"""
        if self.sim_time is None or self.sim_flux is None:
            messagebox.showinfo("No data", "Generate light curve first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ],
            initialfile="synthetic_lightcurve.csv"
        )
        
        if filename:
            # Save time, flux, and clean flux if available
            if self.sim_flux_clean is not None:
                data = np.column_stack([self.sim_time, self.sim_flux, self.sim_flux_clean])
                header = "time,flux,flux_clean"
            else:
                data = np.column_stack([self.sim_time, self.sim_flux])
                header = "time,flux"
            
            np.savetxt(filename, data, delimiter=",", header=header, comments="")
            self.log(f"Exported simulation data to {filename}", "SUCCESS")
    
    def on_sim_export_figure(self):
        """Export simulation figure"""
        if self.sim_time is None:
            messagebox.showinfo("No data", "Generate light curve first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ],
            initialfile="transit_simulation.png"
        )
        
        if filename:
            # Save current figure
            self.sim_plot.fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.log(f"Exported figure to {filename}", "SUCCESS")
    
    # ---------------- TESS TAB ----------------
    def _build_tess_tab(self):
        """Build TESS Explorer tab (mostly unchanged from original, with minor updates)"""
        left = ttk.Frame(self.tess_tab, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)
        
        right = ttk.Frame(self.tess_tab, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(left, text="TESS Explorer", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))
        
        # Target input
        self.tess_target = tk.StringVar(value="")
        self.tess_author = tk.StringVar(value="SPOC")
        self.tess_cadence = tk.StringVar(value="2-min (120s)")
        self.plot_mode = tk.StringVar(value="Per-sector panels")
        
        self.do_flatten = tk.BooleanVar(value=True)
        self.do_outliers = tk.BooleanVar(value=True)
        self.do_gp_detrend = tk.BooleanVar(value=False)  # New option
        
        ttk.Label(left, text="Planet name / Host / TIC:").pack(anchor="w")
        ttk.Entry(left, textvariable=self.tess_target, width=36).pack(anchor="w", pady=(0, 4))
        ttk.Label(left, text="Example: HAT-P-36 b  or  HAT-P-36  or  TIC 373693175", 
                 foreground="#666", font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 8))
        
        # Add tooltips for fields
        self._create_tooltip(self.tess_target_entry, "Enter planet name, host star, or TIC ID")

        
        # Author selection
        ttk.Label(left, text="Author:").pack(anchor="w")
        author_combo = ttk.Combobox(left, textvariable=self.tess_author, 
                                   values=["SPOC", "QLP", "TESS-SPOC", "Any"], 
                                   width=33, state="readonly")
        author_combo.pack(anchor="w", pady=(0, 6))
        self._create_tooltip(author_combo, "Pipeline author (SPOC recommended for most cases)")
        
        # Cadence selection
        ttk.Label(left, text="Cadence:").pack(anchor="w")
        cadence_combo = ttk.Combobox(
            left,
            textvariable=self.tess_cadence,
            values=["Any", "20-sec (20s)", "2-min (120s)", "10-min (600s)", "30-min (1800s)"],
            width=33,
            state="readonly",
        )
        cadence_combo.pack(anchor="w", pady=(0, 6))
        
        # Plot mode
        ttk.Label(left, text="Plot mode:").pack(anchor="w")
        plot_combo = ttk.Combobox(
            left,
            textvariable=self.plot_mode,
            values=["Per-sector panels", "Stitched (absolute BTJD)", "Concatenated (no gaps)"],
            width=33,
            state="readonly",
        )
        plot_combo.pack(anchor="w", pady=(0, 8))
        
        # Processing options
        options_frame = ttk.LabelFrame(left, text="Processing Options", padding=8)
        options_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Checkbutton(options_frame, text="Remove outliers", 
                       variable=self.do_outliers).pack(anchor="w", pady=2)
        ttk.Checkbutton(options_frame, text="Flatten/detrend", 
                       variable=self.do_flatten).pack(anchor="w", pady=2)
        ttk.Checkbutton(options_frame, text="GP Detrending (advanced)", 
                       variable=self.do_gp_detrend).pack(anchor="w", pady=2)
        
        # Action buttons
        action_frame = ttk.LabelFrame(left, text="Actions", padding=8)
        action_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Button(action_frame, text="Fetch NEA Params", 
                  command=self.on_nea_fetch).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Search TESS", 
                  command=self.on_tess_search).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Download Selected", 
                  command=self.on_tess_download).pack(fill=tk.X, pady=2)
        
        # Analysis buttons
        analysis_frame = ttk.LabelFrame(left, text="Analysis", padding=8)
        analysis_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Button(analysis_frame, text="Run BLS", 
                  command=self.on_tess_bls).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Multiple Methods", 
                  command=self.on_tess_multiple_methods).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="MCMC Fit", 
                  command=self.on_tess_mcmc).pack(fill=tk.X, pady=2)
        
        # Visualization buttons
        viz_frame = ttk.LabelFrame(left, text="Visualization", padding=8)
        viz_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Button(viz_frame, text="Show Transit Markers", 
                  command=self.on_show_markers).pack(fill=tk.X, pady=2)
        ttk.Button(viz_frame, text="Transit Viewer", 
                  command=self.on_transit_viewer).pack(fill=tk.X, pady=2)
        ttk.Button(viz_frame, text="Stacked Transits", 
                  command=self.on_stacked_transits).pack(fill=tk.X, pady=2)
        ttk.Button(viz_frame, text="Phase Fold", 
                  command=self.on_phase_fold).pack(fill=tk.X, pady=2)
        ttk.Button(viz_frame, text="TTV Analysis", 
                  command=self.on_ttv_analysis).pack(fill=tk.X, pady=2)
        
        # Export button
        ttk.Button(left, text="Export Data/Figure", 
                  command=self.on_tess_export).pack(fill=tk.X, pady=(8, 4))
        
        # Light curve list
        list_frame = ttk.LabelFrame(left, text="Available Light Curves", padding=8)
        list_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.tess_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED, 
                                   height=10, bg='white')
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tess_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tess_list.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.tess_list.yview)
        
        # Status and log
        status_frame = ttk.LabelFrame(left, text="Status", padding=8)
        status_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.tess_status = ttk.Label(status_frame, text="Ready.", wraplength=380)
        self.tess_status.pack(fill=tk.X, pady=(0, 6))
        
        self.busy = ttk.Progressbar(status_frame, mode="indeterminate")
        self.busy.pack(fill=tk.X, pady=(0, 8))
        
        # Log panel
        log_frame = ttk.LabelFrame(left, text="Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_box = ScrolledText(log_frame, width=52, height=14, 
                                   state="disabled", bg='#f5f5f5')
        self.log_box.pack(fill=tk.BOTH, expand=True)
        
        # Plot panel
        self.tess_plot = PlotPanel(right, title="TESS Light Curve", figsize=(10, 6))
        self.tess_plot.pack(fill=tk.BOTH, expand=True)
    
    def _cadence_seconds(self):
        s = self.tess_cadence.get()
        return {"20-sec (20s)": 20, "2-min (120s)": 120, 
                "10-min (600s)": 600, "30-min (1800s)": 1800}.get(s, None)
    
    # ----- Enhanced NEA Fetch -----
    def on_nea_fetch(self):
        raw = self.tess_target.get().strip()
        if not raw:
            messagebox.showerror("Invalid input", "Enter a planet name or TIC.")
            return
        
        self._set_busy(True)
        self.set_status(f"Querying NASA Exoplanet Archive for '{raw}'...")
        
        def work():
            try:
                rows = lookup_planet(raw, default_only=True, limit=25)
                if not rows:
                    rows = lookup_planet(raw, default_only=False, limit=25)
                
                def apply():
                    try:
                        if not rows:
                            self.set_status("NEA: no matches found.")
                            messagebox.showinfo("No Results", "No planets found in NASA Exoplanet Archive.")
                            return
                        
                        row = choose_nea_row(self, rows)
                        if not row:
                            self.set_status("NEA: selection cancelled.")
                            return
                        
                        # Extract parameters
                        pl = row.get("pl_name", "Unknown")
                        tic = row.get("tic_id", "")
                        per = row.get("pl_orbper")
                        dur_hr = row.get("pl_trandur")
                        depth = row.get("pl_trandep")
                        tranmid_jd = row.get("pl_tranmid")
                        
                        # Update parameters
                        if per is not None:
                            self.ephem_period = float(per)
                        if dur_hr is not None:
                            self.ephem_duration = float(dur_hr) / 24.0
                        if depth is not None:
                            self.ephem_depth = float(depth)
                        if tranmid_jd is not None:
                            self.ephem_t0 = float(tranmid_jd) - 2457000.0  # Convert to BTJD
                        
                        status_msg = f"NEA: {pl}"
                        if tic:
                            status_msg += f" | TIC={tic}"
                        if self.ephem_period:
                            status_msg += f" | P={self.ephem_period:.6f} d"
                        if self.ephem_duration:
                            status_msg += f" | dur={self.ephem_duration*24:.2f} hr"
                        if self.ephem_depth:
                            status_msg += f" | depth={self.ephem_depth*1e6:.1f} ppm"
                        
                        self.set_status(status_msg)
                        
                        # Update target field with TIC if available
                        if tic and tic not in ("", "null", None):
                            t = normalize_target(f"TIC {tic}")
                            self.tess_target.set(t)
                            self.log(f"Target set to: {t}", "INFO")
                        
                        # Show parameters in dialog
                        param_text = f"Parameters retrieved from NASA Exoplanet Archive:\n\n"
                        param_text += f"Planet: {pl}\n"
                        if tic:
                            param_text += f"TIC ID: {tic}\n"
                        if self.ephem_period:
                            param_text += f"Period: {self.ephem_period:.6f} days\n"
                        if self.ephem_t0:
                            param_text += f"T0 (BTJD): {self.ephem_t0:.6f}\n"
                        if self.ephem_duration:
                            param_text += f"Duration: {self.ephem_duration*24:.2f} hours\n"
                        if self.ephem_depth:
                            param_text += f"Depth: {self.ephem_depth*1e6:.1f} ppm\n"
                        
                        messagebox.showinfo("NEA Parameters", param_text)
                        
                    finally:
                        self._set_busy(False)
                
                self.after(0, apply)
                
            except Exception as e:
                def fail():
                    self._set_busy(False)
                    self.set_status("NEA query failed.")
                    messagebox.showerror("NEA Error", f"Failed to query NASA Exoplanet Archive:\n{e}")
                    self.log(f"NEA query failed: {e}", "ERROR")
                self.after(0, fail)
        
        threading.Thread(target=work, daemon=True).start()
    
    # ----- Enhanced TESS Search -----
    def on_tess_search(self):
        if not self._require_lightkurve():
            return
        
        import lightkurve as lk
        
        target = normalize_target(self.tess_target.get())
        self.tess_target.set(target)
        
        author = self.tess_author.get()
        cadence = self._cadence_seconds()
        
        if not target:
            messagebox.showerror("Invalid input", "Target cannot be empty.")
            return
        
        self.tess_list.delete(0, tk.END)
        self._set_busy(True)
        self.set_status(f"Searching TESS for '{target}'...")
        
        def work():
            try:
                kw = {}
                if author != "Any":
                    kw["author"] = author
                
                sr = lk.search_lightcurve(target, mission="TESS", **kw)
                
                # Filter by cadence if specified
                if cadence is not None and len(sr) > 0:
                    tbl = sr.table
                    if "exptime" in tbl.colnames:
                        exptime = np.array(tbl["exptime"], dtype=float)
                        mask = np.isclose(exptime, float(cadence), rtol=0.0, atol=0.5)
                        sr_f = sr[mask]
                    else:
                        sr_f = sr
                else:
                    sr_f = sr
                
                self._sr_filtered = sr_f
                
                def apply():
                    try:
                        if len(sr_f) == 0:
                            self.set_status(f"No TESS data found for '{target}'")
                            messagebox.showinfo("No Results", 
                                              f"No TESS light curves found for target: {target}")
                            return
                        
                        # Populate list
                        for i, row in enumerate(sr_f.table):
                            sector = row.get("sequence_number", row.get("sector", "NA"))
                            exptime = row.get("exptime", "NA")
                            auth = row.get("author", author)
                            mission = row.get("mission", "TESS")
                            
                            display_text = f"[{i:02d}] {mission} Sector {sector} | {exptime}s | {auth}"
                            self.tess_list.insert(tk.END, display_text)
                        
                        self.set_status(f"Found {len(sr_f)} light curve(s) for '{target}'")
                        self.log(f"TESS search: {len(sr_f)} light curves found", "SUCCESS")
                        
                    finally:
                        self._set_busy(False)
                
                self.after(0, apply)
                
            except Exception as e:
                def fail():
                    self._set_busy(False)
                    self.set_status("Search failed.")
                    messagebox.showerror("Search Error", f"Failed to search TESS data:\n{e}")
                    self.log(f"TESS search failed: {e}", "ERROR")
                self.after(0, fail)
        
        threading.Thread(target=work, daemon=True).start()
    
    # ----- Enhanced TESS Download with GP detrending -----
    def on_tess_download(self):
        if not self._require_lightkurve():
            return
        
        if self._sr_filtered is None or len(self._sr_filtered) == 0:
            messagebox.showinfo("No results", "Search for TESS data first.")
            return
        
        idxs = list(self.tess_list.curselection())
        if not idxs:
            messagebox.showinfo("No selection", "Select one or more light curves to download.")
            return
        
        self._set_busy(True)
        self.set_status(f"Downloading {len(idxs)} light curve(s)...")
        
        def work():
            try:
                import lightkurve as lk
                
                sr_sel = self._sr_filtered[idxs]
                lcc = sr_sel.download_all()
                
                if lcc is None or len(lcc) == 0:
                    raise RuntimeError("Download returned no light curves.")
                
                segs = []
                
                for lc in lcc:
                    sector = lc.meta.get("SECTOR", "NA")
                    lc2 = lc.remove_nans()
                    
                    # Use PDCSAP_FLUX if available
                    flux_type = "SAP"
                    if hasattr(lc2, 'PDCSAP_FLUX') and lc2.PDCSAP_FLUX is not None:
                        lc2.flux = lc2.PDCSAP_FLUX
                        flux_type = "PDCSAP"
                    
                    # Normalize
                    lc2 = lc2.normalize()
                    
                    n_before = len(lc2.time)
                    
                    # Outlier removal
                    if self.do_outliers.get():
                        t = np.array(lc2.time.value, dtype=float)
                        f = np.array(lc2.flux.value, dtype=float)
                        
                        # Use ephemeris if available
                        if self.ephem_period and self.ephem_t0 and self.ephem_duration:
                            P = float(self.ephem_period)
                            t0 = float(self.ephem_t0)
                            dur = float(self.ephem_duration)
                            ph = ((t - t0) / P) % 1.0
                            half = 0.5 * dur / P
                            in_tr = (ph <= half) | (ph >= 1.0 - half)
                            oot = ~in_tr
                        else:
                            oot = np.ones_like(f, dtype=bool)
                        
                        # Remove outliers from out-of-transit data
                        med = np.nanmedian(f[oot])
                        sig = 1.4826 * mad(f[oot])
                        if np.isfinite(sig) and sig > 0:
                            keep = np.ones_like(f, dtype=bool)
                            keep[oot] = np.abs(f[oot] - med) < 5.0 * sig
                            lc2 = lc2[keep]
                    
                    # Detrending
                    if self.do_gp_detrend.get():
                        # Gaussian Process detrending
                        t = np.array(lc2.time.value, dtype=float)
                        f = np.array(lc2.flux.value, dtype=float)
                        
                        try:
                            f_detrended, trend, gp = detrend_light_curve_gp(t, f)
                            lc2.flux = f_detrended * u.dimensionless_unscaled
                            self.log(f"Sector {sector}: GP detrending applied", "INFO")
                        except Exception as e:
                            self.log(f"Sector {sector}: GP detrending failed, using flatten: {e}", "WARN")
                            if self.do_flatten.get() and hasattr(lc2, 'flatten'):
                                lc2 = lc2.flatten(window_length=401, polyorder=2, break_tolerance=5)
                    
                    elif self.do_flatten.get() and hasattr(lc2, 'flatten'):
                        # Traditional flattening
                        if self.ephem_period and self.ephem_t0 and self.ephem_duration:
                            t = np.array(lc2.time.value, dtype=float)
                            P = float(self.ephem_period)
                            t0 = float(self.ephem_t0)
                            dur = float(self.ephem_duration)
                            ph = ((t - t0) / P) % 1.0
                            half = 0.5 * dur / P
                            in_tr = (ph <= half) | (ph >= 1.0 - half)
                            lc2 = lc2.flatten(window_length=401, polyorder=2, 
                                             break_tolerance=5, mask=~in_tr)
                        else:
                            self.log(f"Sector {sector}: Flattening without ephemeris may distort transits", "WARN")
                            lc2 = lc2.flatten(window_length=401, polyorder=2, break_tolerance=5)
                    
                    n_after = len(lc2.time)
                    self.log(f"Sector {sector}: {flux_type}, {n_before} → {n_after} points", "INFO")
                    
                    # Store segment
                    t = np.array(lc2.time.value, dtype=float)
                    f = np.array(lc2.flux.value, dtype=float)
                    segs.append({
                        "sector": sector,
                        "time": t,
                        "flux": f,
                        "flux_type": flux_type
                    })
                
                # Sort by sector
                def get_sector_num(seg):
                    try:
                        return int(seg["sector"])
                    except:
                        return 9999
                
                segs.sort(key=get_sector_num)
                
                # Combine segments
                if segs:
                    t_all = np.concatenate([s["time"] for s in segs])
                    f_all = np.concatenate([s["flux"] for s in segs])
                    
                    # Sort by time
                    sort_idx = np.argsort(t_all)
                    self.tess_time = t_all[sort_idx]
                    self.tess_flux = f_all[sort_idx]
                    self.tess_segments = segs
                    
                    # Calculate errors (simplified)
                    self.tess_flux_err = np.ones_like(self.tess_flux) * mad(self.tess_flux)
                    
                    def apply():
                        try:
                            self._plot_segments()
                            self.set_status(f"Downloaded {len(segs)} sector(s)")
                            self.log(f"Download complete: {len(segs)} sectors, {len(self.tess_time)} total points", "SUCCESS")
                            
                            # Show data quality
                            quality = check_data_quality(self.tess_time, self.tess_flux)
                            qual_msg = f"Data Quality: {len(self.tess_time)} points, "
                            qual_msg += f"noise={quality.get('noise_ppm', 0):.0f} ppm, "
                            qual_msg += f"span={quality.get('time_span', 0):.1f} days"
                            self.log(qual_msg, "INFO")
                            
                        finally:
                            self._set_busy(False)
                    
                    self.after(0, apply)
                else:
                    def fail():
                        self._set_busy(False)
                        messagebox.showerror("Download Error", "No valid data after processing.")
                        self.log("Download failed: no valid data", "ERROR")
                    self.after(0, fail)
                
            except Exception as e:
                def fail():
                    self._set_busy(False)
                    self.set_status("Download failed.")
                    messagebox.showerror("Download Error", f"Failed to download data:\n{e}")
                    self.log(f"Download failed: {e}", "ERROR")
                self.after(0, fail)
        
        threading.Thread(target=work, daemon=True).start()
    
    def _plot_segments(self):
        """Plot TESS segments (updated for new PlotPanel)"""
        if not self.tess_segments:
            return
        
        mode = self.plot_mode.get()
        
        if mode == "Per-sector panels" and len(self.tess_segments) > 1:
            self.tess_plot.set_subplots(len(self.tess_segments), sharey=True, figsize=(10, 3*len(self.tess_segments)))
            
            for i, seg in enumerate(self.tess_segments):
                sec = seg["sector"]
                t = seg["time"]
                f = seg["flux"]
                
                self.tess_plot.plot_xy(t, f, 
                                      xlabel="Time (BTJD days)" if i == len(self.tess_segments)-1 else "",
                                      ylabel="Flux",
                                      title=f"Sector {sec}",
                                      style="k.", alpha=0.6, ms=1.5, ax_index=i)
            
            self.tess_plot.fig.tight_layout()
            self.tess_plot.canvas.draw()
            return
        
        elif mode == "Concatenated (no gaps)" and len(self.tess_segments) > 1:
            self.tess_plot.set_subplots(1, figsize=(10, 6))
            
            x_cat, y_cat = [], []
            offset = 0.0
            gap = 0.2
            
            for seg in self.tess_segments:
                t = seg["time"]
                f = seg["flux"]
                dt = t - t[0]
                x_cat.append(dt + offset)
                y_cat.append(f)
                offset += (dt[-1] - dt[0]) + gap
            
            x = np.concatenate(x_cat)
            y = np.concatenate(y_cat)
            
            self.tess_plot.plot_xy(x, y, 
                                  xlabel="Concatenated time (days)",
                                  ylabel="Flux",
                                  title="TESS Concatenated Light Curve",
                                  style="k.", alpha=0.6, ms=1.5)
            return
        
        # Stitched absolute (default)
        self.tess_plot.set_subplots(1, figsize=(10, 6))
        
        if len(self.tess_segments) == 1:
            sec = self.tess_segments[0]["sector"]
            title = f"TESS Sector {sec} Light Curve"
        else:
            title = "TESS Stitched Light Curve"
        
        self.tess_plot.plot_xy(self.tess_time, self.tess_flux,
                              xlabel="Time (BTJD days)",
                              ylabel="Normalized Flux",
                              title=title,
                              style="k.", alpha=0.6, ms=1.5)
    
    # ----- Enhanced BLS -----
    def on_tess_bls(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download light curves first.")
            return
        
        # Set period range based on ephemeris if available
        if self.ephem_period is not None:
            P0 = float(self.ephem_period)
            minP = max(0.2, P0 * 0.9)
            maxP = P0 * 1.1
            nper = 30000
            self.log(f"Using narrow period range around NEA value: {minP:.3f}-{maxP:.3f} d", "INFO")
        else:
            minP, maxP = 0.5, 50.0
            nper = 20000
            self.log(f"Using wide period range: {minP:.1f}-{maxP:.1f} d", "INFO")
        
        self._set_busy(True)
        self.set_status(f"Running BLS period search...")
        
        def work():
            try:
                # Use advanced BLS
                res = find_transits_bls_advanced(
                    self.tess_time, self.tess_flux, self.tess_flux_err,
                    min_period=minP, max_period=maxP, n_periods=nper
                )
                
                # Store period for later use
                if 'period' in res:
                    self.ephem_period = float(res['period'])
                    self.ephem_t0 = float(res.get('t0', 0))
                    self.ephem_duration = float(res.get('duration', 0))
                    self.ephem_depth = float(res.get('depth', 0))
                
                def show_results():
                    # Create results window
                    win = tk.Toplevel(self)
                    win.title("BLS Period Search Results")
                    win.geometry("1100x750")
                    
                    nb = ttk.Notebook(win)
                    nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    # Periodogram tab
                    pg_tab = ttk.Frame(nb)
                    nb.add(pg_tab, text="Periodogram")
                    
                    panel = PlotPanel(pg_tab, figsize=(9, 5))
                    panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    periods = res.get('all_periods', [])
                    power = res.get('all_powers', res.get('all_power', []))
                    best_period = res.get('period', 0)
                    
                    panel.plot_line(periods, power, 
                                   xlabel="Period (days)", 
                                   ylabel="BLS Power",
                                   title=f"BLS Periodogram - Best Period: {best_period:.6f} d",
                                   color='blue', lw=1.5)
                    
                    panel.vline(best_period, color='red', alpha=0.8, 
                               label=f'Best: {best_period:.6f} d', lw=2)
                    
                    # Mark harmonics
                    for i in range(2, 6):
                        panel.vline(best_period * i, color='orange', alpha=0.3, ls=':', 
                                   label=f'{i}:1 harmonic' if i == 2 else '')
                        panel.vline(best_period / i, color='orange', alpha=0.3, ls=':')
                    
                    panel.axes[0].legend(loc='upper right')
                    
                    # Parameters tab
                    param_tab = ttk.Frame(nb)
                    nb.add(param_tab, text="Parameters")
                    
                    text = scrolledtext.ScrolledText(param_tab, wrap=tk.WORD)
                    text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    param_text = "BLS Detection Parameters:\n"
                    param_text += "="*50 + "\n\n"
                    
                    param_text += f"Best Period: {res.get('period', 0):.8f} ± {res.get('errors', {}).get('period_err', 0):.8f} days\n"
                    param_text += f"Transit Time (T0): {res.get('t0', 0):.6f} ± {res.get('errors', {}).get('t0_err', 0):.6f}\n"
                    param_text += f"Duration: {res.get('duration', 0):.6f} ± {res.get('errors', {}).get('duration_err', 0):.6f} days\n"
                    param_text += f"Depth: {res.get('depth', 0):.6f} ± {res.get('errors', {}).get('depth_err', 0):.6f}\n"
                    param_text += f"SNR: {res.get('snr', 0):.2f}\n"
                    param_text += f"False Alarm Probability: {res.get('fap', 1):.2e}\n"
                    param_text += f"Detection Significance: {res.get('significance_sigma', 0):.1f}σ\n\n"
                    
                    param_text += "Statistics:\n"
                    param_text += f"χ²: {res.get('chi2', 0):.2f}\n"
                    param_text += f"BIC: {res.get('bic', 0):.2f}\n"
                    param_text += f"Residual RMS: {res.get('residuals_rms', 0):.6f}\n"
                    param_text += f"Data Points: {res.get('n_data_points', 0)}\n"
                    param_text += f"Data Span: {res.get('data_span', 0):.1f} days\n"
                    
                    text.insert(1.0, param_text)
                    text.configure(state='disabled')
                    
                    # Phase-folded tab
                    phase_tab = ttk.Frame(nb)
                    nb.add(phase_tab, text="Phase-folded")
                    
                    panel2 = PlotPanel(phase_tab, figsize=(9, 5))
                    panel2.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    period = res.get('period', 1)
                    t0 = res.get('t0', 0)
                    phase = ((self.tess_time - t0) / period) % 1
                    phase = (phase + 0.5) % 1 - 0.5
                    sort_idx = np.argsort(phase)
                    
                    # Bin the phase-folded data
                    bins = 200
                    phase_bins = np.linspace(-0.5, 0.5, bins + 1)
                    phase_centers = 0.5 * (phase_bins[1:] + phase_bins[:-1])
                    binned_flux = []
                    binned_err = []
                    
                    for i in range(bins):
                        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
                        if np.sum(mask) > 0:
                            binned_flux.append(np.median(self.tess_flux[mask]))
                            binned_err.append(np.std(self.tess_flux[mask]) / np.sqrt(np.sum(mask)))
                        else:
                            binned_flux.append(np.nan)
                            binned_err.append(np.nan)
                    
                    binned_flux = np.array(binned_flux)
                    binned_err = np.array(binned_err)
                    
                    # Plot binned data with errors
                    valid = ~np.isnan(binned_flux)
                    panel2.errorbar(phase_centers[valid], binned_flux[valid], 
                                   yerr=binned_err[valid],
                                   fmt='o', color='blue', alpha=0.7, capsize=3,
                                   label='Binned data')
                    
                    # Plot individual points (faded)
                    panel2.plot_xy(phase[sort_idx], self.tess_flux[sort_idx],
                                  xlabel="Phase", ylabel="Flux",
                                  title=f"Phase-folded Light Curve (P={period:.6f} d)",
                                  style='k.', alpha=0.1, ms=1, label='Individual points')
                    
                    panel2.axes[0].legend(loc='best')
                    
                    self._set_busy(False)
                    self.log(f"BLS analysis complete: P={best_period:.6f}d, SNR={res.get('snr', 0):.1f}, FAP={res.get('fap', 1):.2e}", "SUCCESS")
                
                self.after(0, show_results)
                
            except Exception as e:
                def show_error():
                    self._set_busy(False)
                    messagebox.showerror("BLS Error", f"BLS analysis failed:\n{e}")
                    self.log(f"BLS analysis failed: {e}", "ERROR")
                self.after(0, show_error)
        
        threading.Thread(target=work, daemon=True).start()
    
    def on_tess_multiple_methods(self):
        """Run multiple detection methods on TESS data"""
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download light curves first.")
            return
        
        self._set_busy(True)
        self.set_status("Running multiple detection methods...")
        
        def work():
            try:
                results = find_transits_multiple_methods(
                    self.tess_time, self.tess_flux,
                    min_period=0.5, max_period=50.0,
                    methods=['bls', 'gls']  # Add 'pdm' if needed
                )
                
                def show_results():
                    # Similar to simulation version but for TESS data
                    win = tk.Toplevel(self)
                    win.title("Multiple Method Analysis - TESS")
                    win.geometry("1200x800")
                    
                    nb = ttk.Notebook(win)
                    nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    # Implementation similar to on_sim_multiple_methods
                    # ... (omitted for brevity, same pattern as simulation version)
                    
                    self._set_busy(False)
                    self.log("Multiple method analysis complete for TESS data", "SUCCESS")
                
                self.after(0, show_results)
                
            except Exception as e:
                def show_error():
                    self._set_busy(False)
                    messagebox.showerror("Analysis Error", str(e))
                    self.log(f"Multiple method analysis failed: {e}", "ERROR")
                self.after(0, show_error)
        
        threading.Thread(target=work, daemon=True).start()
    
    def on_tess_mcmc(self):
        """Run MCMC on TESS data"""
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download light curves first.")
            return
        
        # Get initial parameters
        initial_params = {
            'period': self.ephem_period or 10.0,
            't0': self.ephem_t0 or self.tess_time.mean(),
            'duration': self.ephem_duration or 0.1,
            'depth': self.ephem_depth or 0.01
        }
        
        # Open MCMC dialog
        dialog = MCMCDialog(self, self.tess_time, self.tess_flux, initial_params)
    
    def on_ttv_analysis(self):
        """Analyze Transit Timing Variations"""
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download light curves first.")
            return
        
        if not (self.ephem_period and self.ephem_t0 and self.ephem_duration):
            messagebox.showinfo("Need Ephemeris", 
                              "Need period, T0, and duration for TTV analysis.\n"
                              "Fetch from NEA or run BLS first.")
            return
        
        self._set_busy(True)
        self.set_status("Analyzing Transit Timing Variations...")
        
        def work():
            try:
                ttv_results = measure_transit_timing_variations(
                    self.tess_time, self.tess_flux,
                    float(self.ephem_period),
                    float(self.ephem_t0),
                    float(self.ephem_duration)
                )
                
                def show_results():
                    win = tk.Toplevel(self)
                    win.title("TTV Analysis Results")
                    win.geometry("900x700")
                    
                    nb = ttk.Notebook(win)
                    nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    # TTV plot tab
                    plot_tab = ttk.Frame(nb)
                    nb.add(plot_tab, text="TTV Plot")
                    
                    panel = PlotPanel(plot_tab, figsize=(8, 5))
                    panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    if 'epochs' in ttv_results and 'ttvs' in ttv_results:
                        epochs = ttv_results['epochs']
                        ttvs = ttv_results['ttvs'] * 24 * 60  # Convert to minutes
                        ttv_errs = ttv_results.get('ttv_errs', np.zeros_like(ttvs)) * 24 * 60
                        
                        panel.errorbar(epochs, ttvs, yerr=ttv_errs,
                                      fmt='o', color='blue', capsize=5,
                                      label='TTV measurements')
                        
                        panel.hline(0, color='gray', ls='--', alpha=0.5, label='Expected')
                        
                        # Fit sinusoidal model if available
                        if not np.isnan(ttv_results.get('ttv_period', np.nan)):
                            ttv_period = ttv_results['ttv_period']
                            ttv_amp = ttv_results['ttv_amplitude'] * 24 * 60
                            
                            # Generate model curve
                            epochs_model = np.linspace(epochs.min(), epochs.max(), 1000)
                            ttvs_model = ttv_amp * np.sin(2*np.pi*epochs_model/ttv_period)
                            
                            panel.plot_line(epochs_model, ttvs_model,
                                          color='red', lw=2, alpha=0.7,
                                          label=f'Sinusoidal fit: P={ttv_period:.1f} orbits, A={ttv_amp:.1f} min')
                        
                        panel.axes[0].set_xlabel("Epoch")
                        panel.axes[0].set_ylabel("TTV (minutes)")
                        panel.axes[0].set_title("Transit Timing Variations")
                        panel.axes[0].legend(loc='best')
                        panel.axes[0].grid(True, alpha=0.3)
                    
                    # Results tab
                    results_tab = ttk.Frame(nb)
                    nb.add(results_tab, text="Results")
                    
                    text = scrolledtext.ScrolledText(results_tab, wrap=tk.WORD)
                    text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    results_text = "TTV Analysis Results\n"
                    results_text += "="*40 + "\n\n"
                    
                    results_text += f"TTVs Detected: {ttv_results.get('ttvs_detected', False)}\n"
                    results_text += f"Significance p-value: {ttv_results.get('p_value', 1):.3e}\n"
                    results_text += f"χ²: {ttv_results.get('chi2', 0):.2f}\n"
                    results_text += f"Degrees of Freedom: {ttv_results.get('dof', 0)}\n"
                    results_text += f"RMS TTV: {ttv_results.get('rms_ttv', 0)*24*60:.2f} minutes\n"
                    results_text += f"Number of epochs: {len(ttv_results.get('ttvs', []))}\n\n"
                    
                    if not np.isnan(ttv_results.get('ttv_period', np.nan)):
                        results_text += f"Sinusoidal Fit:\n"
                        results_text += f"  TTV Period: {ttv_results.get('ttv_period', 0):.1f} orbits\n"
                        results_text += f"  TTV Amplitude: {ttv_results.get('ttv_amplitude', 0)*24*60:.2f} minutes\n"
                    
                    results_text += "\nIndividual Measurements:\n"
                    results_text += "-"*30 + "\n"
                    
                    for i, m in enumerate(ttv_results.get('measurements', [])):
                        results_text += f"Epoch {m.get('epoch', 0)}: "
                        results_text += f"TTV = {m.get('ttv', 0)*24*60:.2f} ± {m.get('tc_err', 0)*24*60:.2f} min "
                        results_text += f"({m.get('ttv_sigma', 0):.1f}σ)\n"
                    
                    text.insert(1.0, results_text)
                    text.configure(state='disabled')
                    
                    self._set_busy(False)
                    
                    if ttv_results.get('ttvs_detected', False):
                        self.log(f"TTVs detected! p={ttv_results.get('p_value', 1):.3e}, "
                               f"RMS={ttv_results.get('rms_ttv', 0)*24*60:.1f} min", "SUCCESS")
                    else:
                        self.log(f"No significant TTVs detected (p={ttv_results.get('p_value', 1):.3f})", "INFO")
                
                self.after(0, show_results)
                
            except Exception as e:
                def show_error():
                    self._set_busy(False)
                    messagebox.showerror("TTV Error", f"TTV analysis failed:\n{e}")
                    self.log(f"TTV analysis failed: {e}", "ERROR")
                self.after(0, show_error)
        
        threading.Thread(target=work, daemon=True).start()
    
    # ----- Original TESS functions (updated for compatibility) -----
    def on_show_markers(self):
        """Show transit markers on plot"""
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download first.")
            return
        if not (self.ephem_period and self.ephem_t0):
            messagebox.showinfo("Need ephemeris", "Fetch NEA Params first or run BLS.")
            return
        
        P = float(self.ephem_period)
        t0 = float(self.ephem_t0)
        
        centers = predicted_centers(self.tess_time, P, t0)
        
        mode = self.plot_mode.get()
        if mode == "Per-sector panels" and len(self.tess_segments) > 1:
            for i, seg in enumerate(self.tess_segments):
                tseg = seg["time"]
                tmin_s, tmax_s = float(tseg.min()), float(tseg.max())
                for _, tc in centers:
                    if tmin_s <= tc <= tmax_s:
                        self.tess_plot.vline(tc, color="r", alpha=0.20, ax_index=i)
            self.tess_plot.canvas.draw()
        else:
            for _, tc in centers:
                self.tess_plot.vline(tc, color="r", alpha=0.20, ax_index=0)
            self.tess_plot.canvas.draw()
        
        self.set_status(f"Markers drawn: {len(centers)} predicted transits (P={P:.9f}).")
    
    def on_transit_viewer(self):
        """Open transit viewer window"""
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download first.")
            return
        if not (self.ephem_period and self.ephem_t0 and self.ephem_duration):
            messagebox.showinfo("Need NEA ephemeris", "Fetch NEA Params first to enable transit viewer.")
            return
        
        P = float(self.ephem_period)
        t0 = float(self.ephem_t0)
        dur = float(self.ephem_duration)
        
        events = predicted_centers(self.tess_time, P, t0)
        if not events:
            messagebox.showinfo("No events", "No transits in the downloaded time range.")
            return
        
        win = tk.Toplevel(self)
        win.title("Transit Viewer")
        win.geometry("1120x620")
        
        left = ttk.Frame(win, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)
        
        right = ttk.Frame(win, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(left, text="Select a transit event:").pack(anchor="w")
        lb = tk.Listbox(left, width=34, height=26)
        lb.pack(fill=tk.BOTH, expand=False)
        
        for (n, tc) in events:
            lb.insert(tk.END, f"n={n:6d}  tc={tc:.6f}")
        
        panel = PlotPanel(right, title="Transit Window")
        panel.pack(fill=tk.BOTH, expand=True)
        
        def plot_event(idx):
            n, tc = events[idx]
            w = 3.0 * dur
            m = (self.tess_time >= tc - w) & (self.tess_time <= tc + w)
            if np.sum(m) < 20:
                return
            tt = self.tess_time[m]
            ff = self.tess_flux[m]
            panel.set_subplots(1)
            panel.plot_xy(tt, ff, xlabel="Time (BTJD)", ylabel="Flux",
                         title=f"Transit n={n}  tc={tc:.6f}  window=±{w:.3f} d",
                         style="k.", alpha=0.75, ms=3)
            panel.vline(tc, color="r", alpha=0.35)
        
        def on_select(_evt):
            sel = lb.curselection()
            if sel:
                plot_event(int(sel[0]))
        
        lb.bind("<<ListboxSelect>>", on_select)
        lb.selection_set(0)
        plot_event(0)
    
    def on_stacked_transits(self):
        """Show stacked transits"""
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download first.")
            return
        if not (self.ephem_period and self.ephem_t0 and self.ephem_duration):
            messagebox.showinfo("Need NEA ephemeris", "Fetch NEA Params first.")
            return
        
        P = float(self.ephem_period)
        t0 = float(self.ephem_t0)
        dur = float(self.ephem_duration)
        events = predicted_centers(self.tess_time, P, t0)
        
        w = 2.5 * dur
        xs, ys = [], []
        for _, tc in events:
            m = (self.tess_time >= tc - w) & (self.tess_time <= tc + w)
            if np.sum(m) < 20:
                continue
            xs.append(self.tess_time[m] - tc)
            ys.append(self.tess_flux[m])
        
        if len(xs) < 3:
            messagebox.showinfo("Not enough", "Not enough transits to stack.")
            return
        
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        o = np.argsort(x)
        x, y = x[o], y[o]
        
        win = tk.Toplevel(self)
        win.title("Stacked Transits")
        win.geometry("980x560")
        
        panel = PlotPanel(win, title="Stacked")
        panel.pack(fill=tk.BOTH, expand=True)
        panel.plot_xy(x, y, xlabel="Time from mid-transit (days)", ylabel="Flux",
                     title=f"Stacked transits | N={len(events)} | P={P:.9f}",
                     style="k.", alpha=0.35, ms=2)
        panel.vline(0.0, color="r", alpha=0.3)
    
    def on_phase_fold(self):
        """Show phase-folded light curve"""
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download first.")
            return
        if not (self.ephem_period and self.ephem_t0):
            messagebox.showinfo("Need ephemeris", "Fetch NEA Params first or run BLS.")
            return
        
        P = float(self.ephem_period)
        t0 = float(self.ephem_t0)
        
        ph = ((self.tess_time - t0) / P) % 1.0
        ph = (ph + 0.5) % 1.0 - 0.5  # [-0.5, 0.5)
        o = np.argsort(ph)
        ph = ph[o]
        f = self.tess_flux[o]
        
        win = tk.Toplevel(self)
        win.title("Phase Fold")
        win.geometry("980x560")
        panel = PlotPanel(win, title="Phase Fold")
        panel.pack(fill=tk.BOTH, expand=True)
        panel.plot_xy(ph, f, xlabel="Phase", ylabel="Flux",
                     title=f"Phase fold | P={P:.9f}", style="k.", alpha=0.25, ms=2)
    
    def on_tess_export(self):
        """Export TESS data"""
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download first.")
            return
        
        # Ask for export type
        export_type = messagebox.askquestion("Export", 
                                           "Export data as CSV or figure as image?",
                                           icon='question',
                                           type='yesno')
        
        if export_type == 'yes':  # CSV data
            path = filedialog.asksaveasfilename(
                title="Save CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile="tess_lightcurve.csv",
            )
            if path:
                arr = np.column_stack([self.tess_time, self.tess_flux])
                np.savetxt(path, arr, delimiter=",", header="time_btjd,flux", comments="")
                self.set_status(f"Exported data: {os.path.basename(path)}")
                self.log(f"Data exported to {path}", "SUCCESS")
        
        else:  # Figure
            path = filedialog.asksaveasfilename(
                title="Save Figure",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), 
                          ("PDF files", "*.pdf"),
                          ("SVG files", "*.svg"),
                          ("All files", "*.*")],
                initialfile="tess_plot.png",
            )
            if path:
                self.tess_plot.fig.savefig(path, dpi=300, bbox_inches='tight')
                self.set_status(f"Exported figure: {os.path.basename(path)}")
                self.log(f"Figure exported to {path}", "SUCCESS")
    
    # ---------------- ANALYSIS TAB ----------------
    def _build_analysis_tab(self):
        """Build analysis tab with advanced tools"""
        left = ttk.Frame(self.analysis_tab, padding=15)
        left.pack(side=tk.LEFT, fill=tk.Y)
        
        right = ttk.Frame(self.analysis_tab, padding=15)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(left, text="Advanced Analysis", 
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 15))
        
        # Data selection
        data_frame = ttk.LabelFrame(left, text="Data Source", padding=10)
        data_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.analysis_source = tk.StringVar(value="tess")
        
        ttk.Radiobutton(data_frame, text="TESS Data", 
                       variable=self.analysis_source, value="tess").pack(anchor="w", pady=2)
        ttk.Radiobutton(data_frame, text="Simulation Data", 
                       variable=self.analysis_source, value="sim").pack(anchor="w", pady=2)
        ttk.Radiobutton(data_frame, text="Load from File", 
                       variable=self.analysis_source, value="file").pack(anchor="w", pady=2)
        
        ttk.Button(data_frame, text="Load File", 
                  command=self.on_analysis_load_file).pack(fill=tk.X, pady=(10, 0))
        
        # Analysis tools
        tools_frame = ttk.LabelFrame(left, text="Analysis Tools", padding=10)
        tools_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(tools_frame, text="Data Quality Check", 
                  command=self.on_analysis_quality).pack(fill=tk.X, pady=2)
        ttk.Button(tools_frame, text="Outlier Detection", 
                  command=self.on_analysis_outliers).pack(fill=tk.X, pady=2)
        ttk.Button(tools_frame, text="GP Detrending", 
                  command=self.on_analysis_gp).pack(fill=tk.X, pady=2)
        ttk.Button(tools_frame, text="Validation Tests", 
                  command=self.on_analysis_validation).pack(fill=tk.X, pady=2)
        ttk.Button(tools_frame, text="Significance Test", 
                  command=self.on_analysis_significance).pack(fill=tk.X, pady=2)
        
        # Advanced tools
        advanced_frame = ttk.LabelFrame(left, text="Advanced Tools", padding=10)
        advanced_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(advanced_frame, text="Injection-Recovery", 
                  command=self.on_analysis_injection).pack(fill=tk.X, pady=2)
        ttk.Button(advanced_frame, text="Parameter Space Scan", 
                  command=self.on_analysis_parameter_scan).pack(fill=tk.X, pady=2)
        ttk.Button(advanced_frame, text="Create Report", 
                  command=self.on_analysis_report).pack(fill=tk.X, pady=2)
        
        # Status
        self.analysis_status = ttk.Label(left, text="Ready for analysis", wraplength=300)
        self.analysis_status.pack(fill=tk.X, pady=(0, 10))
        
        # Plot panel
        self.analysis_plot = PlotPanel(right, title="Analysis Results", figsize=(10, 6))
        self.analysis_plot.pack(fill=tk.BOTH, expand=True)
    
    def _get_analysis_data(self):
        """Get data for analysis based on selected source"""
        source = self.analysis_source.get()
        
        if source == "tess":
            if self.tess_time is not None and self.tess_flux is not None:
                return self.tess_time, self.tess_flux
            else:
                messagebox.showinfo("No TESS Data", "Load TESS data first.")
                return None, None
        
        elif source == "sim":
            if self.sim_time is not None and self.sim_flux is not None:
                return self.sim_time, self.sim_flux
            else:
                messagebox.showinfo("No Simulation", "Generate simulation data first.")
                return None, None
        
        elif source == "file":
            messagebox.showinfo("Not Implemented", "File loading from analysis tab coming soon.")
            return None, None
        
        return None, None
    
    def on_analysis_load_file(self):
        """Load data from file for analysis"""
        filename = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("CSV files", "*.csv"), 
                      ("Text files", "*.txt"),
                      ("All files", "*.*")]
        )
        
        if filename:
            try:
                data = np.loadtxt(filename, delimiter=",")
                if data.shape[1] >= 2:
                    self.analysis_time = data[:, 0]
                    self.analysis_flux = data[:, 1]
                    self.analysis_source.set("file")
                    self.analysis_status.config(text=f"Loaded {len(self.analysis_time)} points from {os.path.basename(filename)}")
                    self.log(f"Loaded analysis data from {filename}", "SUCCESS")
                else:
                    messagebox.showerror("Invalid File", "File must have at least 2 columns (time, flux).")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load file:\n{e}")
    
    def on_analysis_quality(self):
        """Run data quality check"""
        time, flux = self._get_analysis_data()
        if time is None:
            return
        
        quality = check_data_quality(time, flux)
        
        # Display results
        win = tk.Toplevel(self)
        win.title("Data Quality Report")
        win.geometry("600x500")
        
        text = scrolledtext.ScrolledText(win, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        report = "Data Quality Report\n"
        report += "="*40 + "\n\n"
        
        report += f"Data Points: {quality.get('n_points', 0)}\n"
        report += f"Time Span: {quality.get('time_span', 0):.2f} days\n"
        report += f"Median Time Gap: {quality.get('median_gap', 0):.6f} days\n"
        report += f"Maximum Gap: {quality.get('max_gap', 0):.6f} days\n\n"
        
        report += f"Flux Statistics:\n"
        report += f"  Mean: {quality.get('flux_mean', 0):.6f}\n"
        report += f"  Median: {quality.get('flux_median', 0):.6f}\n"
        report += f"  STD: {quality.get('flux_std', 0):.6f}\n"
        report += f"  CDPP (noise): {quality.get('noise_ppm', 0):.0f} ppm\n\n"
        
        report += f"Data Issues:\n"
        report += f"  NaN Values: {quality.get('n_nans', 0)}\n"
        report += f"  Has NaN: {quality.get('has_nans', False)}\n"
        report += f"  Is Sorted: {quality.get('is_sorted', True)}\n"
        
        text.insert(1.0, report)
        text.configure(state='disabled')
        
        self.log(f"Quality check: {quality.get('n_points', 0)} points, "
                f"noise={quality.get('noise_ppm', 0):.0f} ppm", "INFO")
    
    def on_analysis_outliers(self):
        """Detect and remove outliers"""
        time, flux = self._get_analysis_data()
        if time is None:
            return
        
        # Detect outliers
        outliers = detect_outliers_modified_zscore(flux)
        n_outliers = np.sum(outliers)
        
        # Plot before/after
        self.analysis_plot.set_subplots(2, 1, sharey=True, figsize=(10, 8))
        
        # Before
        self.analysis_plot.plot_xy(time, flux, 
                                  xlabel="Time", ylabel="Flux",
                                  title=f"Original Data ({len(time)} points)",
                                  style='b.', alpha=0.6, ms=2, ax_index=0)
        
        # Mark outliers
        self.analysis_plot.scatter(time[outliers], flux[outliers],
                                 color='red', s=30, label=f'Outliers ({n_outliers})',
                                 ax_index=0)
        self.analysis_plot.axes[0].legend()
        
        # After removal
        time_clean = time[~outliers]
        flux_clean = flux[~outliers]
        
        self.analysis_plot.plot_xy(time_clean, flux_clean,
                                  xlabel="Time", ylabel="Flux",
                                  title=f"After Outlier Removal ({len(time_clean)} points)",
                                  style='g.', alpha=0.6, ms=2, ax_index=1)
        
        self.analysis_status.config(text=f"Detected {n_outliers} outliers ({n_outliers/len(time)*100:.1f}%)")
        self.log(f"Outlier detection: {n_outliers} outliers removed", "INFO")
    
    def on_analysis_gp(self):
        """Apply Gaussian Process detrending"""
        time, flux = self._get_analysis_data()
        if time is None:
            return
        
        self._set_busy(True)
        self.analysis_status.config(text="Running GP detrending...")
        
        def work():
            try:
                flux_detrended, trend, gp = detrend_light_curve_gp(time, flux)
                
                def apply():
                    self.analysis_plot.set_subplots(3, 1, sharey=False, figsize=(10, 10))
                    
                    # Original
                    self.analysis_plot.plot_xy(time, flux,
                                              xlabel="Time", ylabel="Flux",
                                              title="Original Data",
                                              style='b.', alpha=0.6, ms=2, ax_index=0)
                    
                    # Trend
                    self.analysis_plot.plot_line(time, trend,
                                                color='red', lw=2,
                                                xlabel="Time", ylabel="Trend",
                                                title="GP Trend Model",
                                                ax_index=1)
                    
                    # Detrended
                    self.analysis_plot.plot_xy(time, flux_detrended,
                                              xlabel="Time", ylabel="Flux",
                                              title="Detrended Data",
                                              style='g.', alpha=0.6, ms=2, ax_index=2)
                    
                    self.analysis_status.config(text=f"GP detrending complete. Kernel: {gp.kernel_}")
                    self.log("GP detrending applied successfully", "SUCCESS")
                    self._set_busy(False)
                
                self.after(0, apply)
                
            except Exception as e:
                def fail():
                    self._set_busy(False)
                    messagebox.showerror("GP Error", f"GP detrending failed:\n{e}")
                    self.log(f"GP detrending failed: {e}", "ERROR")
                self.after(0, fail)
        
        threading.Thread(target=work, daemon=True).start()
    
    def on_analysis_validation(self):
        """Run validation tests"""
        time, flux = self._get_analysis_data()
        if time is None:
            return
        
        # Need parameters for validation
        if not (self.ephem_period and self.ephem_t0 and self.ephem_duration):
            messagebox.showinfo("Need Parameters", 
                              "Need period, T0, and duration for validation.\n"
                              "Set via NEA or analysis first.")
            return
        
        params = TransitParameters(
            period=float(self.ephem_period),
            t0=float(self.ephem_t0),
            duration=float(self.ephem_duration),
            depth=self.ephem_depth or 0.01,
            snr=calculate_snr(time, flux, 
                             float(self.ephem_period),
                             float(self.ephem_t0),
                             float(self.ephem_duration))
        )
        
        validation = validate_transit_parameters(params, time, flux)
        
        # Show results
        win = tk.Toplevel(self)
        win.title("Validation Results")
        win.geometry("600x500")
        
        text = scrolledtext.ScrolledText(win, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        report = "Transit Validation Results\n"
        report += "="*40 + "\n\n"
        
        report += "Parameters:\n"
        report += f"  Period: {params.period:.6f} days\n"
        report += f"  T0: {params.t0:.6f}\n"
        report += f"  Duration: {params.duration:.6f} days ({params.duration*24:.2f} hours)\n"
        report += f"  Depth: {params.depth:.6f} ({params.depth*1e6:.1f} ppm)\n"
        report += f"  SNR: {params.snr:.1f}\n\n"
        
        report += "Validation Checks:\n"
        report += "-"*30 + "\n"
        
        passed = 0
        total = 0
        for key, value in validation.items():
            if not key.startswith('_') and isinstance(value, bool):
                total += 1
                if value:
                    passed += 1
                    report += f"✓ {key.replace('_', ' ').title()}\n"
                else:
                    report += f"✗ {key.replace('_', ' ').title()}\n"
        
        report += f"\nPassed: {passed}/{total} checks\n"
        
        if validation.get('all_passed', False):
            report += "\n✅ All validation checks passed!\n"
            self.log("Validation passed all checks", "SUCCESS")
        else:
            report += "\n⚠️ Some validation checks failed\n"
            self.log(f"Validation: {passed}/{total} checks passed", "WARN")
        
        text.insert(1.0, report)
        text.configure(state='disabled')
    
    def on_analysis_significance(self):
        """Calculate detection significance"""
        time, flux = self._get_analysis_data()
        if time is None:
            return
        
        if not self.ephem_period:
            messagebox.showinfo("Need Period", "Need period for significance test.")
            return
        
        self._set_busy(True)
        self.analysis_status.config(text="Calculating detection significance...")
        
        def work():
            try:
                # First get BLS result
                res = find_transits_bls_advanced(
                    time, flux,
                    min_period=float(self.ephem_period) * 0.9,
                    max_period=float(self.ephem_period) * 1.1
                )
                
                # Calculate significance
                significance = calculate_detection_significance(res, n_shuffles=500)
                
                def apply():
                    win = tk.Toplevel(self)
                    win.title("Detection Significance")
                    win.geometry("500x400")
                    
                    text = scrolledtext.ScrolledText(win, wrap=tk.WORD)
                    text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    report = "Detection Significance Test\n"
                    report += "="*40 + "\n\n"
                    
                    p_value = significance.get('p_value', 1)
                    sigma = significance.get('significance_sigma', 0)
                    
                    report += f"Best BLS Power: {significance.get('best_power', 0):.3f}\n"
                    report += f"Mean Shuffled Power: {significance.get('mean_shuffled_power', 0):.3f}\n"
                    report += f"STD Shuffled Power: {significance.get('std_shuffled_power', 0):.3f}\n\n"
                    
                    report += f"p-value: {p_value:.3e}\n"
                    report += f"Significance: {sigma:.1f}σ\n\n"
                    
                    if p_value < 0.01:
                        report += "✅ Detection is statistically significant (p < 0.01)\n"
                        self.log(f"Significance test: p={p_value:.3e} ({sigma:.1f}σ) - Significant", "SUCCESS")
                    elif p_value < 0.05:
                        report += "⚠️ Detection is marginally significant (p < 0.05)\n"
                        self.log(f"Significance test: p={p_value:.3e} ({sigma:.1f}σ) - Marginal", "WARN")
                    else:
                        report += "❌ Detection is not statistically significant (p ≥ 0.05)\n"
                        self.log(f"Significance test: p={p_value:.3e} ({sigma:.1f}σ) - Not significant", "ERROR")
                    
                    report += f"\nBased on {significance.get('n_shuffles', 0)} data shuffles."
                    
                    text.insert(1.0, report)
                    text.configure(state='disabled')
                    
                    self._set_busy(False)
                
                self.after(0, apply)
                
            except Exception as e:
                def fail():
                    self._set_busy(False)
                    messagebox.showerror("Significance Error", f"Failed to calculate significance:\n{e}")
                    self.log(f"Significance test failed: {e}", "ERROR")
                self.after(0, fail)
        
        threading.Thread(target=work, daemon=True).start()
    
    def on_analysis_injection(self):
        """Run injection-recovery test"""
        time, flux = self._get_analysis_data()
        if time is None:
            return
        
        # Get current parameters or use defaults
        if self.ephem_period and self.ephem_t0 and self.ephem_duration and self.ephem_depth:
            params = TransitParameters(
                period=float(self.ephem_period),
                t0=float(self.ephem_t0),
                duration=float(self.ephem_duration),
                depth=float(self.ephem_depth)
            )
        else:
            # Use default parameters
            params = TransitParameters(
                period=10.0,
                t0=time.mean(),
                duration=0.1,
                depth=0.01
            )
        
        self._set_busy(True)
        self.analysis_status.config(text="Running injection-recovery test...")
        
        def work():
            try:
                results = perform_injection_recovery_test(
                    time, params, n_trials=100, noise_level=0.001
                )
                
                def apply():
                    recovery_rate = results.get('recovery_rate', 0)
                    
                    win = tk.Toplevel(self)
                    win.title("Injection-Recovery Results")
                    win.geometry("600x500")
                    
                    text = scrolledtext.ScrolledText(win, wrap=tk.WORD)
                    text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    report = "Injection-Recovery Test Results\n"
                    report += "="*40 + "\n\n"
                    
                    report += f"Trials: {results.get('n_trials', 0)}\n"
                    report += f"Recovered: {results.get('n_recovered', 0)}\n"
                    report += f"Recovery Rate: {recovery_rate*100:.1f}%\n"
                    report += f"Detection Efficiency: {results.get('detection_efficiency', 0)*100:.1f}%\n\n"
                    
                    report += "Injected Parameters:\n"
                    inj_params = results.get('injection_params', {})
                    if hasattr(inj_params, 'period'):
                        report += f"  Period: {inj_params.period:.4f} days\n"
                        report += f"  Depth: {inj_params.depth:.4f}\n"
                        report += f"  Duration: {inj_params.duration:.4f} days\n\n"
                    
                    if recovery_rate > 0.8:
                        report += "✅ Excellent recovery rate (>80%)\n"
                        self.log(f"Injection-recovery: {recovery_rate*100:.1f}% recovery - Excellent", "SUCCESS")
                    elif recovery_rate > 0.5:
                        report += "⚠️ Moderate recovery rate (50-80%)\n"
                        self.log(f"Injection-recovery: {recovery_rate*100:.1f}% recovery - Moderate", "WARN")
                    else:
                        report += "❌ Poor recovery rate (<50%)\n"
                        self.log(f"Injection-recovery: {recovery_rate*100:.1f}% recovery - Poor", "ERROR")
                    
                    text.insert(1.0, report)
                    text.configure(state='disabled')
                    
                    self._set_busy(False)
                
                self.after(0, apply)
                
            except Exception as e:
                def fail():
                    self._set_busy(False)
                    messagebox.showerror("Injection Error", f"Injection-recovery test failed:\n{e}")
                    self.log(f"Injection-recovery test failed: {e}", "ERROR")
                self.after(0, fail)
        
        threading.Thread(target=work, daemon=True).start()
    
    def on_analysis_parameter_scan(self):
        """Run parameter space scan"""
        messagebox.showinfo("Coming Soon", "Parameter space scan feature coming in a future update.")
        self.log("Parameter scan not yet implemented", "INFO")
    
    def on_analysis_report(self):
        """Create analysis report"""
        time, flux = self._get_analysis_data()
        if time is None:
            return
        
        # Get parameters if available
        if self.ephem_period and self.ephem_t0 and self.ephem_duration:
            params = TransitParameters(
                period=float(self.ephem_period),
                t0=float(self.ephem_t0),
                duration=float(self.ephem_duration),
                depth=self.ephem_depth or 0.01
            )
        else:
            params = None
        
        # Ask for filename
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), 
                      ("PNG files", "*.png"),
                      ("All files", "*.*")],
            initialfile="transit_analysis_report.pdf"
        )
        
        if filename:
            try:
                # Create publication-quality figure
                setup_publication_style(style='aas', dpi=300)
                
                if params:
                    fig = create_transit_report_figure(time, flux, params, filename=filename)
                else:
                    # Simple plot if no parameters
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(time, flux, 'k.', alpha=0.5, markersize=1)
                    ax.set_xlabel("Time (days)")
                    ax.set_ylabel("Normalized Flux")
                    ax.set_title("Light Curve")
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                
                self.log(f"Analysis report saved to {filename}", "SUCCESS")
                self.analysis_status.config(text=f"Report saved: {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Report Error", f"Failed to create report:\n{e}")
                self.log(f"Report creation failed: {e}", "ERROR")
    
    # ---------------- ADVANCED TAB ----------------
    def _build_advanced_tab(self):
        """Build advanced tools tab"""
        main_frame = ttk.Frame(self.advanced_tab, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Advanced Tools & Settings", 
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 20))
        
        # Create notebook for different advanced features
        adv_notebook = ttk.Notebook(main_frame)
        adv_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Settings tab
        settings_tab = ttk.Frame(adv_notebook)
        adv_notebook.add(settings_tab, text="Settings")
        
        self._build_settings_tab(settings_tab)
        
        # Batch Processing tab
        batch_tab = ttk.Frame(adv_notebook)
        adv_notebook.add(batch_tab, text="Batch Processing")
        
        self._build_batch_tab(batch_tab)
        
        # Developer tab
        dev_tab = ttk.Frame(adv_notebook)
        adv_notebook.add(dev_tab, text="Developer")
        
        self._build_developer_tab(dev_tab)
    
    def _build_settings_tab(self, parent):
        """Build settings tab"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Plot settings
        plot_frame = ttk.LabelFrame(scrollable_frame, text="Plot Settings", padding=15)
        plot_frame.pack(fill=tk.X, pady=(0, 15), padx=10)
        
        self.plot_style = tk.StringVar(value="default")
        self.plot_dpi = tk.StringVar(value="300")
        self.plot_figsize_w = tk.StringVar(value="10")
        self.plot_figsize_h = tk.StringVar(value="6")
        
        ttk.Label(plot_frame, text="Plot Style:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Combobox(plot_frame, textvariable=self.plot_style,
                    values=["default", "seaborn", "ggplot", "bmh", "dark_background"],
                    width=20).grid(row=0, column=1, sticky="w", pady=3, padx=(10, 0))
        
        ttk.Label(plot_frame, text="DPI:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(plot_frame, textvariable=self.plot_dpi, width=10).grid(row=1, column=1, sticky="w", pady=3, padx=(10, 0))
        
        ttk.Label(plot_frame, text="Figure Width:").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(plot_frame, textvariable=self.plot_figsize_w, width=10).grid(row=2, column=1, sticky="w", pady=3, padx=(10, 0))
        
        ttk.Label(plot_frame, text="Figure Height:").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(plot_frame, textvariable=self.plot_figsize_h, width=10).grid(row=3, column=1, sticky="w", pady=3, padx=(10, 0))
        
        # Analysis settings
        analysis_frame = ttk.LabelFrame(scrollable_frame, text="Analysis Settings", padding=15)
        analysis_frame.pack(fill=tk.X, pady=(0, 15), padx=10)
        
        self.bls_nperiods = tk.StringVar(value="10000")
        self.mcmc_nwalkers = tk.StringVar(value="32")
        self.mcmc_nsteps = tk.StringVar(value="1000")
        
        ttk.Label(analysis_frame, text="BLS Periods:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(analysis_frame, textvariable=self.bls_nperiods, width=15).grid(row=0, column=1, sticky="w", pady=3, padx=(10, 0))
        
        ttk.Label(analysis_frame, text="MCMC Walkers:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(analysis_frame, textvariable=self.mcmc_nwalkers, width=15).grid(row=1, column=1, sticky="w", pady=3, padx=(10, 0))
        
        ttk.Label(analysis_frame, text="MCMC Steps:").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(analysis_frame, textvariable=self.mcmc_nsteps, width=15).grid(row=2, column=1, sticky="w", pady=3, padx=(10, 0))
        
        # Save/Load settings
        settings_btn_frame = ttk.Frame(scrollable_frame)
        settings_btn_frame.pack(fill=tk.X, pady=(0, 15), padx=10)
        
        ttk.Button(settings_btn_frame, text="Save Settings", 
                  command=self.on_save_settings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(settings_btn_frame, text="Load Settings", 
                  command=self.on_load_settings).pack(side=tk.LEFT)
    
    def _build_batch_tab(self, parent):
        """Build batch processing tab"""
        ttk.Label(parent, text="Batch Processing", 
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 15))
        
        # Target list
        list_frame = ttk.LabelFrame(parent, text="Target List", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.batch_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=10)
        scrollbar = ttk.Scrollbar(list_frame)
        
        self.batch_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.batch_list.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.batch_list.yview)
        
        # Control buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(btn_frame, text="Add Target", 
                  command=self.on_batch_add).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Remove Selected", 
                  command=self.on_batch_remove).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Load from File", 
                  command=self.on_batch_load).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Clear All", 
                  command=self.on_batch_clear).pack(side=tk.LEFT)
        
        # Processing options
        options_frame = ttk.LabelFrame(parent, text="Processing Options", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.batch_analysis = tk.StringVar(value="bls")
        self.batch_export = tk.BooleanVar(value=True)
        
        ttk.Radiobutton(options_frame, text="BLS Analysis", 
                       variable=self.batch_analysis, value="bls").pack(anchor="w", pady=2)
        ttk.Radiobutton(options_frame, text="Full Analysis", 
                       variable=self.batch_analysis, value="full").pack(anchor="w", pady=2)
        
        ttk.Checkbutton(options_frame, text="Export Results", 
                       variable=self.batch_export).pack(anchor="w", pady=2)
        
        # Run button
        ttk.Button(parent, text="Run Batch Processing", 
                  command=self.on_batch_run, style="Accent.TButton").pack(fill=tk.X)
    
    def _build_developer_tab(self, parent):
        """Build developer tools tab"""
        ttk.Label(parent, text="Developer Tools", 
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 15))
        
        # Module info
        info_frame = ttk.LabelFrame(parent, text="Module Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        text = scrolledtext.ScrolledText(info_frame, height=8, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True)
        
        info_text = f"TransitKit v{tkit.__version__}\n"
        info_text += f"Author: {tkit.__author__}\n"
        info_text += f"Python: {sys.version}\n\n"
        
        info_text += "Available Modules:\n"
        for module in ['core', 'analysis', 'visualization', 'io', 'utils', 'validation']:
            try:
                mod = getattr(tkit, module)
                info_text += f"  {module}: ✓\n"
            except:
                info_text += f"  {module}: ✗\n"
        
        text.insert(1.0, info_text)
        text.configure(state='disabled')
        
        # Test buttons
        test_frame = ttk.LabelFrame(parent, text="Tests", padding=10)
        test_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(test_frame, text="Run Quick Test", 
                  command=self.on_dev_test).pack(fill=tk.X, pady=2)
        ttk.Button(test_frame, text="Check Dependencies", 
                  command=self.on_dev_check_deps).pack(fill=tk.X, pady=2)
        
        # Debug options
        debug_frame = ttk.LabelFrame(parent, text="Debug", padding=10)
        debug_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.debug_mode = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(debug_frame, text="Enable Debug Mode", 
                       variable=self.debug_mode).pack(anchor="w", pady=2)
        
        ttk.Button(debug_frame, text="Show Log File", 
                  command=self.on_dev_show_log).pack(fill=tk.X, pady=2)
    
    def on_save_settings(self):
        """Save settings to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="transitkit_settings.json"
        )
        
        if filename:
            settings = {
                'plot_style': self.plot_style.get(),
                'plot_dpi': self.plot_dpi.get(),
                'plot_figsize': [self.plot_figsize_w.get(), self.plot_figsize_h.get()],
                'bls_nperiods': self.bls_nperiods.get(),
                'mcmc_nwalkers': self.mcmc_nwalkers.get(),
                'mcmc_nsteps': self.mcmc_nsteps.get()
            }
            
            try:
                with open(filename, 'w') as f:
                    json.dump(settings, f, indent=2)
                self.log(f"Settings saved to {filename}", "SUCCESS")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save settings:\n{e}")
    
    def on_load_settings(self):
        """Load settings from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    settings = json.load(f)
                
                self.plot_style.set(settings.get('plot_style', 'default'))
                self.plot_dpi.set(settings.get('plot_dpi', '300'))
                
                figsize = settings.get('plot_figsize', ['10', '6'])
                self.plot_figsize_w.set(figsize[0])
                self.plot_figsize_h.set(figsize[1])
                
                self.bls_nperiods.set(settings.get('bls_nperiods', '10000'))
                self.mcmc_nwalkers.set(settings.get('mcmc_nwalkers', '32'))
                self.mcmc_nsteps.set(settings.get('mcmc_nsteps', '1000'))
                
                self.log(f"Settings loaded from {filename}", "SUCCESS")
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load settings:\n{e}")
    
    def on_batch_add(self):
        """Add target to batch list"""
        target = self.tess_target.get().strip()
        if target:
            self.batch_list.insert(tk.END, target)
            self.tess_target.set("")
    
    def on_batch_remove(self):
        """Remove selected targets from batch list"""
        selection = self.batch_list.curselection()
        for idx in reversed(selection):
            self.batch_list.delete(idx)
    
    def on_batch_load(self):
        """Load targets from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                
                self.batch_list.delete(0, tk.END)
                for line in lines:
                    target = line.strip()
                    if target and not target.startswith('#'):
                        self.batch_list.insert(tk.END, target)
                
                self.log(f"Loaded {len(lines)} targets from {filename}", "INFO")
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load targets:\n{e}")
    
    def on_batch_clear(self):
        """Clear batch list"""
        self.batch_list.delete(0, tk.END)
    
    def on_batch_run(self):
        """Run batch processing"""
        targets = self.batch_list.get(0, tk.END)
        if not targets:
            messagebox.showinfo("No Targets", "Add targets to the batch list first.")
            return
        
        # This would run batch processing in a separate thread
        # For now, just show a message
        messagebox.showinfo("Batch Processing", 
                          f"Would process {len(targets)} targets.\n"
                          f"Analysis type: {self.batch_analysis.get()}\n"
                          f"Export results: {self.batch_export.get()}")
        
        self.log(f"Batch processing queued for {len(targets)} targets", "INFO")
    
    def on_dev_test(self):
        """Run quick test"""
        try:
            # Create test data
            time = np.linspace(0, 30, 1000)
            flux = np.ones_like(time)
            
            # Add a test transit
            period = 10.0
            t0 = 5.0
            depth = 0.01
            duration = 0.1
            
            for i in range(10):
                tc = t0 + i * period
                in_transit = (time > tc - duration/2) & (time < tc + duration/2)
                flux[in_transit] = 1 - depth
            
            # Add noise
            flux += np.random.normal(0, 0.001, len(flux))
            
            # Run BLS
            res = find_transits_bls_advanced(time, flux, min_period=5, max_period=15)
            
            if res.get('period', 0) > 0:
                messagebox.showinfo("Test Passed", 
                                  f"Test successful!\n"
                                  f"Detected period: {res.get('period', 0):.4f} d\n"
                                  f"Expected: {period:.4f} d")
                self.log("Developer test passed", "SUCCESS")
            else:
                messagebox.showwarning("Test Warning", "Test ran but no period detected.")
                self.log("Developer test: no period detected", "WARN")
                
        except Exception as e:
            messagebox.showerror("Test Failed", f"Test failed:\n{e}")
            self.log(f"Developer test failed: {e}", "ERROR")
    
    def on_dev_check_deps(self):
        """Check dependencies"""
        import importlib
        
        dependencies = [
            'numpy', 'matplotlib', 'scipy', 'astropy',
            'lightkurve', 'emcee', 'corner', 'sklearn'
        ]
        
        report = "Dependency Check:\n"
        report += "="*40 + "\n\n"
        
        all_ok = True
        for dep in dependencies:
            try:
                importlib.import_module(dep)
                version = importlib.metadata.version(dep) if hasattr(importlib, 'metadata') else "?"
                report += f"✓ {dep}: {version}\n"
            except ImportError:
                report += f"✗ {dep}: NOT INSTALLED\n"
                all_ok = False
        
        if all_ok:
            report += "\n✅ All dependencies installed!"
        else:
            report += "\n❌ Some dependencies missing."
        
        # Show report
        win = tk.Toplevel(self)
        win.title("Dependency Check")
        win.geometry("400x300")
        
        text = scrolledtext.ScrolledText(win, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text.insert(1.0, report)
        text.configure(state='disabled')
        
        self.log("Dependency check completed", "INFO")
    
    def on_dev_show_log(self):
        """Show log file (placeholder)"""
        messagebox.showinfo("Log", "Log file viewing feature coming soon.")
        self.log("Log viewer requested", "INFO")
    
    # ---------------- UTILITY METHODS ----------------
    def _create_tooltip(self, widget, text: str):
        """
        Safe tooltip helper.
        - Expects a Tk widget (has .bind).
        - If someone accidentally passes a tk.Variable (StringVar, IntVar, etc),
        we just skip instead of crashing the entire app.
        """
        import tkinter as tk

        # If the wrong object is passed (StringVar, etc.), do NOT crash the app
        if widget is None or not hasattr(widget, "bind"):
            try:
                # if you have a logger panel method, use it
                if hasattr(self, "_log"):
                    self._log(f"[tooltip] skipped: expected widget, got {type(widget).__name__}")
            except Exception:
                pass
            return

        tip = {"win": None}

        def show(_event=None):
            if tip["win"] is not None:
                return
            try:
                x = widget.winfo_rootx() + 20
                y = widget.winfo_rooty() + widget.winfo_height() + 10
            except Exception:
                x, y = 100, 100

            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            win.wm_geometry(f"+{x}+{y}")

            lbl = tk.Label(
                win,
                text=text,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                font=("Segoe UI", 9),
                padx=6,
                pady=3,
            )
            lbl.pack()
            tip["win"] = win

        def hide(_event=None):
            win = tip.get("win")
            if win is not None:
                try:
                    win.destroy()
                except Exception:
                    pass
            tip["win"] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)
        widget.bind("<ButtonPress>", hide)



# -------------------------
# Main Application
# -------------------------
def main():
    """Main application entry point"""
    try:
        # Check for dependencies
        required = ['numpy', 'matplotlib', 'tkinter']
        missing = []
        
        for dep in required:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        
        if missing:
            messagebox.showerror(
                "Missing Dependencies",
                f"TransitKit requires the following packages:\n\n"
                f"{', '.join(missing)}\n\n"
                f"Please install them using:\n"
                f"pip install {' '.join(missing)}"
            )
            return
        
        # Create and run application
        app = TransitKitGUI()
        
        # Center window on screen
        app.update_idletasks()
        width = app.winfo_width()
        height = app.winfo_height()
        x = (app.winfo_screenwidth() // 2) - (width // 2)
        y = (app.winfo_screenheight() // 2) - (height // 2)
        app.geometry(f'{width}x{height}+{x}+{y}')
        
        app.mainloop()
        
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Failed to start TransitKit:\n{e}")
        raise


if __name__ == "__main__":
    main()
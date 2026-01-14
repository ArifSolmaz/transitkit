"""
TransitKit GUI (Tkinter)

Tabs:
- Simulate: build a synthetic transit, plot, run BLS
- TESS: search available sectors/cadences, download selected, stitch, plot, run BLS, export CSV

Requires:
- core: numpy, matplotlib, astropy (already)
- TESS tab: lightkurve (optional: pip install "transitkit[tess]")
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

import transitkit as tkit

# Matplotlib embedding for Tkinter
import matplotlib
matplotlib.use("TkAgg")  # ensures popup figures/canvas work on Windows

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def _safe_float(s: str, default: float) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _safe_int(s: str, default: int) -> int:
    try:
        return int(s)
    except Exception:
        return default


class PlotPanel(ttk.Frame):
    def __init__(self, parent, title=""):
        super().__init__(parent)
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)

    def clear(self):
        self.ax.clear()
        self.canvas.draw()

    def plot_xy(self, x, y, xlabel="", ylabel="", title="", style="k.", alpha=0.6, ms=2):
        self.ax.clear()
        self.ax.plot(x, y, style, alpha=alpha, markersize=ms)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_line(self, x, y, xlabel="", ylabel="", title="", lw=2):
        self.ax.clear()
        self.ax.plot(x, y, linewidth=lw)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def vline(self, x, color="r", ls="--", alpha=0.4, lw=1, label=None):
        self.ax.axvline(x=x, color=color, linestyle=ls, alpha=alpha, linewidth=lw, label=label)


class TransitKitGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TransitKit")
        self.geometry("1100x700")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.sim_tab = ttk.Frame(self.nb)
        self.tess_tab = ttk.Frame(self.nb)

        self.nb.add(self.sim_tab, text="Simulate")
        self.nb.add(self.tess_tab, text="TESS Explorer")

        # State buffers
        self.sim_time = None
        self.sim_flux = None
        self.tess_lc = None  # stitched LightCurve (if available)
        self.tess_time = None
        self.tess_flux = None

        self._build_simulate_tab()
        self._build_tess_tab()

    # ---------------------------
    # Simulate tab
    # ---------------------------
    def _build_simulate_tab(self):
        left = ttk.Frame(self.sim_tab, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(self.sim_tab, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inputs
        ttk.Label(left, text="Synthetic System", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        self.sim_period = tk.StringVar(value="5.0")
        self.sim_depth = tk.StringVar(value="0.02")
        self.sim_duration = tk.StringVar(value="0.15")
        self.sim_baseline = tk.StringVar(value="30.0")
        self.sim_npoints = tk.StringVar(value="3000")
        self.sim_noise = tk.StringVar(value="0.001")
        self.sim_seed = tk.StringVar(value="42")

        grid = ttk.Frame(left)
        grid.pack(fill=tk.X)

        def row(label, var, r):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(grid, textvariable=var, width=12).grid(row=r, column=1, sticky="w", pady=2)

        row("Period (days)", self.sim_period, 0)
        row("Depth (frac)", self.sim_depth, 1)
        row("Duration (days)", self.sim_duration, 2)
        row("Baseline (days)", self.sim_baseline, 3)
        row("N points", self.sim_npoints, 4)
        row("Noise sigma", self.sim_noise, 5)
        row("RNG seed", self.sim_seed, 6)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="BLS Search", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))
        self.sim_minp = tk.StringVar(value="1.0")
        self.sim_maxp = tk.StringVar(value="20.0")
        grid2 = ttk.Frame(left)
        grid2.pack(fill=tk.X)

        row2 = lambda label, var, r: (
            ttk.Label(grid2, text=label).grid(row=r, column=0, sticky="w", pady=2),
            ttk.Entry(grid2, textvariable=var, width=12).grid(row=r, column=1, sticky="w", pady=2),
        )
        row2("Min P (days)", self.sim_minp, 0)
        row2("Max P (days)", self.sim_maxp, 1)

        # Buttons
        ttk.Button(left, text="Simulate & Plot", command=self.on_simulate).pack(fill=tk.X, pady=(10, 4))
        ttk.Button(left, text="Run BLS", command=self.on_sim_bls).pack(fill=tk.X, pady=4)

        self.sim_status = ttk.Label(left, text="Ready.", wraplength=260)
        self.sim_status.pack(fill=tk.X, pady=(10, 0))

        # Plot panel
        self.sim_plot = PlotPanel(right, title="Synthetic Light Curve")
        self.sim_plot.pack(fill=tk.BOTH, expand=True)

    def on_simulate(self):
        P = _safe_float(self.sim_period.get(), 5.0)
        depth = _safe_float(self.sim_depth.get(), 0.02)
        dur = _safe_float(self.sim_duration.get(), 0.15)
        baseline = _safe_float(self.sim_baseline.get(), 30.0)
        n = _safe_int(self.sim_npoints.get(), 3000)
        sigma = _safe_float(self.sim_noise.get(), 0.001)
        seed = _safe_int(self.sim_seed.get(), 42)

        if n < 50:
            messagebox.showerror("Invalid input", "N points must be >= 50.")
            return
        if P <= 0 or dur <= 0 or baseline <= 0:
            messagebox.showerror("Invalid input", "Period, duration, baseline must be positive.")
            return

        rng = np.random.default_rng(seed)
        time = np.linspace(0.0, baseline, n)
        clean = tkit.generate_transit_signal(time, period=P, depth=depth, duration=dur)
        noisy = clean + rng.normal(0.0, sigma, size=n)

        self.sim_time = time
        self.sim_flux = noisy

        self.sim_plot.plot_xy(time, noisy, xlabel="Time (days)", ylabel="Normalized Flux",
                              title="Synthetic Transit Light Curve (Noisy)",
                              style="k.", alpha=0.6, ms=2)

        # mark expected centers (for reference only)
        t0 = P / 2.0
        ntr = int(time[-1] / P) + 1
        for i in range(ntr):
            self.sim_plot.vline(t0 + i * P, color="r", alpha=0.25)

        self.sim_plot.ax.legend_.remove() if self.sim_plot.ax.legend_ else None
        self.sim_plot.canvas.draw()

        self.sim_status.config(text=f"Simulated. P={P:.4f} d, depth={depth:.4f}, duration={dur:.4f} d, noise={sigma:.4f}")

    def on_sim_bls(self):
        if self.sim_time is None or self.sim_flux is None:
            messagebox.showinfo("No data", "Click 'Simulate & Plot' first.")
            return

        minP = _safe_float(self.sim_minp.get(), 1.0)
        maxP = _safe_float(self.sim_maxp.get(), 20.0)

        try:
            res = tkit.find_transits_box(self.sim_time, self.sim_flux, min_period=minP, max_period=maxP)
        except Exception as e:
            messagebox.showerror("BLS error", str(e))
            return

        bestP = res["period"]
        self.sim_status.config(text=f"BLS best period: {bestP:.6f} days")

        # Plot power/score
        periods = res.get("all_periods")
        y = res.get("all_power", None)
        ylabel = "BLS Power"
        if y is None:
            y = res.get("all_scores")
            ylabel = "Detection Score"

        if periods is None or y is None:
            messagebox.showerror("Missing outputs", "find_transits_box must return all_periods and all_power/all_scores.")
            return

        self.sim_plot.plot_line(periods, y, xlabel="Period (days)", ylabel=ylabel, title="Period Search")
        self.sim_plot.vline(bestP, color="g", alpha=0.6, label=f"Detected {bestP:.3f} d")
        self.sim_plot.ax.legend(loc="best")
        self.sim_plot.canvas.draw()

    # ---------------------------
    # TESS tab
    # ---------------------------
    def _build_tess_tab(self):
        left = ttk.Frame(self.tess_tab, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(self.tess_tab, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="TESS Explorer", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        self.tess_target = tk.StringVar(value="HAT-P-36")
        self.tess_author = tk.StringVar(value="SPOC")
        self.tess_cadence = tk.StringVar(value="Any")

        ttk.Label(left, text="Target (name or TIC):").pack(anchor="w")
        ttk.Entry(left, textvariable=self.tess_target, width=28).pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Author:").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.tess_author, values=["SPOC", "QLP", "Any"], width=25, state="readonly").pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Cadence:").pack(anchor="w")
        ttk.Combobox(
            left,
            textvariable=self.tess_cadence,
            values=["Any", "20-sec (20s)", "2-min (120s)", "10-min (600s)", "30-min (1800s)"],
            width=25,
            state="readonly",
        ).pack(anchor="w", pady=(0, 8))

        ttk.Button(left, text="Search", command=self.on_tess_search).pack(fill=tk.X, pady=(6, 4))
        ttk.Button(left, text="Download Selected", command=self.on_tess_download).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Run BLS on Stitched", command=self.on_tess_bls).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Export CSV", command=self.on_tess_export).pack(fill=tk.X, pady=4)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Available Light Curves:", font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self.tess_list = tk.Listbox(left, selectmode=tk.EXTENDED, width=45, height=18)
        self.tess_list.pack(fill=tk.BOTH, expand=False)

        self.tess_status = ttk.Label(left, text="TESS tab requires lightkurve. Install: pip install -e \"./.[tess]\"",
                                     wraplength=300)
        self.tess_status.pack(fill=tk.X, pady=(10, 0))

        # Plot panel
        self.tess_plot = PlotPanel(right, title="TESS Light Curve")
        self.tess_plot.pack(fill=tk.BOTH, expand=True)

        # Internal search result storage
        self._sr = None
        self._sr_filtered = None

    def _require_lightkurve(self):
        try:
            import lightkurve as lk  # noqa: F401
            return True
        except Exception:
            messagebox.showerror(
                "Missing dependency",
                "This feature requires lightkurve.\n\nInstall:\n  python -m pip install -e \".[tess]\""
            )
            return False

    def _cadence_seconds(self):
        s = self.tess_cadence.get()
        if s == "20-sec (20s)":
            return 20
        if s == "2-min (120s)":
            return 120
        if s == "10-min (600s)":
            return 600
        if s == "30-min (1800s)":
            return 1800
        return None  # Any

    def on_tess_search(self):
        if not self._require_lightkurve():
            return

        import lightkurve as lk

        target = self.tess_target.get().strip()
        author = self.tess_author.get()
        cadence = self._cadence_seconds()

        if not target:
            messagebox.showerror("Invalid input", "Target cannot be empty.")
            return

        self.tess_list.delete(0, tk.END)
        self.tess_status.config(text="Searching...")

        def work():
            try:
                kw = {}
                if author != "Any":
                    kw["author"] = author
                sr = lk.search_lightcurve(target, mission="TESS", **kw)

                # Filter by cadence (exptime seconds), if requested
                if cadence is not None and len(sr) > 0:
                    # sr.table has column 'exptime' in seconds in most cases
                    tbl = sr.table
                    if "exptime" in tbl.colnames:
                        mask = np.array(tbl["exptime"]) == cadence
                        sr_f = sr[mask]
                    else:
                        sr_f = sr
                else:
                    sr_f = sr

                self._sr = sr
                self._sr_filtered = sr_f

                # Populate list
                for i, row in enumerate(sr_f.table):
                    sector = row["sequence_number"] if "sequence_number" in row.colnames else row.get("sector", "NA")
                    exptime = row["exptime"] if "exptime" in row.colnames else "NA"
                    auth = row["author"] if "author" in row.colnames else author
                    label = f"[{i:02d}] Sector {sector} | exptime={exptime}s | author={auth}"
                    self.tess_list.insert(tk.END, label)

                msg = f"Found {len(sr_f)} light curve(s). Select one or more and click Download."
                self.after(0, lambda: self.tess_status.config(text=msg))

            except Exception as e:
                self.after(0, lambda: self.tess_status.config(text="Search failed."))
                self.after(0, lambda: messagebox.showerror("Search error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def on_tess_download(self):
        if not self._require_lightkurve():
            return
        if self._sr_filtered is None or len(self._sr_filtered) == 0:
            messagebox.showinfo("No results", "Search first, then select items to download.")
            return

        import lightkurve as lk

        idxs = list(self.tess_list.curselection())
        if not idxs:
            messagebox.showinfo("No selection", "Select one or more light curves to download.")
            return

        self.tess_status.config(text="Downloading selected light curves...")

        def work():
            try:
                sr_sel = self._sr_filtered[idxs]
                lcc = sr_sel.download_all()  # LightCurveCollection
                if lcc is None or len(lcc) == 0:
                    raise RuntimeError("Download returned no light curves.")

                # Stitch and simple cleaning
                lc = lcc.stitch()

                # Drop NaNs
                lc = lc.remove_nans()

                # Save for analysis/export
                self.tess_lc = lc
                self.tess_time = np.array(lc.time.value, dtype=float)
                self.tess_flux = np.array(lc.flux.value, dtype=float)

                # Plot
                def plot_now():
                    self.tess_plot.plot_xy(
                        self.tess_time,
                        self.tess_flux,
                        xlabel="Time (days)",
                        ylabel="Flux",
                        title="TESS Stitched Light Curve",
                        style="k.",
                        alpha=0.6,
                        ms=2,
                    )
                    self.tess_status.config(text=f"Downloaded & stitched {len(idxs)} sector(s). Ready for BLS/export.")

                self.after(0, plot_now)

            except Exception as e:
                self.after(0, lambda: self.tess_status.config(text="Download failed."))
                self.after(0, lambda: messagebox.showerror("Download error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def on_tess_bls(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download and stitch a light curve first.")
            return

        # Sensible defaults for transit-like search on stitched data
        minP, maxP = 0.5, 20.0

        try:
            res = tkit.find_transits_box(self.tess_time, self.tess_flux, min_period=minP, max_period=maxP)
        except Exception as e:
            messagebox.showerror("BLS error", str(e))
            return

        bestP = res["period"]
        periods = res.get("all_periods")
        y = res.get("all_power", None)
        ylabel = "BLS Power"
        if y is None:
            y = res.get("all_scores")
            ylabel = "Detection Score"

        if periods is None or y is None:
            messagebox.showerror("Missing outputs", "find_transits_box must return all_periods and all_power/all_scores.")
            return

        self.tess_plot.plot_line(periods, y, xlabel="Period (days)", ylabel=ylabel, title="TESS Period Search")
        self.tess_plot.vline(bestP, color="g", alpha=0.7, label=f"Detected {bestP:.3f} d")
        self.tess_plot.ax.legend(loc="best")
        self.tess_plot.canvas.draw()

        extra = []
        if "t0" in res:
            extra.append(f"t0={res['t0']:.4f}")
        if "duration" in res:
            extra.append(f"dur={res['duration']:.4f}")
        if "depth" in res:
            extra.append(f"depth={res['depth']:.5f}")

        msg = f"BLS best period: {bestP:.6f} d" + ((" | " + ", ".join(extra)) if extra else "")
        self.tess_status.config(text=msg)

    def on_tess_export(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download and stitch a light curve first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="tess_lightcurve.csv",
        )
        if not path:
            return

        # Export as: time, flux
        arr = np.column_stack([self.tess_time, self.tess_flux])
        header = "time_days,flux"

        np.savetxt(path, arr, delimiter=",", header=header, comments="")
        self.tess_status.config(text=f"Exported: {os.path.basename(path)}")


def main():
    app = TransitKitGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

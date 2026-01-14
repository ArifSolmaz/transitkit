"""
TransitKit GUI (Tkinter)

Tabs:
- Simulate: build a synthetic transit, plot, run BLS
- TESS Explorer: NEA lookup -> auto TIC -> list sectors/cadences -> download selected -> stitch -> plot -> BLS -> export

Requires:
- core: numpy, matplotlib, astropy (already in your deps)
- TESS tab: lightkurve (install with: pip install -e ".[tess]" if you add that extra)
- NEA lookup uses only stdlib (nea.py)
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import transitkit as tkit

# NASA Exoplanet Archive helper
from transitkit.nea import lookup_planet

# Matplotlib embedding for Tkinter
import matplotlib
matplotlib.use("TkAgg")  # good default for Windows desktop popups

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


def choose_nea_row(parent, rows: list[dict]) -> dict | None:
    """If multiple NEA matches exist, let user pick one."""
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]

    win = tk.Toplevel(parent)
    win.title("Select planet (NASA Exoplanet Archive)")
    win.geometry("860x320")

    ttk.Label(win, text="Multiple matches found. Select one:", padding=(10, 8)).pack(anchor="w")

    lb = tk.Listbox(win, selectmode=tk.SINGLE, width=140, height=10)
    lb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    for i, r in enumerate(rows):
        pl = r.get("pl_name")
        host = r.get("hostname")
        tic = r.get("tic_id")
        per = r.get("pl_orbper")
        dur = r.get("pl_trandur")
        lb.insert(tk.END, f"[{i:02d}] {pl} | host={host} | TIC={tic} | P={per} d | dur={dur} hr")

    choice = {"row": None}

    def ok():
        sel = lb.curselection()
        if sel:
            choice["row"] = rows[int(sel[0])]
        win.destroy()

    def cancel():
        win.destroy()

    btns = ttk.Frame(win)
    btns.pack(fill=tk.X, padx=10, pady=(0, 10))
    ttk.Button(btns, text="Use selected", command=ok).pack(side=tk.LEFT)
    ttk.Button(btns, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=(8, 0))

    win.transient(parent)
    win.grab_set()
    parent.wait_window(win)
    return choice["row"]


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
        self.geometry("1120x720")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.sim_tab = ttk.Frame(self.nb)
        self.tess_tab = ttk.Frame(self.nb)

        self.nb.add(self.sim_tab, text="Simulate")
        self.nb.add(self.tess_tab, text="TESS Explorer")

        # buffers
        self.sim_time = None
        self.sim_flux = None

        self.tess_lc = None
        self.tess_time = None
        self.tess_flux = None

        self._sr = None
        self._sr_filtered = None

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

        ttk.Label(grid2, text="Min P (days)").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(grid2, textvariable=self.sim_minp, width=12).grid(row=0, column=1, sticky="w", pady=2)
        ttk.Label(grid2, text="Max P (days)").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(grid2, textvariable=self.sim_maxp, width=12).grid(row=1, column=1, sticky="w", pady=2)

        ttk.Button(left, text="Simulate & Plot", command=self.on_simulate).pack(fill=tk.X, pady=(10, 4))
        ttk.Button(left, text="Run BLS", command=self.on_sim_bls).pack(fill=tk.X, pady=4)

        self.sim_status = ttk.Label(left, text="Ready.", wraplength=280)
        self.sim_status.pack(fill=tk.X, pady=(10, 0))

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

        # reference transit centers consistent with your generator (t0=P/2)
        t0 = P / 2.0
        ntr = int(time[-1] / P) + 1
        for i in range(ntr):
            self.sim_plot.vline(t0 + i * P, color="r", alpha=0.25)

        self.sim_status.config(text=f"Simulated. P={P:.6f} d | depth={depth:.5f} | dur={dur:.4f} d | noise={sigma:.4f}")

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
        self.sim_plot.vline(bestP, color="g", alpha=0.7, label=f"Detected {bestP:.3f} d")
        self.sim_plot.ax.legend(loc="best")
        self.sim_plot.canvas.draw()

        self.sim_status.config(text=f"BLS best period: {bestP:.8f} days")

    # ---------------------------
    # TESS tab
    # ---------------------------
    def _build_tess_tab(self):
        left = ttk.Frame(self.tess_tab, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(self.tess_tab, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="TESS Explorer", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        self.tess_target = tk.StringVar(value="HAT-P-36 b")
        self.tess_author = tk.StringVar(value="SPOC")
        self.tess_cadence = tk.StringVar(value="Any")

        ttk.Label(left, text="Planet name / Host / TIC:").pack(anchor="w")
        ttk.Entry(left, textvariable=self.tess_target, width=34).pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Author:").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.tess_author, values=["SPOC", "QLP", "Any"], width=31, state="readonly").pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Cadence:").pack(anchor="w")
        ttk.Combobox(
            left,
            textvariable=self.tess_cadence,
            values=["Any", "20-sec (20s)", "2-min (120s)", "10-min (600s)", "30-min (1800s)"],
            width=31,
            state="readonly",
        ).pack(anchor="w", pady=(0, 8))

        # Buttons (includes NEA fetch)
        ttk.Button(left, text="Fetch NEA Params", command=self.on_nea_fetch).pack(fill=tk.X, pady=(6, 4))
        ttk.Button(left, text="Search TESS", command=self.on_tess_search).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Download Selected", command=self.on_tess_download).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Run BLS on Stitched", command=self.on_tess_bls).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Export CSV", command=self.on_tess_export).pack(fill=tk.X, pady=4)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Available Light Curves:", font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self.tess_list = tk.Listbox(left, selectmode=tk.EXTENDED, width=52, height=18)
        self.tess_list.pack(fill=tk.BOTH, expand=False)

        self.tess_status = ttk.Label(
            left,
            text='Tip: install Lightkurve for TESS:  python -m pip install -e ".[tess]"',
            wraplength=360
        )
        self.tess_status.pack(fill=tk.X, pady=(10, 0))

        self.tess_plot = PlotPanel(right, title="TESS Light Curve")
        self.tess_plot.pack(fill=tk.BOTH, expand=True)

    def _require_lightkurve(self) -> bool:
        try:
            import lightkurve as lk  # noqa: F401
            return True
        except Exception:
            messagebox.showerror(
                "Missing dependency",
                'This feature requires lightkurve.\n\nInstall:\n  python -m pip install -e ".[tess]"'
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

    # -------- NEA fetch --------
    def on_nea_fetch(self):
        q = self.tess_target.get().strip()
        if not q:
            messagebox.showerror("Invalid input", "Enter a planet name (e.g., HAT-P-36 b) or hostname.")
            return

        self.tess_status.config(text="Querying NASA Exoplanet Archive (NEA)...")

        def work():
            try:
                rows = lookup_planet(q, default_only=True, limit=25)
                if not rows:
                    rows = lookup_planet(q, default_only=False, limit=25)

                def apply():
                    if not rows:
                        self.tess_status.config(text="NEA: no matches found.")
                        return

                    row = choose_nea_row(self, rows)
                    if not row:
                        self.tess_status.config(text="NEA: selection cancelled.")
                        return

                    pl = row.get("pl_name")
                    host = row.get("hostname")
                    tic = row.get("tic_id")
                    per = row.get("pl_orbper")          # days
                    dur_hr = row.get("pl_trandur")      # hours
                    trandep = row.get("pl_trandep")     # percent
                    ratror = row.get("pl_ratror")       # Rp/Rs

                    msg = f"NEA: {pl} | host={host} | TIC={tic} | P={per} d | dur={dur_hr} hr"
                    self.tess_status.config(text=msg)

                    # Auto-fill simulate tab (nice convenience)
                    try:
                        if per is not None:
                            self.sim_period.set(str(float(per)))
                        if dur_hr is not None:
                            self.sim_duration.set(str(float(dur_hr) / 24.0))
                        if trandep is not None:
                            self.sim_depth.set(str(float(trandep) / 100.0))
                        elif ratror is not None:
                            self.sim_depth.set(str(float(ratror) ** 2))
                    except Exception:
                        pass

                    # Auto-search TESS by TIC if available (usually best)
                    if tic not in (None, "", "null"):
                        self.tess_target.set(f"TIC {tic}")
                        self.on_tess_search()
                    else:
                        # If no TIC, user can still click Search TESS on planet/host text
                        pass

                self.after(0, apply)

            except Exception as e:
                self.after(0, lambda: self.tess_status.config(text="NEA query failed."))
                self.after(0, lambda: messagebox.showerror("NEA error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    # -------- TESS search/download/analyze --------
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
        self.tess_status.config(text="Searching TESS light curves...")

        def work():
            try:
                kw = {}
                if author != "Any":
                    kw["author"] = author

                sr = lk.search_lightcurve(target, mission="TESS", **kw)

                # Filter by cadence if exptime is present
                if cadence is not None and len(sr) > 0:
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

                for i, row in enumerate(sr_f.table):
                    # Lightkurve commonly uses 'sequence_number' for sector
                    sector = row["sequence_number"] if "sequence_number" in row.colnames else row.get("sector", "NA")
                    exptime = row["exptime"] if "exptime" in row.colnames else "NA"
                    auth = row["author"] if "author" in row.colnames else author
                    prod = row["productFilename"] if "productFilename" in row.colnames else ""
                    label = f"[{i:02d}] Sector {sector} | exptime={exptime}s | author={auth} | {prod}"
                    self.tess_list.insert(tk.END, label)

                msg = f"Found {len(sr_f)} light curve(s). Select one or more, then Download."
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

        idxs = list(self.tess_list.curselection())
        if not idxs:
            messagebox.showinfo("No selection", "Select one or more light curves to download.")
            return

        self.tess_status.config(text="Downloading selected light curves...")

        def work():
            try:
                sr_sel = self._sr_filtered[idxs]
                lcc = sr_sel.download_all()
                if lcc is None or len(lcc) == 0:
                    raise RuntimeError("Download returned no light curves.")

                lc = lcc.stitch().remove_nans()

                self.tess_lc = lc
                self.tess_time = np.array(lc.time.value, dtype=float)
                self.tess_flux = np.array(lc.flux.value, dtype=float)

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
                    self.tess_status.config(text=f"Downloaded & stitched {len(idxs)} item(s). Ready for BLS/export.")

                self.after(0, plot_now)

            except Exception as e:
                self.after(0, lambda: self.tess_status.config(text="Download failed."))
                self.after(0, lambda: messagebox.showerror("Download error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def on_tess_bls(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download and stitch a light curve first.")
            return

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

        arr = np.column_stack([self.tess_time, self.tess_flux])
        np.savetxt(path, arr, delimiter=",", header="time_days,flux", comments="")
        self.tess_status.config(text=f"Exported: {os.path.basename(path)}")


def main():
    app = TransitKitGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

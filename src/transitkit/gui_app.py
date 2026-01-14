"""
TransitKit GUI (Tkinter)

Restores:
- Simulate tab (generate synthetic system, plot, run BLS)
- TESS Explorer tab with: NEA fetch, search, download, BLS, markers, transit viewer,
  stacked transits, phase fold.

Fixes:
- 20s cadence filtering uses np.isclose (exptime often not exact int)
- avoids numpy "or" truth-value bug when picking arrays
- terminal-like log panel for progress and errors

Notes:
- Requires: numpy, matplotlib, astropy
- For TESS: lightkurve
- For NEA: transitkit.nea.lookup_planet
"""

from __future__ import annotations

import os
import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from datetime import datetime

import numpy as np
import transitkit as tkit
from transitkit.nea import lookup_planet

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# -------------------------
# Plot panel
# -------------------------
class PlotPanel(ttk.Frame):
    def __init__(self, parent, title=""):
        super().__init__(parent)
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.axes = [self.fig.add_subplot(111)]
        self.axes[0].set_title(title)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

    def set_subplots(self, nrows: int, sharey=True):
        self.fig.clf()
        if nrows <= 1:
            ax = self.fig.add_subplot(111)
            self.axes = [ax]
        else:
            axs = self.fig.subplots(nrows, 1, sharey=sharey)
            self.axes = list(axs) if isinstance(axs, (list, np.ndarray)) else [axs]
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_xy(self, x, y, xlabel="", ylabel="", title="", style="k.", alpha=0.6, ms=2, ax_index=0):
        ax = self.axes[ax_index]
        ax.clear()
        ax.plot(x, y, style, alpha=alpha, markersize=ms)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_line(self, x, y, xlabel="", ylabel="", title="", lw=2, ax_index=0):
        ax = self.axes[ax_index]
        ax.clear()
        ax.plot(x, y, linewidth=lw)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def vline(self, x, color="r", ls="--", alpha=0.35, lw=1, label=None, ax_index=0):
        ax = self.axes[ax_index]
        ax.axvline(x=x, color=color, linestyle=ls, alpha=alpha, linewidth=lw, label=label)


# -------------------------
# NEA selection helper
# -------------------------
def choose_nea_row(parent, rows: list[dict]) -> dict | None:
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]

    win = tk.Toplevel(parent)
    win.title("Select planet (NASA Exoplanet Archive)")
    win.geometry("920x340")

    ttk.Label(win, text="Multiple matches found. Select one:", padding=(10, 8)).pack(anchor="w")

    lb = tk.Listbox(win, selectmode=tk.SINGLE, width=150, height=10)
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


# -------------------------
# Small robust helpers
# -------------------------
def mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return np.median(np.abs(x - med))


def normalize_target(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return s
    s = re.sub(r"\s+", " ", s).strip()

    # if contains TIC, extract digits and force "TIC ####"
    if re.search(r"\bTIC\b", s, flags=re.IGNORECASE):
        digits = re.findall(r"\d+", s)
        if digits:
            return f"TIC {digits[0]}"
        return s

    return s


# -------------------------
# Transit utilities (viewer)
# -------------------------
def predicted_centers(time, period, t0):
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
# Main App
# -------------------------
class TransitKitGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TransitKit")
        self.geometry("1280x860")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.sim_tab = ttk.Frame(self.nb)
        self.tess_tab = ttk.Frame(self.nb)

        self.nb.add(self.sim_tab, text="Simulate")
        self.nb.add(self.tess_tab, text="TESS Explorer")

        # state
        self._busy_count = 0
        self._sr_filtered = None

        self.tess_segments = []
        self.tess_time = None
        self.tess_flux = None

        # ephemeris state (BTJD)
        self.ephem_period = None
        self.ephem_t0 = None
        self.ephem_duration = None

        # simulate state
        self.sim_time = None
        self.sim_flux = None

        self._build_sim_tab()
        self._build_tess_tab()

    # ---------------- logging / status ----------------
    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_status(self, msg: str):
        self.tess_status.config(text=msg)
        self.log(msg)

    def _set_busy(self, busy: bool):
        if busy:
            self._busy_count += 1
            if self._busy_count == 1:
                self.busy.start(10)
        else:
            self._busy_count = max(0, self._busy_count - 1)
            if self._busy_count == 0:
                self.busy.stop()

    def _require_lightkurve(self) -> bool:
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
        left = ttk.Frame(self.sim_tab, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(self.sim_tab, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Simulate", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        self.sim_period = tk.StringVar(value="5.0")
        self.sim_depth = tk.StringVar(value="0.02")
        self.sim_dur_hr = tk.StringVar(value="3.6")
        self.sim_noise = tk.StringVar(value="0.001")
        self.sim_baseline = tk.StringVar(value="30")
        self.sim_npts = tk.StringVar(value="3000")

        def row(lbl, var):
            r = ttk.Frame(left)
            r.pack(fill=tk.X, pady=3)
            ttk.Label(r, text=lbl, width=16).pack(side=tk.LEFT)
            ttk.Entry(r, textvariable=var, width=16).pack(side=tk.LEFT)

        row("Period (d):", self.sim_period)
        row("Depth:", self.sim_depth)
        row("Duration (hr):", self.sim_dur_hr)
        row("Noise σ:", self.sim_noise)
        row("Baseline (d):", self.sim_baseline)
        row("N points:", self.sim_npts)

        ttk.Button(left, text="Generate", command=self.on_sim_generate).pack(fill=tk.X, pady=(10, 4))
        ttk.Button(left, text="Run BLS", command=self.on_sim_bls).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Export CSV", command=self.on_sim_export).pack(fill=tk.X, pady=4)

        self.sim_plot = PlotPanel(right, title="Synthetic Light Curve")
        self.sim_plot.pack(fill=tk.BOTH, expand=True)

    def on_sim_generate(self):
        P = float(self.sim_period.get())
        depth = float(self.sim_depth.get())
        dur_hr = float(self.sim_dur_hr.get())
        noise = float(self.sim_noise.get())
        baseline = float(self.sim_baseline.get())
        npts = int(float(self.sim_npts.get()))

        time = np.linspace(0, baseline, npts)
        clean = tkit.generate_transit_signal(time, period=P, depth=depth, duration=dur_hr/24.0)
        flux = tkit.add_noise(clean, noise_level=noise)

        self.sim_time = time
        self.sim_flux = flux

        self.sim_plot.set_subplots(1)
        self.sim_plot.plot_xy(time, flux, xlabel="Time (days)", ylabel="Flux",
                              title="Synthetic Light Curve", style="k.", alpha=0.6, ms=2)

    def on_sim_bls(self):
        if self.sim_time is None or self.sim_flux is None:
            messagebox.showinfo("No data", "Click Generate first.")
            return

        res = tkit.find_transits_box(self.sim_time, self.sim_flux, min_period=0.5, max_period=20.0, n_periods=8000)

        periods = res.get("all_periods")
        y = res["all_power"] if ("all_power" in res and res["all_power"] is not None) else res.get("all_scores")
        bestP = float(res["period"])

        win = tk.Toplevel(self)
        win.title("Simulated BLS Periodogram")
        win.geometry("980x520")
        panel = PlotPanel(win, title="BLS")
        panel.pack(fill=tk.BOTH, expand=True)
        panel.plot_line(periods, y, xlabel="Period (days)", ylabel="Power", title=f"BLS best P={bestP:.6f} d")
        panel.vline(bestP, color="g", alpha=0.8)

    def on_sim_export(self):
        if self.sim_time is None or self.sim_flux is None:
            messagebox.showinfo("No data", "Click Generate first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="synthetic_lightcurve.csv",
        )
        if not path:
            return
        arr = np.column_stack([self.sim_time, self.sim_flux])
        np.savetxt(path, arr, delimiter=",", header="time,flux", comments="")
        messagebox.showinfo("Saved", os.path.basename(path))

    # ---------------- TESS TAB ----------------
    def _build_tess_tab(self):
        left = ttk.Frame(self.tess_tab, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(self.tess_tab, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="TESS Explorer", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        self.tess_target = tk.StringVar(value="")  # user enters
        self.tess_author = tk.StringVar(value="SPOC")
        self.tess_cadence = tk.StringVar(value="2-min (120s)")
        self.plot_mode = tk.StringVar(value="Per-sector panels")

        self.do_flatten = tk.BooleanVar(value=True)
        self.do_outliers = tk.BooleanVar(value=True)

        ttk.Label(left, text="Planet name / Host / TIC:").pack(anchor="w")
        ttk.Entry(left, textvariable=self.tess_target, width=36).pack(anchor="w", pady=(0, 4))
        ttk.Label(left, text="Example: HAT-P-36 b  or  HAT-P-36  or  TIC 373693175", foreground="#666").pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Author:").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.tess_author, values=["SPOC", "QLP", "Any"], width=33, state="readonly").pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Cadence:").pack(anchor="w")
        ttk.Combobox(
            left,
            textvariable=self.tess_cadence,
            values=["Any", "20-sec (20s)", "2-min (120s)", "10-min (600s)", "30-min (1800s)"],
            width=33,
            state="readonly",
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Plot mode:").pack(anchor="w")
        ttk.Combobox(
            left,
            textvariable=self.plot_mode,
            values=["Per-sector panels", "Stitched (absolute BTJD)", "Concatenated (no gaps)"],
            width=33,
            state="readonly",
        ).pack(anchor="w", pady=(0, 8))

        ttk.Checkbutton(left, text="Remove outliers", variable=self.do_outliers).pack(anchor="w")
        ttk.Checkbutton(left, text="Flatten/detrend", variable=self.do_flatten).pack(anchor="w", pady=(0, 8))

        ttk.Button(left, text="Fetch NEA Params", command=self.on_nea_fetch).pack(fill=tk.X, pady=(6, 4))
        ttk.Button(left, text="Search TESS", command=self.on_tess_search).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Download Selected (PDCSAP)", command=self.on_tess_download).pack(fill=tk.X, pady=4)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Button(left, text="Run BLS (narrowed)", command=self.on_tess_bls).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Show Transit Markers", command=self.on_show_markers).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Transit Viewer", command=self.on_transit_viewer).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Stacked Transits", command=self.on_stacked_transits).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Phase Fold", command=self.on_phase_fold).pack(fill=tk.X, pady=4)

        ttk.Button(left, text="Export CSV", command=self.on_tess_export).pack(fill=tk.X, pady=(8, 4))

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Available Light Curves:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.tess_list = tk.Listbox(left, selectmode=tk.EXTENDED, width=52, height=10)
        self.tess_list.pack(fill=tk.X, expand=False)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        self.tess_status = ttk.Label(left, text="Ready.", wraplength=380)
        self.tess_status.pack(fill=tk.X, pady=(0, 6))

        self.busy = ttk.Progressbar(left, mode="indeterminate")
        self.busy.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(left, text="Log:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.log_box = ScrolledText(left, width=52, height=14, state="disabled")
        self.log_box.pack(fill=tk.BOTH, expand=True)

        self.tess_plot = PlotPanel(right, title="TESS Light Curve")
        self.tess_plot.pack(fill=tk.BOTH, expand=True)

    def _cadence_seconds(self):
        s = self.tess_cadence.get()
        return {"20-sec (20s)": 20, "2-min (120s)": 120, "10-min (600s)": 600, "30-min (1800s)": 1800}.get(s, None)

    # ----- NEA -----
    def on_nea_fetch(self):
        raw = self.tess_target.get().strip()
        if not raw:
            messagebox.showerror("Invalid input", "Enter a planet name or TIC.")
            return

        q = raw
        self._set_busy(True)
        self.set_status(f"NEA: querying for '{q}' ...")

        def work():
            try:
                rows = lookup_planet(q, default_only=True, limit=25)
                if not rows:
                    rows = lookup_planet(q, default_only=False, limit=25)

                def apply():
                    try:
                        if not rows:
                            self.set_status("NEA: no matches found.")
                            return

                        row = choose_nea_row(self, rows)
                        if not row:
                            self.set_status("NEA: selection cancelled.")
                            return

                        pl = row.get("pl_name")
                        tic = row.get("tic_id")
                        per = row.get("pl_orbper")
                        dur_hr = row.get("pl_trandur")
                        tranmid_jd = row.get("pl_tranmid")

                        if per is not None:
                            self.ephem_period = float(per)
                        if dur_hr is not None:
                            self.ephem_duration = float(dur_hr) / 24.0
                        if tranmid_jd is not None:
                            self.ephem_t0 = float(tranmid_jd) - 2457000.0

                        self.set_status(f"NEA OK: {pl} | TIC={tic} | P={self.ephem_period} d | dur={dur_hr} hr")

                        if tic not in (None, "", "null"):
                            t = normalize_target(f"TIC {tic}")
                            self.tess_target.set(t)
                            self.set_status(f"Target set to: {t}")

                    finally:
                        self._set_busy(False)

                self.after(0, apply)

            except Exception as e:
                def fail():
                    self._set_busy(False)
                    self.set_status("NEA failed.")
                    messagebox.showerror("NEA error", str(e))
                self.after(0, fail)

        threading.Thread(target=work, daemon=True).start()

    # ----- Search -----
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
        self.set_status(f"Search: target='{target}', author={author}, cadence={cadence or 'Any'} ...")

        def work():
            try:
                kw = {}
                if author != "Any":
                    kw["author"] = author

                sr = lk.search_lightcurve(target, mission="TESS", **kw)

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
                        for i, row in enumerate(sr_f.table):
                            sector = row["sequence_number"] if "sequence_number" in row.colnames else row.get("sector", "NA")
                            exptime = row["exptime"] if "exptime" in row.colnames else "NA"
                            auth = row["author"] if "author" in row.colnames else author
                            self.tess_list.insert(tk.END, f"[{i:02d}] Sector {sector} | exptime={exptime}s | author={auth}")

                        if len(sr_f) == 0 and cadence == 20:
                            self.set_status("Search done: 0 results for 20s. Try 2-min or Any.")
                        else:
                            self.set_status(f"Search done: found {len(sr_f)} light curve(s). Select & Download.")
                    finally:
                        self._set_busy(False)

                self.after(0, apply)

            except Exception as e:
                def fail():
                    self._set_busy(False)
                    self.set_status("Search failed.")
                    messagebox.showerror("Search error", str(e))
                self.after(0, fail)

        threading.Thread(target=work, daemon=True).start()

    # ----- Download + preprocess -----
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

        self._set_busy(True)
        self.set_status(f"Download: fetching {len(idxs)} light curve(s) ...")

        def work():
            try:
                sr_sel = self._sr_filtered[idxs]
                lcc = sr_sel.download_all()
                if lcc is None or len(lcc) == 0:
                    raise RuntimeError("Download returned no light curves.")

                segs = []

                for lc in lcc:
                    sector = lc.meta.get("SECTOR", "NA") if hasattr(lc, "meta") else "NA"
                    lc2 = lc.remove_nans()

                    # Force PDCSAP if available
                    used = "FLUX"
                    try:
                        pd = lc2["PDCSAP_FLUX"]
                        lc2 = lc2.copy()
                        lc2.flux = pd
                        used = "PDCSAP"
                    except Exception:
                        pass

                    # Normalize
                    if hasattr(lc2, "normalize"):
                        lc2 = lc2.normalize()

                    n_before = len(lc2.time)

                    # Outliers
                    if self.do_outliers.get():
                        # safer than remove_outliers for transits: OOT-only if ephemeris exists
                        t = np.array(lc2.time.value, dtype=float)
                        f = np.array(lc2.flux.value, dtype=float)

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

                        med = np.nanmedian(f[oot])
                        sig = 1.4826 * mad(f[oot])
                        if np.isfinite(sig) and sig > 0:
                            keep = np.ones_like(f, dtype=bool)
                            keep[oot] = np.abs(f[oot] - med) < 8.0 * sig
                            lc2 = lc2[keep]

                    # Flatten
                    if self.do_flatten.get() and hasattr(lc2, "flatten"):
                        if self.ephem_period and self.ephem_t0 and self.ephem_duration:
                            t = np.array(lc2.time.value, dtype=float)
                            P = float(self.ephem_period)
                            t0 = float(self.ephem_t0)
                            dur = float(self.ephem_duration)
                            ph = ((t - t0) / P) % 1.0
                            half = 0.5 * dur / P
                            in_tr = (ph <= half) | (ph >= 1.0 - half)
                            lc2 = lc2.flatten(window_length=401, polyorder=2, break_tolerance=5, mask=~in_tr)
                        else:
                            # warning: can distort transits
                            self.after(0, lambda: self.log("WARN: Flatten without ephemeris may distort transits. Fetch NEA first."))
                            lc2 = lc2.flatten(window_length=401, polyorder=2, break_tolerance=5)

                    n_after = len(lc2.time)
                    self.after(0, lambda s=sector, u=used, a=n_after, b=n_before:
                               self.log(f"Sector {s}: flux={u}, points {b} -> {a}"))

                    t = np.array(lc2.time.value, dtype=float)
                    f = np.array(lc2.flux.value, dtype=float)
                    segs.append({"sector": sector, "time": t, "flux": f})

                # sort by sector
                def _sec_key(x):
                    try:
                        return int(x["sector"])
                    except Exception:
                        return 10**9
                segs = sorted(segs, key=_sec_key)

                # stitched arrays
                t_all = np.concatenate([s["time"] for s in segs])
                f_all = np.concatenate([s["flux"] for s in segs])
                o = np.argsort(t_all)
                self.tess_time = t_all[o]
                self.tess_flux = f_all[o]
                self.tess_segments = segs

                def apply():
                    try:
                        self._plot_segments()
                        self.set_status(f"Download done: {len(segs)} sector(s) plotted.")
                    finally:
                        self._set_busy(False)

                self.after(0, apply)

            except Exception as e:
                def fail():
                    self._set_busy(False)
                    self.set_status("Download failed.")
                    messagebox.showerror("Download error", str(e))
                self.after(0, fail)

        threading.Thread(target=work, daemon=True).start()

    def _plot_segments(self):
        if not self.tess_segments:
            return

        mode = self.plot_mode.get()

        if mode == "Per-sector panels" and len(self.tess_segments) > 1:
            self.tess_plot.set_subplots(len(self.tess_segments), sharey=True)
            for i, seg in enumerate(self.tess_segments):
                sec = seg["sector"]
                t = seg["time"]
                f = seg["flux"]
                ax = self.tess_plot.axes[i]
                ax.clear()
                ax.plot(t, f, "k.", alpha=0.6, markersize=2)
                ax.set_title(f"Sector {sec}")
                ax.set_ylabel("Flux")
                ax.grid(True, alpha=0.3)
                if i == len(self.tess_segments) - 1:
                    ax.set_xlabel("Time (BTJD days)")
            self.tess_plot.fig.tight_layout()
            self.tess_plot.canvas.draw()
            return

        if mode == "Concatenated (no gaps)" and len(self.tess_segments) > 1:
            self.tess_plot.set_subplots(1)
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
            self.tess_plot.plot_xy(x, y, xlabel="Concatenated time (days)", ylabel="Flux",
                                   title="TESS Concatenated Light Curve", style="k.", alpha=0.6, ms=2)
            return

        # stitched absolute
        self.tess_plot.set_subplots(1)
        if len(self.tess_segments) == 1:
            sec = self.tess_segments[0]["sector"]
            title = f"TESS Sector {sec} Light Curve"
        else:
            title = "TESS Stitched Light Curve (absolute BTJD)"
        self.tess_plot.plot_xy(self.tess_time, self.tess_flux, xlabel="Time (BTJD days)", ylabel="Flux",
                               title=title, style="k.", alpha=0.6, ms=2)

    # ----- BLS -----
    def on_tess_bls(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download light curves first.")
            return

        if self.ephem_period is not None:
            P0 = float(self.ephem_period)
            minP = max(0.2, P0 - 0.05)
            maxP = P0 + 0.05
            nper = 30000
        else:
            minP, maxP = 0.5, 20.0
            nper = 8000

        self._set_busy(True)
        self.set_status(f"BLS: running (minP={minP}, maxP={maxP}, n={nper}) ...")

        def work():
            try:
                res = tkit.find_transits_box(self.tess_time, self.tess_flux, min_period=minP, max_period=maxP, n_periods=nper)

                bestP = float(res["period"])
                self.ephem_period = bestP

                periods = res.get("all_periods")

                # IMPORTANT: avoid numpy `or`
                if "all_power" in res and res["all_power"] is not None:
                    y = res["all_power"]
                    ylabel = "BLS Power"
                else:
                    y = res.get("all_scores")
                    ylabel = "Score"

                def apply():
                    try:
                        self.tess_plot.set_subplots(1)
                        self.tess_plot.plot_line(periods, y, xlabel="Period (days)", ylabel=ylabel, title="BLS Period Search")
                        self.tess_plot.vline(bestP, color="g", alpha=0.8, label=f"Detected {bestP:.9f} d")
                        self.tess_plot.axes[0].legend(loc="best")
                        self.tess_plot.canvas.draw()
                        self.set_status(f"BLS done: best P={bestP:.9f} d. For transit tools, NEA params are still recommended.")
                    finally:
                        self._set_busy(False)

                self.after(0, apply)

            except Exception as e:
                def fail():
                    self._set_busy(False)
                    self.set_status("BLS failed.")
                    messagebox.showerror("BLS error", str(e))
                self.after(0, fail)

        threading.Thread(target=work, daemon=True).start()

    # ----- Transit markers / viewer / stacked / fold -----
    def on_show_markers(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download first.")
            return
        if not (self.ephem_period and self.ephem_t0):
            messagebox.showinfo("Need ephemeris", "Fetch NEA Params first (best) or run BLS.")
            return

        self._plot_segments()
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

    # ----- Export -----
    def on_tess_export(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download first.")
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
        np.savetxt(path, arr, delimiter=",", header="time_btjd,flux", comments="")
        self.set_status(f"Exported: {os.path.basename(path)}")


def main():
    app = TransitKitGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

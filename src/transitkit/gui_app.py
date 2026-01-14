"""
TransitKit GUI (Tkinter) – improved multi-sector plotting + transit events + better period.

New features:
- Multi-sector plot modes:
    * Per-sector panels (recommended when >1 sector)
    * Stitched (absolute BTJD, with gaps)
    * Concatenated (removes gaps so you "see it like one long sector")
- Transit event tools:
    * Show Transit Markers (vertical lines at predicted/fit mid-transits)
    * Transit Viewer (select individual events and zoom)
- Better period estimation:
    * Preprocess per sector: remove_nans -> remove_outliers -> normalize -> flatten
    * BLS search narrowed around NEA period (if available)
    * Refine ephemeris by measuring mid-transit times and fitting linear ephemeris

Requires:
- numpy, matplotlib, astropy (already)
- lightkurve for TESS tab
- transitkit.nea (NEA TAP client)
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import transitkit as tkit

from transitkit.nea import lookup_planet

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# -------------------------
# small helpers
# -------------------------
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
# Plot panel (supports 1 or N axes)
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

    def set_subplots(self, nrows: int):
        """Rebuild figure layout with nrows subplots."""
        self.fig.clf()
        if nrows <= 1:
            ax = self.fig.add_subplot(111)
            self.axes = [ax]
        else:
            axs = self.fig.subplots(nrows, 1, sharey=True)
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
# Transit mid-time measurement + ephemeris fit
# -------------------------
def _measure_midtime_box_grid(t, f, tc_pred, duration, grid_n=401):
    """
    Very robust mid-transit estimator:
    - scan a grid of trial centers around tc_pred
    - choose the center that minimizes in-transit median flux (max depth)
    - then refine using a local quadratic fit on score vs shift
    """
    # window for local evaluation
    w = 3.0 * duration
    m = (t >= tc_pred - w) & (t <= tc_pred + w)
    if np.sum(m) < 30:
        return None

    tt = t[m]
    ff = f[m]

    # normalize locally to reduce slow trends
    med = np.nanmedian(ff)
    if not np.isfinite(med) or med == 0:
        return None
    ff = ff / med

    # grid of center shifts
    shifts = np.linspace(-duration, duration, grid_n)
    scores = np.full_like(shifts, np.nan, dtype=float)

    half = 0.5 * duration
    for i, s in enumerate(shifts):
        tc = tc_pred + s
        in_tr = (tt >= tc - half) & (tt <= tc + half)
        if np.sum(in_tr) < 6:
            continue
        # score: lower in-transit median is better (deeper)
        scores[i] = np.nanmedian(ff[in_tr])

    if not np.any(np.isfinite(scores)):
        return None

    j = int(np.nanargmin(scores))
    # quadratic refinement around j (use up to 5 points)
    j0 = max(0, j - 2)
    j1 = min(len(shifts), j + 3)
    x = shifts[j0:j1]
    y = scores[j0:j1]
    m2 = np.isfinite(y)
    if np.sum(m2) >= 3:
        x = x[m2]
        y = y[m2]
        # fit y = a x^2 + b x + c ; min at -b/(2a)
        a, b, c = np.polyfit(x, y, 2)
        if a != 0:
            s_ref = -b / (2 * a)
            s_ref = float(np.clip(s_ref, -duration, duration))
            return tc_pred + s_ref

    return tc_pred + float(shifts[j])


def estimate_transit_midtimes(time, flux, period, t0, duration):
    """
    Build predicted transit centers across the time span and measure mid-times locally.
    Returns list of (n, tc_meas).
    """
    tmin, tmax = float(np.nanmin(time)), float(np.nanmax(time))
    n_start = int(np.floor((tmin - t0) / period)) - 1
    n_end = int(np.ceil((tmax - t0) / period)) + 1

    out = []
    for n in range(n_start, n_end + 1):
        tc_pred = t0 + n * period
        if tc_pred < tmin - 2 * duration or tc_pred > tmax + 2 * duration:
            continue
        tc_meas = _measure_midtime_box_grid(time, flux, tc_pred, duration)
        if tc_meas is not None:
            out.append((n, float(tc_meas)))
    return out


def fit_linear_ephemeris(ns, tcs):
    """
    Fit tc = t0 + n*P by least squares.
    Returns t0, P, sigma_t0, sigma_P, residuals
    """
    ns = np.asarray(ns, dtype=float)
    tcs = np.asarray(tcs, dtype=float)
    if ns.size < 3:
        return None

    n0 = np.mean(ns)
    x = ns - n0
    A = np.vstack([np.ones_like(x), x]).T

    coeff, _, _, _ = np.linalg.lstsq(A, tcs, rcond=None)
    c0, P = coeff  # tc ~ c0 + P*(n-n0)
    t0 = c0 - P * n0

    model = t0 + P * ns
    resid = tcs - model

    dof = max(1, ns.size - 2)
    s2 = np.sum(resid**2) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    sigma_c0 = np.sqrt(cov[0, 0])
    sigma_P = np.sqrt(cov[1, 1])

    # sigma_t0 same as sigma_c0 in this parameterization (t0 is derived)
    # Good enough for display purposes:
    sigma_t0 = sigma_c0

    return float(t0), float(P), float(sigma_t0), float(sigma_P), resid


# -------------------------
# Main App
# -------------------------
class TransitKitGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TransitKit")
        self.geometry("1180x760")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.sim_tab = ttk.Frame(self.nb)
        self.tess_tab = ttk.Frame(self.nb)

        self.nb.add(self.sim_tab, text="Simulate")
        self.nb.add(self.tess_tab, text="TESS Explorer")

        # buffers
        self.tess_segments = []  # list of dicts: {"sector":int/str, "time":np.array, "flux":np.array}
        self.tess_time = None
        self.tess_flux = None

        # ephemeris state (NEA or refined)
        self.ephem_period = None
        self.ephem_t0 = None  # BTJD
        self.ephem_duration = None  # days

        self._sr_filtered = None

        self._build_simulate_tab()
        self._build_tess_tab()

    # minimal simulate tab (kept simple)
    def _build_simulate_tab(self):
        ttk.Label(self.sim_tab, text="Use TESS Explorer tab for real data.").pack(anchor="w", padx=12, pady=12)

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
        self.tess_cadence = tk.StringVar(value="2-min (120s)")
        self.plot_mode = tk.StringVar(value="Per-sector panels")  # default best for multi-sector

        self.do_flatten = tk.BooleanVar(value=True)
        self.do_outliers = tk.BooleanVar(value=True)

        ttk.Label(left, text="Planet name / Host / TIC:").pack(anchor="w")
        ttk.Entry(left, textvariable=self.tess_target, width=32).pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Author:").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.tess_author, values=["SPOC", "QLP", "Any"], width=29, state="readonly").pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Cadence:").pack(anchor="w")
        ttk.Combobox(
            left,
            textvariable=self.tess_cadence,
            values=["Any", "20-sec (20s)", "2-min (120s)", "10-min (600s)", "30-min (1800s)"],
            width=29,
            state="readonly",
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Plot mode:").pack(anchor="w")
        ttk.Combobox(
            left,
            textvariable=self.plot_mode,
            values=["Per-sector panels", "Stitched (absolute BTJD)", "Concatenated (no gaps)"],
            width=29,
            state="readonly",
        ).pack(anchor="w", pady=(0, 8))

        ttk.Checkbutton(left, text="Remove outliers", variable=self.do_outliers).pack(anchor="w")
        ttk.Checkbutton(left, text="Flatten/detrend", variable=self.do_flatten).pack(anchor="w", pady=(0, 8))

        ttk.Button(left, text="Fetch NEA Params", command=self.on_nea_fetch).pack(fill=tk.X, pady=(6, 4))
        ttk.Button(left, text="Search TESS", command=self.on_tess_search).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Download Selected", command=self.on_tess_download).pack(fill=tk.X, pady=4)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Button(left, text="Run BLS (better)", command=self.on_tess_bls).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Refine Ephemeris (high precision)", command=self.on_refine_ephemeris).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Show Transit Markers", command=self.on_show_transit_markers).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Transit Viewer", command=self.on_transit_viewer).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Export CSV", command=self.on_tess_export).pack(fill=tk.X, pady=4)

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Available Light Curves:", font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self.tess_list = tk.Listbox(left, selectmode=tk.EXTENDED, width=46, height=14)
        self.tess_list.pack(fill=tk.BOTH, expand=False)

        self.tess_status = ttk.Label(left, text="Ready.", wraplength=340)
        self.tess_status.pack(fill=tk.X, pady=(10, 0))

        self.tess_plot = PlotPanel(right, title="TESS Light Curve")
        self.tess_plot.pack(fill=tk.BOTH, expand=True)

    def _require_lightkurve(self) -> bool:
        try:
            import lightkurve  # noqa: F401
            return True
        except Exception:
            messagebox.showerror(
                "Missing dependency",
                'This feature requires lightkurve.\n\nInstall:\n  python -m pip install lightkurve'
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
        return None

    # -------- NEA fetch --------
    def on_nea_fetch(self):
        q = self.tess_target.get().strip()
        if not q:
            messagebox.showerror("Invalid input", "Enter a planet name (e.g., HAT-P-36 b).")
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
                    tranmid_jd = row.get("pl_tranmid")  # JD
                    msg = f"NEA: {pl} | host={host} | TIC={tic} | P={per} d | dur={dur_hr} hr"
                    self.tess_status.config(text=msg)

                    # store ephemeris from NEA if available
                    if per is not None:
                        self.ephem_period = float(per)
                    if dur_hr is not None:
                        self.ephem_duration = float(dur_hr) / 24.0
                    if tranmid_jd is not None:
                        # TESS time in Lightkurve is typically BTJD = BJD - 2457000
                        self.ephem_t0 = float(tranmid_jd) - 2457000.0

                    # if TIC exists, use it for Lightkurve search (usually best)
                    if tic not in (None, "", "null"):
                        self.tess_target.set(f"TIC {tic}")
                        self.on_tess_search()

                self.after(0, apply)

            except Exception as e:
                self.after(0, lambda: self.tess_status.config(text="NEA query failed."))
                self.after(0, lambda: messagebox.showerror("NEA error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    # -------- TESS search --------
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

                # Filter by cadence if possible
                if cadence is not None and len(sr) > 0:
                    tbl = sr.table
                    if "exptime" in tbl.colnames:
                        mask = np.array(tbl["exptime"]) == cadence
                        sr_f = sr[mask]
                    else:
                        sr_f = sr
                else:
                    sr_f = sr

                self._sr_filtered = sr_f

                for i, row in enumerate(sr_f.table):
                    sector = row["sequence_number"] if "sequence_number" in row.colnames else row.get("sector", "NA")
                    exptime = row["exptime"] if "exptime" in row.colnames else "NA"
                    auth = row["author"] if "author" in row.colnames else author
                    label = f"[{i:02d}] Sector {sector} | exptime={exptime}s | author={auth}"
                    self.tess_list.insert(tk.END, label)

                self.after(0, lambda: self.tess_status.config(text=f"Found {len(sr_f)} light curve(s). Select & Download."))

            except Exception as e:
                self.after(0, lambda: self.tess_status.config(text="Search failed."))
                self.after(0, lambda: messagebox.showerror("Search error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    # -------- Download + preprocess + plot --------
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

                segs = []
                for lc in lcc:
                    # identify sector
                    sector = None
                    try:
                        sector = lc.meta.get("SECTOR", None)
                    except Exception:
                        sector = None
                    if sector is None:
                        try:
                            sector = lc.meta.get("sequence_number", None)
                        except Exception:
                            sector = None
                    if sector is None:
                        sector = "NA"

                    lc2 = lc.remove_nans()

                    if self.do_outliers.get() and hasattr(lc2, "remove_outliers"):
                        lc2 = lc2.remove_outliers(sigma=6)

                    if hasattr(lc2, "normalize"):
                        lc2 = lc2.normalize()

                    if self.do_flatten.get() and hasattr(lc2, "flatten"):
                        # window_length must be odd; tune later if needed
                        lc2 = lc2.flatten(window_length=401, polyorder=2, break_tolerance=5)

                    t = np.array(lc2.time.value, dtype=float)
                    f = np.array(lc2.flux.value, dtype=float)

                    # store
                    segs.append({"sector": sector, "time": t, "flux": f})

                # sort by sector when possible
                def _sec_key(x):
                    try:
                        return int(x["sector"])
                    except Exception:
                        return 1_000_000

                segs = sorted(segs, key=_sec_key)

                # build stitched arrays for BLS/export
                t_all = np.concatenate([s["time"] for s in segs])
                f_all = np.concatenate([s["flux"] for s in segs])
                o = np.argsort(t_all)
                t_all = t_all[o]
                f_all = f_all[o]

                self.tess_segments = segs
                self.tess_time = t_all
                self.tess_flux = f_all

                self.after(0, lambda: self._plot_segments())

                self.after(0, lambda: self.tess_status.config(
                    text=f"Downloaded {len(segs)} sector(s). Plot mode: {self.plot_mode.get()}."
                ))

            except Exception as e:
                self.after(0, lambda: self.tess_status.config(text="Download failed."))
                self.after(0, lambda: messagebox.showerror("Download error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _plot_segments(self):
        if not self.tess_segments:
            return

        mode = self.plot_mode.get()

        if mode == "Per-sector panels" and len(self.tess_segments) > 1:
            self.tess_plot.set_subplots(len(self.tess_segments))
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
            x_cat = []
            y_cat = []
            offset = 0.0
            gap = 0.2  # days between sectors visually
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
                                   title="TESS Concatenated Light Curve (no gaps)",
                                   style="k.", alpha=0.6, ms=2)
            return

        # default: stitched absolute BTJD
        self.tess_plot.set_subplots(1)
        self.tess_plot.plot_xy(self.tess_time, self.tess_flux,
                               xlabel="Time (BTJD days)", ylabel="Flux",
                               title="TESS Stitched Light Curve (absolute BTJD)",
                               style="k.", alpha=0.6, ms=2)

    # -------- Better BLS: narrow around NEA period if available --------
    def on_tess_bls(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download light curves first.")
            return

        # If NEA gave period, use a narrow range around it
        if self.ephem_period is not None:
            P0 = float(self.ephem_period)
            minP = max(0.2, P0 - 0.05)
            maxP = P0 + 0.05
        else:
            minP, maxP = 0.5, 20.0

        # If NEA gave duration, help BLS with duration grid
        durations = None
        if self.ephem_duration is not None:
            d0 = float(self.ephem_duration)
            durations = np.linspace(max(0.01, 0.5 * d0), 1.5 * d0, 20)

        try:
            res = tkit.find_transits_box(
                self.tess_time,
                self.tess_flux,
                min_period=minP,
                max_period=maxP,
                durations=durations,
                n_periods=20000 if (maxP - minP) <= 0.2 else 5000,
            )
        except Exception as e:
            messagebox.showerror("BLS error", str(e))
            return

        bestP = float(res["period"])
        self.ephem_period = bestP  # update ephemeris period from BLS if NEA missing

        periods = res.get("all_periods")
        y = res.get("all_power", None) or res.get("all_scores")
        ylabel = "BLS Power" if res.get("all_power", None) is not None else "Detection Score"

        # Plot power
        self.tess_plot.set_subplots(1)
        self.tess_plot.plot_line(periods, y, xlabel="Period (days)", ylabel=ylabel, title="BLS Period Search")
        self.tess_plot.vline(bestP, color="g", alpha=0.7, label=f"Detected {bestP:.6f} d")
        self.tess_plot.axes[0].legend(loc="best")
        self.tess_plot.canvas.draw()

        extra = []
        if "t0" in res:
            self.ephem_t0 = float(res["t0"])
            extra.append(f"t0={self.ephem_t0:.4f}")
        if "duration" in res:
            self.ephem_duration = float(res["duration"])
            extra.append(f"dur={self.ephem_duration:.4f}")
        if "depth" in res:
            extra.append(f"depth={float(res['depth']):.5f}")

        self.tess_status.config(text=f"BLS best period: {bestP:.8f} d | " + ", ".join(extra))

    # -------- High precision: measure mid-times and fit linear ephemeris --------
    def on_refine_ephemeris(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download light curves first.")
            return

        if self.ephem_period is None or self.ephem_t0 is None or self.ephem_duration is None:
            messagebox.showinfo(
                "Need ephemeris",
                "Fetch NEA Params first (best) OR run BLS once to get initial period/t0/duration."
            )
            return

        P0 = float(self.ephem_period)
        t0 = float(self.ephem_t0)
        dur = float(self.ephem_duration)

        self.tess_status.config(text="Refining ephemeris from individual transit events...")

        def work():
            try:
                mids = estimate_transit_midtimes(self.tess_time, self.tess_flux, P0, t0, dur)
                if len(mids) < 5:
                    raise RuntimeError(f"Only {len(mids)} transit events measured. Not enough for refinement.")

                ns = np.array([m[0] for m in mids], dtype=float)
                tcs = np.array([m[1] for m in mids], dtype=float)

                fit = fit_linear_ephemeris(ns, tcs)
                if fit is None:
                    raise RuntimeError("Ephemeris fit failed.")

                t0_fit, P_fit, s_t0, s_P, resid = fit

                # update ephemeris
                self.ephem_t0 = t0_fit
                self.ephem_period = P_fit

                def apply():
                    self.tess_status.config(
                        text=(
                            f"Refined: P={P_fit:.9f} ± {s_P:.2e} d | "
                            f"t0={t0_fit:.6f} (BTJD) | N={len(ns)} events"
                        )
                    )
                    # after refine, re-plot stitched light curve (if you want) and allow markers
                    self._plot_segments()

                self.after(0, apply)

            except Exception as e:
                self.after(0, lambda: self.tess_status.config(text="Refine failed."))
                self.after(0, lambda: messagebox.showerror("Refine error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    # -------- Show transit markers on current plot --------
    def on_show_transit_markers(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download light curves first.")
            return
        if self.ephem_period is None or self.ephem_t0 is None or self.ephem_duration is None:
            messagebox.showinfo("Need ephemeris", "Fetch NEA Params or run BLS/refine first.")
            return

        P = float(self.ephem_period)
        t0 = float(self.ephem_t0)

        # predicted centers across full timespan
        tmin, tmax = float(np.nanmin(self.tess_time)), float(np.nanmax(self.tess_time))
        n_start = int(np.floor((tmin - t0) / P)) - 1
        n_end = int(np.ceil((tmax - t0) / P)) + 1
        centers = [t0 + n * P for n in range(n_start, n_end + 1) if (t0 + n * P) >= tmin and (t0 + n * P) <= tmax]

        # replot (so we don't accumulate infinite lines)
        self._plot_segments()

        mode = self.plot_mode.get()
        if mode == "Per-sector panels" and len(self.tess_segments) > 1:
            for i, seg in enumerate(self.tess_segments):
                tseg = seg["time"]
                tmin_s, tmax_s = float(tseg.min()), float(tseg.max())
                for tc in centers:
                    if tmin_s <= tc <= tmax_s:
                        self.tess_plot.vline(tc, color="r", alpha=0.25, ax_index=i)
            self.tess_plot.canvas.draw()
        else:
            for tc in centers:
                self.tess_plot.vline(tc, color="r", alpha=0.25, ax_index=0)
            self.tess_plot.canvas.draw()

        self.tess_status.config(text=f"Plotted {len(centers)} transit markers using P={P:.9f} d")

    # -------- Transit Viewer: pick events and zoom --------
    def on_transit_viewer(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download light curves first.")
            return
        if self.ephem_period is None or self.ephem_t0 is None or self.ephem_duration is None:
            messagebox.showinfo("Need ephemeris", "Fetch NEA Params or run BLS/refine first.")
            return

        P = float(self.ephem_period)
        t0 = float(self.ephem_t0)
        dur = float(self.ephem_duration)

        tmin, tmax = float(np.nanmin(self.tess_time)), float(np.nanmax(self.tess_time))
        n_start = int(np.floor((tmin - t0) / P)) - 1
        n_end = int(np.ceil((tmax - t0) / P)) + 1
        events = []
        for n in range(n_start, n_end + 1):
            tc = t0 + n * P
            if tmin <= tc <= tmax:
                events.append((n, tc))

        if not events:
            messagebox.showinfo("No events", "No predicted transit events in this time range.")
            return

        win = tk.Toplevel(self)
        win.title("Transit Viewer")
        win.geometry("1100x600")

        left = ttk.Frame(win, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(win, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Select a transit event:").pack(anchor="w")
        lb = tk.Listbox(left, width=30, height=25)
        lb.pack(fill=tk.BOTH, expand=False)

        for i, (n, tc) in enumerate(events):
            lb.insert(tk.END, f"n={n:6d}  tc={tc:.6f}")

        panel = PlotPanel(right, title="Transit Window")
        panel.pack(fill=tk.BOTH, expand=True)

        def plot_event(idx):
            n, tc = events[idx]
            w = 3.0 * dur
            m = (self.tess_time >= tc - w) & (self.tess_time <= tc + w)
            if np.sum(m) < 10:
                return
            tt = self.tess_time[m]
            ff = self.tess_flux[m]
            panel.set_subplots(1)
            panel.plot_xy(tt, ff, xlabel="Time (BTJD)", ylabel="Flux",
                          title=f"Transit n={n}  (tc={tc:.6f}, window=±{w:.3f} d)",
                          style="k.", alpha=0.7, ms=3)
            panel.vline(tc, color="r", alpha=0.4)
            panel.canvas.draw()

        def on_select(evt):
            sel = lb.curselection()
            if sel:
                plot_event(int(sel[0]))

        lb.bind("<<ListboxSelect>>", on_select)

        # auto select first
        lb.selection_set(0)
        plot_event(0)

    # -------- Export CSV (stitched arrays) --------
    def on_tess_export(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download light curves first.")
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
        self.tess_status.config(text=f"Exported: {os.path.basename(path)}")


def main():
    app = TransitKitGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

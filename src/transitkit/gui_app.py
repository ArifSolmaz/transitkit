"""
TransitKit GUI (Tkinter) – fixes + terminal-like logs + robust TIC parsing.

Fixes:
- Avoid NumPy array truth-value bug in BLS plotting.
- Normalize target so it never becomes "TIC TIC ####".
- Show progress logs in GUI (like terminal) + busy spinner.

Notes:
- Uses lightkurve for TESS search/download
- Uses transitkit.nea.lookup_planet for NEA
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
# App
# -------------------------
class TransitKitGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TransitKit")
        self.geometry("1240x820")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.sim_tab = ttk.Frame(self.nb)
        self.tess_tab = ttk.Frame(self.nb)

        self.nb.add(self.sim_tab, text="Simulate")
        self.nb.add(self.tess_tab, text="TESS Explorer")

        ttk.Label(self.sim_tab, text="Use TESS Explorer for real data.", padding=12).pack(anchor="w")

        # TESS state
        self.tess_segments = []  # list of {"sector":..., "time":..., "flux":...}
        self.tess_time = None
        self.tess_flux = None
        self._sr_filtered = None

        # ephemeris state (BTJD)
        self.ephem_period = None
        self.ephem_t0 = None
        self.ephem_duration = None

        self._busy_count = 0

        self._build_tess_tab()

    # ---------- logging ----------
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

    # ---------- deps ----------
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

    # ---------- target normalization ----------
    def normalize_target(self, raw: str) -> str:
        """
        Normalize user input:
        - If it contains 'TIC' and digits -> return 'TIC <digits>'
        - If it contains just digits after TIC or repeated TIC -> fix it
        - Otherwise return raw as-is
        """
        s = (raw or "").strip()
        if not s:
            return s

        # Collapse multiple spaces
        s = re.sub(r"\s+", " ", s).strip()

        # If it contains TIC anywhere, extract the digits
        if re.search(r"\bTIC\b", s, flags=re.IGNORECASE):
            digits = re.findall(r"\d+", s)
            if digits:
                return f"TIC {digits[0]}"
            # If no digits, just return as-is
            return s

        return s

    def _cadence_seconds(self):
        s = self.tess_cadence.get()
        return {
            "20-sec (20s)": 20,
            "2-min (120s)": 120,
            "10-min (600s)": 600,
            "30-min (1800s)": 1800,
        }.get(s, None)

    # ---------- UI ----------
    def _build_tess_tab(self):
        left = ttk.Frame(self.tess_tab, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(self.tess_tab, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="TESS Explorer", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        self.tess_target = tk.StringVar(value="")  # EMPTY by default (user enters)
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

        ttk.Checkbutton(left, text="Remove outliers (OOT-safe)", variable=self.do_outliers).pack(anchor="w")
        ttk.Checkbutton(left, text="Flatten/detrend (mask transits)", variable=self.do_flatten).pack(anchor="w", pady=(0, 8))

        ttk.Button(left, text="Fetch NEA Params", command=self.on_nea_fetch).pack(fill=tk.X, pady=(6, 4))
        ttk.Button(left, text="Search TESS", command=self.on_tess_search).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Download Selected (PDCSAP)", command=self.on_tess_download).pack(fill=tk.X, pady=4)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Button(left, text="Run BLS (narrowed)", command=self.on_tess_bls).pack(fill=tk.X, pady=4)

        ttk.Button(left, text="Export CSV", command=self.on_tess_export).pack(fill=tk.X, pady=4)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Available Light Curves:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.tess_list = tk.Listbox(left, selectmode=tk.EXTENDED, width=52, height=10)
        self.tess_list.pack(fill=tk.X, expand=False)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        # Status + spinner
        self.tess_status = ttk.Label(left, text="Ready.", wraplength=380)
        self.tess_status.pack(fill=tk.X, pady=(0, 6))

        self.busy = ttk.Progressbar(left, mode="indeterminate")
        self.busy.pack(fill=tk.X, pady=(0, 8))

        # Log box (terminal-like)
        ttk.Label(left, text="Log:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.log_box = ScrolledText(left, width=52, height=14, state="disabled")
        self.log_box.pack(fill=tk.BOTH, expand=True)

        # Plot
        self.tess_plot = PlotPanel(right, title="TESS Light Curve")
        self.tess_plot.pack(fill=tk.BOTH, expand=True)

    # ---------- NEA ----------
    def on_nea_fetch(self):
        raw = self.tess_target.get()
        q = (raw or "").strip()
        if not q:
            messagebox.showerror("Invalid input", "Enter a planet name (e.g., HAT-P-36 b) or a TIC ID.")
            return

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

                        # Auto-fill to TIC if available (but avoid "TIC TIC")
                        if tic not in (None, "", "null"):
                            target = self.normalize_target(f"TIC {tic}")
                            self.tess_target.set(target)
                            self.set_status(f"Target set to: {target} (from NEA)")
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

    # ---------- Search ----------
    def on_tess_search(self):
        if not self._require_lightkurve():
            return
        import lightkurve as lk

        raw = self.tess_target.get()
        target = self.normalize_target(raw)
        self.tess_target.set(target)

        author = self.tess_author.get()
        cadence = self._cadence_seconds()

        if not target:
            messagebox.showerror("Invalid input", "Target cannot be empty.")
            return

        self.tess_list.delete(0, tk.END)

        self._set_busy(True)
        self.set_status(f"Searching TESS: target='{target}', author={author}, cadence={cadence or 'Any'} ...")

        def work():
            try:
                kw = {}
                if author != "Any":
                    kw["author"] = author

                sr = lk.search_lightcurve(target, mission="TESS", **kw)

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

                def apply():
                    try:
                        for i, row in enumerate(sr_f.table):
                            sector = row["sequence_number"] if "sequence_number" in row.colnames else row.get("sector", "NA")
                            exptime = row["exptime"] if "exptime" in row.colnames else "NA"
                            auth = row["author"] if "author" in row.colnames else author
                            self.tess_list.insert(tk.END, f"[{i:02d}] Sector {sector} | exptime={exptime}s | author={auth}")

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

    # ---------- Download + plot ----------
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
        self.set_status(f"Downloading {len(idxs)} selected light curve(s) ...")

        def work():
            try:
                sr_sel = self._sr_filtered[idxs]
                lcc = sr_sel.download_all()
                if lcc is None or len(lcc) == 0:
                    raise RuntimeError("Download returned no light curves.")

                segs = []
                for k, lc in enumerate(lcc):
                    # sector
                    sector = None
                    try:
                        sector = lc.meta.get("SECTOR", None)
                    except Exception:
                        sector = None
                    if sector is None:
                        sector = "NA"

                    # remove NaNs
                    lc2 = lc.remove_nans()

                    # Force PDCSAP if available
                    used = "default"
                    try:
                        pd = lc2["PDCSAP_FLUX"]
                        lc2 = lc2.copy()
                        lc2.flux = pd
                        used = "PDCSAP"
                    except Exception:
                        pass

                    # normalize
                    if hasattr(lc2, "normalize"):
                        lc2 = lc2.normalize()

                    # simple outlier removal (optional)
                    if self.do_outliers.get() and hasattr(lc2, "remove_outliers"):
                        lc2 = lc2.remove_outliers(sigma=6)

                    # flatten (optional)
                    if self.do_flatten.get() and hasattr(lc2, "flatten"):
                        lc2 = lc2.flatten(window_length=401, polyorder=2, break_tolerance=5)

                    t = np.array(lc2.time.value, dtype=float)
                    f = np.array(lc2.flux.value, dtype=float)

                    segs.append({"sector": sector, "time": t, "flux": f})

                    # log from worker thread safely
                    self.after(0, lambda s=sector, u=used: self.log(f"Downloaded sector {s} (flux={u})"))

                # sort
                def _sec_key(x):
                    try:
                        return int(x["sector"])
                    except Exception:
                        return 10**9
                segs = sorted(segs, key=_sec_key)

                # stitched arrays (for BLS/export)
                t_all = np.concatenate([s["time"] for s in segs])
                f_all = np.concatenate([s["flux"] for s in segs])
                o = np.argsort(t_all)
                t_all = t_all[o]
                f_all = f_all[o]

                self.tess_segments = segs
                self.tess_time = t_all
                self.tess_flux = f_all

                def apply():
                    try:
                        # plot per-sector panels if >1 else single
                        if len(self.tess_segments) > 1:
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
                        else:
                            seg = self.tess_segments[0]
                            self.tess_plot.set_subplots(1)
                            self.tess_plot.plot_xy(seg["time"], seg["flux"],
                                                   xlabel="Time (BTJD days)", ylabel="Flux",
                                                   title=f"TESS Sector {seg['sector']} Light Curve",
                                                   style="k.", alpha=0.6, ms=2)

                        self.set_status(f"Download done: {len(self.tess_segments)} sector(s) plotted.")
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

    # ---------- BLS ----------
    def on_tess_bls(self):
        if self.tess_time is None or self.tess_flux is None:
            messagebox.showinfo("No data", "Download light curves first.")
            return

        # narrow around NEA period if available
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
                res = tkit.find_transits_box(
                    self.tess_time,
                    self.tess_flux,
                    min_period=minP,
                    max_period=maxP,
                    n_periods=nper,
                    durations=None,
                )

                bestP = float(res["period"])
                self.ephem_period = bestP

                periods = res.get("all_periods")
                # IMPORTANT FIX: do NOT use `or` with numpy arrays
                if "all_power" in res and res["all_power"] is not None:
                    y = res["all_power"]
                    ylabel = "BLS Power"
                else:
                    y = res.get("all_scores")
                    ylabel = "Detection Score"

                def apply():
                    try:
                        self.tess_plot.set_subplots(1)
                        self.tess_plot.plot_line(periods, y, xlabel="Period (days)", ylabel=ylabel, title="BLS Period Search")
                        self.tess_plot.vline(bestP, color="g", alpha=0.8, label=f"Detected {bestP:.9f} d")
                        self.tess_plot.axes[0].legend(loc="best")
                        self.tess_plot.canvas.draw()
                        self.set_status(f"BLS done: best period = {bestP:.9f} days")
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

    # ---------- Export ----------
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
        self.set_status(f"Exported CSV: {os.path.basename(path)}")


def main():
    app = TransitKitGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

"""
TransitKit v3.0 - Publication Generator

Generate publication-ready outputs:
- LaTeX tables (AAS, MNRAS formats)
- Journal-quality figures
- Complete paper skeleton

Example:
    >>> target = UniversalTarget("WASP-39 b")
    >>> pub = PublicationGenerator(target)
    >>> pub.generate_all("output_dir/")
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from pathlib import Path
import numpy as np


# Lazy import
def _import_matplotlib():
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    return plt


@dataclass
class PublicationConfig:
    """Configuration for publication outputs."""

    journal: str = "AAS"  # AAS, MNRAS, A&A
    figure_format: str = "pdf"  # pdf, png, eps
    figure_dpi: int = 300
    latex_format: str = "aastex"  # aastex, mnras, aa
    include_appendix: bool = True
    author: str = ""
    title: str = ""


class PublicationGenerator:
    """
    Generate publication-ready materials.

    Outputs:
    - LaTeX tables for stellar and planetary parameters
    - Phase-folded transit figure
    - Multi-panel light curve figure
    - Transmission spectrum figure (if available)
    - Complete paper skeleton

    Example:
        >>> target = UniversalTarget("WASP-39 b")
        >>> results = target.analyze()
        >>> pub = PublicationGenerator(target, results)
        >>> pub.generate_all("my_paper/")
    """

    def __init__(
        self,
        target,
        analysis_results: Optional[Dict] = None,
        config: Optional[PublicationConfig] = None,
    ):
        """
        Initialize publication generator.

        Args:
            target: UniversalTarget instance
            analysis_results: Results from analysis pipeline
            config: Publication configuration
        """
        self.target = target
        self.results = analysis_results or {}
        self.config = config or PublicationConfig()

    def generate_all(self, output_dir: str):
        """
        Generate all publication materials.

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Create subdirectories
        (output_path / "figures").mkdir(exist_ok=True)
        (output_path / "tables").mkdir(exist_ok=True)

        # Generate tables
        self.generate_stellar_table(output_path / "tables" / "stellar_params.tex")
        self.generate_planet_table(output_path / "tables" / "planet_params.tex")

        # Generate figures
        if self.results.get("lightcurves"):
            self.generate_lightcurve_figure(output_path / "figures" / "lightcurve.pdf")

        if self.results.get("candidates"):
            self.generate_transit_figure(output_path / "figures" / "transit.pdf")

        if self.results.get("transmission_spectrum"):
            self.generate_spectrum_figure(output_path / "figures" / "transmission.pdf")

        # Generate paper skeleton
        self.generate_paper_skeleton(output_path / "paper.tex")

        print(f"📝 Publication materials generated in {output_dir}")

    def generate_stellar_table(self, filepath: str):
        """Generate LaTeX table of stellar parameters."""
        stellar = self.target.stellar
        ids = self.target.ids

        if self.config.latex_format == "aastex":
            table = self._aastex_stellar_table(stellar, ids)
        else:
            table = self._generic_stellar_table(stellar, ids)

        with open(filepath, "w") as f:
            f.write(table)

    def _aastex_stellar_table(self, stellar, ids) -> str:
        """Generate AASTeX format stellar table."""
        lines = [
            r"\begin{deluxetable}{lcc}",
            r"\tablecaption{Stellar Parameters\label{tab:stellar}}",
            r"\tablewidth{0pt}",
            r"\tablehead{",
            r"  \colhead{Parameter} & \colhead{Value} & \colhead{Source}",
            r"}",
            r"\startdata",
        ]

        # Identifiers
        if ids.tic:
            lines.append(rf"TIC ID & {ids.tic} & TIC \\")
        if ids.gaia_dr3:
            lines.append(rf"Gaia DR3 ID & {ids.gaia_dr3} & Gaia \\")

        lines.append(r"\hline")

        # Coordinates
        if stellar.ra and stellar.dec:
            lines.append(rf"R.A. (J2000) & {stellar.ra:.6f}$^\circ$ & Gaia \\")
            lines.append(rf"Dec. (J2000) & {stellar.dec:.6f}$^\circ$ & Gaia \\")

        lines.append(r"\hline")

        # Physical parameters
        if stellar.teff:
            err = f" $\\pm$ {stellar.teff_err:.0f}" if stellar.teff_err else ""
            lines.append(
                rf"$T_{{\rm eff}}$ (K) & {stellar.teff:.0f}{err} & {stellar.source or 'TIC'} \\"
            )

        if stellar.logg:
            err = f" $\\pm$ {stellar.logg_err:.2f}" if stellar.logg_err else ""
            lines.append(
                rf"$\log g$ (cgs) & {stellar.logg:.2f}{err} & {stellar.source or 'TIC'} \\"
            )

        if stellar.feh:
            err = f" $\\pm$ {stellar.feh_err:.2f}" if stellar.feh_err else ""
            lines.append(rf"[Fe/H] (dex) & {stellar.feh:.2f}{err} & {stellar.source or 'TIC'} \\")

        if stellar.radius:
            err = f" $\\pm$ {stellar.radius_err:.3f}" if stellar.radius_err else ""
            lines.append(
                rf"$R_\star$ ($R_\odot$) & {stellar.radius:.3f}{err} & {stellar.source or 'TIC'} \\"
            )

        if stellar.mass:
            err = f" $\\pm$ {stellar.mass_err:.3f}" if stellar.mass_err else ""
            lines.append(
                rf"$M_\star$ ($M_\odot$) & {stellar.mass:.3f}{err} & {stellar.source or 'TIC'} \\"
            )

        if stellar.distance:
            err = f" $\\pm$ {stellar.distance_err:.1f}" if stellar.distance_err else ""
            lines.append(rf"Distance (pc) & {stellar.distance:.1f}{err} & Gaia \\")

        lines.extend(
            [
                r"\enddata",
                r"\end{deluxetable}",
            ]
        )

        return "\n".join(lines)

    def _generic_stellar_table(self, stellar, ids) -> str:
        """Generate generic LaTeX stellar table."""
        lines = [
            r"\begin{table}",
            r"\centering",
            r"\caption{Stellar Parameters}",
            r"\label{tab:stellar}",
            r"\begin{tabular}{lcc}",
            r"\hline\hline",
            r"Parameter & Value & Source \\",
            r"\hline",
        ]

        if ids.tic:
            lines.append(rf"TIC ID & {ids.tic} & TIC \\")
        if stellar.teff:
            lines.append(
                rf"$T_{{\rm eff}}$ (K) & {stellar.teff:.0f} & {stellar.source or 'TIC'} \\"
            )
        if stellar.radius:
            lines.append(
                rf"$R_\star$ ($R_\odot$) & {stellar.radius:.3f} & {stellar.source or 'TIC'} \\"
            )
        if stellar.mass:
            lines.append(
                rf"$M_\star$ ($M_\odot$) & {stellar.mass:.3f} & {stellar.source or 'TIC'} \\"
            )

        lines.extend(
            [
                r"\hline",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def generate_planet_table(self, filepath: str):
        """Generate LaTeX table of planetary parameters."""
        planets = self.target.planets

        if not planets:
            return

        if self.config.latex_format == "aastex":
            table = self._aastex_planet_table(planets)
        else:
            table = self._generic_planet_table(planets)

        with open(filepath, "w") as f:
            f.write(table)

    def _aastex_planet_table(self, planets) -> str:
        """Generate AASTeX format planet table."""
        lines = [
            r"\begin{deluxetable*}{lccc}",
            r"\tablecaption{Planetary Parameters\label{tab:planets}}",
            r"\tablewidth{0pt}",
            r"\tablehead{",
            r"  \colhead{Parameter} & " + " & ".join([rf"\colhead{{{p.name}}}" for p in planets]),
            r"}",
            r"\startdata",
        ]

        # Orbital parameters
        if any(p.period for p in planets):
            values = [f"{p.period:.6f}" if p.period else "\\nodata" for p in planets]
            lines.append(rf"$P$ (days) & " + " & ".join(values) + r" \\")

        if any(p.t0 for p in planets):
            values = [f"{p.t0:.4f}" if p.t0 else "\\nodata" for p in planets]
            lines.append(rf"$T_0$ (BJD) & " + " & ".join(values) + r" \\")

        if any(p.duration for p in planets):
            values = [f"{p.duration:.3f}" if p.duration else "\\nodata" for p in planets]
            lines.append(rf"$T_{{14}}$ (hr) & " + " & ".join(values) + r" \\")

        if any(p.rp_rs for p in planets):
            values = [f"{p.rp_rs:.4f}" if p.rp_rs else "\\nodata" for p in planets]
            lines.append(rf"$R_p/R_\star$ & " + " & ".join(values) + r" \\")

        if any(p.a_rs for p in planets):
            values = [f"{p.a_rs:.2f}" if p.a_rs else "\\nodata" for p in planets]
            lines.append(rf"$a/R_\star$ & " + " & ".join(values) + r" \\")

        if any(p.inc for p in planets):
            values = [f"{p.inc:.2f}" if p.inc else "\\nodata" for p in planets]
            lines.append(rf"$i$ (deg) & " + " & ".join(values) + r" \\")

        lines.append(r"\hline")

        # Physical parameters
        if any(p.radius for p in planets):
            values = [f"{p.radius:.2f}" if p.radius else "\\nodata" for p in planets]
            lines.append(rf"$R_p$ ($R_\oplus$) & " + " & ".join(values) + r" \\")

        if any(p.mass for p in planets):
            values = [f"{p.mass:.2f}" if p.mass else "\\nodata" for p in planets]
            lines.append(rf"$M_p$ ($M_\oplus$) & " + " & ".join(values) + r" \\")

        if any(p.teq for p in planets):
            values = [f"{p.teq:.0f}" if p.teq else "\\nodata" for p in planets]
            lines.append(rf"$T_{{eq}}$ (K) & " + " & ".join(values) + r" \\")

        if any(p.insol for p in planets):
            values = [f"{p.insol:.1f}" if p.insol else "\\nodata" for p in planets]
            lines.append(rf"$S$ ($S_\oplus$) & " + " & ".join(values) + r" \\")

        lines.extend(
            [
                r"\enddata",
                r"\end{deluxetable*}",
            ]
        )

        return "\n".join(lines)

    def _generic_planet_table(self, planets) -> str:
        """Generate generic LaTeX planet table."""
        n_planets = len(planets)
        col_spec = "l" + "c" * n_planets

        lines = [
            r"\begin{table*}",
            r"\centering",
            r"\caption{Planetary Parameters}",
            r"\label{tab:planets}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\hline\hline",
            "Parameter & " + " & ".join([p.name for p in planets]) + r" \\",
            r"\hline",
        ]

        if any(p.period for p in planets):
            values = [f"{p.period:.6f}" if p.period else "--" for p in planets]
            lines.append("$P$ (days) & " + " & ".join(values) + r" \\")

        if any(p.radius for p in planets):
            values = [f"{p.radius:.2f}" if p.radius else "--" for p in planets]
            lines.append("$R_p$ ($R_\\oplus$) & " + " & ".join(values) + r" \\")

        lines.extend(
            [
                r"\hline",
                r"\end{tabular}",
                r"\end{table*}",
            ]
        )

        return "\n".join(lines)

    def generate_lightcurve_figure(self, filepath: str):
        """Generate multi-panel light curve figure."""
        plt = _import_matplotlib()

        lightcurves = self.results.get("lightcurves", [])
        if not lightcurves:
            return

        n_lc = min(len(lightcurves), 6)

        fig, axes = plt.subplots(n_lc, 1, figsize=(10, 2 * n_lc), sharex=False)
        if n_lc == 1:
            axes = [axes]

        for i, (lc, ax) in enumerate(zip(lightcurves[:n_lc], axes)):
            ax.plot(lc.time, lc.flux, "k.", markersize=0.5, alpha=0.5)
            ax.set_ylabel("Flux", fontsize=10)
            ax.set_title(lc.label, fontsize=10)
            ax.tick_params(labelsize=8)

        axes[-1].set_xlabel("Time (BJD)", fontsize=10)

        plt.tight_layout()
        plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches="tight")
        plt.close()

    def generate_transit_figure(self, filepath: str):
        """Generate phase-folded transit figure."""
        plt = _import_matplotlib()

        candidates = self.results.get("candidates", [])
        if not candidates:
            return

        # Best candidate
        candidate = candidates[0]

        # Get combined light curve
        combined = self.results.get("combined_lc")
        if combined is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Phase fold
        phase = ((combined.time - candidate.t0) % candidate.period) / candidate.period
        phase[phase > 0.5] -= 1

        # Full phase curve
        ax = axes[0]
        ax.plot(phase, combined.flux, "k.", markersize=1, alpha=0.3)
        ax.set_xlabel("Phase", fontsize=12)
        ax.set_ylabel("Normalized Flux", fontsize=12)
        ax.set_title("Full Phase Curve", fontsize=12)
        ax.set_xlim(-0.5, 0.5)

        # Zoomed transit
        ax = axes[1]
        transit_mask = np.abs(phase) < 0.05
        ax.plot(
            phase[transit_mask] * candidate.period * 24,
            combined.flux[transit_mask],
            "k.",
            markersize=2,
            alpha=0.5,
        )
        ax.set_xlabel("Time from mid-transit (hours)", fontsize=12)
        ax.set_ylabel("Normalized Flux", fontsize=12)
        ax.set_title(
            f"Transit (P={candidate.period:.4f} d, depth={candidate.depth:.0f} ppm)", fontsize=12
        )

        plt.tight_layout()
        plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches="tight")
        plt.close()

    def generate_spectrum_figure(self, filepath: str):
        """Generate transmission spectrum figure."""
        plt = _import_matplotlib()

        spectrum = self.results.get("transmission_spectrum")
        if spectrum is None:
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.errorbar(
            spectrum.wavelength,
            spectrum.depth_ppm,
            yerr=(
                spectrum.transit_depth_err * 1e6
                if np.median(spectrum.transit_depth_err) < 0.1
                else spectrum.transit_depth_err
            ),
            fmt="o",
            markersize=4,
            capsize=2,
            color="navy",
            alpha=0.7,
        )

        ax.set_xlabel("Wavelength (μm)", fontsize=12)
        ax.set_ylabel("Transit Depth (ppm)", fontsize=12)
        ax.set_title(f"{spectrum.planet_name} Transmission Spectrum", fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches="tight")
        plt.close()

    def generate_paper_skeleton(self, filepath: str):
        """Generate complete paper skeleton."""
        planet_name = self.target.ids.planet_name or self.target.identifier

        if self.config.latex_format == "aastex":
            skeleton = self._aastex_skeleton(planet_name)
        else:
            skeleton = self._generic_skeleton(planet_name)

        with open(filepath, "w") as f:
            f.write(skeleton)

    def _aastex_skeleton(self, planet_name: str) -> str:
        """Generate AASTeX paper skeleton."""
        return rf"""% Generated by TransitKit v3.0
\documentclass[twocolumn]{{aastex631}}

\usepackage{{graphicx}}
\usepackage{{natbib}}

\begin{{document}}

\title{{{planet_name}: A [Description] Planet Discovered/Characterized with TransitKit}}

\author{{{self.config.author or '[Author Name]'}}}
\affiliation{{[Institution]}}

\begin{{abstract}}
We present the discovery/characterization of {planet_name} using data from 
[TESS/Kepler/JWST]. Using TransitKit v3.0, we analyzed [X] sectors/quarters 
of photometric data, identifying a transit signal with period P = [X] days 
and depth [X] ppm. We derive planetary radius $R_p$ = [X] $R_\oplus$ and 
equilibrium temperature $T_{{eq}}$ = [X] K. [Additional results].
\end{{abstract}}

\keywords{{planets and satellites: detection --- techniques: photometric}}

\section{{Introduction}}

[Introduction text]

\section{{Observations}}

\subsection{{TESS Photometry}}
[TESS observations description]

\subsection{{Kepler/K2 Photometry}}
[Kepler/K2 description if applicable]

\subsection{{JWST Spectroscopy}}
[JWST description if applicable]

\section{{Analysis}}

\subsection{{Transit Detection}}
We used TransitKit v3.0 \citep{{transitkit}} to process the light curves
and search for transit signals. [Methods description]

\subsection{{Transit Fitting}}
[Fitting description]

\subsection{{Atmospheric Analysis}}
[If applicable]

\section{{Results}}

\input{{tables/stellar_params.tex}}

\input{{tables/planet_params.tex}}

\begin{{figure*}}
\includegraphics[width=\textwidth]{{figures/lightcurve.pdf}}
\caption{{Light curves of {planet_name} from [missions]. 
\label{{fig:lightcurve}}}}
\end{{figure*}}

\begin{{figure}}
\includegraphics[width=\columnwidth]{{figures/transit.pdf}}
\caption{{Phase-folded transit of {planet_name}. 
\label{{fig:transit}}}}
\end{{figure}}

\section{{Discussion}}

[Discussion]

\section{{Conclusions}}

[Conclusions]

\acknowledgments

This work made use of TransitKit v3.0 and data from [missions].

\facilities{{TESS, Kepler, JWST}}

\software{{TransitKit \citep{{transitkit}}, 
          lightkurve \citep{{lightkurve}},
          astropy \citep{{astropy}}}}

\bibliography{{references}}

\end{{document}}
"""

    def _generic_skeleton(self, planet_name: str) -> str:
        """Generate generic LaTeX paper skeleton."""
        return rf"""% Generated by TransitKit v3.0
\documentclass[a4paper,11pt]{{article}}

\usepackage{{graphicx}}
\usepackage{{natbib}}
\usepackage{{amsmath}}

\title{{{planet_name}: Analysis with TransitKit}}
\author{{{self.config.author or '[Author Name]'}}}
\date{{\today}}

\begin{{document}}

\maketitle

\begin{{abstract}}
[Abstract]
\end{{abstract}}

\section{{Introduction}}
[Introduction]

\section{{Observations}}
[Observations]

\section{{Analysis}}
[Analysis]

\section{{Results}}
\input{{tables/stellar_params.tex}}
\input{{tables/planet_params.tex}}

\section{{Conclusions}}
[Conclusions]

\bibliographystyle{{mnras}}
\bibliography{{references}}

\end{{document}}
"""

"""Professional Command Line Interface for TransitKit v2.0"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Optional, List, Tuple

import click
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import box

# Import transitkit modules
try:
    import transitkit as tk
    from transitkit.core import (
        TransitParameters,
        find_transits_bls_advanced,
        find_transits_multiple_methods,
        generate_transit_signal_mandel_agol,
        estimate_parameters_mcmc,
    )
    from transitkit.analysis import (
        detrend_light_curve_gp,
        remove_systematics_pca,
        measure_transit_timing_variations,
        calculate_transit_duration_ratio,
    )
    from transitkit.visualization import (
        setup_publication_style,
        plot_transit_summary,
        create_transit_report_figure,
        plot_mcmc_corner,
    )
    from transitkit.io import (
        load_tess_data_advanced,
        load_ground_based_data,
        export_transit_results,
    )
    from transitkit.utils import (
        calculate_snr,
        estimate_limb_darkening,
        calculate_transit_duration_from_parameters,
        check_data_quality,
    )
    from transitkit.validation import (
        validate_transit_parameters,
        compare_with_known_ephemeris,
        perform_injection_recovery_test,
        calculate_detection_significance,
    )
except ImportError as e:
    click.echo(f"Error importing TransitKit modules: {e}", err=True)
    click.echo("Make sure you have installed all dependencies.", err=True)
    sys.exit(1)

console = Console()
warnings.filterwarnings("ignore", category=UserWarning)

# Context settings for CLI
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

@click.group(context_settings=CONTEXT_SETTINGS)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--format", "-f", type=click.Choice(["text", "json", "csv", "latex"]),
              default="text", help="Output format")
@click.pass_context
def cli(ctx, verbose, debug, output, format):
    """
    🪐 TransitKit v2.0: Professional Exoplanet Transit Analysis Toolkit
    
    A comprehensive toolkit for analyzing exoplanet transit light curves.
    Features include: transit detection, parameter estimation, TTV analysis,
    MCMC fitting, publication-quality plots, and validation tools.
    """
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose
    ctx.obj["DEBUG"] = debug
    ctx.obj["OUTPUT_DIR"] = Path(output) if output else Path.cwd()
    ctx.obj["OUTPUT_FORMAT"] = format
    
    # Create output directory if it doesn't exist
    if output:
        ctx.obj["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    
    if verbose:
        console.print(f"[bold blue]TransitKit v{tk.__version__}[/bold blue]")
        console.print(f"Output directory: {ctx.obj['OUTPUT_DIR']}")
        console.print(f"Output format: {format}")

# ==================== INFO COMMANDS ====================

@cli.command()
def version():
    """Display version information"""
    console.print(Panel.fit(
        f"[bold cyan]TransitKit v{tk.__version__}[/bold cyan]\n"
        f"Author: {tk.__author__}\n"
        f"License: {tk.__license__}\n"
        f"Citation: {tk.__citation__}",
        title="Version Information",
        border_style="cyan"
    ))

@cli.command()
def info():
    """Display detailed package information"""
    table = Table(title="TransitKit Modules", box=box.ROUNDED)
    table.add_column("Module", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Status", justify="right")
    
    modules = [
        ("core", "Core transit analysis functions", "✅"),
        ("analysis", "Statistical analysis and detrending", "✅"),
        ("visualization", "Publication-quality plotting", "✅"),
        ("io", "Data I/O and mission support", "✅"),
        ("utils", "Utilities and calculations", "✅"),
        ("validation", "Validation and testing", "✅"),
    ]
    
    for name, desc, status in modules:
        table.add_row(name, desc, status)
    
    console.print(table)
    
    # Display available commands
    console.print("\n[bold]Available Commands:[/bold]")
    commands_table = Table(box=box.SIMPLE)
    commands_table.add_column("Command", style="yellow")
    commands_table.add_column("Description")
    
    for cmd in cli.commands.values():
        if cmd.name not in ["version", "info"]:
            commands_table.add_row(cmd.name, cmd.help.split('\n')[0] if cmd.help else "")
    
    console.print(commands_table)

# ==================== DATA COMMANDS ====================

@cli.group()
def data():
    """Data loading and preprocessing commands"""
    pass

@data.command(name="load")
@click.argument("target")
@click.option("--mission", type=click.Choice(["TESS", "Kepler", "K2", "ground"]),
              default="TESS", help="Mission or data source")
@click.option("--sectors", type=str, help="Sector numbers (comma-separated or 'all')")
@click.option("--author", type=str, default="SPOC", help="Pipeline author")
@click.option("--cadence", type=click.Choice(["fast", "short", "long"]),
              default="short", help="Observation cadence")
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def load_data(ctx, target, mission, sectors, author, cadence, output):
    """Load light curve data from various missions"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Loading {target} from {mission}...", total=None)
        
        try:
            if mission == "TESS":
                # Parse sectors
                if sectors and sectors.lower() != "all":
                    sector_list = [int(s.strip()) for s in sectors.split(",")]
                else:
                    sector_list = "all"
                
                lc_collection = load_tess_data_advanced(
                    target=target,
                    sectors=sector_list,
                    author=author,
                    cadence=cadence
                )
                
                progress.update(task, completed=100, description="Data loaded successfully")
                
                # Display summary
                console.print(f"\n[bold green]✓ Loaded {len(lc_collection)} light curves[/bold green]")
                
                for i, lc in enumerate(lc_collection):
                    sector = lc.meta.get('SECTOR', 'Unknown')
                    n_points = len(lc.time)
                    duration = lc.time[-1] - lc.time[0] if n_points > 1 else 0
                    console.print(f"  Sector {sector}: {n_points} points, {duration:.2f} days")
                
                # Save if requested
                if output:
                    # Convert to numpy arrays
                    all_time = []
                    all_flux = []
                    for lc in lc_collection:
                        all_time.extend(lc.time.value)
                        all_flux.extend(lc.flux.value)
                    
                    data = np.column_stack([all_time, all_flux])
                    np.savetxt(output, data, header="time,flux", delimiter=",")
                    console.print(f"[bold]Data saved to:[/bold] {output}")
            
            elif mission == "ground":
                # Load ground-based data
                data_dict = load_ground_based_data(target)
                console.print(f"[bold green]✓ Loaded ground-based data[/bold green]")
                console.print(f"  Keys: {', '.join(data_dict.keys())}")
                
                if output:
                    export_transit_results(data_dict, output, format="json")
                    console.print(f"[bold]Data saved to:[/bold] {output}")
            
            else:
                console.print(f"[yellow]Mission {mission} support coming soon[/yellow]")
        
        except Exception as e:
            console.print(f"[bold red]Error loading data:[/bold red] {e}")
            if ctx.obj["DEBUG"]:
                import traceback
                console.print(traceback.format_exc())

@data.command(name="preprocess")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--detrend", type=click.Choice(["gp", "pca", "poly", "savgol"]),
              default="gp", help="Detrending method")
@click.option("--remove-outliers", is_flag=True, help="Remove outliers")
@click.option("--normalize", is_flag=True, default=True, help="Normalize flux")
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def preprocess_data(ctx, input_file, detrend, remove_outliers, normalize, output):
    """Preprocess light curve data"""
    # Load data
    data = np.loadtxt(input_file, delimiter=",")
    time, flux = data[:, 0], data[:, 1]
    
    console.print(f"[bold]Preprocessing {len(time)} data points...[/bold]")
    
    # Quality check
    quality = check_data_quality(time, flux)
    console.print(f"  Quality check: {quality['n_nans']} NaNs, "
                  f"noise: {quality.get('noise_ppm', 0):.0f} ppm")
    
    # Remove NaNs
    mask = np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    
    if remove_outliers:
        outliers = tk.utils.detect_outliers_modified_zscore(flux)
        time, flux = time[~outliers], flux[~outliers]
        console.print(f"  Removed {np.sum(outliers)} outliers")
    
    # Normalize
    if normalize:
        flux = flux / np.median(flux)
        console.print("  Normalized flux to median=1")
    
    # Detrend
    if detrend == "gp":
        console.print("  Detrending with Gaussian Process...")
        flux_detrended, trend, gp = detrend_light_curve_gp(time, flux)
        console.print(f"  GP kernel: {gp.kernel_}")
    elif detrend == "pca":
        console.print("  Removing systematics with PCA...")
        results = remove_systematics_pca(time, flux)
        flux_detrended = results['corrected_flux']
        console.print(f"  Explained variance: {results['explained_variance'][:3]}")
    else:
        flux_detrended = flux  # No detrending
    
    # Save results
    if output:
        data_out = np.column_stack([time, flux_detrended])
        np.savetxt(output, data_out, header="time,flux", delimiter=",")
        console.print(f"[bold green]✓ Preprocessed data saved to:[/bold green] {output}")
    
    console.print(f"[bold]Preprocessing complete.[/bold]")
    console.print(f"  Original: {len(data)} points")
    console.print(f"  Clean: {len(time)} points")
    console.print(f"  Time span: {time[-1] - time[0]:.2f} days")

# ==================== ANALYSIS COMMANDS ====================

@cli.group()
def analyze():
    """Transit analysis and detection commands"""
    pass

@analyze.command(name="detect")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--method", "-m", type=click.Choice(["bls", "multiple", "gls", "pdm"]),
              multiple=True, default=["bls"], help="Detection method(s)")
@click.option("--min-period", type=float, default=0.5, help="Minimum period (days)")
@click.option("--max-period", type=float, default=100.0, help="Maximum period (days)")
@click.option("--n-periods", type=int, default=10000, help="Number of period trials")
@click.option("--fap-threshold", type=float, default=0.01,
              help="False alarm probability threshold")
@click.option("--output", "-o", type=click.Path(), help="Output results file")
@click.pass_context
def detect_transits(ctx, input_file, method, min_period, max_period,
                    n_periods, fap_threshold, output):
    """Detect transits in light curve data"""
    # Load data
    data = np.loadtxt(input_file, delimiter=",")
    time, flux = data[:, 0], data[:, 1]
    
    console.print(f"[bold]Analyzing {len(time)} data points...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Run detection
        if "multiple" in method or len(method) > 1:
            task = progress.add_task("Running multiple detection methods...", total=None)
            results = find_transits_multiple_methods(
                time, flux,
                min_period=min_period,
                max_period=max_period,
                n_periods=n_periods,
                methods=list(method) if "multiple" not in method else ["bls", "gls", "pdm"]
            )
            progress.update(task, completed=100, description="Detection complete")
            
            # Display consensus results
            consensus = results.get("consensus", {})
            if consensus:
                console.print(f"\n[bold green]✓ Consensus Detection[/bold green]")
                console.print(f"  Period: {consensus.get('period', 'N/A'):.6f} days")
                console.print(f"  Method agreement: {consensus.get('method_agreement', 0)} methods")
                console.print(f"  Is harmonic: {consensus.get('is_harmonic', False)}")
        
        else:
            # Single method
            method_name = method[0]
            task = progress.add_task(f"Running {method_name.upper()}...", total=None)
            
            if method_name == "bls":
                result = find_transits_bls_advanced(
                    time, flux,
                    min_period=min_period,
                    max_period=max_period,
                    n_periods=n_periods
                )
            elif method_name == "gls":
                result = tk.core.find_period_gls(time, flux)
            elif method_name == "pdm":
                result = tk.core.find_period_pdm(time, flux)
            
            progress.update(task, completed=100, description=f"{method_name.upper()} complete")
            
            # Display results
            console.print(f"\n[bold green]✓ {method_name.upper()} Results[/bold green]")
            console.print(f"  Period: {result.get('period', 'N/A'):.6f} days")
            console.print(f"  Depth: {result.get('depth', 0)*1e6:.1f} ppm")
            console.print(f"  SNR: {result.get('snr', 0):.1f}")
            if 'fap' in result:
                console.print(f"  FAP: {result.get('fap', 1):.2e}")
            
            results = {"primary": result}
        
        # Validation
        task = progress.add_task("Validating detection...", total=None)
        validation = results.get("validation", {})
        if validation:
            console.print(f"\n[bold]Validation Results[/bold]")
            console.print(f"  Passed: {validation.get('passed', False)}")
            console.print(f"  Odd-even p-value: {validation.get('odd_even', {}).get('p_value', 1):.3f}")
        
        progress.update(task, completed=100, description="Validation complete")
    
    # Save results
    if output:
        export_transit_results(results, output)
        console.print(f"\n[bold]Results saved to:[/bold] {output}")

@analyze.command(name="mcmc")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--period", type=float, required=True, help="Initial period guess (days)")
@click.option("--t0", type=float, required=True, help="Initial transit time guess")
@click.option("--duration", type=float, required=True, help="Initial duration guess (days)")
@click.option("--depth", type=float, required=True, help="Initial depth guess")
@click.option("--n-walkers", type=int, default=32, help="Number of MCMC walkers")
@click.option("--n-steps", type=int, default=2000, help="Number of MCMC steps")
@click.option("--burnin", type=int, default=500, help="Burn-in steps")
@click.option("--output", "-o", type=click.Path(), help="Output directory for results")
@click.pass_context
def run_mcmc(ctx, input_file, period, t0, duration, depth, n_walkers, n_steps, burnin, output):
    """Run MCMC to estimate transit parameters with uncertainties"""
    # Load data
    data = np.loadtxt(input_file, delimiter=",")
    time, flux = data[:, 0], data[:, 1]
    flux_err = np.ones_like(flux) * np.std(flux) / np.sqrt(len(flux))
    
    console.print(f"[bold]Running MCMC with {n_walkers} walkers, {n_steps} steps...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running MCMC...", total=n_steps)
        
        # Run MCMC
        samples, errors = estimate_parameters_mcmc(
            time, flux, flux_err,
            period, t0, duration, depth,
            n_walkers=n_walkers,
            n_steps=n_steps,
            burnin=burnin
        )
        
        # Update progress
        for i in range(n_steps):
            progress.update(task, advance=1)
    
    # Display results
    console.print(f"\n[bold green]✓ MCMC Results[/bold green]")
    console.print(f"  Period: {period:.6f} ± {errors.get('period_err', 0):.6f} days")
    console.print(f"  T0: {t0:.6f} ± {errors.get('t0_err', 0):.6f}")
    console.print(f"  Duration: {duration:.6f} ± {errors.get('duration_err', 0):.6f} days")
    console.print(f"  Depth: {depth:.6f} ± {errors.get('depth_err', 0):.6f}")
    
    # Save results
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save samples
        np.save(output_dir / "mcmc_samples.npy", samples)
        
        # Save parameter errors
        with open(output_dir / "parameter_errors.json", "w") as f:
            json.dump(errors, f, indent=2)
        
        # Create corner plot
        fig = plot_mcmc_corner(
            samples,
            param_names=["period", "t0", "duration", "depth"],
            truths=[period, t0, duration, depth]
        )
        fig.savefig(output_dir / "corner_plot.pdf", dpi=300, bbox_inches="tight")
        
        console.print(f"\n[bold]MCMC results saved to:[/bold] {output_dir}")

@analyze.command(name="ttv")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--period", type=float, required=True, help="Orbital period (days)")
@click.option("--t0", type=float, required=True, help="Reference transit time")
@click.option("--duration", type=float, required=True, help="Transit duration (days)")
@click.option("--min-epoch", type=int, help="Minimum epoch to analyze")
@click.option("--max-epoch", type=int, help="Maximum epoch to analyze")
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def analyze_ttvs(ctx, input_file, period, t0, duration, min_epoch, max_epoch, output):
    """Analyze Transit Timing Variations"""
    # Load data
    data = np.loadtxt(input_file, delimiter=",")
    time, flux = data[:, 0], data[:, 1]
    
    console.print(f"[bold]Analyzing TTVs for period={period:.6f} days...[/bold]")
    
    ttv_results = measure_transit_timing_variations(
        time, flux, period, t0, duration
    )
    
    if ttv_results.get("ttvs_detected", False):
        console.print(f"\n[bold green]✓ TTVs Detected![/bold green]")
        console.print(f"  p-value: {ttv_results.get('p_value', 1):.3e}")
        console.print(f"  RMS TTV: {ttv_results.get('rms_ttv', 0)*24*60:.1f} minutes")
        console.print(f"  Number of epochs: {len(ttv_results.get('ttvs', []))}")
        
        if not np.isnan(ttv_results.get('ttv_period', np.nan)):
            console.print(f"  TTV period: {ttv_results.get('ttv_period', 0):.1f} orbits")
            console.print(f"  TTV amplitude: {ttv_results.get('ttv_amplitude', 0)*24*60:.1f} minutes")
    else:
        console.print(f"\n[yellow]No significant TTVs detected[/yellow]")
        console.print(f"  p-value: {ttv_results.get('p_value', 1):.3f}")
    
    # Save results
    if output:
        export_transit_results(ttv_results, output, format="json")
        console.print(f"\n[bold]TTV results saved to:[/bold] {output}")

# ==================== SIMULATION COMMANDS ====================

@cli.group()
def simulate():
    """Transit simulation commands"""
    pass

@simulate.command(name="create")
@click.option("--period", type=float, default=10.0, help="Orbital period (days)")
@click.option("--depth", type=float, default=0.01, help="Transit depth")
@click.option("--duration", type=float, default=0.1, help="Transit duration (days)")
@click.option("--baseline", type=float, default=30.0, help="Observation baseline (days)")
@click.option("--n-points", type=int, default=1000, help="Number of data points")
@click.option("--noise", type=float, default=0.001, help="Noise level")
@click.option("--limb-darkening", "-ld", type=(float, float), default=(0.1, 0.3),
              help="Limb darkening coefficients u1 u2")
@click.option("--output", "-o", type=click.Path(), help="Output file")
def create_simulation(period, depth, duration, baseline, n_points, noise,
                      limb_darkening, output):
    """Create synthetic transit light curve"""
    # Generate time array
    time = np.linspace(0, baseline, n_points)
    
    # Generate transit signal with Mandel & Agol model
    rprs = np.sqrt(depth)  # Planet-to-star radius ratio
    aRs = 10.0  # Scaled semi-major axis (placeholder)
    
    flux_clean = generate_transit_signal_mandel_agol(
        time,
        period=period,
        t0=period/2,
        rprs=rprs,
        aRs=aRs,
        u1=limb_darkening[0],
        u2=limb_darkening[1]
    )
    
    # Add noise
    flux = tk.add_noise(flux_clean, noise_level=noise)
    
    # Save to file
    if output:
        data = np.column_stack([time, flux, flux_clean])
        np.savetxt(output, data, header="time,flux,flux_clean", delimiter=",")
        console.print(f"[bold green]✓ Synthetic light curve saved to:[/bold green] {output}")
        console.print(f"  Period: {period} days")
        console.print(f"  Depth: {depth*1e6:.1f} ppm")
        console.print(f"  Duration: {duration*24:.1f} hours")
        console.print(f"  Noise: {noise*1e6:.0f} ppm")
    else:
        # Display in terminal
        table = Table(title="Synthetic Light Curve Parameters", box=box.ROUNDED)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Period", f"{period:.4f} days")
        table.add_row("Depth", f"{depth*1e6:.1f} ppm")
        table.add_row("Duration", f"{duration*24:.2f} hours")
        table.add_row("Noise", f"{noise*1e6:.0f} ppm")
        table.add_row("N points", f"{n_points}")
        table.add_row("Baseline", f"{baseline:.1f} days")
        
        console.print(table)

@simulate.command(name="recovery")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--n-trials", type=int, default=100, help="Number of injection trials")
@click.option("--noise-level", type=float, default=0.001, help="Injection noise level")
@click.option("--output", "-o", type=click.Path(), help="Output results file")
@click.pass_context
def injection_recovery(ctx, input_file, n_trials, noise_level, output):
    """Perform injection-recovery test"""
    # Load data to get time array
    data = np.loadtxt(input_file, delimiter=",")
    time = data[:, 0]
    
    # Define injection parameters (use values from file if available, or defaults)
    if data.shape[1] >= 4:  # Has clean flux column
        # Estimate parameters from clean signal
        flux_clean = data[:, 2]
        # Simple parameter estimation (in practice, use proper fitting)
        period = 10.0  # Default
        depth = 1 - np.min(flux_clean)
        duration = 0.1  # Default
        t0 = period / 2  # Default
    else:
        period = 10.0
        depth = 0.01
        duration = 0.1
        t0 = period / 2
    
    injection_params = TransitParameters(
        period=period,
        t0=t0,
        depth=depth,
        duration=duration
    )
    
    console.print(f"[bold]Running injection-recovery test ({n_trials} trials)...[/bold]")
    
    recovery_results = perform_injection_recovery_test(
        time, injection_params,
        n_trials=n_trials,
        noise_level=noise_level
    )
    
    recovery_rate = recovery_results.get("recovery_rate", 0)
    
    console.print(f"\n[bold]Injection-Recovery Results[/bold]")
    console.print(f"  Recovery rate: {recovery_rate*100:.1f}%")
    console.print(f"  Recovered: {recovery_results.get('n_recovered', 0)}/{n_trials}")
    console.print(f"  Detection efficiency: {recovery_results.get('detection_efficiency', 0)*100:.1f}%")
    
    if recovery_rate > 0.8:
        console.print(f"[bold green]✓ Excellent recovery rate[/bold green]")
    elif recovery_rate > 0.5:
        console.print(f"[yellow]Moderate recovery rate[/yellow]")
    else:
        console.print(f"[red]Poor recovery rate - consider improving detection[/red]")
    
    # Save results
    if output:
        export_transit_results(recovery_results, output, format="json")
        console.print(f"\n[bold]Results saved to:[/bold] {output}")

# ==================== VISUALIZATION COMMANDS ====================

@cli.group()
def plot():
    """Visualization commands"""
    pass

@plot.command(name="summary")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--period", type=float, help="Orbital period (days)")
@click.option("--t0", type=float, help="Transit time")
@click.option("--duration", type=float, help="Transit duration (days)")
@click.option("--depth", type=float, help="Transit depth")
@click.option("--output", "-o", type=click.Path(), help="Output figure file")
@click.option("--dpi", type=int, default=300, help="Figure DPI")
@click.option("--style", type=click.Choice(["nature", "aas", "default"]),
              default="aas", help="Plot style")
@click.pass_context
def plot_summary(ctx, input_file, period, t0, duration, depth, output, dpi, style):
    """Create publication-quality transit summary plot"""
    # Load data
    data = np.loadtxt(input_file, delimiter=",")
    time, flux = data[:, 0], data[:, 1]
    
    # Create parameters object
    params = TransitParameters(
        period=period or 10.0,
        t0=t0 or (time[0] + (time[-1] - time[0]) / 2),
        duration=duration or 0.1,
        depth=depth or 0.01
    )
    
    console.print(f"[bold]Creating summary plot...[/bold]")
    
    # Setup publication style
    setup_publication_style(style=style, dpi=dpi)
    
    # Create figure
    fig = create_transit_report_figure(time, flux, params)
    
    # Save or show
    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        console.print(f"[bold green]✓ Figure saved to:[/bold green] {output}")
    else:
        plt.show()
    
    plt.close(fig)

@plot.command(name="periodogram")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--method", type=click.Choice(["bls", "gls", "pdm", "all"]),
              default="bls", help="Periodogram method")
@click.option("--min-period", type=float, default=0.5, help="Minimum period (days)")
@click.option("--max-period", type=float, default=100.0, help="Maximum period (days)")
@click.option("--output", "-o", type=click.Path(), help="Output figure file")
@click.pass_context
def plot_periodogram(ctx, input_file, method, min_period, max_period, output):
    """Plot periodogram for transit detection"""
    # Load data
    data = np.loadtxt(input_file, delimiter=",")
    time, flux = data[:, 0], data[:, 1]
    
    console.print(f"[bold]Creating {method.upper()} periodogram...[/bold]")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if method in ["bls", "all"]:
        # Run BLS
        result = find_transits_bls_advanced(
            time, flux,
            min_period=min_period,
            max_period=max_period
        )
        
        periods = result.get("all_periods", [])
        power = result.get("all_powers", result.get("all_power", []))
        
        ax.plot(periods, power, label="BLS", linewidth=2)
        
        # Mark best period
        best_period = result.get("period", 0)
        if best_period > 0:
            ax.axvline(best_period, color='r', linestyle='--', alpha=0.7,
                      label=f'Best: {best_period:.4f} d')
    
    if method in ["all"]:
        # Add other methods if requested
        # (Implementation for GLS and PDM would go here)
        pass
    
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Power")
    ax.set_title(f"Periodogram ({method.upper()})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if max_period / min_period > 100:
        ax.set_xscale("log")
    
    # Save or show
    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        console.print(f"[bold green]✓ Periodogram saved to:[/bold green] {output}")
    else:
        plt.show()
    
    plt.close(fig)

# ==================== VALIDATION COMMANDS ====================

@cli.group()
def validate():
    """Validation and testing commands"""
    pass

@validate.command(name="significance")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--period", type=float, required=True, help="Detected period (days)")
@click.option("--n-shuffles", type=int, default=1000, help="Number of shuffles")
@click.option("--output", "-o", type=click.Path(), help="Output results file")
@click.pass_context
def validate_significance(ctx, input_file, period, n_shuffles, output):
    """Calculate detection significance via data shuffling"""
    # Load data
    data = np.loadtxt(input_file, delimiter=",")
    time, flux = data[:, 0], data[:, 1]
    
    console.print(f"[bold]Calculating significance ({n_shuffles} shuffles)...[/bold]")
    
    # First run BLS to get power
    result = find_transits_bls_advanced(
        time, flux,
        min_period=period * 0.9,
        max_period=period * 1.1
    )
    
    # Calculate significance
    significance = calculate_detection_significance(result, n_shuffles=n_shuffles)
    
    p_value = significance.get("p_value", 1)
    sigma = significance.get("significance_sigma", 0)
    
    console.print(f"\n[bold]Significance Results[/bold]")
    console.print(f"  p-value: {p_value:.3e}")
    console.print(f"  Significance: {sigma:.1f}σ")
    console.print(f"  Best power: {significance.get('best_power', 0):.3f}")
    
    if p_value < 0.01:
        console.print(f"[bold green]✓ Detection is statistically significant[/bold green]")
    elif p_value < 0.05:
        console.print(f"[yellow]Detection is marginally significant[/yellow]")
    else:
        console.print(f"[red]Detection is not statistically significant[/red]")
    
    # Save results
    if output:
        export_transit_results(significance, output, format="json")
        console.print(f"\n[bold]Results saved to:[/bold] {output}")

@validate.command(name="parameters")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--period", type=float, required=True, help="Period (days)")
@click.option("--t0", type=float, required=True, help="Transit time")
@click.option("--duration", type=float, required=True, help="Duration (days)")
@click.option("--depth", type=float, required=True, help="Depth")
@click.pass_context
def validate_parameters(ctx, input_file, period, t0, duration, depth):
    """Validate transit parameters against physical limits"""
    # Load data
    data = np.loadtxt(input_file, delimiter=",")
    time, flux = data[:, 0], data[:, 1]
    
    params = TransitParameters(
        period=period,
        t0=t0,
        duration=duration,
        depth=depth
    )
    
    validation = validate_transit_parameters(params, time, flux)
    
    console.print(f"\n[bold]Parameter Validation[/bold]")
    
    table = Table(box=box.SIMPLE)
    table.add_column("Check", style="cyan")
    table.add_column("Status", justify="right")
    table.add_column("Description")
    
    for key, value in validation.items():
        if not key.startswith("_") and isinstance(value, bool):
            status = "✅" if value else "❌"
            style = "green" if value else "red"
            
            # Description mapping
            desc_map = {
                "period_positive": "Period > 0",
                "period_realistic": "0.1 < Period < 1000 days",
                "duration_positive": "Duration > 0",
                "duration_lt_period": "Duration < Period",
                "duration_realistic": "1-15 hours typical",
                "depth_positive": "Depth > 0",
                "depth_lt_one": "Depth < 1",
                "depth_realistic": "Depth < 3% typical",
                "t0_in_range": "T0 within data range",
                "snr_valid": "SNR > 0",
                "fap_valid": "0 ≤ FAP ≤ 1",
            }
            
            table.add_row(key.replace("_", " ").title(),
                         f"[{style}]{status}[/{style}]",
                         desc_map.get(key, ""))
    
    console.print(table)
    
    if validation.get("all_passed", False):
        console.print(f"[bold green]✓ All parameter checks passed[/bold green]")
    else:
        console.print(f"[red]Some parameter checks failed[/red]")

# ==================== MAIN ====================

def main():
    """Main CLI entry point"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        if "--debug" in sys.argv:
            import traceback
            console.print(f"[bold red]Error:[/bold red] {e}")
            console.print(traceback.format_exc())
        else:
            console.print(f"[bold red]Error:[/bold red] {e}")
            console.print("Use --debug for detailed traceback")
        sys.exit(1)

if __name__ == "__main__":
    main()
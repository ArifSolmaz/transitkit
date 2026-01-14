"""Command line interface for TransitKit."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from transitkit import __version__


def load_target(target_id: str, mission: str, sector: Optional[int] = None):
    """Load a light curve target (placeholder implementation)."""
    return {"target_id": target_id, "mission": mission, "sector": sector}


def fit_transit(path: str, method: str = "least_squares"):
    """Fit a transit model to a file (placeholder implementation)."""
    return {"path": path, "method": method}


class BatchProcessor:
    """Process batches of targets (placeholder implementation)."""

    def __init__(self, output_dir: str, n_workers: int = 1, overwrite: bool = False) -> None:
        self.output_dir = output_dir
        self.n_workers = n_workers
        self.overwrite = overwrite

    def process(self, targets_path: str):
        return []


@click.group()
@click.version_option(__version__, prog_name="transitkit")
def main() -> None:
    """TransitKit command line interface."""


@main.command()
@click.argument("target_id")
@click.option("--mission", default="TESS", show_default=True)
@click.option("--sector", type=int)
def load(target_id: str, mission: str, sector: Optional[int]) -> None:
    """Load a target light curve."""
    load_target(target_id, mission=mission, sector=sector)


@main.command()
@click.argument("input_path")
@click.option("--method", default="least_squares", show_default=True)
@click.option("--output", type=click.Path())
def fit(input_path: str, method: str, output: Optional[str]) -> None:
    """Fit a transit model to an input file."""
    fit_transit(input_path, method=method)
    if output:
        Path(output).write_text("{}")


@main.command()
@click.argument("targets_file")
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--workers", default=1, type=int, show_default=True)
@click.option("--overwrite", is_flag=True, default=False)
def batch(targets_file: str, output_dir: str, workers: int, overwrite: bool) -> None:
    """Process a batch of targets."""
    processor = BatchProcessor(output_dir=output_dir, n_workers=workers, overwrite=overwrite)
    processor.process(targets_file)

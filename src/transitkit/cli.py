"""Command line interface for TransitKit"""

import click

@click.group()
def cli():
    """TransitKit: Exoplanet Transit Light Curve Analysis Toolkit"""
    pass

@cli.command()
def version():
    """Show version"""
    from transitkit import __version__
    click.echo(f"TransitKit v{__version__}")

@cli.command()
@click.argument("target_id")
@click.option("--mission", default="TESS", show_default=True, help="Mission name")
def load(target_id, mission):
    """Load data for a target"""
    click.echo(f"Loading {target_id} from {mission}...")
    click.echo("(This is a placeholder - real loading coming soon!)")

def main():
    cli()

if __name__ == "__main__":
    main()

"""Command line entrypoint for the Traffic Byte-BERT project."""

import typer

from traffic_bert import __version__

app = typer.Typer(help="Traffic Byte-BERT experiment toolkit.")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False,
    help="metinspect: inspection + metrology + uncertainty pipeline",
)

DEFAULT_CONFIG = Path("configs/default.yaml")


@app.command()
def download(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    _ = config
    typer.echo("download: placeholder")


@app.command()
def train(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    _ = config
    typer.echo("train: placeholder")


@app.command()
def eval(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    _ = config
    typer.echo("eval: placeholder")

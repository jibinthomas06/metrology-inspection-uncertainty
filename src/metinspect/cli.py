from __future__ import annotations

from pathlib import Path

import typer

from metinspect.config import load_config

app = typer.Typer(
    add_completion=False,
    help="metinspect: inspection + metrology + uncertainty pipeline",
)

DEFAULT_CONFIG = Path("configs/default.yaml")


@app.command()
def download(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    """
    Milestone 0: validate dataset placement + show available categories.
    """
    cfg = load_config(config)
    from metinspect.data.mvtec import list_categories, validate_mvtec_root

    validate_mvtec_root(cfg.mvtec_dir)
    cats = list_categories(cfg.mvtec_dir)

    typer.echo("MVTec AD looks OK.")
    typer.echo(f"Root: {cfg.mvtec_dir}")
    typer.echo(f'Found categories ({len(cats)}): {", ".join(cats)}')


@app.command()
def train(config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c")) -> None:
    _ = config
    typer.echo("train: placeholder (Milestone 1)")


@app.command()
def eval(config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c")) -> None:
    _ = config
    typer.echo("eval: placeholder (Milestone 1)")

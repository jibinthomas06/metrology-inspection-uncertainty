from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import typer

from metinspect.config import load_config
from metinspect.data.mvtec import (
    index_test_split,
    iter_train_good,
    list_categories,
    validate_mvtec_root,
)
from metinspect.image_io import load_image_tensor, read_mask01, read_rgb, resize_rgb
from metinspect.metrics import image_auroc, pixel_auroc
from metinspect.models.patchcore import PatchCore
from metinspect.viz import save_overlay_figure

app = typer.Typer(
    add_completion=False,
    help="metinspect: inspection + metrology + uncertainty pipeline",
)

DEFAULT_CONFIG = Path("configs/default.yaml")


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)


@app.command()
def download(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    cfg = load_config(config)
    validate_mvtec_root(cfg.mvtec_dir)
    cats = list_categories(cfg.mvtec_dir)
    typer.echo("MVTec AD looks OK.")
    typer.echo(f"Root: {cfg.mvtec_dir}")
    typer.echo(f'Found categories ({len(cats)}): {", ".join(cats)}')


@app.command()
def train(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    backbone: str = typer.Option("resnet18", "--backbone"),
    max_patches: int = typer.Option(20000, "--max-patches"),
) -> None:
    cfg = load_config(config)
    _seed_everything(cfg.seed)
    validate_mvtec_root(cfg.mvtec_dir)

    train_paths = list(iter_train_good(cfg.mvtec_dir, cfg.category))
    if not train_paths:
        raise RuntimeError("No training images found.")

    typer.echo(
        f"Training PatchCore baseline on category={cfg.category} "
        f"with {len(train_paths)} train images"
    )
    pc = PatchCore(backbone=backbone, image_size=cfg.image_size, device=cfg.device, nn_k=1)

    tensors = [load_image_tensor(p, cfg.image_size) for p in train_paths]
    pc.fit_from_tensors(tensors, max_patches=max_patches)

    model_path = cfg.reports_dir / "models" / f"patchcore_{cfg.category}_{backbone}.pt"
    pc.save(model_path)
    typer.echo(f"Saved model: {model_path}")


@app.command()
def eval(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    backbone: str = typer.Option("resnet18", "--backbone"),
    gallery_n: int = typer.Option(12, "--gallery-n"),
) -> None:
    cfg = load_config(config)
    _seed_everything(cfg.seed)
    validate_mvtec_root(cfg.mvtec_dir)

    model_path = cfg.reports_dir / "models" / f"patchcore_{cfg.category}_{backbone}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run `metinspect train` first.")

    pc = PatchCore.load(model_path, device=cfg.device)
    samples = index_test_split(cfg.mvtec_dir, cfg.category)
    if not samples:
        raise RuntimeError("No test samples found.")

    y_true, y_score = [], []
    masks, maps = [], []
    gallery = []

    typer.echo(f"Evaluating on category={cfg.category} with {len(samples)} test images")

    for s in samples:
        x = load_image_tensor(s.image_path, cfg.image_size)
        img_score, score_map = pc.score(x)
        heat = cv2.resize(
            score_map.astype(np.float32),
            (cfg.image_size, cfg.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        if s.mask_path is None:
            m01 = np.zeros((cfg.image_size, cfg.image_size), dtype=np.uint8)
        else:
            m01 = read_mask01(s.mask_path, cfg.image_size)

        y_true.append(int(s.label))
        y_score.append(float(img_score))
        masks.append(m01)
        maps.append(heat)

        img_rgb = resize_rgb(read_rgb(s.image_path), cfg.image_size)
        title = f"{cfg.category}/{s.defect_type} score={img_score:.4f}"
        gallery.append((float(img_score), img_rgb, heat, m01, title))

    y_true_np = np.array(y_true, dtype=np.int32)
    y_score_np = np.array(y_score, dtype=np.float32)

    img_auc = image_auroc(y_true_np, y_score_np)
    pix_auc = pixel_auroc(masks, maps)

    out_metrics = {
        "category": cfg.category,
        "backbone": backbone,
        "n_test": int(len(samples)),
        "image_auroc": float(img_auc),
        "pixel_auroc": float(pix_auc),
    }

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.reports_dir / f"metrics_patchcore_{cfg.category}_{backbone}.json"
    metrics_path.write_text(json.dumps(out_metrics, indent=2), encoding="utf-8")

    typer.echo(f"image AUROC: {img_auc:.4f}")
    typer.echo(f"pixel AUROC: {pix_auc:.4f}")
    typer.echo(f"Saved metrics: {metrics_path}")

    gallery_sorted = sorted(gallery, key=lambda t: t[0], reverse=True)[: int(gallery_n)]
    fig_dir = cfg.reports_dir / "figures" / f"gallery_{cfg.category}_{backbone}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for i, (_sc, img_rgb, heat, m01, title) in enumerate(gallery_sorted):
        out_path = fig_dir / f"{i:03d}.png"
        save_overlay_figure(out_path, img_rgb, heat, m01, title)

    typer.echo(f"Saved gallery: {fig_dir}")

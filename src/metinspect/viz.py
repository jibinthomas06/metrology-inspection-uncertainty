from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def save_overlay_figure(
    out_path: Path,
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    mask01: np.ndarray,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hm01 = normalize01(heatmap)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(image_rgb)
    ax1.set_title("image")
    ax1.axis("off")

    ax2.imshow(image_rgb)
    ax2.imshow(hm01, alpha=0.5)
    ax2.set_title("heatmap")
    ax2.axis("off")

    ax3.imshow(mask01, cmap="gray", vmin=0, vmax=1)
    ax3.set_title("mask")
    ax3.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

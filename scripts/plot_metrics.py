from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    reports = Path("reports")
    csv_path = reports / "metrics_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

    df = pd.read_csv(csv_path)
    # keep only rows with numeric metrics
    df = df.dropna(subset=["image_auroc", "pixel_auroc"]).copy()

    # Sort by image AUROC (ascending)
    df = df.sort_values("image_auroc", ascending=True)

    out_dir = reports / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot: image AUROC by category
    plt.figure(figsize=(10, 6))
    plt.bar(df["category"], df["image_auroc"])
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Image AUROC")
    plt.title("PatchCore (resnet18): Image AUROC by MVTec category")
    plt.tight_layout()
    out1 = out_dir / "image_auroc_by_category.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    # Plot: pixel AUROC by category
    df2 = df.sort_values("pixel_auroc", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.bar(df2["category"], df2["pixel_auroc"])
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Pixel AUROC")
    plt.title("PatchCore (resnet18): Pixel AUROC by MVTec category")
    plt.tight_layout()
    out2 = out_dir / "pixel_auroc_by_category.png"
    plt.savefig(out2, dpi=200)
    plt.close()

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

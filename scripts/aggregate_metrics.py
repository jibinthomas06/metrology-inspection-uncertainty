from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REPORTS_DIR = Path("reports")
PATTERN = "metrics_patchcore_*_*.json"  # e.g. metrics_patchcore_bottle_resnet18.json


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def _infer_from_filename(p: Path) -> tuple[str | None, str | None]:
    # expected: metrics_patchcore_<category>_<backbone>.json
    stem = p.stem
    prefix = "metrics_patchcore_"
    if not stem.startswith(prefix):
        return None, None
    rest = stem[len(prefix) :]
    parts = rest.split("_")
    if len(parts) < 2:
        return None, None
    category = parts[0]
    backbone = "_".join(parts[1:])
    return category, backbone


def _extract_metrics(d: dict[str, Any]) -> tuple[float | None, float | None]:
    """
    Be tolerant to different key names.
    Your current eval prints:
      image AUROC: ...
      pixel AUROC: ...
    and likely stores keys like image_auroc / pixel_auroc in JSON.
    """
    candidates_img = ["image_auroc", "img_auroc", "image_auc", "auroc_image", "imageAUROC"]
    candidates_pix = ["pixel_auroc", "px_auroc", "pixel_auc", "auroc_pixel", "pixelAUROC"]

    img = None
    pix = None

    for k in candidates_img:
        if k in d:
            img = _as_float(d.get(k))
            break

    for k in candidates_pix:
        if k in d:
            pix = _as_float(d.get(k))
            break

    return img, pix


def main() -> int:
    files = sorted(REPORTS_DIR.glob(PATTERN))
    if not files:
        print(f"No metric JSON files found: {REPORTS_DIR / PATTERN}")
        return 1

    rows: list[dict[str, Any]] = []

    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skip (cannot read JSON): {p}  ({e})")
            continue

        category, backbone = None, None

        # Try config-shaped keys first (if present)
        # (Your Config class shows cfg.category/backbone, but JSON may vary.)
        for k in ["category", "mvtec_category", "object", "class_name"]:
            if k in data and isinstance(data[k], str):
                category = data[k]
                break

        for k in ["backbone", "encoder", "feature_backbone"]:
            if k in data and isinstance(data[k], str):
                backbone = data[k]
                break

        # Fallback: infer from filename
        f_cat, f_bb = _infer_from_filename(p)
        category = category or f_cat
        backbone = backbone or f_bb

        image_auroc, pixel_auroc = _extract_metrics(data)

        rows.append(
            {
                "category": category,
                "backbone": backbone,
                "image_auroc": image_auroc,
                "pixel_auroc": pixel_auroc,
                "metrics_file": p.as_posix(),
            }
        )

    # Write CSV
    out_csv = REPORTS_DIR / "metrics_table.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["category", "backbone", "image_auroc", "pixel_auroc", "metrics_file"],
        )
        w.writeheader()
        for r in sorted(rows, key=lambda x: (str(x["backbone"]), str(x["category"]))):
            w.writerow(r)

    # Write short markdown summary
    out_md = REPORTS_DIR / "summary_eval.md"

    img_vals = [r["image_auroc"] for r in rows if isinstance(r["image_auroc"], (int, float))]
    pix_vals = [r["pixel_auroc"] for r in rows if isinstance(r["pixel_auroc"], (int, float))]

    def _fmt(x: Any) -> str:
        if x is None:
            return "NA"
        if isinstance(x, (int, float)):
            return f"{x:.4f}"
        return str(x)

    lines: list[str] = []
    lines.append("# Evaluation summary\n")
    lines.append(f"- Metric files parsed: {len(rows)}")
    lines.append(f"- CSV: `{out_csv.as_posix()}`")
    lines.append(f"- Generated: `{out_md.as_posix()}`\n")

    if img_vals:
        lines.append(
            f"- image AUROC: mean={sum(img_vals)/len(img_vals):.4f}  "
            f"min={min(img_vals):.4f}  max={max(img_vals):.4f}"
        )
    if pix_vals:
        lines.append(
            f"- pixel AUROC: mean={sum(pix_vals)/len(pix_vals):.4f}  "
            f"min={min(pix_vals):.4f}  max={max(pix_vals):.4f}"
        )

    lines.append("\n## Per-category\n")
    lines.append("| category | backbone | image_auroc | pixel_auroc |")
    lines.append("|---|---|---:|---:|")
    for r in sorted(rows, key=lambda x: (str(x["backbone"]), str(x["category"]))):
        lines.append(
            f"| {_fmt(r['category'])} | {_fmt(r['backbone'])} | {_fmt(r['image_auroc'])} | {_fmt(r['pixel_auroc'])} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

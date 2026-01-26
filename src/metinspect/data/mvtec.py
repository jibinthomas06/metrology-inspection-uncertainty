from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DEFAULT_MVTEC_ROOT = Path("data") / "mvtec_ad"
ENV_MVTEC_ROOT = "METINSPECT_DATA_ROOT"


def get_mvtec_root() -> Path:
    """
    Resolve MVTec dataset root.

    Priority:
      1) env var METINSPECT_DATA_ROOT
      2) default: data/mvtec_ad (relative to repo root)

    Examples (PowerShell):
      $env:METINSPECT_DATA_ROOT = "D:\\datasets\\mvtec_anomaly_detection"
    """
    env = os.environ.get(ENV_MVTEC_ROOT, "").strip()
    if env:
        return Path(env).expanduser()
    return DEFAULT_MVTEC_ROOT


@dataclass(frozen=True)
class MvtecSample:
    image_path: Path
    mask_path: Path | None
    label: int  # 0=good, 1=anomaly
    defect_type: str  # "good" or defect folder name


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in VALID_IMAGE_EXTS


def validate_mvtec_root(mvtec_root: Path) -> None:
    """
    Expected structure:
      mvtec_root/<category>/train/good/*.png
      mvtec_root/<category>/test/<type>/*.png
      mvtec_root/<category>/ground_truth/<type>/*.png   (not for good)
    """
    if not mvtec_root.exists():
        raise FileNotFoundError(
            f"MVTec AD not found at: {mvtec_root}\n"
            f"Set {ENV_MVTEC_ROOT} to your dataset path, or place the dataset at: {DEFAULT_MVTEC_ROOT}\n"
            "The folder should contain category subfolders like 'bottle', 'cable', etc."
        )

    cats = [p for p in mvtec_root.iterdir() if p.is_dir()]
    if not cats:
        raise ValueError(f"No category folders found under: {mvtec_root}")

    ok_any = False
    for cat in cats:
        if (cat / "train" / "good").exists() and (cat / "test").exists():
            ok_any = True
            break

    if not ok_any:
        raise ValueError(
            f"Did not find expected MVTec structure under: {mvtec_root}\n"
            "Expected: <category>/train/good and <category>/test"
        )


def list_categories(mvtec_root: Path) -> list[str]:
    validate_mvtec_root(mvtec_root)
    cats = []
    for p in mvtec_root.iterdir():
        if (p / "train" / "good").exists() and (p / "test").exists():
            cats.append(p.name)
    return sorted(cats)


def iter_train_good(mvtec_root: Path, category: str) -> Iterable[Path]:
    d = mvtec_root / category / "train" / "good"
    if not d.exists():
        raise FileNotFoundError(f"Missing train/good for category={category}: {d}")
    for p in sorted(d.rglob("*")):
        if p.is_file() and _is_image(p):
            yield p


def index_test_split(mvtec_root: Path, category: str) -> list[MvtecSample]:
    test_root = mvtec_root / category / "test"
    gt_root = mvtec_root / category / "ground_truth"
    if not test_root.exists():
        raise FileNotFoundError(f"Missing test folder: {test_root}")

    samples: list[MvtecSample] = []
    for defect_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
        defect_type = defect_dir.name
        is_good = defect_type == "good"
        for img_path in sorted(defect_dir.rglob("*")):
            if not img_path.is_file() or not _is_image(img_path):
                continue

            if is_good:
                samples.append(MvtecSample(img_path, None, 0, "good"))
            else:
                mask_dir = gt_root / defect_type
                # common naming in MVTec: <stem>_mask.png
                mask_path = mask_dir / f"{img_path.stem}_mask.png"
                if not mask_path.exists():
                    # fallback: any matching stem
                    candidates = list(mask_dir.glob(f"{img_path.stem}*"))
                    if candidates:
                        mask_path = candidates[0]
                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Missing ground truth mask for {img_path}\nExpected: {mask_path}"
                    )

                samples.append(MvtecSample(img_path, mask_path, 1, defect_type))

    return samples

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb(img_rgb: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor_1chw_float01(img_rgb: np.ndarray) -> torch.Tensor:
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    t = torch.from_numpy(x).unsqueeze(0)  # 1CHW
    return t


def load_image_tensor(path: Path, size: int) -> torch.Tensor:
    img = read_rgb(path)
    img = resize_rgb(img, size)
    return to_tensor_1chw_float01(img)


def read_mask01(path: Path, size: int) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def pixel_auroc(y_true_masks: list[np.ndarray], y_score_maps: list[np.ndarray]) -> float:
    yt = np.concatenate([m.reshape(-1) for m in y_true_masks]).astype(np.int32)
    ys = np.concatenate([s.reshape(-1) for s in y_score_maps]).astype(np.float32)
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, ys))

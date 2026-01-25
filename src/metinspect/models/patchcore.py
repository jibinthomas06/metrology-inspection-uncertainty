from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.neighbors import NearestNeighbors


@dataclass
class PatchCoreArtifacts:
    backbone: str
    image_size: int
    coreset: np.ndarray
    nn_k: int
    feat_hw: tuple[int, int]


class PatchCore:
    def __init__(self, backbone: str, image_size: int, device: str = "cpu", nn_k: int = 1) -> None:
        self.backbone = backbone
        self.image_size = image_size
        self.device = torch.device(device)
        self.nn_k = nn_k

        self.model = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            out_indices=(3,),
        ).to(self.device)
        self.model.eval()

        self.coreset: np.ndarray | None = None
        self.nn: NearestNeighbors | None = None
        self.feat_hw: tuple[int, int] | None = None

    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)[0]  # (B,C,Hf,Wf)
        self.feat_hw = (int(feats.shape[2]), int(feats.shape[3]))
        patches = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
        return patches

    def fit_embeddings(self, embeddings: np.ndarray) -> None:
        self.coreset = embeddings.astype(np.float32)
        self.nn = NearestNeighbors(n_neighbors=self.nn_k, algorithm="auto")
        self.nn.fit(self.coreset)

    def fit_from_tensors(self, tensors_1chw: list[torch.Tensor], max_patches: int = 20000) -> None:
        embs = []
        for t in tensors_1chw:
            t = t.to(self.device)
            e = self._embed(t).cpu().numpy()
            embs.append(e)
        X = np.concatenate(embs, axis=0).astype(np.float32)
        if X.shape[0] > max_patches:
            idx = np.random.choice(X.shape[0], size=max_patches, replace=False)
            X = X[idx]
        self.fit_embeddings(X)

    def _check(self) -> None:
        if self.coreset is None or self.nn is None or self.feat_hw is None:
            raise RuntimeError("Model not fitted. Run train first.")

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> tuple[float, np.ndarray]:
        self._check()
        x = x.to(self.device)
        patches = self._embed(x).cpu().numpy().astype(np.float32)
        dists, _ = self.nn.kneighbors(patches)
        patch_scores = dists[:, 0]
        Hf, Wf = self.feat_hw
        score_map = patch_scores.reshape(Hf, Wf)
        image_score = float(np.max(score_map))
        return image_score, score_map

    def save(self, path: Path) -> None:
        self._check()
        art = PatchCoreArtifacts(
            backbone=self.backbone,
            image_size=self.image_size,
            coreset=self.coreset,
            nn_k=self.nn_k,
            feat_hw=self.feat_hw,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(art, path)

    @staticmethod
    def load(path: Path, device: str = "cpu") -> PatchCore:
        art: PatchCoreArtifacts = torch.load(path, map_location="cpu", weights_only=False)
        pc = PatchCore(art.backbone, art.image_size, device=device, nn_k=art.nn_k)
        pc.coreset = art.coreset
        pc.feat_hw = art.feat_hw
        pc.nn = NearestNeighbors(n_neighbors=art.nn_k, algorithm="auto")
        pc.nn.fit(pc.coreset)
        return pc

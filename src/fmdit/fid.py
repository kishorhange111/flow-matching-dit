"""Fréchet Inception Distance (Heusel et al. 2017).

Fit a Gaussian to Inception-v3 pool features of each image set, then

    FID = ||μ₁ − μ₂||² + Tr(Σ₁ + Σ₂ − 2 (Σ₁ Σ₂)^{1/2}).

``frechet_distance`` is written here. The square root of Σ₁Σ₂, which is not symmetric, is
computed as ``Σ₁^{1/2} (Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2} Σ₁^{-1/2}``. Its trace equals the
trace of ``(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2}``, a symmetric PSD matrix, so two symmetric
eigendecompositions suffice, with no complex sqrtm. The feature extractor is the
TF-ported Inception from ``pytorch-fid``, which reproduces the reference FID weights.
"""

from __future__ import annotations

import numpy as np
import torch


def _sqrt_psd(a):
    w, v = np.linalg.eigh((a + a.T) / 2)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def frechet_distance(mu1, s1, mu2, s2) -> float:
    r = _sqrt_psd(s1)
    tr_cov = np.trace(s1) + np.trace(s2) - 2 * np.trace(_sqrt_psd(r @ s2 @ r))
    return float(((mu1 - mu2) ** 2).sum() + tr_cov)


def stats(feats: np.ndarray):
    feats = feats.astype(np.float64)
    return feats.mean(0), np.cov(feats, rowvar=False)


class InceptionFeatures:
    """2048-d pool features from pytorch-fid's InceptionV3; images in [0, 1]."""

    def __init__(self, device="cuda"):
        from pytorch_fid.inception import InceptionV3

        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, bs: int = 250) -> np.ndarray:
        out = []
        for i in range(0, len(images), bs):
            x = images[i : i + bs].to(self.device).float()
            out.append(self.net(x)[0].squeeze(-1).squeeze(-1).cpu())
        return torch.cat(out).numpy()

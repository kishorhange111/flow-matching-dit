"""A class-conditional Diffusion Transformer (DiT, Peebles & Xie 2023).

The image is cut into p×p patches. Each patch becomes a token, and fixed 2-D sin-cos
position embeddings are added. The blocks are transformer blocks conditioned through
**adaLN-Zero**: the time and class embedding c regresses a shift, a scale and a gate for
both the attention and the MLP branch,

    x = x + gate_a · Attn(LN(x) · (1 + scale_a) + shift_a)
    x = x + gate_m · MLP (LN(x) · (1 + scale_m) + shift_m)

and the regression layer is zero-initialised, so every block starts as the identity. The
final layer is zero-initialised too, so the untrained model outputs exactly 0. Classifier-
free guidance needs an unconditional model, which is the same network given the extra
"null" class ``n_classes``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    """Sinusoidal embedding of t ∈ [0, 1], scaled by 1000 as in diffusion codebases."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    ang = 1000.0 * t.float()[:, None] * freqs[None]
    return torch.cat([ang.cos(), ang.sin()], -1)


def sincos_2d(grid: int, dim: int) -> torch.Tensor:
    """(grid², dim) fixed 2-D position embedding: half the channels for rows, half for
    columns."""
    q = dim // 4
    omega = 1.0 / 10000 ** (torch.arange(q, dtype=torch.float64) / q)
    pos = torch.arange(grid, dtype=torch.float64)
    ang = pos[:, None] * omega[None]  # (grid, q)
    emb1 = torch.cat([ang.sin(), ang.cos()], 1)  # (grid, dim/2)
    rows = emb1[:, None, :].expand(grid, grid, dim // 2)
    cols = emb1[None, :, :].expand(grid, grid, dim // 2)
    return torch.cat([rows, cols], -1).reshape(grid * grid, dim).float()


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(approximate="tanh"), nn.Linear(h, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def attn(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(b, n, d))

    def forward(self, x, c):
        sa, ca, ga, sm, cm, gm = self.ada(c).chunk(6, -1)
        x = x + ga[:, None] * self.attn(modulate(self.n1(x), sa, ca))
        return x + gm[:, None] * self.mlp(modulate(self.n2(x), sm, cm))


class DiT(nn.Module):
    def __init__(self, img=32, ch=3, patch=2, d=384, depth=12, heads=6, n_classes=10):
        super().__init__()
        self.img, self.ch, self.p, self.n_classes = img, ch, patch, n_classes
        self.grid = img // patch
        self.embed = nn.Linear(ch * patch * patch, d)
        self.register_buffer("pos", sincos_2d(self.grid, d)[None], persistent=False)
        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.y_emb = nn.Embedding(n_classes + 1, d)  # index n_classes = unconditional
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.n_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, ch * patch * patch)
        for m in (self.ada_out[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.y_emb.weight, std=0.02)

    def patchify(self, x):
        b, c, H, W = x.shape
        p = self.p
        x = x.view(b, c, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, (H // p) * (W // p), c * p * p)

    def unpatchify(self, x):
        b, g, p, c = x.shape[0], self.grid, self.p, self.ch
        x = x.view(b, g, g, c, p, p).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, c, g * p, g * p)

    def forward(self, x, t, y):
        """``x: (b, ch, img, img)``, ``t: (b,)`` in [0, 1], ``y: (b,)`` class or
        ``n_classes`` for unconditional. Returns the predicted velocity, shaped like x."""
        h = self.embed(self.patchify(x)) + self.pos
        c = self.t_mlp(timestep_embedding(t, 256)) + self.y_emb(y)
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.ada_out(c).chunk(2, -1)
        return self.unpatchify(self.out(modulate(self.n_out(h), shift, scale)))

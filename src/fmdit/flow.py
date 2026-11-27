"""Rectified flow / conditional flow matching (Lipman et al. 2023, Liu et al. 2023).

Noise z ~ N(0, I) sits at t = 0 and data x at t = 1, joined by the straight line

    x_t = (1 − t)·z + t·x,      target velocity  v = x − z.

The network regresses v. The regression's minimiser is the conditional expectation
``E[x − z | x_t]``, a velocity field whose ODE ``dx/dt = v(x, t)`` carries N(0, I) exactly
onto the data distribution. Sampling integrates that ODE from t = 0 to 1 with a fixed number
of steps: Euler (first order, one evaluation per step) or Heun (second order, two).

Time is sampled uniformly or from a **logit-normal** (SD3, Esser et al. 2024), which
concentrates training on intermediate t, where the velocity is hardest to predict.

**Classifier-free guidance** (Ho & Salimans 2022): labels are dropped to the null class
with probability p during training, and sampling uses
``v = v_uncond + w·(v_cond − v_uncond)``. w = 1 is plain conditional sampling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_t(n, device, schedule="logit_normal", generator=None):
    if schedule == "uniform":
        return torch.rand(n, device=device, generator=generator)
    if schedule == "logit_normal":
        return torch.sigmoid(torch.randn(n, device=device, generator=generator))
    raise ValueError(schedule)


def fm_loss(model, x, y, schedule="logit_normal", p_uncond=0.1, n_classes=None, generator=None):
    """Flow-matching loss on a batch of data ``x`` with labels ``y``."""
    b = x.shape[0]
    z = torch.randn(x.shape, device=x.device, generator=generator)
    t = sample_t(b, x.device, schedule, generator)
    tt = t.view(b, *([1] * (x.dim() - 1)))
    xt = (1 - tt) * z + tt * x
    if p_uncond and n_classes is not None:
        drop = torch.rand(b, device=x.device, generator=generator) < p_uncond
        y = torch.where(drop, torch.full_like(y, n_classes), y)
    return F.mse_loss(model(xt, t, y), x - z)


def guided(model, n_classes, w: float):
    """Wrap ``model`` into a velocity field with classifier-free guidance scale w."""

    def v(x, t, y):
        if w == 1.0:
            return model(x, t, y)
        null = torch.full_like(y, n_classes)
        both = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]))
        vc, vu = both.chunk(2)
        return vu + w * (vc - vu)

    return v


@torch.no_grad()
def sample(v, z, y, steps: int, method: str = "euler"):
    """Integrate dx/dt = v(x, t, y) from t = 0 (noise z) to t = 1. Returns x(1) and the
    number of velocity evaluations used.

    ``heun``'s last corrector evaluates the network at exactly t = 1, which logit-normal
    training almost never samples. ``edm_heun`` takes a plain Euler step there instead, as
    in Karras et al.'s EDM sampler, so every evaluation is at t < 1."""
    x, b = z, z.shape[0]
    ts = torch.linspace(0, 1, steps + 1, device=z.device)
    nfe = 0
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        h = t1 - t0
        k1 = v(x, t0.expand(b), y)
        nfe += 1
        last = i == steps - 1
        if method == "euler" or (method == "edm_heun" and last):
            x = x + h * k1
        elif method in ("heun", "edm_heun"):
            xe = x + h * k1
            k2 = v(xe, t1.expand(b), y)
            nfe += 1
            x = x + h * 0.5 * (k1 + k2)
        else:
            raise ValueError(method)
    return x, nfe

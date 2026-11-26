import math

import numpy as np
import pytest
import torch

from fmdit.fid import frechet_distance, stats
from fmdit.flow import fm_loss, guided, sample
from fmdit.model import DiT, sincos_2d, timestep_embedding
from fmdit.train import EMA

M, S = 1.5, 0.4  # data distribution N(M, S²) per pixel, for the analytic tests


def exact_velocity(x, t, y=None):
    """E[x1 − z | x_t] for z ~ N(0,1), x1 ~ N(M, S²), x_t = (1−t)z + t·x1."""
    t = t.view(-1, *([1] * (x.dim() - 1))).double()
    var = (1 - t) ** 2 + t**2 * S**2
    return M + (t * S**2 - (1 - t)) / var * (x - t * M)


def tiny(**kw):
    torch.manual_seed(0)
    return DiT(img=8, ch=3, patch=2, d=32, depth=2, heads=4, n_classes=10, **kw)


def test_patchify_roundtrip():
    m = tiny()
    x = torch.randn(2, 3, 8, 8)
    p = m.patchify(x)
    assert p.shape == (2, 16, 12)
    torch.testing.assert_close(m.unpatchify(p), x)
    # a patch holds a contiguous 2×2 window of every channel
    torch.testing.assert_close(p[0, 1].view(3, 2, 2), x[0, :, 0:2, 2:4])


def test_untrained_dit_is_zero_and_blocks_are_identity():
    m = tiny()
    x, t, y = torch.randn(3, 3, 8, 8), torch.rand(3), torch.tensor([0, 4, 10])
    assert torch.equal(m(x, t, y), torch.zeros_like(x))
    h, c = torch.randn(3, 16, 32), torch.randn(3, 32)
    for blk in m.blocks:
        torch.testing.assert_close(blk(h, c), h)


def test_every_parameter_trains_within_three_steps():
    m = tiny()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x, y = torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,))
    # zero-init unlocks in stages: output layer, then the gates, then everything behind them
    for _ in range(3):
        opt.zero_grad()
        fm_loss(m, x, y, p_uncond=0.0, n_classes=10).backward()
        opt.step()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_embeddings():
    e = timestep_embedding(torch.tensor([0.0, 0.5]), 256)
    assert e.shape == (2, 256)
    torch.testing.assert_close(e[0], torch.cat([torch.ones(128), torch.zeros(128)]))
    pe = sincos_2d(4, 16)
    assert pe.shape == (16, 16)
    assert len({tuple(r.tolist()) for r in pe}) == 16  # every position distinct
    torch.testing.assert_close(pe[1, :8], pe[0, :8])  # same row ⇒ same row half
    assert not torch.allclose(pe[1, 8:], pe[0, 8:])
    torch.testing.assert_close(pe[4, 8:], pe[0, 8:])  # same column ⇒ same column half


def test_exact_velocity_is_the_loss_minimiser():
    """The closed-form E[x − z | x_t] scores a lower flow-matching loss than perturbations
    of it, which checks the loss's conventions (direction, interpolation, target)."""
    g = torch.Generator().manual_seed(0)
    x = M + S * torch.randn(20000, 1, 2, 2, generator=g, dtype=torch.float64)

    def loss(scale, bias):
        class V(torch.nn.Module):
            n_classes = 1

            def forward(self, xt, t, y):
                return scale * exact_velocity(xt, t) + bias

        gg = torch.Generator().manual_seed(1)
        z = torch.randn(x.shape, generator=gg, dtype=torch.float64)
        t = torch.rand(x.shape[0], generator=gg, dtype=torch.float64)
        tt = t.view(-1, 1, 1, 1)
        xt = (1 - tt) * z + tt * x
        return float(((V()(xt, t, None) - (x - z)) ** 2).mean())

    best = loss(1.0, 0.0)
    for s, b in [(0.9, 0.0), (1.1, 0.0), (1.0, 0.1), (1.0, -0.1)]:
        assert best < loss(s, b)


@pytest.mark.parametrize("method", ["euler", "heun"])
def test_sampling_the_exact_field_reaches_the_data_distribution(method):
    z = torch.randn(200_000, 1, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    x, nfe = sample(exact_velocity, z, None, 200, method)
    assert nfe == (200 if method == "euler" else 400)
    assert abs(x.mean().item() - M) < 0.01
    assert abs(x.std().item() - S) < 0.01


def test_heun_is_second_order_and_euler_first_order():
    """Error in the transported standard deviation against step count."""
    z = torch.randn(100_000, 1, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    ref, _ = sample(exact_velocity, z, None, 4096, "heun")

    def err(n, m):
        return (sample(exact_velocity, z, None, n, m)[0] - ref).abs().max().item()

    e_ratio = err(16, "euler") / err(32, "euler")
    h_ratio = err(16, "heun") / err(32, "heun")
    assert 1.6 < e_ratio < 2.4  # halving h halves Euler's error
    assert 3.2 < h_ratio < 4.8  # and quarters Heun's
    assert err(8, "heun") < err(16, "euler")  # Heun wins at equal NFE


def test_guidance_mixes_conditional_and_unconditional():
    m = tiny()
    for p in m.parameters():
        p.data.normal_(0, 0.1)
    x, t, y = torch.randn(2, 3, 8, 8), torch.rand(2), torch.tensor([3, 7])
    null = torch.full_like(y, 10)
    vc, vu = m(x, t, y), m(x, t, null)
    torch.testing.assert_close(guided(m, 10, 1.0)(x, t, y), vc)
    torch.testing.assert_close(guided(m, 10, 0.0)(x, t, y), vu, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        guided(m, 10, 3.0)(x, t, y), vu + 3 * (vc - vu), atol=1e-6, rtol=1e-5
    )


def test_label_dropout_rate():
    seen = []

    class Spy(torch.nn.Module):
        def forward(self, xt, t, y):
            seen.append(y)
            return torch.zeros_like(xt)

    g = torch.Generator().manual_seed(0)
    x, y = torch.zeros(20000, 1, 1, 1), torch.zeros(20000, dtype=torch.long)
    fm_loss(Spy(), x, y, p_uncond=0.1, n_classes=10, generator=g)
    frac = (seen[0] == 10).float().mean().item()
    assert abs(frac - 0.1) < 0.01


def test_logit_normal_concentrates_on_the_middle():
    from fmdit.flow import sample_t

    g = torch.Generator().manual_seed(0)
    ln = sample_t(100_000, "cpu", "logit_normal", g)
    un = sample_t(100_000, "cpu", "uniform", g)
    mid = lambda t: ((t > 0.25) & (t < 0.75)).float().mean().item()
    assert mid(ln) > 0.65 and abs(mid(un) - 0.5) < 0.01
    assert abs(ln.median().item() - 0.5) < 0.01


def test_frechet_distance_matches_scipy_and_known_cases():
    scipy = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(500, 6)), rng.normal(size=(500, 6)) @ rng.normal(size=(6, 6))
    mu1, s1 = stats(a)
    mu2, s2 = stats(b + 0.3)
    want = ((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * scipy.sqrtm(s1 @ s2).real)
    assert frechet_distance(mu1, s1, mu2, s2) == pytest.approx(want, rel=1e-6)
    assert frechet_distance(mu1, s1, mu1, s1) == pytest.approx(0.0, abs=1e-8)
    # N(0, I) vs N(m, 4I) in d dims: |m|² + d·(1 + 4 − 2·2) = |m|² + d
    d, m = 5, np.full(5, 0.5)
    assert frechet_distance(np.zeros(d), np.eye(d), m, 4 * np.eye(d)) == pytest.approx(
        (m**2).sum() + d
    )


def test_ema_tracks_the_weights():
    m = torch.nn.Linear(2, 2)
    ema = EMA(m, 0.9)
    w0 = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    for _ in range(3):
        ema.update(m)
    want = w0 + 1.0 * (1 - 0.9**3)
    torch.testing.assert_close(ema.shadow["weight"], want)


def test_training_reduces_the_loss_on_a_tiny_dataset():
    m = tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randn(8, 3, 8, 8, generator=torch.Generator().manual_seed(0)).clamp(-1, 1)
    y = torch.arange(8) % 10

    def held_loss():
        with torch.no_grad():
            g = torch.Generator().manual_seed(123)
            xs, ys = x.repeat(64, 1, 1, 1), y.repeat(64)
            return fm_loss(m, xs, ys, p_uncond=0.0, n_classes=10, generator=g).item()

    before = held_loss()
    for _ in range(300):
        loss = fm_loss(m, x, y, p_uncond=0.0, n_classes=10)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = held_loss()
    assert math.isfinite(after) and after < 0.8 * before

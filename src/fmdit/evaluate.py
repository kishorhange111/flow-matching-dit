"""FID of a trained DiT across samplers, step counts and guidance scales (GPU, resumable).

    python -m fmdit.evaluate --ckpt results/main/ema_last.pt --out results/main/eval

The reference statistics are Inception features of the 50,000 CIFAR-10 training images,
computed once and cached. Each configuration draws ``--n`` samples with balanced classes
and a fixed noise seed, so configurations differ only in how the ODE is solved. FID with
5,000 samples is biased upward relative to the usual 50,000, so numbers are comparable
with each other but not with published 50k FIDs. Each row is appended to ``fid.jsonl``
as it finishes.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .fid import InceptionFeatures, frechet_distance, stats
from .flow import guided, sample
from .model import DiT
from .train import CONFIGS, cifar10

GRID = [("euler", s, 1.5) for s in (8, 16, 32, 64)]
GRID += [("heun", s, 1.5) for s in (4, 8, 16, 32)]
GRID += [("euler", 32, w) for w in (1.0, 2.0, 3.0, 4.0)]


@torch.no_grad()
def generate(model, n, steps, method, w, seed=0, bs=500, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    v = guided(model, model.n_classes, w)
    out, nfe = [], 0
    for i in range(0, n, bs):
        m = min(bs, n - i)
        y = (torch.arange(i, i + m, device=device) % model.n_classes).long()
        z = torch.randn(m, model.ch, model.img, model.img, device=device, generator=g)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            x, nfe = sample(v, z, y, steps, method)
        out.append(((x.float().clamp(-1, 1) + 1) / 2).cpu())
    return torch.cat(out), nfe * (2 if w != 1.0 else 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", choices=list(CONFIGS), default="S")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--grid", default="full", choices=["full", "best"])
    ap.add_argument("--ref", default="results/ref_cifar10_train.npz")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    dev = a.device
    os.makedirs(a.out, exist_ok=True)
    model = DiT(**CONFIGS[a.size]).to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    inc = InceptionFeatures(dev)
    if os.path.exists(a.ref):
        r = np.load(a.ref)
        mu_r, s_r = r["mu"], r["sigma"]
    else:
        X, _ = cifar10(a.data, "cpu")
        mu_r, s_r = stats(inc(X.float() / 255.0))
        os.makedirs(os.path.dirname(a.ref) or ".", exist_ok=True)
        np.savez(a.ref, mu=mu_r, sigma=s_r)
    path = os.path.join(a.out, "fid.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["method"], r["steps"], r["cfg"], r["n"]) for r in map(json.loads, open(path))}
    grid = GRID if a.grid == "full" else [("heun", 16, 1.5)]
    for method, steps, w in grid:
        if (method, steps, w, a.n) in done:
            continue
        t = time.time()
        imgs, nfe = generate(model, a.n, steps, method, w, device=dev)
        gen_s = time.time() - t
        mu, s = stats(inc(imgs))
        rec = {"method": method, "steps": steps, "cfg": w, "n": a.n, "nfe": nfe,
               "fid": frechet_distance(mu, s, mu_r, s_r), "gen_seconds": gen_s,
               "ckpt": os.path.basename(a.ckpt)}  # fmt: skip
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    from torchvision.utils import save_image

    imgs, _ = generate(model, 100, 32, "heun", 2.0, seed=1, bs=100, device=dev)
    order = torch.arange(100).view(10, 10).t().flatten()  # one row per class
    save_image(imgs[order], os.path.join(a.out, "samples_heun32_cfg2.png"), nrow=10)


if __name__ == "__main__":
    main()

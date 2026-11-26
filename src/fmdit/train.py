"""Train a class-conditional DiT on CIFAR-10 with flow matching (GPU, resumable).

    python -m fmdit.train --out results/main --budget-min 35

Mixed precision (fp16 autocast with a grad scaler), AdamW, linear warm-up, and an EMA of
the weights used for sampling. Everything needed to continue (model, EMA, optimiser,
scaler, step, RNG) is saved every ``--ckpt-every`` steps and when the time budget runs out,
so a run continues across GPU sessions that are reclaimed. EMA snapshots are kept every
``--snap-every`` steps, so runs can be compared at equal step counts.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .flow import fm_loss
from .model import DiT

CONFIGS = {
    "S": dict(d=256, depth=8, heads=4),  # 9.7M parameters
    "B": dict(d=384, depth=12, heads=6),  # 32.5M parameters (DiT-S/2 size)
}


def cifar10(root, device, train=True):
    """The whole split as a uint8 tensor on ``device`` plus labels: 150 MB, so augmentation
    and batching stay on the GPU."""
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root, train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    return x.to(device), torch.tensor(ds.targets, device=device)


def to_model_range(x_u8, flip=None):
    x = x_u8.float() / 127.5 - 1.0
    if flip is not None:
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    return x


def half(sd):
    """fp16 copy of a state dict, for the sampling snapshots (half the size)."""
    return {k: v.half() for k, v in sd.items()}


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--data", default="data")
    ap.add_argument("--size", choices=list(CONFIGS), default="S")
    ap.add_argument("--schedule", default="logit_normal", choices=["logit_normal", "uniform"])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--p-uncond", type=float, default=0.1)
    ap.add_argument("--budget-min", type=float, default=1e9)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--snap-every", type=int, default=10_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    deadline = time.time() + 60 * a.budget_min
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    cuda = dev == "cuda"
    torch.manual_seed(a.seed)
    X, Y = cifar10(a.data, dev)
    model = DiT(**CONFIGS[a.size]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    ema = EMA(model, a.ema)
    step = 0
    ck = os.path.join(a.out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"])
        ema.shadow = s["ema"]
        step = s["step"]
        torch.set_rng_state(s["rng"])
        if cuda and s.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(s["cuda_rng"])
    n_params = sum(p.numel() for p in model.parameters())
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    def save():
        torch.save(
            {
                "model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "ema": ema.shadow, "step": step,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if cuda else None,
                "args": vars(a),
            },
            ck + ".tmp",
        )  # fmt: skip
        os.replace(ck + ".tmp", ck)

    print(f"start at step {step}, {n_params / 1e6:.2f}M parameters", flush=True)
    t0, seen, run_loss = time.time(), 0, 0.0
    while step < a.steps and time.time() < deadline:
        idx = torch.randint(0, len(X), (a.bs,), device=dev)
        flip = torch.rand(a.bs, device=dev) < 0.5
        x = to_model_range(X[idx], flip)
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            loss = fm_loss(model, x, Y[idx], a.schedule, a.p_uncond, model.n_classes)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model)
        step += 1
        seen += 1
        run_loss += loss.item() if step % 50 == 0 else 0.0
        if step % 500 == 0:
            rate = seen / (time.time() - t0)
            rec = {"step": step, "loss": run_loss / 10, "steps_per_s": rate}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            run_loss = 0.0
        if step % a.ckpt_every == 0:
            save()
        if step % a.snap_every == 0:
            torch.save(half(ema.shadow), os.path.join(a.out, f"ema_{step}.pt"))
    save()
    torch.save(half(ema.shadow), os.path.join(a.out, "ema_last.pt"))
    print(f"stopped at step {step}", flush=True)


if __name__ == "__main__":
    main()

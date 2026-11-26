# flow-matching-dit

[![CI](https://github.com/kishorhange111/flow-matching-dit/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/flow-matching-dit/actions/workflows/ci.yml)

**Rectified flow matching** with a class-conditional **Diffusion Transformer**, written from
scratch and trained on CIFAR-10. The recipe is the same family as Stable Diffusion 3 and
Flux, at toy scale:
- a DiT with adaLN-Zero conditioning;
- straight-line flow matching with logit-normal timesteps;
- classifier-free guidance;
- Euler and Heun ODE samplers;
- an FID implementation to measure them.

```
noise z ──(1−t)z + t·x──► x_t ──DiT(x_t, t, class)──► v̂ ≈ x − z        train: ‖v̂ − (x − z)‖²
sample: integrate dx/dt = v̂_guided(x, t) from t = 0 (noise) to t = 1 (image)
```

## Pieces

**[`model.py`](src/fmdit/model.py): the DiT.**
- The image becomes 2×2 patches with fixed 2-D sin-cos positions.
- The time and class embedding drives **adaLN-Zero** in every block: a regressed shift,
  scale and gate around attention and around the MLP.
- The regression layers and the output layer start at zero. Every block therefore starts
  as the identity and the model's first prediction is exactly 0, which is the stability
  trick of the DiT paper.
- Class index 10 is the "null" class that classifier-free guidance needs.

**[`flow.py`](src/fmdit/flow.py): flow matching and sampling.**
- The interpolant is `x_t = (1−t)z + t·x` and the regression target is `x − z`.
- Time is drawn uniformly or from a logit-normal, as in SD3.
- Labels are dropped to the null class with probability 0.1.
- Guidance is `v_u + w(v_c − v_u)`, and the Euler and Heun integrators report how many
  velocity evaluations (NFE) they used.

**[`fid.py`](src/fmdit/fid.py): FID.** It computes the Fréchet distance between Gaussian
fits of Inception-v3 features. The matrix square root of the non-symmetric `Σ₁Σ₂` is avoided
by taking the trace of `(Σ₁^{½} Σ₂ Σ₁^{½})^{½}`, which is similar to it. That leaves two
symmetric eigendecompositions and no complex `sqrtm`. Features come from pytorch-fid's
TF-ported Inception, so the numbers use the standard network.

**[`train.py`](src/fmdit/train.py) / [`evaluate.py`](src/fmdit/evaluate.py).**
- Training: the whole of CIFAR-10 sits on the GPU, with fp16 autocast, AdamW, an EMA of the
  weights, and checkpoints that resume across reclaimed GPU sessions.
- Evaluation: FID over samplers, step counts and guidance scales from a fixed noise seed,
  plus a grid of samples per class.

## Tests

There are 14, on CPU. Several are **analytic**: for data `N(m, s²)` the optimal velocity
field `E[x − z | x_t]` has a closed form, and the tests check:
- the closed-form field **scores a lower flow-matching loss** than perturbations of it, which
  pins the loss's direction and target;
- **integrating it transports N(0, 1) onto N(m, s²)** with both samplers;
- **Euler is first order and Heun second order**: halving the step halves and quarters the
  error, and Heun beats Euler at equal NFE;
- the untrained DiT outputs exactly 0, every block starts as the identity, and every
  parameter trains within three steps as the zero-inits unlock;
- guidance at w = 1 is the conditional model and at w = 0 the unconditional one, and the
  label-dropout rate is 10%;
- patchify round-trips, and the 2-D position embedding shares a row half within a row and a
  column half within a column;
- **FID matches `scipy.linalg.sqrtm`**, is 0 for identical statistics, and matches the
  closed form `|m|² + d` for N(0, I) against N(m, 4I);
- the EMA update is exact, and training lowers the loss on a tiny dataset.

```
pip install -e ".[dev]"
pytest -q                                                    # 14 tests, CPU
pip install -e ".[train]"
python -m fmdit.train --out results/main --budget-min 35     # GPU, resumable
python -m fmdit.evaluate --ckpt results/main/ema_last.pt --out results/main/eval
```

## Results

**Pending** the Colab T4 runs.

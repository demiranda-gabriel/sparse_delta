# 5-composite_staged / stage3_sweep_sparsity

**Status:** done-success (all three runs complete)
**Date:** 2026-05-28
**Parent:** [5-composite_staged](../)

## Intent

Stage 3 with `sparsity_coeff=1.0` (parent run, wandb `vzqofs9q`)
**regressed** vs stage 2 (`ay5b5zes`): test forces_mae went from 0.073
eV/Å → 0.096 eV/Å when the gate was added. With `EnergyForceLoss`
weighting at `coeffs={total_energy: 1.0, forces: 1.0}` plus the
sparsity metric at `coeff=1.0`, `MetricsManager` normalises so all
three contribute equally to `weighted_sum`. The gate trades useful M1
correction for sparsity — the wrong balance for the science target.

This sweep keeps stage 1 + stage 2 weights frozen (gate-only training,
identical to the parent stage 3) and varies only `sparsity_coeff`:

| Tag | sparsity_coeff | Hypothesis |
|---|---|---|
| `sc_0.01` | 0.01 | Sparsity penalty negligible. Gate likely saturates at 1.0 everywhere (~equivalent to stage 2 const-gate). Sets the **lower-bound forces_mae** for the gated model. |
| `sc_0.1` | 0.1 | Mild penalty. Gate should retain ~most of M1's contribution while still going to 0 where M1 is irrelevant. Expected sweet spot. |
| `sc_0.5` | 0.5 | Intermediate. |

Existing `sc_1.0` is the parent stage-3 run (`vzqofs9q`); no need to
rerun.

## Hypothesis

`mean(λ)` on val should approach 1.0 as `sparsity_coeff → 0` and
`forces_mae` should approach stage 2's 0.073 eV/Å. The sweep
identifies the largest `sparsity_coeff` that still keeps `forces_mae`
within (say) 5 % of stage 2 — that's the design target for the gate
(maximum sparsity that doesn't hurt accuracy materially).

## Setup

All three runs reuse `runs/5-composite_staged/stage2_m1/stage2_weights.pt`
via the existing `LoadWeightsCallback`. Only difference per config is
the `sparsity_coeff` hydra interpolation. Submitted across the 4-
partition list per `feedback_gpu_partitions.md`.

## Run

```bash
sbatch sc_0.01/train.sh
sbatch sc_0.1/train.sh
sbatch sc_0.5/train.sh
```

Independent jobs (no dependency chain — each is self-contained,
loads from stage 2 directly).

## Outcome (2026-05-28)

| Tag | Status | Wall | Test forces_mae | Test per_atom_E mae | Verdict |
|---|---|---|---|---|---|
| sc_0.01 | done | 47 min | **0.0798** | 0.0068 | gate ON, ~stage 2's 0.073 baseline |
| sc_0.1  | done | 1h06m | 0.0956 | 0.0081 | **collapsed** (gate → 0) |
| sc_0.5  | done | 53 min | 0.0956 | 0.0081 | collapsed |
| sc_1.0 (parent) | done | 44 min | 0.0956 | 0.0081 | collapsed |

**Key finding:** sharp transition between sparsity_coeff=0.01 and 0.1.
Beyond 0.1, the optimizer collapses λ → 0 uniformly. Test metrics
become identical to the bit (forces_mae=0.0956, per_atom_E=0.0081)
across sc=0.1 / 0.5 / 1.0 because the composite reduces to frozen
M0-only — a deterministic fixed point.

**Why so sharp:** `MetricsManager` normalises loss coefficients so they
sum to 1. With `total_energy: 1.0`, `forces: 1.0`, `sparsity: sc`, the
sparsity weight is `sc / (2 + sc)`. At `sc=0.01` → 0.5% of loss → gate
free to stay near 1. At `sc=0.1` → 4.8% → enough to overpower the
accuracy term once the optimizer finds the λ=0 fixed point. The
penalty's linearity in λ means the gradient is constant (no
self-balancing) — once λ starts shrinking, nothing stops it.

**Verdict on the linear sparsity penalty:** unusable as-is. Either:

1. **No gate (stage 2 const-λ wins):** forces_mae=0.073, but no
   compute savings — M1 runs everywhere.
2. **Gate collapse:** forces_mae=0.096, equivalent to M0 alone.

The middle ground (sparse but accurate) requires a different penalty
shape. Options to try in a follow-up:

- **Quadratic penalty** `mean(λ²)` — gradient ∝ λ, vanishes near 0,
  self-balancing.
- **Entropy-style penalty** `mean(λ · log λ + (1−λ) · log(1−λ))` —
  drives λ to bistable {0, 1} but not uniformly zero.
- **Anneal sparsity_coeff** from 0 → target over training; gate first
  learns to be useful, then carves out the sparse subset.
- **Initialise λ near 1** instead of 0.05 — start where M1 is needed,
  prune outward.

None of these are blockers for the current science — stage 2 with
constant λ already gives the best forces_mae we've seen (0.073). Gate
work is a follow-up R&D thread.

Wandb URLs filled per tag:

| Tag | WandB |
|---|---|
| sc_0.01 | filled from runs/.../sc_0.01/wandb/ |
| sc_0.1  | filled from runs/.../sc_0.1/wandb/  |
| sc_0.5  | filled from runs/.../sc_0.5/wandb/  |
| sc_1.0 (parent) | [vzqofs9q](https://wandb.ai/demiranda-gabriel/sparse-delta/runs/vzqofs9q) |

# 5-composite_staged / stage3_sweep_sparsity

**Status:** planned
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

## Outcome

Filled per tag after each finishes.

| Tag | Status | WandB | Test forces_mae | Test per_atom_E mae | val mean(λ) | Verdict |
|---|---|---|---|---|---|---|
| sc_0.01 | planned | — | — | — | — | — |
| sc_0.1  | planned | — | — | — | — | — |
| sc_0.5  | planned | — | — | — | — | — |
| sc_1.0 (parent) | done | [vzqofs9q](https://wandb.ai/demiranda-gabriel/sparse-delta/runs/vzqofs9q) | 0.096 | 0.0081 | (see analyze_gate) | regression |

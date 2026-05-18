# 4-composite_warmstart

**Status:** running (queued)
**Date:** 2026-05-17
**Outer SHA:** `2677aca`
**Submodule SHAs:** `nequip-private=74c7689  allegro-private=82d7258  sparse-delta-core=9e10612`
**SLURM job:** `13516824` on `kozinsky_gpu` (1-day walltime, 200 epoch cap, patience 40).
**WandB:** _filled once the job starts_ — project `sparse-delta`, group `4-composite_warmstart`

**Smoke (job 13513507):** infra OK — 3 epochs on gpu_test with cu128 torch + cuEquivariance + compile_mode=compile. Train weighted_sum 3.18 → 1.52; val 2.70 → 1.46; λ_mean 0.05 → 0.84 (gate opens fast with tiny M0 + sparsity_coeff=0.1; expected to rebalance once M0 has capacity at the full config sizes). Logs: `_smoke/logs/sd_warmstart_smoke_13513507.{out,err}`.

**Acceleration bisect (2026-05-18):** v1/v2 fast runs with `compile_mode: compile` + cuEquivariance both NaN'd mid-training (v1: epoch 8 step 118 wp2qrft3; v2: epoch 15 dd394qyh). _diag (eager, no cueq, 8xtfq0f3) trained 30 epochs cleanly. Bisect runs:
- `_diag_b1` (job 13687384, eager + cuEquivariance ON): 30 epochs clean (run 8mjd3yhz). val_weighted_sum 0.13 → 0.075, val_forces_mae 0.13.
- `_diag_b2` (job 13687385, compile + cuEquivariance OFF): FAILED 9:27 with `RuntimeError: Function 'CompiledFunctionBackward' returned nan values in its 8th output.`

**Conclusion: PT2 compile is the NaN source, cuEquivariance is innocent.** Production config now `compile_mode: eager` with cuEquivariance modifier kept on. Upstream PT2 compile + warm-start composite bug needs separate investigation; not blocking science.

## Intent

First training run of the **from-scratch warm-start composite** described in [`notes/m1_warmstart_design.md`](../../notes/m1_warmstart_design.md). M0 and M1 are both fresh `Allegro_Module`s built in a single `SequentialGraphNetwork`; M1's `tensor_features_in_field` is **aliased** to M0's `_allegro_pre_final_tp_out` so M1's strided tensor input is the pre-final TP output of M0 — every irrep there carries gradient signal from the energy/force loss.

Distinct from [`3-composite_phase_A`](../3-composite_phase_A/), which loads two **pre-trained** packages and only trains the gate. Here everything trains jointly from random init.

## Hypothesis

Joint training of (M0, gate, M1) with M1's tensor input architecturally warm-started from M0's pre-final TP converges to a useful composite faster than:

(a) training M1 standalone first and then plugging it into the Phase A composite (= exp 3 path), or
(b) joint training where M1 carries its own `TwoBodySphericalHarmonicTensorEmbed`.

The mechanism: M1 inherits a pre-trained two-body tensor embedding for free, and gradients keep adapting M0's penultimate features to be useful for M1.

Falsified if (a) joint training is unstable (NaN, energy diverges) at the chosen LRs, (b) `mean(λ)` collapses to 0 or saturates to 1 within the first few epochs, or (c) val force MAE at convergence is no better than a same-budget standalone Allegro of equal total parameter count.

## Architecture

Built by `sparse_delta_core.model.build_warmstart_composite`. Constraints enforced at build time:
- `m0_num_layers >= 2` (need pre-final TP).
- `m1_num_tensor_features == m0_num_tensor_features` (no channel adapter in v1).
- `m1_l_max <= m0_l_max` (M1 can't synthesize higher-ℓ content than M0 supplies).

### M0 (small, runs everywhere)

| | |
|---|---|
| `l_max` | 2 |
| `parity` | true |
| `num_layers` | 2 |
| `num_scalar_features` | 32 |
| `num_tensor_features` | 16 |
| `allegro_mlp` | depth 1, width 64, silu |
| `readout_mlp` | depth 1, width 32 |
| Bessel radial | 8 bessels, p=6, embed_dim=128 |

### M1 (larger, gated, strictly local)

| | |
|---|---|
| `l_max` | 2 (inherited) |
| `parity` | true |
| `num_layers` | 1 |
| `num_scalar_features` | 128 |
| `num_tensor_features` | 16 (must match M0) |
| `allegro_mlp` | depth 1, width 128, silu |
| `readout_mlp` | depth 1, width 64 |

### Gate (`EdgeFeaturesToNodeLambda`)

| | |
|---|---|
| Input | `EDGE_FEATURES_KEY` (M0's DenseNet concat) |
| Edge MLP | linear, output 64 |
| Pool | sum to receiver atom |
| Node MLP | depth 2, hidden 64, silu, final linear scalar |
| Function | `sigmoid` warm-up (`gate_function="sigmoid"`) |
| Init | zero-weight final layer, bias = `logit(0.05)` ⇒ `λ ≈ 0.05` at step 0 |

Swap `gate_function: poly5` for Phase D (exact sparsity); requires a calibrated `(s_low, s_high)` window from a `λ_i` histogram across the val set. Deferred to a follow-up experiment.

## Training

| | |
|---|---|
| `batch_size` | 4 |
| optimizer | Adam, `lr=1e-3`, `grad_clip=0.8` |
| LR schedule | `ReduceLROnPlateau` patience=20, factor=0.8, `min_lr=1e-6` |
| early-stop | patience=50 on `val0_epoch/weighted_sum` |
| `max_epochs` | 500 |
| `max_time` | 3 days (`#SBATCH --time=3-00:00:00` on `kozinsky_gpu`) |
| loss | `EnergyForceLoss(per_atom_energy=True, total_energy=1.0, forces=1.0)` + `sparsity_penalty.coeff=0.1` (`GateMeanMetric` on `GATE_KEY`) |
| `seed` | 217 |

## Dataset

Same splits used by experiments 0 / 1 / 2 / 3 — `data/optb88/cameron/split_dataset_r5.0_{train,val,test}.xyz`. 1956 / 244 / 244 frames, species `{C, O, Pt}`.

On FASRC the canonical persistent path is `/n/holylabs/LABS/kozinsky_lab/Users/demiranda/projects/sparse_delta/data/`; until that path is set up, the multifidelity sibling's copy at `/n/holylabs/LABS/kozinsky_lab/Users/demiranda/projects/multifidelity/data/` is the source.

## Success criteria

- Training completes (early-stop or `max_epochs=500`) without NaN.
- `mean(λ)` on val ∈ [0.05, 0.5] at convergence.
- λ distribution shows separation between Pt-only frames and mixed/CO frames (qualitative check).
- Test force MAE **strictly better** than a same-budget standalone Allegro (compare to the M0-alone baseline-B + the L-config from experiment 2 if its ckpt is available).
- NVE drift on a held-out 5-step CO/Pt trajectory < 1 meV/atom/step (smoke check; full conservation already enforced at the test level in `tests/test_warmstart_composite.py`).

## Run

```bash
sbatch experiments/4-composite_warmstart/train.sh
```

Outputs:
- Hydra run dir: `runs/4-composite_warmstart/` — checkpoints (`best.ckpt`, `last.ckpt`), wandb dir, lightning logs.
- SLURM logs: `experiments/4-composite_warmstart/logs/`.

## Blockers

1. **Data path.** `data/optb88/cameron/` is not yet present on this clone of `sparse_delta`. Either symlink the sibling `multifidelity` data dir or pull from gdrive.
2. **Pyright import noise.** The venv is a uv-managed `.venv` that Pyright doesn't auto-detect — no code-level issue.

## Outcome

_Filled in after the run._

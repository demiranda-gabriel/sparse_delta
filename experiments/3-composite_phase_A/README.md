# 3-composite_phase_A

**Status:** planned (do **not** submit before swapping the M1 package path — see "Blockers" below)
**Date:** 2026-05-15
**Outer SHA:** _filled at commit time_
**Submodule SHAs:** `nequip-private=…  allegro-private=…`
**WandB:** _filled at submission_ — project `sparse-delta`, group `3-composite_phase_A`

## Intent

First training run of the sparse_delta composite (M0 + gate + M1). Phase A schedule from [`notes/sparse_local_correction.md`](../../notes/sparse_local_correction.md) §16.3:

- **M0 frozen** (baseline-B, inherited from sibling `multifidelity` project)
- **M1 frozen** (warm-started from [`2-allegro_L_correction`](../2-allegro_L_correction/))
- **Only the gate MLP trains**
- Loss = energy MSE + force MSE + `α · mean(λ)` (sparsity penalty), with α the relative weight in MetricsManager's normalized weighted_sum
- Sigmoid gate, initial `mean(λ) = 0.05` (Phase A warm-up; sigmoid→poly_5 swap is Phase D)

Gate input: **C0 only** (per-atom sum-pooled scalar features from M0's Allegro DenseNet). P1 plumbing exists in `sparse_delta_core.features.M0InvariantFeatures` but `compute_p1=False` skips it. If C0-only Phase A underperforms, revisit P1 (no new code needed — flip the flag and concat in the gate).

## Gate MLP

| | |
|---|---|
| Input dim | 96 (baseline-B M0: `num_scalar_features=32` × (`num_layers=2` + 1)) |
| Hidden | 1 layer of width 64, silu |
| Output | scalar → sigmoid → λ ∈ (0, 1) |
| Final-layer init | zero weights, bias = logit(0.05) → λ = 0.05 everywhere at step 0 |

Per [test_gate.py::test_gate_gradient_flow_at_init], the hidden layer has zero gradient at step 0 (because the final weight is zero); after one optimizer step the final weight becomes non-zero and the hidden layer joins in. This is intentional — it costs one step of warm-up, not learning capacity.

## Composite

`build_phase_a_composite(...)` in [`sparse_delta_core.model`](../../software/sparse-delta-core/sparse_delta_core/model/factory.py):

1. Loads M0 + M1 from `.nequip.zip` packages via `ModelFromPackage` (compile_mode="eager" for both — baseline-B carries a pre-patch class snapshot; the M1 from exp 2 will also use eager until repackaged with the patched code).
2. Wraps M0 in `M0InvariantFeatures(compute_p1=False)` so the per-atom C0 feature is written to the AtomicDataDict as `LAST_LAYER_SCALARS_KEY`.
3. Builds the `GateMLP`.
4. Constructs `SparseDeltaComposite(m0_extractor, gate, m1)` which composes `E_total^i = E_0^i + λ_i · E_1^i` and reduces to total energy via `AtomwiseReduce`. Inner `ForceStressOutput` modules on both M0 and M1 are auto-disabled at construction (CLAUDE.md gotcha "no nested ForceStressOutput").
5. Freezes M0 and M1 parameters (`requires_grad = False`).
6. Wraps the composite in a single top-level `ForceStressOutput` for conservative forces.

## Training config

| | |
|---|---|
| `batch_size` | 8 (composite memory footprint is ~2× a single Allegro) |
| optimizer | Adam, `lr=2e-3`, `grad_clip=0.8`. **Param groups filtered to trainable only** via `sparse_delta_core.train.trainable_param_group` — Adam state is allocated only for the gate. |
| LR schedule | `ReduceLROnPlateau` patience=20, factor=0.8, `min_lr=1e-6` |
| early-stop | patience=50 on `val0_epoch/weighted_sum` |
| `max_epochs` | 500 |
| `max_time` | 1 day (cap; Frontier 1-node walltime is 2h per job, chain via `--dependency=afterok`) |
| loss | `EnergyForceLoss(per_atom_energy=True, total_energy=1.0, forces=1.0)` + `sparsity_penalty.coeff=0.1` (`GateMeanMetric` on `GATE_KEY`) |
| `seed` | 185 |

## Dataset

[`data/optb88/cameron/split_dataset_r5.0_{train,val,test}.xyz`](../../data/optb88/cameron/) — same splits used by experiments 0 / 1 / 2. 1956 train / 244 val / 244 test frames. Species `{C, O, Pt}`.

## Hypothesis

Phase A trains the gate to identify "complex" atoms in CO/Pt frames — bulk-Pt atoms should converge to λ → 0 and reactive interface atoms / gas-phase CO atoms to λ > 0. Concretely:

- Test force MAE **should improve over pure M0** (baseline-B alone, as measured by experiment 0).
- `mean(λ)` should converge to a value < 1 (sparsity penalty drives it down).
- λ distribution on val frames should **be bimodal** — separating Pt-only frames from mixed frames.

Falsified if (a) λ collapses to 0 everywhere (gate starves; nothing learned), (b) λ saturates to 1 everywhere (M1 takes over; sparsity penalty too weak), or (c) test force MAE is no better than M0 alone (gate doesn't carry useful signal).

## Success criteria

- Training completes (early-stop or `max_epochs=500`) without NaN.
- `mean(λ)` on val ∈ [0.05, 0.5] at convergence (well-defined active fraction).
- Per-class histograms of `λ` show bimodal separation (Pt-only ≪ mixed ≪ CO-only or similar).
- Test force MAE **strictly better** than baseline-B alone.

## Run

```bash
cd /lustre/orion/mat281/scratch/demirand/projects/sparse_delta
sbatch experiments/3-composite_phase_A/train.sh
```

Outputs:

- Hydra run dir: `/lustre/orion/mat281/scratch/demirand/projects/sparse_delta/runs/3-composite_phase_A/` — checkpoints (`best.ckpt`, `last.ckpt`), wandb local dir, lightning logs.
- SLURM logs: `experiments/3-composite_phase_A/logs/`.

## Blockers

1. **M1 package path is a placeholder.** [`config.yaml`](config.yaml) currently points `m1_package_path` to `M0-baseline-B.nequip.zip`, so the run would compose `E = E_0 + λ · E_0 = (1 + λ) · E_0` — meaningless training. Swap to the experiment 2 checkpoint (`saved_models/packages/2-allegro_L_correction.nequip.zip`) once experiment 2 produces it. Then verify with the canonical sparse_delta WandB URL convention before sbatch.
2. **Experiment 2 must finish first.** Without it, there's no M1 to warm-start from. If the user wants to smoke-test the plumbing before exp 2 lands, the `test_factory.py` suite covers shape/grad/forward — running the full training with M0 as a stand-in M1 would just burn GPU hours.

On success: copy best ckpt + nequip package to `saved_models/packages/3-composite_phase_A.nequip.zip`, append a row to `saved_models/best_checkpoint_paths.csv`, flip status to `done-success`, paste the wandb run URL above.

## Outcome

_Filled in after the run._

# 2-allegro_L_correction

**Status:** planned
**Date:** 2026-05-15
**Outer SHA:** `14cf92ae`
**Submodule SHAs:** `nequip-private=c2ec1f2b  allegro-private=82d7258`
**WandB:** project [wandb.ai/demiranda-gabriel/sparse-delta](https://wandb.ai/demiranda-gabriel/sparse-delta), group `2-allegro_L_correction`. Run URL: _filled when job 4590559 starts._

## Intent

First training run on the cameron CO/Pt dataset in `sparse_delta`. Train a large Allegro model with the `l-l3-paper` architecture from [`mtfd/experiments/12-bulk_pt_relabel_sweep`](../../../mtfd/experiments/12-bulk_pt_relabel_sweep/) — the largest tag in that 5-tag bulk-Pt sweep. Adapted for:

- **Frontier (AMD MI250X / ROCm 6.2.4).** Drop `enable_CuEquivarianceContracter` (NVIDIA-only); keep `compile_mode: compile`.
- **3 species** `{C, O, Pt}` (not bulk Pt only).
- **sparse_delta paths and wandb.** Project `sparse-delta`, entity `demiranda-gabriel`.

This trains a candidate **M1 correction model** for the sparse_delta gated multi-fidelity design. Allegro stays strictly local at any depth because its receptive field is `r_max` regardless of `num_layers` (see [`notes/sparse_local_correction.md`](../../notes/sparse_local_correction.md) §2.3), so `num_layers = 4` is fine. Training as a *standalone* model first (not yet inside the `E = E_0 + Σ_i λ_i E_1^i` composite) — this baseline anchors what "large Allegro alone" can do on cameron CO/Pt and gives a checkpoint to plug into the gated composite once that scaffold lands.

## Architecture (from `l-l3-paper`)

| | |
|---|---|
| `l_max` | 3 |
| `r_max` | 7.0 Å |
| `parity` | true |
| `num_layers` | 4 |
| `num_scalar_features` | 128 |
| `num_tensor_features` | 64 |
| `allegro_mlp` depth × width | 2 × 128 (silu) |
| `readout_mlp` | depth 1, width 64, linear out |
| Bessel radial | 12 bessels, p=6, embed_dim=512, scalar-embed MLP depth-1 width-512 |

## Training

| | |
|---|---|
| `batch_size` | 16 |
| optimizer | Adam, `lr=2e-3`, `grad_clip=0.8` |
| LR schedule | `ReduceLROnPlateau` patience=20 factor=0.8 `min_lr=1e-6` |
| early-stop | patience=1000 on `val0_epoch/weighted_sum` |
| `max_epochs` | 5000 |
| `max_time` | 1 day |
| loss | `EnergyForceLoss(per_atom_energy=True, total_energy=1.0, forces=1.0)` |
| `seed` | 185 |

## Dataset

[`data/optb88/cameron/split_dataset_r5.0_{train,val,test}.xyz`](../../data/optb88/cameron/) (the pre-existing splits from the multifidelity tarball; same files used by `0-m0_complexity_probe`).

- train: 1956 frames
- val:    244 frames
- test:   244 frames

Species `{C, O, Pt}`. Mostly mixed CO/Pt frames with bulk-Pt and gas-phase CO classes (see `0-m0_complexity_probe/README.md` for composition class breakdown).

## Hypothesis

The L-config Allegro generalises cleanly to the 3-species cameron set. We expect per-atom energy MAE on the test split well below `0-m0_complexity_probe`'s `baseline-B` reference, since L has ~10x more parameters and `l_max=3` vs `l_max=1`. Falsified if test force MAE is worse than baseline-B, suggesting capacity/regularisation issues specific to multi-species training.

## Success criteria

- Training completes (early-stop or `max_epochs=5000`) without NaN.
- Test force MAE notably better than `baseline-B`.
- `compile_mode: compile` works on Frontier ROCm (no fallback to eager).
- Checkpoint packages without errors and round-trips through `ModelFromPackage`.

## Run

```bash
cd /lustre/orion/mat281/scratch/demirand/projects/sparse_delta
sbatch experiments/2-allegro_L_correction/train.sh
```

Outputs:

- Hydra run dir: `/lustre/orion/mat281/scratch/demirand/projects/sparse_delta/runs/2-allegro_L_correction/` — checkpoints (`best*.ckpt`, `last.ckpt`), wandb local dir, lightning logs.
- SLURM logs: `experiments/2-allegro_L_correction/logs/`.

**Frontier walltime policy.** OLCF's bin policy caps 1-node jobs at 2 h walltime regardless of QoS (`debug`, `normal`, `extended` all enforce it on small node counts). `train.sh` therefore requests `--time=01:55:00`. To reach the lightning-side `max_time=1d`, chain submissions: each job restarts from `last.ckpt` in the hydra run dir, and the next is enqueued via `--dependency=afterok:<jobid>` (or manually after the first finishes). The wandb run resumes by `id` automatically as long as the run dir is reused.

On success, follow [`notes/workflow.md`](../../notes/workflow.md) curation step: copy best ckpt + nequip package to `saved_models/packages/2-allegro_L_correction.nequip.zip`, append a row to `saved_models/best_checkpoint_paths.csv`, flip status to `done-success`, paste the wandb run URL above.

## Environment notes (Frontier)

This is the first training run on Frontier from sparse_delta. The project's `pyproject.toml` exposes cluster-specific deps as extras:

- `uv sync --extra rocm62` → AMD MI250X / ROCm 6.2.4 wheels (this cluster).
- `uv sync --extra cuda` → NVIDIA wheels + cuequivariance (FASRC or similar).

The two extras are declared `conflicts` so only one can be installed at a time. On Frontier we use `rocm62`. Notes:

- `cuequivariance-*` is in the `cuda` extra and not installed here; the `enable_CuEquivarianceContracter` modifier is omitted from the config accordingly.
- ROCm runtime: `module load amd-mixed/6.2.4` (in `train.sh`).
- `compile_mode: compile` retained — ROCm-compatible.
- Verified: `M0-baseline-B.nequip.zip` loads and forward-passes on the login-node MI210; compute nodes have MI250X (same `gfx90a`).

## Outcome

_Filled in after the run._

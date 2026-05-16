# 2-allegro_L_correction

**Status:** running
**Date:** 2026-05-15
**Outer SHA:** `e3fb27c1`
**Submodule SHAs:** `nequip-private=c2ec1f2b  allegro-private=82d7258`
**WandB:** [wandb.ai/demiranda-gabriel/sparse-delta/runs/o2p1l6zr](https://wandb.ai/demiranda-gabriel/sparse-delta/runs/o2p1l6zr) (group `2-allegro_L_correction`). Run logs to `WANDB_MODE=offline` first; publish with `wandb sync runs/2-allegro_L_correction/wandb/offline-run-20260515_235128-o2p1l6zr` from a login node.

## Intent

First training run on the cameron CO/Pt dataset in `sparse_delta`. Train a large Allegro model with the `l-l3-paper` architecture from [`mtfd/experiments/12-bulk_pt_relabel_sweep`](../../../mtfd/experiments/12-bulk_pt_relabel_sweep/) — the largest tag in that 5-tag bulk-Pt sweep. Adapted for:

- **Frontier (AMD MI250X / ROCm 6.4.2).** Drop `enable_CuEquivarianceContracter` (NVIDIA-only); keep `compile_mode: compile`.
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

First training run on Frontier from sparse_delta. The project's `pyproject.toml` exposes cluster-specific deps as extras:

- `uv sync --extra rocm64` → AMD MI250X / ROCm 6.4 wheels (this cluster); pins `torch==2.9.1+rocm6.4` + `pytorch-triton-rocm==3.5.1`.
- `uv sync --extra cuda` → NVIDIA wheels + cuequivariance (FASRC or similar).

The two extras are declared `conflicts` so only one can be installed at a time. On Frontier we use `rocm64`. Notes:

- The earlier `rocm62` extra was abandoned: the rocm6.2 wheel index tops out at torch 2.5.1, which is below nequip/allegro's `>=2.6` PT2-compile requirement. Migrating to rocm6.4 wheels + the `amd-mixed/6.4.2` module solves this.
- `cuequivariance-*` is in the `cuda` extra and not installed here; the `enable_CuEquivarianceContracter` modifier is omitted from the config accordingly.
- ROCm runtime: `module load amd-mixed/6.4.2` (in `train.sh`).
- `compile_mode: compile` retained — ROCm-compatible from torch 2.6 onward.
- Verified: torch 2.9.1+rocm6.4 reports `torch.version.hip = "6.4.43484-..."` and `torch.cuda.device_count() == 1` on the login-node MI210; compute nodes have MI250X (same `gfx90a`).

## Outcome

_In progress as of 2026-05-15 23:51._ Job 4592647 (the first to actually train) reached Epoch 1 and wrote `runs/2-allegro_L_correction/best.ckpt` after ~5 minutes of wall-clock; wandb run id `o2p1l6zr`.

### Deviations from the spec above (driven by Frontier MI250X / 64 GiB GCD memory)

Reaching a running state required eleven failed submissions before training started; each fix is a commit on master. The cumulative deviations from the README's "Architecture" and "Training" tables are:

| Field | Spec | Running with | Reason |
|---|---|---|---|
| `num_layers` | 4 | **3** | Backward stored ~33 GiB intermediates at 4 layers (job 4592631). |
| `num_tensor_features` | 64 | **32** | `tp_path_channel_coupling=true` forward intermediate hit 67 GiB at 64 channels (jobs 4592436 / 75 / 85). |
| `batch_size` | 16 (global) | **8 (global = 1 × 8 DDP ranks)** | Per-GCD memory budget. |
| `compile_mode` | `compile` | **`eager`** | Tried to rule out PT2 graph-capture peak; OOM persisted under eager, so this stayed for the run. |
| `Trainer.precision` | (default fp32) | **`bf16-mixed`** | AMP-style; halves activation memory. Params stay fp32 (nequip's `model_dtype` doesn't accept bf16). |

What's **unchanged** vs spec: `l_max=3`, `r_max=7.0`, `parity=true`, `num_scalar_features=128`, all MLP shapes, the Bessel basis, optimizer/loss/scheduler/early-stop config, dataset paths, seed.

The Frontier-side environment + DDP / SLURM setup landed along the way (some of it is project infra rather than experiment-specific):

- Migrated the project's ROCm extra from `rocm62` (torch 2.5.1 — below nequip's PT2 floor of 2.6) to `rocm64` (torch 2.9.1) + `amd-mixed/6.4.2` module.
- Pinned the venv to a uv-managed `cpython-3.12.13` (`/.python-version`) — system Python 3.12 is installed on Frontier without dev headers, breaking triton's HIP shim compile.
- `srun` + `--ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=7` for DDP × 8 GCDs.
- `-q debug` for ~10× faster queue placement vs `normal` at the same 2 h walltime cap.
- `WANDB_MODE=offline` (compute nodes have no outbound internet).
- `PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512` to tame backward-pass fragmentation.
- A circular-import fix in [`software/sparse-delta-core/sparse_delta_core/__init__.py`](../../software/sparse-delta-core/sparse_delta_core/__init__.py) (lazy `__getattr__` on `.model` / `.features` / `.nn`) plus a narrower entry point in that package's `pyproject.toml` (`init_always = "sparse_delta_core._keys"`) — sparse-delta-core's eager `.model` import re-entered nequip during nequip's own `init_always` entry-point load.

These are tracked in commits between `665f7d5` (experiment scaffold) and `b8c433d` (the running configuration).

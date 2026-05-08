# sparse_delta

Sparse, smooth, locally-triggered multi-fidelity correction for machine-learning interatomic potentials.

A small foundation model `M0` (Allegro or 1-layer NequIP) runs everywhere. Its hidden features feed a smooth, compactly-supported gate `λ_i ∈ [0, 1]` that is *exactly zero* in simple environments. A larger correction model `M1` (strictly local) runs only on the active subgraph. Total energy:

```
E_total = E_0 + Σ_i λ_i · E_1^i
```

Forces are conservative (single autograd backward through the whole graph) and `M1` compute scales with `|active atoms| / N_total`, not with `N_total`.

The full design rationale, math, halo analysis, training recipes, and first-prototype skeleton live in [`notes/sparse_local_correction.md`](notes/sparse_local_correction.md). Read that file before contributing.

## Layout

- `software/` — editable uv-workspace packages (code repos live here).
  - `nequip-private/` — NequIP fork (mir-group, read-only). Cloned via `software/install.sh`.
  - `allegro-private/` — Allegro fork (mir-group, read-only). Cloned via `software/install.sh`.
  - `sparse-delta-core/` — sparse-delta-specific package (gate, complexity score, two-pass driver, custom losses, …). Editable workspace member, this is where new code lands.
- `experiments/` — git source-of-truth for runs: per-experiment configs, SLURM scripts, READMEs. One folder per experiment with reverse-chronological index in [`experiments/README.md`](experiments/README.md). See [`notes/workflow.md`](notes/workflow.md).
- `runs/` — scratch outputs from training jobs. **Gitignored**, disposable, expected to be wiped (FASRC scratch 90-day purge). Nothing authoritative lives here.
- `data/` — datasets. **Gitignored**. Persistent storage on holylabs at `/n/holylabs/LABS/kozinsky_lab/Users/demiranda/projects/sparse_delta/data/` once that path is set up; symlink in.
- `models/` — model artifacts (pre-trained M0 packages, etc.). **Gitignored**. Persistent storage on holylabs analogously.
- `saved_models/` — curated `.nequip.zip` packages + `best_checkpoint_paths.csv` registry, organized as `saved_models/packages/<exp_name>/`. Symlink to holylabs.
- `notes/` — design and status docs (`sparse_local_correction.md`, `workflow.md`, `next_steps.md`, …).
- `scripts/` — utility scripts (sweep helpers, checkpoint surgery, plotting, …).
- `pyproject.toml` / `uv.lock` — uv workspace config.

## Environment / tooling

- Python 3.10 via `uv` workspace (see `pyproject.toml`).
- Bootstrap: `cd software && bash install.sh`. This clones the read-only mir-group forks of `nequip-private` and `allegro-private`, then `uv sync` registers them and `sparse-delta-core` as editable workspace members.
- Code may live on scratch, but persistent data and model reads/writes should go through `/n/holylabs/LABS/kozinsky_lab/Users/demiranda/projects/sparse_delta/`.
- SLURM: A100 80 GB for fine-tuning; H200 141 GB if M1 grows large. Use existing kozinsky_lab partitions.
- WandB project: `sparse-delta`.

## Workflow for runs and experiments

Three tiers, mirroring the convention used in the sibling `multifidelity` project:

- **git** (`experiments/<name>/`) — reproducibility source-of-truth. Every experiment commits its config, SLURM script, and README *before* `sbatch` so the SHA pins the submitted state.
- **holylabs** (`saved_models/`, `models/`, `data/`, `archive/`) — preserved artifacts.
- **scratch** (`runs/<name>/`) — disposable outputs, 90-day purge. Nothing authoritative.

### Per-experiment recipe

1. Copy `experiments/_template/` → `experiments/<name>/`.
2. Fill the README header: status, intent, hypothesis, success criteria, outer + submodule SHAs.
3. Commit and push **before** `sbatch`. Bundle submodule pointer bumps in the same commit.
4. Add a row to the `## Experiments index` table in [`experiments/README.md`](experiments/README.md) (top of table, reverse-chronological) in the same commit that creates the experiment directory.
5. Run via SLURM. SLURM stdout/stderr live next to the run (`<exp>/<tag>/logs/` for sweeps, `runs/<name>/logs/` for single runs).
6. On success: copy best checkpoint + `.nequip.zip` to `saved_models/packages/<exp>-<tag>.nequip.zip`; append a row to `saved_models/best_checkpoint_paths.csv` (`experiment, tag, holylabs_path, wandb_url, git_sha, date`); update the experiment README to `Status: done-success`.

### Sweeps

Use the nested-config pattern: `tags.txt` + `configs/{base,<tag>}.yaml` + `<tag>/train.sh` + a self-locating `submit_all.sh`. Single-run experiments stay flat (`config.yaml + train.sh`).

### Status vocabulary

`planned | running | done-success | done-fail | abandoned`

### WandB

`project=sparse-delta`, `group=<experiment>`, `name=<tag>`. Paste the run URL into the per-tag README on submission.

## Architecture notes (preview — full version in design doc)

- Two-pass inference: M0 forward (full system) → score `s_i` → gate `λ_i` → identify active set `A` → M1 forward on subgraph (edges with receiver `∈ A`) → compose `E_total = E_0 + Σ_{i ∈ A} λ_i · E_1^i` → single autograd backward → conservative forces.
- M1 must remain **strictly local** (Allegro at any depth, or NequIP with `num_layers=1`) so the receptive-field halo stays at one neighbor shell. Multi-layer message-passing M1 voids the savings argument.
- Gate is a polynomial cutoff (`1 − 6u^5 + 15u^4 − 10u^3` style) chosen because it is identically zero below threshold and `C^2`-smooth across the boundary. This gives both exact sparsity and conservative forces.

## Critical gotchas (carried from sibling project — applies here too)

- **Type embedding**: Allegro foundation-model packages use a 89-element vocab (C→5, O→7, Pt→77, …). Hardcoding `["C","O","Pt"]` maps to rows 0/1/2, which are NaN. Use `type_names_from_package:${model_path}`.
- **Compile mode**: all Allegro / `ModelFromPackage` runs should set `compile_mode: compile`. Do not submit models with `compile_mode: eager` outside short debugging experiments.
- **Checkpoints**: prefer `best-vN.ckpt` over `best.ckpt` if you observe NaN (known `check_finite: false` save bug in older nequip).
- **PerTypeScaleShift**: when loading a checkpoint onto a new dataset, wrap with `nequip.model.modify` + `modify_PerTypeScaleShift` using training-data stats.
- **r_max**: combined neighbor lists use `max(r_max_per_model)`; per-submodel cutoffs applied via edge trimming at forward time.

## Conventions

- Don't introduce nested `ForceStressOutput` on submodels — breaks the single-backward-pass guarantee that makes forces conservative.
- All submodels share `type_names` from the outer composite model.
- DFT reference functional follows the parent project: **optb88-vdW**. Canonical INCAR template inherited from sibling project.

## First task for the agent

See [`START_HERE.md`](START_HERE.md). The first job is to bootstrap the environment: clone the read-only forks, install the uv workspace, and verify that `nequip` / `allegro` import and a trivial training step works.

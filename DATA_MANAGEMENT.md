# DATA_MANAGEMENT.md

Classification of every top-level entry in this project. See user-scoped policy in `~/.claude/CLAUDE.md` and the `backup-to-gdrive` skill for the authoritative push/pull workflow.

Project name (auto-resolved from git root): `sparse_delta`.
Drive layout: `gdrive:projects/sparse_delta/<subpath>/`.
Holylabs root: `/n/holylabs/LABS/kozinsky_lab/Users/demiranda/projects/sparse_delta/`.

## git-tracked

Committed; GitHub is source-of-truth.

- `CLAUDE.md` — project instructions for Claude Code.
- `README.md` — top-level project overview.
- `START_HERE.md` — first-task pointer for new agents.
- `DATA_MANAGEMENT.md` — this file.
- `pyproject.toml` — uv workspace config.
- `uv.lock` — pinned dependency graph.
- `.gitignore` — ignore rules.
- `.python-version` — Python version pin (3.12.13).
- `experiments/` — per-experiment configs, SLURM scripts, READMEs, results/ subdirs. Reproducibility source-of-truth. Small SLURM stdout/stderr kept next to a run (`<exp>/<tag>/logs/*.{out,err}`) is tracked as provenance; bulky run logs may be left untracked (local-only) rather than committed.
- `notes/` — design and status docs.
- `scripts/` — utility scripts.
- `software/install.sh` — bootstrap script for the workspace (clones read-only forks).
- `software/sparse-delta-core/` — editable workspace member; sparse-delta-specific code lands here.

## gdrive-tracked

Bulk data; mirrored to `gdrive:projects/sparse_delta/<subpath>/` via `gdrive-push` / `gdrive-pull` / `gdrive-archive`. Not committed. Persistent local copy lives on holylabs (symlinked in at runtime).

- `data/` → **shared with sibling project**, pulled from `gdrive:projects/multifidelity/data/` (not `sparse_delta/data/`). Restore with `PROJECT_NAME=multifidelity gdrive-pull data/data-YYYY-MM-DD.tar.gz data` then `tar -xzf` in place. Current layout:
  - `optb88/bulk_pt/` — bulk-Pt train/val/test splits, EOS-augmented training set, relabel audit, EOS curve.
  - `optb88/cameron/` — Cameron CO/Pt nanoparticle + slab dataset; `full_dataset_r5.xyz` + `split_dataset_r5.0_{train,val,test}.xyz`.
  - `original/` — source `full_dataset_r5.xyz` and the `multi_preprocess.py` / `preprocess.ipynb` used to derive the splits.
  Authoritative copy stays in the multifidelity project's Drive subtree; do **not** re-archive into `gdrive:projects/sparse_delta/data/` unless sparse_delta diverges from the shared dataset.
- `models/` → `gdrive:projects/sparse_delta/models/`. Pre-trained model artifacts (e.g. packaged M0). Symlink target: `/n/holylabs/.../models/`. Currently absent.
- `saved_models/` → `gdrive:projects/sparse_delta/saved_models/`. Curated `.nequip.zip` packages + `best_checkpoint_paths.csv` registry. Currently a real directory on scratch; promote to a holylabs symlink once that path is provisioned.

## local-only

Ephemeral, regenerable, environment-specific. Not committed, not backed up.

- `.git/` — local git metadata.
- `.venv/` — uv-managed virtualenv.
- `.vscode/` — editor settings.
- `runs/` — scratch training outputs (FASRC 90-day purge). Includes `lightning_logs/`, `wandb/`, `slurm-*.out` written alongside.
- `software/nequip-private/` — read-only mir-group fork, cloned by `software/install.sh`. Track upstream SHA via `notes/`, not via outer git.
- `software/allegro-private/` — read-only mir-group fork, ditto.
- `__pycache__/`, `*.egg-info/`, `.ipynb_checkpoints/`, `.uv/`, `uv.lock.bak`, OS junk (`.DS_Store`, `.idea/`, `*.swp`) — build/editor cruft.
- `termpdf.log` (any dir) — md-view / termpdf temp output. Gitignored.
- Bulky untracked per-experiment SLURM logs (e.g. `experiments/*/logs/`, `experiments/*/_*/logs/`) not committed as provenance — disposable.

## Rules (recap)

1. Anything in `.gitignore` is gdrive-tracked or local-only.
2. Anything not in `.gitignore` is git-tracked.
3. New top-level entry → classify in the same commit that introduces it.
4. Before deleting a local copy of a gdrive-tracked path, confirm a recent push exists (`gdrive-push -n` first).

## Polaris mirror (2026-06-12)

Mirrored to `/lus/eagle/projects/HetRxnEnergy/demiranda/projects/sparse_delta`
on Polaris (ALCF) ahead of FASRC maintenance. Cluster-specific deltas:

- `data/` pulled from the shared multifidelity Drive subtree
  (`optb88/`, `original/`) per the policy above; `saved_models/` pulled
  from `sparse_delta/saved_models/`. Both are real directories (no
  holylabs on Polaris).
- `software/` forks pinned to the FASRC state: `nequip-private`
  @ `74c7689f`, `allegro-private` @ `82d7258` with both
  `sparse-delta-core/patches/` applied by `software/install.sh`
  (verified byte-identical to the FASRC working tree).
- `software/sparse-delta-core` is a nested standalone repo
  (`demiranda-gabriel/sparse-delta-core`, here @ `2d274e3`) — the
  superproject records it as a bare gitlink, so fresh clones get an
  empty directory until it is cloned explicitly.
- SLURM launchers do not apply on Polaris — submit via HyperQueue.

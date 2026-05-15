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
- `experiments/` — per-experiment configs, SLURM scripts, READMEs, results/ subdirs. Reproducibility source-of-truth.
- `notes/` — design and status docs.
- `scripts/` — utility scripts.
- `software/install.sh` — bootstrap script for the workspace (clones read-only forks).
- `software/sparse-delta-core/` — editable workspace member; sparse-delta-specific code lands here.

## gdrive-tracked

Bulk data; mirrored to `gdrive:projects/sparse_delta/<subpath>/` via `gdrive-push` / `gdrive-pull` / `gdrive-archive`. Not committed. Persistent local copy lives on holylabs (symlinked in at runtime).

- `data/` → `gdrive:projects/sparse_delta/data/`. Datasets. Symlink target: `/n/holylabs/LABS/kozinsky_lab/Users/demiranda/projects/sparse_delta/data/`. Currently absent; symlink in once dataset is staged. See [`memory/project_dataset.md`].
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

## Rules (recap)

1. Anything in `.gitignore` is gdrive-tracked or local-only.
2. Anything not in `.gitignore` is git-tracked.
3. New top-level entry → classify in the same commit that introduces it.
4. Before deleting a local copy of a gdrive-tracked path, confirm a recent push exists (`gdrive-push -n` first).

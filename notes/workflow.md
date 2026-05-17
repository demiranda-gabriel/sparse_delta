# Workflow

This file captures the day-to-day workflow for runs and experiments in `sparse_delta`. It mirrors the convention used in the sibling `multifidelity` project.

## Three tiers of persistence

| Tier | Path | Tracked? | Purpose |
|---|---|---|---|
| git | `experiments/<name>/` | yes | Reproducibility source-of-truth: configs, SLURM scripts, READMEs, scripts, **and result-grade artifacts (figures, output datasets, summary files) under `experiments/<name>/results/`**. |
| holylabs | `saved_models/`, `models/`, `data/`, `archive/` | symlinks only | Preserved artifacts (pre-trained models, datasets, curated checkpoints, large result artifacts > ~50 MB that don't belong in git). |
| scratch | `runs/<name>/` | no | Disposable training outputs, FASRC 90-day purge. Workspace only. |

Nothing authoritative lives on scratch. If it matters, copy it to `experiments/<name>/results/` (small, tracked) or `saved_models/` (large, holylabs) and update the registry.

### Where do figures and output datasets go?

**Result-grade artifacts go in `experiments/<name>/results/`**, not `runs/`. This includes:
- Figures (`*.png`, `*.pdf`).
- Output datasets the experiment was supposed to produce (e.g. an extxyz with custom per-atom fields, a summary `.npz`, a `summary.json`).
- The captured run log if it's the only record of run-time metrics (when there is no WandB run).

`runs/<name>/` is for working / intermediate output during a run, not the deliverable. If a result file is over ~50 MB, ask before adding to git — it likely belongs in `saved_models/` (holylabs) with a row in `saved_models/best_checkpoint_paths.csv` pointing at the holylabs path.

## Per-experiment recipe

1. Pick the next experiment number `N = max(existing N) + 1` from the index in [`../experiments/README.md`](../experiments/README.md) (oldest is `0`). Copy `experiments/_template/` → `experiments/<N>-<exp_name>/`.
2. Fill the README header (status, intent, hypothesis, success criteria, outer + submodule SHAs).
3. Commit and push the experiment directory **before** `sbatch`. This pins the SHA to the submitted state. Bundle any submodule pointer bumps in the same commit.
4. Add a row to the `## Index` table in [`../experiments/README.md`](../experiments/README.md) (top of table, reverse-chronological by date; the `N` column is monotonically increasing) in the **same** commit that creates the experiment directory.
5. Run via SLURM. SLURM stdout/stderr live next to the run:
   - sweeps: `<exp>/<tag>/logs/`
   - single runs: `runs/<name>/logs/`
6. WandB: `project=sparse-delta`, `group=<experiment>`, `name=<tag>`. Paste the run URL into the per-tag README on submission.
7. On success:
   - Copy the best checkpoint and `.nequip.zip` to `saved_models/packages/<exp>-<tag>.nequip.zip`.
   - Append a row to `saved_models/best_checkpoint_paths.csv` with columns `experiment, tag, holylabs_path, wandb_url, git_sha, date`.
   - Update the experiment README to `Status: done-success`.

## Sweep pattern

Use the nested-config layout:

```
<exp_name>/
├── tags.txt              # one tag per line
├── configs/
│   ├── base.yaml         # shared defaults
│   ├── <tagA>.yaml       # diffs from base
│   └── <tagB>.yaml
├── <tagA>/train.sh       # per-tag SLURM script
├── <tagB>/train.sh
└── submit_all.sh         # self-locating wrapper, sbatch-es all tags
```

Single-run experiments stay flat (`config.yaml + train.sh`).

## Submodule push order

When updating either of the read-only forks (`software/nequip-private`, `software/allegro-private`), do not push — they are upstream and read-only. Local commits are fine, but rely on outer-repo SHA pinning to capture them.

For any project-owned submodule (e.g. a future `sparse-delta-core` extracted into a separate repo): always push the inner submodule first, then bump the outer pointer in a separate commit. Outer-only pointer bumps without a pushed inner SHA leave the project unbuildable.

## Status vocabulary

`planned | running | done-success | done-fail | abandoned`

## What goes in `notes/`

Design docs, status snapshots, and analyses that survive the experiment that produced them. Specifically:

- `sparse_local_correction.md` — the project design doc. Read this first.
- `next_steps.md` — running list of follow-up ideas, refreshed as work progresses.
- `summary_<topic>.md` — periodic status snapshots when work finishes a phase.

Avoid putting per-experiment scratch in `notes/`; that lives next to the run.

# START HERE — agent initialization for sparse_delta

If you have just been spawned in this repository, read this file first. Then read [CLAUDE.md](CLAUDE.md) and [notes/sparse_local_correction.md](notes/sparse_local_correction.md) before doing anything else.

## What this project is

`sparse_delta` is a research project on **sparse, smooth, locally-triggered multi-fidelity correction** for machine-learning interatomic potentials.

Concept in one paragraph:

> A small foundation model `M0` (Allegro or 1-layer NequIP) runs everywhere on a system. Its hidden features feed a smooth, compactly-supported gate `λ_i ∈ [0, 1]` that is *exactly zero* in simple environments and ramps to 1 in complex / out-of-distribution environments. A larger, strictly-local correction model `M1` (Allegro at any depth or NequIP with `num_layers=1`) runs only on the active subgraph. Total energy is `E = E_0 + Σ_i λ_i · E_1^i`. Forces are conservative because everything is differentiable and the gate is `C^2`-smooth across the on/off boundary. `M1` compute scales with `|active| / N`, not with `N`.

The full motivation, math, halo analysis, choice of complexity signal, force-conservation argument, training degeneracies, and first-prototype skeleton are spelled out in [notes/sparse_local_correction.md](notes/sparse_local_correction.md). **Read that file before writing any code.**

## What this project is *not*

- Not Δ-learning (which evaluates the correction everywhere — no compute savings).
- Not adaptive resolution simulation (which uses spatial regions and suffers boundary artifacts).
- Not Top-K mixture-of-experts (which is non-conservative).
- Not active learning (the gate is an inference-time conditional-computation mechanism, not an offline data acquisition strategy).

## Your first task: bootstrap the environment

The repository has been scaffolded but no code is installed. Your job is to get a working Python environment with `nequip-private` and `allegro-private` installed editable, plus the empty `sparse-delta-core` package recognized as a workspace member. Follow the steps below and report any deviations.

### Step 0 — verify the layout

Confirm the following directories exist:

```
sparse_delta/
├── CLAUDE.md
├── README.md
├── START_HERE.md
├── pyproject.toml
├── .gitignore
├── software/
│   ├── install.sh
│   └── sparse-delta-core/
│       ├── pyproject.toml
│       └── src/sparse_delta_core/__init__.py
├── experiments/
│   ├── README.md
│   └── _template/
├── notes/
│   ├── sparse_local_correction.md
│   └── workflow.md
├── runs/         # empty, gitignored
└── scripts/      # empty
```

If anything is missing, stop and ask.

### Step 1 — clone the read-only forks

`software/install.sh` is set up to clone `mir-group/nequip-private` and `mir-group/allegro-private` from GitHub via SSH. Both are **read-only** — never push to them. (This is a hard project rule; the sibling `multifidelity` project memory carries the same constraint.)

Run:

```bash
cd software
bash install.sh
```

Verify each clone landed at `software/nequip-private/` and `software/allegro-private/` and that the `develop` branch is checked out.

### Step 2 — uv workspace sync

From the project root:

```bash
uv sync
```

This should pick up the workspace members declared in `pyproject.toml` (`software/nequip-private`, `software/allegro-private`, `software/sparse-delta-core`) and resolve a venv at `.venv/`.

### Step 3 — sanity check imports

```bash
uv run python -c "import nequip; import allegro; import sparse_delta_core; print('ok')"
```

Expected output: `ok`.

If any of these imports fail, do not work around — diagnose. Common causes:

- SSH key not configured for GitHub on this host (forks won't clone).
- Wrong Python version (project requires `>=3.10`).
- `cuequivariance-*` wheels missing for the local CUDA toolkit (these are listed as runtime dependencies; if they fail to resolve, ask the user before pinning a different version).

### Step 4 — submodules vs plain clones

`install.sh` uses plain `git clone` for speed. The current scaffold does **not** add `nequip-private` or `allegro-private` as git submodules of the outer repo, by design — defer that decision until experiments start. When the first experiment is created, convert the relevant trees to submodules so the experiment commit pins exact submodule SHAs (per the workflow in [CLAUDE.md](CLAUDE.md)).

Note: this differs from the sibling `multifidelity` project, which has them as submodules already. We're starting clean here.

### Step 5 — set up holylabs symlinks (defer — not part of first task)

When the first experiment runs, you will need:

- `data/` → `/n/holylabs/LABS/kozinsky_lab/Users/demiranda/projects/sparse_delta/data/`
- `models/` → `/n/holylabs/LABS/kozinsky_lab/Users/demiranda/projects/sparse_delta/models/`
- `saved_models/` → analogous

Do not create these now. Ask the user to create the holylabs side first; then we symlink in.

### Step 6 — report

Report back to the user with:

1. Whether install.sh completed without errors.
2. Output of `uv run python -c "import nequip; import allegro; import sparse_delta_core; print('ok')"`.
3. Any deviations from the expected layout.
4. Anything ambiguous you needed to decide and what you chose.

Then **stop and wait** before proceeding to any code work. The user wants to confirm the environment before we start building the gate / complexity score / two-pass driver.

## After bootstrap — likely next steps (preview only)

Following sections of [notes/sparse_local_correction.md](notes/sparse_local_correction.md):

- §11 — first-prototype skeleton (train M0 → build cheap complexity signal → calibrate `(s_low, s_high)` → train M1 on residual → two-pass inference → NVE conservation check).
- §12 — distribution diagnostics for `s_i` across system types (decides whether the idea is worth pursuing).

Do not start any of this until the user confirms the bootstrap is healthy.

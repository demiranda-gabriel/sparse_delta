# 5-composite_staged

**Status:** done-success (stages 1-3 complete; sparsity_coeff sweep pending)
**Date:** 2026-05-25 (submit) → 2026-05-28 (chain complete)
**Outer SHA:** `f91c90b` (chain) / `26c8caf` (parent)
**Submodule SHAs:** `sparse-delta-core=2d274e3`
**WandB:**
- stage 1 → [`vrjc4hiv`](https://wandb.ai/demiranda-gabriel/sparse-delta/runs/vrjc4hiv)
- stage 2 → [`ay5b5zes`](https://wandb.ai/demiranda-gabriel/sparse-delta/runs/ay5b5zes)
- stage 3 → [`vzqofs9q`](https://wandb.ai/demiranda-gabriel/sparse-delta/runs/vzqofs9q)

## Outcome (2026-05-28)

| Stage | Partition | Wall | Test forces_mae | Test per_atom_E mae | Notes |
|---|---|---|---|---|---|
| 1 (M0 alone) | kozinsky_gpu | 59 min | — | — | Stage 1 metrics not surfaced (M0 standalone tail). See wandb. |
| 2 (M1, λ=1 const) | seas_gpu | 1h41m | **0.073 eV/Å** | 0.0058 eV/atom | **Beats exp 4 joint baseline (0.089).** |
| 3 (learned gate) | seas_gpu | 44 min | 0.096 eV/Å | 0.0081 eV/atom | Regresses vs stage 2 — `sparsity_coeff=1.0` too aggressive (see follow-up). |

**Pipeline validation:**
- LoadWeightsCallback / FreezeByNameCallback both fire correctly at
  every stage. Stage 2: 22 M0 params loaded, 15 frozen. Stage 3: 42
  params loaded, 30 frozen (M0+M1+adapter+edge_sh).
- Adapter handles ℓ=1→3 lift; M1's ℓ=2 and ℓ=3 input channels start
  zero, gain capacity through the dedicated `m1_edge_sh`.
- Two-partition delay diagnosed first chain: lab `kozinsky_gpu`
  saturated, jobs sat queued >24 h. Resubmit with multi-partition list
  (`gpu,seas_gpu,kozinsky_gpu,gpu_h200`) + relaxed `--gres=gpu:1`
  scheduled within hours on `seas_gpu`. Saved as
  `feedback_gpu_partitions.md` for future jobs.

**Headline finding:** the staged curriculum's **stage 2** (full-strength
M1 correction everywhere, M0 frozen) produces a **better** model than
exp 4's joint training. Stage 3's learned gate then degrades the model
because the sparsity penalty (`coeff=1.0`, weighted equally with E+F)
trades useful M1 correction for sparsity. The right move is a
sparsity-coefficient sweep (see follow-up below).

## Artifacts

Best checkpoints + companion `state_dict.pt` files mirrored to
holylabs:

```
/n/holylabs/LABS/kozinsky_lab/Users/demiranda/projects/sparse-delta/saved_models/packages/
├── 5-composite_staged-stage1_m0.ckpt       (~500 KB, M0 standalone)
├── 5-composite_staged-stage1_weights.pt    (stage 2 input)
├── 5-composite_staged-stage2_m1.ckpt       (~5.4 MB, M0+adapter+M1 best)
├── 5-composite_staged-stage2_weights.pt    (stage 3 input)
└── 5-composite_staged-stage3_gate.ckpt     (~3.5 MB, learned-gate composite)
```

CSV rows added to `saved_models/best_checkpoint_paths.csv` under
experiment `5-composite_staged` (tags `stage1_m0`, `stage2_m1`,
`stage3_gate`). `.nequip.zip` packaging deferred per the same
entry-point bug noted on exp 4 (sparse-delta-core's
`nequip.extension` entry currently scopes to `._keys` only; needs to
broaden to `sparse_delta_core` so the packager interns the
`nn`/`model` submodules).

## Follow-up: sparsity_coeff sweep

Stage 3 regression motivates a focused sweep of `sparsity_coeff`
keeping stage 1 + stage 2 weights frozen (only the gate retrains).
See `stage3_sweep_sparsity/`.

## Intent

Staged-from-scratch counterpart to [`4-composite_warmstart`](../4-composite_warmstart/).
Decompose joint training into three sequential stages instead of one joint
end-to-end optimisation:

1. **Stage 1 — M0 standalone.** Train a fresh M0 (Allegro, `l_max=1`,
   `num_layers=2`) with `ForceStressOutput`, energy+force MSE. Freeze.
2. **Stage 2 — M1 with constant λ.** Build warm-start composite with
   `gate_mode="constant"`, `lambda_value=1.0` (gate is a constant — every
   atom gets full M1 correction). Load M0 weights from stage 1, freeze
   M0. Train **adapter + M1** only. Loss = energy + force MSE (no
   sparsity term, no gate to penalise).
3. **Stage 3 — gate.** Build composite with `gate_mode="learned"` (the
   exp 4 `EdgeFeaturesToNodeLambda`). Load M0+adapter+M1 weights from
   stage 2, freeze them all. Train **gate only**. Loss = E + F MSE + α
   · mean(λ) sparsity penalty.

Architectural innovation vs exp 4: M0 `l_max=1` (cheap everywhere-pass)
+ M1 `l_max=2`. Requires a new **TensorTrackAdapter** + a **second
edge-spherical-harmonic embedding** at `l_max_M1=2` so M1's TP can
synthesise higher-ℓ content.

## Hypothesis

Staged optimisation isolates the failure modes of joint warm-start
training. Each stage has a clean, well-conditioned loss landscape:

- Stage 1: identical to a vanilla Allegro fit. Known-good optimisation.
- Stage 2: M1 sees a fixed environment (M0 frozen, λ=1). Convex-like;
  M1 just learns the residual `E_target − E_M0` per edge. No
  competition with M0 over capacity.
- Stage 3: only the gate moves. ≤10K params. Identical to exp 3 path
  but with this-project-trained M0/M1.

**Expectations to falsify:**

- (a) Stage 3's `mean(λ)` settles in `(0.05, 0.30)` on the cameron CO/Pt
  validation set (sparse-but-non-trivial). If it collapses to 0 or
  saturates to 1, the staged curriculum has failed to build a useful
  gate target.
- (b) Final test `forces_mae` ≤ exp 4's 0.089 eV/Å. If staged matches
  or beats joint, the curriculum is preferable for stability. If it
  loses by a lot, joint is better despite the PT2 compile fragility.
- (c) Stage 2 M1 converges with `val_forces_mae` already below the
  stage-1 M0 standalone baseline (M1 should be reducing residual). If
  M1 does **not** improve over M0 alone with λ=1, the M0 representation
  is starving M1 and the warm-start aliasing isn't carrying useful
  signal. Diagnostic, not a falsifier.

## Success criteria

Concrete:

- Stage 1: `test0_epoch/forces_mae` < 0.20 eV/Å (calibration target —
  small M0 is the everywhere-pass, doesn't need to be SOTA).
- Stage 2: `test0_epoch/forces_mae` < 0.15 eV/Å (M1 should bring error
  down meaningfully versus stage 1).
- Stage 3: `test0_epoch/forces_mae` ≤ 0.10 eV/Å AND validation
  `mean(λ)` ∈ `(0.05, 0.40)` on the bulk-Pt sub-split. If gate
  collapses to 0, the model degenerates to stage 1 (and exp fails).

## Architecture (proposed)

### M0 (small, runs everywhere)

| | |
|---|---|
| `l_max` | **1** |
| `parity` | true |
| `num_layers` | 2 |
| `num_scalar_features` | 32 |
| `num_tensor_features` | 32 |
| `allegro_mlp` | depth 1, width 64, silu |
| `readout_mlp` | depth 1, width 32 |
| Bessel radial | 8 bessels, p=6, embed_dim=128 |

### M1 (larger, gated, strictly local)

| | |
|---|---|
| `l_max` | **3** (lifted from M0 via adapter — see below) |
| `parity` | true |
| `num_layers` | **3** (Allegro strictly local at any depth — each layer only sees 1-shell of receiver atom, so multi-layer M1 still has 1-shell halo per active atom) |
| `num_scalar_features` | 128 |
| `num_tensor_features` | 32 (must match M0; v1 constraint) |
| `allegro_mlp` | depth 1, width 128, silu |
| `readout_mlp` | depth 1, width 64 |

### Tensor-track adapter (NEW)

`sparse_delta_core.nn.TensorTrackAdapter`: per-ℓ channel-mixing module
operating in Allegro's strided layout `[E, mul, k]`. Reads M0's
`_allegro_pre_final_tp_out` (irreps `32x0e + 32x1o`, dim = 32×4 = 128
per edge), writes `_warmstart_tensor_adapted` (irreps
`32x0e + 32x1o + 32x2e + 32x3o`, dim = 32×16 = 512 per edge).

- Shared-ℓ paths (ℓ=0, ℓ=1): two learnable `[32, 32]` matrices applied
  to the channel axis (shared across the `2ℓ+1` components — preserves
  equivariance).
- New ℓ paths (ℓ=2, ℓ=3): zero-init by construction (no source-ℓ block
  to mix from). M1's ℓ=2 / ℓ=3 input channels start at zero; they
  gain capacity inside M1's TPs via mixing of
  ℓ=1_input ⊗ ℓ≥2_edge_SH → ℓ ∈ [0, 3] (filtered to ≤ ℓ_max_M1=3).

### M1 edge SH (NEW)

Second `TwoBodySphericalHarmonicTensorEmbed` at `l_max_M1=3`, writes
to a new key `_m1_edge_attrs` (irreps `1x0e + 1x1o + 1x2e + 1x3o`).
M1's `tensor_basis_in_field` reads this key. The M0 edge SH (at
`l_max_M0=1`) is left untouched on the canonical `EDGE_ATTRS_KEY`
since M0's own layers still need it.

Only inserted when `m1_l_max > m0_l_max`. When `m1_l_max == m0_l_max`
the builder takes the existing path (M1 reads M0's edge SH directly).

### Gate

| Stage | `gate_mode` | Implementation |
|---|---|---|
| 1 | n/a | No gate; M0 standalone via `build_m0_standalone`. |
| 2 | `"constant"` | `ConstantNodeLambda(value=1.0)`. Zero params. |
| 3 | `"learned"` | `EdgeFeaturesToNodeLambda` (same as exp 4). Sigmoid, `initial_lambda_mean=0.05`. |

### Loss

| Stage | Loss |
|---|---|
| 1 | `total_energy: 1.0` + `forces: 1.0` (per-atom energy). |
| 2 | same as stage 1; no sparsity term (gate is constant). |
| 3 | stage 1 + `sparsity_penalty: ${sparsity_coeff}` via `GateMeanMetric`. |

### Compile / cuEquivariance

`compile_mode: eager` + `enable_CuEquivarianceContracter` everywhere
(per `notes/pt2_compile_warmstart_nan_study.md`). Eager is the
supported training path; PT2 compile reproducibly NaN'd the warm-start
composite in exp 4's bisect.

## Implementation notes

### New modules in `sparse-delta-core`

1. **`sparse_delta_core.nn.TensorTrackAdapter`**
   `GraphModuleMixin` wrapping `o3.Linear`. Reads M0's pre-final TP
   output, writes adapted tensor. Pure plumbing — see
   "Tensor-track adapter" above for the equivariance argument.

2. **`sparse_delta_core.nn.ConstantNodeLambda`**
   Writes `GATE_KEY` = constant tensor `(num_atoms, 1)` with value
   `lambda_value`. No trainable parameters. Used in stage 2.

3. **`sparse_delta_core.train.LoadWeightsCallback`**
   Lightning callback. In `setup(stage="fit")`, loads a
   `state_dict.pt` into the lightning module's `model` via
   `load_state_dict(..., strict=False)`. Reports matched and missing
   keys to wandb.

4. **`sparse_delta_core.train.FreezeByNameCallback`**
   Lightning callback. In `setup(stage="fit")`, iterates over
   `model.named_parameters()` and sets `requires_grad=False` on any
   whose name matches one of the supplied prefix patterns.

### Modified modules

- **`sparse_delta_core.model.build_warmstart_composite`**
  - Drop `m1_l_max <= m0_l_max` check.
  - New kwarg `gate_mode: Literal["learned", "constant"]` (default
    `"learned"`).
  - New kwarg `lambda_value: float = 1.0` (used when `gate_mode ==
    "constant"`).
  - When `m1_l_max > m0_l_max`: insert second
    `TwoBodySphericalHarmonicTensorEmbed` (writes
    `_m1_edge_attrs`) + `TensorTrackAdapter` (writes
    `_warmstart_tensor_adapted`) after the gate, before M1. Rewire M1's
    `tensor_basis_in_field` and `tensor_features_in_field`
    accordingly.

- **NEW `sparse_delta_core.model.build_m0_standalone`**
  Builds M0-only model via `_build_m0_modules` + `AtomwiseReduce` +
  `ForceStressOutput`. Stage 1 entry point. Keeps
  `expose_pre_final_tp_out=True` so the resulting checkpoint contains
  the architecture needed for stage 2's warmstart aliasing.

### Auxiliary scripts

- **`scripts/extract_model_weights.py`**
  Takes a Lightning `.ckpt` and writes the `model.*` portion of the
  state_dict to a `.pt` file. Used between stages.

### Field-key additions to `sparse_delta_core._keys`

- `M1_EDGE_ATTRS_KEY = "_m1_edge_attrs"` — edge-tensor field, register
  as `edge_fields`.
- `WARMSTART_TENSOR_ADAPTED_KEY = "_warmstart_tensor_adapted"` — edge
  field (Allegro's tensor track lives on edges), register as
  `edge_fields`.

## Stage execution flow

```
stage1_m0/      sbatch train.sh
  → runs/5-composite_staged/stage1_m0/best.ckpt
  → scripts/extract_model_weights.py best.ckpt → stage1_weights.pt

stage2_m1/      sbatch train.sh  (--dependency=afterok:STAGE1)
  config loads stage1_weights.pt via LoadWeightsCallback,
  freezes m0_* via FreezeByNameCallback.
  → runs/5-composite_staged/stage2_m1/best.ckpt
  → scripts/extract_model_weights.py best.ckpt → stage2_weights.pt

stage3_gate/    sbatch train.sh  (--dependency=afterok:STAGE2)
  config loads stage2_weights.pt via LoadWeightsCallback,
  freezes m0_*, m1_*, tensor_track_adapter*, m1_edge_sh* via
  FreezeByNameCallback.
  → runs/5-composite_staged/stage3_gate/best.ckpt
```

`submit_all.sh` chains the three with SLURM `--dependency=afterok`.

## Run

```bash
sbatch experiments/5-composite_staged/submit_all.sh
```

## Outcome

Filled per stage after each completes.

| Stage | Status | WandB | test forces_mae | test per_atom_E mae | mean(λ) |
|---|---|---|---|---|---|
| 1 | planned | — | — | — | n/a |
| 2 | planned | — | — | — | 1.0 (fixed) |
| 3 | planned | — | — | — | — |

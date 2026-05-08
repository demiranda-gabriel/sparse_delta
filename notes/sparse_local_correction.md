# Sparse, Smooth, Locally-Triggered Multi-Fidelity Correction

**Status:** design sketch / seed doc for new project
**Date:** 2026-05-08

## One-liner

Run a small Allegro/NequIP model `M0` everywhere; use its hidden features to compute a smooth per-atom (or per-edge) "complexity" score `s_i`; pass `s_i` through a compactly-supported smooth gate `λ_i = cutoff(s_i)` so it is exactly zero in simple environments; evaluate a larger correction model `M1` only where `λ > 0`. Total energy

```
E_total = E_0 + Σ_i λ_i · E_1^i
```

is conservative (single autograd backward) and *truly sparse* (no `M1` compute on simple atoms), with no boundary discontinuity.

---

## 1. Motivation

In heterogeneous systems (e.g. CO/Pt with both bulk crystal and reactive interfaces), most atoms are in simple, well-sampled environments where a small foundation model is already accurate. Only a fraction sit in distorted / reactive / out-of-distribution environments where a larger, more expressive model is worth the cost.

Existing approaches:

- **Δ-learning** (`E = E_low + E_high_correction`): correction is computed *everywhere*. No savings.
- **Adaptive resolution simulation (AdResS)**: spatial switching function blends two force fields. Notorious for thermodynamic-consistency artifacts at the boundary.
- **Mixture-of-Experts** with Top-K routing: discrete routing → non-conservative forces. Unusable for MD.
- **Active learning / FLARE-style uncertainty triggers**: same complexity signal, but used offline for data acquisition, not as an inference-time gate.

The goal here is genuine inference-time conditional computation for MLIPs that (a) preserves conservative forces and (b) gives real wall-clock savings.

---

## 2. Background: per-atom and per-edge features

This section pins down what "hidden features" look like in the two architectures, since the gate has to be derived from them. Numbers/paths refer to the upstream NequIP / Allegro repos; transplant into the new project as needed.

### 2.1 NequIP — per-node features

Field key: `AtomicDataDict.NODE_FEATURES_KEY = "node_features"`. Tensor shape `(N_atoms, dim)` throughout the model.

| Stage | Module | Irreps | Notes |
|---|---|---|---|
| Type embed | `NodeTypeEmbed` | `num_features × 0e` | Pure scalars, dim `(N, num_features)` |
| Hidden layers `1..L-1` | `ConvNetLayer` / `InteractionBlock` | `feature_irreps_hidden = Nx0e + Nx1o + Nx1e + Nx2e + Nx2o + ...` up to `l_max` | Full equivariant tensor. Per-node dim = Σ_l N_l · (2l+1). |
| Final conv `L` | `ConvNetLayer` | `num_features[0] × 0e` | Forced scalar — only invariants flow into readout. |
| Readout | `ScalarMLP` | `(N, num_features[0]) → (N, 1)` | Per-atom energy. |
| Aggregation | `AtomwiseReduce` | scatter-sum over batch index | Total energy. |

Concrete example: `num_features=64`, `l_max=2`, parity=True.
Hidden per-node = 64 × (1 + 3 + 3 + 5 + 5) = 64 × 17 = 1088 floats.
Pre-readout (final hidden) = 64 scalars per atom.
Per-atom energy = 1 scalar.

The decomposition `1 + 3 + 3 + 5 + 5 = 17` is: `0e` (dim 1) + `1o` (3) + `1e` (3) + `2e` (5) + `2o` (5). With parity=True, every `l > 0` carries both even and odd; `l = 0` only `0e` (`0o` is the pseudoscalar, dropped).

### 2.2 Allegro — per-edge features

Field key: `AtomicDataDict.EDGE_FEATURES_KEY`. Tensor shape `(N_edges, dim)`.

- **Pre-readout edge embedding**: scalar-only (`Nx0e`). Dim = `num_scalar_features × (num_layers + 1)` (DenseNet-style concat of scalar slices from each layer + initial two-body embedding). Higher-`l` content has been contracted out before this point.
- **Final readout**: `ScalarMLP` projecting `(E, dim) → (E, 1)` per-edge energy.
- **Aggregation**: `EdgewiseReduce` scatter-sums per-edge energies over receiver atom (`edge_index[0]`), normalizes by `√(2 · avg_num_neighbors)`, yielding per-atom energies.

Concrete example: `num_scalar_features=64`, `num_layers=2` → edge embedding dim = 64 × 3 = 192. Allegro M1 ≈ 384.

### 2.3 Locality

Both architectures, when configured strictly local, have receptive field exactly `r_max` regardless of depth:

- **Allegro**: by construction. Edge `ij` features depend only on atoms within `r_max` of `i` (edge updates use only edges sharing a center).
- **NequIP**: `num_layers = 1` only. With `L > 1` the receptive field grows multiplicatively: `L · r_max`.

This locality is critical for the sparse-evaluation argument below.

---

## 3. Core idea

1. Run `M0` (small, cheap) on the full system. Record its hidden features.
2. Compute a per-atom complexity score `s_i` from those features.
3. Pass `s_i` through a compactly-supported smooth gate `λ_i = cutoff(s_i)` with `λ_i = 0` for `s_i < s_low` and `λ_i = 1` for `s_i > s_high`.
4. Identify the active set `A = {i : λ_i > 0}`.
5. Run `M1` (larger, strictly local) only on the subgraph induced by `A` and its neighbors.
6. Total energy: `E = E_0 + Σ_{i ∈ A} λ_i · E_1^i`.
7. Forces by single autograd backward through the whole graph.

---

## 4. Mathematical formulation

```
E_total = E_0(R) + Σ_i λ_i(R) · E_1^i(R)

λ_i(R)  = c(s_i(R))                 # compactly-supported smooth gate
s_i(R)  = f(h_i^{M0}(R))            # complexity score from M0 hidden features
```

Forces (well-defined and conservative):

```
F_a = -∂E/∂r_a
    = -Σ_i ∂E_0^i/∂r_a                   # M0 force
      - Σ_i (∂λ_i/∂r_a) · E_1^i           # gate-gradient force (don't forget!)
      - Σ_i λ_i · ∂E_1^i/∂r_a             # gated M1 force
```

The middle term is what makes the dynamics conservative across the switching region. It comes free from autograd as long as `s_i` is differentiable through `M0`.

### Choice of blend

Two reasonable forms:

- **Residual** `E = E_0 + λ E_1`. `M1` learns the *residual* `E_DFT − E_0` in complex regions. Simplest; `M1` cannot be pretrained alone on raw energies.
- **Convex blend** `E = (1 − λ) E_0 + λ E_1`. Each model predicts a full energy; smooth interpolation. Both models run wherever `0 < λ < 1`.

Recommendation: start with the residual form (simpler, stronger sparsity).

---

## 5. The gate: smooth *and* exact zero

A sigmoid never produces an exact zero, so it gives *no compute savings*. A hard threshold gives savings but breaks differentiability and non-conservative forces.

Solution: borrow the polynomial cutoff trick already used for radial cutoffs in NequIP/Allegro:

```
            ⎧ 0                                         s ≤ s_low
λ(s) = c(s) = ⎨ p((s − s_low)/(s_high − s_low))            s_low < s < s_high
            ⎩ 1                                         s ≥ s_high
```

with `p(u) = 1 − 6 u^5 + 15 u^4 − 10 u^3` (Behler / NequIP polynomial cutoff). All derivatives up to order 2 vanish at the boundary. Forces remain `C^2` across the switching region, and the gate is *identically* zero below threshold — so skipping `M1` for those atoms is exact, not an approximation.

For total energy continuity in MD: the polynomial smoothness around `s_low` keeps energies and forces continuous as atoms move in and out of the active set. No discontinuous energy jump when an atom crosses the threshold.

---

## 6. Halo problem and why strictly-local `M1` resolves it

If `M1` is a multi-layer message-passing GNN, evaluating `E_1^i` for `i ∈ A` reaches atoms within `num_layers · r_max` of `i`, regardless of their `λ`. You cannot prune them — they participate in message passing through `A`. Active-set savings collapse if the halo is large.

**With a strictly-local `M1`** (Allegro, or NequIP with `num_layers = 1`):

- Receptive field is exactly `r_max^{M1}`, depth-independent.
- Halo `H = (⋃_{i ∈ A} N_{r_max^{M1}}(i)) \ A` is one neighbor shell thick.
- Halo atoms enter as *neighbor inputs* to `E_1^i` for `i ∈ A`; they do **not** get their own readout, MLP eval, or per-atom energy contribution.
- Compute scales as `O(|A| · deg)`, not `O(|A ∪ H| · deg)`.

Effective speedup ≈ `|A| / N` (modulo subgraph-construction overhead).

### Force on halo atoms

`r_j` for `j ∈ H` enters `E_1^i` for active `i`. By autograd:

```
∂(λ_i · E_1^i)/∂r_j = λ_i · ∂E_1^i/∂r_j  +  (∂λ_i/∂r_j) · E_1^i ≠ 0
```

Halo atoms feel `M1` forces *as neighbors of active atoms*, even though no `E_1` lives on them. Fully conservative; just bookkeeping.

### Edge-level pruning (Allegro-native)

Decompose `E_1 = Σ_{i ∈ A} λ_i · Σ_{j → i} E_ij`. Edge `j → i` is needed iff its receiver `i` is active. Edges with both endpoints inactive: dropped entirely. Edges from halo `j` to active `i`: kept. Build the M1 graph as `{edges with receiver ∈ A}` — single mask over the full neighbor list.

---

## 7. Per-atom complexity signal `s_i`

### From NequIP per-node features

Take node features at output of last hidden conv layer (scalar-only, `(N, num_features[0])`) or any earlier layer (full equivariant). Score is a function of those:

- `s_i = ‖h_i‖₂` — norm of scalar block. One line, zero params.
- `s_i = (h_i − μ)ᵀ Σ^{-1} (h_i − μ)` — Mahalanobis distance to a "simple" reference distribution (e.g. bulk Pt frames). OOD-style, no learned head.
- `s_i = MLP(h_i)` — learnable head; needs regularization to avoid degeneracy.
- Supervised proxy: train head to predict `|E_DFT − E_0|` (rough, but works empirically).

If using equivariant intermediate features, only the scalar (`l=0`) sub-block is invariant; restrict `f(·)` to scalars or use rotation-invariant norms (`‖h_i^{(l)}‖²` per `l`).

### From Allegro per-edge features

Pool over edges incident to atom `i` (the same `EdgewiseReduce` machinery used for energies):

```
z_i = Σ_{j → i} h_ij        # sum-pool of edge embeddings
s_i = MLP(z_i)              # or norm, Mahalanobis, etc.
```

Why sum-pool is the right default:

- **Sum**: `Σ_j h_ij` — smooth in positions because Allegro edge embeddings already carry the radial cutoff envelope (`h_ij → 0` smoothly as `r_ij → r_max`). Atoms entering/leaving the neighbor list cause no jumps. ✓
- **Sum of norms / softplus**: also smooth, density-aware. ✓
- **Mean**: `(1/|N(i)|) · Σ_j h_ij` — denominator is integer, discontinuous across `r_max` crossings. ✗
- **Cutoff-weighted mean**: `Σ_j w(r_ij) h_ij / Σ_j w(r_ij)` with smooth `w` — recovers smoothness. ✓
- **Max-pool**: non-smooth at neighbor swaps. ✗

Concrete dims for Allegro M1 small (`num_scalar_features=64`, `num_layers=2`):
edge feature `h_ij` ∈ ℝ^{192} → pooled `z_i` ∈ ℝ^{192} → `s_i` ∈ ℝ → `λ_i` ∈ [0, 1].

### Per-edge gate (alternative)

Skip the per-atom step entirely:

```
λ_ij = c(t_ij),   t_ij = g(h_ij)
E_1  = Σ_{ij} λ_ij · E_ij^{(1)}
```

Finer-grained sparsity (an atom in mostly-bulk environment with one weird neighbor activates only that edge), no pooling needed, smoothness inherited from `h_ij`.

| | Per-atom gate | Per-edge gate |
|---|---|---|
| Granularity | atom | edge |
| Sparsity ceiling | lower | higher |
| Halo reasoning | clean | murkier (active subgraph defined by edge mask) |
| Native to Allegro | requires pool | yes |
| Interpretability | "active atom" | "active interaction" |

Recommendation: start per-atom; if active fraction is too high, drop to per-edge.

---

## 8. Force conservation across the switching region

Conservative forces require `E_total` to be a single scalar function of `R` and forces to be its negative gradient, computed by a single backward pass. With the smooth compact gate:

- `λ_i` is `C^2` everywhere (poly cutoff).
- `λ_i = 0` exactly outside the active region — skipping `M1` evaluation there is *exact*, not approximate.
- Inside the transition region (`s_low < s_i < s_high`), all three force terms are nonzero and well-defined.
- Inside `λ_i = 1` plateau, gate gradient vanishes (poly cutoff is `C^2` at upper boundary), so dynamics is `M0 + M1` with no switching effect.

Verification protocol: short NVE MD runs spanning the switching region. Energy drift should be at machine precision (or ODE integrator floor); any drift indicates non-conservative implementation.

---

## 9. Training degeneracies and how to avoid them

End-to-end joint training of `(M0, gate, M1)` has two failure modes:

- **Gate → 1 everywhere**: `M1` becomes the model. No savings.
- **Gate → 0 everywhere**: `M1` starves. No correction.

Mitigations:

- **Two-stage**: train `M0` alone; freeze; train gate (supervised by complexity proxy) and `M1` (residual) jointly.
- **Budget constraint**: enforce `mean(λ) ≤ β` via Lagrangian or hard projection. Gives explicit speedup knob.
- **Sparsity penalty**: add `α · ‖λ‖_1` to loss. Tune `α` to target activation fraction.
- **Supervised gate**: bypass end-to-end; train head to predict `|E_DFT^i − E_0^i|` per atom (or use ensemble disagreement, evidential, density). Decouples gate learning from `M1` capacity.

---

## 10. Inference workflow (sparse two-pass)

1. **M0 forward** on full system → per-node or per-edge features → score `s_i` → gate `λ_i`. Cheap.
2. **Active mask**: `A = {i : λ_i > 0}`. Build subgraph: edges with receiver `∈ A`. O(N · deg) mask.
3. **M1 forward** on subgraph → per-atom energies for `i ∈ A` only.
4. **Compose**: `E = E_0 + Σ_{i ∈ A} λ_i · E_1^i`.
5. **Single autograd backward** through both M0 and M1 graphs → conservative forces on all atoms.

Halo atoms participate in step 3 as neighbor positions (not as readout targets) and pick up forces from step 5.

---

## 11. First-prototype skeleton

1. Train `M0` (Allegro small or 1-layer NequIP) alone on the full dataset. Standard config.
2. Build complexity signal cheaply:
   - Allegro: `s_i = ‖Σ_j h_ij‖_2` or Mahalanobis distance against bulk-Pt reference set.
   - NequIP: `s_i = ‖h_i^{(L)}‖_2` (last hidden scalar block) similarly.
3. Pick `(s_low, s_high)` so e.g. ~80% of bulk-Pt atoms have `λ = 0` and ~80% of CO/Pt-surface or reactive atoms have `λ > 0`. Eyeball from histogram first; tune later.
4. Train `M1` (larger Allegro, strictly local) on residual `E_DFT − E_0` with per-atom weight `λ_i`. Forces: total `M0 + λ M1` matched to DFT forces.
5. Inference: implement two-pass with edge-mask subgraph build.
6. Verify conservation: short NVE MD trajectories spanning active/inactive regions.

---

## 12. What to measure early (decides whether the idea is worth pursuing)

- **Distribution of `s_i` across system types.** If unimodal, the cutoff has nothing to bite — the idea fails. If bimodal (bulk vs reactive cleanly separated), the idea wins.
- **Active fraction `|A| / N`** across realistic frames. Determines compute savings ceiling.
- **Halo fraction `|H| / N`** at chosen `r_max^{M1}`. Real speedup is governed by both.
- **Force error in transition region** (`0 < λ < 1`) compared to pure-`M0` and pure-`M1`. If errors spike at the boundary, the smooth gate isn't smooth enough or the residual is too aggressive.
- **NVE energy drift.** Conservation sanity check.
- **Wall-clock M1 forward** for active-only subgraph vs full system. Does the theoretical `|A|/N` speedup actually materialize given GPU/sparsity overhead?

---

## 13. Open design choices

- Per-atom vs per-edge gate (start per-atom).
- Residual `E_0 + λ E_1` vs convex `(1−λ) E_0 + λ E_1` (start residual).
- Complexity signal: norm vs Mahalanobis vs ensemble vs supervised proxy (start Mahalanobis, no extra training).
- Gate cutoff function: poly_5 (NequIP-style) vs other smooth bumps (start poly_5, matches existing radial cutoffs).
- `M0` architecture: Allegro small vs 1-layer NequIP. Allegro preferred (cleaner edge-level decomposition; native to per-edge gating fallback).
- `M1` architecture: Allegro larger or higher `r_max`. Must remain strictly local.

---

## 14. Literature pointers

- Ramakrishnan et al. 2015 — Δ-learning.
- Praprotnik, Delle Site, Kremer 2005+ — AdResS, adaptive resolution simulation. Thermodynamic-consistency literature relevant to switching-function design.
- Shazeer et al. 2017; Fedus et al. 2021 — Mixture-of-Experts, Switch Transformer. The "discrete routing breaks gradients" cautionary tale.
- Graves 2016 — Adaptive Computation Time / pondering networks.
- Vandermause et al. 2020 (FLARE) — uncertainty-driven on-the-fly learning. Source of complexity-signal ideas.
- Behler 2011 — polynomial cutoff function used here for the gate (and inherited from NequIP / Allegro radial cutoffs).
- Batatia et al. 2022 (MACE) — community-relevant MLIP architecture, useful for comparison if `M1` is changed later.

---

## 15. Glossary (for the new project)

- **`M0`**: small, cheap MLIP. Runs everywhere. Target: `Allegro_S` or 1-layer NequIP.
- **`M1`**: larger correction MLIP. Strictly local. Runs only on active subgraph.
- **`s_i`**: per-atom complexity score, derived from `M0` hidden features.
- **`λ_i = c(s_i)`**: per-atom gate, smooth and compactly supported.
- **Active set `A`**: `{i : λ_i > 0}`.
- **Halo `H`**: atoms within `r_max^{M1}` of any active atom but with `λ = 0` themselves.
- **Compactly-supported smooth cutoff**: function that is identically zero below a threshold and `C^k`-smooth across the boundary (e.g. poly_5 = `1 − 6u⁵ + 15u⁴ − 10u³`).

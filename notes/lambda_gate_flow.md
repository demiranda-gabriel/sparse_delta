# How λ is produced from M0 and used to activate M1

**Author:** Gabriel de Miranda (with Claude Opus 4.7)
**Date:** 2026-05-29
**Scope:** the data flow that turns M0's hidden features into a per-atom
gate `λ_i ∈ [0, 1]` and uses it to weight (and, at inference, skip) the
M1 correction. Points to every node in the code. Companion to
[`training_and_loss.md`](training_and_loss.md) and
[`sparse_local_correction.md`](sparse_local_correction.md) §9–§10.

---

## 0. One-paragraph summary

M0 runs on the **whole** system and writes a per-edge scalar feature
stack. A small gate node pools those scalars to each receiver atom,
runs an MLP, and squashes the result to `λ_i ∈ [0, 1]`. The composite
energy is `E_total = Σ_i (E0_i + λ_i · E1_i)`, so λ multiplies M1's
per-atom contribution. **At training time M1 runs densely** and λ is a
*soft* scaling. The compute-saving "activation" — actually skipping M1
on atoms where λ=0 — is an **inference-time** property unlocked by the
`poly5` gate (exactly zero below a threshold); it is not yet a masked
forward in this codebase.

```
M0 (everywhere) ──► EDGE_FEATURES (per-edge scalars)
                         │
                         ├──► gate: pool→MLP→squash ──► λ_i  (per atom)
                         │
                         └──► M1 (per-edge) ──► E1_i (per atom)
                                                   │
                  E_total = Σ_i ( E0_i + λ_i · E1_i )
```

---

## 1. Where λ comes from: M0's per-edge scalars

The gate's input is **M0's DenseNet scalar concatenation**, stored under
`AtomicDataDict.EDGE_FEATURES_KEY`.

In the warm-start composite, M0's `Allegro_Module` is configured with
`scalar_out_field = EDGE_FEATURES_KEY`
(`software/sparse-delta-core/sparse_delta_core/model/warmstart_composite.py:237`):

```python
allegro = Allegro_Module(
    ...
    scalar_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
    scalar_out_field=AtomicDataDict.EDGE_FEATURES_KEY,   # ← gate reads this
    expose_pre_final_tp_out=True,                        # ← M1 warm-start reads this
)
```

So after M0's forward, `EDGE_FEATURES_KEY` holds, per edge `(i,j)`, the
concatenation of M0's per-layer ℓ=0 latents — the invariant scalar
descriptors of that edge's local environment. These are exactly the
features whose distribution we probed in exp 0 (the complexity-score
`s_i` study) to confirm they carry a usable signal for "is this
environment simple or complex?".

> **Two gate families in the codebase.** The *warm-start* composite
> (exp 4 / exp 5, the active path) uses `EdgeFeaturesToNodeLambda`,
> which pools M0's **per-edge** scalars internally. The older *Phase-A*
> composite (`build_phase_a_composite`) instead used `GateMLP` fed by
> `M0InvariantFeatures` (per-atom C0, pooled *outside* the gate). This
> note documents the warm-start path; the Phase-A path is analogous
> with the pool moved upstream.

---

## 2. The gate node: scalars → λ

Implementation:
`software/sparse-delta-core/sparse_delta_core/nn/gate_edge.py:EdgeFeaturesToNodeLambda`
(class at line 95).

Constructed in the composite builder
(`warmstart_composite.py:461`, inside the `gate_mode == "learned"`
branch):

```python
gate = EdgeFeaturesToNodeLambda(
    field_in=AtomicDataDict.EDGE_FEATURES_KEY,   # M0's per-edge scalars
    out_field=GATE_KEY,                          # "sparse_delta_lambda"
    edge_hidden_dim=64, node_hidden_dim=64, node_hidden_depth=2,
    gate_function="sigmoid",                     # or "poly5"
    initial_lambda_mean=0.05,
)
```

### 2.1 Forward (gate_edge.py:248)

```python
def forward(self, data):
    edge_feats = data[self.field_in]                  # (E, F_in) M0 scalars
    edge_index = data[EDGE_INDEX_KEY]                 # (2, E)
    edge_hidden = self.edge_mlp(edge_feats)           # (E, H)  per-edge MLP

    # sum-pool edges onto their RECEIVER atom (edge_index[0])
    node_hidden = edge_hidden.new_zeros((num_atoms, H))
    node_hidden.index_add_(0, edge_index[0], edge_hidden)   # (N, H)

    s = self.node_mlp(node_hidden)                    # (N, 1) per-atom score
    lam = self._apply_gate(s)                         # (N, 1) λ ∈ [0,1]
    data[self.out_field] = lam                        # write GATE_KEY
    return data
```

Three steps:

1. **Per-edge MLP** — projects M0's scalar stack to a hidden width.
2. **Sum-pool to receiver atom** via `index_add_` on `edge_index[0]`
   (the receiver). This is the "gather the environment of atom `i`"
   step. Written with `index_add_` (not `torch_scatter`) so it stays
   TorchScript / `torch.compile` friendly.
3. **Per-atom MLP → squash** — a small MLP produces a scalar score
   `s_i`, then `_apply_gate` maps it to `λ_i`.

### 2.2 The squashing functions (gate_edge.py:240)

```python
def _apply_gate(self, s):
    if self._gate_function == "sigmoid":
        return torch.sigmoid(s)                       # smooth, never exactly 0
    # poly5:
    u = ((s - s_low) / (s_high - s_low)).clamp(0, 1)
    return _poly_gate(u)                              # 6u⁵ − 15u⁴ + 10u³
```

- **`sigmoid`** (warm-up, what exp 4/5 use): `λ = σ(s − b)`. Smooth, but
  asymptotically approaches 0/1 without reaching them — **no exact
  sparsity**, so no compute savings yet. The final linear layer is
  zero-weight-initialised with bias set to `logit(initial_lambda_mean)`,
  so at step 0 every atom gets `λ ≈ initial_lambda_mean` regardless of
  input (gate_edge.py:212-226).
- **`poly5`** (the C²-smooth cutoff `6u⁵−15u⁴+10u³`, gate_edge.py:52):
  **exactly 0** for `s ≤ s_low`, **exactly 1** for `s ≥ s_high`, C² at
  both boundaries. This is what makes the *sparse two-pass* exact — λ=0
  means M1's contribution is provably zero, so the atom can be dropped
  from M1's compute. Swapping sigmoid→poly5 is the planned Phase-D move
  once `s_i` is calibrated; both are conservative (the gate path is
  differentiable so forces stay exact — see §4).

The output `λ` lives in `GATE_KEY = "sparse_delta_lambda"`
(`software/sparse-delta-core/sparse_delta_core/_keys.py:GATE_KEY`),
registered as a per-node field so batching treats it correctly.

### 2.3 Constant-gate variant (staged training stage 2)

In exp 5 stage 2 the gate is replaced by a zero-parameter
`ConstantNodeLambda`
(`software/sparse-delta-core/sparse_delta_core/nn/constant_lambda.py`)
that writes `λ ≡ lambda_value` (1.0) for every atom. Same `GATE_KEY`,
same downstream contract — M1 just gets full-strength everywhere while
it learns, before a learned gate is fit in stage 3. Selected by
`gate_mode="constant"` in the builder (`warmstart_composite.py`, gate
construction block).

---

## 3. How λ activates M1: the compose step

M1 is a strictly-local `Allegro_Module` that runs over edges and, via
`EdgewiseReduce`, produces a per-atom correction energy `E1_i` under
`PER_ATOM_ENERGY_M1_KEY`. M0's per-atom energy was moved aside earlier
to `PER_ATOM_ENERGY_M0_KEY` by a `RenameKey` (so M1's edgewise reduce
doesn't overwrite it).

The gate is applied in `GatedPerAtomEnergyCompose`
(`software/sparse-delta-core/sparse_delta_core/nn/compose.py:42`),
constructed at `warmstart_composite.py` (compose block):

```python
compose = GatedPerAtomEnergyCompose(
    e0_field=PER_ATOM_ENERGY_M0_KEY,     # E0_i
    e1_field=PER_ATOM_ENERGY_M1_KEY,     # E1_i
    lambda_field=GATE_KEY,               # λ_i
    out_field=PER_ATOM_ENERGY_KEY,       # E0_i + λ_i·E1_i
)
```

Its forward is literally (compose.py:101):

```python
data[self.out_field] = e0 + lam * e1
```

A downstream `AtomwiseReduce` sums `PER_ATOM_ENERGY_KEY` over atoms to
the total energy, and the top-level `ForceStressOutput` differentiates
that total w.r.t. positions for forces.

### 3.1 Node order in the graph

The composite is a single `SequentialGraphNetwork`
(`warmstart_composite.py`, assemble block). Execution order:

```
m0_edge_norm → m0_radial_chemical_embed → m0_scalar_embed_mlp
  → m0_tensor_embed → m0_allegro            ← writes EDGE_FEATURES (M0 scalars)
  → m0_edge_readout → m0_edge_eng_sum → m0_per_type_scale_shift
  → m0_rename_energy                        ← E0 moved to PER_ATOM_ENERGY_M0_KEY
  → gate                                    ← reads EDGE_FEATURES, writes λ (GATE_KEY)
  [→ m1_edge_sh → tensor_track_adapter]     ← only if m1_l_max > m0_l_max
  → m1_allegro → m1_edge_readout → m1_edge_eng_sum → m1_per_type_scale_shift
                                            ← writes E1 (PER_ATOM_ENERGY_M1_KEY)
  → compose                                 ← E = E0 + λ·E1
  → total_energy_sum                        ← Σ_i E_i
```

The gate runs **after** M0 (so M0's `EDGE_FEATURES` exist) and **before**
M1 (so λ is ready when compose needs it). M1 reads M0's `EDGE_FEATURES`
as *its own* scalar input (`scalar_in_field=EDGE_FEATURES_KEY`) but
writes to a *different* key (`scalar_out_field=M1_EDGE_FEATURES_KEY`), so
the gate's input is never clobbered.

---

## 4. λ keeps forces conservative

Because λ depends on positions (through M0's features) and is composed
multiplicatively before the single `autograd.grad(E, positions)` in
`ForceStressOutput`, the force on atom `a` correctly includes the
`Σ_i (∂λ_i/∂r_a) · E1_i` term. There is **no** `.detach()` on the
M0→gate path — detaching it would silently drop that term and break
conservativeness. This is asserted by the finite-difference +
position-dependent-gate tests in
`software/sparse-delta-core/tests/test_warmstart_composite.py`
(`TestRotation`, the conservativeness finite-diff cases).

---

## 5. Training-time (dense) vs inference-time (sparse) — important

| | Training (now) | Inference (design target, §10 of design doc) |
|---|---|---|
| M1 compute | runs on **all** atoms/edges (dense) | runs only on the **active subgraph** (atoms with λ>0 + their halo) |
| Role of λ | soft multiplicative weight on `E1_i` | exact on/off switch (poly5 gives true zeros) |
| Compute savings | none | `|active| / N` scaling — the whole point of the project |
| Gate function | `sigmoid` (warm-up) | `poly5` (exact sparsity) |

The current code implements the **dense, soft-gated** form. The sparse
two-pass driver (score → threshold → build active subgraph → run M1 only
there) is described in `sparse_local_correction.md` §10 and is **not yet
wired** as a masked forward. The `poly5` gate + the `GateMeanMetric`
sparsity penalty are the pieces in place to make the learned gate
*produce* a sparse λ; turning that sparsity into skipped compute is the
remaining inference-engineering step.

> **Caveat carried from exp 5:** the linear `mean(λ)` sparsity penalty
> collapses λ→0 uniformly past `sparsity_coeff≈0.1` (see
> `experiments/5-composite_staged/stage3_sweep_sparsity/README.md` and
> `training_and_loss.md` §5). So the gate doesn't yet yield a *useful*
> sparse pattern — that's open R&D (quadratic/entropy penalty, annealing).

---

## 6. Code reference index

| Concept | File:line |
|---|---|
| λ field key | `software/sparse-delta-core/sparse_delta_core/_keys.py:GATE_KEY` |
| M0 writes per-edge scalars (`EDGE_FEATURES_KEY`) | `warmstart_composite.py:237` (`scalar_out_field`) |
| M0 exposes pre-final TP for M1 warm-start | `warmstart_composite.py:238` (`expose_pre_final_tp_out=True`) |
| Gate construction (learned) | `warmstart_composite.py:461` |
| Gate node class | `software/sparse-delta-core/sparse_delta_core/nn/gate_edge.py:95` (`EdgeFeaturesToNodeLambda`) |
| Gate forward (pool + MLP + squash) | `gate_edge.py:248` |
| sigmoid vs poly5 squash | `gate_edge.py:240` (`_apply_gate`), `gate_edge.py:52` (`_poly_gate`) |
| init-λ bias solve | `gate_edge.py:212-226`, `gate_edge.py:75` (`_solve_poly5_bias`) |
| Constant gate (stage 2) | `software/sparse-delta-core/sparse_delta_core/nn/constant_lambda.py` |
| M0 energy moved aside | `warmstart_composite.py` (`m0_rename_energy`, `RenameKey`) |
| M1 per-atom energy | `warmstart_composite.py` (`m1_edge_eng_sum`, `EdgewiseReduce` → `PER_ATOM_ENERGY_M1_KEY`) |
| Compose `E = E0 + λ·E1` | `software/sparse-delta-core/sparse_delta_core/nn/compose.py:101` |
| Conservative-force wrap | `software/nequip-private/nequip/nn/grad_output.py:ForceStressOutput` |
| Sparsity penalty on λ | `software/sparse-delta-core/sparse_delta_core/train/sparsity.py:GateMeanMetric` |
| Conservativeness tests | `software/sparse-delta-core/tests/test_warmstart_composite.py` |
| Design rationale (score, gate, two-pass) | `notes/sparse_local_correction.md` §9–§10 |

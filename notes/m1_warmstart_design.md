# M1 Warm-Start from M0 — Design and Implementation Plan

**Status:** design, not yet implemented
**Parent doc:** [`sparse_local_correction.md`](sparse_local_correction.md) (read first for the overall sparse-delta motivation)
**Target experiment:** [`experiments/3-composite_phase_A/`](../experiments/3-composite_phase_A/)

---

## 0. Quick start for the next agent

You are picking this up on a new cluster. To get oriented:

1. **Read the parent doc** `notes/sparse_local_correction.md` for the broader sparse-delta motivation, gate math, and halo analysis.
2. **Skim this doc top-to-bottom** to internalize the warm-start design — sections 1–4 are the conceptual background that justifies the implementation plan in section 5.
3. **The implementation is greenfield**: nothing in `software/sparse-delta-core/src/sparse_delta_core/` implements the composite yet. The existing M0 fork (`software/allegro-private/`) already has the necessary hook (`expose_pre_final_tp_out`); you only need to wire consumers.
4. **First concrete task:** Section 5 step 1 — extend `Allegro_Module.irreps_out` to advertise `_allegro_pre_final_tp_out` when the flag is on. Everything else stacks on top of that.

Working dirs assumed:
- `software/allegro-private/` — read-only fork; only minimal surgical edits to `_allegro.py`.
- `software/nequip-private/` — read-only fork; no edits expected.
- `software/sparse-delta-core/` — where all new modules live.
- `experiments/3-composite_phase_A/` — first integration experiment.

---

## 1. TL;DR

Train two stacked Allegro models in series: a small `M0` runs everywhere; a larger `M1` runs only where a smooth per-atom gate `λ_i ∈ [0,1]` is non-zero. To save the parameters and compute that `M1` would otherwise spend re-learning a two-body tensor embedding, **`M1` warm-starts from the pre-final tensor features of `M0`** — i.e., the per-edge tensor `V_ij` that `M0` produced one TP before its final readout, which has already been trained against the energy/force loss.

Total energy is
```
E_total = E_0 + Σ_i λ_i · E_1^i
```
with a single autograd backward through both models, yielding conservative forces. Per-atom gating preserves the "skip whole atom" compute-saving argument from the parent doc.

---

## 2. Background: how Allegro builds features (reference)

This section is a condensed reference of what `M0` produces, so you can locate fields and irreps without re-reading the upstream codebase. Line refs are against `software/allegro-private/`.

### 2.1 Edge geometry (pre-Allegro)

`EdgeLengthNormalizer` (`allegro/model/allegro_models.py:134`) populates:
- `EDGE_INDEX_KEY` — `(2, E)`; row 0 = center `i`, row 1 = neighbor `j`.
- `EDGE_VECTORS_KEY` — `r_ij` (lazily, via `with_edge_vectors_`).
- `EDGE_LENGTHS_KEY`, `NORM_LENGTH_KEY` — length and per-edge-type-normalized length.

All Allegro features below live on **edges**. There is *no per-node tensor track*.

### 2.2 Scalar embedding

`TwoBodyBesselScalarEmbed` (`allegro/nn/scalarembed.py:19`) → `ProductTypeEmbedding` (`allegro/nn/_edgeembed.py:14`):
1. Bessel-radial encode `‖r_ij‖`, multiply by polynomial cutoff envelope.
2. Linearly project Bessel coefficients to `D = num_scalar_features`.
3. Multiply elementwise by `concat(center_type_embed(i), neighbor_type_embed(j))` of dim `D`.
4. Pass through `scalar_embed_mlp` (1 hidden layer, SiLU) → write to `EDGE_EMBEDDING_KEY` at width `num_scalar_features = S`.

This is the only place atom types enter the model.

### 2.3 Tensor embedding

`TwoBodySphericalHarmonicTensorEmbed` (`allegro/nn/tensorembed.py:17`):
1. `Y_ij = SH(r̂_ij)` with irreps `o3.Irreps.spherical_harmonics(l_max, p=-1)` → write to `EDGE_ATTRS_KEY`. **Frozen for all layers.**
2. Linear projection of `EDGE_EMBEDDING_KEY` → per-edge scalar weights `(E, num_irreps · U)` where `U = num_tensor_features`.
3. `V_ij^{(0)} = MakeWeightedChannels(Y_ij, weights)` of shape `(E, U, dim(SH))` → write to `EDGE_FEATURES_KEY`.

Each `(channel u, irrep r)` of `Y_ij` is rescaled by one learned scalar broadcast across the `(2ℓ+1)` `m`-components, preserving equivariance.

### 2.4 Allegro layers (`allegro/nn/_allegro.py:246`)

At entry, layer 0 reads:
- `tensor_basis = Y_ij` from `EDGE_ATTRS_KEY` (covariant, frozen).
- `tensor_features = V_ij^{(0)}` from `EDGE_FEATURES_KEY` (covariant, `(E, U, dim(SH))`).
- `twobody_scalar_embed` from `EDGE_EMBEDDING_KEY` (invariant, `(E, S)`).

Then `first_layer_env_embed_projection` (`_allegro.py:99–103`) linearly maps the scalar embed to `S + env_w_numel`, splitting into:
- `x_ij^{(0)}` — initial invariant latent of width `S` (appended to `accumulated_scalar_features`).
- `w_ij^{(0)}` — invariant per-edge env weights of width `num_irreps · U`.

Each layer then runs (`_allegro.py:269–334`):

1. **Stage A — build per-node env tensor:**
   `env_w_edges = w_ij ⊙ Y_ij` (env weighter), scatter-sum to centers → `V_i^env = Σ_{j ∈ N(i)} w_ij ⊙ Y_ij`, normalize by `1/√⟨n_nbr⟩`.
2. **Stage B — tensor product:** `V_ij^{(l+1)} = TP(V_ij^{(l)}, V_i^env)` gathered to edges via `edge_center`. Channels never mix (depthwise); paths mix per-channel if `tp_path_channel_coupling=True`.
3. **Stage C — extract invariants:** take the `0e` slot of `V_ij^{(l+1)}` → `scalars` of shape `(E, U)`.
4. **Stage D — latent MLP:** `latent(concat(accumulated_scalar_features + [scalars]))` → first `S` cols are the new `x_ij^{(l+1)}`; remaining `env_w_numel` cols are `w_ij^{(l+1)}` (skipped on last layer).

### 2.5 Key asymmetry: `V_ij` vs `w_ij`

- `V_ij` is **equivariant** (carries irreps), evolves through TPs, is the LHS of each TP, and can grow into exotic parities in interior layers.
- `w_ij` is **invariant** (pure scalars), is rebuilt fresh each layer from the current scalar latent, and only acquires irreps when multiplied with the frozen `Y_ij`.

This asymmetry is what makes Allegro strictly local: the RHS of every TP is "current scalar weights × frozen SH summed over one-hop neighbors of `i`", never a multi-hop message.

### 2.6 Readout

After `num_layers` iterations (`_allegro.py:337`):
```
data[EDGE_FEATURES_KEY] = torch.cat(accumulated_scalar_features, dim=-1)
                       # shape (E, S · (L + 1))
```

`edge_readout` (a `ScalarMLP`, `allegro_models.py:213`) reads the entire concatenated DenseNet stack and projects to `(E, 1)`. **Every layer's `x_ij^{(l)}` plus the initial two-body `x_ij^{(0)}` feed the energy** — not just the final layer.

### 2.7 Last-layer pruning

`tps_irreps[-1] = {0e}` is hardcoded (`_allegro.py:130`). Backward pruning (`_allegro.py:152–167`) propagates this: at layer `L-1`, only irreps reachable from `0e` via a TP with `env_embed_irreps` survive — i.e., the SH-natural set `{0e, 1o, 2e}` for `l_max=2`. Interior layers can be wider (exotic parities).

### 2.8 The pre-final TP hook (already present in our fork)

`software/allegro-private/allegro/nn/_allegro.py:46–49, 304–305`:

```python
expose_pre_final_tp_out: bool = False,
...
if self._expose_pre_final_tp_out and layer_index == self.num_layers - 2:
    data["_allegro_pre_final_tp_out"] = tensor_features
```

When the flag is on, the `V_ij` *after* layer `L-2`'s TP (= input to the final TP) is published. This is the "penultimate" `V_ij`. Every irrep in this tensor has gradient signal from energy/force loss because the final TP reads them all to produce the `0e` scalars. **This is the warm-start source for `M1`.**

Flag is typed `bool`, conditional branches are constant-foldable; the hook is TorchScript/`torch.compile` safe by design (see the comment at `_allegro.py:47–48`).

---

## 3. Why this design

Three design choices were made after walking through Allegro's internals:

### 3.1 Per-atom gating, but logits from edge features

The natural Allegro output is per-edge scalars. The cleanest implementation is:
```
s_i = MLP_node(  Σ_{j ∈ N(i)} φ(x_ij)  )
λ_i = poly_cutoff(s_i)        # smooth, compactly supported, exactly 0 below threshold
```
- The pool stabilizes the signal (lower variance than a per-edge logit).
- Atomic-energy decomposition remains physical (`E_1^i` per atom, not per edge).
- Compute savings argument is intact: when `λ_i = 0`, the entire atom `i`'s `M1` neighborhood evaluation is skipped, including its `V_i^env` construction.

Per-edge gating was considered and rejected: it does not save more compute (computing M1 edge features still requires building `V_i^env` over all of `i`'s edges), and it muddies the physical interpretation.

### 3.2 Warm-start from M0's pre-final V_ij (not post-final)

The post-final `V_ij` (output of M0's last TP) has only its `0e` slot trained — non-scalar irrep paths there have zero gradient signal from energy loss. The pre-final `V_ij` has every irrep trained, because M0's last TP consumes them all to produce the `0e` scalars that drive the readout.

Concretely for M0 with `num_layers=2, l_max=2`: pre-final `V_ij` lives in `{0e, 1o, 2e}` (after backward pruning at layer 0), shape `(E, U_{M0}, 9)`. This is what `_allegro_pre_final_tp_out` exposes.

If joint M0+M1 fine-tuning is performed, gradients flow through M1 → through the dict-stored tensor → back into M0's pre-final-layer parameters. So even with warm-start, M0's penultimate features will continue adapting to be useful for M1.

### 3.3 M1 stays strictly local

M1 inherits the locality of M0's pre-final `V_ij`, which has receptive field = first neighbor shell of `i`. Every additional M1 Allegro layer adds zero halo (this is the defining property of Allegro). So `M1` with any `num_layers_M1 ≥ 1` is strictly local — the savings argument from the parent doc holds.

### 3.4 Conservativeness

A single `ForceStressOutput` wraps the whole composite. Energy = `E_0 + λ · E_1` is a differentiable function of positions. Autograd produces conservative forces in one backward. No `.detach()` anywhere on the path from positions to total energy.

---

## 4. Architecture

### 4.1 Data flow

```
data (positions, types, neighborlist)
  │
  ├─► M0_Allegro  (expose_pre_final_tp_out=True)
  │     writes:
  │       EDGE_ATTRS_KEY                = Y_ij              (frozen SH; shared with M1)
  │       EDGE_EMBEDDING_KEY            = scalar_embed (M0)
  │       EDGE_FEATURES_KEY             = concat scalar stack (M0); width S_{M0} · (L_{M0}+1)
  │       _allegro_pre_final_tp_out     = V_ij^{(M0, pre-final)}; irreps tps_irreps[L_{M0}-1]
  │       EDGE_ENERGY_KEY (from M0's edge_readout)
  │       PER_ATOM_ENERGY_KEY (from M0's edge_eng_sum + scale/shift)
  │       → renamed to PER_ATOM_ENERGY_M0
  │
  ├─► GateModule
  │     reads EDGE_FEATURES_KEY (M0's edge scalars), EDGE_INDEX_KEY
  │     pools edges to nodes, MLPs to s_i, applies polynomial cutoff → λ_i
  │     writes LAMBDA_KEY (= "node_lambda")
  │
  ├─► M1_Allegro  (warm-start)
  │     reads:
  │       EDGE_ATTRS_KEY        = Y_ij              (reused, no recomputation)
  │       EDGE_FEATURES_KEY     = M0's scalar stack (used as scalar_in for M1)
  │       _allegro_pre_final_tp_out = V_ij^{(M0, pre-final)} (used as tensor_features_in for M1)
  │     writes:
  │       _m1_edge_features     = M1's own DenseNet scalar stack
  │       _m1_edge_energy
  │       _m1_per_atom_energy   = E_1^i
  │
  ├─► CompositeEnergy
  │     PER_ATOM_ENERGY_KEY = PER_ATOM_ENERGY_M0 + λ_i · _m1_per_atom_energy
  │
  ├─► AtomwiseReduce → TOTAL_ENERGY_KEY
  │
  └─► ForceStressOutput → forces (single backward)
```

### 4.2 Module inventory

All new modules live in `software/sparse-delta-core/src/sparse_delta_core/`:

| Module | Role | Approx LOC |
|---|---|---|
| `EdgeFeaturesToNodeLambda` | Pool edge scalars → MLP → polynomial-cutoff gate | ~80 |
| `WarmStartAllegro` | Thin subclass / configured `Allegro_Module` that consumes `_allegro_pre_final_tp_out` | ~30 (mostly config) |
| `GatedPerAtomEnergyCompose` | `PER_ATOM_ENERGY = E0 + λ · E1` | ~30 |
| `RenameKey` | Utility module: rename `K_in → K_out` in data dict (for moving M0's `PER_ATOM_ENERGY_KEY` aside before M1 overwrites it) | ~20 |
| `CompositeAllegroModel` (`@model_builder`) | Assembles M0, gate, M1, compose, reduce, ForceStressOutput | ~150 |

One minimal upstream edit:
| File | Edit |
|---|---|
| `allegro-private/allegro/nn/_allegro.py` | Add `_allegro_pre_final_tp_out` to `self.irreps_out` when `expose_pre_final_tp_out=True` |

---

## 5. Implementation plan

Steps are ordered so each one is independently testable.

### Step 1 — Advertise pre-final irreps in M0's `irreps_out`

**Why:** Downstream `_init_irreps` calls (M1, gate) need to know the irrep structure of every data dict key they read. `_allegro_pre_final_tp_out` is currently written at forward time but not registered.

**File:** `software/allegro-private/allegro/nn/_allegro.py`

**Edit (near `_allegro.py:226–232`, the `irreps_out` block at end of `__init__`):**

```python
self.irreps_out.update(
    {
        self.scalar_out_field: Irreps(
            [(self.num_scalar_features * (self.num_layers + 1), (0, 1))]
        ),
    }
)
if self._expose_pre_final_tp_out:
    # Pre-final TP output irreps = output of layer (L-2) = tps_irreps_out[L-2]
    # which we computed as `tps_irreps_out` above; persist it.
    pre_final_irreps = tps_irreps_out[self.num_layers - 2]
    # With strided layout, the actual stored shape is (E, U, dim(pre_final_irreps)),
    # i.e. multiplicity num_tensor_features broadcast across irrep slots.
    # In `irreps_out`, advertise the *flattened equivalent* using mul=num_tensor_features
    # per irrep, so consumers get the right path-pruning when they call `_init_irreps`.
    self.irreps_out["_allegro_pre_final_tp_out"] = Irreps(
        [(self.num_tensor_features, ir) for _, ir in pre_final_irreps]
    )
```

**Caveat:** `tps_irreps_out` is `del`'d at `_allegro.py:174`. Hoist it to a `self._tps_irreps_out = tps_irreps_out` before the `del`, then use it for the irreps registration. Or alternatively compute `pre_final_irreps` once before the irrep loop and store it.

**Test:** Construct `Allegro_Module(..., expose_pre_final_tp_out=True)` and assert `module.irreps_out["_allegro_pre_final_tp_out"]` matches the expected pre-final irreps for several `(num_layers, l_max, parity)` combinations.

---

### Step 2 — Build `EdgeFeaturesToNodeLambda`

**Why:** Produces `λ_i` from M0's edge scalars without an extra equivariant pass.

**File:** `software/sparse-delta-core/src/sparse_delta_core/gate.py` (new).

**Sketch:**

```python
from typing import Sequence
import torch
from e3nn.o3._irreps import Irreps
from e3nn.util.jit import compile_mode
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, ScalarMLPFunction, scatter
from nequip.nn.embedding import PolynomialCutoff


@compile_mode("script")
class EdgeFeaturesToNodeLambda(GraphModuleMixin, torch.nn.Module):
    """Pool per-edge scalars → per-node MLP → polynomial-cutoff gate λ_i ∈ [0, 1]."""

    def __init__(
        self,
        # MLP shape
        hidden_layers_depth: int = 1,
        hidden_layers_width: int = 64,
        nonlinearity: str = "silu",
        # gate
        cutoff_threshold: float = 1.0,    # s_i ≥ threshold → λ_i = 1
        cutoff_p: int = 5,                # polynomial degree
        # bookkeeping
        edge_features_in_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        node_lambda_out_field: str = "node_lambda",
        irreps_in=None,
    ):
        super().__init__()
        self.edge_features_in_field = edge_features_in_field
        self.node_lambda_out_field = node_lambda_out_field

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.edge_features_in_field],
            irreps_out={
                self.node_lambda_out_field: Irreps("1x0e"),
            },
        )
        in_dim = self.irreps_in[self.edge_features_in_field].num_irreps

        # per-edge → invariant scalar logit
        self.edge_to_logit = ScalarMLPFunction(
            input_dim=in_dim,
            output_dim=hidden_layers_width,
            hidden_layers_depth=hidden_layers_depth,
            hidden_layers_width=hidden_layers_width,
            nonlinearity=nonlinearity,
        )
        # per-node MLP from pooled hidden vector → s_i
        self.node_to_s = ScalarMLPFunction(
            input_dim=hidden_layers_width,
            output_dim=1,
            hidden_layers_depth=hidden_layers_depth,
            hidden_layers_width=hidden_layers_width,
            nonlinearity=nonlinearity,
        )
        self._cutoff_threshold = float(cutoff_threshold)
        self._cutoff_p = int(cutoff_p)

    def _poly_cutoff(self, s: torch.Tensor) -> torch.Tensor:
        # Map s ∈ ℝ → λ ∈ [0,1], with λ = 0 for s ≤ 0, λ = 1 for s ≥ threshold,
        # C^2-smooth polynomial in between. Standard 1 - 6u^5 + 15u^4 - 10u^3 form
        # used elsewhere in nequip (see `nequip.nn.embedding.PolynomialCutoff`).
        u = (s / self._cutoff_threshold).clamp(min=0.0, max=1.0)
        # smoothstep^2 family (the "1-6u^5+15u^4-10u^3" cutoff is exactly C^2)
        return 1.0 - (1.0 - u).pow(3) * (1.0 + 3.0 * u + 6.0 * u * u)
        # ^ verify the polynomial against the one used for the parent design doc; tweak as needed.

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        num_atoms: int = AtomicDataDict.num_nodes(data)

        edge_feats = data[self.edge_features_in_field]              # (E, D)
        edge_hidden = self.edge_to_logit(edge_feats)                # (E, H)
        node_hidden = scatter(edge_hidden, edge_center, dim=0,
                              dim_size=num_atoms)                   # (N, H)
        # optional: divide by sqrt(avg_num_neighbors); keep simple for v1.
        s = self.node_to_s(node_hidden).squeeze(-1)                 # (N,)
        data[self.node_lambda_out_field] = self._poly_cutoff(s)     # (N,)
        return data
```

**Decisions to revisit:**
- Whether to apply `AvgNumNeighborsNorm` between the scatter and the per-node MLP.
- Whether to bias `s` so the network starts with `λ ≈ 0` everywhere at init (so `M0` predictions aren't immediately polluted by random `M1` corrections).

**Test:**
- Forward an `Allegro_Module` output through it, check `data["node_lambda"]` shape `(N,)`, all values in `[0, 1]`.
- Initialize the head so `s_i` is centered at 0 → `λ_i` should be approximately 0 at init.
- Permutation equivariance: shuffle atoms, check the output permutes accordingly.

---

### Step 3 — Configure `M1` via `Allegro_Module` field aliases

**Insight:** No subclassing needed. `Allegro_Module.__init__` already exposes `tensor_features_in_field`, `scalar_in_field`, `scalar_out_field`, `tensor_basis_in_field` as constructor kwargs. We just point them at M0's outputs and rename M1's outputs to avoid collisions.

For M1, set:
```python
M1 = Allegro_Module(
    num_layers=num_layers_M1,
    num_scalar_features=S_M1,
    num_tensor_features=U_M1,
    tensor_track_allowed_irreps=tensor_track_allowed_irreps_M1,
    avg_num_neighbors=avg_num_neighbors,
    type_names=type_names,
    latent_kwargs={...},
    tensor_basis_in_field=AtomicDataDict.EDGE_ATTRS_KEY,         # reuse M0's SH
    tensor_features_in_field="_allegro_pre_final_tp_out",        # warm-start from M0
    scalar_in_field=AtomicDataDict.EDGE_FEATURES_KEY,            # M0's scalar stack
    scalar_out_field="_m1_edge_features",                        # M1's own
    expose_pre_final_tp_out=False,                               # M1 has no consumer for this
    irreps_in=...,  # must include irreps for all three input fields, taken from M0
)
```

**Gotcha:** M1's `first_layer_env_embed_projection` has `input_dim = num_scalar_features` from the irrep counts at `scalar_in_field`. M0's `EDGE_FEATURES_KEY` has width `S_{M0} · (L_{M0} + 1)` — typically much larger than `S_{M1}`. That's fine; the projection's input dim is determined from `self.irreps_in[scalar_in_field].num_irreps`, so it'll size correctly. Just be aware that this is an *implicit* learned projection from M0's full scalar stack down to `S_{M1} + env_w_numel_{M1}`.

**Decision to make:** Should M1's `num_tensor_features` (`U_{M1}`) match M0's? If yes, the warm-start tensor `(E, U_{M0}, dim(SH))` slots in directly. If `U_{M1} > U_{M0}`, you need an explicit channel-projection adapter — easiest is an extra `MakeWeightedChannels`-like module placed before M1 that maps `U_{M0} → U_{M1}` per irrep.

**Recommendation for v1:** `U_{M1} = U_{M0}`, no channel adapter. Iterate on channel widening only after the basic composite works.

**Test:**
- Construct `M1` standalone (without `M0`), feed it a synthetic `data` dict with the warm-start fields populated by hand, check forward runs and output shapes are correct.
- Equivariance test: rotate input positions, check that M0+M1 forward produces correctly-rotated forces (this is the end-to-end test and runs after step 5).

---

### Step 4 — `GatedPerAtomEnergyCompose`

**File:** `software/sparse-delta-core/src/sparse_delta_core/compose.py` (new).

**Sketch:**

```python
@compile_mode("script")
class GatedPerAtomEnergyCompose(GraphModuleMixin, torch.nn.Module):
    """PER_ATOM_ENERGY_KEY = E0_per_atom + λ_i · E1_per_atom."""

    def __init__(
        self,
        e0_field: str = "_per_atom_energy_M0",
        e1_field: str = "_per_atom_energy_M1",
        lambda_field: str = "node_lambda",
        out_field: str = AtomicDataDict.PER_ATOM_ENERGY_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self.e0_field = e0_field
        self.e1_field = e1_field
        self.lambda_field = lambda_field
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[e0_field, e1_field, lambda_field],
            irreps_out={out_field: Irreps("1x0e")},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        e0 = data[self.e0_field]                  # (N, 1) or (N,)
        e1 = data[self.e1_field]                  # (N, 1) or (N,)
        lam = data[self.lambda_field]             # (N,)
        # broadcast-safe
        if e0.dim() == 2:
            lam = lam.unsqueeze(-1)
        data[self.out_field] = e0 + lam * e1
        return data
```

**Test:** Synthetic input, hand-compute expected output, assert equality.

---

### Step 5 — `CompositeAllegroModel` builder

**File:** `software/sparse-delta-core/src/sparse_delta_core/composite.py` (new).

**Skeleton:**

```python
from nequip.data import AtomicDataDict
from nequip.model import model_builder
from nequip.nn import SequentialGraphNetwork, ScalarMLP, EdgewiseReduce, AtomwiseReduce, PerTypeScaleShift, ForceStressOutput
from nequip.nn.embedding import EdgeLengthNormalizer
from allegro.nn import TwoBodySphericalHarmonicTensorEmbed, Allegro_Module
from .gate import EdgeFeaturesToNodeLambda
from .compose import GatedPerAtomEnergyCompose
from .rename import RenameKey  # tiny helper, see step 5.5


@model_builder
def CompositeAllegroModel(
    # shared
    r_max: float,
    type_names,
    l_max: int,
    parity: bool = True,
    avg_num_neighbors=None,
    # M0
    m0_radial_chemical_embed,
    m0_num_layers: int = 2,
    m0_num_scalar_features: int = 64,
    m0_num_tensor_features: int = 16,
    m0_allegro_mlp_hidden_layers_depth: int = 1,
    m0_allegro_mlp_hidden_layers_width: int = 64,
    m0_readout_mlp_hidden_layers_depth: int = 1,
    m0_readout_mlp_hidden_layers_width: int = 32,
    # M1
    m1_num_layers: int = 1,
    m1_num_scalar_features: int = 128,
    m1_num_tensor_features: int = 16,   # match M0 for v1
    m1_allegro_mlp_hidden_layers_depth: int = 1,
    m1_allegro_mlp_hidden_layers_width: int = 128,
    m1_readout_mlp_hidden_layers_depth: int = 1,
    m1_readout_mlp_hidden_layers_width: int = 64,
    # gate
    gate_hidden_layers_depth: int = 1,
    gate_hidden_layers_width: int = 64,
    gate_cutoff_threshold: float = 1.0,
    # per-type scale/shift
    per_type_energy_scales=None,
    per_type_energy_shifts=None,
    # derivatives
    do_derivatives: bool = True,
):
    # === M0 ===
    # Reuse the FullAllegroModel internals but stop *before* ForceStressOutput,
    # and configure with expose_pre_final_tp_out=True. Rename M0's per-atom energy
    # field out of the way so M1's readout doesn't overwrite it.
    m0_modules = _build_m0_modules(...)            # returns dict of named modules
    m0_modules["m0_rename_energy"] = RenameKey(
        in_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field="_per_atom_energy_M0",
        irreps_in=m0_modules[<last>].irreps_out,
    )

    # === Gate ===
    gate = EdgeFeaturesToNodeLambda(
        hidden_layers_depth=gate_hidden_layers_depth,
        hidden_layers_width=gate_hidden_layers_width,
        cutoff_threshold=gate_cutoff_threshold,
        edge_features_in_field=AtomicDataDict.EDGE_FEATURES_KEY,
        node_lambda_out_field="node_lambda",
        irreps_in=m0_modules["m0_rename_energy"].irreps_out,
    )

    # === M1 ===
    m1 = Allegro_Module(
        num_layers=m1_num_layers,
        num_scalar_features=m1_num_scalar_features,
        num_tensor_features=m1_num_tensor_features,
        tensor_track_allowed_irreps=tensor_track_allowed_irreps,
        avg_num_neighbors=avg_num_neighbors,
        type_names=type_names,
        latent_kwargs={
            "hidden_layers_depth": m1_allegro_mlp_hidden_layers_depth,
            "hidden_layers_width": m1_allegro_mlp_hidden_layers_width,
            "nonlinearity": "silu",
            "bias": False,
        },
        tensor_basis_in_field=AtomicDataDict.EDGE_ATTRS_KEY,
        tensor_features_in_field="_allegro_pre_final_tp_out",
        scalar_in_field=AtomicDataDict.EDGE_FEATURES_KEY,
        scalar_out_field="_m1_edge_features",
        irreps_in=gate.irreps_out,
    )

    m1_readout = ScalarMLP(
        output_dim=1,
        hidden_layers_depth=m1_readout_mlp_hidden_layers_depth,
        hidden_layers_width=m1_readout_mlp_hidden_layers_width,
        field="_m1_edge_features",
        out_field="_m1_edge_energy",
        irreps_in=m1.irreps_out,
    )
    m1_eng_sum = EdgewiseReduce(
        field="_m1_edge_energy",
        out_field="_per_atom_energy_M1",
        avg_num_neighbors=avg_num_neighbors,
        type_names=type_names,
        irreps_in=m1_readout.irreps_out,
    )

    # === Compose ===
    compose = GatedPerAtomEnergyCompose(
        e0_field="_per_atom_energy_M0",
        e1_field="_per_atom_energy_M1",
        lambda_field="node_lambda",
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        irreps_in=m1_eng_sum.irreps_out,
    )

    total_energy_sum = AtomwiseReduce(
        irreps_in=compose.irreps_out,
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
    )

    energy_model = SequentialGraphNetwork({
        **m0_modules,
        "gate": gate,
        "m1": m1,
        "m1_readout": m1_readout,
        "m1_eng_sum": m1_eng_sum,
        "compose": compose,
        "total_energy_sum": total_energy_sum,
    })
    return ForceStressOutput(energy_model, do_derivatives)
```

**`_build_m0_modules`** is a refactor of `FullAllegroModel`'s body in `allegro_models.py:132` that returns the `modules` dict *without* wrapping in `ForceStressOutput` and with `expose_pre_final_tp_out=True`. The cleanest approach is to copy-paste the upstream builder into `composite.py` rather than monkey-patching.

### Step 5.5 — `RenameKey` helper

Tiny utility — see step 5 sketch. Lives in `software/sparse-delta-core/src/sparse_delta_core/utils/rename.py`. Just copies `data[in_field]` to `data[out_field]` and propagates irreps in `irreps_out`.

---

### Step 6 — Training recipe

Two variants. Start with (a); only fall back to (b) if (a) is unstable.

**(a) Joint training from scratch.**
- Single optimizer, all parameters trainable.
- Loss: standard `energy + λ_F · forces` (NequIP defaults).
- **Gate initialization bias:** initialize the gate so `λ_i ≈ 0` at init (e.g., negative bias on the final layer of `node_to_s`). Otherwise M1 starts adding random correction noise to M0 and training is unstable.
- Optional: small sparsity penalty `α · mean(λ_i)` to encourage the gate to stay closed unless `M1` provides real benefit.

**(b) Staged training.**
1. Train M0 to convergence on the energy/force loss as a standalone Allegro model.
2. Freeze M0; train gate + M1 jointly. The gate parameters are free; M1 is trained from random init.
3. Optional: unfreeze M0 for joint fine-tuning at the end.

Staged is safer but requires two checkpoints to manage. Joint requires care with gate init but is operationally simpler.

**Conservativeness sanity check during training:** periodically log `‖∇_x E_total − F_pred‖` — should be machine precision. If not, something is detaching gradients (almost certainly an `.item()` or `.detach()` somewhere in the composite forward).

---

### Step 7 — Tests

Mirror the upstream nequip/allegro test layout. New tests in `software/sparse-delta-core/tests/`:

- **`test_pre_final_irreps.py`**: Step 1 above. Construct `Allegro_Module(expose_pre_final_tp_out=True)` for several `(num_layers, l_max, parity)` triples; verify `irreps_out["_allegro_pre_final_tp_out"]`.
- **`test_gate.py`**: Forward `EdgeFeaturesToNodeLambda` on synthetic data; verify output range `[0, 1]`, permutation equivariance, gradient flow.
- **`test_compose.py`**: Hand-computed addition.
- **`test_composite_equivariance.py`**: Rotate a small test system, verify total energy is invariant and forces are equivariant under the rotation. **This is the load-bearing test.**
- **`test_composite_conservativeness.py`**: Compare `-∂E/∂x` (via finite differences or autograd on `E` alone) to the model's reported forces. Tolerance should be `< 1e-5` in float32.
- **`test_composite_lambda_zero_reduction.py`**: Set `gate_cutoff_threshold` very high so `λ_i = 0` everywhere; total energy must equal pure M0 energy.
- **`test_composite_scripts.py`**: TorchScript the composite, verify forward matches eager.
- **`test_composite_compiles.py`** (optional, gated on hardware): `torch.compile(composite)`, verify forward + backward match eager. Likely deferred given the OOM history with PT2 capture on AMD GCDs — see recent commits in experiment 2.

---

### Step 8 — First experiment (`experiments/3-composite_phase_A/`)

The experiment directory already exists with a stub `config.yaml`, `README.md`, `train.sh`. Wire it to the new `CompositeAllegroModel`:

- Pick a small dataset where the sparsity argument is meaningful (CO/Pt or a curated subset with mixed bulk/defect environments).
- M0 config: small Allegro (`num_layers=2`, `num_scalar_features=64`, `num_tensor_features=16`, `l_max=2`).
- M1 config: larger Allegro (`num_layers=1` or `2`, `num_scalar_features=128`, same `num_tensor_features=16` for v1).
- Gate: 1-layer MLP, width 64.
- Sparsity penalty: `α = 0.01 · mean(λ_i)` initially; tune.

Update the experiment README header with: hypothesis, success criteria, outer + submodule SHAs. Follow the workflow in `notes/workflow.md`. Commit before `sbatch`.

---

## 6. Open questions and decisions to revisit

These are deliberately deferred — punt them past v1 unless they bite during implementation.

1. **`U_{M1}` vs `U_{M0}`.** v1 uses the same `num_tensor_features` to skip the channel adapter. If M1 needs more capacity in the tensor track, add a per-irrep linear channel-widener at M1's input. The widener weights would also receive gradient through M1's TPs.

2. **`l_max_{M1}` vs `l_max_{M0}`.** v1 uses the same `l_max`. M1 cannot expose higher-ℓ content than M0 provides at its pre-final layer (M0's SH evaluation bounds the bandwidth forever). If M1 wants higher ℓ, M0 must be reconfigured.

3. **Alternative warm-start sources.** Currently using `_allegro_pre_final_tp_out` (layer-`L-2` output). Could also try:
   - **M0's penultimate scalar slice only** (no tensor warm-start; M1 re-builds its own `V_ij` from SH but inherits M0's scalar latent). Cheaper to implement; weaker warm-start.
   - **M0's per-edge env-weighted SH from some intermediate layer.** Stronger warm-start but requires extra hooks.
   - **Joint exposure of multiple M0 layers, with M1's scalar track ingesting all of them via DenseNet concat.** Maximally rich; more parameters.

4. **Joint M0 training vs frozen.** The "warm start" framing assumes M0 is at least partially trainable. If a fully-pretrained M0 checkpoint is loaded and frozen, only the random non-scalar paths at M0's last TP would be untrained — which is *not* the issue with pre-final warm-start (pre-final has all paths trained). So frozen M0 is fine with pre-final warm-start.

5. **Per-node `avg_num_neighbors` normalization in the gate.** Probably want it, mirroring Allegro's other reductions; v1 skips it for simplicity.

6. **Gate sparsity loss.** Linear `α · mean(λ)`? Or an L0-ish surrogate? Or just rely on the polynomial cutoff being exactly zero below threshold? Decide empirically.

7. **Force-field deployment.** When packaged via `nequip-compile --mode aotinductor`, the dict-side-channel `_allegro_pre_final_tp_out` becomes a real tensor in the compiled graph. Verify the export still works once the composite is functional.

---

## 7. Key code paths to know

For navigation on the new cluster:

| What | Where |
|---|---|
| Allegro module (the L-iteration loop) | `software/allegro-private/allegro/nn/_allegro.py:246–338` |
| Pre-final TP hook (already present) | `software/allegro-private/allegro/nn/_allegro.py:46–49, 74, 304–305` |
| Allegro `irreps_out` registration | `software/allegro-private/allegro/nn/_allegro.py:226–232` |
| Tensor embedding (`V_ij^{(0)}`) | `software/allegro-private/allegro/nn/tensorembed.py:17` |
| Scalar embedding | `software/allegro-private/allegro/nn/scalarembed.py:19` |
| Contracter (the TP) | `software/allegro-private/allegro/nn/_strided/_contract.py:11` |
| `MakeWeightedChannels` | `software/allegro-private/allegro/nn/_strided/_channels.py:8` |
| `FullAllegroModel` builder | `software/allegro-private/allegro/model/allegro_models.py:91–283` |
| AtomicDataDict keys | `software/nequip-private/nequip/data/_keys.py` (via the `AtomicDataDict` re-export) |
| `ScalarMLPFunction`, `ScalarMLP` | `software/nequip-private/nequip/nn/mlp.py` (approx; grep if moved) |
| `EdgewiseReduce`, `AtomwiseReduce` | `nequip.nn` re-exports |
| `ForceStressOutput` | `nequip.nn._grad_output` (single autograd backward wrapper) |
| `SequentialGraphNetwork`, `GraphModuleMixin` | `nequip.nn._graph_mixin` |
| Parent design doc | `notes/sparse_local_correction.md` |
| Workflow conventions | `notes/workflow.md` |
| Experiment template | `experiments/_template/` |

---

## 8. What this doc deliberately does not cover

- Dataset selection and curation — handled in the experiment README.
- Hyperparameter search — out of scope until v1 works end-to-end.
- Multi-GPU / DDP — inherit whatever the parent project's nequip config uses.
- LAMMPS / ASE deployment — defer until the composite trains.
- Comparisons to baselines (delta learning, AdResS, MoE) — covered in `sparse_local_correction.md` section 1.

# NOTES — 0-m0_complexity_probe

Theory and methodology for the §12 distribution diagnostic. The script [`probe.py`](probe.py) implements what is described here.

This document is intentionally long. The science of sparse_delta hinges on whether the M0 hidden representation can separate "simple" from "complex" atomic environments, and the right mathematical framing matters. Skip to §6 ("Methods tested") if you only want the recipe.

---

## 1. Setup recap

- **M0**: packaged Allegro `baseline-B` from sibling `multifidelity` project. Trained on cameron CO/Pt dataset (optb88-vdW DFT). Architecture: `r_max=7.0`, `l_max=1`, `parity=true`, `num_layers=2`, `num_scalar_features=32`, `num_tensor_features=32`. **Strictly local** because Allegro's receptive field is `r_max` regardless of `num_layers` (each Allegro layer mixes only edges sharing a center, not multi-hop messages between atoms — see [`notes/sparse_local_correction.md`](../../notes/sparse_local_correction.md) §2.3).
- **Dataset**: cameron CO/Pt train/val splits at `split_dataset_r5.0_*`. 131 val frames, 43–339 atoms each, mixture of bulk Pt, Pt-slab, Pt-nanoparticle, gas-phase CO, and CO/Pt interface configurations.
- **Class labels**: composition heuristic per frame: `Pt-only` (no C, no O), `CO-only` (no Pt), `mixed` (has Pt and at least one of C, O). Robust to filename loss in the split, fast, and sufficient to ask the bimodality question.

---

## 2. What hidden features Allegro exposes

Allegro is a **per-edge** model. Per-atom quantities (energies, forces) are computed by aggregating per-edge quantities at the receiver atom. The relevant fields in the `AtomicDataDict` after a forward pass are:

| Key | String | Per-edge dim (baseline-B) | Meaning |
|---|---|---|---|
| `EDGE_INDEX_KEY` | `edge_index` | shape `(2, E)` | row 0 = receiver atom index, row 1 = sender. **Convention from [`allegro/nn/edgewise.py`](../../software/allegro-private/allegro/nn/edgewise.py): receiver is row 0.** |
| `EDGE_EMBEDDING_KEY` | `edge_embedding` | 32 | Initial 2-body scalar embed: `radial_chemical_embed(r_ij, t_i, t_j)` → `scalar_embed_mlp` → 32 scalars. Updated each Allegro layer with the per-layer scalar latents (DenseNet style; see §3). |
| `EDGE_FEATURES_KEY` | `edge_features` | **96 = 32×3** | The DenseNet concat of all scalar slices: `[twobody, layer1, layer2]`. **This is what we call F2 in §6.** Final readout reads from this. |
| `EDGE_ATTRS_KEY` | `edge_attrs` | irreps tensor | Spherical harmonics of `r̂_ij`. Fixed (not learned). Skip. |
| `EDGE_ENERGY_KEY` | `edge_energy` | 1 | Scalar per edge; `readout_mlp(edge_features)`. |
| `PER_ATOM_ENERGY_KEY` | `atomic_energy` | 1 | `EdgewiseReduce` sum-pool of `edge_energy` to receiver atoms, normalized by `√(2·avg_num_neighbors_recv)`. |

**The DenseNet concat that produces the dim-96 vector is the central object of this experiment.** It is unpacked in §3.

---

## 3. F2 unpacked: what `edge_features[ij]` is, mathematically

After Allegro's full forward pass, every directed edge `(j → i)` carries a **96-dim scalar vector** `h_ij` stored in `data["edge_features"]`. It is built by concatenating three groups of 32 scalars, in this order:

```
h_ij  =  [  s^{(0)}_ij  ,  s^{(1)}_ij  ,  s^{(2)}_ij  ]   ∈  ℝ^{32+32+32} = ℝ^{96}
            └ F2a ──┘    └ F2b ──┘    └ F2c ──┘
            two-body     layer-1      layer-2
            slice        latents      latents
```

Read [`allegro/nn/_allegro.py:239–323`](../../software/allegro-private/allegro/nn/_allegro.py) to follow along.

### 3.1 The two-body slice F2a (`s^{(0)}`)

Built before any Allegro layer runs. The pipeline is:

1. `EdgeLengthNormalizer(r_ij; r_max, t_i, t_j)` — produces a normalized edge length and a polynomial cutoff envelope `c(r_ij)` that goes smoothly to zero at `r_ij = r_max`.
2. `radial_chemical_embed(...)` — projects the normalized length and the (sender, receiver) chemical pair into a `radial_chemical_embed_dim = 32`-dim vector. For baseline-B this is `TwoBodyBesselScalarEmbed`: Bessel basis + per-(t_i, t_j) chemical mixing weights.
3. `scalar_embed_mlp` — small MLP (depth 2, width 64) projecting that to the `num_scalar_features = 32` dim.
4. `Allegro_Module.first_layer_env_embed_projection(scalar_embed)` — a linear that produces the `num_scalar_features` slice + the layer-1 environment-weight allocation. The first 32 entries of its output are stored as `accumulated_scalar_features[0]`. **These 32 entries are F2a.**

In symbols (for atom types `t_i, t_j` and scalar distance `r_ij`):

```
s^{(0)}_ij  =  W^{(0)} · MLP_2B( B_chem(r_ij; t_i, t_j) ) · c(r_ij)
```

where `c(r_ij)` is the polynomial radial cutoff (Behler-style `1 − 6u^5 + 15u^4 − 10u^3` with `u = r_ij / r_max`), guaranteeing `s^{(0)}_ij → 0` smoothly as `r_ij → r_max`. **This vanishing at the cutoff is what makes the sum-pool to atoms `C^k`-smooth in atomic positions** — atoms entering or leaving the neighbor list cause no jumps. See §5.

F2a depends only on the pair `(r_ij, t_i, t_j)`. **No environment information.** Two Pt atoms at the same separation always have identical F2a, regardless of whether they sit in bulk or at a reactive interface. We therefore expect F2a to be the **weakest** signal for the bimodality test.

### 3.2 The layer-1 slice F2b (`s^{(1)}`)

Allegro layer 1 does the following:

1. **Environment weighting.** From the previous layer's output, a chunk of the scalars (`env_w` portion) is sliced off and used to weight the spherical-harmonic tensor basis on the same edge. `_env_weighter(tensor_basis, env_w)` produces tensor-valued weighted-channel features per edge.
2. **Scatter to centers.** These weighted tensors are summed over edges with the same **center** (sender of edge `i → ·`) to form per-atom environment tensors. This is the only place Allegro looks beyond the single edge — but only over the immediate neighbor shell of `i` (no multi-hop).
3. **Tensor product (Contracter).** The edge's own tensor features are tensor-product-contracted with the centered environment tensors of the receiver. Output irreps include scalars and (for non-final layers) higher-`l` harmonics.
4. **Latent MLP.** All scalars accumulated so far (twobody + new TP scalars) are fed into a `latent` MLP with depth 2, width 256. The first 32 entries of its output are appended as `s^{(1)}_ij` (F2b). The remaining entries are `env_w` for the next layer.

The DenseNet concat `[F2a, F2b]` is what enters the next layer's latent MLP — every prior layer's scalar slice persists into all subsequent layers' MLP inputs. This is the "DenseNet-style" architecture noted in [`_allegro.py:198`](../../software/allegro-private/allegro/nn/_allegro.py#L198).

### 3.3 The layer-2 slice F2c (`s^{(2)}`)

Same structure as layer 1 but:
- Layer 2 is the **last layer**, so its tensor-product output is restricted to scalars (`Irreps([(1, (0, 1))])`) — no higher-`l` content survives. See [`_allegro.py:120-135`](../../software/allegro-private/allegro/nn/_allegro.py#L120-L135).
- Latent MLP input is `[F2a, F2b, layer2_TP_scalars]` (DenseNet); output is just `s^{(2)}_ij` (32 dim) since no further layer needs an `env_w` allocation.
- F2c is the most environment-mixed slice. **Expected to carry the strongest bulk-vs-OOD signal.**

### 3.4 Final assembly

```python
data["edge_features"] = torch.cat(
    [accumulated_scalar_features[0],   # F2a, 32
     accumulated_scalar_features[1],   # F2b, 32
     accumulated_scalar_features[2]],  # F2c, 32
    dim=-1)                            # → (E, 96)
```

This is the 96-dim vector the readout MLP turns into per-edge scalar energies. **It is also exactly what we hook to capture the M0 hidden state for the complexity score.**

### 3.5 Per-edge invariance

Every entry of `h_ij` is rotation- and translation-invariant in the input atomic positions (Allegro's whole point). Permutation-equivariance over atoms is preserved through the receiver index. This means **any function of `h_ij` we build is automatically a valid invariant per-atom signal once pooled to the receiver**.

---

## 4. Per-atom signal: pooling F2 to atoms

Given per-edge `h_ij ∈ ℝ^{96}`, a per-atom signal `z_i` requires aggregating over edges incident to atom `i`. The **receiver-side sum-pool** is the design-doc default (see [`notes/sparse_local_correction.md`](../../notes/sparse_local_correction.md) §7) and is what `EdgewiseReduce` already does for energies. We use:

```
z_i  =  Σ_{j: (j → i) ∈ E}  h_ij   ∈   ℝ^{96}
```

This is the simplest density-aware pool. **Smoothness is inherited from `h_ij`**: every `h_ij` carries the radial cutoff envelope `c(r_ij)` from F2a (and indirectly from later layers, since `c(r_ij)` is inside every TP and latent computation). As `r_ij → r_max` from below, `h_ij → 0` continuously, so the sum is `C^k`-smooth in atomic positions across neighbor-list integer crossings. This is **why we sum and not mean** — see [`sparse_local_correction.md`](../../notes/sparse_local_correction.md) §7 for the rejection of mean and max pools.

We do *not* normalize by `√(2·avg_num_neighbors)` here as `EdgewiseReduce` does for energies. That normalization is a per-type rescaling chosen to make the model output well-conditioned at training time; for our complexity score it's a constant and either choice gives identical bimodality. We'll absorb any per-type rescaling into the per-type Mahalanobis statistics in §6.

The same pool, restricted to a slice of `h_ij`, gives `z_i^{F2a}, z_i^{F2b}, z_i^{F2c}` (each in `ℝ^{32}`).

---

## 5. Smoothness — why this matters at all

The whole project hinges on `λ_i = c(s_i(R))` being a `C^2` function of atomic positions `R`, so that forces from a single autograd backward through `E_total = E_0 + Σ_i λ_i E_1^i` are conservative. The chain is:

```
positions R  →  edge lengths r_ij  →  radial cutoff c(r_ij)  →  h_ij(R)  →  z_i(R) = Σ_j h_ij  →  s_i(R) = f(z_i)  →  λ_i = c_gate(s_i)
```

Each arrow is `C^k` for some `k ≥ 2`:
- `r_ij` is a smooth function of positions away from `r_ij = 0` (no two atoms ever overlap in physically meaningful frames).
- `c(r_ij)` is the polynomial cutoff in NequIP/Allegro — `C^2` at `r_ij = r_max` and identically zero beyond.
- All MLPs and tensor products in `h_ij(R)` are smooth (silu activations, no max/abs/relu in the path that survives to F2).
- The sum `z_i = Σ_j h_ij` is smooth because `h_ij = 0` smoothly at the boundary.
- `f(z_i)` for our methods (norm, Mahalanobis quadratic form) is smooth in `z_i`.
- `c_gate` is the polynomial gate — `C^2`.

So `λ_i(R)` is `C^2` in `R`. Conservative forces follow automatically. **None of this depends on training** — the smoothness is structural.

This experiment does not yet touch the gate `c_gate`. It only checks whether `s_i` has a bimodal distribution in the first place. If yes, calibrating `c_gate` is the next step.

---

## 6. Methods tested

Six per-atom methods + one per-edge method. Naming convention: `<feature_source>_<scoring_function>`.

### Atom-level (six)

For each method we compute one scalar per atom on the val set and write it as a custom per-atom array in the output extxyz.

| ID | Source slice of F2 | Pool | Score | Per-atom output field |
|---|---|---|---|---|
| **F2_norm** | full F2, dim 96 | sum to receiver | `‖z_i‖₂` | `s_F2_norm` |
| **F2a_norm** | F2a only (twobody slice), dim 32 | sum to receiver | `‖z_i^{F2a}‖₂` | `s_F2a_norm` |
| **F2b_norm** | F2b only (layer-1 slice), dim 32 | sum to receiver | `‖z_i^{F2b}‖₂` | `s_F2b_norm` |
| **F2c_norm** | F2c only (layer-2 slice), dim 32 | sum to receiver | `‖z_i^{F2c}‖₂` | `s_F2c_norm` |
| **F2_maha** | full F2, dim 96 | sum to receiver | per-type Mahalanobis (see §6.1) | `s_F2_maha` |
| **F2c_maha** | F2c only, dim 32 | sum to receiver | per-type Mahalanobis | `s_F2c_maha` |

### Edge-level (one, no per-atom field)

| ID | Source | Pool | Score | Output |
|---|---|---|---|---|
| **F2_edge_norm** | full F2 | none | `‖h_ij‖₂` per edge | per-pair histogram only |

### 6.1 What S3 (per-type Mahalanobis) is, in full

**Goal**: a number per atom that says how far this atom's pooled feature `z_i` is from the typical pooled feature of atoms of the **same chemical type** in the training distribution. Atoms in unfamiliar environments → far from training mean → high score.

**Why "Mahalanobis" and not just Euclidean**: features in different directions have very different variances. A direction with large training spread is "expected variability" and shouldn't count as anomaly. Mahalanobis whitens by the covariance.

**Definition.** For atom type `t ∈ {C, O, Pt}` collect the training-set pooled features

```
Z_t  =  { z_i  :  i ∈ atoms_train,  type(i) = t }    ⊂   ℝ^d,    d ∈ {96, 32}
```

Compute the per-type sample mean and covariance:

```
μ_t  =  (1/N_t) Σ_{i ∈ Z_t}  z_i
Σ_t  =  (1/N_t) Σ_{i ∈ Z_t}  (z_i − μ_t)(z_i − μ_t)ᵀ        ∈ ℝ^{d×d}
```

Add a small ridge on the diagonal for numerical invertibility and to avoid Σ_t being singular when `d ≈ N_t`:

```
Σ̃_t  =  Σ_t  +  ε · tr(Σ_t)/d · I_d           with ε = 10^{-4}
```

The per-type Mahalanobis score for an atom `i` of type `t = type(i)` evaluated on a val frame is:

```
s_i  =  (z_i − μ_{t(i)})ᵀ  Σ̃_{t(i)}^{-1}  (z_i − μ_{t(i)})        ∈ ℝ_{≥0}
```

This is the squared Mahalanobis distance. Equivalent geometric picture: `s_i` is the squared L2 norm of the **whitened** residual `Σ̃_t^{-1/2} (z_i − μ_t)`. Under a Gaussian model `z | t ∼ N(μ_t, Σ_t)`, the score is `−2 · log p(z | t)` plus a constant — i.e. the log-likelihood up to additive shift. It is the standard **OOD detector** when no labelled OOD set is available; only requires training data.

**Why per-type, not global.** baseline-B per-type force-RMS in training: `C=4.95, O=4.86, Pt=0.94`. An order of magnitude per-type asymmetry. A single global `(μ, Σ)` would map Pt onto the C/O scale and ruin the test — every Pt atom would look "anomalous" relative to the global mixture-of-types mean. Per-type stats normalize this out.

**Why Mahalanobis is differentiable in `z_i`.** `Σ̃_t^{-1}` is constant per type (computed once on the training set, then frozen). The only part that depends on positions is `(z_i − μ_t)`. So:

```
∂s_i/∂z_i  =  2 · Σ̃_t^{-1} (z_i − μ_t)
```

a smooth linear function of `z_i`. Combined with `z_i(R)` being `C^∞` in positions (as established in §5), `s_i(R)` is `C^∞`. **Conservative-force-compatible.**

**Computation cost at inference.** Quadratic form on a 96-dim vector: ~10⁴ FLOPs per atom. Negligible compared to a single forward pass through M0. Per-type `Σ̃_t^{-1}` precomputed once.

**Computation cost during stats pass.** Storing `Σ_t = (1/N_t) Σ z_i z_iᵀ − μ_t μ_tᵀ`. Streaming-stable form: accumulate `S1_t = Σ z_i ∈ ℝ^d`, `S2_t = Σ z_i z_iᵀ ∈ ℝ^{d×d}`, `N_t ∈ ℕ`; then `μ_t = S1_t / N_t`, `Σ_t = S2_t/N_t − μ_t μ_tᵀ`. We use the streaming form. For `d = 96` the per-type S2 matrix is 96×96 = 9.2k floats — trivial. We also store the F2c-only (32-dim) submatrix as a sub-block of the full S2.

### 6.2 Sub-block trick for F2c statistics

The F2c-only mean is just the last 32 components of the full mean: `μ_t^{F2c} = μ_t[64:96]`. The F2c-only covariance is the bottom-right 32×32 block of the full covariance: `Σ_t^{F2c} = Σ_t[64:96, 64:96]`. So we only need one streaming pass storing the 96×96 cross-moments, and the F2c stats fall out by slicing. No second pass.

---

## 7. M7: per-edge norm by edge-type pair

For every directed edge `(j → i)` we record the triple `(type(i), type(j), ‖h_ij‖₂)`. We then plot histograms split by the **directed** pair `(type_recv, type_send)`. If the pair-resolved distributions are well-separated (e.g. Pt-Pt edges concentrated low, C-Pt edges concentrated high), the per-edge gate of [`sparse_local_correction.md`](../../notes/sparse_local_correction.md) §7 has a clear signal — useful fallback if the per-atom diagnostic is mushy.

The per-edge score has no clean "per-atom" interpretation, so it does not go into the OVITO extxyz file. It is a histogram-only diagnostic.

---

## 8. Procedure

The script does three things:

### 8.1 Forward hook

Locate the `Allegro_Module` instance inside the loaded `GraphModel` (via name match `type(m).__name__ == "Allegro_Module"`). Register a forward hook that grabs `out_dict["edge_features"]` and stores it (detached, cloned) in a buffer. Each call to `model(data)` thus refreshes the buffer with the post-Allegro F2 tensor for that frame.

### 8.2 Stats pass over training set

For each train frame:
1. Run forward → buffer holds `h_ij ∈ ℝ^{E × 96}`.
2. Sum-pool to receivers: `z_i = Σ_j h_ij` using `torch.zeros(N, 96).index_add_(0, edge_index[0], h_ij)`. Receiver convention from `EdgewiseReduce` ([`allegro/nn/edgewise.py:44`](../../software/allegro-private/allegro/nn/edgewise.py#L44)).
3. Accumulate per-type streaming stats: `S1_t += z_t.sum(0)`, `S2_t += z_t.T @ z_t`, `N_t += z_t.shape[0]`, where `z_t = z[atom_types == t]`.
4. Promote accumulators to `float64` to avoid numerical drift over thousands of frames.

After the pass:
- `μ_t = S1_t / N_t`
- `Σ_t = S2_t / N_t − μ_t μ_tᵀ`
- `Σ̃_t = Σ_t + ε · tr(Σ_t)/d · I`
- `Σ̃_t^{-1}` via `torch.linalg.inv`

Save as `experiments/0-m0_complexity_probe/results/stats_train.npz` for reuse.

### 8.3 Score pass over val set

For each val frame:
1. Run forward → buffer holds `h_ij`.
2. Sum-pool to receivers → `z_i`.
3. Compute all six atom-level scores and the M0 per-atom energy.
4. Append the original ASE Atoms with new `arrays` entries (one per method) and a frame-level `info["composition_class"]` tag.
5. Also accumulate per-edge `(type_recv, type_send, ‖h_ij‖₂)` triples for M7.

Output:
- `experiments/0-m0_complexity_probe/results/val_with_si.xyz` — concatenated extxyz, one frame block per val frame, with per-atom `s_F2_norm`, `s_F2a_norm`, `s_F2b_norm`, `s_F2c_norm`, `s_F2_maha`, `s_F2c_maha`, `m0_per_atom_energy`. **Loadable in OVITO; switch atomic-property coloring to compare methods visually.**
- `experiments/0-m0_complexity_probe/results/figures/hist_<method>.png` — overlaid histograms by composition class.
- `experiments/0-m0_complexity_probe/results/figures/violin_<method>.png` — per-class violin plots.
- `experiments/0-m0_complexity_probe/results/figures/edge_norms_by_pair.png` — M7 distributions.

### 8.4 What success looks like

For at least one of the six atom-level methods:
- The val-set histogram **should** be non-Gaussian (multimodal or heavy-tailed).
- Splitting by composition class **should** show distinct modal locations: `Pt-only` low, `mixed` high (or partially overlapping but visibly shifted).
- Per-type Mahalanobis (S3) **should** outperform plain norm (S1) — especially on F2c — because the heavy-tailed direction in feature space is whitened out.

If none of the six show separation, the next move is supervised (S6 in the design doc): train a small head to regress `|E_DFT^i − E_0^i|` per atom, since that is exactly what the M1 correction will need to predict.

---

## 9. Caveats and subtleties

- **Edge index convention.** In `allegro/nn/edgewise.py:44`, `edge_dst = data[EDGE_INDEX_KEY][0]` is the receiver. We replicate that. Getting it backwards would silently invert sender/receiver and produce wrong M7 pair labels (and a different — but still valid — pooling). Sanity-check by confirming `‖z_Pt‖` is much bigger than `‖z_C‖` on bulk-Pt frames (Pt has more neighbors; sum-pool grows with degree).
- **Receiver vs sender for M7.** We label edges by `(recv, send)` since that's what the receiver-side aggregation cares about. C–Pt and Pt–C will appear as separate histograms — they have the same `r_ij` but the sender/receiver chemical identity differs in `radial_chemical_embed`, so `h_ij ≠ h_ji` in general.
- **Train/val matmul precision.** The packaged model is `model_dtype: float32`. We disable mixed precision and run forward in fp32 to match training conditions. Train-time stats accumulators are in fp64.
- **Compile mode.** We load the package in `eager` mode. The `compile` mode runs torch.compile, which can interfere with hooks because compiled subgraphs are atomic units. Eager is the safe choice for instrumentation.
- **Force gradients.** The package wraps the energy model in `ForceStressOutput`, which sets `positions.requires_grad_(True)` and runs autograd to derive forces. Our hook captures `edge_features` *before* any autograd-on-positions backward, so the captured tensor is just the forward activations. We `.detach().clone()` in the hook to be defensive about graph leaks.
- **Frames are processed one at a time, not batched.** Cameron frames vary 43–339 atoms; padding into a batch buys little speed and complicates the per-frame extxyz output. Per-frame loop is simple and fast enough on A100.
- **Per-type stats need enough atoms.** If the train set has very few atoms of one type (e.g. C in mostly-Pt frames), `Σ_t` may be poorly conditioned. The ridge regulator handles this. We log per-type counts at stats time; if any type has fewer than ~10·d samples (`< 960` for d=96, `< 320` for d=32), the Mahalanobis score for that type is treated as a rough proxy and noted in the outcome.
- **Cameron CO/Pt species.** All three of C, O, Pt are present per cameron metadata. Per-type sample sizes will be checked and logged.

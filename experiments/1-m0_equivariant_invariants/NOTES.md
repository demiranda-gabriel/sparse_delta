# NOTES — 1-m0_equivariant_invariants

Theory and methodology for the equivariant-invariants probe. The script [`probe.py`](probe.py) implements what is described here.

Read this after [`0-m0_complexity_probe/NOTES.md`](../0-m0_complexity_probe/NOTES.md), which covers the Allegro DenseNet feature layout, sum-pooling smoothness, and the per-type Mahalanobis machinery we re-use conceptually here.

---

## 1. Why pre-final-layer

By construction, Allegro's **last** layer restricts its tensor-product output to scalars: in [`allegro/nn/_allegro.py:120-135`](../../software/allegro-private/allegro/nn/_allegro.py#L120-L135),

```python
if layer_idx == self.num_layers - 1:
    ir_out = Irreps([(1, (0, 1))])  # last layer: scalars only
else:
    ir_out = self.tensor_track_allowed_irreps  # interior: all allowed l
```

So `tps[-1].irreps_out = "1x0e"`, and `edge_features` (the DenseNet concat fed to the readout MLP) carries no `l > 0` content. To get equivariant per-edge features we hook the **second-to-last** TP, `tps[num_layers - 2]`, whose `irreps_out` includes `tensor_track_allowed_irreps` (modulo path-pruning). For `baseline-B` with `tensor_track_allowed_irreps = 1x0e + 1x1o + 1x1e` and `l_max = 1, parity = true`, the pre-final TP outputs `mul × (1x0e + 1x1o + 1x1e)` per edge with `mul = num_tensor_features = 32`. The trailing dimension of the Contracter output is `1 + 3 + 3 = 7` per channel.

This requires `num_layers >= 2`. For `num_layers = 1` the only TP is the final scalar-only one, and there is no pre-final layer to hook — the probe asserts and fails.

The hook is generic: at load time we read `allegro_mod.num_layers`, `allegro_mod.tps[num_layers - 2].mul`, and `tps[num_layers - 2].irreps_out`, then build the slot table dynamically. Same script works at any `num_layers >= 2` and any `l_max` (though we only compute bispectrum for `l = 1` blocks).

---

## 2. Per-node equivariant feature

Per-edge tensor `t_{j→i} ∈ ℝ^{mul × base_dim}` where `base_dim = Σ_{ir ∈ irreps_out} (2l_ir + 1)` and the last axis is strided in the irrep-out order (scalars first; see `Contracter.__init__` assertion `assert all(ir == SCALAR for _, ir in tp.irreps_out[:n_scalar_outs])`).

Sum-pool to the receiver atom (Allegro convention `edge_index[0] = receiver`, [`allegro/nn/edgewise.py:44`](../../software/allegro-private/allegro/nn/edgewise.py#L44)):

```
z_i = Σ_{j: (j → i) ∈ E} t_{j→i}      ∈ ℝ^{mul × base_dim}
```

Splitting `z_i` along the last axis by `irreps_out`:

```
z_i = [ z_i^{(l=0, ...)}, z_i^{(l_1, p_1)}, z_i^{(l_2, p_2)}, ... ]
```

where each `z_i^{(l,p)} ∈ ℝ^{mul × (2l+1)}`. The `l = 0` blocks are pure scalars (and overlap conceptually with what the existing `s_F2c_norm` already uses); we skip them here and focus on `l > 0`.

**Smoothness in `R`.** Every entry of `t_{j→i}` is built from operations that inherit the radial cutoff envelope `c(r_{ij}) → 0` as `r_{ij} → r_max`, so the sum is `C^k`-smooth in atomic positions (no integer jumps when neighbors enter/leave the list). Polynomials of smooth quantities are smooth, so all invariants below remain smooth in `R` — conservative-force-compatible.

---

## 3. Cross-channel power spectrum

For one `(l > 0, parity)` irrep block, with `mul = C` channels:

```
P^{ab}_{i, l} = Σ_m  z_{i, a, l}^m  z_{i, b, l}^m       a, b ∈ [0, C)
```

This is the `(a, b)` entry of the per-atom Gram matrix on the `(2l+1)`-dim irrep.

### 3.1 Why it is SO(3)-invariant

Under rotation `R ∈ SO(3)`, each channel transforms as `z_{i, a, l} → D^l(R) z_{i, a, l}`. e3nn's real Wigner-D matrix is **orthogonal**: `D^l(R)^⊤ D^l(R) = I_{2l+1}`. Hence

```
P^{ab}_{i, l}  →  (D^l z_{i, a, l})^⊤ (D^l z_{i, b, l})
              =   z_{i, a, l}^⊤ (D^l)^⊤ D^l z_{i, b, l}
              =   z_{i, a, l}^⊤ z_{i, b, l}
              =   P^{ab}_{i, l}
```

Each `(a, b)` pair is a separate invariant. The Gram matrix is symmetric in `(a, b)` since dot products commute, so we keep the upper triangle including the diagonal:

```
n_pairs = C(C+1)/2 = 528    for C = 32.
```

### 3.2 Connection to the SH power spectrum

This is exactly the construction in §7.1 of [`notes/signals_on_spheres.pdf`](../../notes/signals_on_spheres.pdf), with two extensions:

- We work with **per-channel** features rather than a single signal on the sphere. The signals here are the 32 learned `l = 1` channels per atom, not the raw atomic-density expansion.
- We keep the full Gram (cross-channels), not just the diagonal `Σ_m |f_l^m|²`. The cross-channel terms `P^{ab}` with `a ≠ b` carry information that the per-channel norms discard: they measure alignment between channel `a` and channel `b` at the same atom. This is the SOAP construction (Bartók et al. 2013).

### 3.3 Cost

Per atom: `O(C² (2l+1))` multiply-adds. For `C = 32, 2l+1 = 3`: ~3.1k FLOPs. For 20k val atoms: ~60M FLOPs total. Trivial compared to the M0 forward (~5–10 GFLOPs over val).

---

## 4. Bispectrum: (1,1,1) → 0, antisymmetric channel triplets

The bispectrum on a single signal (one channel) is defined (§7.2 of the PDF) as

```
B_{l1, l2, l3}  =  Σ_{m1, m2, m3}  C_{l1 m1, l2 m2}^{l3 m3}  f_{l1}^{m1}  f_{l2}^{m2}  f_{l3}^{m3}
```

with the CG selection rule `|l1 − l2| ≤ l3 ≤ l1 + l2`. For multi-channel features we generalize to channel triples `(a, b, c)`:

```
B^{a, b, c}_{l1, l2, l3}  =  Σ_{m1, m2, m3}  C_{l1 m1, l2 m2}^{l3 m3}
                              z_{a, l1}^{m1}  z_{b, l2}^{m2}  z_{c, l3}^{m3}
```

### 4.1 The only genuinely-cubic family at l_max = 1

Enumerating triples with `0 ≤ l_i ≤ 1` and triangle inequality, the bispectrum entries that yield a scalar output (`l3` value present in our rep) are:

| `(l1, l2, l3)` | What it computes |
|---|---|
| `(0, 0, 0)` | Triple scalar product — equivalent to a higher-order cross-channel descriptor on the `l=0` slot, already covered by scalar features. |
| `(0, 1, 1)`, `(1, 0, 1)`, `(1, 1, 0)` | Each is a scalar times an `l=1` inner product — already covered by cross-channel power spectrum at `l=1`. |
| `(1, 1, 1)` | Genuinely cubic in `l=1` features. **This is the only new family.** |

So the `(1, 1, 1)` triple is the unique cubic invariant at `l_max = 1`. Higher `l_max` would unlock `(1, 1, 2), (2, 2, 0), (2, 2, 2)`, etc.

### 4.2 (1, 1, 1) → 0 is the triple product

The Clebsch-Gordan coupling `1 ⊗ 1 → 1` is **antisymmetric** in `(m1, m2)`. Geometrically, on `ℝ^3` vectors, this is the cross product (up to a normalization constant):

```
(z_a ⊗ z_b)_{l=1, m} ∝ (z_a × z_b)_m
```

Then contracting with `z_c` (l=1) gives a scalar — the standard triple product:

```
B^{a, b, c}_{1,1,1} ∝ (z_a × z_b) · z_c = det[ z_a | z_b | z_c ]
```

The script uses `torch.cross` on the strided `[m=−1, m=0, m=+1]` axis directly. The cross product is basis-independent (an axial vector is the Hodge dual of an antisymmetric 2-form), and the determinant of three vectors is invariant under cyclic relabeling of the basis, so e3nn's `(y, z, x)` ordering of the real `l=1` basis vs. the Cartesian `(x, y, z)` ordering does not affect the value of the triple product (up to an overall sign that is absorbed downstream when we use `|B|` or square the invariants).

### 4.3 Why "diagonal" is zero

The triple product is fully antisymmetric in channel indices `(a, b, c)`:

```
det[z_a | z_a | z_c] = 0           (two identical columns)
det[z_a | z_b | z_a] = 0
det[z_a | z_b | z_b] = 0
```

Any repetition of a channel index kills the determinant. The only nonzero entries are those with **three distinct channels**, and by full antisymmetry, only the ordered subset `a < b < c` gives independent invariants:

```
n_triplets = C(C-1)(C-2)/6 = 4960    for C = 32.
```

"Diagonal-only" (`a = b = c`) would give zero — this was an error in earlier discussion. Antisymmetric triplets are the minimal nontrivial channel restriction.

### 4.4 Parity

Both `1o` (true vectors, polar) and `1e` (pseudovectors, axial) blocks can appear in `tps[-2].irreps_out` when `parity = true`. The probe computes the bispectrum **within each parity block separately** (not across blocks). The resulting invariants are:

- From `1o × 1o × 1o`: pseudoscalar (0o) — the standard chirality detector for polar vectors.
- From `1e × 1e × 1e`: true scalar (0e).

Both are SO(3)-invariant. Under inversion the `0o` flips sign while the `0e` is preserved. For our gate we treat both as invariants (the gate is SO(3)-invariant by design; inversion-invariance is a stronger condition we do not need).

### 4.5 Why this is the natural chirality detector

Curie's principle (§9 of [`notes/signals_on_spheres.pdf`](../../notes/signals_on_spheres.pdf)): every symmetry element of the cause is preserved in the effect. Centrosymmetric atomic environments (bulk-Pt, gas-phase CO with `O_h` or `D_∞h` symmetry on average) project onto only the `l` values whose branching under the local point group contains the trivial irrep `A_1`. For `O_h`, the surviving values are `l = 0, 4, 6, ...`; `l = 1` is killed. So the bispectrum entries built from `l = 1` features are forced to **zero** for sufficiently symmetric environments and grow with deviation from that symmetry. This is exactly the smooth complexity signal we need for the gate.

### 4.6 Cost

Per atom: each triple product is one cross product (6 mul + 3 add) followed by one dot product (3 mul + 2 add) = ~14 FLOPs. With 4960 triplets: ~70k FLOPs/atom. Over 20k val atoms: ~1.4 GFLOPs total. Memory: chunked over atoms (default 2048) to keep peak memory bounded at ~120 MB for the `[chunk, T, 3]` intermediate. Below the model forward cost.

---

## 5. Summary scalars and what to inspect

The probe stores raw invariants in `val_invariants.npz` and stamps a few summary scalars onto the val extxyz for OVITO:

| Field | Meaning |
|---|---|
| `s_F2c_norm` | The existing baseline from `0-m0_complexity_probe` (post-Allegro F2c block norm). |
| `s_power_l1{o,e}_norm` | L2 norm of the cross-channel Gram (upper triangle) for the `l=1, parity=o/e` block. One per `(l, parity)` slot present in the pre-final-layer irreps. |
| `s_power_total_norm` | L2 norm of the concatenated power-spectrum vector across all `l > 0` slots. |
| `s_bispec_l1{o,e}_norm` | L2 norm of the antisymmetric-triplet bispectrum for the `l=1, parity=o/e` block. |
| `s_bispec_total_norm` | L2 norm across all `l=1` bispectrum blocks. |

Comparisons of interest:

- `s_power_l1*_norm` vs. `s_F2c_norm` per-class histogram. Expect at least equal separation; ideally tighter.
- `s_bispec_l1*_norm` per-class histogram. Expect bimodal: low for `Pt-only` (more centrosymmetric on average), higher for `mixed` (CO/Pt interface, broken inversion).
- Spatial map in OVITO: color val frames by `s_bispec_total_norm`. Adsorbate sites and asymmetric interfaces should light up.

---

## 6. Caveats

- **F2c slice derivation.** The probe derives the F2c slice from `F2_dim = num_scalar_features × (num_layers + 1)`; for `baseline-B` (num_scalar_features=32, num_layers=2) this gives 96 and F2c is `[64:96]`. If a future M0 has a different `num_layers`, the slice still tracks the **last** `num_scalar_features` of the DenseNet concat (which is the latents from the final Allegro layer's TP). This is the same slice the prior probe called F2c.
- **Sum-pool sign.** `torch.cross` and the determinant pick an overall sign tied to the right-hand rule on the `[m=−1, m=0, m=+1]` axis ordering. Squared norms `s_bispec_*_norm` are sign-invariant. If we ever want signed bispectrum values for chirality discrimination, we should fix a convention (and probably switch to `s_bispec_signed`) — for now we only look at norms.
- **Per-type stats not yet computed.** Unlike `0-m0_complexity_probe`, this probe does **not** run a stats pass over train. With 5500-dim invariant vectors, a full per-type Mahalanobis is impractical (`5500 × 5500` covariance per type). A useful next step if `s_bispec_*_norm` alone is not sharp enough is to compute per-type Mahalanobis on the cross-channel power-spectrum vector alone (528-dim per `(l, parity)` slot) or on PCA-reduced features. Defer until needed.

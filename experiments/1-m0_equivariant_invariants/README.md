# 1-m0_equivariant_invariants

**Status:** planned
**Date:** 2026-05-15
**Outer SHA:** _filled at commit time_
**Submodule SHAs:** `nequip-private=…  allegro-private=…`
**WandB:** none — diagnostic only, no training

## Intent

Extends [`0-m0_complexity_probe`](../0-m0_complexity_probe/) with **equivariant** per-atom invariants. The original probe used the post-Allegro `edge_features` tensor (96-dim scalar DenseNet concat — the final Allegro layer contracts its TP output to `0e` only, dropping all `l > 0` content). This probe instead hooks the **second-to-last** Allegro layer's Contracter output, which retains higher-l content. We sum-pool the per-edge tensor to receiver atoms and build SO(3)-invariant descriptors from the resulting per-node equivariant feature.

Two invariant families:

1. **Cross-channel power spectrum** for each `l > 0` irrep block: `P^{ab}_i = Σ_m h^m_{i,a,l} h^m_{i,b,l}`. Upper triangle (incl. diagonal): `C(C+1)/2` invariants per atom per block. For `baseline-B` (`C = 32`, `l_max = 1`, parity true) that's 528 per `(l, parity)` slot.
2. **(1,1,1) → 0 bispectrum** on `l = 1` blocks, antisymmetric channel triplets `a < b < c`. Geometrically the triple product `det[h_a | h_b | h_c]`. With `C = 32`: 4960 invariants per atom per `(l=1, parity)` slot.

The "diagonal" `(a = b = c)` version is identically zero because `(v × v) · v = 0` — the `(1,1,1) → 0` CG path is antisymmetric in the first two indices, so any channel repetition kills the invariant. Antisymmetric triplets `a < b < c` are the minimal nontrivial choice; cost is still well below the M0 forward.

The probe is `num_layers`-agnostic: it reads `num_layers` and `num_tensor_features` and `irreps_out` from the loaded `Allegro_Module` at runtime, and always hooks `tps[num_layers - 2]`. Requires `num_layers >= 2`; fails loudly otherwise. See [NOTES.md](NOTES.md) for the math.

## Hypothesis

Equivariant invariants should strictly extend the discriminative power of `s_F2c_norm` from the prior probe, because they expose information that the all-scalar `edge_features` discards. Concretely:

- Cross-channel power spectrum at `l = 1` carries information about angular alignment among the 32 `l = 1` channels at each atom — this is *missing* from scalar features.
- The `(1,1,1)` bispectrum carries chirality/handedness information. Centrosymmetric environments (bulk-Pt, gas-phase CO with inversion symmetry) should give zero or near-zero bispectrum values; asymmetric environments (interface Pt, distorted geometries) should give nonzero values. This is **the natural soft chirality detector** flagged in §9 of [`notes/signals_on_spheres.pdf`](../../notes/signals_on_spheres.pdf) (Curie's principle: symmetric inputs zero out matching spectral entries).

Falsified if (a) class histograms of all the new summary scalars overlap perfectly across composition classes, or (b) the bispectrum norm is uniformly small everywhere, indicating `l_max = 1` is too low for the bispectrum to capture meaningful asymmetry.

## Success criteria

Visual, on val frames:

- `s_power_l1{o,e}_norm` per-class histograms show separation at least as good as `s_F2c_norm`.
- `s_bispec_l1{o,e}_norm` is **bimodal** across composition classes — concentrated near zero for `Pt-only` (high local symmetry on average) and shifted upward for `mixed` frames containing CO/Pt interfaces.
- `s_F2c_norm` is reproduced consistent with [`0-m0_complexity_probe`](../0-m0_complexity_probe/results/) within float32 noise (sanity check that the dual hook setup is correct).

## Run

Interactive on a local A100 (no SLURM). Same machine and data path as the prior probe.

```bash
cd /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta
uv run python experiments/1-m0_equivariant_invariants/probe.py
```

Outputs in `experiments/1-m0_equivariant_invariants/results/`:

- `val_invariants.npz` — raw invariants per atom, stacked over val frames (`power_l1{o,e}`, `bispec_l1{o,e}` arrays + `frame_idx`, `atom_type`, `composition_class`).
- `val_with_si.xyz` — extxyz with per-atom summary scalars (`s_power_*_norm`, `s_bispec_*_norm`, `s_F2c_norm`) for OVITO inspection.
- `figures/hist_*_by_class.png` and `hist_*_by_atomtype.png` — per-class and per-atom-type histograms of summary scalars.
- `summary.json` — counts, slot info, output paths.
- `probe.log` — run log.

## Outcome

_Filled in after the run._

## What lives here

- [README.md](README.md) — this file.
- [NOTES.md](NOTES.md) — math: power-spectrum invariance proof, why `(1,1,1)` diagonal is zero, antisymmetric-triplet derivation, smoothness chain, cost analysis.
- [probe.py](probe.py) — the script.

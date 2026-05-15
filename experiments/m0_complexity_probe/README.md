# m0_complexity_probe

**Status:** done-success
**Date:** 2026-05-08
**Outer SHA:** _filled at commit time_
**Submodule SHAs:** `nequip-private=74c7689f  allegro-private=82d7258`
**WandB:** none — diagnostic only, no training

## Intent

§12 distribution diagnostic from [`notes/sparse_local_correction.md`](../../notes/sparse_local_correction.md). Decide whether the sparse-delta idea has a chance on the cameron CO/Pt dataset by checking whether the per-atom complexity score `s_i` derived from the M0 (`baseline-B` Allegro) hidden features is **bimodal** across structure classes.

If `s_i` is unimodal across simple-vs-complex environments, the smooth gate has no boundary to bite, and the project pivots before any training is invested. If bimodal, we proceed to §11 (gate calibration + M1 residual training).

No training, inference-only, single-pass over train + val sets to compute statistics + scores.

## Hypothesis

Given the cameron CO/Pt dataset is a heterogeneous mixture of bulk-Pt, Pt-slab, Pt-nanoparticle, gas-phase CO, and CO/Pt interface frames, we expect:

- Bulk-Pt atoms in well-coordinated environments → low `s_i`.
- CO/Pt interface atoms → high `s_i`.
- Gas-phase CO atoms → either very low (M0 trained on plenty of CO) or very high (rare environments) — informative either way.

Falsified if all three classes overlap on a single mode for every method tested. Falsified if Mahalanobis adds nothing over a plain norm — would indicate the feature distribution is too isotropic for OOD-style scoring.

## Success criteria

- At least one of the 6 atom-level methods produces a histogram with **clear bimodal separation** between {bulk-Pt} and {CO/Pt interface} atoms (visual inspection sufficient at this stage; quantify with KS-test later if needed).
- Per-edge `‖h_ij‖₂` distribution shows separation by edge-type pair (e.g. C–Pt edges sit at higher norm than Pt–Pt edges).
- Generated extxyz dataset visualizes meaningfully in OVITO when atoms are colored by `s_i`.

## Run

Interactive on local A100. No SLURM.

```bash
cd /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta
uv run python experiments/m0_complexity_probe/probe.py
```

Outputs land in `experiments/m0_complexity_probe/results/`:
- `val_with_si.xyz` — extxyz with per-atom `s_i` arrays for OVITO.
- `figures/*.png` — histograms, per-class violins.
- `probe.log` — stdout/stderr.
- `stats_train.npz` — train-set per-type μ, Σ for reuse.

## Outcome

Bimodality test **passed clearly**. Run stats:

- 1956 train frames, 244 val frames. ~157k Pt atoms in train, 25k C, 25k O. Val: 13.3k Pt, 3.3k C, 3.3k O. Classes: 17320 mixed-frame atoms, 1969 Pt-only-frame atoms, 672 CO-only-frame atoms.
- Forward time: stats pass 14.7 s (~133 fps), score pass 2.2 s (~110 fps) on A100 40GB.

### Key result — `s_F2c_norm` (sum-pool of layer-2 latents, L2 norm)

Trimodal by composition class:
- `Pt-only` frames: peak ~7–8 (bulk-like Pt atoms with familiar coordination).
- `mixed` (CO/Pt) frames: clearly bimodal — one peak ~7 (bulk-like Pt below the surface), another ~13–15 (interface Pt + adsorbate-influenced atoms), and ~17–18 (CO atoms).
- `CO-only` frames: tight peak at ~17–18 (gas-phase CO).
- Crucially, **bimodality WITHIN the mixed class** — exactly what the gate needs. Atoms in the "bulk-like" sub-mode of mixed are candidates for `λ_i = 0`; atoms in the "interface" sub-mode are candidates for `λ_i > 0`.
- Initial gate threshold candidates from this distribution: `s_low ≈ 9`, `s_high ≈ 13` (per-atom Allegro F2c sum-norm). To be calibrated against an active-fraction target later.

### Method ranking (visual, by separability)

1. **`s_F2c_norm`** — cleanest per-class separation. Layer-2 latents only. **Recommended for the v0 gate.**
2. **`s_F2_norm`** — full 96-dim norm. Also trimodal but separations dominated by F2a (degree) component. Coarser.
3. **`s_F2b_norm`** — moderate, layer-1 latents only. Pt cleanly bimodal but C/O peaks closer together.
4. **`s_F2a_norm`** — twobody slice only. Looks like a degree (coordination-number) proxy: Pt atoms peak ~50–65 (high coordination), C/O cluster ~15–22. Driven by neighbor count, not by environment chemistry. Confirms the §6 prediction in NOTES.md.
5. **`s_F2_maha`** / **`s_F2c_maha`** (per-type Mahalanobis) — surprisingly worse for *between-class* separation, because Mahalanobis whitens out the per-type asymmetry that drives most of the visible signal here. They concentrate near zero for in-distribution atoms (esp. O — every val O is in a CO molecule, perfectly familiar; very tight delta at zero) and produce heavy tails for OOD. **Better OOD detector for within-type "find the rare configs" use case** — but for the gate-trigger task, plain `s_F2c_norm` is already superior. Keep Mahalanobis as a per-type-fairness fallback if `s_F2c_norm` proves biased toward Pt.

### Per-edge (M7)

Per-pair `‖h_ij‖₂` distributions are sharply peaked near zero for most pairs, with a distinct high-norm tail for Pt–Pt edges (clipped at ~26–28). Suggests per-edge gating could give an additional axis of sparsity for *which* Pt–Pt interactions to gate, but the per-atom signal is already strong enough that we don't need it for v0.

### Files for inspection / next steps

- [`results/val_with_si.xyz`](results/val_with_si.xyz) — 244 val frames, 19961 atoms, with per-atom arrays `s_F2_norm`, `s_F2a_norm`, `s_F2b_norm`, `s_F2c_norm`, `s_F2_maha`, `s_F2c_maha`, `m0_per_atom_energy`, `atom_type_index`. Frame `info["composition_class"]` ∈ {Pt-only, CO-only, mixed}. **Open in OVITO; switch atom coloring to any `s_*` array.**
- [`results/stats_train.npz`](results/stats_train.npz) — `μ_t`, `Σ_t`, `Σ̃_t`, `Σ̃_t^{-1}`, per-type counts (`N_per_type`). Can be reused to score new frames.
- [`results/figures/`](results/figures/) — 19 PNGs (12 histograms = 6 methods × {by-class, by-atom-type}, 6 violins by class, 1 edge-norm overlay).
- [`results/edge_norms.npz`](results/edge_norms.npz) — per-edge raw arrays for further M7 analysis.
- [`results/summary.json`](results/summary.json) — counts + paths.

### Decision for next round

Proceed to §11. Use `s_F2c_norm` as the complexity score for the v0 gate. Calibrate `(s_low, s_high)` against a target `mean(λ)` over the val set (proposal: `mean(λ) ≈ 0.15`) once an active-fraction objective is fixed.

## What lives here

- [README.md](README.md) — this file.
- [NOTES.md](NOTES.md) — theory: what F2 is (Allegro DenseNet edge embedding), what S3 (per-type Mahalanobis) computes and why, smoothness argument.
- [probe.py](probe.py) — the script.

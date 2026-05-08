# sparse_delta

Sparse, smooth, locally-triggered multi-fidelity correction for ML interatomic potentials.

```
E_total = E_0 + Σ_i λ_i · E_1^i
```

A small foundation model `M0` runs everywhere. Its hidden features feed a smooth gate `λ_i` that is exactly zero in simple environments. A larger correction model `M1` (strictly local) runs only on the active subgraph. Forces remain conservative; `M1` compute scales with the active fraction.

## Quick start

```bash
cd software
bash install.sh        # clones nequip-private + allegro-private (read-only forks), uv sync registers workspace
```

## Read first

- [CLAUDE.md](CLAUDE.md) — project layout, environment, run/experiment workflow.
- [notes/sparse_local_correction.md](notes/sparse_local_correction.md) — full design rationale, math, halo analysis, prototype skeleton.
- [START_HERE.md](START_HERE.md) — first task for an agent dropped into this repo.

## Status

Planning / scaffolding. No experiments yet.

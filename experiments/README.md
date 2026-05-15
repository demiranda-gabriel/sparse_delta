# Experiments

Each subdirectory is a single experiment (or a sweep). One commit per experiment, pinned **before** `sbatch` so the SHA reproduces the submitted state.

## Naming convention

Experiment directories are named **`N-exp_name`** where `N` is a zero-padded-not-required integer assigned in creation order starting from `0` (oldest first). The numeric prefix gives a stable, sortable directory listing and a short reference for cross-links (`exp 1` is unambiguous). The `_template/` directory is exempt (it's not a real experiment).

When creating a new experiment, take `N = max(existing N) + 1` and use `<N>-<exp_name>/`. Do **not** renumber existing experiments after the fact — their `N` is permanent. (If two experiments are created in parallel and collide on `N`, resolve by creation timestamp at commit time.)

## Layout per experiment

Mirrors `_template/`:

```
<N>-<exp_name>/
├── README.md          # status, intent, hypothesis, success criteria, SHAs
├── config.yaml        # nequip / allegro training config
├── train.sh           # SLURM script
└── logs/              # SLURM stdout/stderr (created at run time)
```

For sweeps:

```
<N>-<exp_name>/
├── README.md
├── tags.txt
├── configs/
│   ├── base.yaml
│   ├── <tagA>.yaml
│   └── <tagB>.yaml
├── <tagA>/train.sh
├── <tagB>/train.sh
└── submit_all.sh      # self-locating wrapper
```

## Status vocabulary

`planned | running | done-success | done-fail | abandoned`

## Index

Reverse-chronological. Add a row in the same commit that creates the experiment directory.

| N | Date | Experiment | Status | Intent | WandB |
|---|---|---|---|---|---|
| 2 | 2026-05-15 | [2-allegro_L_correction](2-allegro_L_correction/) | planned | Train L-config Allegro (`l-l3-paper` from mtfd) on cameron CO/Pt as the M1 correction baseline on Frontier (ROCm). | _filled at submission_ |
| 1 | 2026-05-15 | [1-m0_equivariant_invariants](1-m0_equivariant_invariants/) | done-success | §12 extension: cross-channel power spectrum + (1,1,1) antisymmetric-triplet bispectrum on M0 pre-final-layer equivariant features. | _no run; inference-only_ |
| 0 | 2026-05-08 | [0-m0_complexity_probe](0-m0_complexity_probe/) | done-success | §12 distribution diagnostic on cameron CO/Pt: do M0 (baseline-B) hidden features yield bimodal `s_i`? | _no run; inference-only_ |

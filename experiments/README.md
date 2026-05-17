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
| 4 | 2026-05-17 | [4-composite_warmstart](4-composite_warmstart/) | planned | First training run of the from-scratch warm-start composite. M0 and M1 are both fresh `Allegro_Module`s; M1's `tensor_features_in_field` is aliased to M0's `_allegro_pre_final_tp_out`. Joint training; nothing frozen. FASRC `kozinsky_gpu`. | _filled at submission_ |
| 3 | 2026-05-15 | [3-composite_phase_A](3-composite_phase_A/) | planned | Phase A composite training: load pre-trained M0 (baseline-B) and M1 (from exp 2) `.nequip.zip` packages, freeze both, train only the gate. Distinct from exp 4 (architectural warm-start, joint training). | _blocked on exp 2 ckpt_ |
| 2 | 2026-05-15 | [2-allegro_L_correction](2-allegro_L_correction/) | running | Train L-config Allegro (`l-l3-paper` from mtfd) on cameron CO/Pt as the M1 correction baseline on Frontier (ROCm). Frontier-fit deviations: `num_layers=3`, `num_tensor_features=32`, BS=1/GCD × 8 DDP, eager, `bf16-mixed`. | [o2p1l6zr](https://wandb.ai/demiranda-gabriel/sparse-delta/runs/o2p1l6zr) (offline; sync from login node) |
| 1 | 2026-05-15 | [1-m0_equivariant_invariants](1-m0_equivariant_invariants/) | done-success | §12 extension: cross-channel power spectrum + (1,1,1) antisymmetric-triplet bispectrum on M0 pre-final-layer equivariant features. | _no run; inference-only_ |
| 0 | 2026-05-08 | [0-m0_complexity_probe](0-m0_complexity_probe/) | done-success | §12 distribution diagnostic on cameron CO/Pt: do M0 (baseline-B) hidden features yield bimodal `s_i`? | _no run; inference-only_ |

# Experiments

Each subdirectory is a single experiment (or a sweep). One commit per experiment, pinned **before** `sbatch` so the SHA reproduces the submitted state.

Layout per experiment (mirrors `_template/`):

```
<exp_name>/
├── README.md          # status, intent, hypothesis, success criteria, SHAs
├── config.yaml        # nequip / allegro training config
├── train.sh           # SLURM script
└── logs/              # SLURM stdout/stderr (created at run time)
```

For sweeps:

```
<exp_name>/
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

| Date | Experiment | Status | Intent | WandB |
|---|---|---|---|---|
| | | | | |

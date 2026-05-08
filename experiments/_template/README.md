# <experiment-name>

**Status:** planned
**Date:** YYYY-MM-DD
**Outer SHA:** `<git rev-parse HEAD before submission>`
**Submodule SHAs:** `nequip-private=…  allegro-private=…`
**WandB:** `<run url, paste at submission>`

## Intent

One paragraph: what is this run for? What knob is being tested?

## Hypothesis

One paragraph: what do you expect to see? What would falsify the hypothesis?

## Success criteria

Concrete, measurable. E.g.:
- Validation force RMSE < X meV/Å.
- Active fraction `mean(λ)` between 0.05 and 0.20 on bulk-Pt frames.
- NVE energy drift < Y meV/atom/ps over a 5 ps trajectory.

## Run

```bash
sbatch train.sh
```

## Outcome

Filled in after the run. Brief. Link to plots in `notes/` or `runs/<exp>/figures/` if any.

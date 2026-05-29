# Training procedure and loss function

**Author:** Gabriel de Miranda (with Claude Opus 4.7)
**Date:** 2026-05-29
**Scope:** how the sparse_delta composite models are actually trained,
what the loss function is mathematically, and where every piece lives
in the code. Covers experiments 4 (joint) and 5 (staged). Read
alongside [`sparse_local_correction.md`](sparse_local_correction.md)
§9, §16 (loss design) and [`m1_warmstart_design.md`](m1_warmstart_design.md).

---

## 0. Executive summary

We train with **nequip's Lightning harness** (`nequip-train`), not a
bespoke loop. The optimised quantity is a **single scalar**
`weighted_sum`, produced by a `MetricsManager` that combines:

1. per-atom **energy MSE**,
2. **force MSE**,
3. (composite gate runs only) a **sparsity penalty** `mean(λ)`.

The three terms' coefficients are **auto-normalised to sum to 1**, so
the configured `coeffs` are *relative* weights, not absolute multipliers.
Forces are **conservative** — computed by autograd-differentiating the
predicted total energy w.r.t. positions inside `ForceStressOutput`, so
the same backward pass that trains the energy also trains the forces.

```
loss = c_E · MSE(E_pred/N, E_DFT/N)  +  c_F · MSE(F_pred, F_DFT)  [ + c_S · mean(λ) ]
       └──────────── per-atom energy ┘    └──── forces ────┘         └── sparsity ──┘
with c_E + c_F [ + c_S ] = 1 after normalisation.
```

---

## 1. The training harness

We do **not** write a custom training loop. Every run is launched with:

```bash
srun .venv/bin/nequip-train -cd <config-dir> -cn <config-name> [+ckpt_path=...]
```

Entry point: `nequip.scripts.train.main`
(`software/nequip-private/nequip/scripts/train.py`). It instantiates a
`NequIPLightningModule` and a `lightning.Trainer` from the Hydra config,
then calls `trainer.fit(...)` (and `trainer.test(...)` for the `test`
phase of `run: [train, test]`).

The Lightning module is
`software/nequip-private/nequip/train/lightning.py:NequIPLightningModule`.
Key wiring in `__init__` (lightning.py:130):

```python
self.loss = instantiate(loss, type_names=type_names)
assert self.loss.do_weighted_sum  # loss MUST define coefficients
```

### 1.1 The training step

`NequIPLightningModule.training_step` (lightning.py:239):

```python
target = self.process_target(batch, ...)   # DFT energy/forces
output = self(batch)                        # forward → predicted E, F, λ, ...
loss_dict = self.loss(output, target, ...)  # MetricsManager.forward
loss = loss_dict[".../weighted_sum"] * self.world_size  # DDP-correction factor
return loss
```

Lightning then calls `loss.backward()` and the optimizer step under the
hood. The returned `weighted_sum` **is** the objective. The
`* self.world_size` term only matters under DDP (cancels the implicit
`1/n_rank` gradient averaging); single-GPU runs have `world_size=1`.

Validation/test use the same forward but feed `output` into
`val_metrics` / `test_metrics` (a separate `MetricsManager` reporting
MAE/RMSE, no gradient) — see `validation_step` (lightning.py:285).

---

## 2. The loss function in detail

### 2.1 `MetricsManager` — the engine

Both the **loss** and the **monitoring metrics** are instances of the
same class:
`software/nequip-private/nequip/train/metrics_manager.py:MetricsManager`
(line 36). It is a `torch.nn.ModuleDict` holding one `torchmetrics`
metric per named term.

**Coefficient normalisation** (the single most important behavioural
detail) — `MetricsManager.set_coeffs` (metrics_manager.py:372):

```python
tot = sum(v for v in coeff_dict.values() if v is not None)
metric_dict["coeff"] = coeff[name] / tot      # normalised to sum to 1
```

So `{total_energy: 1.0, forces: 1.0}` becomes effective weights
`{0.5, 0.5}`. Adding `sparsity: 1.0` makes it `{0.333, 0.333, 0.333}`.
Adding `sparsity: 0.1` makes it `{0.476, 0.476, 0.048}`. **This is why
the sparsity coefficient behaves the way it does** — see §5.

**Weighted sum** — computed in `MetricsManager.forward`
(metrics_manager.py:322):

```python
for metric_name, metric_params in self.metrics.items():
    metric = self[metric_name](preds_field, target_field)  # e.g. MSE
    if coeff is not None:
        weighted_sum = weighted_sum + metric * coeff
metric_dict[".../weighted_sum"] = weighted_sum
```

`weighted_sum` is differentiable (the underlying torchmetrics objects
propagate gradients through `sum`/`count` state — see §2.4).

### 2.2 `EnergyForceLoss` — the wrapper we configure

In every config we use the convenience factory
`nequip.train.EnergyForceLoss` (metrics_manager.py:410), which builds a
`MetricsManager` with exactly two default terms:

```python
metrics = [
    {                                                   # per-atom energy
        "name": "per_atom_energy_mse",
        "field": PerAtomModifier(TOTAL_ENERGY_KEY),     # E / N_atoms
        "coeff": coeffs[TOTAL_ENERGY_KEY],
        "metric": MeanSquaredError(),
    },
    {                                                   # forces
        "name": "forces_mse",
        "field": FORCE_KEY,
        "coeff": coeffs[FORCE_KEY],
        "metric": MeanSquaredError(),
    },
]
return MetricsManager(_with_extra_metrics(metrics, extra_metrics), ...)
```

Config snippet (identical across exp 4 + exp 5 stages 1/2; stage 3 adds
the sparsity term via `extra_metrics`):

```yaml
loss:
  _target_: nequip.train.EnergyForceLoss
  per_atom_energy: true
  coeffs:
    total_energy: 1.0
    forces: 1.0
```

- `per_atom_energy: true` → the energy term is `E / N_atoms`, via
  `PerAtomModifier` (`software/nequip-private/nequip/data/modifier.py:43`).
  It divides the graph's total energy by node count so large and small
  structures contribute comparably. Forces are **not** normalised
  (they're already per-atom by nature).
- `MeanSquaredError` is nequip's torchmetrics MSE
  (`nequip.train.MeanSquaredError`).

So term-by-term the loss is, per batch:

```
per_atom_energy_mse = mean_over_graphs[ (E_pred/N − E_DFT/N)² ]
forces_mse          = mean_over_atoms,xyz[ (F_pred − F_DFT)² ]
weighted_sum        = c̃_E · per_atom_energy_mse + c̃_F · forces_mse   (+ sparsity, §2.3)
```

with `c̃` the normalised coefficients.

### 2.3 The sparsity penalty (composite gate runs only)

Stage 3 of exp 5 (and the parent exp-4 joint run) add a third loss term
through `EnergyForceLoss`'s `extra_metrics`:

```yaml
loss:
  _target_: nequip.train.EnergyForceLoss
  per_atom_energy: true
  coeffs:
    total_energy: 1.0
    forces: 1.0
  extra_metrics:
    - name: sparsity_penalty
      coeff: ${sparsity_coeff}          # relative weight (NOT absolute α)
      metric:
        _target_: sparse_delta_core.train.GateMeanMetric
```

`GateMeanMetric` is **ours**:
`software/sparse-delta-core/sparse_delta_core/train/sparsity.py:40`.
It is a `field=None` custom metric (receives the full `(preds, target)`
dicts) that returns the running mean of the gate field `λ`:

```python
class GateMeanMetric(_MeanX):
    def update(self, preds, target):
        super().update(preds[self.field])   # field = GATE_KEY = "sparse_delta_lambda"
```

So the penalty is literally `mean_over_atoms(λ_i)`. Minimising it pushes
λ toward 0 (fewer atoms get the M1 correction → more compute savings).
The energy/force MSE pulls the other way (λ→1 wherever M1 helps). The
balance point is the "active set".

`λ` itself is written into `preds[GATE_KEY]` by the gate node
(`EdgeFeaturesToNodeLambda`,
`software/sparse-delta-core/sparse_delta_core/nn/gate_edge.py`), and the
composite energy is assembled as `E = E0 + λ·E1` in
`GatedPerAtomEnergyCompose`
(`software/sparse-delta-core/sparse_delta_core/nn/compose.py:101`):

```python
data[self.out_field] = e0 + lam * e1
```

### 2.4 Why the metrics are differentiable

`GateMeanMetric` subclasses `_MeanX`
(`software/nequip-private/nequip/data/stats.py:8`), which keeps `sum`
and `count` as tensor state and computes `sum/count`. Gradients flow
through that division back into λ, so the running mean works as a true
loss term, not just a logged number. `MeanSquaredError` is differentiable
the same way.

---

## 3. Forces are conservative (single backward)

The composite is wrapped at the top in **one**
`nequip.nn.ForceStressOutput`
(`software/nequip-private/nequip/nn/grad_output.py`). Inside its
`forward` (paraphrased; grad_output.py ~line 215):

```python
positions.requires_grad_(True)
data = self.func(data)                 # energy net forward → total_energy
E = data[TOTAL_ENERGY_KEY]
forces = -autograd.grad(E.sum(), positions, create_graph=True)[0]
data[FORCE_KEY] = forces
```

`create_graph=True` makes the force-MSE term differentiable w.r.t. model
parameters, i.e. training on forces is second-order through this graph.
This is **the** reason forces are conservative (`F = −∂E/∂r` exactly,
not a separately-predicted vector) and the reason exp 4's PT2 compile
NaN bisect mattered — see
[`pt2_compile_warmstart_nan_study.md`](pt2_compile_warmstart_nan_study.md).
The composite builders enforce exactly one `ForceStressOutput` at the
top and never nest one inside a submodel
(`software/sparse-delta-core/sparse_delta_core/model/warmstart_composite.py`,
end of `build_warmstart_composite`).

---

## 4. Optimizer, schedule, stopping, checkpointing

All set in the config's `training_module` / `trainer` blocks (see
`experiments/5-composite_staged/configs/stage*.yaml`).

| Component | Setting | Where |
|---|---|---|
| Optimizer | `torch.optim.Adam`, `lr` per stage (1e-3 stage 1, 5e-4 stages 2/3) | `training_module.optimizer` |
| Param filter | `sparse_delta_core.train.trainable_param_group` — only `requires_grad=True` params reach Adam (frozen blocks carry no optimizer state) | `software/sparse-delta-core/sparse_delta_core/train/optim.py` |
| LR schedule | `ReduceLROnPlateau` on `val0_epoch/weighted_sum`, patience 20, factor 0.8, min 1e-6 | `training_module.lr_scheduler` |
| Grad clip | `gradient_clip_val: 0.3` | `trainer` |
| Early stop | `EarlyStopping` on `val0_epoch/weighted_sum`, patience 40 | `trainer.callbacks` |
| Checkpoint | `ModelCheckpoint(monitor=val0_epoch/weighted_sum, save_last=true, filename=best)` | `trainer.callbacks` |
| Max epochs | 200 (medium-validation budget) | `target_max_epochs` |
| Precision | `model_dtype: float32`, eager compile + cuEquivariance modifier | `model` block |

`configure_optimizers` (lightning.py:174) wires the optimizer +
scheduler. The monitored quantity for both LR-plateau and early-stop is
the **validation** `weighted_sum` — i.e. the same loss formula evaluated
on the val split.

**`best.ckpt` selection** = lowest val `weighted_sum`. This is what
makes the gate-collapse issue self-limiting in principle: the checkpoint
saved is the best-on-val one, not the last. (In stage 3 it didn't help
because the gate collapsed early and stayed collapsed — §5.)

---

## 5. How the loss shaped the experimental results

This is the practical payoff of understanding the loss normalisation.

**Exp 5 stage 3 gate collapse** (see
`experiments/5-composite_staged/stage3_sweep_sparsity/README.md`): with
`coeffs={total_energy:1, forces:1}` + `sparsity_coeff` swept, the
normalised sparsity weight is `sc / (2 + sc)`:

| `sparsity_coeff` | normalised sparsity weight | outcome |
|---|---|---|
| 0.01 | 0.5 % | gate stays ~on, forces_mae 0.0798 |
| 0.1 | 4.8 % | **λ collapses to 0**, forces_mae 0.0956 |
| 0.5 | 20 % | collapsed (identical metrics) |
| 1.0 | 33 % | collapsed |

The penalty `mean(λ)` is **linear** in λ, so its gradient is a constant
(independent of λ). Once λ starts shrinking nothing damps it — it runs
to the λ=0 fixed point, where the composite degenerates to frozen M0 and
the test metrics become identical bit-for-bit across `sc ≥ 0.1`.

Fixes (documented, not yet run): quadratic `mean(λ²)` (gradient ∝ λ,
self-damping near 0), entropy-style penalty (bistable {0,1}), annealed
`sparsity_coeff`, or λ-init near 1. `GateMeanMetric` already supports a
`modifier` arg (`torch.square` → L² penalty) so a quadratic penalty is a
one-line config change:

```yaml
metric:
  _target_: sparse_delta_core.train.GateMeanMetric
  modifier:
    _target_: torch.square   # mean(λ²) instead of mean(λ)
```

---

## 6. Per-experiment loss configuration

| Experiment / stage | Energy term | Force term | Sparsity term | Trainable params |
|---|---|---|---|---|
| exp 4 (joint) | per_atom MSE, c=1 | MSE, c=1 | `GateMeanMetric`, c=0.1 | M0 + gate + M1 (all) |
| exp 5 stage 1 (M0) | per_atom MSE, c=1 | MSE, c=1 | — | M0 only |
| exp 5 stage 2 (M1, λ≡1) | per_atom MSE, c=1 | MSE, c=1 | — (gate is constant) | adapter + M1 (M0 frozen) |
| exp 5 stage 3 (gate) | per_atom MSE, c=1 | MSE, c=1 | `GateMeanMetric`, c=1.0 | gate only (M0+M1+adapter frozen) |

Stage 2 has **no** sparsity term because the gate is a constant
(`ConstantNodeLambda`, λ≡1) with no parameters — penalising it would be
meaningless.

---

## 7. Code reference index

| Concept | File:line |
|---|---|
| Train entry point | `software/nequip-private/nequip/scripts/train.py` → `main` |
| Lightning module | `software/nequip-private/nequip/train/lightning.py:NequIPLightningModule` |
| `training_step` (returns weighted_sum) | `lightning.py:239` |
| `configure_optimizers` | `lightning.py:174` |
| `MetricsManager` (loss engine) | `software/nequip-private/nequip/train/metrics_manager.py:36` |
| coeff normalisation | `metrics_manager.py:372` (`set_coeffs`) |
| weighted_sum assembly | `metrics_manager.py:322` (`forward`) |
| `EnergyForceLoss` factory | `metrics_manager.py:410` |
| `EnergyForceMetrics` (val/test) | `metrics_manager.py:485` |
| `PerAtomModifier` (E/N) | `software/nequip-private/nequip/data/modifier.py:43` |
| `_MeanX` (differentiable running mean) | `software/nequip-private/nequip/data/stats.py:8` |
| sparsity penalty `GateMeanMetric` | `software/sparse-delta-core/sparse_delta_core/train/sparsity.py:40` |
| gate node (writes λ) | `software/sparse-delta-core/sparse_delta_core/nn/gate_edge.py:EdgeFeaturesToNodeLambda` |
| constant gate (stage 2) | `software/sparse-delta-core/sparse_delta_core/nn/constant_lambda.py:ConstantNodeLambda` |
| compose `E = E0 + λ·E1` | `software/sparse-delta-core/sparse_delta_core/nn/compose.py:101` |
| conservative forces | `software/nequip-private/nequip/nn/grad_output.py:ForceStressOutput` |
| optimizer param filter | `software/sparse-delta-core/sparse_delta_core/train/optim.py:trainable_param_group` |
| stage weight transfer / freeze | `software/sparse-delta-core/sparse_delta_core/train/stage_transfer.py` |
| loss configs | `experiments/5-composite_staged/configs/stage{1,2,3}.yaml` (`training_module.loss`) |

#!/usr/bin/env python3
"""Extract per-atom gate ``λ`` values from the stage 3 model on a sample
of the validation set, plot histograms (overall + per atom type), and
dump summary statistics.

Stage 3 outputs ``test0_epoch/forces_mae = 0.096`` vs stage 2's 0.073.
The regression suggests the learned gate is suppressing M1 in regions
where it was contributing useful correction. This script answers
"where is the gate doing what" by looking at the actual λ distribution
on validation data.

Usage::

    python experiments/5-composite_staged/results/analyze_gate.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ase.io
import torch

# Ensure sparse_delta_core is loaded (registers GATE_KEY etc.) and
# nequip is fully initialised before we touch ModelFromCheckpoint.
import nequip  # noqa: F401
import sparse_delta_core  # noqa: F401

from nequip.data import AtomicDataDict, from_ase
from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
)
from nequip.model import ModelFromCheckpoint
from nequip.utils.global_state import set_global_state

# Required by @model_builder before constructing any model. Same call
# nequip-train makes at startup. Match exp 5's training config
# (allow_tf32=False, model_dtype=float32).
set_global_state(allow_tf32=False)

from sparse_delta_core._keys import GATE_KEY


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


PROJECT_ROOT = Path(
    "/n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta"
)
CKPT = (
    PROJECT_ROOT
    / "runs"
    / "5-composite_staged"
    / "5-composite_staged-stage3_gate"
    / "best.ckpt"
)
VAL_XYZ = PROJECT_ROOT / "data/optb88/cameron_plus_bulkpt/split_val.xyz"
TYPE_NAMES = ["C", "O", "Pt"]
R_MAX = 5.0
N_FRAMES = 50  # subsample for speed; whole val set is ~274 frames

OUTDIR = (
    PROJECT_ROOT / "experiments/5-composite_staged/results"
)


def main() -> int:
    # === load model ===
    logger.info(f"loading checkpoint: {CKPT}")
    wrapper = ModelFromCheckpoint(str(CKPT), compile_mode="eager")
    # NequIPLightningModule wraps the model in ModuleDict({'sole_model': ...}).
    model = (
        wrapper["sole_model"]
        if isinstance(wrapper, torch.nn.ModuleDict)
        else wrapper.sole_model
        if hasattr(wrapper, "sole_model")
        else wrapper
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"moving model to {device}")
    model = model.to(device)
    model.eval()

    # === load val frames ===
    logger.info(f"loading val xyz: {VAL_XYZ}")
    all_frames = ase.io.read(str(VAL_XYZ), index=":")
    logger.info(f"val frames available: {len(all_frames)}")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(all_frames), size=min(N_FRAMES, len(all_frames)), replace=False)
    frames = [all_frames[int(i)] for i in idx]

    # === transforms ===
    species_map = ChemicalSpeciesToAtomTypeMapper(model_type_names=TYPE_NAMES)
    neighborlist = NeighborListTransform(r_max=R_MAX)

    # === forward each frame, collect λ + atom types ===
    all_lambdas: List[np.ndarray] = []
    all_types: List[np.ndarray] = []

    for fi, atoms in enumerate(frames):
        # ASE → AtomicDataDict (exclude force / energy keys; we only
        # need positions / cell / pbc / numbers).
        data = from_ase(atoms, exclude_keys=["energy", "forces", "dipole"])
        data = species_map(data)
        data = neighborlist(data)
        # Required nequip-internal sentinel field — must be set on a
        # single-frame batch since the model expects it.
        if AtomicDataDict.BATCH_KEY not in data:
            n_atoms = data[AtomicDataDict.POSITIONS_KEY].shape[0]
            data[AtomicDataDict.BATCH_KEY] = torch.zeros(
                n_atoms, dtype=torch.long
            )
        if AtomicDataDict.NUM_NODES_KEY not in data:
            n_atoms = data[AtomicDataDict.POSITIONS_KEY].shape[0]
            data[AtomicDataDict.NUM_NODES_KEY] = torch.tensor(
                [n_atoms], dtype=torch.long
            )

        # Move every tensor in the data dict onto the model's device.
        data = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in data.items()
        }
        with torch.no_grad():
            out = model(data)

        lam = out[GATE_KEY].detach().cpu().numpy().reshape(-1)
        types = out[AtomicDataDict.ATOM_TYPE_KEY].detach().cpu().numpy().reshape(-1)
        all_lambdas.append(lam)
        all_types.append(types)

        if (fi + 1) % 10 == 0:
            logger.info(
                f"  frame {fi + 1}/{len(frames)} "
                f"(n_atoms={len(lam)}, λ_mean={lam.mean():.3f})"
            )

    lam_arr = np.concatenate(all_lambdas)
    type_arr = np.concatenate(all_types)

    logger.info(f"total atoms scored: {len(lam_arr)}")

    # === summary stats ===
    def stats(x):
        return {
            "n": int(len(x)),
            "mean": float(x.mean()) if len(x) else None,
            "std": float(x.std()) if len(x) else None,
            "median": float(np.median(x)) if len(x) else None,
            "frac_below_0.05": float((x < 0.05).mean()) if len(x) else None,
            "frac_above_0.95": float((x > 0.95).mean()) if len(x) else None,
            "min": float(x.min()) if len(x) else None,
            "max": float(x.max()) if len(x) else None,
        }

    summary: Dict[str, Dict] = {"overall": stats(lam_arr)}
    for ti, tname in enumerate(TYPE_NAMES):
        mask = type_arr == ti
        summary[tname] = stats(lam_arr[mask])

    logger.info("summary:")
    logger.info(json.dumps(summary, indent=2))

    # === save outputs ===
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTDIR / "gate_distribution.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"wrote {json_path}")

    # Histogram: overall + one per atom type stacked.
    fig, axes = plt.subplots(
        1, len(TYPE_NAMES) + 1, figsize=(4 * (len(TYPE_NAMES) + 1), 4),
        sharey=False,
    )
    bins = np.linspace(0.0, 1.0, 51)
    axes[0].hist(lam_arr, bins=bins, color="steelblue", edgecolor="black")
    axes[0].set_title(
        f"overall (n={len(lam_arr)})\n"
        f"mean={summary['overall']['mean']:.3f}, "
        f"<0.05={summary['overall']['frac_below_0.05']:.2f}, "
        f">0.95={summary['overall']['frac_above_0.95']:.2f}"
    )
    axes[0].set_xlabel("λ")
    axes[0].set_ylabel("count")

    for ti, tname in enumerate(TYPE_NAMES):
        ax = axes[ti + 1]
        mask = type_arr == ti
        sub = lam_arr[mask]
        if len(sub) == 0:
            ax.set_title(f"{tname} — no atoms")
            continue
        ax.hist(sub, bins=bins, color="tab:green", edgecolor="black")
        ax.set_title(
            f"{tname} (n={len(sub)})\n"
            f"mean={summary[tname]['mean']:.3f}, "
            f"<0.05={summary[tname]['frac_below_0.05']:.2f}, "
            f">0.95={summary[tname]['frac_above_0.95']:.2f}"
        )
        ax.set_xlabel("λ")

    fig.suptitle(
        "Stage-3 learned gate λ distribution on cameron_plus_bulkpt val "
        f"(n={N_FRAMES} frames)"
    )
    fig.tight_layout()
    fig_path = OUTDIR / "gate_distribution.png"
    fig.savefig(fig_path, dpi=130)
    logger.info(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

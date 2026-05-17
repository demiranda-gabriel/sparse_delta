"""§12 extension: equivariant invariants from M0 pre-final-layer features.

The original 0-m0_complexity_probe used the post-Allegro DenseNet ``edge_features``
tensor, which is all-scalar by Allegro's design (the final layer contracts to
0e only). This probe hooks the **second-to-last** Allegro layer's Contracter
output, which retains higher-l content (up to ``tensor_track_allowed_irreps``).
After sum-pooling per-edge tensors to receiver atoms, we get a per-node feature

    h_{i, c, l, m}    with c in [0, mul),  irreps from tps[-2].irreps_out

and compute two families of rotation-invariant descriptors:

1.  Cross-channel power spectrum for each l > 0 irrep block.
    P^{ab}_{i, l} = sum_m h_{i, a, l, m} h_{i, b, l, m}
    Invariant because D^l is orthogonal in e3nn's real basis. With C channels
    and l > 0, we keep the upper-triangle (including diagonal):
    C(C+1)/2 invariants per atom per block. Cost: O(N * C^2 * (2l+1)).

2.  (1,1,1) -> 0 bispectrum, antisymmetric channel triplets a < b < c, on l=1
    blocks. The (1,1,1) CG path is antisymmetric in (a, b) and the result is
    again contracted antisymmetrically with channel c, so only fully-distinct
    channel indices give nonzero values. Geometrically the invariant is
    det[h_a | h_b | h_c] (the triple product). Cost: O(N * C^3) but with the
    small (2l+1)=3 dimension. With C=32 we get C(C-1)(C-2)/6 = 4960 invariants.

The probe is num_layers-agnostic so long as num_layers >= 2; the hook target is
always ``allegro_mod.tps[num_layers - 2]``. For num_layers = 1 the last (and
only) TP outputs scalars only, so we fail loudly.

We also re-hook the Allegro module's ``edge_features`` output and recompute the
scalar baseline s_F2c_norm so this probe's output is directly comparable to
0-m0_complexity_probe without cross-experiment file reads.

Outputs (under ``experiments/1-m0_equivariant_invariants/results/``):

    val_invariants.npz   raw invariants per atom (large, fp32)
    val_with_si.xyz      extxyz with per-atom summary scalars + s_F2c_norm
    figures/             per-class histograms + violin plots
    summary.json         counts, paths, scalar summaries
    probe.log            stdout/stderr
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import ase.io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nequip.data import AtomicDataDict, from_ase
from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
)
from nequip.model.saved_models import ModelFromPackage


# === paths / config ===
# Frontier (lustre) layout: data and saved_models live under the project tree
# directly (no holylabs symlink on this cluster). Override PROJECT_ROOT via the
# SPARSE_DELTA_ROOT env var if running from a different mount point.
import os as _os

PROJECT_ROOT = Path(
    _os.environ.get(
        "SPARSE_DELTA_ROOT",
        "/lustre/orion/mat281/scratch/demirand/projects/sparse_delta",
    )
)
M0_PACKAGE = PROJECT_ROOT / "saved_models/packages/M0-baseline-B.nequip.zip"
DATA_ROOT = PROJECT_ROOT / "data/optb88/cameron"
VAL_XYZ = DATA_ROOT / "split_dataset_r5.0_val.xyz"
OUT_DIR = PROJECT_ROOT / "experiments/1-m0_equivariant_invariants/results"
FIG_DIR = OUT_DIR / "figures"

CHEMICAL_SYMBOLS = ["C", "O", "Pt"]
N_TYPES = len(CHEMICAL_SYMBOLS)
R_MAX = 7.0  # baseline-B

# baseline-B reference: num_scalar_features=32, num_layers=2 -> F2 dim = 96.
# We re-read this from the module at runtime instead of hard-coding F2_DIM.
EXCLUDE_KEYS = ["dipole", "momenta", "magmoms", "magmom", "free_energy", "model_index"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging():
    LOG_DIR = OUT_DIR
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "probe.log"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="w")],
    )
    return logging.getLogger("probe")


def composition_class(atoms):
    syms = set(atoms.get_chemical_symbols())
    has_pt = "Pt" in syms
    has_c = "C" in syms
    has_o = "O" in syms
    if has_pt and (has_c or has_o):
        return "mixed"
    if has_pt:
        return "Pt-only"
    if has_c or has_o:
        return "CO-only"
    return "other"


def find_allegro_module(model):
    for _, sub in model.named_modules():
        if type(sub).__name__ == "Allegro_Module":
            return sub
    raise RuntimeError("Could not locate Allegro_Module in the loaded model")


def make_pipeline(log):
    """Load model and register two forward hooks:
      - on tps[num_layers - 2]: pre-final-layer per-edge tensor (equivariant)
      - on Allegro_Module: post-Allegro edge_features (scalar, for s_F2c baseline)
    """
    wrapper = ModelFromPackage(str(M0_PACKAGE), compile_mode="eager").to(DEVICE).eval()
    model = wrapper.sole_model
    allegro_mod = find_allegro_module(model)

    num_layers = int(allegro_mod.num_layers)
    if num_layers < 2:
        raise RuntimeError(
            f"this probe requires num_layers >= 2 to capture pre-final-layer "
            f"equivariant features; loaded model has num_layers={num_layers}. "
            f"For num_layers=1, the only TP outputs scalars only and there is no "
            f"pre-final layer to hook."
        )

    pre_final_tp = allegro_mod.tps[num_layers - 2]
    mul = int(pre_final_tp.mul)
    irreps_out = pre_final_tp.irreps_out

    # parse irreps into slot info; the Contracter strides each output irrep
    # contiguously in the trailing dim, in declared order
    slot_info = []
    offset = 0
    for _mul1, ir in irreps_out:
        psym = "e" if int(ir.p) == 1 else ("o" if int(ir.p) == -1 else "x")
        slot_info.append(
            {
                "l": int(ir.l),
                "parity": int(ir.p),  # +1 (even) or -1 (odd)
                "psym": psym,
                "label": f"l{int(ir.l)}{psym}",
                "start": offset,
                "end": offset + ir.dim,
                "dim": int(ir.dim),
            }
        )
        offset += ir.dim
    base_dim_out = offset

    log.info(f"Allegro_Module: num_layers={num_layers}, num_tensor_features={mul}")
    log.info(f"hooking pre-final-layer TP: tps[{num_layers - 2}]")
    log.info(
        f"pre-final irreps_out: {irreps_out} (base_dim={base_dim_out}, mul={mul})"
    )
    for s in slot_info:
        log.info(
            f"  irrep slot: {s['label']}  slice=[{s['start']}:{s['end']}]  dim={s['dim']}"
        )

    f2_dim = int(allegro_mod.irreps_out[AtomicDataDict.EDGE_FEATURES_KEY].num_irreps)
    log.info(f"post-Allegro edge_features dim (F2): {f2_dim}")
    num_scalar = f2_dim // (num_layers + 1)
    f2c_slice = (num_scalar * num_layers, f2_dim)  # last 32 scalars = F2c (layer-2 latents)
    log.info(f"F2c slice for s_F2c_norm baseline: [{f2c_slice[0]}:{f2c_slice[1]}]")

    buffer = {"pre_final": None, "edge_features": None}

    def hook_tp(_mod, _inp, out):
        # out shape: [E, mul, base_dim_out]
        buffer["pre_final"] = out.detach().clone()

    def hook_allegro(_mod, _inp, out_dict):
        buffer["edge_features"] = (
            out_dict[AtomicDataDict.EDGE_FEATURES_KEY].detach().clone()
        )

    pre_final_tp.register_forward_hook(hook_tp)
    allegro_mod.register_forward_hook(hook_allegro)

    type_mapper = ChemicalSpeciesToAtomTypeMapper(model_type_names=CHEMICAL_SYMBOLS)
    nl_transform = NeighborListTransform(r_max=R_MAX)

    return {
        "model": model,
        "buffer": buffer,
        "mul": mul,
        "slot_info": slot_info,
        "base_dim_out": base_dim_out,
        "num_layers": num_layers,
        "f2_dim": f2_dim,
        "f2c_slice": f2c_slice,
        "type_mapper": type_mapper,
        "nl_transform": nl_transform,
    }


def atoms_to_data(atoms, type_mapper, nl_transform):
    data = from_ase(atoms, exclude_keys=EXCLUDE_KEYS)
    data = type_mapper(data)
    data = nl_transform(data)
    return AtomicDataDict.to_(data, DEVICE)


def sum_pool_to_receiver_3d(h_edge, edge_index, n_atoms):
    """Sum-pool per-edge [E, mul, D] to per-receiver-node [N, mul, D].

    receiver = edge_index[0] (allegro/nn/edgewise.py convention).
    """
    E, mul, D = h_edge.shape
    z = torch.zeros(n_atoms, mul, D, device=h_edge.device, dtype=h_edge.dtype)
    z.index_add_(0, edge_index[0], h_edge)
    return z


def sum_pool_to_receiver_2d(h_edge, edge_index, n_atoms):
    """Same as above but for [E, D] -> [N, D]."""
    z = torch.zeros(n_atoms, h_edge.shape[1], device=h_edge.device, dtype=h_edge.dtype)
    z.index_add_(0, edge_index[0], h_edge)
    return z


def cross_channel_power_spectrum(h_l):
    """Cross-channel power spectrum for one irrep block.

    h_l : [N, mul, 2l+1]  (the components of one l > 0 irrep block per atom)

    Returns the upper-triangle (including diagonal) of the Gram matrix
    P^{ab}_i = sum_m h_l[i, a, m] h_l[i, b, m]:

        out shape: [N, mul * (mul + 1) // 2]

    Invariant under SO(3) because the real Wigner-D matrix is orthogonal.
    """
    # Gram[i, a, b] = sum_m h_l[i, a, m] * h_l[i, b, m]
    gram = torch.einsum("nam,nbm->nab", h_l, h_l)  # [N, mul, mul]
    mul = h_l.shape[1]
    iu = torch.triu_indices(mul, mul, device=h_l.device)
    return gram[:, iu[0], iu[1]]


def l1_bispectrum_antisym(h_1, triplets, chunk_atoms=2048):
    """(1,1,1) -> 0 antisymmetric channel-triplet bispectrum on an l=1 block.

    h_1      : [N, mul, 3]    one l=1 irrep block per atom
    triplets : [T, 3] long    (a, b, c) with a < b < c, T = mul*(mul-1)*(mul-2)//6

    Returns [N, T], where each entry is det[h_a | h_b | h_c] = (h_a x h_b) . h_c.
    Antisymmetric under any pair swap, so a < b < c covers all independent values.

    Chunks over atoms to bound peak memory (forms a [chunk, T, 3] intermediate).
    """
    N = h_1.shape[0]
    T = triplets.shape[0]
    out = torch.empty(N, T, device=h_1.device, dtype=h_1.dtype)
    for start in range(0, N, chunk_atoms):
        end = min(start + chunk_atoms, N)
        h_chunk = h_1[start:end]  # [n, mul, 3]
        h_a = h_chunk[:, triplets[:, 0]]  # [n, T, 3]
        h_b = h_chunk[:, triplets[:, 1]]  # [n, T, 3]
        h_c = h_chunk[:, triplets[:, 2]]  # [n, T, 3]
        cross_ab = torch.cross(h_a, h_b, dim=-1)  # [n, T, 3]
        out[start:end] = (cross_ab * h_c).sum(dim=-1)
    return out


def build_triplets(mul, device):
    """Build (a, b, c) index triplets with a < b < c, lex order."""
    idx = []
    for a in range(mul):
        for b in range(a + 1, mul):
            for c in range(b + 1, mul):
                idx.append((a, b, c))
    if not idx:
        return torch.empty(0, 3, dtype=torch.long, device=device)
    return torch.tensor(idx, dtype=torch.long, device=device)


def score_pass(pipeline, frames, triplets, log):
    """Forward each val frame, capture pre-final TP output + edge_features, and
    compute per-atom invariants. Returns lists of ASE Atoms (with summary
    scalars stamped as arrays) plus stacked raw invariant arrays.
    """
    model = pipeline["model"]
    buffer = pipeline["buffer"]
    mul = pipeline["mul"]
    slot_info = pipeline["slot_info"]
    base_dim_out = pipeline["base_dim_out"]
    f2c_slice = pipeline["f2c_slice"]
    type_mapper = pipeline["type_mapper"]
    nl_transform = pipeline["nl_transform"]

    out_atoms = []
    # we collect per-atom rows for histograms
    power_blocks = {s["label"]: [] for s in slot_info if s["l"] > 0}
    bispec_blocks = {s["label"]: [] for s in slot_info if s["l"] == 1}
    per_atom_meta = {"frame_idx": [], "atom_type": [], "class": []}

    t0 = time.time()
    with torch.no_grad():
        for fi, atoms in enumerate(frames):
            data = atoms_to_data(atoms, type_mapper, nl_transform)
            with torch.enable_grad():
                _out = model(data)  # ForceStressOutput needs grad enabled internally

            # === pre-final-layer per-edge tensor [E, mul, base_dim_out] ===
            h_edge = buffer["pre_final"]
            assert h_edge is not None, "pre_final hook did not fire"
            assert h_edge.shape[1:] == (mul, base_dim_out), (
                f"unexpected pre-final shape {tuple(h_edge.shape)}, "
                f"expected (E, {mul}, {base_dim_out})"
            )

            edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
            n_atoms = int(data[AtomicDataDict.POSITIONS_KEY].shape[0])
            atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].reshape(-1)

            # sum-pool to receiver atoms: [N, mul, base_dim_out]
            z_node = sum_pool_to_receiver_3d(h_edge, edge_index, n_atoms)

            # === invariants per irrep block ===
            frame_power = {}  # key -> [N, n_pairs]
            frame_bispec = {}  # key -> [N, T]
            for s in slot_info:
                if s["l"] == 0:
                    continue  # scalars are already invariant; we use F2c instead
                key = s["label"]
                h_block = z_node[:, :, s["start"] : s["end"]]  # [N, mul, 2l+1]
                frame_power[key] = cross_channel_power_spectrum(h_block)
                if s["l"] == 1:
                    frame_bispec[key] = l1_bispectrum_antisym(h_block, triplets)

            # === baseline s_F2c_norm from post-Allegro edge_features ===
            h_ij = buffer["edge_features"]  # [E, F2_dim]
            z_F2 = sum_pool_to_receiver_2d(h_ij, edge_index, n_atoms)
            s0, s1 = f2c_slice
            s_F2c_norm = torch.linalg.norm(z_F2[:, s0:s1], dim=-1).cpu().numpy()

            # === per-atom summary scalars (one per irrep block per family) ===
            new_atoms = atoms.copy()
            new_atoms.arrays["s_F2c_norm"] = s_F2c_norm.astype(np.float32)
            new_atoms.arrays["atom_type_index"] = (
                atom_types.cpu().numpy().astype(np.int32)
            )

            total_power_sq = torch.zeros(n_atoms, device=DEVICE)
            for key, P in frame_power.items():
                norm = torch.linalg.norm(P, dim=-1).cpu().numpy()
                new_atoms.arrays[f"s_power_{key}_norm"] = norm.astype(np.float32)
                total_power_sq = total_power_sq + (P * P).sum(dim=-1)
            new_atoms.arrays["s_power_total_norm"] = (
                total_power_sq.sqrt().cpu().numpy().astype(np.float32)
            )

            total_bispec_sq = torch.zeros(n_atoms, device=DEVICE)
            for key, B in frame_bispec.items():
                norm = torch.linalg.norm(B, dim=-1).cpu().numpy()
                new_atoms.arrays[f"s_bispec_{key}_norm"] = norm.astype(np.float32)
                total_bispec_sq = total_bispec_sq + (B * B).sum(dim=-1)
            new_atoms.arrays["s_bispec_total_norm"] = (
                total_bispec_sq.sqrt().cpu().numpy().astype(np.float32)
            )

            klass = composition_class(atoms)
            new_atoms.info["composition_class"] = klass
            new_atoms.info["frame_index"] = fi
            out_atoms.append(new_atoms)

            # accumulate raw invariants
            for key, P in frame_power.items():
                power_blocks[key].append(P.cpu().numpy().astype(np.float32))
            for key, B in frame_bispec.items():
                bispec_blocks[key].append(B.cpu().numpy().astype(np.float32))
            atype_np = atom_types.cpu().numpy()
            per_atom_meta["frame_idx"].extend([fi] * n_atoms)
            per_atom_meta["atom_type"].extend(atype_np.tolist())
            per_atom_meta["class"].extend([klass] * n_atoms)

            if (fi + 1) % 25 == 0:
                log.info(
                    f"score pass: frame {fi+1}/{len(frames)} "
                    f"({(fi+1)/(time.time()-t0):.1f} fps)"
                )

    log.info(f"score pass done in {time.time()-t0:.1f}s")

    # stack across frames
    power_stacked = {
        k: np.concatenate(v, axis=0) for k, v in power_blocks.items() if v
    }
    bispec_stacked = {
        k: np.concatenate(v, axis=0) for k, v in bispec_blocks.items() if v
    }
    meta = {
        "frame_idx": np.array(per_atom_meta["frame_idx"], dtype=np.int32),
        "atom_type": np.array(per_atom_meta["atom_type"], dtype=np.int32),
        "class": np.array(per_atom_meta["class"]),
    }
    return out_atoms, power_stacked, bispec_stacked, meta


def plot_summary_histograms(out_atoms, fig_dir, log):
    """Per-class histograms for each summary scalar stamped onto out_atoms."""
    # collect summary keys from the first frame
    keys = [k for k in out_atoms[0].arrays.keys() if k.startswith("s_")]
    rows = []
    for atoms in out_atoms:
        klass = atoms.info["composition_class"]
        atype = atoms.arrays["atom_type_index"]
        for i in range(len(atoms)):
            row = {"class": klass, "atom_type": int(atype[i])}
            for k in keys:
                row[k] = float(atoms.arrays[k][i])
            rows.append(row)

    classes = sorted(set(r["class"] for r in rows))
    atom_types = ["C", "O", "Pt"]
    for key in keys:
        vals_all = np.array([r[key] for r in rows], dtype=np.float64)
        if not np.isfinite(vals_all).any():
            continue
        vmin = np.nanpercentile(vals_all, 0.5)
        vmax = np.nanpercentile(vals_all, 99.5)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            continue
        bins = np.linspace(vmin, vmax, 80)

        # by class
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in classes:
            v = np.array([r[key] for r in rows if r["class"] == c])
            if v.size == 0:
                continue
            ax.hist(v, bins=bins, alpha=0.5, label=f"{c} (n={v.size})", density=True)
        ax.set_xlabel(key)
        ax.set_ylabel("density")
        ax.set_title(f"{key} by composition class")
        ax.legend()
        fig.tight_layout()
        path = fig_dir / f"hist_{key}_by_class.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        log.info(f"wrote {path}")

        # by atom type
        fig, ax = plt.subplots(figsize=(6, 4))
        for ti, tn in enumerate(atom_types):
            v = np.array([r[key] for r in rows if r["atom_type"] == ti])
            if v.size == 0:
                continue
            ax.hist(v, bins=bins, alpha=0.5, label=f"{tn} (n={v.size})", density=True)
        ax.set_xlabel(key)
        ax.set_ylabel("density")
        ax.set_title(f"{key} by atom type")
        ax.legend()
        fig.tight_layout()
        path = fig_dir / f"hist_{key}_by_atomtype.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        log.info(f"wrote {path}")


def main():
    log = setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"device: {DEVICE}")
    log.info(f"package: {M0_PACKAGE}")
    log.info(f"val:     {VAL_XYZ}")

    log.info("loading model package...")
    pipeline = make_pipeline(log)

    triplets = build_triplets(pipeline["mul"], DEVICE)
    log.info(
        f"l=1 antisymmetric triplets: T = C(C-1)(C-2)/6 = {triplets.shape[0]} "
        f"(C = {pipeline['mul']})"
    )

    log.info("loading val frames...")
    val_atoms_all = ase.io.read(str(VAL_XYZ), index=":")
    log.info(f"val frames: {len(val_atoms_all)}")
    max_frames_env = _os.environ.get("MAX_FRAMES")
    if max_frames_env is not None:
        n_keep = int(max_frames_env)
        log.info(f"MAX_FRAMES={n_keep} set — truncating val to first {n_keep} frames")
        val_atoms_all = val_atoms_all[:n_keep]

    # sanity probe
    log.info("sanity probe on first val frame...")
    data = atoms_to_data(
        val_atoms_all[0], pipeline["type_mapper"], pipeline["nl_transform"]
    )
    with torch.enable_grad():
        pipeline["model"](data)
    pre_final = pipeline["buffer"]["pre_final"]
    edge_features = pipeline["buffer"]["edge_features"]
    log.info(
        f"sanity: pre_final shape = {tuple(pre_final.shape)}, "
        f"edge_features shape = {tuple(edge_features.shape)}"
    )

    # score pass over val
    log.info("running score pass over val...")
    out_atoms, power_stacked, bispec_stacked, meta = score_pass(
        pipeline, val_atoms_all, triplets, log
    )

    # save raw invariants
    npz_path = OUT_DIR / "val_invariants.npz"
    log.info(f"writing raw invariants to {npz_path}")
    save_dict = {f"power_{k}": v for k, v in power_stacked.items()}
    save_dict.update({f"bispec_{k}": v for k, v in bispec_stacked.items()})
    save_dict["frame_idx"] = meta["frame_idx"]
    save_dict["atom_type"] = meta["atom_type"]
    save_dict["composition_class"] = meta["class"]
    np.savez_compressed(npz_path, **save_dict)
    for k, v in power_stacked.items():
        log.info(f"  power_{k}: shape {v.shape}")
    for k, v in bispec_stacked.items():
        log.info(f"  bispec_{k}: shape {v.shape}")

    # extxyz with summary arrays for OVITO
    xyz_path = OUT_DIR / "val_with_si.xyz"
    log.info(f"writing extxyz to {xyz_path}")
    ase.io.write(str(xyz_path), out_atoms, format="extxyz")

    # plots
    log.info("plotting per-class histograms of summary scalars...")
    plot_summary_histograms(out_atoms, FIG_DIR, log)

    # summary json
    per_class_counts = {}
    for c in meta["class"]:
        per_class_counts[c] = per_class_counts.get(c, 0) + 1
    per_type_counts = {
        CHEMICAL_SYMBOLS[t]: int((meta["atom_type"] == t).sum())
        for t in range(N_TYPES)
    }
    summary = {
        "n_val_frames": len(val_atoms_all),
        "n_val_atoms": int(meta["atom_type"].shape[0]),
        "num_layers": pipeline["num_layers"],
        "num_tensor_features": pipeline["mul"],
        "f2_dim": pipeline["f2_dim"],
        "pre_final_slot_info": pipeline["slot_info"],
        "n_l1_triplets": int(triplets.shape[0]),
        "per_class_atom_counts": per_class_counts,
        "per_type_atom_counts": per_type_counts,
        "power_blocks": {k: list(v.shape) for k, v in power_stacked.items()},
        "bispec_blocks": {k: list(v.shape) for k, v in bispec_stacked.items()},
        "outputs": {
            "extxyz": str(xyz_path),
            "invariants_npz": str(npz_path),
            "figures": str(FIG_DIR),
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"wrote {OUT_DIR/'summary.json'}")
    log.info("DONE")


if __name__ == "__main__":
    main()

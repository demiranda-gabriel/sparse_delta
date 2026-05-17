"""§12 distribution diagnostic for sparse_delta.

Reads the packaged baseline-B Allegro M0, runs forward passes over the
cameron CO/Pt train and val splits, captures the post-Allegro per-edge
DenseNet feature h_ij in R^96 via a forward hook, sum-pools to receiver
atoms to get z_i in R^96, and computes per-atom complexity scores using
six methods:

    F2_norm, F2a_norm, F2b_norm, F2c_norm, F2_maha, F2c_maha

Plus per-edge norms by directed (recv_type, send_type) pair (M7).

Outputs:
    experiments/0-m0_complexity_probe/results/stats_train.npz
    experiments/0-m0_complexity_probe/results/val_with_si.xyz
    experiments/0-m0_complexity_probe/results/figures/*.png

See experiments/0-m0_complexity_probe/NOTES.md for the math.
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
PROJECT_ROOT = Path("/n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta")
M0_PACKAGE = PROJECT_ROOT / "saved_models/packages/M0-baseline-B.nequip.zip"
DATA_ROOT = Path(
    "/n/holylabs/LABS/kozinsky_lab/Users/demiranda/projects/multifidelity/data/optb88/cameron"
)
TRAIN_XYZ = DATA_ROOT / "split_dataset_r5.0_train.xyz"
VAL_XYZ = DATA_ROOT / "split_dataset_r5.0_val.xyz"
OUT_DIR = PROJECT_ROOT / "experiments/0-m0_complexity_probe/results"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR  # log goes alongside results, no separate logs/ subdir

CHEMICAL_SYMBOLS = ["C", "O", "Pt"]
N_TYPES = len(CHEMICAL_SYMBOLS)
R_MAX = 7.0  # baseline-B
NUM_SCALAR = 32  # baseline-B num_scalar_features
NUM_LAYERS = 2  # baseline-B num_layers
F2_DIM = NUM_SCALAR * (NUM_LAYERS + 1)  # 96

EXCLUDE_KEYS = ["dipole", "momenta", "magmoms", "magmom", "free_energy", "model_index"]
RIDGE_EPS = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging():
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


def make_pipeline():
    """Return (model, hook_buffer, transforms).

    NOTE: ``ModelFromPackage`` returns a ``ModuleDict({'sole_model': GraphModel})``
    wrapper, not the GraphModel itself. We unwrap here so callers get something
    callable as ``model(data)``.
    """
    wrapper = ModelFromPackage(str(M0_PACKAGE), compile_mode="eager").to(DEVICE).eval()
    model = wrapper.sole_model
    allegro_mod = find_allegro_module(model)

    buffer = {"edge_features": None}

    def hook(_module, _inputs, out_dict):
        # out_dict is the AtomicDataDict; edge_features is set by Allegro_Module.forward
        buffer["edge_features"] = (
            out_dict[AtomicDataDict.EDGE_FEATURES_KEY].detach().clone()
        )

    allegro_mod.register_forward_hook(hook)

    type_mapper = ChemicalSpeciesToAtomTypeMapper(model_type_names=CHEMICAL_SYMBOLS)
    nl_transform = NeighborListTransform(r_max=R_MAX)
    return model, buffer, type_mapper, nl_transform


def atoms_to_data(atoms, type_mapper, nl_transform):
    data = from_ase(atoms, exclude_keys=EXCLUDE_KEYS)
    data = type_mapper(data)
    data = nl_transform(data)
    return AtomicDataDict.to_(data, DEVICE)


def sum_pool_to_receiver(h_ij, edge_index, n_atoms):
    """z_i = sum_{j: (j -> i)} h_ij. edge_index[0] is receiver (per allegro/nn/edgewise.py)."""
    z = torch.zeros(n_atoms, h_ij.shape[1], device=h_ij.device, dtype=h_ij.dtype)
    z.index_add_(0, edge_index[0], h_ij)
    return z


def stats_pass(model, buffer, type_mapper, nl_transform, frames, log):
    """Streaming per-type sample mean / cov of z_i over the train set."""
    S1 = torch.zeros(N_TYPES, F2_DIM, dtype=torch.float64)
    S2 = torch.zeros(N_TYPES, F2_DIM, F2_DIM, dtype=torch.float64)
    N = torch.zeros(N_TYPES, dtype=torch.int64)

    t0 = time.time()
    with torch.no_grad():
        for fi, atoms in enumerate(frames):
            data = atoms_to_data(atoms, type_mapper, nl_transform)
            with torch.enable_grad():
                _ = model(data)  # ForceStressOutput needs grad enabled internally
            h_ij = buffer["edge_features"]  # (E, 96)
            assert h_ij.shape[1] == F2_DIM, f"unexpected F2 dim {h_ij.shape[1]}"
            edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
            n_atoms = data[AtomicDataDict.POSITIONS_KEY].shape[0]
            z = sum_pool_to_receiver(h_ij, edge_index, n_atoms)
            z = z.cpu().to(torch.float64)
            atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].cpu().reshape(-1)
            for t in range(N_TYPES):
                mask = atom_types == t
                if not mask.any():
                    continue
                z_t = z[mask]
                S1[t] += z_t.sum(dim=0)
                S2[t] += z_t.T @ z_t
                N[t] += z_t.shape[0]
            if (fi + 1) % 50 == 0:
                log.info(
                    f"stats pass: frame {fi+1}/{len(frames)} "
                    f"({(fi+1)/(time.time()-t0):.1f} fps)"
                )
    log.info(f"stats pass done in {time.time()-t0:.1f}s. per-type N: {N.tolist()}")

    mu = S1 / N.clamp(min=1).unsqueeze(-1).double()
    cov = S2 / N.clamp(min=1).view(-1, 1, 1).double() - mu.unsqueeze(-1) * mu.unsqueeze(
        -2
    )
    # ridge using mean diagonal as scale
    diag_mean = cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1)  # (N_TYPES,)
    eye = torch.eye(F2_DIM, dtype=torch.float64).unsqueeze(0)
    cov_reg = cov + RIDGE_EPS * diag_mean.view(-1, 1, 1) * eye
    cov_inv = torch.linalg.inv(cov_reg)

    return {
        "mu": mu.numpy(),
        "cov": cov.numpy(),
        "cov_reg": cov_reg.numpy(),
        "cov_inv": cov_inv.numpy(),
        "N_per_type": N.numpy(),
        "type_names": np.array(CHEMICAL_SYMBOLS),
    }


def slice_stats(mu, cov_reg, sl):
    """Sub-block stats for a slice of dim coordinates."""
    mu_s = mu[:, sl]
    cov_s = cov_reg[:, sl, sl]
    cov_inv_s = np.stack([np.linalg.inv(cov_s[t]) for t in range(cov_s.shape[0])])
    return mu_s, cov_inv_s


def maha_score(z_np, atom_types_np, mu_t, cov_inv_t):
    """Per-type Mahalanobis. z: (N, d) np float64. mu_t: (T, d). cov_inv_t: (T, d, d)."""
    out = np.zeros(z_np.shape[0], dtype=np.float64)
    for t in range(mu_t.shape[0]):
        mask = atom_types_np == t
        if not mask.any():
            continue
        d = z_np[mask] - mu_t[t]  # (n_t, d)
        # quadratic form: diag(d @ A @ d.T) = sum(d * (d @ A.T), axis=1)
        out[mask] = np.einsum("ij,jk,ik->i", d, cov_inv_t[t], d)
    return out


def score_pass(model, buffer, type_mapper, nl_transform, frames, stats, log):
    """Run val frames; compute all six per-atom scores + per-edge norms; return ASE atoms list + edge records."""
    SL = {
        "F2": np.s_[0:F2_DIM],
        "F2a": np.s_[0:NUM_SCALAR],
        "F2b": np.s_[NUM_SCALAR : 2 * NUM_SCALAR],
        "F2c": np.s_[2 * NUM_SCALAR : 3 * NUM_SCALAR],
    }

    # build per-slice mu / cov_inv only for the two we Mahalanobize
    mu_full = stats["mu"]  # (T, 96) float64
    cov_reg_full = stats["cov_reg"]  # (T, 96, 96) float64
    mu_F2 = mu_full
    cov_inv_F2 = stats["cov_inv"]  # already inverted on full
    # F2c sub-block: take rows/cols [64:96] of cov_reg_full and invert
    sl_c = SL["F2c"]
    cov_F2c = cov_reg_full[:, sl_c, sl_c]
    cov_inv_F2c = np.stack(
        [np.linalg.inv(cov_F2c[t]) for t in range(cov_F2c.shape[0])]
    )
    mu_F2c = mu_full[:, sl_c]

    out_atoms = []
    edge_recv_type = []
    edge_send_type = []
    edge_norms = []

    t0 = time.time()
    with torch.no_grad():
        for fi, atoms in enumerate(frames):
            data = atoms_to_data(atoms, type_mapper, nl_transform)
            with torch.enable_grad():
                out = model(data)
            h_ij = buffer["edge_features"]
            edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
            n_atoms = data[AtomicDataDict.POSITIONS_KEY].shape[0]
            z = sum_pool_to_receiver(h_ij, edge_index, n_atoms)
            z_np = z.cpu().to(torch.float64).numpy()
            atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].cpu().reshape(-1).numpy()

            # M0 per-atom energy (post per-type scale-shift)
            e_per_atom = (
                out[AtomicDataDict.PER_ATOM_ENERGY_KEY]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            # six per-atom scores
            s_F2_norm = np.linalg.norm(z_np[:, SL["F2"]], axis=-1)
            s_F2a_norm = np.linalg.norm(z_np[:, SL["F2a"]], axis=-1)
            s_F2b_norm = np.linalg.norm(z_np[:, SL["F2b"]], axis=-1)
            s_F2c_norm = np.linalg.norm(z_np[:, SL["F2c"]], axis=-1)
            s_F2_maha = maha_score(z_np[:, SL["F2"]], atom_types, mu_F2, cov_inv_F2)
            s_F2c_maha = maha_score(z_np[:, SL["F2c"]], atom_types, mu_F2c, cov_inv_F2c)

            new_atoms = atoms.copy()
            new_atoms.arrays["s_F2_norm"] = s_F2_norm.astype(np.float32)
            new_atoms.arrays["s_F2a_norm"] = s_F2a_norm.astype(np.float32)
            new_atoms.arrays["s_F2b_norm"] = s_F2b_norm.astype(np.float32)
            new_atoms.arrays["s_F2c_norm"] = s_F2c_norm.astype(np.float32)
            new_atoms.arrays["s_F2_maha"] = s_F2_maha.astype(np.float32)
            new_atoms.arrays["s_F2c_maha"] = s_F2c_maha.astype(np.float32)
            new_atoms.arrays["m0_per_atom_energy"] = e_per_atom.astype(np.float32)
            new_atoms.arrays["atom_type_index"] = atom_types.astype(np.int32)
            klass = composition_class(atoms)
            new_atoms.info["composition_class"] = klass
            new_atoms.info["frame_index"] = fi
            out_atoms.append(new_atoms)

            # M7: per-edge norm by directed (recv, send) pair
            ei = edge_index.cpu().numpy()
            recv_t = atom_types[ei[0]]
            send_t = atom_types[ei[1]]
            norms = h_ij.norm(dim=-1).cpu().numpy()
            edge_recv_type.append(recv_t)
            edge_send_type.append(send_t)
            edge_norms.append(norms)

            if (fi + 1) % 25 == 0:
                log.info(
                    f"score pass: frame {fi+1}/{len(frames)} "
                    f"({(fi+1)/(time.time()-t0):.1f} fps)"
                )
    log.info(f"score pass done in {time.time()-t0:.1f}s")

    return (
        out_atoms,
        np.concatenate(edge_recv_type),
        np.concatenate(edge_send_type),
        np.concatenate(edge_norms),
    )


def collect_per_atom_arrays(out_atoms, methods):
    """Stack per-atom values across all val frames, plus parallel arrays for class + atom_type."""
    rows = []
    for atoms in out_atoms:
        klass = atoms.info["composition_class"]
        types = atoms.arrays["atom_type_index"]
        for i in range(len(atoms)):
            row = {"class": klass, "atom_type": int(types[i])}
            for m in methods:
                row[m] = float(atoms.arrays[m][i])
            rows.append(row)
    return rows


def plot_histograms(rows, methods, fig_dir, log):
    classes = sorted(set(r["class"] for r in rows))
    types = ["C", "O", "Pt"]
    for m in methods:
        # by class
        fig, ax = plt.subplots(figsize=(6, 4))
        vals_all = np.array([r[m] for r in rows], dtype=np.float64)
        vmin = np.nanpercentile(vals_all, 0.5)
        vmax = np.nanpercentile(vals_all, 99.5)
        bins = np.linspace(vmin, vmax, 80)
        for k in classes:
            v = np.array([r[m] for r in rows if r["class"] == k])
            if v.size == 0:
                continue
            ax.hist(v, bins=bins, alpha=0.5, label=f"{k} (n={v.size})", density=True)
        ax.set_xlabel(m)
        ax.set_ylabel("density")
        ax.set_title(f"{m} — by composition class")
        ax.legend()
        fig.tight_layout()
        path = fig_dir / f"hist_{m}_by_class.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        log.info(f"wrote {path}")

        # by atom type
        fig, ax = plt.subplots(figsize=(6, 4))
        for ti, tn in enumerate(types):
            v = np.array([r[m] for r in rows if r["atom_type"] == ti])
            if v.size == 0:
                continue
            ax.hist(v, bins=bins, alpha=0.5, label=f"{tn} (n={v.size})", density=True)
        ax.set_xlabel(m)
        ax.set_ylabel("density")
        ax.set_title(f"{m} — by atom type")
        ax.legend()
        fig.tight_layout()
        path = fig_dir / f"hist_{m}_by_atomtype.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        log.info(f"wrote {path}")


def plot_violin(rows, methods, fig_dir, log):
    classes = sorted(set(r["class"] for r in rows))
    for m in methods:
        fig, ax = plt.subplots(figsize=(6, 4))
        data = []
        labels = []
        for k in classes:
            v = np.array([r[m] for r in rows if r["class"] == k])
            if v.size == 0:
                continue
            # clip extreme tails for the violin so the body is visible
            vmax = np.nanpercentile(v, 99.5)
            v = v[v <= vmax]
            data.append(v)
            labels.append(f"{k}\n(n={v.size})")
        if not data:
            continue
        parts = ax.violinplot(data, showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(m)
        ax.set_title(f"{m} — violins by class (clipped at 99.5th pct)")
        fig.tight_layout()
        path = fig_dir / f"violin_{m}_by_class.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        log.info(f"wrote {path}")


def plot_edge_norms(recv_t, send_t, norms, fig_dir, log):
    pairs = sorted(set(zip(recv_t.tolist(), send_t.tolist())))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vmax = np.nanpercentile(norms, 99.5)
    bins = np.linspace(0, vmax, 80)
    for r, s in pairs:
        mask = (recv_t == r) & (send_t == s)
        v = norms[mask]
        if v.size == 0:
            continue
        ax.hist(
            v,
            bins=bins,
            alpha=0.4,
            label=f"{CHEMICAL_SYMBOLS[r]}<-{CHEMICAL_SYMBOLS[s]} (n={v.size})",
            density=True,
        )
    ax.set_xlabel("|h_ij|_2 (per-edge F2 norm)")
    ax.set_ylabel("density")
    ax.set_title("Per-edge F2 norm by directed (recv<-send) type pair")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "edge_norms_by_pair.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info(f"wrote {path}")


def main():
    log = setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"device: {DEVICE}")
    log.info(f"package: {M0_PACKAGE}")
    log.info(f"train: {TRAIN_XYZ}")
    log.info(f"val:   {VAL_XYZ}")

    log.info("loading model package...")
    model, buffer, type_mapper, nl_transform = make_pipeline()
    log.info(
        f"model loaded; allegro module found: {type(find_allegro_module(model)).__name__}"
    )

    # sanity probe on first val frame
    log.info("sanity probe on first val frame...")
    val_atoms_all = ase.io.read(str(VAL_XYZ), index=":")
    log.info(f"val frames: {len(val_atoms_all)}")
    data = atoms_to_data(val_atoms_all[0], type_mapper, nl_transform)
    with torch.enable_grad():
        out = model(data)
    h_ij = buffer["edge_features"]
    log.info(
        f"sanity: frame0 atoms={data[AtomicDataDict.POSITIONS_KEY].shape[0]} "
        f"edges={h_ij.shape[0]} F2_dim={h_ij.shape[1]} (expect {F2_DIM})"
    )
    assert h_ij.shape[1] == F2_DIM

    # train pass: compute per-type stats
    log.info(f"loading train xyz: {TRAIN_XYZ}")
    train_atoms = ase.io.read(str(TRAIN_XYZ), index=":")
    log.info(f"train frames: {len(train_atoms)}")
    log.info("running stats pass over train...")
    stats = stats_pass(model, buffer, type_mapper, nl_transform, train_atoms, log)
    np.savez(OUT_DIR / "stats_train.npz", **stats)
    log.info(f"saved stats to {OUT_DIR/'stats_train.npz'}")
    for ti, tn in enumerate(CHEMICAL_SYMBOLS):
        log.info(f"  type {tn}: N={stats['N_per_type'][ti]} atoms")

    # score pass on val
    log.info("running score pass over val...")
    out_atoms, edge_recv_t, edge_send_t, edge_norms = score_pass(
        model, buffer, type_mapper, nl_transform, val_atoms_all, stats, log
    )

    # write extxyz with custom per-atom arrays
    xyz_out = OUT_DIR / "val_with_si.xyz"
    log.info(f"writing extxyz to {xyz_out}")
    ase.io.write(str(xyz_out), out_atoms, format="extxyz")

    # save edge records compactly
    np.savez(
        OUT_DIR / "edge_norms.npz",
        recv_type=edge_recv_t,
        send_type=edge_send_t,
        norm=edge_norms,
    )

    # plots
    methods = [
        "s_F2_norm",
        "s_F2a_norm",
        "s_F2b_norm",
        "s_F2c_norm",
        "s_F2_maha",
        "s_F2c_maha",
    ]
    rows = collect_per_atom_arrays(out_atoms, methods)
    log.info(f"per-atom rows: {len(rows)}")
    # class breakdown
    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    log.info(f"per-atom counts by class: {counts}")
    type_counts = {tn: 0 for tn in CHEMICAL_SYMBOLS}
    for r in rows:
        type_counts[CHEMICAL_SYMBOLS[r["atom_type"]]] += 1
    log.info(f"per-atom counts by type: {type_counts}")

    plot_histograms(rows, methods, FIG_DIR, log)
    plot_violin(rows, methods, FIG_DIR, log)
    plot_edge_norms(edge_recv_t, edge_send_t, edge_norms, FIG_DIR, log)

    # summary json
    summary = {
        "n_train_frames": len(train_atoms),
        "n_val_frames": len(val_atoms_all),
        "n_val_atoms": len(rows),
        "per_type_train_counts": {
            tn: int(stats["N_per_type"][ti]) for ti, tn in enumerate(CHEMICAL_SYMBOLS)
        },
        "per_class_val_counts": counts,
        "per_type_val_counts": type_counts,
        "methods": methods,
        "F2_dim": F2_DIM,
        "outputs": {
            "extxyz": str(xyz_out),
            "stats": str(OUT_DIR / "stats_train.npz"),
            "edge_norms": str(OUT_DIR / "edge_norms.npz"),
            "figures": str(FIG_DIR),
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("DONE")


if __name__ == "__main__":
    main()

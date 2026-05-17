#!/bin/bash
# Bootstrap the sparse_delta software stack.
# Run from the project's `software/` directory:
#   cd software && bash install.sh
#
# Clones the read-only mir-group forks of NequIP and Allegro, then runs `uv sync`
# from the project root to register all workspace members and resolve the venv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${SCRIPT_DIR}"

# 1. Clone read-only mir-group forks. NEVER push to these.
if [[ ! -d nequip-private ]]; then
    echo "[install] cloning nequip-private (mir-group, read-only)"
    git clone --branch develop git@github.com:mir-group/nequip-private.git
else
    echo "[install] nequip-private already present, skipping clone"
fi

if [[ ! -d allegro-private ]]; then
    echo "[install] cloning allegro-private (mir-group, read-only)"
    git clone --branch develop git@github.com:mir-group/allegro-private.git
else
    echo "[install] allegro-private already present, skipping clone"
fi

# 1b. Apply sparse_delta-local patches to the read-only forks.
# These are sparse_delta-only modifications; they are NEVER pushed back to
# the mir-group remotes (see sparse_delta CLAUDE.md gotcha "never push forks").
# Idempotent: detected by grepping for the patch's distinguishing symbol.
PATCH_DIR="${SCRIPT_DIR}/sparse-delta-core/patches"

apply_patch() {
    local patch_file="$1"
    local marker="$2"
    local marker_file="$3"
    if [[ ! -f "${patch_file}" ]]; then
        echo "[install] patch ${patch_file} not present, skipping"
        return
    fi
    if grep -q "${marker}" "${marker_file}"; then
        echo "[install] $(basename ${patch_file}) already applied, skipping"
    else
        echo "[install] applying $(basename ${patch_file})"
        # --fuzz=0: refuse to silently apply a patch whose context has
        # drifted. If upstream Allegro changes, fail loudly so the patch
        # is regenerated against the new state.
        (cd allegro-private && patch -p1 --fuzz=0 < "${patch_file}")
    fi
}

# Patch 1: surface Allegro's pre-final TP output as an AtomicDataDict
# entry and advertise its irreps in irreps_out. Required by
# sparse_delta_core.{features.M0InvariantFeatures, model.build_warmstart_composite}.
apply_patch \
    "${PATCH_DIR}/allegro_expose_pre_final_tp_out.patch" \
    "expose_pre_final_tp_out" \
    "allegro-private/allegro/nn/_allegro.py"

# Patch 2: `Contracter.enable_{Triton,CuEquivariance}Contracter` use
# `load_state_dict(..., strict=False)` so the cuet kernel's deterministic
# Clebsch-Gordan buffers (which the original Contracter doesn't have)
# don't trigger a strict-mode mismatch. Required for the
# `enable_CuEquivarianceContracter` model modifier to work end-to-end.
apply_patch \
    "${PATCH_DIR}/allegro_contracter_strict_false.patch" \
    "strict=False" \
    "allegro-private/allegro/nn/_strided/_contract.py"

# 2. uv workspace sync from the project root.
#
# Extra selection: pyproject.toml declares mutually exclusive `cuda` and
# `rocm64` extras for the per-cluster torch wheel. Pass the extra via
# ``SPARSE_DELTA_UV_EXTRA=cuda`` (FASRC, NVIDIA) or
# ``SPARSE_DELTA_UV_EXTRA=rocm64`` (Frontier). Defaults to ``cuda`` on
# Linux x86_64 hosts that look NVIDIA (``nvidia-smi`` present) and to
# ``rocm64`` if ``rocm-smi`` is present; if neither is detectable, the
# user must set the env var explicitly.
cd "${PROJECT_ROOT}"

if [[ -z "${SPARSE_DELTA_UV_EXTRA:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        SPARSE_DELTA_UV_EXTRA=cuda
    elif command -v rocm-smi >/dev/null 2>&1; then
        SPARSE_DELTA_UV_EXTRA=rocm64
    else
        echo "[install] WARNING: could not auto-detect cluster (no nvidia-smi or rocm-smi)." >&2
        echo "          Set SPARSE_DELTA_UV_EXTRA=cuda or rocm64 explicitly." >&2
        echo "          Continuing with no extra; torch wheels may be missing." >&2
        SPARSE_DELTA_UV_EXTRA=""
    fi
fi

if [[ -n "${SPARSE_DELTA_UV_EXTRA}" ]]; then
    echo "[install] running uv sync --extra ${SPARSE_DELTA_UV_EXTRA} at ${PROJECT_ROOT}"
    uv sync --extra "${SPARSE_DELTA_UV_EXTRA}"
else
    echo "[install] running uv sync at ${PROJECT_ROOT}"
    uv sync
fi

echo "[install] done. Sanity check:"
echo "  uv run python -c 'import nequip; import allegro; import sparse_delta_core; print(\"ok\")'"

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
ALLEGRO_PATCH="${SCRIPT_DIR}/sparse-delta-core/patches/allegro_expose_pre_final_tp_out.patch"
if [[ -f "${ALLEGRO_PATCH}" ]]; then
    if grep -q "expose_pre_final_tp_out" allegro-private/allegro/nn/_allegro.py; then
        echo "[install] allegro pre-final-TP patch already applied, skipping"
    else
        echo "[install] applying allegro pre-final-TP patch"
        # --fuzz=0: refuse to silently apply a patch whose context has
        # drifted. If upstream Allegro changes the forward, fail loudly so
        # the patch is regenerated against the new state.
        (cd allegro-private && patch -p1 --fuzz=0 < "${ALLEGRO_PATCH}")
    fi
fi

# 2. uv workspace sync from the project root.
cd "${PROJECT_ROOT}"
echo "[install] running uv sync at ${PROJECT_ROOT}"
uv sync

echo "[install] done. Sanity check:"
echo "  uv run python -c 'import nequip; import allegro; import sparse_delta_core; print(\"ok\")'"

#!/bin/bash
# submit_all.sh — submit all three sparsity_coeff variants
# independently (no inter-dependencies; each loads stage 2 weights).

set -euo pipefail
SWEEP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for sc in 0.01 0.1 0.5; do
  out=$(sbatch "$SWEEP/sc_${sc}/train.sh")
  jid=$(echo "$out" | awk '{print $NF}')
  echo "sc_${sc} -> $jid"
done

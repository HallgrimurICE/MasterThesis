#!/usr/bin/env bash
set -euo pipefail

# Example usage:
#   QUEUE=normal NCORES=8 MEM=16G WALLTIME=02:00 \
#   EXTRA_ARGS="--mode br_vs_neg --weights_path data/sl_params.npz --rounds 50" \
#   ./cluster/submit_single.sh

QUEUE="${QUEUE:-normal}"
NCORES="${NCORES:-4}"
NGPU="${NGPU:-0}"
WALLTIME="${WALLTIME:-01:00}"
MEM="${MEM:-8G}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CMD="${CMD:-python -m diplomacy.experiment_runner}"

mkdir -p logs

GPU_ARGS=()
if [[ "${NGPU}" -gt 0 ]]; then
  GPU_ARGS=(-gpu "num=${NGPU}")
fi

bsub \
  -q "${QUEUE}" \
  -n "${NCORES}" \
  -W "${WALLTIME}" \
  -M "${MEM}" \
  -R "rusage[mem=${MEM}]" \
  "${GPU_ARGS[@]}" \
  -oo "logs/%J.out" \
  -eo "logs/%J.err" \
  ${CMD} ${EXTRA_ARGS}

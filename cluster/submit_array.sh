#!/usr/bin/env bash
set -euo pipefail

# Example usage (array over seeds):
#   ARRAY="1-50" QUEUE=normal NCORES=4 MEM=8G WALLTIME=01:00 \
#   EXTRA_ARGS="--mode br_vs_neg --weights_path data/sl_params.npz --rounds 50 --seeds 0:49" \
#   ./cluster/submit_array.sh

QUEUE="${QUEUE:-normal}"
NCORES="${NCORES:-4}"
NGPU="${NGPU:-0}"
WALLTIME="${WALLTIME:-01:00}"
MEM="${MEM:-8G}"
ARRAY="${ARRAY:-1-10}"
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
  -J "diplomacy[${ARRAY}]" \
  -oo "logs/%J.%I.out" \
  -eo "logs/%J.%I.err" \
  ${CMD} ${EXTRA_ARGS}

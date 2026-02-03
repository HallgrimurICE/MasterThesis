# LSF Cluster Utilities

This folder provides simple submission helpers for LSF (`bsub`) along with usage guidance for the
`diplomacy.experiment_runner` CLI.

## Quick start

### Single job
```bash
QUEUE=normal NCORES=8 MEM=16G WALLTIME=02:00 \
EXTRA_ARGS="--mode br_vs_neg --weights_path data/sl_params.npz --rounds 50" \
./cluster/submit_single.sh
```

### Job array (multi-seed)
```bash
ARRAY="1-50" QUEUE=normal NCORES=4 MEM=8G WALLTIME=01:00 \
EXTRA_ARGS="--mode br_vs_neg --weights_path data/sl_params.npz --rounds 50 --seeds 0:49 --exp_name tom2" \
./cluster/submit_array.sh
```

## Common flags

* `QUEUE`: LSF queue (example: `normal`, `gpul40s`).
* `NCORES`: CPU cores.
* `NGPU`: GPUs to request. If `0`, no GPU is requested.
* `WALLTIME`: Job wall-clock time (e.g. `01:00`, `02:30`).
* `MEM`: Memory (e.g. `8G`, `16G`).
* `ARRAY`: Job array index range for `submit_array.sh` (e.g. `1-100`).
* `EXTRA_ARGS`: Arguments passed to `python -m diplomacy.experiment_runner`.

## Logs and monitoring

* Logs are written to `logs/` with the LSF job id and (for arrays) the index.
* Use `bjobs -l <jobid>` for detailed job info.
* Use `bjobs` to list running jobs and `bkill <jobid>` to stop a job.

## CPU vs GPU guidance

* **CPU queue**: preferred when using heuristic values or when model inference is minimal.
* **GPU queue (e.g. `gpul40s`)**: use when policy/value inference is heavy and you enable batching
  (`--inference_batch_size` and `--profile` to confirm GPU utilization).

## Notes

* LSF job arrays automatically map `LSB_JOBINDEX` to the experiment seed.
* Each run writes outputs to `runs/<exp_name>/<seed>/<timestamp>/`.
* Use job arrays (multi-seed parallelism) for best throughput instead of Python threading.

#!/bin/bash
set -euo pipefail

# Run from repository root:
#   bash examples/Camera/eval_files/eval_hstar_vla_parallel.sh

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ===== Please modify for your local env =====
export STARVLA_PYTHON="/share/project/lmz/miniconda3/envs/starVLA_ap/bin/python3"
parallel_config="/share/project/zhouenshen/hpfs/code/ActivePerception/starVLA/examples/Camera/eval_files/hstar_eval_parallel.yaml"
# ============================================

"${STARVLA_PYTHON}" examples/Camera/eval_files/eval_hstar_vla_parallel.py \
  --config "${parallel_config}"

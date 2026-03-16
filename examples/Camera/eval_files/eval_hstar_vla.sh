#!/bin/bash
set -euo pipefail

# Run from repository root:
#   bash examples/Camera/eval_files/eval_hstar_vla.sh

# cd "$(dirname "$0")/../../../.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}" # let eval import project modules

# ===== Please modify for your local env =====
export STARVLA_PYTHON="/share/project/lmz/miniconda3/envs/starVLA_ap/bin/python3"
config_path="/share/project/zhouenshen/hpfs/code/ActivePerception/starVLA/examples/Camera/eval_files/hstar_eval_part.yaml"
# ============================================

"${STARVLA_PYTHON}" examples/Camera/eval_files/eval_hstar_vla.py \
  --config "${config_path}"

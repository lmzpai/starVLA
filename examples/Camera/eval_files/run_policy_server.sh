#!/bin/bash
set -euo pipefail

# Run from repository root:
#   bash examples/Camera/eval_files/run_policy_server.sh

export PYTHONPATH="$(pwd):${PYTHONPATH:-}" # let examples import deployment tools
export star_vla_python="/share/project/lmz/miniconda3/envs/starVLA_ap/bin/python3"
your_ckpt=/share/project/zhouenshen/hpfs/code/ActivePerception/starVLA/checkpoints/QwenCamera_v1/QwenCamera_hstar_pano_chunk_size_2_history_stride_2_sample_stride_2_history_mode_random/checkpoints/epoch_5_pytorch_model.pt
gpu_id=0
port=10002
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################

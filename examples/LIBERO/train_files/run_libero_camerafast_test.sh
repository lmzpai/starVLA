###########################################################################################
# === Please modify the following paths according to your environment ===
## Model
Framework_name=CameraFast
base_vlm=/share/project/zhouenshen/hpfs/ckpt/vlm/Qwen3-VL-4B-Instruct-Action
## Data
oxe_data_root=/share/project/zhouenshen/sfs/dataset/libero
data_mix=libero_all
fast_tokenizer=/share/project/zhouenshen/hpfs/ckpt/vla/fast
# === End of environment variable configuration ===

# network config
## Pick a valid network interface for torch.distributed (fallback to eth0)
IFACE="bond0"
if [ ! -d "/sys/class/net/${IFACE}" ]; then
  IFACE="eth0"
fi
export GLOO_SOCKET_IFNAME="${IFACE}"
export NCCL_SOCKET_IFNAME="${IFACE}"

## If no InfiniBand is visible, disable IB to avoid NCCL init errors
if [ ! -d "/sys/class/infiniband" ] || [ -z "$(ls -A /sys/class/infiniband 2>/dev/null)" ]; then
  export NCCL_IB_DISABLE=1
fi

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)
export NCCL_DEBUG=INFO
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export OMP_NUM_THREADS=4
export NCCL_IB_HCA=mlx5_0,mlx5_1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_BLOCKING_WAIT=1
export NCCL_PORT=25161
###########################################################################################


# mv this script to the output dir
export WANDB_MODE=disableds

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_camerafast.py \
  --config_yaml /share/project/zhouenshen/hpfs/code/ActivePerception/starVLA/examples/CameraFast/train_files/test.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${oxe_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --framework.action_model.tokenizer ${fast_tokenizer} \
  # --is_debug True


# multi-node launch example

# accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --main_process_ip $MASTER_ADDR \
#   --main_process_port $MASTER_PORT \
#   --machine_rank $SLURM_PROCID \
#   --num_machines $SLURM_NNODES \
#   --num_processes=${TOTAL_GPUS} \
#   starVLA/training/train_starvla.py \
#   --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
#   --framework.framework_py QwenGR00T \
#   --framework.qwenvl.base_vlm microsoft/Florence-2-large \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --wandb_project your_project \
#   --wandb_entity your_name


#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-/root/autodl-tmp/MRTOPSD-ENV}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/Qwen3-4B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/opsd_runs}"
RUN_CONFIG="${RUN_CONFIG:-qwen3_4b_single_gpu}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12949}"

LEARNING_RATE="${LEARNING_RATE:-5e-6}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-32}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-30}"
MAX_LENGTH="${MAX_LENGTH:-20000}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
SAVE_STEPS="${SAVE_STEPS:-25}"
LOGGING_STEPS="${LOGGING_STEPS:-2}"

TEMPERATURE="${TEMPERATURE:-1.1}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
BETA="${BETA:-0}"
LMBDA="${LMBDA:-1}"
JSD_TOKEN_CLIP="${JSD_TOKEN_CLIP:-0.05}"
USE_VLLM="${USE_VLLM:-0}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"

if [[ ! -x "${ENV_DIR}/bin/accelerate" ]]; then
  echo "Missing accelerate in ${ENV_DIR}. Finish the install scripts first." >&2
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing model directory: ${MODEL_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
mkdir -p /root/autodl-tmp/.cache/huggingface
mkdir -p /root/autodl-tmp/.cache/trl
mkdir -p /root/autodl-tmp/.cache/triton

export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRL_HOME="${TRL_HOME:-/root/autodl-tmp/.cache/trl}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/.cache/triton}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"

CMD=(
  "${ENV_DIR}/bin/accelerate" launch
  --config_file "${ROOT_DIR}/accelerate.single_gpu.yaml"
  --num_processes 1
  --main_process_port "${MAIN_PROCESS_PORT}"
  "${ROOT_DIR}/opsd_train.py"
  --model_name_or_path "${MODEL_DIR}"
  --trust_remote_code
  --learning_rate "${LEARNING_RATE}"
  --max_grad_norm 0.1
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --gradient_checkpointing
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --output_dir "${OUTPUT_ROOT}"
  --run_config "${RUN_CONFIG}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --save_steps "${SAVE_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --attn_implementation flash_attention_2
  --torch_dtype bfloat16
  --max_length "${MAX_LENGTH}"
  --beta "${BETA}"
  --use_peft
  --lora_r 64
  --lora_alpha 128
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --top_k "${TOP_K}"
  --lmbda "${LMBDA}"
  --fixed_teacher
  --jsd_token_clip "${JSD_TOKEN_CLIP}"
  --wandb_project OPSD
)

if [[ -n "${MAX_STEPS:-}" ]]; then
  CMD+=(--max_steps "${MAX_STEPS}")
fi

if [[ "${USE_VLLM}" == "1" ]]; then
  CMD+=(
    --use_vllm
    --vllm_mode "${VLLM_MODE}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
  )
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"

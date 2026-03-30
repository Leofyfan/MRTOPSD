#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUN_CONFIG="${RUN_CONFIG:-qwen3_4b_smoke}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/opsd_smoke_runs}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
export SAVE_STEPS="${SAVE_STEPS:-1}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export MAX_STEPS="${MAX_STEPS:-1}"
export USE_VLLM="${USE_VLLM:-0}"

"${ROOT_DIR}/scripts/run_opsd_4b_local.sh"

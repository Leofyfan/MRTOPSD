#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="/home/yuanfan/envs/uv_envs/MRTOPSD-ENV"
MODEL_DIR="/home/shenyl/hf/model/Qwen/Qwen3-1.7B"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/train}"




PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/third_party/verl:${PYTHONPATH:-}" \
  "${ENV_DIR}/bin/python" -m verl_opsd.prepare_dataset \
  --model-path "${MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

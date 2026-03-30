#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-/root/autodl-tmp/MRTOPSD-ENV}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/Qwen3-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/processed}"

PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/third_party/verl:${PYTHONPATH:-}" \
  "${ENV_DIR}/bin/python" -m verl_opsd.prepare_dataset \
  --model-path "${MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

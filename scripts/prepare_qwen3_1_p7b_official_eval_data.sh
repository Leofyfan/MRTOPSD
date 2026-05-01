#!/usr/bin/env bash
set -xeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="/home/yuanfan/envs/uv_envs/MRTOPSD-ENV"
PYTHON_BIN="${ENV_DIR}/bin/python"
MODEL_DIR="/home/shenyl/hf/model/Qwen/Qwen3-1.7B"

TRAIN_OUTPUT_DIR="${ROOT_DIR}/data/processed_all_boxed"
BENCH_OUTPUT_DIR="${ROOT_DIR}/data/benchmarks"

export PATH="${ENV_DIR}/bin:$PATH"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/third_party/verl:${PYTHONPATH:-}"
export HF_HOME="${ROOT_DIR}/eval/data/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

mkdir -p "${BENCH_OUTPUT_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"



"${PYTHON_BIN}" -m verl_opsd.prepare_official_bench_eval \
  --model-path "${MODEL_DIR}" \
  --output-dir "${BENCH_OUTPUT_DIR}" \
  --datasets aime24 aime25 hmmt25

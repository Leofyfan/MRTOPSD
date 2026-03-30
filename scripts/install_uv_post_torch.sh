#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-/root/autodl-tmp/MRTOPSD-ENV}"
PYTHON_BIN="${ENV_DIR}/bin/python"
ROOT_DIR="${ROOT_DIR:-/root/MRTOPSD}"
VERL_SRC="${VERL_SRC:-${ROOT_DIR}/third_party/verl}"
CACHE_ROOT="${CACHE_ROOT:-/root/autodl-tmp/.cache}"
TMP_ROOT="${TMP_ROOT:-/root/autodl-tmp/.tmp}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${VERL_SRC}" ]]; then
  echo "Missing verl source tree: ${VERL_SRC}" >&2
  exit 1
fi

mkdir -p "${CACHE_ROOT}/uv" "${CACHE_ROOT}/pip" "${TMP_ROOT}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export TMPDIR="${TMPDIR:-${TMP_ROOT}}"
export TMP="${TMP:-${TMP_ROOT}}"
export TEMP="${TEMP:-${TMP_ROOT}}"

"${PYTHON_BIN}" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch is not installed in the uv env: {exc}")

version = torch.__version__
cuda = torch.version.cuda

if not version.startswith("2.8."):
    raise SystemExit(f"Expected torch 2.8.x, found {version}")

if cuda is None or not cuda.startswith("12.8"):
    raise SystemExit(f"Expected CUDA 12.8 torch wheel, found torch.version.cuda={cuda}")

print(f"Detected torch={version}, cuda={cuda}")
PY

ABI_FLAG="$("${PYTHON_BIN}" - <<'PY'
import torch
print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")
PY
)"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

FLASH_ATTN_WHL="flash_attn-2.8.1+cu12torch2.8cxx11abi${ABI_FLAG}-cp312-cp312-linux_x86_64.whl"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/${FLASH_ATTN_WHL}"

uv pip install --python "${PYTHON_BIN}" \
  "accelerate==1.11.0" \
  "trl==0.26.0" \
  "peft==0.17.1" \
  "deepspeed==0.18.2" \
  "bitsandbytes==0.48.2" \
  "xformers==0.0.32.post1" \
  "triton==3.4.0" \
  "vllm==0.11.0" \
  "flashinfer-python==0.3.1" \
  "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
  "torchdata" \
  "ray[default]" \
  "codetiming" \
  "hydra-core" \
  "pylatexenc" \
  "qwen-vl-utils" \
  "dill" \
  "pybind11" \
  "liger-kernel" \
  "tensorboard" \
  "pytest" \
  "py-spy" \
  "pre-commit" \
  "ruff" \
  "fastapi[standard]>=0.115.0" \
  "optree>=0.13.0" \
  "pydantic>=2.9" \
  "grpcio>=1.62.1"

wget -nv -O "${TMP_DIR}/${FLASH_ATTN_WHL}" "${FLASH_ATTN_URL}"
uv pip install --python "${PYTHON_BIN}" --no-cache-dir "${TMP_DIR}/${FLASH_ATTN_WHL}"

uv pip install --python "${PYTHON_BIN}" \
  "opencv-python" \
  "opencv-fixer"

"${PYTHON_BIN}" - <<'PY'
from opencv_fixer import AutoFix

AutoFix()
print("opencv-fixer completed")
PY

uv pip install --python "${PYTHON_BIN}" --no-deps -e "${VERL_SRC}"

echo "Post-torch dependencies installed into ${ENV_DIR}"

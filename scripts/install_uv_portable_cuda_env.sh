#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_DIR="${ENV_DIR:-$HOME/.local/share/mrtopsd/venvs/mrtopsd-cu128}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/mrtopsd}"
TMP_ROOT="${TMP_ROOT:-$CACHE_ROOT/tmp}"
UV_BIN_DIR="${UV_BIN_DIR:-$HOME/.local/bin}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCH_CUDA_CHANNEL="${TORCH_CUDA_CHANNEL:-cu128}"
CUDA_SERIES="${CUDA_SERIES:-12.8}"
INSTALL_UV_IF_MISSING="${INSTALL_UV_IF_MISSING:-1}"
INSTALL_NVCC_STUBS="${INSTALL_NVCC_STUBS:-0}"
VERIFY_ENV="${VERIFY_ENV:-1}"
MODEL_PATH="${MODEL_PATH:-}"

need_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

ensure_uv() {
  export PATH="${UV_BIN_DIR}:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  if [[ "${INSTALL_UV_IF_MISSING}" != "1" ]]; then
    echo "uv is not installed. Set INSTALL_UV_IF_MISSING=1 or install uv manually." >&2
    exit 1
  fi

  need_cmd curl
  mkdir -p "${UV_BIN_DIR}"
  export UV_UNMANAGED_INSTALL="${UV_BIN_DIR}"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${UV_BIN_DIR}:${PATH}"

  if ! command -v uv >/dev/null 2>&1; then
    echo "uv installation completed but uv is still not in PATH." >&2
    exit 1
  fi
}

warn_driver_state() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: nvidia-smi not found. uv can manage the Python env and user-space CUDA libs," >&2
    echo "but it cannot install the NVIDIA kernel driver." >&2
    return 0
  fi

  echo "Detected NVIDIA driver:"
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -n 4
}

ensure_repo_state() {
  if [[ ! -d "${ROOT_DIR}/third_party/verl" ]]; then
    echo "Missing ${ROOT_DIR}/third_party/verl" >&2
    echo "Copy the full workspace to the new machine before running this installer." >&2
    exit 1
  fi

  if [[ ! -f "${ROOT_DIR}/scripts/install_uv_pre_torch.sh" ]]; then
    echo "Missing pre-torch installer under ${ROOT_DIR}/scripts" >&2
    exit 1
  fi

  if [[ ! -f "${ROOT_DIR}/scripts/install_uv_post_torch.sh" ]]; then
    echo "Missing post-torch installer under ${ROOT_DIR}/scripts" >&2
    exit 1
  fi
}

main() {
  need_cmd bash
  need_cmd wget
  ensure_uv
  ensure_repo_state
  warn_driver_state

  mkdir -p "${CACHE_ROOT}/uv" "${CACHE_ROOT}/pip" "${TMP_ROOT}" "$(dirname "${ENV_DIR}")"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
  export TMPDIR="${TMPDIR:-${TMP_ROOT}}"
  export TMP="${TMP:-${TMP_ROOT}}"
  export TEMP="${TEMP:-${TMP_ROOT}}"
  export UV_PYTHON_INSTALL_DIR

  uv python install "${PYTHON_VERSION}"
  if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    uv venv "${ENV_DIR}" --python "${PYTHON_VERSION}" --seed
  fi

  ENV_DIR="${ENV_DIR}" bash "${ROOT_DIR}/scripts/install_uv_pre_torch.sh"

  uv pip install --python "${ENV_DIR}/bin/python" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL}" \
    "torch==${TORCH_VERSION}"

  if [[ "${INSTALL_NVCC_STUBS}" == "1" ]]; then
    uv pip install --python "${ENV_DIR}/bin/python" \
      "nvidia-cuda-nvcc-cu12==${CUDA_SERIES}.*"
  fi

  ENV_DIR="${ENV_DIR}" \
  ROOT_DIR="${ROOT_DIR}" \
  CACHE_ROOT="${CACHE_ROOT}" \
  TMP_ROOT="${TMP_ROOT}" \
  bash "${ROOT_DIR}/scripts/install_uv_post_torch.sh"

  if [[ "${VERIFY_ENV}" == "1" ]]; then
    if [[ -n "${MODEL_PATH}" ]]; then
      "${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/verify_uv_env.py" --skip-dataset --model-path "${MODEL_PATH}"
    else
      echo "Skipping verify_uv_env.py because MODEL_PATH is not set." >&2
    fi
  fi

  cat <<EOF

Portable uv environment is ready.

Environment path:
  ${ENV_DIR}

User-level CUDA activation:
  source "${ROOT_DIR}/scripts/source_uv_user_cuda_env.sh"

Example:
  ENV_DIR="${ENV_DIR}" source "${ROOT_DIR}/scripts/source_uv_user_cuda_env.sh"

Notes:
  - This script reproduces the Python env and CUDA user-space libraries inside the uv env.
  - It does not install the NVIDIA driver.
  - The project weights and datasets still need to be copied separately.
EOF
}

main "$@"

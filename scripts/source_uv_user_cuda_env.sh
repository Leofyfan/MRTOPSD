#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead of executing it:" >&2
  echo "  source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

ENV_DIR="${ENV_DIR:-$HOME/.local/share/mrtopsd/venvs/mrtopsd-cu128}"
PYTHON_BIN="${ENV_DIR}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python interpreter: ${PYTHON_BIN}" >&2
  return 1
fi

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

SITE_PACKAGES="$("${PYTHON_BIN}" - <<'PY'
import site

paths = [p for p in site.getsitepackages() if p.endswith("site-packages")]
if not paths:
    raise SystemExit("Unable to locate site-packages for the uv environment.")
print(paths[0])
PY
)"

NVIDIA_ROOT="${SITE_PACKAGES}/nvidia"
if [[ ! -d "${NVIDIA_ROOT}" ]]; then
  echo "Missing NVIDIA package tree under ${NVIDIA_ROOT}" >&2
  return 1
fi

mapfile -t CUDA_LIB_DIRS < <(find "${NVIDIA_ROOT}" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
mapfile -t CUDA_INCLUDE_DIRS < <(find "${NVIDIA_ROOT}" -mindepth 2 -maxdepth 2 -type d -name include | sort)

join_by_colon() {
  local IFS=:
  echo "$*"
}

prepend_env() {
  local var_name="$1"
  local joined="$2"
  local current="${!var_name:-}"

  if [[ -z "${joined}" ]]; then
    return 0
  fi

  if [[ -n "${current}" ]]; then
    export "${var_name}=${joined}:${current}"
  else
    export "${var_name}=${joined}"
  fi
}

LIB_JOINED="$(join_by_colon "${CUDA_LIB_DIRS[@]}")"
INCLUDE_JOINED="$(join_by_colon "${CUDA_INCLUDE_DIRS[@]}")"

prepend_env LD_LIBRARY_PATH "${LIB_JOINED}"
prepend_env LIBRARY_PATH "${LIB_JOINED}"
prepend_env CPATH "${INCLUDE_JOINED}"

export CUDA_HOME="${NVIDIA_ROOT}/cuda_runtime"
export CUDA_PATH="${CUDA_HOME}"

if [[ -x "${NVIDIA_ROOT}/cuda_nvcc/bin/nvcc" ]]; then
  export CUDACXX="${NVIDIA_ROOT}/cuda_nvcc/bin/nvcc"
  prepend_env PATH "${NVIDIA_ROOT}/cuda_nvcc/bin"
elif [[ -x "${NVIDIA_ROOT}/cuda_nvcc/bin/ptxas" ]]; then
  prepend_env PATH "${NVIDIA_ROOT}/cuda_nvcc/bin"
fi

echo "Activated uv env: ${ENV_DIR}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "LD_LIBRARY_PATH prefixed with CUDA user-space libraries from ${NVIDIA_ROOT}"

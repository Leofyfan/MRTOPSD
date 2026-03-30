#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-/root/autodl-tmp/MRTOPSD-ENV}"
PYTHON_BIN="${ENV_DIR}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python interpreter: ${PYTHON_BIN}" >&2
  echo "Create the uv environment first: uv venv ${ENV_DIR} --python 3.12 --seed" >&2
  exit 1
fi

uv pip install --python "${PYTHON_BIN}" --upgrade \
  pip \
  setuptools \
  wheel \
  packaging \
  ninja \
  cmake

uv pip install --python "${PYTHON_BIN}" \
  "transformers==4.57.1" \
  "datasets==3.6.0" \
  "wandb==0.22.3" \
  "einops==0.8.1" \
  "safetensors==0.5.3" \
  "sentencepiece==0.1.99" \
  "tiktoken==0.9.0" \
  "math-verify==0.8.0" \
  "numpy<2.0.0" \
  "pyarrow>=15.0.0" \
  "pandas" \
  "hf-transfer" \
  "nvidia-ml-py>=12.560.30"

# These packages are installed without dependency resolution until torch is present.
uv pip install --python "${PYTHON_BIN}" --no-deps \
  "accelerate==1.11.0" \
  "trl==0.26.0" \
  "peft==0.17.1"

echo "Pre-torch dependencies installed into ${ENV_DIR}"

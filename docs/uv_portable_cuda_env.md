# Portable `uv` + User-Space CUDA Environment

This project can be migrated to another machine by rebuilding the Python environment with `uv` and reusing CUDA user-space libraries from the Python environment itself.

## What `uv` can manage

- Python version
- virtual environment location
- PyTorch CUDA wheel selection, for example `cu128`
- NVIDIA user-space CUDA component wheels pulled in by PyTorch or installed explicitly with `uv pip`

## What `uv` cannot manage

- the NVIDIA kernel driver
- a full system CUDA Toolkit installation under `/usr/local/cuda`
- a guaranteed `nvcc` compiler binary from pip wheels

For this repo, that is acceptable because the current environment uses:

- PyTorch `2.8.0` with `cu128`
- prebuilt `flash-attn` wheel
- editable local `verl`

## One-command installer

On the new machine, after copying the repository, run:

```bash
bash scripts/install_uv_portable_cuda_env.sh
```

Useful overrides:

```bash
ENV_DIR=$HOME/.local/share/mrtopsd/venvs/mrtopsd-cu128 \
PYTHON_VERSION=3.12 \
TORCH_VERSION=2.8.0 \
TORCH_CUDA_CHANNEL=cu128 \
CUDA_SERIES=12.8 \
MODEL_PATH=/path/to/Qwen3-4B \
bash scripts/install_uv_portable_cuda_env.sh
```

If you want pip-installed CUDA compiler stubs such as `ptxas` and NVVM headers:

```bash
INSTALL_NVCC_STUBS=1 bash scripts/install_uv_portable_cuda_env.sh
```

## Activate the user-space CUDA environment

After installation, activate the environment with:

```bash
ENV_DIR=$HOME/.local/share/mrtopsd/venvs/mrtopsd-cu128 \
source scripts/source_uv_user_cuda_env.sh
```

This will:

- activate the uv virtual environment
- discover the `site-packages/nvidia/*` directories inside that environment
- prepend their `lib` directories to `LD_LIBRARY_PATH`
- prepend their `include` directories to `CPATH`
- export `CUDA_HOME` as the environment-local CUDA runtime root

## Recommended migration workflow

1. Install a compatible NVIDIA driver on the target machine.
2. Copy this repository, the model weights, and any prepared datasets.
3. Run `bash scripts/install_uv_portable_cuda_env.sh`.
4. `source scripts/source_uv_user_cuda_env.sh`.
5. Run `python scripts/verify_uv_env.py --skip-dataset --model-path /path/to/Qwen3-4B`.

## Practical note

If you later need a full `nvcc`-based development toolkit rather than user-space runtime libraries, do not rely on `uv` alone. Keep `uv` for Python and either:

- use prebuilt wheels whenever possible, or
- add a separate user-space CUDA toolkit manager outside `uv`


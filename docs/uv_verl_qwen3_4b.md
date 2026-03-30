# Qwen3-4B Reproduction With `uv` + `verl`

This repository is upstream `OPSD`, which is implemented with `trl.experimental.gold` rather than `verl`.

For this workspace, the environment is aligned to the `verl` official `cu128 + torch 2.8.0 + vllm 0.11.0` stack, and an `verl_opsd` adaptation layer is added for Qwen3-4B reproduction.

## Local paths

- Environment: `/root/autodl-tmp/MRTOPSD-ENV`
- Model: `/root/autodl-tmp/Qwen3-4B`
- verl source: `/root/MRTOPSD/third_party/verl`

## Installation order

1. Create the environment:

```bash
uv venv /root/autodl-tmp/MRTOPSD-ENV --python 3.12 --seed
```

2. Install pre-torch dependencies:

```bash
bash scripts/install_uv_pre_torch.sh
```

3. Install the CUDA-enabled torch wheel into the uv env:

```bash
uv pip install --python /root/autodl-tmp/MRTOPSD-ENV/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0
```

4. Finish post-torch dependencies, including local editable `verl`, `vllm`, `deepspeed`, and `flash-attn`:

```bash
bash scripts/install_uv_post_torch.sh
```

5. Verify the environment:

```bash
/root/autodl-tmp/MRTOPSD-ENV/bin/python scripts/verify_uv_env.py
```

## Training entrypoints

Prepare verl parquet data:

```bash
bash scripts/prepare_opsd_verl_data.sh
```

Dual-GPU local smoke test:

```bash
bash scripts/run_opsd_4b_verl_smoke.sh
```

Default dual-GPU local training (2 epochs by default):

```bash
bash scripts/run_opsd_4b_verl.sh
```

More aggressive dual-GPU throughput preset:

```bash
bash scripts/run_opsd_4b_verl_fast_dual_gpu.sh
```

Multi-GPU style launch:

```bash
NUM_GPUS=4 ROLLOUT_TP=1 TEACHER_TP=1 bash scripts/run_opsd_4b_verl.sh
```

`run_opsd_4b_verl.sh` now defaults to a dual-GPU setup with `TOTAL_EPOCHS=2`. For Qwen3-4B on 2 GPUs, keeping `ROLLOUT_TP=1` and `TEACHER_TP=1` usually gives better throughput because it creates two single-GPU data-parallel inference replicas instead of one tensor-parallel replica.

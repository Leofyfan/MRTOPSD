# `run_opsd_4b_verl.sh` Parameter Reference

This document explains the parameters used by [`run_opsd_4b_verl.sh`](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh) in both Chinese and English.

本文档对 [`run_opsd_4b_verl.sh`](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh) 里的参数做中英文说明。

## 1. Script Role / 脚本作用

This script is the main launcher for the `OPSD-on-verl` training pipeline. It does four things:

这个脚本是 `OPSD-on-verl` 训练流程的主启动脚本，主要做四件事：

1. Resolve local paths and default environment variables.  
   解析本地路径和默认环境变量。
2. Check whether the parquet training/validation data exists and optionally rebuild it.  
   检查 parquet 训练/验证数据是否存在，必要时自动重建。
3. Export runtime cache and logging environment variables.  
   导出运行时缓存和日志相关环境变量。
4. Convert shell variables into Hydra overrides and launch `verl_opsd.main_ppo_opsd`.  
   把 shell 变量转换成 Hydra override，然后启动 `verl_opsd.main_ppo_opsd`。

## 2. Shell Preamble / Shell 头部

| Parameter | Default | 中文说明 | English meaning |
|---|---:|---|---|
| `set -e` | n/a | 任意命令返回非零时立即退出，避免脚本带着错误继续跑。 | Exit immediately when any command fails. |
| `set -u` | n/a | 访问未定义变量时报错，避免拼写错误悄悄变成空字符串。 | Treat unset variables as an error. |
| `set -o pipefail` | n/a | 管道中任意一段失败都算失败，不只看最后一个命令。 | Make a pipeline fail if any command in it fails. |

## 3. Path And Experiment Variables / 路径与实验命名参数

| Variable | Default | 中文说明 | English meaning |
|---|---:|---|---|
| `ROOT_DIR` | auto | 仓库根目录，由脚本位置自动推断。 | Repository root inferred from the script location. |
| `ENV_DIR` | `/root/autodl-tmp/MRTOPSD-ENV` | `uv`/Python 虚拟环境目录，里面必须有 `bin/python`。 | Python/uv environment directory. |
| `MODEL_DIR` | `/root/autodl-tmp/Qwen3-4B` | 学生模型与 teacher 共用的基础模型路径。 | Base model path shared by the student and teacher. |
| `TRAIN_FILE` | `${ROOT_DIR}/data/processed/opsd_train.parquet` | 训练集 parquet 文件路径。 | Training parquet file path. |
| `VAL_FILE` | `${ROOT_DIR}/data/processed/opsd_val.parquet` | 验证集 parquet 文件路径。 | Validation parquet file path. |
| `PROJECT_NAME` | `opsd_verl` | 训练项目名，会映射到 `trainer.project_name` 和 `WANDB_PROJECT`。 | Project name used by trainer and W&B. |
| `EXPERIMENT_NAME` | `qwen3_4b_opsd_verl` | 具体实验名，会映射到输出目录和 wandb run name。 | Run / experiment name used in output and W&B. |
| `OUTPUT_DIR` | `/root/autodl-tmp/opsd_verl_runs/${EXPERIMENT_NAME}` | 本地输出目录，保存 checkpoint、wandb 本地文件、rollout dump 等。 | Local output directory for checkpoints and logs. |

## 4. Topology And Scale / 拓扑与训练规模参数

| Variable | Default | 中文说明 | English meaning |
|---|---:|---|---|
| `NUM_GPUS` | `2` | 当前节点用于训练的 GPU 数量。 | Number of GPUs used on the current node. |
| `NNODES` | `1` | 训练节点数，多机时使用。 | Number of training nodes. |
| `ROLLOUT_TP` | `1` | rollout/vLLM 的 tensor parallel 大小。 | Tensor parallel size for rollout vLLM. |
| `TEACHER_TP` | `1` | teacher vLLM 的 tensor parallel 大小。 | Tensor parallel size for teacher vLLM. |
| `DEFAULT_WORKER_COUNT` | `NUM_GPUS * 8` | 默认 CPU worker 数，不直接传 Hydra，只作为其他参数默认值来源。 | Derived default worker count used by several worker-related variables. |
| `DEFAULT_TRAIN_BATCH_SIZE` | `NUM_GPUS * 32` | 默认全局 batch size。 | Derived default global training batch size. |
| `DEFAULT_MICRO_BATCH_SIZE` | `NUM_GPUS * 2` | 默认每卡 micro batch size。 | Derived default micro batch size per GPU. |

## 5. Dataloader, Batch, Length, Memory / 数据加载、批大小、长度与显存参数

| Variable | Default | 中文说明 | English meaning |
|---|---:|---|---|
| `ROLLOUT_AGENT_NUM_WORKERS` | `${DEFAULT_WORKER_COUNT}` | rollout/agent loop 的 worker 数。 | Worker count for rollout agent loop. |
| `REWARD_NUM_WORKERS` | `${DEFAULT_WORKER_COUNT}` | reward 计算 worker 数。 | Worker count for reward computation. |
| `DATALOADER_NUM_WORKERS` | `${DEFAULT_WORKER_COUNT}` | dataloader 的 worker 数。 | Number of dataloader workers. |
| `TRAIN_BATCH_SIZE` | `${DEFAULT_TRAIN_BATCH_SIZE}` | 全局训练 batch size。 | Global training batch size. |
| `MICRO_BATCH_SIZE` | `${DEFAULT_MICRO_BATCH_SIZE}` | 每卡 actor update 的 micro batch size。 | Per-GPU micro batch size for actor updates. |
| `PPO_MAX_TOKEN_LEN_PER_GPU` | `65536` | actor/ref 计算 logprob 或更新时每卡最大 token 预算。 | Max token budget per GPU for PPO/logprob computation. |
| `MAX_PROMPT_LENGTH` | `2048` | 学生 prompt 最大长度，超长样本会被过滤。 | Maximum student prompt length. |
| `MAX_RESPONSE_LENGTH` | `1024` | 模型生成回复的最大长度。 | Maximum response generation length. |
| `TEACHER_MAX_MODEL_LEN` | `6144` | teacher vLLM 允许的最大上下文长度。 | Max context length for the teacher vLLM. |
| `ROLLOUT_MAX_MODEL_LEN` | `3072` | rollout vLLM 允许的最大上下文长度。 | Max context length for rollout vLLM. |
| `ROLLOUT_MAX_BATCHED_TOKENS` | `49152` | rollout vLLM 单次调度可装载的最大 token 总量。 | Max batched tokens scheduled by rollout vLLM. |
| `TEACHER_MAX_BATCHED_TOKENS` | `65536` | teacher vLLM 单次调度可装载的最大 token 总量。 | Max batched tokens scheduled by teacher vLLM. |
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | `0.45` | rollout vLLM 目标显存利用率。 | Target GPU memory utilization for rollout vLLM. |
| `TEACHER_GPU_MEMORY_UTILIZATION` | `0.40` | teacher vLLM 目标显存利用率。 | Target GPU memory utilization for teacher vLLM. |
| `ROLLOUT_NAME` | `vllm` | rollout/teacher 推理后端名。当前默认是 `vllm`。 | Inference backend name, currently `vllm`. |

## 6. Training Control, Logging, Offload / 训练控制、日志与 Offload 参数

| Variable | Default | 中文说明 | English meaning |
|---|---:|---|---|
| `SAVE_FREQ` | `200` | 每多少个 training step 保存一次 checkpoint。 | Save checkpoint every N training steps. |
| `TEST_FREQ` | `200` | 每多少个 training step 进行一次验证。 | Run validation every N training steps. |
| `TOTAL_EPOCHS` | `2` | 总训练 epoch 数。 | Total training epochs. |
| `VAL_BEFORE_TRAIN` | `false` | 训练开始前是否先跑一次验证。 | Whether to run validation before training. |
| `WANDB_LOGGER` | `["console","wandb"]` | logger 列表，默认同时输出到控制台和 wandb。 | Active loggers, defaulting to console and W&B. |
| `LOG_VAL_GENERATIONS` | `16` | 每次验证最多记录多少条样本回答。 | Number of validation generations to log each validation run. |
| `LOG_VAL_ERROR_GENERATIONS` | `16` | 每次验证最多记录多少条错误回答。 | Number of incorrect validation generations to log. |
| `ACTOR_PARAM_OFFLOAD` | `False` | 是否把 actor 参数 offload 到 CPU。 | Whether to offload actor parameters to CPU. |
| `ACTOR_OPTIMIZER_OFFLOAD` | `False` | 是否把 actor optimizer state offload 到 CPU。 | Whether to offload actor optimizer states to CPU. |
| `REF_PARAM_OFFLOAD` | `False` | 是否把 reference policy 参数 offload 到 CPU。 | Whether to offload reference model parameters to CPU. |
| `TEACHER_ENABLE_RESOURCE_POOL` | `False` | 是否启用 teacher resource pool 模式。 | Whether to enable teacher resource pool mode. |

## 7. OPSD Rubric Distillation Parameters / OPSD Rubric 自蒸馏参数

| Variable | Default | 中文说明 | English meaning |
|---|---:|---|---|
| `OPSD_RUBRIC_ENABLED` | `true` | 是否启用 rubric-based self-distillation。 | Enable rubric-based self-distillation. |
| `OPSD_RUBRIC_WARMUP_STEPS` | `100` | curriculum warmup 步数，前期偏向 generic rubric。 | Warmup steps before self-mined rubrics become active. |
| `OPSD_RUBRIC_MIX_STEPS` | `300` | curriculum 混合阶段步数。 | Number of curriculum mixing steps. |
| `OPSD_RUBRIC_SEED` | `0` | rubric 课程与采样相关随机种子。 | Random seed for rubric curriculum behavior. |
| `OPSD_RUBRIC_MIN_RESPONSE_CHARS` | `32` | 回答低于该字符数时，不作为高质量 rubric 挖掘候选。 | Minimum response length for rubric mining candidates. |
| `OPSD_RUBRIC_MAX_PENDING_REQUESTS` | `128` | 后台 rubric 更新队列的最大待处理请求数。 | Maximum pending requests in the rubric update queue. |
| `ROLLOUT_DATA_DIR` | `${OUTPUT_DIR}/rollout_data` | 训练 rollout 全量回答落盘目录。 | Directory for dumping all rollout generations. |
| `ROLLOUT_ERROR_DATA_DIR` | `${OUTPUT_DIR}/rollout_error_data` | 训练 rollout 错误回答单独落盘目录。 | Directory for dumping incorrect rollout generations only. |

## 8. Helper Functions / 辅助函数

### `parquet_num_rows`

- 中文：读取 parquet 文件并返回样本数，用来判断训练/验证集是否存在、是否过小。  
- English: Reads a parquet file and returns its row count, used for dataset existence and size checks.

### `prepare_dataset_if_needed`

- 中文：当 `TRAIN_FILE`/`VAL_FILE` 缺失，或者训练集小于 `TRAIN_BATCH_SIZE` 时，自动调用 [`prepare_opsd_verl_data.sh`](/root/MRTOPSD/scripts/prepare_opsd_verl_data.sh) 重新构建数据。  
- English: Rebuilds the dataset with [`prepare_opsd_verl_data.sh`](/root/MRTOPSD/scripts/prepare_opsd_verl_data.sh) when parquet files are missing or undersized.

The derived teacher prompt length limit used in data preparation is:

数据重建时 teacher prompt 最大长度使用下面的推导值：

- `max_teacher_prompt_length = TEACHER_MAX_MODEL_LEN - MAX_RESPONSE_LENGTH`

Reason / 原因:
- 中文：给 teacher prompt 预留 response 生成空间，避免总上下文超过 teacher vLLM 的上限。  
- English: Reserves generation space so the teacher prompt plus response stays within `TEACHER_MAX_MODEL_LEN`.

## 9. Exported Runtime Environment Variables / 导出的运行时环境变量

| Variable | Default | 中文说明 | English meaning |
|---|---:|---|---|
| `HF_HOME` | `/root/autodl-tmp/.cache/huggingface` | Hugging Face 主缓存目录。 | Main Hugging Face cache directory. |
| `HUGGINGFACE_HUB_CACHE` | `${HF_HOME}/hub` | Hugging Face hub cache 目录。 | Hugging Face hub cache directory. |
| `VLLM_CACHE_ROOT` | `/root/autodl-tmp/.cache/vllm` | vLLM 缓存目录。 | vLLM cache directory. |
| `RAY_TMPDIR` | `/root/autodl-tmp/.cache/ray` | Ray 临时目录。 | Ray temporary directory. |
| `TRITON_CACHE_DIR` | `/root/autodl-tmp/.cache/triton` | Triton 编译缓存目录。 | Triton compilation cache directory. |
| `WANDB_ENTITY` | `leofyfan-east-china-normal-university` | wandb team/entity。 | W&B entity/team slug. |
| `WANDB_USERNAME` | `leofyfan` | wandb 用户名。 | W&B username. |
| `WANDB_PROJECT` | `${PROJECT_NAME}` | wandb project 名。 | W&B project name. |
| `WANDB_NAME` | `${EXPERIMENT_NAME}` | wandb run name。 | W&B run name. |
| `WANDB_DIR` | `${OUTPUT_DIR}/wandb` | wandb 本地运行目录。 | Local W&B run directory. |
| `WANDB_CACHE_DIR` | `/root/autodl-tmp/.cache/wandb` | wandb cache 目录。 | W&B cache directory. |
| `WANDB_CONFIG_DIR` | `/root/autodl-tmp/.config/wandb` | wandb 配置目录。 | W&B config directory. |
| `WANDB_DATA_DIR` | `/root/autodl-tmp/.local/share/wandb` | wandb 数据目录。 | W&B data directory. |
| `TOKENIZERS_PARALLELISM` | `false` | 关闭 tokenizer 并行警告，避免日志污染。 | Disable tokenizer parallelism warnings. |
| `WANDB_MODE` | `online` | wandb 在线/离线模式。 | W&B online/offline mode. |
| `VERL_LOGGING_LEVEL` | `INFO` | `verl` 日志级别。 | Logging level for `verl`. |
| `VERL_USE_OPSD_TEACHER` | `1` | 启用 OPSD 特化 teacher 逻辑的开关。 | Flag for enabling OPSD-specific teacher logic. |
| `PYTHONPATH` | `${ROOT_DIR}:${ROOT_DIR}/third_party/verl:${PYTHONPATH:-}` | 把本仓库与 vendored `verl` 加入 Python 搜索路径。 | Adds the repo and vendored `verl` to Python import paths. |

## 10. Fixed LoRA Target Modules / 固定的 LoRA 目标模块

| Parameter | Value | 中文说明 | English meaning |
|---|---:|---|---|
| `TARGET_MODULES` | `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` | 对 Qwen 系列常见注意力层和 MLP 层打 LoRA。 | LoRA target modules for common Qwen attention and MLP projections. |

## 11. Hydra Override Parameters / Hydra 覆盖参数说明

The script finally constructs a `CMD` array and passes these values into Hydra.  
脚本最后会构造 `CMD` 数组，把下面这些值作为 Hydra override 传给 `verl`。

### 11.1 Algorithm / 算法参数

| Hydra key | Value | 中文说明 | English meaning |
|---|---:|---|---|
| `algorithm.adv_estimator` | `grpo` | 优势估计器使用 GRPO 风格，而不是 GAE。 | Use the GRPO-style advantage estimator instead of GAE. |
| `algorithm.use_kl_in_reward` | `False` | 不把 KL 项直接并入 reward。 | Do not add KL directly into the reward. |

### 11.2 Data / 数据参数

| Hydra key | Source | 中文说明 | English meaning |
|---|---:|---|---|
| `data.train_files` | `TRAIN_FILE` | 训练数据文件。 | Training dataset file(s). |
| `data.val_files` | `VAL_FILE` | 验证数据文件。 | Validation dataset file(s). |
| `data.train_batch_size` | `TRAIN_BATCH_SIZE` | 全局训练 batch。 | Global training batch size. |
| `data.max_prompt_length` | `MAX_PROMPT_LENGTH` | 最大 prompt 长度。 | Maximum prompt length. |
| `data.max_response_length` | `MAX_RESPONSE_LENGTH` | 最大回复长度。 | Maximum response length. |
| `data.filter_overlong_prompts` | `True` | 自动过滤超长 prompt。 | Filter overlong prompts automatically. |
| `data.dataloader_num_workers` | `DATALOADER_NUM_WORKERS` | dataloader worker 数。 | Number of dataloader workers. |
| `data.truncation` | `error` | 超长时不静默截断，而是报错/过滤。 | Use strict truncation behavior rather than silent clipping. |
| `data.shuffle` | `False` | 当前脚本默认不在 `verl` 这一层打乱数据。 | Disable data shuffling at the `verl` layer. |
| `data.return_raw_chat` | `True` | 返回原始 chat messages。 | Return raw chat messages. |
| `+data.apply_chat_template_kwargs.enable_thinking` | `False` | 学生 prompt 不启用 thinking 模式。 | Disable thinking mode when applying the student chat template. |

### 11.3 Reward / 奖励参数

| Hydra key | Source | 中文说明 | English meaning |
|---|---:|---|---|
| `reward.num_workers` | `REWARD_NUM_WORKERS` | reward 侧 worker 数。 | Reward worker count. |

Important note / 重要说明:
- 中文：当前数据是由 [`prepare_dataset.py`](/root/MRTOPSD/verl_opsd/prepare_dataset.py#L87) 写入 `data_source="math_dapo"`，所以 reward dispatch 会走 [`reward_score/__init__.py`](/root/MRTOPSD/third_party/verl/verl/utils/reward_score/__init__.py#L59) 并调用 [`math_dapo.compute_score`](/root/MRTOPSD/third_party/verl/verl/utils/reward_score/math_dapo.py#L242)。  
- English: The dataset writer sets `data_source="math_dapo"`, so reward dispatch goes through [`reward_score/__init__.py`](/root/MRTOPSD/third_party/verl/verl/utils/reward_score/__init__.py#L59) and calls [`math_dapo.compute_score`](/root/MRTOPSD/third_party/verl/verl/utils/reward_score/math_dapo.py#L242).

### 11.4 Student Model / 学生模型参数

| Hydra key | Value | 中文说明 | English meaning |
|---|---:|---|---|
| `actor_rollout_ref.model.path` | `${MODEL_DIR}` | 学生模型权重路径。 | Student model path. |
| `actor_rollout_ref.model.trust_remote_code` | `True` | 允许 Hugging Face 自定义模型代码。 | Trust remote code from HF model repo. |
| `actor_rollout_ref.model.use_remove_padding` | `True` | 使用 remove padding 优化训练效率。 | Enable remove-padding optimization. |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `True` | 开启梯度检查点以节省显存。 | Enable gradient checkpointing. |
| `actor_rollout_ref.model.lora_rank` | `64` | LoRA rank。 | LoRA rank. |
| `actor_rollout_ref.model.lora_alpha` | `128` | LoRA alpha。 | LoRA alpha. |
| `actor_rollout_ref.model.target_modules` | `${TARGET_MODULES}` | LoRA 注入模块。 | LoRA target modules. |

### 11.5 Actor / Actor 训练参数

| Hydra key | Source / Value | 中文说明 | English meaning |
|---|---:|---|---|
| `actor_rollout_ref.actor.strategy` | `fsdp2` | actor 训练策略，使用 FSDP2。 | Actor training strategy, using FSDP2. |
| `actor_rollout_ref.actor.optim.lr` | `5e-6` | actor 学习率。 | Actor learning rate. |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `TRAIN_BATCH_SIZE` | PPO mini-batch size。 | PPO mini-batch size. |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `MICRO_BATCH_SIZE` | 每卡 micro-batch size。 | Per-GPU PPO micro-batch size. |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `PPO_MAX_TOKEN_LEN_PER_GPU` | actor 每卡 token 预算。 | Max actor token budget per GPU. |
| `actor_rollout_ref.actor.use_dynamic_bsz` | `True` | 启用动态 batch。 | Enable dynamic batching. |
| `actor_rollout_ref.actor.use_kl_loss` | `False` | 关闭 KL loss。 | Disable actor KL loss. |
| `actor_rollout_ref.actor.entropy_coeff` | `0` | 熵正则系数。 | Entropy regularization coefficient. |
| `actor_rollout_ref.actor.fsdp_config.model_dtype` | `bf16` | 模型权重 dtype。 | Model dtype set to bf16. |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | `ACTOR_PARAM_OFFLOAD` | actor 参数 offload 开关。 | Actor parameter offload switch. |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `ACTOR_OPTIMIZER_OFFLOAD` | actor optimizer offload 开关。 | Actor optimizer offload switch. |

### 11.6 Rollout / Rollout 推理参数

| Hydra key | Source / Value | 中文说明 | English meaning |
|---|---:|---|---|
| `actor_rollout_ref.rollout.name` | `ROLLOUT_NAME` | rollout 推理后端。 | Rollout inference backend. |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `ROLLOUT_TP` | rollout TP 大小。 | Rollout tensor-parallel size. |
| `actor_rollout_ref.rollout.n` | `1` | 每个 prompt 采样的训练回复数。 | Number of training responses sampled per prompt. |
| `actor_rollout_ref.rollout.temperature` | `1.1` | rollout 采样温度。 | Rollout sampling temperature. |
| `actor_rollout_ref.rollout.top_p` | `0.95` | rollout top-p。 | Rollout top-p. |
| `actor_rollout_ref.rollout.top_k` | `20` | rollout top-k。 | Rollout top-k. |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `ROLLOUT_GPU_MEMORY_UTILIZATION` | rollout 显存利用率目标。 | Rollout GPU memory utilization target. |
| `actor_rollout_ref.rollout.enforce_eager` | `True` | 强制 eager 模式。 | Force eager execution. |
| `actor_rollout_ref.rollout.load_format` | `safetensors` | rollout 模型加载格式。 | Rollout model load format. |
| `actor_rollout_ref.rollout.prompt_length` | `MAX_PROMPT_LENGTH` | rollout prompt 长度限制。 | Rollout prompt length limit. |
| `actor_rollout_ref.rollout.response_length` | `MAX_RESPONSE_LENGTH` | rollout 回复长度限制。 | Rollout response length limit. |
| `actor_rollout_ref.rollout.max_model_len` | `ROLLOUT_MAX_MODEL_LEN` | rollout 最大总上下文长度。 | Rollout max total context length. |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `ROLLOUT_MAX_BATCHED_TOKENS` | rollout 单步最大 batched tokens。 | Rollout max batched tokens. |
| `actor_rollout_ref.rollout.max_num_seqs` | `TRAIN_BATCH_SIZE` | rollout 单步最大并行序列数。 | Max rollout sequences scheduled at once. |
| `actor_rollout_ref.rollout.agent.num_workers` | `ROLLOUT_AGENT_NUM_WORKERS` | agent loop worker 数。 | Agent loop worker count. |
| `actor_rollout_ref.rollout.log_prob_use_dynamic_bsz` | `True` | rollout logprob 计算使用动态 batch。 | Use dynamic batching for rollout logprob. |
| `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu` | `PPO_MAX_TOKEN_LEN_PER_GPU` | rollout logprob 每卡 token 限额。 | Max rollout logprob token budget per GPU. |

### 11.7 Reference Policy / 参考策略参数

| Hydra key | Source / Value | 中文说明 | English meaning |
|---|---:|---|---|
| `actor_rollout_ref.ref.log_prob_use_dynamic_bsz` | `True` | reference policy 的 logprob 计算使用动态 batch。 | Use dynamic batching for reference logprob. |
| `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | `PPO_MAX_TOKEN_LEN_PER_GPU` | reference policy 每卡 token 限额。 | Max reference logprob token budget per GPU. |
| `actor_rollout_ref.ref.fsdp_config.param_offload` | `REF_PARAM_OFFLOAD` | reference 参数 offload 开关。 | Reference parameter offload switch. |

### 11.8 Distillation / 蒸馏参数

| Hydra key | Source / Value | 中文说明 | English meaning |
|---|---:|---|---|
| `distillation.enabled` | `True` | 启用蒸馏。 | Enable distillation. |
| `distillation.num_workers` | `NUM_GPUS` | distillation worker 数。 | Distillation worker count. |
| `distillation.teacher_model.enable_resource_pool` | `TEACHER_ENABLE_RESOURCE_POOL` | teacher resource pool 开关。 | Teacher resource pool switch. |
| `distillation.teacher_model.model_path` | `MODEL_DIR` | teacher 模型路径。 | Teacher model path. |
| `distillation.teacher_model.n_gpus_per_node` | `NUM_GPUS` | 每节点 teacher 可用 GPU 数。 | Number of GPUs per node for teacher serving. |
| `distillation.teacher_model.nnodes` | `NNODES` | teacher 节点数。 | Number of teacher nodes. |
| `distillation.teacher_model.inference.name` | `ROLLOUT_NAME` | teacher 推理后端。 | Teacher inference backend. |
| `distillation.teacher_model.inference.tensor_model_parallel_size` | `TEACHER_TP` | teacher TP 大小。 | Teacher tensor-parallel size. |
| `distillation.teacher_model.inference.temperature` | `1.0` | teacher 温度，通常设为 1 保持 logprob 稳定。 | Teacher temperature, usually fixed at 1.0 for stable logprobs. |
| `distillation.teacher_model.inference.gpu_memory_utilization` | `TEACHER_GPU_MEMORY_UTILIZATION` | teacher 显存利用率目标。 | Teacher GPU memory utilization target. |
| `distillation.teacher_model.inference.enforce_eager` | `True` | teacher 使用 eager 模式。 | Force eager mode for teacher inference. |
| `distillation.teacher_model.inference.max_model_len` | `TEACHER_MAX_MODEL_LEN` | teacher 最大上下文长度。 | Teacher max context length. |
| `distillation.teacher_model.inference.max_num_batched_tokens` | `TEACHER_MAX_BATCHED_TOKENS` | teacher batched token 上限。 | Teacher max batched tokens. |
| `distillation.teacher_model.inference.max_num_seqs` | `TRAIN_BATCH_SIZE` | teacher 一次最多处理多少序列。 | Max number of teacher sequences per scheduling step. |
| `distillation.distillation_loss.loss_mode` | `forward_kl_topk` | 蒸馏损失类型。 | Distillation loss mode. |
| `distillation.distillation_loss.topk` | `128` | teacher top-k logprobs 数量。 | Number of teacher top-k logprobs used by distillation. |
| `distillation.distillation_loss.use_task_rewards` | `False` | 蒸馏损失中不再额外叠加 task reward。 | Do not add task rewards into the distillation loss. |
| `distillation.distillation_loss.use_policy_gradient` | `False` | 关闭 policy gradient 项。 | Disable policy gradient term in the distillation loss. |
| `distillation.distillation_loss.loss_max_clamp` | `0.05` | 蒸馏损失上限裁剪值。 | Upper clamp for distillation loss. |
| `distillation.distillation_loss.log_prob_min_clamp` | `-10.0` | logprob 下限裁剪值。 | Lower clamp for log probability values. |

### 11.9 Rubric Config In Hydra / Hydra 中的 rubric 参数

| Hydra key | Source | 中文说明 | English meaning |
|---|---:|---|---|
| `+opsd_rubric.enabled` | `OPSD_RUBRIC_ENABLED` | 启用 rubric 模块。 | Enable rubric module. |
| `+opsd_rubric.warmup_steps` | `OPSD_RUBRIC_WARMUP_STEPS` | rubric warmup 步数。 | Rubric warmup steps. |
| `+opsd_rubric.mix_steps` | `OPSD_RUBRIC_MIX_STEPS` | rubric mix 阶段步数。 | Rubric curriculum mix steps. |
| `+opsd_rubric.seed` | `OPSD_RUBRIC_SEED` | rubric 相关随机种子。 | Rubric random seed. |
| `+opsd_rubric.min_response_chars` | `OPSD_RUBRIC_MIN_RESPONSE_CHARS` | 用于 rubric mining 的最小回答长度。 | Minimum response length used for rubric mining. |
| `+opsd_rubric.max_pending_requests` | `OPSD_RUBRIC_MAX_PENDING_REQUESTS` | rubric 队列最大积压。 | Maximum pending rubric update requests. |

### 11.10 Trainer / 训练器参数

| Hydra key | Source / Value | 中文说明 | English meaning |
|---|---:|---|---|
| `trainer.use_legacy_worker_impl` | `disable` | 使用新版 worker 实现。 | Disable legacy worker implementation. |
| `trainer.val_before_train` | `VAL_BEFORE_TRAIN` | 是否先验证再训练。 | Validate before training starts. |
| `trainer.critic_warmup` | `0` | critic warmup 步数；当前 critic 关闭，因此基本无作用。 | Critic warmup steps; mostly irrelevant here because critic is disabled. |
| `trainer.logger` | `WANDB_LOGGER` | logger 列表。 | Active logger list. |
| `trainer.log_val_generations` | `LOG_VAL_GENERATIONS` | 验证回答表记录条数。 | Number of validation generations to log. |
| `trainer.log_val_error_generations` | `LOG_VAL_ERROR_GENERATIONS` | 验证错答表记录条数。 | Number of incorrect validation generations to log. |
| `trainer.project_name` | `PROJECT_NAME` | 项目名。 | Trainer project name. |
| `trainer.experiment_name` | `EXPERIMENT_NAME` | 实验名。 | Trainer experiment name. |
| `trainer.default_local_dir` | `OUTPUT_DIR` | 本地输出根目录。 | Local output root directory. |
| `trainer.rollout_data_dir` | `ROLLOUT_DATA_DIR` | rollout 全量回答落盘目录。 | Directory for full rollout dumps. |
| `trainer.rollout_error_data_dir` | `ROLLOUT_ERROR_DATA_DIR` | rollout 错答落盘目录。 | Directory for incorrect rollout dumps. |
| `trainer.n_gpus_per_node` | `NUM_GPUS` | 每节点 GPU 数。 | Number of GPUs per node. |
| `trainer.nnodes` | `NNODES` | 节点数。 | Number of nodes. |
| `trainer.save_freq` | `SAVE_FREQ` | checkpoint 保存频率。 | Checkpoint save frequency. |
| `trainer.test_freq` | `TEST_FREQ` | 验证频率。 | Validation frequency. |
| `trainer.total_epochs` | `TOTAL_EPOCHS` | 总 epoch 数。 | Total epoch count. |
| `trainer.resume_mode` | `disable` | 禁用自动续训。 | Disable automatic resume. |

## 12. Important Notes / 重要说明

### 12.1 Validation `@k` Metrics / 验证 `@k` 指标

- 中文：`mean@k`、`best@k` 这类指标是否出现，取决于验证时 `actor_rollout_ref.rollout.val_kwargs.n`。如果 `n=1`，只会得到 `mean@1`。  
- English: Whether metrics like `mean@k` and `best@k` appear depends on `actor_rollout_ref.rollout.val_kwargs.n` during validation. If `n=1`, you only get `mean@1`.

### 12.2 Current Reward Semantics / 当前 reward 语义

- 中文：对 `math_dapo`，`score = 1.0` 表示判对，`score = -1.0` 表示判错；这不是 `0~1` 分数。  
- English: For `math_dapo`, `score = 1.0` means correct and `score = -1.0` means incorrect; it is not a `0~1` score.

### 12.3 Current Output Length Risk / 当前输出长度风险

- 中文：如果 `MAX_RESPONSE_LENGTH` 太小，模型可能在给出最终答案前被截断，从而被 `math_dapo` 判成 `[INVALID]` 或错误。  
- English: If `MAX_RESPONSE_LENGTH` is too small, generations may be truncated before the final answer appears, causing `math_dapo` to mark them as `[INVALID]` or incorrect.

### 12.4 Prompt And Verifier Alignment / Prompt 与判分器的一致性

- 中文：当前 student prompt 要求把最终答案放进 `\boxed{}`，但 `math_dapo` 默认分支使用的是 `is_correct_minerva` 风格提取器；如果格式不对齐，分数会系统性偏低。  
- English: The student prompt asks for the final answer in `\boxed{}`, while the default `math_dapo` branch uses a Minerva-style answer extractor. If the formats are misaligned, scores can become systematically low.

## 13. Suggested Reading / 建议联读文档

- [`uv_verl_qwen3_4b.md`](/root/MRTOPSD/docs/uv_verl_qwen3_4b.md)
- [`verl_training_logic.md`](/root/MRTOPSD/docs/verl_training_logic.md)
- [`verl_framework_full_flow.md`](/root/MRTOPSD/docs/verl_framework_full_flow.md)


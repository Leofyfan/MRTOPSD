#!/usr/bin/env bash
set -xeuo pipefail

############################ Quick Config ############################

ROOT_DIR=/home/yuanfan/projects/MRTOPSD
ENV_DIR=/home/yuanfan/envs/uv_envs/MRTOPSD-ENV
PYTHON_BIN=${ENV_DIR}/bin/python
MODEL_DIR=/home/shenyl/hf/model/Qwen/Qwen3-1.7B

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export NCCL_P2P_DISABLE=1
# Avoid the cuMem NCCL path that is fragile on this 4x3090 topology.
# vLLM rollout/teacher servers set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
# on their own; exporting it globally breaks verl WorkerDict when
# CUDA_VISIBLE_DEVICES is a subset like 2,3.
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
export NCCL_CUMEM_ENABLE=0
export TOKENIZERS_PARALLELISM=false
# vLLM V1 sleep mode uses CuMemAllocator, which rejects expandable segments.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi
export VERL_USE_OPSD_TEACHER=1

ROLLOUT_NAME="vllm"
PROJECT_NAME="opsd_verl"

TRAIN_FILE="${ROOT_DIR}/data/train/opsd_train.parquet"
VAL_FILE="${ROOT_DIR}/data/processed_new/opsd_val.parquet"
AIME24_VAL_FILE="${ROOT_DIR}/data/benchmarks/aime24_official_eval.parquet"
AIME25_VAL_FILE="${ROOT_DIR}/data/benchmarks/aime25_official_eval.parquet"
HMMT25_VAL_FILE="${ROOT_DIR}/data/benchmarks/hmmt25_official_eval.parquet"
VAL_FILES="['${AIME24_VAL_FILE}','${AIME25_VAL_FILE}','${HMMT25_VAL_FILE}']"

MAX_PROMPT=4096
MAX_RESPONSE_LENGTH=4096
VAL_RESPONSE_LENGTH=${VAL_RESPONSE_LENGTH:-32768}
TRAIN_MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 256 ))
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$(( MAX_PROMPT + VAL_RESPONSE_LENGTH + 256 ))}
ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS:-${ROLLOUT_MAX_MODEL_LEN}}
VAL_MAX_TOKENS=${VAL_MAX_TOKENS:-${VAL_RESPONSE_LENGTH}}

TRAIN_PROMPT_BSZ=32
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
STUDENT_MICRO_BATCH_SIZE_PER_GPU=16
STUDENT_MAX_TOKEN_LEN_PER_GPU=${TRAIN_MAX_NUM_TOKENS}
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=2
TEACHER_RESOURCE_POOL=False
TEACHER_WORLD_SIZE=2

ROLLOUT_TP=1
ROLLOUT_DP=${ROLLOUT_DP:-1}
TEACHER_TP=1
ROLLOUT_MAX_NUM_SEQS=128
TEACHER_REQUIRED_CONTEXT_LEN=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
# OPSD teacher uses rubric/scoring prompt + student response, so it needs
# more context than the original student prompt + response budget.
OPSD_TEACHER_PROMPT_EXTRA_TOKENS=${OPSD_TEACHER_PROMPT_EXTRA_TOKENS:-$(( MAX_PROMPT / 2 ))}
TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-$(( TEACHER_REQUIRED_CONTEXT_LEN + OPSD_TEACHER_PROMPT_EXTRA_TOKENS + 2048 ))}
TEACHER_MAX_BATCHED_TOKENS=${TEACHER_MAX_BATCHED_TOKENS:-${TEACHER_MAX_MODEL_LEN}}
TEACHER_MAX_NUM_SEQS=1
VLLM_DTYPE="float16"

SAVE_FREQ=400
TEST_FREQ=10
TOTAL_EPOCHS=2
VAL_REPEAT_N=16

DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-"opsd_multi_forward_kl_topk"}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-False}
DISTILLATION_TOPK=128
DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

OPSD_RUBRIC_ENABLED=${OPSD_RUBRIC_ENABLED:-True}
OPSD_RUBRIC_WARMUP_STEPS=${OPSD_RUBRIC_WARMUP_STEPS:-0}
OPSD_RUBRIC_MIX_STEPS=${OPSD_RUBRIC_MIX_STEPS:-0}
OPSD_DYNAMIC_RUBRIC_ENABLED=${OPSD_DYNAMIC_RUBRIC_ENABLED:-True}
OPSD_DYNAMIC_RUBRIC_SELECTION_MODE=${OPSD_DYNAMIC_RUBRIC_SELECTION_MODE:-top_m}
OPSD_DYNAMIC_RUBRIC_MAX_PER_SAMPLE=${OPSD_DYNAMIC_RUBRIC_MAX_PER_SAMPLE:-3}
OPSD_DYNAMIC_RUBRIC_WEIGHT_MODE=${OPSD_DYNAMIC_RUBRIC_WEIGHT_MODE:-abs_points}
OPSD_DYNAMIC_RUBRIC_WEIGHT_TEMPERATURE=${OPSD_DYNAMIC_RUBRIC_WEIGHT_TEMPERATURE:-1.0}
OPSD_DYNAMIC_RUBRIC_INCLUDE_NEGATIVE=${OPSD_DYNAMIC_RUBRIC_INCLUDE_NEGATIVE:-True}
OPSD_DYNAMIC_RUBRIC_MINING_MAX_TOKENS=${OPSD_DYNAMIC_RUBRIC_MINING_MAX_TOKENS:-1024}


ROLLOUT_GPU_MEMORY_UTILIZATION=0.65
# Leave headroom for teacher-side prompt_logprobs over long official-eval sequences.
TEACHER_GPU_MEMORY_UTILIZATION=0.65

LORA_RANK=32
LORA_ALPHA=64

MODEL_TAG="$(basename "${MODEL_DIR}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g; s/_\\+/_/g; s/^_//; s/_$//')"
LOSS_TAG="$(echo "${DISTILLATION_LOSS_MODE}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g; s/_\\+/_/g; s/^_//; s/_$//')"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${MODEL_TAG}_dynamic_rubric_policy_gradient_${USE_POLICY_GRADIENT}_${DISTILLATION_TOPK}_sample_${OPSD_DYNAMIC_RUBRIC_MAX_PER_SAMPLE}_mode_${OPSD_DYNAMIC_RUBRIC_WEIGHT_MODE}_forwardkl_${LOSS_TAG}_official_eval_bs${TRAIN_PROMPT_BSZ}_mb${STUDENT_MICRO_BATCH_SIZE_PER_GPU}_ws${STUDENT_WORLD_SIZE}_seq${ROLLOUT_MAX_NUM_SEQS}_${RUN_TIMESTAMP}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${EXP_NAME}"

############################ Runtime Paths ############################

export PATH="${ENV_DIR}/bin:$PATH"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/third_party/verl:${PYTHONPATH:-}"
export HF_HOME="${ROOT_DIR}/eval/data/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export VLLM_CACHE_ROOT="${ROOT_DIR}/.cache/vllm"
export TRITON_CACHE_DIR="${ROOT_DIR}/.cache/triton"
export RAY_TMPDIR="${ROOT_DIR}/.r"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_CACHE_DIR="${ROOT_DIR}/.cache/wandb"
export WANDB_CONFIG_DIR="${ROOT_DIR}/.cache/wandb-config"
export WANDB_DATA_DIR="${ROOT_DIR}/.cache/wandb-data"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-leofyfan-east-china-normal-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-${PROJECT_NAME}}"
export WANDB_NAME="${WANDB_NAME:-${EXP_NAME}}"
export WANDB_API_KEY="${WANDB_API_KEY-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}"
export VERL_LOGGING_LEVEL=INFO
export VERL_ZMQ_NAMESPACE="u$(id -u)-$(date +%s)-official-eval"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

mkdir -p \
    "${OUTPUT_DIR}" \
    "${OUTPUT_DIR}/rollout_data" \
    "${HF_HOME}" \
    "${HF_DATASETS_CACHE}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${VLLM_CACHE_ROOT}" \
    "${TRITON_CACHE_DIR}" \
    "${RAY_TMPDIR}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}" \
    "${WANDB_CONFIG_DIR}" \
    "${WANDB_DATA_DIR}"

############################ Parameter Groups ############################

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILES}"
    data.max_prompt_length=${MAX_PROMPT}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.train_batch_size=${TRAIN_PROMPT_BSZ}
    # verl defaults val_batch_size to the full validation dataset, which is
    # catastrophic here because validation immediately repeats each sample N times.
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.filter_overlong_prompts=True
    data.dataloader_num_workers=4
    data.truncation=error
    data.shuffle=False
    data.return_raw_chat=True
    +data.apply_chat_template_kwargs.enable_thinking=False
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_DIR}"
    actor_rollout_ref.model.external_lib=verl_opsd.losses
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.enable_activation_offload=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'
)

STUDENT=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=5e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_BSZ}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${STUDENT_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${STUDENT_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${STUDENT_MAX_TOKEN_LEN_PER_GPU}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME}
    actor_rollout_ref.rollout.dtype=${VLLM_DTYPE}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.temperature=1.1
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.top_k=20
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.agent.num_workers=4
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${STUDENT_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_REPEAT_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    +actor_rollout_ref.rollout.val_kwargs.min_p=0.0
    +actor_rollout_ref.rollout.val_kwargs.presence_penalty=0.0
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=${VAL_MAX_TOKENS}
    +actor_rollout_ref.rollout.val_kwargs.response_length=${VAL_RESPONSE_LENGTH}
    +actor_rollout_ref.rollout.val_kwargs.enable_thinking=True
)

DISTILLATION=(
    distillation.enabled=True
    distillation.num_workers=1
    distillation.teacher_model.enable_resource_pool=${TEACHER_RESOURCE_POOL}
    distillation.teacher_model.model_path="${MODEL_DIR}"
    distillation.teacher_model.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.teacher_model.nnodes=1
    distillation.teacher_model.inference.name=${ROLLOUT_NAME}
    distillation.teacher_model.inference.dtype=${VLLM_DTYPE}
    distillation.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_model.inference.temperature=1.0
    distillation.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION}
    distillation.teacher_model.inference.enforce_eager=True
    distillation.teacher_model.inference.max_model_len=${TEACHER_MAX_MODEL_LEN}
    distillation.teacher_model.inference.max_num_batched_tokens=${TEACHER_MAX_BATCHED_TOKENS}
    distillation.teacher_model.inference.max_num_seqs=${TEACHER_MAX_NUM_SEQS}
    distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
    distillation.distillation_loss.topk=${DISTILLATION_TOPK}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
    distillation.distillation_loss.loss_max_clamp=${DISTILLATION_LOSS_MAX_CLAMP}
    distillation.distillation_loss.log_prob_min_clamp=${DISTILLATION_LOG_PROB_MIN_CLAMP}
)

RUBRIC=(
    +opsd_rubric.enabled=${OPSD_RUBRIC_ENABLED}
    +opsd_rubric.warmup_steps=${OPSD_RUBRIC_WARMUP_STEPS}
    +opsd_rubric.mix_steps=${OPSD_RUBRIC_MIX_STEPS}
    +opsd_rubric.seed=0
    +opsd_rubric.min_response_chars=32
    +opsd_rubric.max_pending_requests=128
    +opsd_rubric.dynamic.enabled=${OPSD_DYNAMIC_RUBRIC_ENABLED}
    +opsd_rubric.dynamic.selection_mode=${OPSD_DYNAMIC_RUBRIC_SELECTION_MODE}
    +opsd_rubric.dynamic.max_rubrics_per_sample=${OPSD_DYNAMIC_RUBRIC_MAX_PER_SAMPLE}
    +opsd_rubric.dynamic.weight_mode=${OPSD_DYNAMIC_RUBRIC_WEIGHT_MODE}
    +opsd_rubric.dynamic.weight_temperature=${OPSD_DYNAMIC_RUBRIC_WEIGHT_TEMPERATURE}
    +opsd_rubric.dynamic.include_negative=${OPSD_DYNAMIC_RUBRIC_INCLUDE_NEGATIVE}
    +opsd_rubric.dynamic.mining_max_tokens=${OPSD_DYNAMIC_RUBRIC_MINING_MAX_TOKENS}
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXP_NAME}
    trainer.default_local_dir=${OUTPUT_DIR}
    trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data
    trainer.n_gpus_per_node=${STUDENT_WORLD_SIZE}
    trainer.nnodes=1
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.val_before_train=True
    trainer.use_legacy_worker_impl=disable
    trainer.critic_warmup=0
    trainer.resume_mode=disable
    trainer.log_val_generations=0
    +trainer.log_val_error_generations=0
)

RUNTIME=(
    reward.num_workers=8
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_ENABLE='0'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_OPSD_TEACHER='${VERL_USE_OPSD_TEACHER}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL='INFO'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${ROOT_DIR}:${ROOT_DIR}/third_party/verl'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE='${WANDB_MODE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY='${WANDB_ENTITY}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_PROJECT='${WANDB_PROJECT}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_NAME='${WANDB_NAME}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY='${WANDB_API_KEY}'"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD='spawn'"
)

############################ Launch ############################

"${PYTHON_BIN}" -m verl_opsd.main_ppo_opsd \
    --config-path="${ROOT_DIR}/third_party/verl/verl/trainer/config" \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${RUBRIC[@]}" \
    "${TRAINER[@]}" \
    "${RUNTIME[@]}" \
    "$@"

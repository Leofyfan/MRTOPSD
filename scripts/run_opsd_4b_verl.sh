#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-/root/autodl-tmp/MRTOPSD-ENV}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/Qwen3-4B}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT_DIR}/data/processed/opsd_train.parquet}"
VAL_FILE="${VAL_FILE:-${ROOT_DIR}/data/processed/opsd_val.parquet}"
PROJECT_NAME="${PROJECT_NAME:-opsd_verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_opsd_verl}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/opsd_verl_runs/${EXPERIMENT_NAME}}"

NUM_GPUS="${NUM_GPUS:-2}"
NNODES="${NNODES:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
TEACHER_TP="${TEACHER_TP:-1}"
DEFAULT_WORKER_COUNT="$(( NUM_GPUS * 8 ))"
DEFAULT_TRAIN_BATCH_SIZE="$(( NUM_GPUS * 32 ))"
DEFAULT_MICRO_BATCH_SIZE="$(( NUM_GPUS * 2 ))"

ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-${DEFAULT_WORKER_COUNT}}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-${DEFAULT_WORKER_COUNT}}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-${DEFAULT_WORKER_COUNT}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${DEFAULT_MICRO_BATCH_SIZE}}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:--1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-6144}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-3072}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-49152}"
TEACHER_MAX_BATCHED_TOKENS="${TEACHER_MAX_BATCHED_TOKENS:-65536}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.40}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
SAVE_FREQ="${SAVE_FREQ:-200}"
TEST_FREQ="${TEST_FREQ:-200}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
WANDB_LOGGER="${WANDB_LOGGER:-[\"console\",\"wandb\"]}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-16}"
LOG_VAL_ERROR_GENERATIONS="${LOG_VAL_ERROR_GENERATIONS:-16}"
VAL_REPEAT_N="${VAL_REPEAT_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
VAL_TOP_K="${VAL_TOP_K:--1}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-False}"
TEACHER_ENABLE_RESOURCE_POOL="${TEACHER_ENABLE_RESOURCE_POOL:-False}"
OPSD_RUBRIC_ENABLED="${OPSD_RUBRIC_ENABLED:-true}"
OPSD_RUBRIC_WARMUP_STEPS="${OPSD_RUBRIC_WARMUP_STEPS:-100}"
OPSD_RUBRIC_MIX_STEPS="${OPSD_RUBRIC_MIX_STEPS:-300}"
OPSD_RUBRIC_SEED="${OPSD_RUBRIC_SEED:-0}"
OPSD_RUBRIC_MIN_RESPONSE_CHARS="${OPSD_RUBRIC_MIN_RESPONSE_CHARS:-32}"
OPSD_RUBRIC_MAX_PENDING_REQUESTS="${OPSD_RUBRIC_MAX_PENDING_REQUESTS:-128}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollout_data}"
ROLLOUT_ERROR_DATA_DIR="${ROLLOUT_ERROR_DATA_DIR:-${OUTPUT_DIR}/rollout_error_data}"

parquet_num_rows() {
  local parquet_path="$1"
  "${ENV_DIR}/bin/python" - "$parquet_path" <<'PY'
import sys
from datasets import load_dataset

path = sys.argv[1]
dataset = load_dataset("parquet", data_files=path)["train"]
print(len(dataset))
PY
}

parquet_prompt_format_version() {
  local parquet_path="$1"
  "${ENV_DIR}/bin/python" - "$parquet_path" <<'PY'
import sys
from datasets import load_dataset

path = sys.argv[1]
dataset = load_dataset("parquet", data_files=path)["train"]
if len(dataset) == 0:
    print("")
else:
    row = dataset[0]
    extra_info = row.get("extra_info") or {}
    print(extra_info.get("prompt_format_version", ""))
PY
}

prepare_dataset_if_needed() {
  local train_parent
  local val_parent
  train_parent="$(dirname "${TRAIN_FILE}")"
  val_parent="$(dirname "${VAL_FILE}")"

  if [[ "${train_parent}" != "${val_parent}" ]]; then
    echo "TRAIN_FILE and VAL_FILE must share the same parent directory for auto-prepare." >&2
    return 1
  fi

  bash "${ROOT_DIR}/scripts/prepare_opsd_verl_data.sh" \
    --output-dir "${train_parent}" \
    --train-file "$(basename "${TRAIN_FILE}")" \
    --val-file "$(basename "${VAL_FILE}")" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --max-student-prompt-length "${MAX_PROMPT_LENGTH}" \
    --max-teacher-prompt-length "$((TEACHER_MAX_MODEL_LEN - MAX_RESPONSE_LENGTH))"
}

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
  prepare_dataset_if_needed
else
  TRAIN_ROWS="$(parquet_num_rows "${TRAIN_FILE}")"
  VAL_ROWS="$(parquet_num_rows "${VAL_FILE}")"
  PROMPT_FORMAT_VERSION="$(parquet_prompt_format_version "${TRAIN_FILE}")"
  if (( TRAIN_ROWS < TRAIN_BATCH_SIZE || VAL_ROWS < 1 )); then
    echo "Detected undersized parquet data: train_rows=${TRAIN_ROWS}, val_rows=${VAL_ROWS}, train_batch_size=${TRAIN_BATCH_SIZE}. Rebuilding dataset." >&2
    prepare_dataset_if_needed
  elif [[ "${PROMPT_FORMAT_VERSION}" != "opsd_boxed_last_line_v2" ]]; then
    echo "Detected outdated prompt format in parquet data (${PROMPT_FORMAT_VERSION:-missing}). Rebuilding dataset." >&2
    prepare_dataset_if_needed
  fi
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/wandb"
mkdir -p /root/autodl-tmp/.cache/huggingface
mkdir -p /root/autodl-tmp/.cache/vllm
mkdir -p /root/autodl-tmp/.cache/ray
mkdir -p /root/autodl-tmp/.cache/triton
mkdir -p /root/autodl-tmp/.cache/wandb
mkdir -p /root/autodl-tmp/.config/wandb
mkdir -p /root/autodl-tmp/.local/share/wandb

export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/.cache/huggingface/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/root/autodl-tmp/.cache/vllm}"
export RAY_TMPDIR="${RAY_TMPDIR:-/root/autodl-tmp/.cache/ray}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/.cache/triton}"
export WANDB_ENTITY="${WANDB_ENTITY:-leofyfan-east-china-normal-university}"
export WANDB_USERNAME="${WANDB_USERNAME:-leofyfan}"
export WANDB_PROJECT="${WANDB_PROJECT:-${PROJECT_NAME}}"
export WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/root/autodl-tmp/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/root/autodl-tmp/.config/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/root/autodl-tmp/.local/share/wandb}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export VERL_USE_OPSD_TEACHER="${VERL_USE_OPSD_TEACHER:-1}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/third_party/verl:${PYTHONPATH:-}"

TARGET_MODULES='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'

CMD=(
  "${ENV_DIR}/bin/python" -m verl_opsd.main_ppo_opsd
  --config-path="${ROOT_DIR}/third_party/verl/verl/trainer/config"
  --config-name=ppo_trainer.yaml
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
  "data.truncation=error"
  "data.shuffle=False"
  "data.return_raw_chat=True"
  "+data.apply_chat_template_kwargs.enable_thinking=False"
  "reward.num_workers=${REWARD_NUM_WORKERS}"
  "actor_rollout_ref.model.path=${MODEL_DIR}"
  "actor_rollout_ref.model.trust_remote_code=True"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.model.lora_rank=64"
  "actor_rollout_ref.model.lora_alpha=128"
  "actor_rollout_ref.model.target_modules=${TARGET_MODULES}"
  "actor_rollout_ref.actor.strategy=fsdp2"
  "actor_rollout_ref.actor.optim.lr=5e-6"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16"
  "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
  "actor_rollout_ref.rollout.name=${ROLLOUT_NAME}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
  "actor_rollout_ref.rollout.n=1"
  "actor_rollout_ref.rollout.temperature=1.1"
  "actor_rollout_ref.rollout.top_p=0.95"
  "actor_rollout_ref.rollout.top_k=20"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.enforce_eager=True"
  "actor_rollout_ref.rollout.load_format=safetensors"
  "actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}"
  "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.max_num_seqs=${TRAIN_BATCH_SIZE}"
  "actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS}"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_REPEAT_N}"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}"
  "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
  "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
  "actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}"
  "distillation.enabled=True"
  "distillation.num_workers=${NUM_GPUS}"
  "distillation.teacher_model.enable_resource_pool=${TEACHER_ENABLE_RESOURCE_POOL}"
  "distillation.teacher_model.model_path=${MODEL_DIR}"
  "distillation.teacher_model.n_gpus_per_node=${NUM_GPUS}"
  "distillation.teacher_model.nnodes=${NNODES}"
  "distillation.teacher_model.inference.name=${ROLLOUT_NAME}"
  "distillation.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}"
  "distillation.teacher_model.inference.temperature=1.0"
  "distillation.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION}"
  "distillation.teacher_model.inference.enforce_eager=True"
  "distillation.teacher_model.inference.max_model_len=${TEACHER_MAX_MODEL_LEN}"
  "distillation.teacher_model.inference.max_num_batched_tokens=${TEACHER_MAX_BATCHED_TOKENS}"
  "distillation.teacher_model.inference.max_num_seqs=${TRAIN_BATCH_SIZE}"
  "distillation.distillation_loss.loss_mode=forward_kl_topk"
  "distillation.distillation_loss.topk=128"
  "distillation.distillation_loss.use_task_rewards=False"
  "distillation.distillation_loss.use_policy_gradient=False"
  "distillation.distillation_loss.loss_max_clamp=0.05"
  "distillation.distillation_loss.log_prob_min_clamp=-10.0"
  "+opsd_rubric.enabled=${OPSD_RUBRIC_ENABLED}"
  "+opsd_rubric.warmup_steps=${OPSD_RUBRIC_WARMUP_STEPS}"
  "+opsd_rubric.mix_steps=${OPSD_RUBRIC_MIX_STEPS}"
  "+opsd_rubric.seed=${OPSD_RUBRIC_SEED}"
  "+opsd_rubric.min_response_chars=${OPSD_RUBRIC_MIN_RESPONSE_CHARS}"
  "+opsd_rubric.max_pending_requests=${OPSD_RUBRIC_MAX_PENDING_REQUESTS}"
  "trainer.use_legacy_worker_impl=disable"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
  "trainer.critic_warmup=0"
  "trainer.logger=${WANDB_LOGGER}"
  "trainer.log_val_generations=${LOG_VAL_GENERATIONS}"
  "trainer.log_val_error_generations=${LOG_VAL_ERROR_GENERATIONS}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.default_local_dir=${OUTPUT_DIR}"
  "trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}"
  "trainer.rollout_error_data_dir=${ROLLOUT_ERROR_DATA_DIR}"
  "trainer.n_gpus_per_node=${NUM_GPUS}"
  "trainer.nnodes=${NNODES}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.resume_mode=disable"
)

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"

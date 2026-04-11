#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_opsd_verl}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/opsd_verl_runs/${EXPERIMENT_NAME}}"
SESSION_NAME="${SESSION_NAME:-${EXPERIMENT_NAME}}"
SCREEN_LOG_FILE="${SCREEN_LOG_FILE:-${OUTPUT_DIR}/screen.log}"

mkdir -p "${OUTPUT_DIR}"

if screen -ls | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
  echo "screen session already exists: ${SESSION_NAME}" >&2
  echo "attach with: screen -r ${SESSION_NAME}" >&2
  exit 1
fi

screen -dmS "${SESSION_NAME}" bash -lc "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/run_opsd_4b_verl.sh' 2>&1 | tee -a '${SCREEN_LOG_FILE}'"

echo "started screen session: ${SESSION_NAME}"
echo "attach: screen -r ${SESSION_NAME}"
echo "log: ${SCREEN_LOG_FILE}"

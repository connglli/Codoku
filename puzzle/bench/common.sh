#!/usr/bin/env bash
# -----------------------------------------------------------------------
# common.sh — Shared utility logic for RefractIR agent runners.
# -----------------------------------------------------------------------
# Sourced by agent runner scripts (e.g. claude.sh, opencode.sh) to avoid
# duplicating the parsing, paths, templating, and timeout orchestration.
# -----------------------------------------------------------------------
set -euo pipefail

# — Parse options ————————————————————————————————————————————————————————
PUZZLE_DIR="${1:?usage: ${0##*/} <puzzle-dir> --model MODEL [options]}"
shift

MODEL=""
TIMEOUT=0         # 0 = unlimited
MAX_TURNS=0       # 0 = unlimited
MAX_BUDGET_USD=0  # 0 = unlimited

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2";          shift 2 ;;
    --timeout)        TIMEOUT="$2";        shift 2 ;;
    --max-turns)      MAX_TURNS="$2";      shift 2 ;;
    --max-budget-usd) MAX_BUDGET_USD="$2"; shift 2 ;;
    *)                echo "${0##*/}: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "${MODEL}" ]; then
  echo "${0##*/}: --model is required" >&2
  exit 2
fi

# — Paths ————————————————————————————————————————————————————————————————
PUZZLE_FILE="${PUZZLE_DIR}/puzzle.sir"
SOLUTION_FILE="${PUZZLE_DIR}/solution.sir"
WORKSPACE="${PUZZLE_DIR}/workspace"
TRAJECTORY="${PUZZLE_DIR}/trajectory.jsonl"
SYSTEM_MD="${PUZZLE_DIR}/system.md"
STDERR_LOG="${PUZZLE_DIR}/stderr.log"

mkdir -p "${WORKSPACE}" "${PUZZLE_DIR}/cache"

# — Process system.md template ———————————————————————————————————————————
# The template (mounted at /opt/rypuz/system.md) contains placeholders
# like {{PUZZLE_FILE}} that we resolve to this puzzle's concrete paths.
sed -e "s|{{PUZZLE_FILE}}|${PUZZLE_FILE}|g"     \
    -e "s|{{SOLUTION_FILE}}|${SOLUTION_FILE}|g" \
    -e "s|{{WORKSPACE}}|${WORKSPACE}|g"         \
    /opt/rypuz/system.md > "${SYSTEM_MD}"

# — Helper to run a command with optional timeout —————————————————————————
run_agent() {
  local cmd=("$@")
  cd "${PUZZLE_DIR}"

  if [ "${TIMEOUT}" -gt 0 ] 2>/dev/null; then
    timeout --signal TERM --kill-after 30 "${TIMEOUT}" \
      "${cmd[@]}" 2>"${STDERR_LOG}" | tee "${TRAJECTORY}"
  else
    "${cmd[@]}" 2>"${STDERR_LOG}" | tee "${TRAJECTORY}"
  fi
}

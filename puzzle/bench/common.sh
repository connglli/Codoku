#!/usr/bin/env bash
# -----------------------------------------------------------------------
# common.sh — Shared utility logic for puzzle agent runners.
# -----------------------------------------------------------------------
# Unified agent interface (all agents must follow this contract):
#
#   <agent>.sh <puzzle-dir> --model MODEL  [--timeout S]
#               [--max-turns N] [--max-budget-usd F]
#
# Inputs (inside <puzzle-dir>):
#   puzzle.<ext>        The <ext> puzzle to solve
#   tools/              Symlinks to puzzle toolchain (rypuzchk, etc.)
#   workspace/          Writable scratch space
#   system.md           Instructions (pre-processed with runtime paths)
#
# Outputs (agent must produce):
#   solution.<ext>      The solved puzzle (if successful)
#   trajectory.jsonl    Raw agent output / log
#   agent_cache/        Agent session / cache data
#
# Sourced by agent runner scripts (e.g. claude.sh, opencode.sh) to avoid
# duplicating the parsing, paths, templating, and timeout orchestration.
# See ./claude.sh and ./opencode.sh for examples of how to use this common.sh.
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
if [ -f "${PUZZLE_DIR}/puzzle.py" ]; then
  PUZZLE_FILE="${PUZZLE_DIR}/puzzle.py"
  SOLUTION_FILE="${PUZZLE_DIR}/solution.py"
  TEMPLATE_FILE="/opt/rypuz/puzzle/system_python.md"
elif [ -f "${PUZZLE_DIR}/puzzle.sir" ]; then
  PUZZLE_FILE="${PUZZLE_DIR}/puzzle.sir"
  SOLUTION_FILE="${PUZZLE_DIR}/solution.sir"
  TEMPLATE_FILE="/opt/rypuz/puzzle/system_sir.md"
else
  PUZZLE_FILE="${PUZZLE_DIR}/puzzle.c"
  SOLUTION_FILE="${PUZZLE_DIR}/solution.c"
  TEMPLATE_FILE="/opt/rypuz/puzzle/system_c.md"
fi
WORKSPACE="${PUZZLE_DIR}/workspace"
TRAJECTORY="${PUZZLE_DIR}/trajectory.jsonl"
SYSTEM_MD="${PUZZLE_DIR}/system.md"
CACHE_DIR="${PUZZLE_DIR}/agent_cache"

mkdir -p "${WORKSPACE}"

# — Process system.md template ———————————————————————————————————————————
# The template contains placeholders like {{PUZZLE_FILE}} that we resolve to this puzzle's concrete paths.
sed -e "s|{{PUZZLE_FILE}}|${PUZZLE_FILE}|g"     \
    -e "s|{{SOLUTION_FILE}}|${SOLUTION_FILE}|g" \
    -e "s|{{WORKSPACE}}|${WORKSPACE}|g"         \
    "${TEMPLATE_FILE}" > "${SYSTEM_MD}"

# — Helper to run a command with optional timeout —————————————————————————
run_agent() {
  local cmd=("$@")
  cd "${PUZZLE_DIR}"

  if [ "${TIMEOUT}" -gt 0 ] 2>/dev/null; then
    timeout --signal TERM --kill-after 30 "${TIMEOUT}" \
      "${cmd[@]}" | tee "${TRAJECTORY}"
  else
    "${cmd[@]}" | tee "${TRAJECTORY}"
  fi
}

save_cache() {
  local src_dir="$1"
  local dest_dir="${CACHE_DIR}"
  if [ -d "${src_dir}" ]; then
    cp -r "${src_dir}" "${dest_dir}" 2>/dev/null || true
  else
    echo "Warning: the agent cache directory ${src_dir} does not exist." >&2
    mkdir -p "${dest_dir}"
  fi
}

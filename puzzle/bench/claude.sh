#!/usr/bin/env bash
# -----------------------------------------------------------------------
# claude.sh — Run Claude Code on a RefractIR puzzle.
# -----------------------------------------------------------------------
#
# Unified agent interface (all agents must follow this contract):
#
#   <agent>.sh <puzzle-dir> --model MODEL  [--timeout S]
#               [--max-turns N] [--max-budget-usd F]
#
# Inputs (inside <puzzle-dir>):
#   puzzle.sir          The puzzle to solve
#   tools/              Symlinks to RefractIR tools + SMT solvers
#   references/         Symlinks to reference documentation
#   workspace/          Writable scratch space
#   system.md           Instructions (pre-processed with runtime paths)
#
# Outputs (agent must produce):
#   solution.sir        The solved puzzle (if successful)
#   trajectory.json     Raw agent output / log
#   cache/              Agent session / cache data
# -----------------------------------------------------------------------
set -euo pipefail

# — Source common setup —————————————————————————————————————————————————
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh"

SYSTEM_PROMPT="$(cat "${SYSTEM_MD}")"

# — Build Claude CLI args ————————————————————————————————————————————————
CLAUDE_ARGS=(
  --print
  --verbose
  --model "${MODEL}"
  --output-format stream-json
  --system-prompt "${SYSTEM_PROMPT}"
  --dangerously-skip-permissions
)

# Conditional flags (0 / empty = omit = unlimited)
[ "${MAX_TURNS}" -gt 0 ]       2>/dev/null && CLAUDE_ARGS+=(--max-turns "${MAX_TURNS}")
[ "${MAX_BUDGET_USD}" -gt 0 ]  2>/dev/null && CLAUDE_ARGS+=(--max-budget-usd "${MAX_BUDGET_USD}")

USER_PROMPT="Solve the puzzle."

# — Run ——————————————————————————————————————————————————————————————————
run_agent claude "${CLAUDE_ARGS[@]}" "${USER_PROMPT}"
EXIT_CODE=$?

# — Save cache ———————————————————————————————————————————————————————————
# Claude Code stores session data under ~/.claude/.  Copy whatever is
# there into the puzzle's cache/ directory for post-mortem analysis.
if [ -d "${HOME}/.claude" ]; then
  cp -r "${HOME}/.claude/." "${PUZZLE_DIR}/cache/" 2>/dev/null || true
fi

exit ${EXIT_CODE}

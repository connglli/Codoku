#!/usr/bin/env bash
# -----------------------------------------------------------------------
# claude.sh — Run Claude Code on a C puzzle.
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

# Set default Claude model environment variables
export ANTHROPIC_DEFAULT_OPUS_MODEL="${MODEL}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${MODEL}"
export CLAUDE_CODE_SUBAGENT_MODEL="${MODEL}"

USER_PROMPT="Solve the puzzle."

# — Run ——————————————————————————————————————————————————————————————————
run_agent claude "${CLAUDE_ARGS[@]}" "${USER_PROMPT}"
EXIT_CODE=$?

# — Save cache ———————————————————————————————————————————————————————————
# Claude Code stores session data under ~/.claude/.
save_cache "${HOME}/.claude"

exit ${EXIT_CODE}

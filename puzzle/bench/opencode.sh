#!/usr/bin/env bash
# -----------------------------------------------------------------------
# opencode.sh — Run OpenCode on a RefractIR puzzle.
# -----------------------------------------------------------------------
set -euo pipefail

# — Source common setup —————————————————————————————————————————————————
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh"

AGENTS_MD="${PUZZLE_DIR}/AGENTS.md"
cp "${SYSTEM_MD}" "${AGENTS_MD}"

# — Build OpenCode CLI args ——————————————————————————————————————————————
OPENCODE_ARGS=(
  run
  "Solve the puzzle."
  --model "${MODEL}"
  --auto
  --format json
)

# OpenCode does not support
# - MAX_TURNS
# - MAX_BUDGET_USD

# — Run ——————————————————————————————————————————————————————————————————
run_agent opencode "${OPENCODE_ARGS[@]}"
EXIT_CODE=$?

# — Save cache ———————————————————————————————————————————————————————————
# OpenCode stores session data under ~/.local/share/opencode/.
# Copy whatever is there into the puzzle's cache/ directory.
if [ -d "${HOME}/.local/share/opencode" ]; then
  cp -r "${HOME}/.local/share/opencode" "${PUZZLE_DIR}/cache/" 2>/dev/null || true
fi

exit ${EXIT_CODE}

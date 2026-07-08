#!/usr/bin/env bash
#
# agent.sh — Agent launcher script for the rypuz-c benchmark sandbox.
#
# Prepares the tools directory inside the puzzle directory and executes the agent.
# Note: This runs in the environment activated by entrypoint.sh.

set -euo pipefail

AGENT_SCRIPT="${1:?usage: agent.sh <agent-script> <puzzle-dir> [agent-args...]}"
PUZZLE_DIR="${2:?usage: agent.sh <agent-script> <puzzle-dir> [agent-args...]}"
shift 2

# — Set up a tools/ directory with the C puzzle scripts —Target-oriented
# We create a directory in the puzzle dir and symlink each tool in.
TOOLS_DIR="${PUZZLE_DIR}/tools"
mkdir -p "${TOOLS_DIR}"

for tool in rypuzchk-c puzzle_common.py; do
  ln -sf "/opt/rypuz/bin/${tool}" "${TOOLS_DIR}/${tool}"
done

# SMT solvers (installed to /usr/local/bin during image build)
for solver in z3 cvc5 bitwuzla; do
  bin="$(command -v "$solver" 2>/dev/null || true)"
  if [ -n "$bin" ]; then
    ln -sf "$bin" "${TOOLS_DIR}/${solver}"
  fi
done

# — Execute the agent ————————————————————————————————————————————————————
exec bash "${AGENT_SCRIPT}" "${PUZZLE_DIR}" "$@"

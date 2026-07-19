#!/usr/bin/env bash
#
# entrypoint.sh — Docker entrypoint for the rypuz benchmark sandbox.
#
# Invoked by the Docker container with:
#   entrypoint.sh <agent-script> <puzzle-dir> [agent-args...]
#
# Responsibilities:
#   1. Source the global environment from rypuzbench.env.
#   2. Provision tools/ and references/ inside the puzzle directory.
#   3. Exec the agent script.
#
set -euo pipefail

AGENT_SCRIPT="${1:?usage: entrypoint.sh <agent-script> <puzzle-dir> [agent-args...]}"
PUZZLE_DIR="${2:?usage: entrypoint.sh <agent-script> <puzzle-dir> [agent-args...]}"
shift 2

# — Source global environment ————————————————————————————————————————————
# rypuzbench.env is mounted read-only by run.py at /opt/rypuz/rypuzbench.env.
# `set -a` exports every variable so child processes (the agent, Claude, etc.) inherit them.
if [ -f /opt/rypuz/rypuzbench.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /opt/rypuz/rypuzbench.env
  set +a
fi

# — Set up a tools/ directory combining RefractIR + SMT solvers ——————————
# The puzzle banner refers to ./tools/rypuzchk, ./tools/symiri, etc.
# We create a real directory in the puzzle dir and symlink each tool in,
# because /opt/refractir/tools is image-owned and may not be writable.
TOOLS_DIR="${PUZZLE_DIR}/tools"
mkdir -p "${TOOLS_DIR}"

# RefractIR tools
for tool in symiri symirc symirsolve rypuzchk; do
  ln -sf "/opt/rypuz/tools/${tool}" "${TOOLS_DIR}/${tool}"
done

# SMT solvers (installed to /usr/local/bin during image build)
for solver in z3 cvc5 bitwuzla; do
  bin="$(command -v "$solver" 2>/dev/null || true)"
  if [ -n "$bin" ]; then
    ln -sf "$bin" "${TOOLS_DIR}/${solver}"
  fi
done

# — Set up references/ symlink ———————————————————————————————————————————
ln -sfn /opt/rypuz/references "${PUZZLE_DIR}/references"

# — Set up a virtual Python environment ————————————————————————————
python3 -m venv "${PUZZLE_DIR}/venv"
source "${PUZZLE_DIR}/venv/bin/activate"

# — Execute the agent ————————————————————————————————————————————————————
exec bash "${AGENT_SCRIPT}" "${PUZZLE_DIR}" "$@"

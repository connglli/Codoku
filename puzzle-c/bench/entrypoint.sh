#!/usr/bin/env bash
#
# entrypoint.sh — Docker image entrypoint for the rypuz-c benchmark sandbox.
#
# Activates the Python virtual environment and sources global environment variables,
# then executes the passed command.

set -euo pipefail

# — Source global environment ————————————————————————————————————————————
if [ -f /opt/rypuz/rypuzbench.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /opt/rypuz/rypuzbench.env
  set +a
fi

# — Activate the virtual Python environment ————————————————————————————
# shellcheck disable=SC1091
source /opt/rypuz/venv/bin/activate

# — Execute the command ————————————————————————————————————————————————————
exec "$@"

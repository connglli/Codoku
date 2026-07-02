#!/usr/bin/env python3
"""Per-agent trajectory parsers for the rypuz benchmark.

Each agent produces a trajectory file in its own format.  This module
provides a *parse_trajectory(agent, path)* dispatcher that returns a
standardised stats dict regardless of the underlying format.

To add support for a new agent:
    1. Write a *_parse_<agent>(path)* function below.
    2. Register it in the PARSERS dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Standardised empty-stats template
# ---------------------------------------------------------------------------


def _empty_stats() -> dict[str, Any]:
  return {
    "num_rounds": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
    "tool_calls": {},
  }


# ---------------------------------------------------------------------------
# Claude Code  (--output-format stream-json  →  JSONL)
# ---------------------------------------------------------------------------


def _parse_claude(trajectory_file: Path) -> dict[str, Any]:
  stats = _empty_stats()
  if not trajectory_file.exists():
    return stats

  try:
    with open(trajectory_file) as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          event = json.loads(line)
        except json.JSONDecodeError:
          continue

        event_type = event.get("type", "")

        if event_type == "assistant":
          stats["num_rounds"] += 1

        if event_type == "result":
          usage = event.get("usage", {})
          stats["input_tokens"] += usage.get("input_tokens", 0)
          stats["output_tokens"] += usage.get("output_tokens", 0)
          stats["cache_read_tokens"] += usage.get("cache_read_input_tokens", 0)
          stats["cache_write_tokens"] += usage.get("cache_creation_input_tokens", 0)

        stats["cost_usd"] += event.get("total_cost_usd", 0.0)

        if event_type == "tool_use":
          tool_name = event.get("name", "unknown")
          stats["tool_calls"][tool_name] = stats["tool_calls"].get(tool_name, 0) + 1
  except Exception:
    pass  # Partial or corrupt trajectory

  stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]
  return stats


def _parse_opencode(trajectory_file: Path) -> dict[str, Any]:
  stats = _empty_stats()
  if not trajectory_file.exists():
    return stats

  try:
    with open(trajectory_file) as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          event = json.loads(line)
        except json.JSONDecodeError:
          continue

        event_type = event.get("type", "")

        if event_type == "step_start":
          stats["num_rounds"] += 1

        usage = event.get("usage", {})
        if not usage and "metrics" in event:
          usage = event["metrics"]
        if not usage and event_type == "step_finish":
          usage = event

        if usage:
          stats["input_tokens"] += (
            usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0
          )
          stats["output_tokens"] += (
            usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0
          )
          stats["cache_read_tokens"] += usage.get("cache_read_input_tokens", 0) or 0
          stats["cache_write_tokens"] += (
            usage.get("cache_creation_input_tokens", 0) or 0
          )
          stats["cost_usd"] += usage.get("cost_usd", 0.0) or 0.0

        if event_type == "tool_use":
          tool_name = event.get("name", "unknown")
          stats["tool_calls"][tool_name] = stats["tool_calls"].get(tool_name, 0) + 1
  except Exception:
    pass

  stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]
  return stats


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

PARSERS: dict[str, Any] = {
  "claude": _parse_claude,
  "opencode": _parse_opencode,
  # Future: "codex": _parse_codex,
}


def parse_trajectory(agent: str, trajectory_file: Path) -> dict[str, Any]:
  """Parse an agent's trajectory file into standardised stats."""
  parser = PARSERS.get(agent)
  if parser is None:
    return _empty_stats()
  return parser(trajectory_file)

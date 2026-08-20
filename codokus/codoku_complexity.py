"""codoku_complexity.py - realized puzzle metrics and complexity estimation.

Measures the generated puzzle's static structure, dynamic execution path,
masking, and constant-budget constraints; collapses them into a heuristic
(not calibrated) complexity estimate.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class PuzzleMetrics:
  """Properties measured from the generated puzzle file."""

  # Static structure
  cfg_nodes: int  # distinct CFG nodes named in the //@ CFG_EDGE markers and EXEC_PATH
  cfg_edges: int  # distinct edges declared in the //@ CFG_EDGE markers
  cyclomatic_complexity: int  # E - N + 2 over the declared CFG (min 1)

  # Dynamic execution (from the EXEC_PATH)
  exec_path_length: (
    int  # number of blocks executed on the prescribed path (incl. repeats)
  )
  unique_path_blocks: int  # distinct blocks visited on the path
  repeated_block_visits: int  # extra visits beyond the first for each path block
  max_block_visits: int  # visit count of the most-visited block on the path

  # Information hiding
  total_masks: int  # total <FILL_*> tokens in the puzzle body
  masks_by_kind: Mapping[str, int]  # count per mask kind, e.g. {"<FILL_VAR>": 3}

  # Constant-budget constraints
  const_budget_entries: int  # distinct values in the //@ <FILL_CONST> budget
  const_budget_total: int  # total slot count across all budget entries

  # Source size
  source_lines: int  # total lines of the puzzle file
  non_comment_source_lines: int  # lines that are not comment-only

  def flattened(self) -> dict[str, float]:
    result: dict[str, float] = {
      "cfg_nodes": self.cfg_nodes,
      "cfg_edges": self.cfg_edges,
      "cyclomatic_complexity": self.cyclomatic_complexity,
      "exec_path_length": self.exec_path_length,
      "unique_path_blocks": self.unique_path_blocks,
      "repeated_block_visits": self.repeated_block_visits,
      "max_block_visits": self.max_block_visits,
      "total_masks": self.total_masks,
      "const_budget_entries": self.const_budget_entries,
      "const_budget_total": self.const_budget_total,
      "source_lines": self.source_lines,
      "non_comment_source_lines": self.non_comment_source_lines,
    }
    for kind, count in self.masks_by_kind.items():
      result[f"mask_{kind.lower()}"] = count
    return result


@dataclass(frozen=True)
class ComplexityEstimate:
  """Heuristic, uncalibrated complexity estimate from the realized metrics."""

  size: float  # CFG size and source volume (cfg_nodes + non-comment lines)
  static_structure: float  # CFG edges and cyclomatic complexity
  dynamic_trace: float  # execution-path length and repeated block visits
  masking: float  # weighted sum of <FILL_*> mask counts
  constraints: float  # constant-budget size and slot interactions
  total: float  # sum of the five axes above


MASK_WEIGHTS: Mapping[str, float] = {
  "<FILL_VAR>": 1.0,
  "<FILL_CONST>": 1.2,
  "<FILL_OP>": 1.8,
  "<FILL_TYPE>": 1.0,
  "<FILL_LABEL>": 1.5,
  "<FILL_FUNC>": 1.8,
  "<FILL_FIELD>": 1.2,
  "<FILL_CTRL>": 2.5,
}

CFG_EDGE_RE = re.compile(r"#//@\s*CFG_EDGE\s*:\s*(.+)")
EXEC_PATH_RE = re.compile(r"#//@\s*EXEC_PATH\s*:\s*(.+)")
CONST_BUDGET_RE = re.compile(r"#//@\s*<FILL_CONST>\s*:\s*(.+?)\s+(\d+)\s*$")
BLOCK_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*|\d+")
KNOWN_MASKS = (
  "<FILL_VAR>",
  "<FILL_CONST>",
  "<FILL_OP>",
  "<FILL_TYPE>",
  "<FILL_LABEL>",
  "<FILL_FUNC>",
  "<FILL_FIELD>",
  "<FILL_CTRL>",
)


def extract_block_tokens(payload: str) -> list[str]:
  return BLOCK_TOKEN_RE.findall(payload)


def count_code_masks(text: str) -> Counter[str]:
  """Count <FILL_*> tokens in Python code, excluding comments and strings."""
  counts: Counter[str] = Counter()
  # The masked puzzle is not valid Python (tokens like `<FILL_VAR>`), so count
  # line-oriented while excluding comment-only lines.  No \b anchors: the
  # angle brackets are the token boundary.
  for line in text.splitlines():
    stripped = line.lstrip()
    if stripped.startswith("#"):
      continue
    for mask in re.findall(r"<FILL_[A-Z_]+>", line):
      counts[mask] += 1
  return counts


def analyze_puzzle(path: Path) -> PuzzleMetrics:
  """Measure realized properties of the generated puzzle file."""
  text = path.read_text()
  lines = text.splitlines()

  cfg_edges: list[tuple[str, str]] = []
  path_blocks: list[str] = []
  const_budget_entries = 0
  const_budget_total = 0

  for line in lines:
    edge_match = CFG_EDGE_RE.search(line)
    if edge_match:
      tokens = extract_block_tokens(edge_match.group(1))
      if len(tokens) >= 2:
        cfg_edges.append((tokens[0], tokens[-1]))

    path_match = EXEC_PATH_RE.search(line)
    if path_match:
      path_blocks.extend(extract_block_tokens(path_match.group(1)))

    budget_match = CONST_BUDGET_RE.search(line)
    if budget_match:
      const_budget_entries += 1
      const_budget_total += int(budget_match.group(2))

  cfg_nodes = {node for s, t in cfg_edges for node in (s, t)}
  cfg_nodes.update(path_blocks)
  node_count = len(cfg_nodes)
  edge_count = len(cfg_edges)
  cyclomatic = max(1, edge_count - node_count + 2) if node_count else 0

  path_counts = Counter(path_blocks)
  unique_path_blocks = len(path_counts)
  repeated_visits = sum(max(0, c - 1) for c in path_counts.values())
  max_block_visits = max(path_counts.values(), default=0)

  masks = count_code_masks(text)
  for known_mask in KNOWN_MASKS:
    masks.setdefault(known_mask, 0)

  non_comment_lines = sum(
    1 for line in lines if line.strip() and not line.lstrip().startswith("#")
  )

  return PuzzleMetrics(
    cfg_nodes=node_count,
    cfg_edges=edge_count,
    cyclomatic_complexity=cyclomatic,
    exec_path_length=len(path_blocks),
    unique_path_blocks=unique_path_blocks,
    repeated_block_visits=repeated_visits,
    max_block_visits=max_block_visits,
    total_masks=sum(masks.values()),
    masks_by_kind=dict(sorted(masks.items())),
    const_budget_entries=const_budget_entries,
    const_budget_total=const_budget_total,
    source_lines=len(lines),
    non_comment_source_lines=non_comment_lines,
  )


def estimate_complexity(metrics: PuzzleMetrics) -> ComplexityEstimate:
  """Estimate complexity from realized puzzle properties.

  The model is a transparent, hand-written heuristic over five independent
  axes; it is NOT a calibrated measure of solving difficulty.  Weights should
  eventually be fitted against solver outcomes (pass rate, time, attempts).

  Axes (all computed from the realized puzzle, not the generator knobs):

  - size: structural volume.
      size = 1.0 * cfg_nodes + 0.05 * non_comment_source_lines
    Nodes dominate; raw source lines add only a small linear term so that
    long-but-simple bodies do not inflate the score.

  - static_structure: how much branching/looping must be understood.
      static_structure = 1.0 * cfg_edges + 2.0 * cyclomatic_complexity
    Cyclomatic complexity counts decision points, so it is weighted twice as
    heavily as a single edge.

  - dynamic_trace: how long the prescribed execution must be followed.
      dynamic_trace = 0.6 * exec_path_length + 0.5 * max_block_visits
    Path length is the primary term; the most-visited block adds a small
    bonus because deep repetition is harder to track than an equally long
    straight-line path (repeated visits are already inside path length, so
    they are not counted again at full weight).

  - masking: how many blanks must be filled and how costly each kind is.
      masking = sum(MASK_WEIGHTS[kind] * count for kind, count in masks)
    Control-flow masks (<FILL_CTRL>) are the most expensive (they steer the
    whole path), followed by operators and function names; plain variable
    names are cheapest.

  - constraints: the constant-budget matching burden.
      if const_budget_entries:
        constraints = 0.75 * <FILL_CONST> count
                    + 0.5 * const_budget_entries
                    + 0.25 * (const_budget_total - const_budget_entries)
    A budget is an additive cost, not a multiplier: each constant slot costs
    weight 0.75, each distinct budget value costs 0.5 (more distinct values
    make the value-count matching harder), and repeated duplicates of a value
    add a small 0.25 term for the global-interaction aspect.

  - total: sum of the five axes, so two puzzles can share a total while
    differing in style (e.g. many masks vs. a long path).
  """
  size = 1.0 * metrics.cfg_nodes + 0.05 * metrics.non_comment_source_lines
  static_structure = 1.0 * metrics.cfg_edges + 2.0 * metrics.cyclomatic_complexity
  dynamic_trace = 0.6 * metrics.exec_path_length + 0.5 * metrics.max_block_visits
  masking = sum(
    MASK_WEIGHTS.get(kind, 1.0) * count for kind, count in metrics.masks_by_kind.items()
  )
  constraints = 0.0
  if metrics.const_budget_entries:
    const_masks = metrics.masks_by_kind.get("<FILL_CONST>", 0)
    constraints += 0.75 * const_masks
    constraints += 0.5 * metrics.const_budget_entries
    constraints += 0.25 * max(
      0, metrics.const_budget_total - metrics.const_budget_entries
    )
  total = size + static_structure + dynamic_trace + masking + constraints
  return ComplexityEstimate(
    size=round(size, 2),
    static_structure=round(static_structure, 2),
    dynamic_trace=round(dynamic_trace, 2),
    masking=round(masking, 2),
    constraints=round(constraints, 2),
    total=round(total, 2),
  )

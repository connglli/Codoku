"""codoku_complexity.py - realized puzzle metrics and complexity estimation.

Measures the generated puzzle's static structure, dynamic execution path,
masking, constant-budget constraints, and an enumerated solution-space size;
collapses them into a heuristic (not calibrated) complexity estimate.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from codoku_common import (
  BINARY_OP_SPANS,
  COMPARISON_OP_SPANS,
  CONTROL_FLOW_KEYWORDS,
  IFEXP_KEYWORDS,
  INTERNAL_HELPER_FUNCS,
  UNARY_OP_SPANS,
  collect_python_leaf_locals,
  find_python_leaf_function,
)


@dataclass(frozen=True)
class PuzzleMetrics:
  """Properties measured from the generated puzzle file."""

  # Static structure
  cfg_nodes: int  # distinct CFG nodes named in the //@ CFG_EDGE markers and EXEC_PATH
  cfg_edges: int  # distinct edges declared in the //@ CFG_EDGE markers
  cyclomatic_complexity: int  # E - N + 2 over the declared CFG (min 1)
  n_loops: int  # distinct loops (back-edge targets) exercised on the path

  # Dynamic execution (from the EXEC_PATH)
  exec_path_length: (
    int  # number of blocks executed on the prescribed path (incl. repeats)
  )
  unique_path_blocks: int  # distinct blocks visited on the path
  repeated_block_visits: int  # extra visits beyond the first for each path block
  max_block_visits: int  # visit count of the most-visited block on the path
  loop_iterations_total: int  # summed trip counts over those loops
  loop_iterations_avg: float  # loop_iterations_total / n_loops (0 if no loops)

  # Information hiding
  total_masks: int  # total <FILL_*> tokens in the puzzle body
  masks_by_kind: Mapping[str, int]  # count per mask kind, e.g. {"<FILL_VAR>": 3}

  # Solution space (enumerated fill combinations, reported as log10)
  sol_space_log10: float

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
      "n_loops": self.n_loops,
      "exec_path_length": self.exec_path_length,
      "unique_path_blocks": self.unique_path_blocks,
      "repeated_block_visits": self.repeated_block_visits,
      "max_block_visits": self.max_block_visits,
      "loop_iterations_total": self.loop_iterations_total,
      "loop_iterations_avg": self.loop_iterations_avg,
      "total_masks": self.total_masks,
      "const_budget_entries": self.const_budget_entries,
      "const_budget_total": self.const_budget_total,
      "source_lines": self.source_lines,
      "non_comment_source_lines": self.non_comment_source_lines,
    }
    for kind, count in self.masks_by_kind.items():
      result[f"mask_{kind.lower()}"] = count
    result["sol_space_log10"] = self.sol_space_log10
    return result


@dataclass(frozen=True)
class ComplexityEstimate:
  """Heuristic, uncalibrated complexity estimate from the realized metrics."""

  static_struct: float  # CFG size/edges, cyclomatic complexity, and source volume
  dynamic_trace: float  # execution-path length and repeated block visits
  masking: float  # weighted sum of <FILL_*> mask counts
  constraints: float  # constant-budget size and slot interactions
  total: float  # sum of the four axes above


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
MASK_TOKEN_RE = re.compile(r"<FILL_[A-Z_]+>")
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

# ---------------------------------------------------------------------------
# Solution-space estimation
# ---------------------------------------------------------------------------

# Nominal vocabulary for an unconstrained <FILL_CONST> slot: puzzles generated
# with lift_consts carry no budget, so any literal is admissible.
UNBOUNDED_FILL_CONST_CHOICES = 65536

# Per-kind placeholders used only so the de-masked source parses.  Each kind
# always occurs in positions where its placeholder is grammatically valid:
# identifiers for names/lvalues, a literal for constants, `pass` for
# break/continue statements, and an infix operator for operators (including
# the if/else keywords of a ternary, both of which are masked together).
FILL_PARSE_PLACEHOLDERS: Mapping[str, str] = {
  "<FILL_VAR>": "_var_slot",
  "<FILL_CONST>": "0",
  "<FILL_OP>": "+",
  "<FILL_FUNC>": "_fun_slot",
  "<FILL_CTRL>": "pass",
}


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
    for mask in MASK_TOKEN_RE.findall(line):
      counts[mask] += 1
  return counts


def collect_fill_choices(text: str, budget_values: set[str]) -> dict[str, int]:
  """Candidate-token vocabulary size for one slot of each mask kind.

  Per-kind sets, mirroring the masking rules in codoku_common:

  - <FILL_VAR>:   local/parameter names visible in the leaf function.
  - <FILL_CONST>: distinct values admitted by the //@ <FILL_CONST> budget;
                  with no budget any literal is admissible, approximated by
                  UNBOUNDED_FILL_CONST_CHOICES.
  - <FILL_OP>:    operator/keyword vocabulary masked as <FILL_OP>.
  - <FILL_FUNC>:  functions defined in the file except internal helpers.
  - <FILL_CTRL>:  break / continue.
  - <FILL_TYPE>/<FILL_LABEL>/<FILL_FIELD>: unused by the Python target (0).

  If the de-masked source fails to parse, the AST-derived entries
  (<FILL_VAR>, <FILL_FUNC>) stay 0.
  """
  choices: dict[str, int] = {kind: 0 for kind in KNOWN_MASKS}
  op_symbols = {
    sym.decode("ascii")
    for spans in (BINARY_OP_SPANS, COMPARISON_OP_SPANS, UNARY_OP_SPANS, IFEXP_KEYWORDS)
    for sym in spans
  }
  choices["<FILL_OP>"] = len(op_symbols)
  choices["<FILL_CTRL>"] = len(CONTROL_FLOW_KEYWORDS)
  choices["<FILL_CONST>"] = (
    len(budget_values) if budget_values else UNBOUNDED_FILL_CONST_CHOICES
  )

  try:
    tree = ast.parse(
      MASK_TOKEN_RE.sub(lambda m: FILL_PARSE_PLACEHOLDERS.get(m.group(0), "None"), text)
    )
  except SyntaxError:
    return choices

  leaf, _ = find_python_leaf_function(tree, b"")
  if leaf is not None:
    choices["<FILL_VAR>"] = len(collect_python_leaf_locals(leaf))
  defined_funcs = {
    node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
  }
  choices["<FILL_FUNC>"] = len(defined_funcs - INTERNAL_HELPER_FUNCS)
  return choices


def estimate_sol_space_log10(
  masks_by_kind: Mapping[str, int], fill_choices: Mapping[str, int]
) -> float:
  """log10 of the solution space: the cartesian product of per-slot candidate counts across all mask slots.

  Each mask of a kind contributes fill_choices[kind] options, so
  log10(sol_space) = sum(count[kind] * log10(choices[kind])).  Returns -inf
  when a masked kind has no enumerable candidates.
  """
  total = 0.0
  for kind, count in masks_by_kind.items():
    if count <= 0:
      continue
    n_choices = fill_choices.get(kind, 0)
    if n_choices <= 0:
      return float("-inf")
    total += count * math.log10(n_choices)
  return total


def analyze_puzzle(path: Path) -> PuzzleMetrics:
  """Measure realized properties of the generated puzzle file."""
  text = path.read_text()
  lines = text.splitlines()

  cfg_edges: list[tuple[str, str]] = []
  path_blocks: list[str] = []
  const_budget_entries = 0
  const_budget_total = 0
  const_budget_values: set[str] = set()

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
      const_budget_values.add(budget_match.group(1))

  cfg_nodes = {node for s, t in cfg_edges for node in (s, t)}
  cfg_nodes.update(path_blocks)
  node_count = len(cfg_nodes)
  edge_count = len(cfg_edges)
  cyclomatic = max(1, edge_count - node_count + 2) if node_count else 0

  path_counts = Counter(path_blocks)
  unique_path_blocks = len(path_counts)
  repeated_visits = sum(max(0, c - 1) for c in path_counts.values())
  max_block_visits = max(path_counts.values(), default=0)

  # Loops on the prescribed path.  A declared CFG edge whose target first
  # occurs earlier on the path than its source is a back-edge; its target
  # heads a loop, and the header's occurrence count estimates that loop's
  # trip count.  One visit is subtracted when the trace's final step departs
  # from the header itself: the failing test exits without completing an
  # iteration (exits from inside the body need no correction).
  first_seen: dict[str, int] = {}
  for idx, block in enumerate(path_blocks):
    first_seen.setdefault(block, idx)
  edge_set = set(cfg_edges)
  headers = {
    dst
    for src, dst in edge_set
    if dst in first_seen and src in first_seen and first_seen[dst] < first_seen[src]
  }
  iterations_by_header = {h: path_counts[h] for h in headers}
  if len(path_blocks) >= 2 and path_blocks[-2] in headers:
    iterations_by_header[path_blocks[-2]] -= 1
  n_loops = len(headers)
  loop_iters_total = sum(iterations_by_header.values())
  loop_iters_avg = loop_iters_total / n_loops if n_loops else 0.0

  masks = count_code_masks(text)
  for known_mask in KNOWN_MASKS:
    masks.setdefault(known_mask, 0)

  fill_choices = collect_fill_choices(text, const_budget_values)
  sol_space_log10 = estimate_sol_space_log10(masks, fill_choices)

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
    n_loops=n_loops,
    loop_iterations_total=loop_iters_total,
    loop_iterations_avg=round(loop_iters_avg, 2),
    total_masks=sum(masks.values()),
    masks_by_kind=dict(sorted(masks.items())),
    sol_space_log10=round(sol_space_log10, 4),
    const_budget_entries=const_budget_entries,
    const_budget_total=const_budget_total,
    source_lines=len(lines),
    non_comment_source_lines=non_comment_lines,
  )


def estimate_complexity(metrics: PuzzleMetrics) -> ComplexityEstimate:
  """Estimate complexity from realized puzzle properties.

  The model is a transparent, hand-written heuristic over four independent
  axes; it is NOT a calibrated measure of solving difficulty.  Weights should
  eventually be fitted against solver outcomes (pass rate, time, attempts).

  Axes (all computed from the realized puzzle, not the generator knobs):

  - static_struct: structural volume and control-flow complexity.
      static_struct = 1.0 * cfg_nodes
                    + 1.0 * cfg_edges
                    + 2.0 * cyclomatic_complexity
                    + 0.05 * non_comment_source_lines
    Nodes and edges describe the graph skeleton; cyclomatic complexity
    weights decision points; non-comment lines add a minor term for code volume.

  - dynamic_trace: how long the prescribed execution must be followed.
      dynamic_trace = 0.6 * exec_path_length + 0.5 * loop_iterations_avg
    Path length is the primary term and already carries the repetition
    volume (every iteration re-executes its body on the path).  The average
    loop depth adds a small bonus because many consecutive passes through
    one loop are harder to track than the same number of blocks spread over
    distinct code; it is body-size-independent and sees every loop, unlike
    the previous max_block_visits term, which conflated loop depth with
    loop-body size and ignored all but the hottest loop.
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

  - total: sum of the four axes, so two puzzles can share a total while
    differing in style (e.g. many masks vs. a long path).
  """
  static_struct = (
    1.0 * metrics.cfg_nodes
    + 1.0 * metrics.cfg_edges
    + 2.0 * metrics.cyclomatic_complexity
    + 0.05 * metrics.non_comment_source_lines
  )
  dynamic_trace = 0.6 * metrics.exec_path_length + 0.5 * metrics.loop_iterations_avg
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
  total = static_struct + dynamic_trace + masking + constraints
  return ComplexityEstimate(
    static_struct=round(static_struct, 2),
    dynamic_trace=round(dynamic_trace, 2),
    masking=round(masking, 2),
    constraints=round(constraints, 2),
    total=round(total, 2),
  )

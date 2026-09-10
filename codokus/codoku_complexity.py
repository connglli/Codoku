"""codoku_complexity.py - realized puzzle metrics and complexity estimation.

Measures the generated puzzle's static structure, dynamic execution path,
masking, constant-budget constraints, and an enumerated solution-space size;
collapses them into a heuristic (not calibrated) complexity estimate.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
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

  # Vocabulary complexity (Halstead)
  hal_operators: int = 0  # distinct operators
  hal_operands: int = 0  # distinct operands
  hal_total_operators: int = 0  # total operator occurrences
  hal_total_operands: int = 0  # total operand occurrences
  hal_vocabulary: int = 0  # distinct operators + distinct operands
  hal_length: int = 0  # total operators + total operands
  hal_volume: float = 0.0  # length * log2(vocabulary)
  hal_difficulty: float = 0.0  # (operators / 2) * (total_operands / operands)
  hal_effort: float = 0.0  # difficulty * volume

  # Control-flow complexity (CFG and McCabe)
  cfg_nodes: int = 0  # distinct CFG nodes in CFG_EDGE markers and EXEC_PATH
  cfg_edges: int = 0  # distinct edges declared in the //@ CFG_EDGE markers
  n_loops: int = 0  # distinct loops (back-edge targets) exercised on the path
  cyclomatic: int = 0  # E - N + 2 over the declared CFG (min 1)

  # Data-flow complexity (DepDegree over statement def-use edges)
  dep_nodes: int = 0  # |V|: params entry + definition/use statements
  dep_edges: int = 0  # |E|: resolved def -> use edges
  dep_avg_degree: float = 0.0  # 2|E| / |V|
  dep_max_degree: int = 0  # max per-node in + out degree
  dep_density: float = 0.0  # |E| / (|V| * (|V| - 1))

  # Dynamic execution (from the EXEC_PATH)
  exec_path_length: int = 0  # blocks executed on the path (incl. repeats)
  unique_path_blocks: int = 0  # distinct blocks visited on the path
  repeated_block_visits: int = 0  # extra visits beyond the first for each path block
  max_block_visits: int = 0  # visit count of the most-visited block on the path
  loop_iterations_total: int = 0  # summed trip counts over those loops
  loop_iterations_avg: float = 0.0  # loop_iterations_total / n_loops (0 if no loops)

  # Information hiding
  total_masks: int = 0  # total <FILL_*> tokens in the puzzle body
  masks_by_kind: Mapping[str, int] = (
    field(  # count per mask kind, e.g. {"<FILL_VAR>": 3}
      default_factory=dict
    )
  )

  # Solution space (enumerated fill combinations, reported as log10)
  sol_space_log10: float = 0.0

  # Constant-budget constraints
  const_budget_entries: int = 0  # distinct values in the //@ <FILL_CONST> budget
  const_budget_total: int = 0  # total slot count across all budget entries

  # Source size
  source_lines: int = 0  # total lines of the puzzle file
  non_comment_source_lines: int = 0  # lines that are not comment-only

  def flattened(self) -> dict[str, float]:
    result: dict[str, float] = {
      "cfg_nodes": self.cfg_nodes,
      "cfg_edges": self.cfg_edges,
      "cyclomatic": self.cyclomatic,
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
      "hal_operators": self.hal_operators,
      "hal_operands": self.hal_operands,
      "hal_total_operators": self.hal_total_operators,
      "hal_total_operands": self.hal_total_operands,
      "hal_vocabulary": self.hal_vocabulary,
      "hal_length": self.hal_length,
      "hal_volume": self.hal_volume,
      "hal_difficulty": self.hal_difficulty,
      "hal_effort": self.hal_effort,
      "dep_nodes": self.dep_nodes,
      "dep_edges": self.dep_edges,
      "dep_avg_degree": self.dep_avg_degree,
      "dep_max_degree": self.dep_max_degree,
      "dep_density": self.dep_density,
    }
    for kind, count in self.masks_by_kind.items():
      result[f"mask_{kind.lower()}"] = count
    result["sol_space_log10"] = self.sol_space_log10
    return result


@dataclass(frozen=True)
class ComplexityEstimate:
  """Heuristic, uncalibrated complexity estimate from the realized metrics."""

  static_struct: float  # CFG, vocabulary, data flow, and source volume
  dynamic_trace: float  # execution-path length and repeated block visits
  masking: float  # weighted sum of <FILL_*> mask counts
  constraints: float  # constant-budget size and slot interactions
  total: float = 0.0  # sum of the four axes above


@dataclass(frozen=True)
class McCabeMetrics:
  """Control-flow complexity (McCabe) over the declared CFG.

  Nodes and edges surface on PuzzleMetrics as cfg_nodes/cfg_edges, so they
  are not duplicated under a second name; this dataclass is the single
  computation site for all three values.
  """

  nodes: int  # |N|: distinct CFG nodes
  edges: int  # |E|: distinct CFG edges
  cyclomatic: int  # E - N + 2 over one connected routine (min 1; 0 if empty)

  @staticmethod
  def zero() -> McCabeMetrics:
    return McCabeMetrics(0, 0, 0)


@dataclass(frozen=True)
class HalsteadMetrics:
  """Vocabulary complexity from operator/operand counts in the leaf function.

  Operators are AST node types other than names, literals, parameter
  bindings, and expression contexts; operands are variable names, literals,
  and parameter names.  Only vocabulary, length, volume, difficulty, and
  effort are reported: the historical estimated-time (E / 18) and
  estimated-defects (V / 3000) heuristics are deliberately omitted as
  unreliable predictors for an individual puzzle.
  """

  operators: int  # distinct operators
  operands: int  # distinct operands
  total_operators: int  # total operator occurrences
  total_operands: int  # total operand occurrences
  vocabulary: int  # distinct operators + distinct operands
  length: int  # total operators + total operands
  volume: float  # length * log2(vocabulary)
  difficulty: float  # (operators / 2) * (total_operands / operands)
  effort: float  # difficulty * volume

  @staticmethod
  def zero() -> HalsteadMetrics:
    return HalsteadMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class DepDegreeMetrics:
  """Data-flow complexity from statement-level def-use edges in the leaf.

  Nodes are a params-entry plus, in source order, every assignment, loop/if
  test, and return.  Each use of a variable resolves to its nearest
  preceding definition (params pre-defined), giving one directed edge per
  (definer, user) pair.  Per-node DepDegree is in + out degree; the routine
  reports node/edge counts, average degree (2|E| / |V|), maximum degree,
  and directed density (|E| / (|V| * (|V| - 1))).

  Approximations, documented so values are only compared within this tool:
  straight-line reaching definitions (branch merges and loop-carried flows
  resolve to the textually nearest prior definition), whole-variable
  granularity for subscripts/attributes (no index analysis), no alias
  analysis, and nested scopes excluded.  Names defined outside the leaf
  (preamble helpers, module sentinels) simply never open edges.
  """

  nodes: int  # |V|
  edges: int  # |E|
  avg_degree: float  # 2|E| / |V|
  max_degree: int  # max per-node in + out degree
  density: float  # |E| / (|V| * (|V| - 1))

  @staticmethod
  def zero() -> DepDegreeMetrics:
    return DepDegreeMetrics(0, 0, 0.0, 0, 0.0)


# ---------------------------------------------------------------------------
# Vocabulary (Halstead) and data-flow (DepDegree) measurement
# ---------------------------------------------------------------------------

# Expression contexts carry no operator/operand information of their own.
_HALSTEAD_SKIP = (ast.Load, ast.Store, ast.Del)


def _is_trace_block(node: ast.AST) -> bool:
  """True for DUMP_TRACE instrumentation blocks (never puzzle logic).

  Mirrors the trace-block exclusion in codoku_common's maskable-statement
  walk: these `if __import__("os").environ.get("DUMP_TRACE")` prints are
  scaffolding, so vocabulary and data-flow measurement skips them.
  """
  return isinstance(node, ast.If) and any(
    isinstance(n, ast.Constant) and n.value == "DUMP_TRACE" for n in ast.walk(node.test)
  )


def _trace_subtree_ids(leaf: ast.FunctionDef) -> set[int]:
  """Ids of every node under a DUMP_TRACE instrumentation block."""
  ids: set[int] = set()
  for node in ast.walk(leaf):
    if _is_trace_block(node):
      for sub in ast.walk(node):
        ids.add(id(sub))
  return ids


def compute_halstead(leaf: ast.FunctionDef) -> HalsteadMetrics:
  """Count Halstead operators/operands over the leaf function subtree."""
  operator_counts: Counter[str] = Counter()
  operand_counts: Counter[str] = Counter()
  skip = _trace_subtree_ids(leaf)
  for node in ast.walk(leaf):
    if id(node) in skip or isinstance(node, _HALSTEAD_SKIP):
      continue
    if isinstance(node, ast.Name):
      operand_counts[node.id] += 1
    elif isinstance(node, ast.Constant):
      operand_counts[repr(node.value)] += 1
    elif isinstance(node, ast.arg):
      operand_counts[node.arg] += 1
    else:
      operator_counts[type(node).__name__] += 1
  distinct_operators = len(operator_counts)
  distinct_operands = len(operand_counts)
  total_operators = sum(operator_counts.values())
  total_operands = sum(operand_counts.values())
  vocabulary = distinct_operators + distinct_operands
  length = total_operators + total_operands
  volume = round(length * math.log2(vocabulary), 2) if vocabulary > 1 else 0.0
  difficulty = (
    round((distinct_operators / 2) * (total_operands / distinct_operands), 2)
    if distinct_operators and distinct_operands
    else 0.0
  )
  effort = round(difficulty * volume, 2)
  return HalsteadMetrics(
    distinct_operators,
    distinct_operands,
    total_operators,
    total_operands,
    vocabulary,
    length,
    volume,
    difficulty,
    effort,
  )


class _DepCollector(ast.NodeVisitor):
  """Collect (defs, uses) per statement in source order; skips nested scopes."""

  def __init__(self) -> None:
    self.stmts: list[tuple[set[str], set[str]]] = []

  @staticmethod
  def _names(tree: ast.AST | None) -> set[str]:
    if tree is None:
      return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

  @staticmethod
  def _target_defs(targets: list[ast.expr]) -> set[str]:
    defs: set[str] = set()
    for target in targets:
      defs |= _DepCollector._names(target)
    return defs

  def visit_Assign(self, node: ast.Assign) -> None:
    uses = self._names(node.value)
    for target in node.targets:
      if isinstance(target, ast.Subscript):
        uses |= self._names(target.slice)
    self.stmts.append((self._target_defs(node.targets), uses))

  def visit_AugAssign(self, node: ast.AugAssign) -> None:
    if isinstance(node.target, ast.Subscript):
      # Whole-variable granularity (as for Assign): the base object is
      # redefined while the slice is read.
      base = self._names(node.target.value)
      self.stmts.append((set(base), self._names(node.value) | self._names(node.target)))
      return
    touched = self._names(node.target)
    self.stmts.append((set(touched), self._names(node.value) | set(touched)))

  def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
    self.stmts.append((self._names(node.target), self._names(node.value)))

  def visit_For(self, node: ast.For) -> None:
    self.stmts.append((self._names(node.target), self._names(node.iter)))
    self.generic_visit(node)

  def visit_If(self, node: ast.If) -> None:
    if _is_trace_block(node):
      return
    self.stmts.append((set(), self._names(node.test)))
    self.generic_visit(node)

  def visit_While(self, node: ast.While) -> None:
    self.stmts.append((set(), self._names(node.test)))
    self.generic_visit(node)

  def visit_Return(self, node: ast.Return) -> None:
    self.stmts.append((set(), self._names(node.value)))

  def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
    return

  def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
    return

  def visit_Lambda(self, node: ast.Lambda) -> None:
    return

  def visit_ClassDef(self, node: ast.ClassDef) -> None:
    return


def compute_depdegree(leaf: ast.FunctionDef) -> DepDegreeMetrics:
  """Build the statement def-use graph and report routine-level DepDegree."""
  collector = _DepCollector()
  for stmt in leaf.body:
    collector.visit(stmt)
  params = [a.arg for a in leaf.args.args]
  params += [a.arg for a in leaf.args.posonlyargs]
  params += [a.arg for a in leaf.args.kwonlyargs]
  if leaf.args.vararg is not None:
    params.append(leaf.args.vararg.arg)
  if leaf.args.kwarg is not None:
    params.append(leaf.args.kwarg.arg)

  nodes = [(set(params), set())] + [(defs, uses) for defs, uses in collector.stmts]
  # Node 0 pre-defines the params; every other node resolves each distinct
  # used name to its nearest preceding definer.
  last_def: dict[str, int] = dict.fromkeys(params, 0)
  edges: set[tuple[int, int]] = set()
  for user in range(1, len(nodes)):
    _, uses = nodes[user]
    for name in sorted(uses):
      if name in last_def:
        edges.add((last_def[name], user))
    for name in sorted(nodes[user][0]):
      last_def[name] = user

  node_count = len(nodes)
  edge_count = len(edges)
  if not node_count:
    return DepDegreeMetrics.zero()
  degrees: Counter[int] = Counter()
  for definer, user in edges:
    degrees[definer] += 1
    degrees[user] += 1
  avg_degree = round(2 * edge_count / node_count, 2)
  max_degree = max(degrees.values(), default=0)
  density = (
    round(edge_count / (node_count * (node_count - 1)), 4) if node_count > 1 else 0.0
  )
  return DepDegreeMetrics(node_count, edge_count, avg_degree, max_degree, density)


def compute_mccabe(
  cfg_nodes: set[str], cfg_edges: list[tuple[str, str]]
) -> McCabeMetrics:
  """McCabe complexity (E - N + 2, min 1) over the declared CFG."""
  node_count = len(cfg_nodes)
  edge_count = len(cfg_edges)
  cyclomatic = max(1, edge_count - node_count + 2) if node_count else 0
  return McCabeMetrics(node_count, edge_count, cyclomatic)


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
  "<FILL_LABEL>": "exit",
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


def collect_fill_choices(
  text: str, budget_values: set[str], n_cfg_nodes: int
) -> dict[str, int]:
  """Candidate-token vocabulary size for one slot of each mask kind.

  Per-kind sets, mirroring the masking rules in codoku_common:

  - <FILL_VAR>:   local/parameter names visible in the leaf function.
  - <FILL_CONST>: distinct values admitted by the //@ <FILL_CONST> budget;
                  with no budget any literal is admissible, approximated by
                  UNBOUNDED_FILL_CONST_CHOICES.
  - <FILL_OP>:    operator/keyword vocabulary masked as <FILL_OP>.
  - <FILL_FUNC>:  functions defined in the file except internal helpers.
  - <FILL_CTRL>:  break / continue.
  - <FILL_LABEL>: CFG nodes, i.e., the possible destinations of a goto flag
                  (the concrete flag names are hidden behind the masks, so
                  the node count is the approximation).
  - <FILL_TYPE>/<FILL_FIELD>: unused by the Python target (0).

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
  choices["<FILL_LABEL>"] = max(1, n_cfg_nodes)
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


def _leaf_from_source(source: str) -> ast.FunctionDef | None:
  """Parse source and return its leaf function, or None when unavailable."""
  try:
    tree = ast.parse(source)
  except SyntaxError:
    return None
  leaf, _ = find_python_leaf_function(tree, b"")
  return leaf


def resolve_vocab_source(path: Path, gt_path: Path | None = None) -> Path | None:
  """First usable ground-truth file for a puzzle, or None.

  Preference order is the explicit gt_path, the oracle sibling, then the
  same-directory <stem>.gt.py.  Usable means the file reads and contains a
  leaf function.
  """
  for candidate in _find_gt_candidates(path, gt_path):
    try:
      content = candidate.read_text()
    except (OSError, UnicodeError):
      continue
    if _leaf_from_source(content) is not None:
      return candidate
  return None


def _find_gt_candidates(path: Path, gt_path: Path | None) -> list[Path]:
  """Candidate ground-truth files for a puzzle, in preference order."""
  candidates = []
  if gt_path is not None:
    candidates.append(Path(gt_path))
  candidates.append(path.parent / "oracle" / "puzzle.gt.py")
  candidates.append(path.parent / (path.stem + ".gt.py"))
  unique = []
  for candidate in candidates:
    if candidate not in unique:
      unique.append(candidate)
  return unique


def analyze_puzzle(path: Path, gt_path: Path | None = None) -> PuzzleMetrics:
  """Measure realized properties of the generated puzzle file.

  CFG markers, masks, budgets, and sizes come from the puzzle itself.
  Vocabulary (Halstead) and data flow (DepDegree) are measured on the
  ground-truth leaf when available — the explicit gt_path (generation),
  else the oracle sibling or same-directory <stem>.gt.py — because masked
  holes carry no vocabulary of their own.  Without any usable GT, fall back
  to the placeholder-demasked puzzle (same mapping as solution-space
  parsing); anything still unparsable, or without a leaf, measures zero.
  """
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
  mccabe = compute_mccabe(cfg_nodes, cfg_edges)
  node_count = mccabe.nodes
  edge_count = mccabe.edges
  cyclomatic = mccabe.cyclomatic

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

  fill_choices = collect_fill_choices(text, const_budget_values, node_count)
  sol_space_log10 = estimate_sol_space_log10(masks, fill_choices)

  non_comment_lines = sum(
    1 for line in lines if line.strip() and not line.lstrip().startswith("#")
  )

  # Vocabulary and data-flow prefer the ground-truth leaf (see docstring);
  # the placeholder-demasked puzzle is only the fallback.
  gt_source = resolve_vocab_source(path, gt_path)
  leaf_node = None
  if gt_source is not None:
    try:
      leaf_node = _leaf_from_source(gt_source.read_text())
    except (OSError, UnicodeError):
      leaf_node = None
  if leaf_node is None:
    demasked = MASK_TOKEN_RE.sub(
      lambda m: FILL_PARSE_PLACEHOLDERS.get(m.group(0), "None"), text
    )
    leaf_node = _leaf_from_source(demasked)
  if leaf_node is None:
    halstead = HalsteadMetrics.zero()
    depdegree = DepDegreeMetrics.zero()
  else:
    halstead = compute_halstead(leaf_node)
    depdegree = compute_depdegree(leaf_node)

  return PuzzleMetrics(
    cfg_nodes=node_count,
    cfg_edges=edge_count,
    cyclomatic=cyclomatic,
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
    hal_operators=halstead.operators,
    hal_operands=halstead.operands,
    hal_total_operators=halstead.total_operators,
    hal_total_operands=halstead.total_operands,
    hal_vocabulary=halstead.vocabulary,
    hal_length=halstead.length,
    hal_volume=halstead.volume,
    hal_difficulty=halstead.difficulty,
    hal_effort=halstead.effort,
    dep_nodes=depdegree.nodes,
    dep_edges=depdegree.edges,
    dep_avg_degree=depdegree.avg_degree,
    dep_max_degree=depdegree.max_degree,
    dep_density=depdegree.density,
  )


def estimate_complexity(metrics: PuzzleMetrics) -> ComplexityEstimate:
  """Estimate complexity from realized puzzle properties.

  The model is a transparent, hand-written heuristic over four independent
  axes; it is NOT a calibrated measure of solving difficulty.  Weights should
  eventually be fitted against solver outcomes (pass rate, time, attempts).

  Axes (all computed from the realized puzzle, not the generator knobs):

  - static_struct: everything statically visible in the puzzle.
      static_struct = 1.0 * cfg_nodes
                    + 1.0 * cfg_edges
                    + 2.0 * cyclomatic
                    + 0.05 * non_comment_source_lines
                    + 0.01 * hal_volume + 0.5 * hal_difficulty
                    + 2.0 * dep_avg_degree + 1.0 * dep_max_degree
    Nodes and edges describe the graph skeleton; cyclomatic complexity
    weights decision points; non-comment lines add a minor term for code
    volume.  Vocabulary (Halstead) and data flow (DepDegree) are static
    properties too: raw volume runs in the thousands, so it is scaled by
    1/100 while difficulty carries weight 1/2 (estimated time and defects
    are excluded upstream); average degree captures overall change
    propagation while maximum degree captures the hottest hub (often a
    checksum-style accumulator).

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
    + 2.0 * metrics.cyclomatic
    + 0.05 * metrics.non_comment_source_lines
    + 0.01 * metrics.hal_volume
    + 0.5 * metrics.hal_difficulty
    + 2.0 * metrics.dep_avg_degree
    + 1.0 * metrics.dep_max_degree
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


# ---------------------------------------------------------------------------
# CLI (mirrors codoku_checker.py; `codoku analyze` forwards here)
# ---------------------------------------------------------------------------


def run_analyze(
  puzzle: str, as_json: bool = False, ground_truth: str | None = None
) -> int:
  """Analyze one puzzle file and print its metrics and complexity estimate."""
  path = Path(puzzle)
  if not path.is_file():
    print(f"codoku: error: puzzle file not found: {puzzle}", file=sys.stderr)
    return 2
  gt_path = Path(ground_truth) if ground_truth is not None else None
  if gt_path is not None:
    # An explicitly requested GT file must be usable; silently substituting
    # a sibling (or the placeholder fallback) would misattribute the source.
    resolved = resolve_vocab_source(path, gt_path)
    if resolved is None or resolved.resolve() != gt_path.resolve():
      print(
        f"codoku: error: unusable --ground-truth file: {ground_truth}",
        file=sys.stderr,
      )
      return 2
  try:
    metrics = analyze_puzzle(path, gt_path=gt_path)
  except (OSError, UnicodeError) as e:
    print(f"codoku: error: cannot read puzzle '{puzzle}': {e}", file=sys.stderr)
    return 2
  estimate = estimate_complexity(metrics)
  vocab_source = resolve_vocab_source(path, gt_path)

  if as_json:
    payload = {
      "target": "python",
      "puzzle": str(path),
      "vocab_source": str(vocab_source) if vocab_source is not None else None,
      "realized_metrics": asdict(metrics),
      "complexity_estimate": asdict(estimate),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

  print(f"Puzzle analysis: {path}")
  print("  structure:")
  print(f"    cfg_nodes={metrics.cfg_nodes} cfg_edges={metrics.cfg_edges}")
  print(f"    cyclomatic={metrics.cyclomatic}")
  print(
    f"    source_lines={metrics.source_lines}"
    f" non_comment_source_lines={metrics.non_comment_source_lines}"
  )
  print("  execution path:")
  print(
    f"    exec_path_length={metrics.exec_path_length}"
    f" unique_path_blocks={metrics.unique_path_blocks}"
  )
  print(
    f"    repeated_block_visits={metrics.repeated_block_visits}"
    f" max_block_visits={metrics.max_block_visits}"
  )
  print(
    f"    n_loops={metrics.n_loops}"
    f" loop_iterations_total={metrics.loop_iterations_total}"
    f" loop_iterations_avg={metrics.loop_iterations_avg:.2f}"
  )
  print("  masks:")
  print(f"    total_masks={metrics.total_masks}")
  kinds = " ".join(f"{kind}={count}" for kind, count in metrics.masks_by_kind.items())
  print(f"    {kinds}")
  print("  constant budget:")
  print(
    f"    entries={metrics.const_budget_entries}"
    f" total_slots={metrics.const_budget_total}"
  )
  print("  vocabulary (Halstead, leaf-scoped, no time/defect heuristics):")
  print(
    f"    operators={metrics.hal_operators} operands={metrics.hal_operands}"
    f" total_operators={metrics.hal_total_operators}"
    f" total_operands={metrics.hal_total_operands}"
    f" vocabulary={metrics.hal_vocabulary} length={metrics.hal_length}"
  )
  print(
    f"    volume={metrics.hal_volume:.2f}"
    f" difficulty={metrics.hal_difficulty:.2f}"
    f" effort={metrics.hal_effort:.2f}"
  )
  print("  data flow (DepDegree, nearest-prior reaching defs):")
  print(
    f"    nodes={metrics.dep_nodes} edges={metrics.dep_edges}"
    f" avg_degree={metrics.dep_avg_degree:.2f} max_degree={metrics.dep_max_degree}"
    f" density={metrics.dep_density:.4f}"
  )
  if vocab_source is not None:
    print(f"  vocab source: {vocab_source}")
  else:
    print("  vocab source: placeholder fallback (no usable GT file found)")
  print(f"  solution space: log10={metrics.sol_space_log10:.2f}")
  print("  complexity estimate (heuristic, uncalibrated):")
  for axis in ("static_struct", "dynamic_trace", "masking", "constraints"):
    print(f"    {axis}={getattr(estimate, axis):.2f}")
  print(f"    total={estimate.total:.2f}")
  return 0


def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="Analyze a codoku puzzle file and report its realized metrics "
    "and the heuristic complexity estimate.",
  )
  p.add_argument(
    "puzzle",
    nargs="?",
    default="puzzle.py",
    help="puzzle file path (.py with <FILL_XXX> marks; default: puzzle.py)",
  )
  p.add_argument(
    "--json", action="store_true", help="emit machine-readable JSON instead of text"
  )
  p.add_argument(
    "--ground-truth",
    default=None,
    help="ground-truth file for vocabulary/dataflow measurement "
    "(default: oracle/puzzle.gt.py sibling or same-directory <stem>.gt.py)",
  )
  return p


def main(argv: list[str] | None = None) -> int:
  args = build_arg_parser().parse_args(argv)
  return run_analyze(args.puzzle, args.json, args.ground_truth)


if __name__ == "__main__":
  sys.exit(main())

"""Shared framework for agent-free Codoku baselines.

A puzzle is inferred from its text alone: the masked spans, the kind of
each cell, the candidate domain of each kind, and the harness examples.
``PuzzleModel`` owns that inference and candidate assembly; ``Baseline``
owns the check loop.  A concrete solver therefore only implements
``propose()``: produce one candidate filling for the model.
"""

from __future__ import annotations

import ast
import bisect
import hashlib
import io
import itertools
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import tokenize
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

_CODOKU_DIR = Path(__file__).resolve().parent.parent
if str(_CODOKU_DIR) not in sys.path:
  sys.path.insert(0, str(_CODOKU_DIR))

from codoku_checker import (  # noqa: E402
  CheckFailure,
  parse_puzzle_requirements,
)
from codoku_common import (  # noqa: E402
  BINARY_OP_SPANS,
  COMPARISON_OP_SPANS,
  CONTROL_FLOW_KEYWORDS,
  IFEXP_KEYWORDS,
  INTERNAL_HELPER_FUNCS,
  UNARY_OP_SPANS,
  apply_replacements,
  block_label_for_line,
  collect_canonical_cells,
  collect_python_leaf_locals,
  count_harness_examples,
  find_python_block_comments,
  find_python_leaf_function,
  get_byte_offsets,
  get_python_maskable_statements,
  stmt_is_on_live_path,
  strip_refractir_prefix,
)

MASK_TOKEN_RE = re.compile(r"<FILL_[A-Z_]+>")
COMPOUND_GO_RE = re.compile(r"_go_<FILL_LABEL>")
COMPOUND_CTRL_RE = re.compile(r"_<FILL_CTRL>_<FILL_LABEL>")
PLACEHOLDER_NAME_RE = re.compile(r"^zz[a-z]+\d+$")

SENTINEL_CONSTANTS = frozenset({"0", "1", "0.0", "1.0"})
CHECKER_CLI = Path(__file__).resolve().parent.parent / "codoku.py"
CHECK_TIMEOUT = 10.0
CHECK_MEMORY_LIMIT = 1024 * 1024 * 1024
WORD_OPERATORS = frozenset({"not", "is", "is not", "in", "not in", "if", "else"})
ALL_OP_SYMBOLS = tuple(
  dict.fromkeys(
    symbol.decode("ascii")
    for spans in (
      BINARY_OP_SPANS,
      COMPARISON_OP_SPANS,
      UNARY_OP_SPANS,
      IFEXP_KEYWORDS,
    )
    for symbol in spans
  )
)

OP_REPAIR_ORDER = ("if", "else", "-", "not", "==", "//", "*", "+", "~", "&", "|", "^")


class PuzzleModelError(Exception):
  """The puzzle text cannot be turned into a cell model."""


class UnsupportedPuzzle(Exception):
  """The puzzle uses a construct outside a baseline's fragment."""


class SolverUnavailable(Exception):
  """The external SMT solver is not installed."""


class SolverTimeout(Exception):
  """The external SMT solver exceeded its per-call deadline."""


@dataclass
class Cell:
  """One masked cell, in puzzle coordinates."""

  index: int
  start: int
  end: int
  probe_start: int
  probe_end: int
  kind: str
  token: str
  stmt_index: int
  is_lhs: bool
  line: int
  block: str | None
  live: bool
  domain: list[str]
  context: str = ""
  prefix: str = ""

  @property
  def is_constant(self) -> bool:
    return self.kind == "CONST"

  @property
  def value_choices(self) -> int:
    return max(1, len(self.domain))


@dataclass
class BaselineOptions:
  """Options shared by the three baselines."""

  seed: int = 0
  smt_solver: str = "z3"
  solver_timeout: float = 10.0
  const_pool_size: int = 64
  max_enumeration: int = 200000
  max_path_examples: int = 16
  smt_domain_limit: int = 0
  keep_solver_files: str | None = None
  verbose: bool = False
  memory_path: str | None = None
  memory_flush_every: int = 1000
  memory_max_entries: int = 100_000


class SolutionMemory:
  """Bounded SHA256 memory of candidate solutions.

  Only 32-byte digests are kept in memory, so a run with many candidates
  stays small.  When *path* is set, digests are appended to disk every
  *flush_every* additions; reloading the file restores the memory across
  runs.  The in-memory set is bounded by *max_entries* (oldest evicted).
  """

  RECORD_SIZE = 32

  def __init__(
    self,
    path: str | Path | None = None,
    *,
    flush_every: int = 1000,
    max_entries: int = 100_000,
  ) -> None:
    self.path = Path(path) if path is not None else None
    self.flush_every = max(1, flush_every)
    self.max_entries = max(1, max_entries)
    self._seen: set[bytes] = set()
    self._order: deque[bytes] = deque()
    self._pending: list[bytes] = []
    if self.path is not None and self.path.exists():
      self._load()

  def _load(self) -> None:
    with self.path.open("rb") as handle:
      while True:
        record = handle.read(self.RECORD_SIZE)
        if len(record) < self.RECORD_SIZE:
          break
        self._remember(record, persist=False)

  def _remember(self, digest: bytes, *, persist: bool) -> None:
    if digest in self._seen:
      return
    self._seen.add(digest)
    self._order.append(digest)
    if len(self._order) > self.max_entries:
      self._seen.discard(self._order.popleft())
    if persist:
      self._pending.append(digest)
      if len(self._pending) >= self.flush_every:
        self.flush()

  @staticmethod
  def digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()

  def seen_text(self, text: str) -> bool:
    return self.digest(text) in self._seen

  def add_text(self, text: str) -> None:
    self._remember(self.digest(text), persist=self.path is not None)

  def flush(self) -> None:
    if self.path is None or not self._pending:
      return
    self.path.parent.mkdir(parents=True, exist_ok=True)
    if self.path.exists():
      size = self.path.stat().st_size
      if size % self.RECORD_SIZE:
        with self.path.open("r+b") as handle:
          handle.truncate(size - size % self.RECORD_SIZE)
    with self.path.open("ab") as handle:
      handle.write(b"".join(self._pending))
      handle.flush()
    self._pending.clear()

  def close(self) -> None:
    self.flush()

  def __len__(self) -> int:
    return len(self._seen)


@dataclass
class RunResult:
  """Outcome of one baseline run on one puzzle."""

  solver: str
  status: str
  attempts: int
  elapsed: float
  solution: str | None = None
  message: str = ""
  check_result: str | None = None
  metadata: dict = field(default_factory=dict)


@dataclass
class HarnessExample:
  """One replay in ``main``: concrete arguments and expected return."""

  args: list[int | float]
  expected: int | float


@dataclass
class _Occurrence:
  start: int
  end: int
  token: str
  kind: str
  replacement: str = ""


def _comment_start(line: str) -> int:
  """Index of the first textual comment marker outside a string."""
  quote = None
  escaped = False
  for idx, char in enumerate(line):
    if quote:
      if escaped:
        escaped = False
      elif char == "\\":
        escaped = True
      elif char == quote:
        quote = None
    elif char in ("'", '"'):
      quote = char
    elif char == "#":
      return idx
  return len(line)


def _kind_of_token(token: str) -> str:
  return token.removeprefix("<FILL_").removesuffix(">")


def _scan_masked_occurrences(text: str) -> list[_Occurrence]:
  """Find code mask tokens, treating each goto-flag compound as one cell."""
  occurrences: list[_Occurrence] = []
  offset = 0
  for line in text.splitlines(keepends=True):
    stripped = line.lstrip()
    if stripped.startswith("#"):
      offset += len(line)
      continue
    code_end = _comment_start(line)
    code = line[:code_end]
    taken: list[tuple[int, int]] = []

    for match in COMPOUND_GO_RE.finditer(code):
      occurrences.append(
        _Occurrence(
          offset + match.start(),
          offset + match.end(),
          "_go_<FILL_LABEL>",
          "FLAG_GO",
        )
      )
      taken.append((match.start(), match.end()))
    for match in COMPOUND_CTRL_RE.finditer(code):
      occurrences.append(
        _Occurrence(
          offset + match.start(),
          offset + match.end(),
          "_<FILL_CTRL>_<FILL_LABEL>",
          "FLAG_CTRL",
        )
      )
      taken.append((match.start(), match.end()))

    for match in MASK_TOKEN_RE.finditer(code):
      if any(lo <= match.start() and match.end() <= hi for lo, hi in taken):
        continue
      token = match.group(0)
      occurrences.append(
        _Occurrence(
          offset + match.start(),
          offset + match.end(),
          token,
          _kind_of_token(token),
        )
      )
    offset += len(line)

  occurrences.sort(key=lambda occ: (occ.start, occ.end))
  return occurrences


def _replacement_text(kind: str, index: int) -> str:
  if kind == "VAR":
    return f"zzvar{index}"
  if kind == "CONST":
    return str(900000000000 + index)
  if kind == "OP":
    return "+"
  if kind == "FUNC":
    return f"zzfun{index}"
  if kind == "CTRL":
    return "break"
  if kind == "FLAG_GO":
    return f"_go_zzlbl{index}"
  if kind == "FLAG_CTRL":
    return f"_break_zzlbl{index}"
  if kind == "LABEL":
    return f"zzlbl{index}"
  if kind == "TYPE":
    return "int"
  if kind == "FIELD":
    return "zzfield"
  raise PuzzleModelError(f"unknown mask kind {kind!r}")


@dataclass
class _Probe:
  text: str
  spans: dict[int, tuple[int, int]]
  occurrences: list[_Occurrence]


def _render_probe(text: str, occurrences: list[_Occurrence]) -> _Probe:
  replacements = sorted(occurrences, key=lambda occ: occ.start)
  chunks: list[str] = []
  spans: dict[int, tuple[int, int]] = {}
  pos = 0
  out_len = 0
  for occ in replacements:
    if occ.start < pos:
      raise PuzzleModelError("overlapping mask spans")
    fixed = text[pos : occ.start]
    chunks.append(fixed)
    out_len += len(fixed)
    start = out_len
    chunks.append(occ.replacement)
    out_len += len(occ.replacement)
    spans[id(occ)] = (start, out_len)
    pos = occ.end
  chunks.append(text[pos:])
  return _Probe("".join(chunks), spans, occurrences)


def _parse_error_offset(probe_text: str, error: SyntaxError) -> int:
  lineno = error.lineno or 1
  offset = error.offset or 1
  lines = probe_text.splitlines(keepends=True)
  return sum(len(line) for line in lines[: lineno - 1]) + max(0, offset - 1)


def _token_repair(text: str, occurrences: list[_Occurrence]) -> dict[int, str]:
  """Match visible ternary keywords against masked counterparts by depth.

  ``if`` and ``else`` in one conditional expression sit at the same
  parenthesis depth.  The masked side is the nearest unassigned operator
  cell at that depth before an ``else`` or after an ``if``.
  """
  replacements = []
  token_to_index: dict[str, int] = {}
  for index, occ in enumerate(occurrences):
    token = _token_placeholder(occ.kind, index)
    if occ.kind == "OP":
      token_to_index[token] = index
    replacements.append((occ.start, occ.end, f" {token} "))
  token_text = apply_replacements(text.encode("utf-8"), replacements).decode("utf-8")
  # Indentation is irrelevant to keyword matching; dedenting keeps
  # tokenize from rejecting a replacement that shifts a line's first column.
  token_text = "\n".join(line.lstrip() for line in token_text.splitlines())
  try:
    tokens = list(tokenize.generate_tokens(io.StringIO(token_text).readline))
  except (tokenize.TokenError, IndentationError, SyntaxError):
    return {}

  significant = []
  line_starts: list[bool] = []
  line_ids: list[int] = []
  at_line_start = True
  logical_line = 0
  for token in tokens:
    if token.type == tokenize.NEWLINE:
      logical_line += 1
      at_line_start = True
      continue
    if token.type in {
      tokenize.NL,
      tokenize.INDENT,
      tokenize.DEDENT,
      tokenize.COMMENT,
      tokenize.ENCODING,
      tokenize.ENDMARKER,
    }:
      continue
    significant.append(token)
    line_starts.append(at_line_start)
    line_ids.append(logical_line)
    at_line_start = False

  depths: list[int] = []
  depth = 0
  for token in significant:
    if token.string in (")", "]", "}"):
      depth -= 1
    depths.append(depth)
    if token.string in ("(", "[", "{"):
      depth += 1

  def is_expression_if(position: int) -> bool:
    return significant[position].string == "if" and not line_starts[position]

  def is_expression_else(position: int) -> bool:
    token = significant[position]
    if token.string != "else" or line_starts[position]:
      return False
    if position + 1 >= len(significant):
      return False
    return significant[position + 1].string != ":"

  assigned_if: set[int] = set()
  assigned_else: set[int] = set()

  def masked_at(position: int) -> int | None:
    token = significant[position]
    if token.type != tokenize.NAME:
      return None
    return token_to_index.get(token.string)

  for position, token in enumerate(significant):
    if is_expression_else(position):
      level = depths[position]
      matched = False
      for back in range(position - 1, -1, -1):
        if line_ids[back] != line_ids[position] or depths[back] < level:
          break
        if depths[back] != level:
          continue
        if is_expression_if(back):
          matched = True
          break
        index = masked_at(back)
        if index is not None and index in assigned_if:
          matched = True
          break
        index = masked_at(back)
        if index is not None and index in assigned_else:
          continue
      if matched:
        continue
      for back in range(position - 1, -1, -1):
        if line_ids[back] != line_ids[position] or depths[back] < level:
          break
        if depths[back] != level:
          continue
        index = masked_at(back)
        if index is None or index in assigned_if or index in assigned_else:
          continue
        assigned_if.add(index)
        break
    elif is_expression_if(position):
      level = depths[position]
      matched = False
      for forward in range(position + 1, len(significant)):
        if line_ids[forward] != line_ids[position] or depths[forward] < level:
          break
        if depths[forward] != level:
          continue
        if is_expression_else(forward):
          matched = True
          break
        index = masked_at(forward)
        if index is not None and index in assigned_else:
          matched = True
          break
      if matched:
        continue
      for forward in range(position + 1, len(significant)):
        if line_ids[forward] != line_ids[position] or depths[forward] < level:
          break
        if depths[forward] != level:
          continue
        index = masked_at(forward)
        if index is None or index in assigned_if or index in assigned_else:
          continue
        assigned_else.add(index)
        break

  choices: dict[int, str] = {}
  for index in assigned_if:
    choices[index] = "if"
  for index in assigned_else:
    choices[index] = "else"
  return choices


def _token_placeholder(kind: str, index: int) -> str:
  return f"ZZ{kind}{index}"


def _parse_with_repair(
  text: str, occurrences: list[_Occurrence]
) -> tuple[_Probe, ast.Module]:
  """Build the probe, repairing masked operators until it parses.

  The all-``+`` filling parses most puzzles.  A masked ternary keyword can
  turn the same cells into a different grammar, so errors are repaired by a
  breadth-first search over the nearby operator and control cells.  Only
  ``if``/``else`` (and ``break``/``continue``) change the grammar; the other
  operators parse in every operand position.
  """
  for index, occ in enumerate(occurrences):
    occ.replacement = _replacement_text(occ.kind, index)
  for index, choice in _token_repair(text, occurrences).items():
    occurrences[index].replacement = _pad_probe_choice(choice)

  repair_occurrences = [occ for occ in occurrences if occ.kind in {"OP", "CTRL"}]
  initial = tuple(occ.replacement for occ in repair_occurrences)
  queue: deque[tuple[str, ...]] = deque([initial])
  tried: set[tuple[str, ...]] = {initial}
  nodes = 0
  max_nodes = 3000

  def apply_state(state: tuple[str, ...]) -> None:
    for occ, value in zip(repair_occurrences, state):
      occ.replacement = value

  while queue and nodes < max_nodes:
    state = queue.popleft()
    nodes += 1
    apply_state(state)
    probe = _render_probe(text, occurrences)
    try:
      return probe, ast.parse(probe.text)
    except SyntaxError as error:
      position = _parse_error_offset(probe.text, error)
      candidates = _error_candidates(
        probe, occurrences, repair_occurrences, error, position
      )
      preferred = _preferred_choices(error.msg)
      for occ in candidates:
        index = repair_occurrences.index(occ)
        current = state[index]
        for choice in _neighbor_choices(occ, current, preferred):
          replacement = _pad_probe_choice(choice)
          if replacement == current:
            continue
          neighbor = state[:index] + (replacement,) + state[index + 1 :]
          if neighbor in tried:
            continue
          tried.add(neighbor)
          queue.append(neighbor)

  raise PuzzleModelError("cannot build a parseable probe within the repair budget")


def _error_candidates(
  probe: _Probe,
  occurrences: list[_Occurrence],
  repair_occurrences: list[_Occurrence],
  error: SyntaxError,
  position: int,
) -> list[_Occurrence]:
  """Pick nearby cells whose replacement can repair this syntax error."""
  del occurrences
  line = error.lineno or 0
  same_line = [
    occ
    for occ in repair_occurrences
    if _line_of(probe.text, probe.spans[id(occ)][0]) == line
  ]
  pool = same_line or repair_occurrences
  window = probe.text[max(0, position - 20) : position + 20]
  if "expected 'else'" in error.msg:
    after = [occ for occ in pool if probe.spans[id(occ)][0] >= position]
    after.sort(key=lambda occ: abs(probe.spans[id(occ)][0] - position))
    return after[:12]
  if "else" in window:
    before = [occ for occ in pool if probe.spans[id(occ)][0] < position]
    before.sort(key=lambda occ: -abs(probe.spans[id(occ)][0] - position))
    return before[:12]
  ordered = sorted(pool, key=lambda occ: abs(probe.spans[id(occ)][0] - position))
  return ordered[:12]


def _neighbor_choices(
  occ: _Occurrence, current: str, preferred: tuple[str, ...]
) -> tuple[str, ...]:
  if occ.kind == "CTRL":
    choices = [choice for choice in ("break", "continue") if choice != current]
  else:
    choices = [choice for choice in ("if", "else", "not") if choice != current]
    if current not in ("+", ""):
      choices.append("+")
  choices.sort(
    key=lambda choice: preferred.index(choice) if choice in preferred else 99
  )
  return tuple(choices)


def _preferred_choices(message: str) -> tuple[str, ...]:
  if "expected 'else'" in message:
    return ("else", "if", "+", "-", "not")
  if "expected ':'" in message or "invalid syntax" in message:
    return ("if", "else", "not", "+", "-")
  return ("if", "else", "not", "+", "-", "==", "//", "*", "&", "|", "^")


def _pad_probe_choice(choice: str) -> str:
  return f" {choice} " if choice in WORD_OPERATORS else choice


def _op_span(
  src: bytes,
  left: ast.AST,
  right: ast.AST,
  symbols: tuple[bytes, ...],
) -> tuple[int, int] | None:
  end_left = get_byte_offsets(
    src, left.lineno, left.col_offset, left.end_lineno, left.end_col_offset
  )[1]
  start_right = get_byte_offsets(
    src, right.lineno, right.col_offset, right.end_lineno, right.end_col_offset
  )[0]
  sl = src[end_left:start_right]
  for symbol in symbols:
    idx = sl.find(symbol)
    if idx != -1:
      return end_left + idx, end_left + idx + len(symbol)
  return None


def _annotate_op_contexts(
  leaf: ast.FunctionDef, src: bytes
) -> dict[tuple[int, int], str]:
  """Map operator spans to their syntactic construct."""
  contexts: dict[tuple[int, int], str] = {}
  for node in ast.walk(leaf):
    if isinstance(node, ast.BinOp):
      span = _op_span(src, node.left, node.right, BINARY_OP_SPANS)
      if span:
        contexts[span] = "binary"
    elif isinstance(node, ast.Compare):
      previous = node.left
      for op, comparator in zip(node.ops, node.comparators):
        del op
        span = _op_span(src, previous, comparator, COMPARISON_OP_SPANS)
        if span:
          contexts[span] = "comparison"
        previous = comparator
    elif isinstance(node, ast.UnaryOp):
      op_start = get_byte_offsets(
        src, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset
      )[0]
      operand_start = get_byte_offsets(
        src,
        node.operand.lineno,
        node.operand.col_offset,
        node.operand.end_lineno,
        node.operand.end_col_offset,
      )[0]
      sl = src[op_start:operand_start]
      for symbol in UNARY_OP_SPANS:
        mark = sl.find(symbol)
        if mark != -1:
          contexts[(op_start + mark, op_start + mark + len(symbol))] = "unary"
          break
    elif isinstance(node, ast.IfExp):
      body_end = get_byte_offsets(
        src,
        node.body.lineno,
        node.body.col_offset,
        node.body.end_lineno,
        node.body.end_col_offset,
      )[1]
      test_start = get_byte_offsets(
        src,
        node.test.lineno,
        node.test.col_offset,
        node.test.end_lineno,
        node.test.end_col_offset,
      )[0]
      mark = src[body_end:test_start].find(IFEXP_KEYWORDS[0])
      if mark != -1:
        contexts[(body_end + mark, body_end + mark + len(IFEXP_KEYWORDS[0]))] = (
          "ternary_if"
        )
      test_end = get_byte_offsets(
        src,
        node.test.lineno,
        node.test.col_offset,
        node.test.end_lineno,
        node.test.end_col_offset,
      )[1]
      orelse_start = get_byte_offsets(
        src,
        node.orelse.lineno,
        node.orelse.col_offset,
        node.orelse.end_lineno,
        node.orelse.end_col_offset,
      )[0]
      mark = src[test_end:orelse_start].find(IFEXP_KEYWORDS[1])
      if mark != -1:
        contexts[(test_end + mark, test_end + mark + len(IFEXP_KEYWORDS[1]))] = (
          "ternary_else"
        )
  return contexts


def _annotate_call_contexts(
  leaf: ast.FunctionDef, src: bytes
) -> dict[tuple[int, int], str]:
  """Map an argument span to a variable-role hint."""
  hints: dict[tuple[int, int], str] = {}
  list_first = {"_rd", "_vrd", "_Ptr"}
  pointer_first = {
    "_load",
    "_store",
    "_padd",
    "_pdiff",
    "_peq",
    "_prel",
    "_pidx",
    "_pfield",
    "_chk_ptr",
    "_chk_deref",
    "_live",
  }
  for node in ast.walk(leaf):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
      continue
    name = node.func.id
    for idx, arg in enumerate(node.args):
      span = get_byte_offsets(
        src, arg.lineno, arg.col_offset, arg.end_lineno, arg.end_col_offset
      )
      if name in list_first and idx == 0:
        hint = "list"
      elif name in pointer_first and idx == 0:
        hint = "pointer"
      elif name == "_cast_int" and idx == 0:
        hint = "scalar"
      else:
        hint = "scalar"
      hints[span] = hint
    for keyword in node.keywords:
      span = get_byte_offsets(
        src,
        keyword.value.lineno,
        keyword.value.col_offset,
        keyword.value.end_lineno,
        keyword.value.end_col_offset,
      )
      hints[span] = "scalar"
  return hints


def _contained_key(
  hints: dict[tuple[int, int], str], start: int, end: int
) -> str | None:
  best: tuple[int, str] | None = None
  for (lo, hi), hint in hints.items():
    if lo <= start and end <= hi:
      size = hi - lo
      if best is None or size < best[0]:
        best = (size, hint)
  return best[1] if best else None


def _infer_variable_roles(leaf: ast.FunctionDef) -> tuple[set[str], set[str], set[str]]:
  """Split locals into list-valued, pointer-valued, and vector-object names."""
  lists: set[str] = set()
  pointers: set[str] = set()
  vectors: set[str] = set()
  for node in ast.walk(leaf):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
      continue
    target = node.targets[0]
    if not isinstance(target, ast.Name):
      continue
    name = target.id
    value = node.value
    if isinstance(value, ast.List):
      lists.add(name)
    elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
      if value.func.id == "_Ptr":
        pointers.add(name)
      elif value.func.id.startswith("_vec_"):
        vectors.add(name)
  return lists, pointers, vectors


def _infer_list_indices(leaf: ast.FunctionDef) -> dict[str, list[int]]:
  """Plausible subscript indices for every list-valued local."""
  lengths: dict[str, int] = {}
  seen: dict[str, set[int]] = {}
  for node in ast.walk(leaf):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      target, value = node.targets[0], node.value
      if isinstance(target, ast.Name) and isinstance(value, ast.List):
        lengths[target.id] = len(value.elts)
    if isinstance(node, ast.Subscript):
      base = node.value
      while isinstance(base, ast.Subscript):
        base = base.value
      if isinstance(base, ast.Name) and isinstance(node.slice, ast.Constant):
        if isinstance(node.slice.value, int):
          seen.setdefault(base.id, set()).add(int(node.slice.value))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
      if node.func.id in {"_rd", "_vrd", "_Ptr"} and len(node.args) >= 2:
        base, index = node.args[0], node.args[1]
        if isinstance(base, ast.Name) and isinstance(index, ast.Constant):
          if isinstance(index.value, int):
            seen.setdefault(base.id, set()).add(int(index.value))
  result: dict[str, list[int]] = {}
  for name, length in lengths.items():
    indices = set(range(min(max(length, 1), 16))) | seen.get(name, set())
    result[name] = sorted(idx for idx in indices if idx >= 0)
  for name, indices in seen.items():
    if name not in result:
      result[name] = sorted(idx for idx in indices if idx >= 0)
  return result


def _subscript_domain(names: set[str], indices: dict[str, list[int]]) -> list[str]:
  out: list[str] = []
  for name in sorted(names):
    for idx in indices.get(name, [0]):
      out.append(f"{name}[{idx}]")
  return out


def _free_const_pool(text: str, limit: int) -> list[str]:
  """A finite, type-aware pool for puzzles without a constant budget."""
  values: list[str] = []
  seen: set[str] = set()

  def add(value: str) -> None:
    if value in SENTINEL_CONSTANTS or value in seen:
      return
    seen.add(value)
    values.append(value)

  for line in text.splitlines():
    if line.lstrip().startswith("#"):
      continue
    code = line[: _comment_start(line)]
    for literal in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", code):
      add(literal)
  for value in (
    2,
    3,
    4,
    5,
    7,
    8,
    16,
    31,
    32,
    64,
    127,
    128,
    255,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32767,
    32768,
    65535,
    65536,
    1048576,
    1073741824,
    2147483647,
  ):
    add(str(value))
  target = min(limit, 4096)
  candidate = 2
  while len(values) < target and candidate <= target * 4 + 1024:
    add(str(candidate))
    candidate += 1
  return values[:limit]


def _budget_lists(
  budget: dict[str, int | tuple[int, int]],
) -> tuple[list[str], list[str], list[str], bool]:
  """Expand a budget into total, live, and dead multisets."""
  total: list[str] = []
  live: list[str] = []
  dead: list[str] = []
  split = any(isinstance(value, tuple) for value in budget.values())
  for value, count in budget.items():
    if isinstance(count, tuple):
      live_count, dead_count = count
    else:
      live_count, dead_count = count, 0
    total.extend([value] * (live_count + dead_count))
    live.extend([value] * live_count)
    dead.extend([value] * dead_count)
  if not split:
    live = list(total)
  return total, live, dead, split


def _block_comments(text: str) -> list[tuple[int, int, str, int]]:
  return find_python_block_comments(text.encode("utf-8"))


def _line_of(text: str, offset: int) -> int:
  return text.count("\n", 0, offset) + 1


class PuzzleModel:
  """A puzzle's masked cells, domains, and harness examples."""

  def __init__(
    self,
    text: str,
    *,
    path: Path | None = None,
    options: BaselineOptions | None = None,
  ) -> None:
    self.text = strip_refractir_prefix(text.encode("utf-8")).decode("utf-8")
    self.path = path
    self.options = options or BaselineOptions()

    try:
      requirements = parse_puzzle_requirements(self.text)
    except CheckFailure as error:
      raise PuzzleModelError(error.message) from error

    self.expected_path = requirements.expected_path
    self.cfg_edges = requirements.cfg_edges
    if not self.expected_path:
      raise PuzzleModelError("puzzle is missing //@ EXEC_PATH")
    if not self.cfg_edges:
      raise PuzzleModelError("puzzle is missing //@ CFG_EDGE markers")
    self.disabled_masks = requirements.disabled_masks
    self.const_budget = requirements.const_budget
    self.nodes = sorted(
      {node for edge in self.cfg_edges for node in edge} | set(self.expected_path)
    )
    self.live_blocks = frozenset(self.expected_path)
    self.n_examples = count_harness_examples(self.text)

    occurrences = _scan_masked_occurrences(self.text)
    if not occurrences:
      raise PuzzleModelError("puzzle has no mask cells")
    probe, tree = _parse_with_repair(self.text, occurrences)
    self.probe_text = probe.text
    self.probe_bytes = probe.text.encode("utf-8")
    self.probe_spans = probe.spans
    self.occurrences = occurrences

    leaf, leaf_name = find_python_leaf_function(tree, self.probe_bytes)
    if leaf is None:
      raise PuzzleModelError("probe has no leaf function")
    self.tree = tree
    self.leaf = leaf
    self.leaf_name = leaf_name

    maskable, entry_line, exit_line = get_python_maskable_statements(
      leaf, self.probe_bytes
    )
    if not maskable or not entry_line or not exit_line:
      raise PuzzleModelError("probe has no maskable statements or block markers")
    self.maskable = maskable
    self.entry_line = entry_line
    self.exit_line = exit_line
    self.comments = _block_comments(self.probe_text)

    probe_local_names = collect_python_leaf_locals(leaf)
    placeholder_vars = {occ.replacement for occ in occurrences if occ.kind == "VAR"}
    placeholder_funcs = {occ.replacement for occ in occurrences if occ.kind == "FUNC"}
    local_names = {
      name for name in probe_local_names if not PLACEHOLDER_NAME_RE.match(name)
    }
    self.local_names = local_names
    self.placeholder_names = placeholder_vars | placeholder_funcs
    self.defined_funcs = {
      node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    canonical, _lhs, _const_values = collect_canonical_cells(
      maskable,
      entry_line,
      self.probe_bytes,
      probe_local_names | placeholder_vars,
      self.defined_funcs | placeholder_funcs,
      self.disabled_masks,
    )

    occ_spans = [
      (probe.spans[id(occ)][0], probe.spans[id(occ)][1], occ)
      for occ in occurrences
      if occ.kind not in {"TYPE", "FIELD", "LABEL"}
    ]

    def match_occurrence(start: int, end: int) -> _Occurrence | None:
      best: tuple[int, _Occurrence] | None = None
      for lo, hi, occurrence in occ_spans:
        if lo <= start and end <= hi:
          size = hi - lo
          if best is None or size < best[0]:
            best = (size, occurrence)
      return best[1] if best else None

    op_contexts = _annotate_op_contexts(leaf, self.probe_bytes)
    call_contexts = _annotate_call_contexts(leaf, self.probe_bytes)
    lists, pointers, vectors = _infer_variable_roles(leaf)
    list_indices = _infer_list_indices(leaf)
    lists = {name for name in lists if not PLACEHOLDER_NAME_RE.match(name)}
    pointers = {name for name in pointers if not PLACEHOLDER_NAME_RE.match(name)}
    vectors = {name for name in vectors if not PLACEHOLDER_NAME_RE.match(name)}
    list_indices = {
      name: indices
      for name, indices in list_indices.items()
      if not PLACEHOLDER_NAME_RE.match(name)
    }

    scalar_names = set(local_names) - lists - vectors
    self.list_names = lists
    self.pointer_names = pointers
    self.vector_names = vectors
    self.scalar_names = scalar_names
    self.list_indices = list_indices

    total_budget, live_budget, dead_budget, split = _budget_lists(self.const_budget)
    self.const_split = split
    self.const_total_pool = total_budget
    self.const_live_pool = live_budget
    self.const_dead_pool = dead_budget
    self.const_pool = (
      total_budget
      if total_budget
      else _free_const_pool(self.text, max(1, self.options.const_pool_size))
    )

    cells: list[Cell] = []
    for start, end, token, stmt_index, is_lhs in canonical:
      occ = match_occurrence(start, end)
      if occ is None:
        continue
      kind = occ.kind
      if token.startswith("_go_<FILL_LABEL>"):
        kind = "FLAG_GO"
      elif token.startswith("_<FILL_CTRL>_<FILL_LABEL>"):
        kind = "FLAG_CTRL"
      line = _line_of(self.probe_text, start)
      block = block_label_for_line(self.comments, line)
      live = stmt_is_on_live_path(
        self.comments, maskable[stmt_index].lineno, self.live_blocks
      )
      context = _contained_key(call_contexts, start, end) or ""
      if kind == "OP":
        context = op_contexts.get((start, end), "binary")
      probe_start, probe_end = start, end
      cell = Cell(
        index=len(cells),
        start=occ.start,
        end=occ.end,
        probe_start=probe_start,
        probe_end=probe_end,
        kind=kind,
        token=occ.token,
        stmt_index=stmt_index,
        is_lhs=is_lhs,
        line=line,
        block=block,
        live=live,
        domain=[],
        context=context,
        prefix=occ.token.split("<", 1)[0],
      )
      cells.append(cell)

    if not cells:
      raise PuzzleModelError("no masked cell survived canonical inference")
    self.cells = cells
    self.cells_by_start = {cell.start: cell for cell in cells}
    self.cells_by_probe_span = {
      (cell.probe_start, cell.probe_end): cell for cell in cells
    }
    self.bound_names_at = self._bound_names(leaf, cells)
    self.ternary_if_cells, self.ternary_else_cells = self._ternary_pairs(leaf)
    for cell in cells:
      cell.domain = self._domain_for(cell)
    self.harness_examples = self._harness_examples(tree)
    self._validate_const_budget()

  @classmethod
  def from_file(
    cls, path: str | Path, options: BaselineOptions | None = None
  ) -> PuzzleModel:
    puzzle = Path(path)
    return cls(puzzle.read_text(), path=puzzle, options=options)

  def _ternary_pairs(self, leaf: ast.FunctionDef) -> tuple[set[int], set[int]]:
    """Cells that can form ``A <OP> B <OP> C`` as ``A if B else C``.

    The probe freezes one parse, so a boolean chain whose two operators are
    both masked reads as a binary chain.  The same cells also admit the
    ternary completion, so its keywords join their domains.
    """
    inner_cells: set[int] = set()
    outer_cells: set[int] = set()
    for node in ast.walk(leaf):
      if not isinstance(node, ast.BinOp) or not isinstance(node.left, ast.BinOp):
        continue
      inner = node.left
      if (
        isinstance(inner.left, ast.BinOp)
        or isinstance(inner.right, ast.BinOp)
        or isinstance(node.right, ast.BinOp)
      ):
        continue
      inner_span = _op_span(self.probe_bytes, inner.left, inner.right, BINARY_OP_SPANS)
      outer_span = _op_span(self.probe_bytes, node.left, node.right, BINARY_OP_SPANS)
      inner_cell = self.cells_by_probe_span.get(inner_span)
      outer_cell = self.cells_by_probe_span.get(outer_span)
      if (
        inner_cell is not None
        and outer_cell is not None
        and inner_cell.kind == "OP"
        and outer_cell.kind == "OP"
      ):
        inner_cells.add(inner_cell.index)
        outer_cells.add(outer_cell.index)
    return inner_cells, outer_cells

  def _bound_names(
    self, leaf: ast.FunctionDef, cells: list[Cell]
  ) -> dict[int, set[str]]:
    """Names definitely bound before each cell, by source order.

    Pre-entry and on-path assignments bind visible names.  A masked
    assignment target hides its concrete name, so it binds nothing; using
    such a name on a right-hand side would be a local-before-assignment
    error even though the checker's re-masking accepts the token.
    """
    params = {arg.arg for arg in leaf.args.args}
    events: list[tuple[int, str]] = []
    for node in ast.walk(leaf):
      if isinstance(node, ast.Assign):
        targets = node.targets
      elif isinstance(node, ast.AugAssign):
        targets = [node.target]
      else:
        continue
      line = getattr(node, "lineno", 0)
      block = block_label_for_line(self.comments, line)
      if block is not None and block not in self.live_blocks:
        continue
      for target in targets:
        if not isinstance(target, ast.Name):
          continue
        name = target.id
        if PLACEHOLDER_NAME_RE.match(name):
          continue
        offset = get_byte_offsets(
          self.probe_bytes,
          node.lineno,
          node.col_offset,
          node.end_lineno,
          node.end_col_offset,
        )[1]
        events.append((offset, name))
    events.sort()
    offsets = [offset for offset, _ in events]
    result: dict[int, set[str]] = {}
    for cell in cells:
      cut = bisect.bisect_left(offsets, cell.probe_start)
      bound = set(params)
      bound.update(name for _, name in events[:cut])
      result[cell.index] = bound
    return result

  def _domain_for(self, cell: Cell) -> list[str]:
    if cell.kind == "CONST":
      if self.const_split:
        if cell.live:
          return sorted(set(self.const_live_pool))
        return sorted(set(self.const_dead_pool))
      if self.const_budget:
        return sorted(set(self.const_total_pool))
      return list(self.const_pool)
    if cell.kind == "OP":
      if cell.context == "unary":
        preferred = [sym.decode() for sym in UNARY_OP_SPANS]
      elif cell.context == "comparison":
        preferred = [sym.decode() for sym in COMPARISON_OP_SPANS]
      elif cell.context == "ternary_if":
        preferred = ["if"]
      elif cell.context == "ternary_else":
        preferred = ["else"]
      else:
        preferred = [sym.decode() for sym in BINARY_OP_SPANS]
        preferred += [sym.decode() for sym in COMPARISON_OP_SPANS]
        if cell.index in self.ternary_if_cells:
          preferred.append("if")
        if cell.index in self.ternary_else_cells:
          preferred.append("else")
      return _unique(preferred + list(ALL_OP_SYMBOLS))
    if cell.kind == "VAR":
      if cell.is_lhs:
        return _unique(
          sorted(self.local_names)
          + _subscript_domain(self.list_names, self.list_indices)
        )
      bound = self.bound_names_at.get(cell.index, set())

      def readable(names: list[str]) -> list[str]:
        if not bound:
          return names
        return [name for name in names if name.split("[", 1)[0] in bound]

      generic = _unique(
        sorted(self.scalar_names)
        + sorted(self.pointer_names)
        + sorted(self.list_names)
        + sorted(self.vector_names)
        + _subscript_domain(self.list_names, self.list_indices)
      )
      if cell.context == "list":
        preferred = sorted(self.list_names)
      elif cell.context == "pointer":
        preferred = sorted(self.pointer_names) + _subscript_domain(
          self.list_names, self.list_indices
        )
      elif cell.context == "vec":
        preferred = sorted(self.vector_names)
      elif cell.context == "scalar":
        preferred = sorted(self.scalar_names) + _subscript_domain(
          self.list_names, self.list_indices
        )
      else:
        preferred = generic
      candidates = readable(_unique(preferred + generic))
      return candidates or generic
    if cell.kind == "FUNC":
      return sorted(
        name
        for name in self.defined_funcs - INTERNAL_HELPER_FUNCS
        if not PLACEHOLDER_NAME_RE.match(name)
      )
    if cell.kind == "CTRL":
      return list(CONTROL_FLOW_KEYWORDS)
    if cell.kind == "FLAG_GO":
      return [f"_go_{node}" for node in self.nodes]
    if cell.kind == "FLAG_CTRL":
      return [
        f"{prefix}{node}"
        for prefix in ("_brk_", "_break_", "_cnt_", "_continue_")
        for node in self.nodes
      ]
    if cell.kind == "LABEL":
      return list(self.nodes)
    return ["None"]

  def _validate_const_budget(self) -> None:
    const_cells = [cell for cell in self.cells if cell.is_constant]
    if not const_cells:
      return
    if not self.const_budget:
      return
    if self.const_split:
      live_cells = sum(1 for cell in const_cells if cell.live)
      dead_cells = len(const_cells) - live_cells
      expected_live = sum(
        count[0] if isinstance(count, tuple) else count
        for count in self.const_budget.values()
      )
      expected_dead = sum(
        count[1] if isinstance(count, tuple) else 0
        for count in self.const_budget.values()
      )
      if (live_cells, dead_cells) != (expected_live, expected_dead):
        raise PuzzleModelError(
          "constant cells do not match the live/dead budget: "
          f"cells=({live_cells}, {dead_cells}) "
          f"budget=({expected_live}, {expected_dead})"
        )
    elif len(const_cells) != len(self.const_total_pool):
      raise PuzzleModelError(
        "constant cell count does not match the budget: "
        f"cells={len(const_cells)} budget={len(self.const_total_pool)}"
      )

  def _harness_examples(self, tree: ast.Module) -> list[HarnessExample]:
    main = next(
      (
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
      ),
      None,
    )
    if main is None:
      return []
    examples: list[HarnessExample] = []
    pending_args: list[list[int | float]] = []
    for stmt in ast.walk(main):
      if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
        continue
      call = stmt.value
      if isinstance(call.func, ast.Name) and call.func.id == self.leaf_name:
        args = _literal_args(call.args)
        if args is not None:
          pending_args.append(args)
      elif isinstance(call.func, ast.Name) and call.func.id == "_in_check_chksum":
        if len(call.args) != 2 or not pending_args:
          continue
        expected = _literal_value(call.args[0])
        if expected is None:
          continue
        args = pending_args.pop(0)
        examples.append(HarnessExample(args, expected))
    if not examples and pending_args:
      return []
    return examples[: self.options.max_path_examples]

  def assemble(self, values: list[str]) -> str:
    if len(values) != len(self.cells):
      raise PuzzleModelError(
        f"expected {len(self.cells)} cell values, got {len(values)}"
      )
    replacements = [
      (cell.start, cell.end, _pad_value(value))
      for cell, value in zip(self.cells, values)
    ]
    return apply_replacements(self.text.encode("utf-8"), replacements).decode("utf-8")

  def sample_values(self, rng: random.Random) -> list[str]:
    """Draw one type-aware filling, respecting the constant multiset."""
    values: list[str] = ["" for _ in self.cells]
    const_cells = [cell for cell in self.cells if cell.is_constant]
    if const_cells:
      if self.const_split:
        live_cells = [cell for cell in const_cells if cell.live]
        dead_cells = [cell for cell in const_cells if not cell.live]
        live_pool = list(self.const_live_pool)
        dead_pool = list(self.const_dead_pool)
        rng.shuffle(live_pool)
        rng.shuffle(dead_pool)
        for cell, value in zip(live_cells, live_pool):
          values[cell.index] = value
        for cell, value in zip(dead_cells, dead_pool):
          values[cell.index] = value
      elif self.const_budget:
        pool = list(self.const_total_pool)
        rng.shuffle(pool)
        for cell, value in zip(const_cells, pool):
          values[cell.index] = value
      else:
        for cell in const_cells:
          values[cell.index] = _random_free_const(rng)
    for cell in self.cells:
      if cell.is_constant:
        continue
      if not cell.domain:
        raise PuzzleModelError(f"cell {cell.index} ({cell.kind}) has no domain")
      values[cell.index] = rng.choice(cell.domain)
    return values

  def check_text(self, candidate: str) -> tuple[str, str]:
    """Run the shipped checker in an isolated, resource-bounded process.

    The checker executes the candidate.  A random filling can build a huge
    integer or loop, so the child gets a memory and CPU cap; the outer
    timeout is a second backstop.
    """
    with tempfile.NamedTemporaryFile(
      "w", suffix=".py", delete=False, encoding="utf-8"
    ) as puzzle_handle:
      puzzle_handle.write(self.text)
      puzzle_path = puzzle_handle.name
    with tempfile.NamedTemporaryFile(
      "w", suffix=".py", delete=False, encoding="utf-8"
    ) as handle:
      handle.write(candidate)
      solution_path = handle.name
    try:
      run = subprocess.run(
        _checker_command(Path(puzzle_path), solution_path),
        capture_output=True,
        text=True,
        timeout=CHECK_TIMEOUT + 5.0,
        preexec_fn=_child_limits,
      )
    except subprocess.TimeoutExpired:
      return "FAIL_TIMEOUT", "checker process exceeded its wall-clock cap"
    except OSError as error:
      return "ERROR", f"checker launch failed: {error}"
    finally:
      for path in (puzzle_path, solution_path):
        try:
          os.unlink(path)
        except OSError:
          pass
    output = (run.stdout or "") + (run.stderr or "")
    match = re.search(r"\[(PASS|FAIL_[A-Z_]+|ERROR)\]", output)
    if match is None:
      return "ERROR", output.strip() or f"checker exited {run.returncode}"
    result = match.group(1)
    message = output[match.end() :].strip()
    return result, message

  def domain_log10(self) -> float:
    total = 0.0
    for cell in self.cells:
      choices = cell.value_choices
      if choices <= 0:
        return float("-inf")
      total += math.log10(choices)
    return total


def _pad_value(value: str) -> str:
  """Keep word operators from gluing onto an adjacent operand."""
  if value in WORD_OPERATORS:
    return f" {value} "
  return value


def _checker_command(puzzle: Path, solution: str) -> list[str]:
  if CHECKER_CLI.exists():
    return [sys.executable, str(CHECKER_CLI), "check", str(puzzle), solution]
  checker = CHECKER_CLI.parent / "codoku_checker.py"
  if checker.exists():
    return [sys.executable, str(checker), str(puzzle), solution]
  raise OSError(f"no checker entry point near {CHECKER_CLI}")


def _child_limits() -> None:
  """Cap checker memory and CPU; unavailable limits are skipped."""
  try:
    import resource
  except ImportError:
    return
  try:
    resource.setrlimit(resource.RLIMIT_AS, (CHECK_MEMORY_LIMIT, CHECK_MEMORY_LIMIT))
  except (ValueError, OSError):
    pass
  try:
    resource.setrlimit(resource.RLIMIT_CPU, (int(CHECK_TIMEOUT), int(CHECK_TIMEOUT)))
  except (ValueError, OSError):
    pass


def _random_free_const(rng: random.Random) -> str:
  """Draw an integer with unbounded bit length, plus occasional floats."""
  if rng.random() < 0.05:
    raw = rng.getrandbits(64).to_bytes(8, "little")
    value = struct.unpack("<d", raw)[0]
    if math.isfinite(value) and value > 0 and value != 1.0:
      return repr(value)
  bits = 1 + int(rng.expovariate(1 / 16))
  value = rng.getrandbits(bits)
  if value in (0, 1):
    value = 2
  return str(value)


def _unique(items: list[str]) -> list[str]:
  seen: set[str] = set()
  out: list[str] = []
  for item in items:
    if item not in seen:
      seen.add(item)
      out.append(item)
  return out


def _literal_value(node: ast.AST) -> int | float | None:
  if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
    if isinstance(node.value, bool):
      return None
    return node.value
  if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
    value = _literal_value(node.operand)
    if value is None:
      return None
    return -value if isinstance(node.op, ast.USub) else value
  if (
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "_f32"
    and len(node.args) == 1
  ):
    return _literal_value(node.args[0])
  return None


def _literal_args(args: list[ast.AST]) -> list[int | float] | None:
  values: list[int | float] = []
  for arg in args:
    value = _literal_value(arg)
    if value is None:
      return None
    values.append(value)
  return values


class Baseline:
  """Shared budgeted wrapper around a one-candidate ``propose`` core."""

  name = "baseline"

  def __init__(
    self, model: PuzzleModel, options: BaselineOptions | None = None
  ) -> None:
    self.model = model
    self.options = options or BaselineOptions()
    self.rng = random.Random(self.options.seed)
    self._attempt = 0
    self.memory = SolutionMemory(
      self.options.memory_path,
      flush_every=self.options.memory_flush_every,
      max_entries=self.options.memory_max_entries,
    )

  def propose(self, remaining: float) -> str | None:
    raise NotImplementedError

  def on_failure(self, check_result: str, message: str) -> None:
    """Hook for check-guided solvers that want to block a failed choice."""

  def run(
    self,
    *,
    time_limit: float = 60.0,
    max_attempts: int = 100,
  ) -> RunResult:
    try:
      return self._run(time_limit=time_limit, max_attempts=max_attempts)
    finally:
      self.memory.close()

  def _run(
    self,
    *,
    time_limit: float,
    max_attempts: int,
  ) -> RunResult:
    started = time.monotonic()
    attempts = 0
    duplicates = 0
    timeouts = 0
    duplicate_limit = min(max(100, max_attempts), 1_000_000)
    last_result: str | None = None
    last_message = ""
    while attempts < max_attempts:
      remaining = time_limit - (time.monotonic() - started)
      if remaining <= 0:
        return RunResult(
          self.name,
          "timeout",
          attempts,
          time.monotonic() - started,
          message=last_message or "time budget exhausted",
          check_result=last_result,
        )
      try:
        candidate = self.propose(remaining)
      except UnsupportedPuzzle as error:
        return RunResult(
          self.name,
          "unsupported",
          attempts,
          time.monotonic() - started,
          message=str(error),
          check_result=last_result,
        )
      except SolverUnavailable as error:
        return RunResult(
          self.name,
          "unsupported",
          attempts,
          time.monotonic() - started,
          message=str(error),
          check_result=last_result,
        )
      except SolverTimeout as error:
        # A single hard query should not discard the remaining budget.
        timeouts += 1
        last_message = str(error)
        if timeouts > 100000:
          return RunResult(
            self.name,
            "timeout",
            attempts,
            time.monotonic() - started,
            message=last_message,
            check_result=last_result,
          )
        continue
      except Exception as error:  # noqa: BLE001
        return RunResult(
          self.name,
          "error",
          attempts,
          time.monotonic() - started,
          message=f"{type(error).__name__}: {error}",
          check_result=last_result,
        )
      if candidate is None:
        return RunResult(
          self.name,
          "exhausted",
          attempts,
          time.monotonic() - started,
          message="no further candidates",
          check_result=last_result,
        )
      if self.memory.seen_text(candidate):
        duplicates += 1
        if duplicates > duplicate_limit:
          return RunResult(
            self.name,
            "exhausted",
            attempts,
            time.monotonic() - started,
            message="only duplicate candidates were generated",
            check_result=last_result,
          )
        continue
      self.memory.add_text(candidate)
      attempts += 1
      result, message = self.model.check_text(candidate)
      last_result, last_message = result, message
      if result == "PASS":
        return RunResult(
          self.name,
          "solved",
          attempts,
          time.monotonic() - started,
          solution=candidate,
          check_result=result,
        )
      self.on_failure(result, message)
      if result == "ERROR":
        return RunResult(
          self.name,
          "error",
          attempts,
          time.monotonic() - started,
          message=message,
          check_result=result,
        )
    return RunResult(
      self.name,
      "exhausted",
      attempts,
      time.monotonic() - started,
      message=last_message or "attempt budget exhausted",
      check_result=last_result,
    )


def enumerate_templates(
  cells: list[Cell],
  rng: random.Random,
  *,
  limit: int | None = None,
) -> Iterator[dict[int, str]]:
  """Lazily enumerate non-constant fillings, smallest domains first."""
  chosen = [cell for cell in cells if not cell.is_constant]
  kind_order = {
    "VAR": 0,
    "FUNC": 1,
    "CTRL": 2,
    "FLAG_GO": 3,
    "FLAG_CTRL": 4,
    "OP": 5,
  }
  chosen.sort(
    key=lambda cell: (
      kind_order.get(cell.kind, 9),
      len(cell.domain),
      cell.index,
    )
  )
  domains: list[list[str]] = []
  for cell in chosen:
    domain = list(cell.domain)
    rng.shuffle(domain)
    domains.append(domain)
  produced = 0
  for combo in itertools.product(*domains):
    if limit is not None and produced >= limit:
      return
    produced += 1
    yield {cell.index: value for cell, value in zip(chosen, combo)}


def run_smt2(query: str, *, smt_solver: str, timeout: float) -> tuple[str, str]:
  """Run an SMT-LIB2 query through an external solver binary.

  A solver binary that cannot be executed (for example a mount with no
  execute permission) falls back to the other installed solver.  Both are
  invoked from a writable temporary directory so a solver that writes
  scratch files does not depend on the caller's working directory.
  """
  candidates = [smt_solver]
  if smt_solver == "z3":
    candidates.append("cvc5")
  last_error: OSError | None = None
  for name in candidates:
    binary = shutil.which(name)
    if binary is None:
      continue
    command = [binary, "-smt2", "-in"]
    if Path(binary).name.startswith("cvc5"):
      command = [binary, "--lang=smt2"]
    try:
      with tempfile.TemporaryDirectory(prefix="codoku_smt_") as workdir:
        environment = dict(os.environ)
        environment["TMPDIR"] = workdir
        run = subprocess.run(
          command,
          input=query,
          capture_output=True,
          text=True,
          timeout=timeout,
          cwd=workdir,
          env=environment,
        )
      return run.stdout, run.stderr
    except subprocess.TimeoutExpired as error:
      raise SolverTimeout(
        f"{name} exceeded the {timeout:g}s solver deadline"
      ) from error
    except OSError as error:
      last_error = error
      continue
  if last_error is not None:
    raise SolverUnavailable(f"cannot execute SMT solver {smt_solver!r}: {last_error}")
  raise SolverUnavailable(f"SMT solver {smt_solver!r} is not installed")

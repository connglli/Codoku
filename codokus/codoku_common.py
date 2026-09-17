"""codoku_common.py - Python-only masking and analysis helpers for codoku.

Python-only: mask tokens are the angle-bracketed <FILL_XXX> forms, and the
helpers cover only the Python target's masking and CFG needs.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Masking vocabularies
#
# Single source of truth for what each <FILL_XXX> kind can be filled with;
# used by the masker below and by the solution-space estimator in
# codoku_complexity.py.  Op symbols/keywords are byte spans because masking
# locates them inside byte slices; helper/function names are text.
# ---------------------------------------------------------------------------

# The instrumented checksum function every puzzle splices in.
GLOBAL_CHKSUM_FUNC = "_in_global_chksum"

# File-internal functions never masked as <FILL_FUNC>. The pointer/memory
# model helpers and ``_cast_int`` are excluded on purpose: a masked puzzle
# is a module a third party fills in, so their calls stay visible and the
# provenance/value machinery cannot be swapped out by a fill; their
# arguments are masked independently.
INTERNAL_HELPER_FUNCS: frozenset[str] = frozenset(
  {
    "_crc32_update_i32",
    "_check_chksum_i32",
    "_in_check_chksum",
    GLOBAL_CHKSUM_FUNC,
    "_trap",
    "_f32",
    "_cast_int",
    "_Ptr",
    "_rd",
    "_vrd",
    "_idx",
    "_padd",
    "_pdiff",
    "_peq",
    "_prel",
    "_load",
    "_store",
    "_pidx",
    "_pfield",
    "_need_int",
    "_chk_geom",
    "_live",
    "_chk_ptr",
    "_chk_deref",
    "_derive",
  }
)

# Mask token per goto-flag kind. ``_go_`` keeps its kind (a general goto
# pairs with no keyword). ``_brk_``/``_cnt_`` pair 1:1 with ``break``/
# ``continue``, so their kind hides behind ``<FILL_CTRL>`` too. Both
# spellings (``_brk_``/``_break_``, ``_cnt_``/``_continue_``) map alike.
# Counting reuses existing vocabularies: one <FILL_CTRL> (2 kinds) times
# one <FILL_LABEL> (N targets).
GOTO_FLAG_MASK_TEMPLATE: dict[str, str] = {
  "_go_": "_go_<FILL_LABEL>",
  "_brk_": "_<FILL_CTRL>_<FILL_LABEL>",
  "_break_": "_<FILL_CTRL>_<FILL_LABEL>",
  "_cnt_": "_<FILL_CTRL>_<FILL_LABEL>",
  "_continue_": "_<FILL_CTRL>_<FILL_LABEL>",
}


def goto_flag_target(name: str) -> str | None:
  """Return the CFG target encoded in a goto-flag name, or None."""
  for prefix in GOTO_FLAG_MASK_TEMPLATE:
    if name.startswith(prefix):
      return name[len(prefix) :]
  return None


def goto_flag_mask(name: str) -> str | None:
  """Return the mask token for a goto-flag name, or None."""
  for prefix, template in GOTO_FLAG_MASK_TEMPLATE.items():
    if name.startswith(prefix):
      return template
  return None


# Operators masked as <FILL_OP>, by construct.  Within each tuple the order
# matters: longer symbols must precede their prefixes (e.g. b"//" before
# b"/") so the byte-slice search picks the intended span.
BINARY_OP_SPANS: tuple[bytes, ...] = (
  b"**",
  b"<<",
  b">>",
  b"//",
  b"+",
  b"-",
  b"*",
  b"/",
  b"%",
  b"&",
  b"|",
  b"^",
)
COMPARISON_OP_SPANS: tuple[bytes, ...] = (
  b"==",
  b"!=",
  b"<=",
  b">=",
  b"<",
  b">",
  b"is not",
  b"is",
  b"not in",
  b"in",
)
UNARY_OP_SPANS: tuple[bytes, ...] = (b"not", b"~", b"+", b"-")
IFEXP_KEYWORDS: tuple[bytes, ...] = (b"if", b"else")

# Control keywords masked as <FILL_CTRL>.
CONTROL_FLOW_KEYWORDS: tuple[str, ...] = ("break", "continue")

# ---------------------------------------------------------------------------
# Global checksum scaffold
# ---------------------------------------------------------------------------


def leftmost_base_name(node) -> str | None:
  """Name of the leftmost base of a target/operand expression.

  ``t0[4]`` resolves to ``t0``; ``vec1.lanes[0]`` to ``vec1``; a plain
  ``Name`` to itself; a non-name expression (call, constant) to None.
  """
  base = node
  while isinstance(base, (ast.Subscript, ast.Attribute)):
    base = base.value
  return base.id if isinstance(base, ast.Name) else None


def _is_global_chksum_call(node) -> bool:
  """True for an instrumented checksum call: either a bare
  ``_in_global_chksum(...)`` expression statement or an assignment whose
  value calls it (mid-path accumulates the running checksum back) --
  wrapped or not."""
  if not isinstance(node, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)):
    return False
  return any(
    isinstance(n, ast.Call)
    and isinstance(n.func, ast.Name)
    and n.func.id == GLOBAL_CHKSUM_FUNC
    for n in ast.walk(node.value if isinstance(node, ast.Expr) else node)
  )


# ---------------------------------------------------------------------------
# Prefix Stripping
# ---------------------------------------------------------------------------


def strip_refractir_prefix(src_bytes: bytes) -> bytes:
  """Strip the ``refractir_`` name prefix from all identifiers.

  rysmith prefixes every generated identifier with ``refractir_`` (or
  ``_refractir_`` for private helpers).  Stripping them makes the puzzle
  source more concise and readable.

  Ordering matters: ``_refractir_`` must be replaced before ``refractir_``
  so that ``_refractir_foo`` becomes ``_foo`` (not ``__foo``).
  """
  src = src_bytes.replace(b"_refractir_", b"_")
  src = src.replace(b"refractir_", b"")
  return src


# ---------------------------------------------------------------------------
# Emission preamble swap
#
# rysmith emits its leaf modules guard-free: every arithmetic operator is
# inlined as a plain Python expression, so none of the guarded emission's
# arithmetic helpers (``_ichk``, ``_iadd``, ``_sdiv``, ...) is referenced
# and none is emitted. The guard-free pointer/memory model drops the
# provenance checks with them: a pointer is plain list indexing, so an
# invalid pointer access silently reads or overwrites a flat leaf slot it
# cannot name. A masked puzzle is a module a third party fills in, so the
# swap replaces the emitted preamble with the guarded one before masking:
# the generated program is UB-free and never trips a guard, and a fill
# whose invalid pointer access would otherwise smuggle values through
# the leaf-slot list traps instead. The guarded preamble's ``_trap``
# raises, so a fill that executes unreachable code or trips an intrinsic
# UB precondition (an inert no-op in the replaced emission) traps too.
# ---------------------------------------------------------------------------

# Names of the emitted emission-preamble statements, by top-level kind.
# Imports carry their module names; classes/functions their names;
# sentinels ``_NULL``/``_UNDEF``/``_PAD`` their assignment targets.
GUARD_FREE_PREAMBLE_NAMES: frozenset[str] = frozenset(
  {
    "math",
    "struct",
    "RefractIRTrap",
    "_trap",
    "_cast_int",
    "_f32",
    "_need_int",
    "_chk_geom",
    "_Ptr",
    "_NULL",
    "_UNDEF",
    "_PAD",
    "_live",
    "_rd",
    "_idx",
    "_vrd",
    "_chk_ptr",
    "_chk_deref",
    "_derive",
    "_padd",
    "_pdiff",
    "_peq",
    "_prel",
    "_load",
    "_store",
    "_pidx",
    "_pfield",
  }
)

# The subset the leaf function and the intrinsic helpers reference. The
# emitted preamble is a fixed text with all of them present; one missing
# name means the emission shape drifted, which must fail loudly rather
# than replace only part of a preamble a call site still references.
REQUIRED_PREAMBLE_NAMES: frozenset[str] = GUARD_FREE_PREAMBLE_NAMES - {
  "_need_int",
  "_chk_geom",
  "_live",
  "_chk_ptr",
  "_chk_deref",
  "_derive",
}

# The guarded memory model lives in codoku_preamble.py, next to
# kPreamble in py_backend.cpp; that file owns the consistency contract.

_PREAMBLE_MODULE_NAME = "codoku_preamble.py"


def _guarded_preamble_bytes() -> bytes:
  """The splice text: the preamble module from its first import on.

  The docstring stays out of the splice, so puzzles carry the preamble
  code only.
  """
  path = Path(__file__).resolve().parent / _PREAMBLE_MODULE_NAME
  try:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith("import "))
  except (OSError, StopIteration) as e:
    raise RuntimeError(f"{_PREAMBLE_MODULE_NAME} is missing or malformed") from e
  # The file's final newline stays out of the splice (the ast span
  # excludes the last statement's line terminator), so re-swapping an
  # already-swapped module is byte-idempotent.
  return "".join(lines[start:]).rstrip("\n").encode("utf-8")


def _preamble_stmt_names(node):
  """Names a top-level statement contributes to the emission preamble."""
  if isinstance(node, ast.Import):
    return {alias.name for alias in node.names}
  if isinstance(node, ast.ImportFrom):
    return {alias.name for alias in node.names}
  if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
    return {node.name}
  if (
    isinstance(node, ast.Assign)
    and len(node.targets) == 1
    and isinstance(node.targets[0], ast.Name)
  ):
    return {node.targets[0].id}
  return None


def swap_preamble(src_bytes: bytes) -> bytes:
  """Replace the guard-free emission preamble with the guarded one.

  The swap is by top-level names, not by bytes: the emitted preamble's
  comment text has no contract, and the guarded preamble reuses the
  emitted one's names, so re-swapping an already-swapped module is a
  no-op. Definitions after the preamble (vec-lowering strategy classes,
  intrinsic helpers, the main wrapper and the leaf) are kept.
  """
  try:
    tree = ast.parse(src_bytes)
  except SyntaxError as e:
    raise RuntimeError(f"rysmith output is not valid Python: {e}") from e

  preamble: list[ast.stmt] = []
  for stmt in tree.body:
    names = _preamble_stmt_names(stmt)
    if names is None or not names <= GUARD_FREE_PREAMBLE_NAMES:
      break
    preamble.append(stmt)

  present: set[str] = set()
  for stmt in preamble:
    present |= _preamble_stmt_names(stmt)
  missing = REQUIRED_PREAMBLE_NAMES - present
  if missing:
    raise RuntimeError(
      "rysmith output preamble is unrecognizable (missing: "
      + ", ".join(sorted(missing))
      + ")"
    )

  start, end = get_byte_offsets(
    src_bytes,
    preamble[0].lineno,
    preamble[0].col_offset,
    preamble[-1].end_lineno,
    preamble[-1].end_col_offset,
  )
  return src_bytes[:start] + _guarded_preamble_bytes() + src_bytes[end:]


# ---------------------------------------------------------------------------
# Byte offsets and text replacement
# ---------------------------------------------------------------------------


def get_byte_offsets(
  src_bytes: bytes,
  lineno: int,
  col_offset: int,
  end_lineno: int,
  end_col_offset: int,
) -> tuple[int, int]:
  """Convert 1-indexed lineno and 0-indexed byte col_offset to absolute byte offsets."""
  lines = src_bytes.split(b"\n")
  start = 0
  for i in range(lineno - 1):
    start += len(lines[i]) + 1
  start += col_offset

  end = 0
  for i in range(end_lineno - 1):
    end += len(lines[i]) + 1
  end += end_col_offset
  return start, end


def apply_replacements(src_bytes: bytes, replacements: list) -> bytes:
  """Apply a list of ``(start, end, text)`` replacements to *src_bytes*.

  Replacements are applied in reverse byte-offset order so earlier offsets
  remain valid after each substitution.  Overlapping ranges are silently
  dropped (the outermost wins, consistent with the parent-first walk order).
  """
  sorted_repls = sorted(set(replacements), key=lambda x: (x[0], x[1]), reverse=True)

  last_start = len(src_bytes)
  valid_repls = []
  for start, end, repl in sorted_repls:
    if end <= last_start:
      valid_repls.append((start, end, repl))
      last_start = start

  res = bytearray(src_bytes)
  for start, end, repl in valid_repls:
    res[start:end] = repl.encode("utf-8")
  return bytes(res)


def canonical_cells(repls: list) -> list:
  """Order mask spans by offset and resolve overlaps like the puzzle render.

  ``apply_replacements`` drops overlapping ranges (outermost wins); the
  cell list follows the same rule, so the rendered puzzle and this list
  always agree on which spans a mask occupies.
  """
  sorted_repls = sorted(set(repls), key=lambda x: (x[0], x[1]), reverse=True)
  last_start = None
  valid = []
  for start, end, repl in sorted_repls:
    if last_start is None or end <= last_start:
      valid.append((start, end, repl))
      last_start = start
  valid.reverse()
  return valid


def collect_canonical_cells(
  maskable: list,
  entry_line: int,
  src_bytes: bytes,
  local_names: set[str],
  defined_funcs: set[str],
  disabled_masks: frozenset[str] = frozenset(),
) -> tuple[list, set, dict]:
  """Collect, filter, and canonicalize every mask cell of the maskable statements.

  Returns (cells, lhs_spans, const_values): *cells* lists every kept cell
  in source order as ``(start, end, token, stmt_index, is_lhs)``;
  *lhs_spans* tags ``<FILL_VAR>`` spans that sit in an assignment target
  (left-hand side); *const_values* maps every ``<FILL_CONST>`` span to its
  literal. Kinds in *disabled_masks* stay visible: their spans are dropped
  before this list is built, so they are never masked or budgeted.
  """
  cells: list = []
  lhs_spans: set = set()
  const_values: dict = {}
  for stmt_index, stmt in enumerate(maskable):
    repls: list = []
    collect_python_replacements(
      stmt,
      src_bytes,
      stmt.lineno > entry_line,
      repls,
      local_names,
      defined_funcs,
      lhs_spans=lhs_spans,
      const_values=const_values,
    )
    for start, end, token in canonical_cells(
      filter_disabled_masks(repls, disabled_masks)
    ):
      cells.append((start, end, token, stmt_index, (start, end) in lhs_spans))
  return cells, lhs_spans, const_values


def filter_disabled_masks(repls: list, disabled: frozenset[str]) -> list:
  """Drop replacements whose mask text is in *disabled*.

  Masking locates each blank as a ``(start, end, mask_text)`` span; a
  profile that disables kinds keeps those constructs visible by dropping
  their spans before the cell list is built, so they are never masked or
  budgeted. Exact-token filtering is why goto flags survive:
  ``_go_<FILL_LABEL>`` and ``_<FILL_CTRL>_<FILL_LABEL>`` are compound
  tokens, not known kinds, so disabling any kind leaves every flag target
  hidden.
  """
  if not disabled:
    return repls
  return [repl for repl in repls if repl[2] not in disabled]


# ---------------------------------------------------------------------------
# Python block comments
# ---------------------------------------------------------------------------


def find_python_block_comments(src: bytes) -> list[tuple[int, int, str, int]]:
  """Find all comments starting with '# ^' in the source.

  Returns a list of tuples: (start_byte, end_byte, label, line_number)
  """
  comments = []
  lines = src.split(b"\n")
  current_offset = 0
  for idx, line in enumerate(lines, 1):
    line_str = line.decode("utf-8", errors="ignore").strip()
    if line_str.startswith("# ^"):
      label = line_str[3:].strip()
      start_byte = current_offset + line.find(b"#")
      end_byte = current_offset + len(line)
      comments.append((start_byte, end_byte, label, idx))
    current_offset += len(line) + 1
  return comments


def block_label_for_line(comments, lineno: int) -> str | None:
  """Label of the last 'block comment' at or before *lineno*, or None.

  One block comment is emitted per block that carries instructions, so a
  statement's block is the last comment at or above its line. A labelless
  prefix (before the first block comment) yields None.
  """
  current = None
  for _, _, label, line in comments:
    if line <= lineno:
      current = label
    else:
      break
  return current


# ---------------------------------------------------------------------------
# Harness examples
#
# rysmith's --n-examples replays the leaf once per example inside the fixed
# @main wrapper: r = <leaf>(...) followed by r = _in_check_chksum(<expected>,
# r), one pair per example. The checker counts the examples from the puzzle
# text alone, and the creator's checksum recalibration patches every pair.
# ---------------------------------------------------------------------------


def count_harness_examples(text: str) -> int:
  """Count the @main harness examples a puzzle replays.

  One `_in_check_chksum(<expected>, r)` call per example; the helper's
  def line names no literal, so only the replay calls match.
  """
  return len(HARNESS_CHECK_RE.findall(text))


# The harness replay's check statement: the call's expected argument is a
# literal, and the local it checks is the leaf's return-value holder. The
# def line carries named parameters, so only call statements match.
HARNESS_CHECK_RE = re.compile(r"r\s*=\s*_in_check_chksum\(-?\d+\s*,\s*r\s*\)")


# Constant budget: each value maps to its (live, dead) slot split (when
# livedead_const_budget is True), or to its total count across the solution.
ConstBudget = dict[str, tuple[int, int]] | dict[str, int]


def merge_const_split(
  live_counts: dict[str, int], dead_counts: dict[str, int]
) -> ConstBudget:
  """Merge per-region counts into one budget split per value.

  The creator and the checker both build their splits this way so the
  two sides cannot drift: every value seen on either side appears once,
  missing on one side as zero.
  """
  return {
    val: (live_counts.get(val, 0), dead_counts.get(val, 0))
    for val in live_counts.keys() | dead_counts.keys()
  }


def stmt_is_on_live_path(comments, lineno: int, live_blocks: frozenset[str]) -> bool:
  """True when a statement at *lineno* counts toward the live budget.

  Statements before the first block comment are pre-entry declarations:
  they always execute, so they are live. Any other statement is live
  exactly when its block lies on the prescribed execution path.
  """
  label = block_label_for_line(comments, lineno)
  return label is None or label in live_blocks


def run_dumps_trace(py_path: str | Path, timeout: float) -> tuple[list[str], int]:
  """Run a Python module with DUMP_TRACE=1 and collect its block-entry trace.

  The run is capped at *timeout* seconds; a timeout raises RuntimeError.
  The trace labels match the block comments the leaf prints, and the
  caller interprets the exit code.
  """
  env = dict(os.environ)
  env["DUMP_TRACE"] = "1"
  try:
    r_run = subprocess.run(
      [sys.executable, str(py_path)],
      capture_output=True,
      text=True,
      timeout=timeout,
      env=env,
    )
  except subprocess.TimeoutExpired:
    raise RuntimeError(f"exceeded the {timeout:g}s execution cap")
  trace = [
    line[1:].rstrip(":") for line in r_run.stdout.splitlines() if line.startswith("^")
  ]
  return trace, r_run.returncode


def get_line_indent(src_bytes: bytes, start_byte: int) -> str:
  """Get the whitespace indentation of the line containing start_byte."""
  line_start = src_bytes.rfind(b"\n", 0, start_byte)
  if line_start == -1:
    line_start = 0
  else:
    line_start += 1
  indent = []
  for i in range(line_start, start_byte):
    char = src_bytes[i : i + 1]
    if char in (b" ", b"\t"):
      indent.append(char)
    else:
      break
  return b"".join(indent).decode("utf-8")


# ---------------------------------------------------------------------------
# Python leaf-function discovery
# ---------------------------------------------------------------------------


def find_python_leaf_function(
  tree, src: bytes
) -> tuple[ast.FunctionDef | None, str | None]:
  """Locate the non-helper, non-main leaf function in the Python AST module.

  Returns (node, name) or (None, None).
  """
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
      # Exclude main and all private helpers (starting with '_').
      if node.name != "main" and not node.name.startswith("_"):
        return node, node.name
  return None, None


def collect_python_leaf_locals(leaf_node: ast.FunctionDef) -> set[str]:
  """Collect all local variable and parameter names in the leaf function."""
  names = set()
  # Parameters
  for arg in leaf_node.args.args:
    names.add(arg.arg)
  # Local assignments
  for stmt in ast.walk(leaf_node):
    if isinstance(stmt, ast.Assign):
      targets = stmt.targets
    elif isinstance(stmt, ast.AugAssign):
      targets = [stmt.target]
    else:
      continue
    for target in targets:
      if isinstance(target, ast.Name):
        names.add(target.id)
      elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        names.add(target.value.id)
  # Exclude scratch variables starting with '_' or 'v__'
  return {
    name for name in names if not name.startswith("_") and not name.startswith("v__")
  }


# ---------------------------------------------------------------------------
# Python masking
# ---------------------------------------------------------------------------


def mentions_go_flag(node: ast.AST) -> bool:
  """True if the node references a goto flag of any known spelling."""
  return any(
    isinstance(n, ast.Name) and goto_flag_target(n.id) is not None
    for n in ast.walk(node)
  )


def get_python_maskable_statements(
  leaf_node: ast.FunctionDef, src: bytes
) -> tuple[list[ast.AST], int, int]:
  """Collect maskable statements and the entry/exit line numbers for the Python leaf.

  Returns (maskable_list, entry_line_number, exit_line_number).
  """
  comments = find_python_block_comments(src)
  comments = [c for c in comments if leaf_node.lineno <= c[3] <= leaf_node.end_lineno]
  comment_map = {}
  for _, _, label, idx in comments:
    if label not in comment_map:
      comment_map[label] = idx

  if "entry" not in comment_map or "exit" not in comment_map:
    return [], 0, 0

  entry_line = comment_map["entry"]
  exit_line = comment_map["exit"]

  decls_before_entry = []
  body_statements = []

  # Pre-entry assigns. Scratch names stay visible, except goto flags.
  # Inits nest inside the `try:` body since the `_frame` wrapper, so collect
  # at any depth in source order. Skip nested defs and DUMP_TRACE guards.

  def collect_decls(node):
    for child in ast.iter_child_nodes(node):
      if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        continue
      if _is_dump_trace_guard(child):
        continue
      if isinstance(child, ast.Assign) and child.lineno < entry_line:
        is_scratch = False
        for target in child.targets:
          if isinstance(target, ast.Name) and (
            target.id.startswith("_") or target.id.startswith("v__")
          ):
            is_scratch = True
        if not is_scratch or mentions_go_flag(child):
          decls_before_entry.append(child)
      collect_decls(child)

  collect_decls(leaf_node)
  decls_before_entry.sort(key=lambda s: (s.lineno, s.col_offset))

  def get_block_for_line(lineno):
    return block_label_for_line(comments, lineno)

  def get_loop_header(loop_node):
    for _, _, label, line in comments:
      if loop_node.lineno <= line <= loop_node.end_lineno:
        return label
    return "exit"

  def walk(node):
    if not hasattr(node, "lineno"):
      return
    # Skip trace blocks
    if _is_dump_trace_guard(node):
      return
    # Skip the global-checksum instrumentation: its operand arguments are
    # creator-shaped scaffold, not puzzle content.
    if _is_global_chksum_call(node):
      return
    # Flag setters/resets are maskable in any block: a revealed setter next
    # to masked guards would answer the blank.
    if isinstance(node, (ast.Assign, ast.Expr)) and mentions_go_flag(node):
      body_statements.append(node)
      return
    block = get_block_for_line(node.lineno)
    if isinstance(node, (ast.While, ast.For)):
      block = get_loop_header(node)

    # Exclude entry and exit blocks -- except goto-flag plumbing, which is
    # maskable wherever it appears.
    if block is not None and block != "entry" and block != "exit":
      if isinstance(
        node, (ast.Assign, ast.AugAssign, ast.Break, ast.Continue, ast.Expr)
      ):
        body_statements.append(node)
        return
      elif isinstance(node, (ast.If, ast.While)):
        # Note: ast.For is intentionally absent here - rysmith does not
        # generate Python for loops, so its maskable-statement traversal is
        # not modeled.
        body_statements.append(node.test)
        for child in node.body:
          walk(child)
        for child in getattr(node, "orelse", []):
          walk(child)
        return
    elif block is not None and isinstance(node, ast.If) and mentions_go_flag(node.test):
      # Goto plumbing outside body blocks: mask the test plus the keyword in
      # its body (resets are caught by the walk's top branch).
      body_statements.append(node.test)
      append_guard_keyword(node.body)
      append_guard_keyword(node.orelse)

    for child in ast.iter_child_nodes(node):
      walk(child)

  def append_guard_keyword(stmts):
    """Append break/continue inside a goto-guard body."""
    for stmt in stmts:
      if isinstance(stmt, (ast.Break, ast.Continue)):
        body_statements.append(stmt)
      elif isinstance(stmt, (ast.If, ast.While)):
        append_guard_keyword(stmt.body)
        append_guard_keyword(getattr(stmt, "orelse", []))

  for stmt in leaf_node.body:
    walk(stmt)

  # Guard and main walks can overlap: dedupe by identity, then sort by
  # source position. Creator flips, checker inference and remasking all
  # index this list, so its order must not depend on traversal shape.
  ordered = []
  seen = set()
  for stmt in decls_before_entry + body_statements:
    if id(stmt) not in seen:
      seen.add(id(stmt))
      ordered.append(stmt)
  ordered.sort(key=lambda s: (s.lineno, s.col_offset))
  return ordered, entry_line, exit_line


def collect_python_replacements(
  node: ast.AST,
  src_bytes: bytes,
  is_body: bool,
  replacements: list,
  local_names: set[str],
  defined_funcs: set[str],
  is_lhs: bool = False,
  lhs_spans: set[tuple[int, int]] | None = None,
  const_values: dict[tuple[int, int], str] | None = None,
) -> None:
  """Recursively walk node and append (start, end, <FILL_XXX>) replacement tuples.

  Masking rules (mirrors puzzle_common.hpp's SIRMaskedPrinter):
  - lvalues whose base is in *local_names* → ``<FILL_VAR>``
  - number literals → ``<FILL_CONST>``
  - break/continue → ``<FILL_CTRL>``
  - function calls to functions defined in the same file
    (*defined_funcs*) → ``<FILL_FUNC>`` (excluding internal helpers like
    _trap, _f32, _cast_int, and the pointer/memory model, whose calls
    stay visible while their arguments are masked independently)
  - binary operators in BinOp → ``<FILL_OP>``
  - comparison operators → ``<FILL_OP>``
  - unary operators → ``<FILL_OP>``
  - ternary if/else in IfExp → ``<FILL_OP>``

  The walk emits EVERY maskable cell; whether a cell is masked is not
  decided here.  ``is_lhs`` tracks assignment-target contexts (the left
  side of ``=`` and ``+=``), so a ``<FILL_VAR>`` cell inside one is
  recorded in *lhs_spans*, and ``<FILL_CONST>`` cells record their literal
  in *const_values* (span → value), which the budget reads at masked
  positions only.

  Trusted layout metadata is never masked: ``_Ptr`` geometry arguments
  (off/stride/lo/hi) come from the frontend's object layout and
  ``_cast_int`` widths come from the IR's integer types, so both stay
  visible and out of the constant budget. Only the ``_Ptr`` buffer (which
  object is pointed to) masks normally.

  ``is_body`` is True for statements strictly inside the function body (between
  entry/exit); it separates let-initialiser assigns (whose non-flag lvalues
  stay visible) from body assigns.  The sentinels ``0``, ``1``, ``0.0``, and
  ``1.0`` are left visible in every zone.

  Regardless of zone: every goto-flag identifier is replaced by its
  kind's mask token (see GOTO_FLAG_MASK_TEMPLATE). The target must be derived
  from the CFG/EXEC_PATH markers.
  """
  if not hasattr(node, "lineno"):
    return

  def get_node_offsets(n):
    return get_byte_offsets(
      src_bytes, n.lineno, n.col_offset, n.end_lineno, n.end_col_offset
    )

  if isinstance(node, ast.Name):
    flag_mask = goto_flag_mask(node.id)
    if flag_mask is not None:
      start, end = get_node_offsets(node)
      replacements.append((start, end, flag_mask))
      return

  # Subscripts mask whole (`t0[0]` is one VAR cell); attributes recurse so
  # only their base masks (`vec1` in `vec1.lanes`). This stops at Attribute
  # on purpose, unlike leftmost_base_name which resolves through both for
  # binding analysis.
  def get_py_subscript_base(n):
    if isinstance(n, ast.Subscript):
      return get_py_subscript_base(n.value)
    return n

  leftmost = get_py_subscript_base(node)

  if isinstance(node, ast.Assign):
    if not is_body:
      # let-initialisers: non-flag lvalues stay visible; the value masks.
      for target in node.targets:
        if mentions_go_flag(target):
          collect_python_replacements(
            target,
            src_bytes,
            is_body,
            replacements,
            local_names,
            defined_funcs,
            is_lhs,
            lhs_spans,
            const_values,
          )
      collect_python_replacements(
        node.value,
        src_bytes,
        is_body,
        replacements,
        local_names,
        defined_funcs,
        False,
        lhs_spans,
        const_values,
      )
      return
    # Body assignment: its target subtrees (left-hand side) and value mask.
    for target in node.targets:
      collect_python_replacements(
        target,
        src_bytes,
        is_body,
        replacements,
        local_names,
        defined_funcs,
        True,
        lhs_spans,
        const_values,
      )
    collect_python_replacements(
      node.value,
      src_bytes,
      is_body,
      replacements,
      local_names,
      defined_funcs,
      False,
      lhs_spans,
      const_values,
    )
    return
  if isinstance(node, ast.AugAssign):
    # Augmented assignment reads and writes its target (as x = x + v does),
    # so the target is left-hand side. The operator (+=) stays structural,
    # as does `=`: neither takes a mask.
    collect_python_replacements(
      node.target,
      src_bytes,
      is_body,
      replacements,
      local_names,
      defined_funcs,
      True,
      lhs_spans,
      const_values,
    )
    collect_python_replacements(
      node.value,
      src_bytes,
      is_body,
      replacements,
      local_names,
      defined_funcs,
      False,
      lhs_spans,
      const_values,
    )
    return
  if isinstance(leftmost, ast.Name) and leftmost.id in local_names:
    start, end = get_node_offsets(node)
    replacements.append((start, end, "<FILL_VAR>"))
    if is_lhs and lhs_spans is not None:
      lhs_spans.add((start, end))
    return

  if isinstance(node, ast.Constant):
    if isinstance(node.value, (int, float, bool)) or node.value is None:
      if isinstance(node.value, bool) or node.value is None:
        return
      # 0, 1, 0.0, and 1.0 are structural sentinels (counters, identity
      # elements): guessing them is noise, so they stay visible in every
      # zone and never enter a <FILL_CONST> position or the budget. A
      # fill that puts one into a mark fails the re-mask skeleton match.
      if node.value in (0, 1, 0.0, 1.0):
        return
      val_str = str(node.value)
      start, end = get_node_offsets(node)
      replacements.append((start, end, "<FILL_CONST>"))
      if const_values is not None:
        const_values[(start, end)] = val_str
      return

  if isinstance(node, (ast.Break, ast.Continue)):
    start, end = get_node_offsets(node)
    replacements.append((start, end, "<FILL_CTRL>"))
    return

  if isinstance(node, ast.Call):
    if isinstance(node.func, ast.Name):
      func_name = node.func.id
      if func_name in defined_funcs and func_name not in INTERNAL_HELPER_FUNCS:
        start, end = get_node_offsets(node.func)
        replacements.append((start, end, "<FILL_FUNC>"))
      # Trusted layout metadata stays visible (see the docstring): _Ptr
      # geometry (positions 1-4) and the _cast_int width (position 1).
      # Anything else in these calls masks normally: the _Ptr buffer
      # (arg 0, which object is pointed to) and the _cast_int value. The
      # frame arg is the scratch `_frame` cell and never masks.
      skip_args: frozenset[int] = frozenset()
      if func_name == "_Ptr":
        skip_args = frozenset({1, 2, 3, 4})
      elif func_name == "_cast_int":
        skip_args = frozenset({1})
      if skip_args:
        for idx, arg in enumerate(node.args):
          if idx not in skip_args:
            collect_python_replacements(
              arg,
              src_bytes,
              is_body,
              replacements,
              local_names,
              defined_funcs,
              False,
              lhs_spans,
              const_values,
            )
        for keyword in node.keywords:
          collect_python_replacements(
            keyword.value,
            src_bytes,
            is_body,
            replacements,
            local_names,
            defined_funcs,
            False,
            lhs_spans,
            const_values,
          )
        return
    for arg in node.args:
      collect_python_replacements(
        arg,
        src_bytes,
        is_body,
        replacements,
        local_names,
        defined_funcs,
        False,
        lhs_spans,
        const_values,
      )
    for keyword in node.keywords:
      collect_python_replacements(
        keyword.value,
        src_bytes,
        is_body,
        replacements,
        local_names,
        defined_funcs,
        False,
        lhs_spans,
        const_values,
      )
    return

  if isinstance(node, ast.BinOp):
    start_left, end_left = get_node_offsets(node.left)
    start_right, end_right = get_node_offsets(node.right)
    op_slice = src_bytes[end_left:start_right]
    for op_sym in BINARY_OP_SPANS:
      idx = op_slice.find(op_sym)
      if idx != -1:
        op_start = end_left + idx
        op_end = op_start + len(op_sym)
        replacements.append((op_start, op_end, "<FILL_OP>"))
        break
    collect_python_replacements(
      node.left,
      src_bytes,
      is_body,
      replacements,
      local_names,
      defined_funcs,
      False,
      lhs_spans,
      const_values,
    )
    collect_python_replacements(
      node.right,
      src_bytes,
      is_body,
      replacements,
      local_names,
      defined_funcs,
      False,
      lhs_spans,
      const_values,
    )
    return

  if isinstance(node, ast.Compare):
    prev_end = get_node_offsets(node.left)[1]
    for op, comp in zip(node.ops, node.comparators):
      comp_start, comp_end = get_node_offsets(comp)
      op_slice = src_bytes[prev_end:comp_start]
      for op_sym in COMPARISON_OP_SPANS:
        idx = op_slice.find(op_sym)
        if idx != -1:
          op_start = prev_end + idx
          op_end = op_start + len(op_sym)
          replacements.append((op_start, op_end, "<FILL_OP>"))
          break
      prev_end = comp_end
      collect_python_replacements(
        comp,
        src_bytes,
        is_body,
        replacements,
        local_names,
        defined_funcs,
        False,
        lhs_spans,
        const_values,
      )
    collect_python_replacements(
      node.left,
      src_bytes,
      is_body,
      replacements,
      local_names,
      defined_funcs,
      False,
      lhs_spans,
      const_values,
    )
    return

  if isinstance(node, ast.UnaryOp):
    op_start, op_end = get_node_offsets(node)
    operand_start, operand_end = get_node_offsets(node.operand)
    op_slice = src_bytes[op_start:operand_start]
    for op_sym in UNARY_OP_SPANS:
      idx = op_slice.find(op_sym)
      if idx != -1:
        start = op_start + idx
        end = start + len(op_sym)
        replacements.append((start, end, "<FILL_OP>"))
        break
    collect_python_replacements(
      node.operand,
      src_bytes,
      is_body,
      replacements,
      local_names,
      defined_funcs,
      False,
      lhs_spans,
      const_values,
    )
    return

  if isinstance(node, ast.IfExp):
    body_start, body_end = get_node_offsets(node.body)
    test_start, test_end = get_node_offsets(node.test)
    orelse_start, orelse_end = get_node_offsets(node.orelse)
    kw_if, kw_else = IFEXP_KEYWORDS
    if_slice = src_bytes[body_end:test_start]
    idx_if = if_slice.find(kw_if)
    if idx_if != -1:
      replacements.append(
        (body_end + idx_if, body_end + idx_if + len(kw_if), "<FILL_OP>")
      )
    else_slice = src_bytes[test_end:orelse_start]
    idx_else = else_slice.find(kw_else)
    if idx_else != -1:
      replacements.append(
        (test_end + idx_else, test_end + idx_else + len(kw_else), "<FILL_OP>")
      )
    collect_python_replacements(
      node.body,
      src_bytes,
      is_body,
      replacements,
      local_names,
      defined_funcs,
      False,
      lhs_spans,
      const_values,
    )
    collect_python_replacements(
      node.test,
      src_bytes,
      is_body,
      replacements,
      local_names,
      defined_funcs,
      False,
      lhs_spans,
      const_values,
    )
    collect_python_replacements(
      node.orelse,
      src_bytes,
      is_body,
      replacements,
      local_names,
      defined_funcs,
      False,
      lhs_spans,
      const_values,
    )
    return

  for child in ast.iter_child_nodes(node):
    collect_python_replacements(
      child,
      src_bytes,
      is_body,
      replacements,
      local_names,
      defined_funcs,
      is_lhs,
      lhs_spans,
      const_values,
    )


# ---------------------------------------------------------------------------
# Python CFG extraction
# ---------------------------------------------------------------------------


def loop_header_label(comments, loop_node) -> str:
  """First block comment in a loop range: what `continue` targets."""
  for _, _, label, line in comments:
    if loop_node.lineno <= line <= loop_node.end_lineno:
      return label
  return "exit"


def exit_label_after(comments, node) -> str:
  """First block comment past a node: what `break`/`return` targets."""
  for _, _, label, line in comments:
    if line > node.end_lineno:
      return label
  return "exit"


def _is_dump_trace_guard(node) -> bool:
  """True for a ``DUMP_TRACE`` instrumentation guard (never program code)."""
  return isinstance(node, ast.If) and any(
    isinstance(n, ast.Constant) and n.value == "DUMP_TRACE" for n in ast.walk(node.test)
  )


def _is_flag_set(node) -> bool:
  """True for a ``flag = True`` statement (a SetFlag emission site).

  ``flag = False`` is a reset (it unmasks a flag for later dispatches) and
  is deliberately not a set: resets are fall-through plumbing.
  """
  if not (
    isinstance(node, ast.Assign)
    and len(node.targets) == 1
    and isinstance(node.targets[0], ast.Name)
  ):
    return False
  name = node.targets[0].id
  return (
    goto_flag_target(name) is not None
    and isinstance(node.value, ast.Constant)
    and bool(node.value.value)
  )


def _is_flag_dispatch_test(test) -> bool:
  """True for a dispatch guard's test: a positive test of one or more flags.

  The Python backend emits in-flight transfer dispatches as ``if flag:`` /
  ``if flag and flag2 ...:``; a negated test (``if not flags:``) is a
  Guarded body wrapping the normal path instead.
  """
  if isinstance(test, ast.Name):
    return goto_flag_target(test.id) is not None
  if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
    return all(
      isinstance(v, ast.Name) and goto_flag_target(v.id) is not None
      for v in test.values
    )
  return False


def _is_guarded_test(test) -> bool:
  """True for a Guarded body's test: ``not flag`` / ``not f1 and not f2``.

  A Guarded body wraps a normal-path item against in-flight join flags;
  the flag-in-flight path is unreachable here (it bypassed the guard at
  each set site), so the walk treats the body as fall-through-only.
  """
  if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
    operand = test.operand
    return isinstance(operand, ast.Name) and goto_flag_target(operand.id) is not None
  if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
    return all(
      isinstance(v, ast.UnaryOp)
      and isinstance(v.op, ast.Not)
      and isinstance(v.operand, ast.Name)
      and goto_flag_target(v.operand.id) is not None
      for v in test.values
    )
  return False


def build_python_cfg(leaf_node: ast.FunctionDef, src: bytes) -> set[tuple[str, str]]:
  """Extract the CFG edges from the Python leaf function's AST.

  The Python backend lowers multi-level transfers into flag state and
  peepholes the loop structure, so a purely structural walk misreports
  the SIR CFG in three ways: the ``break`` paired with a ``flag = True``
  site is the mechanical inner-loop unwind (its structural target is not
  the SIR edge), dispatch guards re-enter a chain that bypasses the
  structural targets, and self-edge filtering drops a legitimate
  self-loop. The walk is flag-aware:

  - a prepass turns each ``flag = True`` site into its true edge
    (source block -> the flag's encoded target) and marks the breaks in
    the set's straight-line run as mechanical (suppressed);
  - ``if flag:`` dispatch guards are fall-through-only: their taken
    branch re-enters a chain already recorded at the set site;
  - ``if not flags:`` guarded bodies wrap the normal path only;
  - ``DUMP_TRACE`` instrumentation guards are scaffolding, never edges;
  - loop bodies whose tail pendings re-anchor at a header comment carry
    the implicit backedge, self-loops included;
  - a ``while <cond>:`` loop's condition-false side exits to the loop's
    successor: the post-loop pendings anchor at the exit edge's source.
  The backend's ``try:`` is the ``_frame`` provenance wrapper: its body
  is the leaf's code; ``finally:``, ``orelse``, and ``handlers`` are
  mechanical or off-path plumbing without SIR edges.
  """
  comments = find_python_block_comments(src)
  comments = [c for c in comments if leaf_node.lineno <= c[3] <= leaf_node.end_lineno]

  def get_loop_header(loop_node):
    return loop_header_label(comments, loop_node)

  def get_exit_label_after(node):
    return exit_label_after(comments, node)

  edges = set()
  suppressed = set()
  processed_comments = set()

  # --- Prepass: SetFlag sites and the mechanical breaks they pair with. ---
  # Each statement list anchors its own comment region: the first arm of
  # an `if`/`while` runs from the keyword's line (a comment right after
  # belongs to that arm), a second arm from just above its first
  # statement. A set site's source block is the arm's last comment before
  # it, falling back to the block that branches into the arm.

  def scan(stmts, floor, node):
    """Collect set-site edges into *edges* and mark mechanical breaks."""
    arm_set = False
    for stmt in stmts:
      if _is_dump_trace_guard(stmt):
        continue
      if _is_flag_set(stmt):
        arm_comments = [
          lbl for _, _, lbl, line in comments if floor <= line <= stmt.lineno
        ]
        source = (
          arm_comments[-1]
          if arm_comments
          else block_label_for_line(comments, node.lineno)
        )
        if source is not None:
          edges.add((source, goto_flag_target(stmt.targets[0].id)))
        arm_set = True
        continue
      if isinstance(stmt, ast.Break):
        # A `break` inside a set's straight-line run is the mechanical
        # dispatch-chain unwind; the real transfer is the flag's edge.
        # The run marker survives plain statements, so an unwind stays
        # mechanical even if the backend reorders around it.
        if arm_set:
          suppressed.add(stmt.lineno)
        arm_set = False
        continue
      if isinstance(stmt, ast.If) and _is_flag_dispatch_test(stmt.test):
        arm_set = False
        continue
      if isinstance(stmt, ast.If):
        arm_set = False
        scan(stmt.body, stmt.lineno, stmt)
        if stmt.orelse:
          scan(stmt.orelse, stmt.orelse[0].lineno - 1, stmt)
      elif isinstance(stmt, ast.While):
        arm_set = False
        scan(stmt.body, stmt.lineno, stmt)
        if stmt.orelse:
          scan(stmt.orelse, stmt.orelse[0].lineno - 1, stmt)
      elif isinstance(stmt, ast.Try):
        arm_set = False
        scan(stmt.body, stmt.lineno + 1, stmt)
      elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        continue

  scan(leaf_node.body, leaf_node.lineno, leaf_node)

  # --- Structural walk (flag-aware). ---

  def walk(node, pending, loop_stack):
    if not hasattr(node, "lineno"):
      return pending
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      return pending

    # Process any comments that occur before or at this node's line
    for _, _, label, line in comments:
      if line not in processed_comments and line <= node.lineno:
        processed_comments.add(line)
        for p in pending:
          if p != label:
            edges.add((p, label))
        pending = [label]

    if _is_dump_trace_guard(node):
      return pending
    if _is_flag_set(node):
      # The arm's flow terminates at the flag's target (the set site is
      # always the last flow item of its arm, next to the suppressed
      # unwind or nothing at all for a jump-join).
      return []
    if isinstance(node, ast.Break):
      if node.lineno in suppressed:
        return []
      if loop_stack:
        innermost_loop = loop_stack[-1]
        loop_exit = get_exit_label_after(innermost_loop)
        for p in pending:
          edges.add((p, loop_exit))
      return []
    elif isinstance(node, ast.Continue):
      if loop_stack:
        innermost_loop = loop_stack[-1]
        loop_header = get_loop_header(innermost_loop)
        for p in pending:
          edges.add((p, loop_header))
      return []
    elif isinstance(node, ast.Return):
      # A `ret` terminates its own block, never jumps into it: when the
      # return sits in the exit block's code, its pendings re-anchor at
      # the `# ^exit` comment and adding (exit, exit) is spurious.
      for p in pending:
        if p != "exit":
          edges.add((p, "exit"))
      return []
    elif isinstance(node, ast.If):
      if _is_flag_dispatch_test(node.test):
        # Dispatch plumbing: the taken branch re-enters a chain already
        # recorded at its set site.
        return pending
      if _is_guarded_test(node.test):
        # Guarded bodies wrap the normal path only: the flag-in-flight
        # path is unreachable here, so the walk is fall-through-only.
        pending = walk(node.test, pending, loop_stack)
        for child in node.body:
          pending = walk(child, pending, loop_stack)
        return pending
      pending = walk(node.test, pending, loop_stack)

      then_pending = pending
      for child in node.body:
        then_pending = walk(child, then_pending, loop_stack)

      else_pending = pending
      for child in node.orelse:
        else_pending = walk(child, else_pending, loop_stack)

      return then_pending + else_pending
    elif isinstance(node, ast.While):
      loop_stack.append(node)
      loop_header = get_loop_header(node)
      loop_exit = get_exit_label_after(node)

      for p in pending:
        if p != loop_header:
          edges.add((p, loop_header))

      body_pending = [loop_header]
      for child in node.body:
        body_pending = walk(child, body_pending, loop_stack)

      # Body-tail pendings repeat the loop: an implicit backedge, and a
      # legitimate self-loop when the body's first block is the loop head.
      for bp in body_pending:
        edges.add((bp, loop_header))

      loop_stack.pop()

      # A constant-truthy test (rysmith emits `while True:` for an
      # unconditional Loop node) never exits via its condition; every
      # escape is an explicit transfer recorded at its own site.
      is_infinite = isinstance(node.test, ast.Constant) and bool(node.test.value)
      if not is_infinite:
        # `while <cond>:`: the loop's condition-false side is a real
        # SIR edge from the loop header to the successor block. When
        # the header was rotated (its body tail duplicates the
        # pre-loop code) the pre-loop block is that header; the
        # pendings after the loop anchor there.
        header_label = block_label_for_line(comments, node.lineno)
        if header_label is not None:
          edges.add((header_label, loop_exit))
          return [header_label]
      return []

    elif isinstance(node, ast.Try):
      # The backend's `try:` is the `_frame` provenance wrapper: its body
      # is the leaf's code; `finally:`, `orelse`, and `handlers` are
      # mechanical or off-path plumbing without SIR edges.
      for child in node.body:
        pending = walk(child, pending, loop_stack)
      return pending

    cb = pending
    for child in ast.iter_child_nodes(node):
      cb = walk(child, cb, loop_stack)
    return cb

  cb = ["entry"]
  for stmt in leaf_node.body:
    cb = walk(stmt, cb, [])

  return edges


def iter_flag_dispatches(leaf_node: ast.FunctionDef, src: bytes):
  """Yield (flag, encoded, structural, lineno) per dispatch guard transfer.

  A dispatch guard tests one goto flag (`if F:` / `if not F:`) and moves
  control with break/continue/return in the taken branch. The structural
  destination uses the same loop model as build_python_cfg. All spellings
  (`_go_`, `_brk_` / `_cnt_` and their full-word forms) are yielded; the
  checker applies transfer equality where it is meaningful - the one-hop
  `_brk_` / `_cnt_` casts - while `_go_` flags unwind through multi-hop
  chains (a guard's break lands mid-chain), so their equality does not
  hold and their target is validated by topology instead. DUMP_TRACE
  instrumentation guards are scaffolding and yield nothing. Guards
  without a loop transfer yield nothing.
  """
  comments = find_python_block_comments(src)
  comments = [c for c in comments if leaf_node.lineno <= c[3] <= leaf_node.end_lineno]
  found = []

  def find_keyword(node):
    if isinstance(node, (ast.Break, ast.Continue, ast.Return)):
      return node
    if isinstance(
      node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.While)
    ):
      return None
    for child in ast.iter_child_nodes(node):
      hit = find_keyword(child)
      if hit is not None:
        return hit
    return None

  def walk(node, loop_stack):
    if _is_dump_trace_guard(node):
      return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      if node is not leaf_node:
        return
    if isinstance(node, ast.If):
      test, branch = node.test, node.body
      while isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        test, branch = test.operand, node.orelse if branch is node.body else node.body
      if isinstance(test, ast.Name) and goto_flag_target(test.id) is not None:
        for stmt in branch:
          keyword = find_keyword(stmt)
          if keyword is None:
            continue
          if isinstance(keyword, ast.Break) and loop_stack:
            dest = exit_label_after(comments, loop_stack[-1])
          elif isinstance(keyword, ast.Continue) and loop_stack:
            dest = loop_header_label(comments, loop_stack[-1])
          elif isinstance(keyword, ast.Return):
            dest = "exit"
          else:
            continue
          found.append((test.id, goto_flag_target(test.id), dest, node.lineno))
          break
    if isinstance(node, ast.While):
      loop_stack.append(node)
      for child in ast.iter_child_nodes(node):
        walk(child, loop_stack)
      loop_stack.pop()
    else:
      for child in ast.iter_child_nodes(node):
        walk(child, loop_stack)

  walk(leaf_node, [])
  return found

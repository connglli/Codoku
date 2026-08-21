#!/usr/bin/env python3
"""codoku_checker.py - validates a solution against a Python codoku puzzle.

Vendored from puzzle/target/rypuzchk.py, reduced to the Python target and
adapted to the angle-bracketed <FILL_XXX> mask tokens.  Validation is exposed
programmatically via check() (raises CheckFailure on failure) and via the CLI.

Checks performed in strict order from easiest to hardest to reason about:

  Stage 1 - FAIL_BASICS     : Basics (missing markers, unfilled <FILL_XXX> marks).
  Stage 2 - FAIL_PARSE      : Parse (solution fails to parse).
  Stage 3 - FAIL_REMASKING  : Re-masked skeleton does not match the puzzle.
  Stage 4 - FAIL_COMPILE    : Compile (solution fails to compile).
  Stage 5 - FAIL_CFG        : CFG topology matches the declared //@ CFG_EDGE: markers exactly.
  Stage 6 - FAIL_TIMEOUT    : The solution run exceeded the 5s execution cap.
  Stage 7 - FAIL_PATH       : Execution did not follow the prescribed path exactly.
  Stage 8 - FAIL_OUTPUT     : check_chksum reports a wrong result (non-zero exit).
  Stage 9 - FAIL_FILL_CONST : Constant budget multiset mismatch.

Each stage is a strict prerequisite for the next.  When a stage fails, later
stages are skipped, making it unambiguous *why* a solution is wrong.
"""

import argparse
import ast
import os
import subprocess
import sys
import tempfile
from enum import Enum

from codoku_common import (
  apply_replacements,
  build_python_cfg,
  collect_python_leaf_locals,
  collect_python_replacements,
  find_python_leaf_function,
  get_byte_offsets,
  get_python_maskable_statements,
  strip_refractir_prefix,
)

# ---------------------------------------------------------------------------
# Check result - ordered from easiest to hardest to satisfy.
# ---------------------------------------------------------------------------


class CheckResult(str, Enum):
  PASS = "PASS"
  FAIL_BASICS = "FAIL_BASICS"
  FAIL_PARSE = "FAIL_PARSE"
  FAIL_REMASKING = "FAIL_REMASKING"
  FAIL_COMPILE = "FAIL_COMPILE"
  FAIL_CFG = "FAIL_CFG"
  FAIL_PATH = "FAIL_PATH"
  FAIL_OUTPUT = "FAIL_OUTPUT"
  FAIL_FILL_CONST = "FAIL_FILL_CONST"
  FAIL_TIMEOUT = "FAIL_TIMEOUT"


class CheckFailure(Exception):
  """Raised when a validation stage fails; carries the failing stage."""

  def __init__(self, result: CheckResult, message: str) -> None:
    super().__init__(message)
    self.result = result
    self.message = message


def fail(result: CheckResult, msg: str) -> None:
  """Raise a tagged failure carrying the failing stage and message."""
  raise CheckFailure(result, msg)


# ---------------------------------------------------------------------------
# Comment / whitespace stripping for structural comparison
# ---------------------------------------------------------------------------


def strip_comments_and_whitespace(text: str) -> str:
  """Remove Python comments and all whitespace outside strings.

  The stripped form is used for the re-masking byte-for-byte comparison so
  that formatting differences between the puzzle and the re-masked solution
  are ignored.
  """
  res: list[str] = []
  in_line_comment = False
  in_string = False
  string_delim = None  # single char ('"' or "'") or triple ('"""' or "'''")
  i = 0
  while i < len(text):
    if in_line_comment:
      if text[i] == "\n":
        in_line_comment = False
      i += 1
    elif in_string:
      res.append(text[i])
      if text[i] == "\\" and i + 1 < len(text):
        res.append(text[i + 1])
        i += 2
      elif len(string_delim) == 3 and text[i : i + 3] == string_delim:
        # Close triple-quoted string: first char already appended above.
        res.append(text[i + 1])
        res.append(text[i + 2])
        in_string = False
        i += 3
      elif len(string_delim) == 1 and text[i] == string_delim:
        in_string = False
        i += 1
      else:
        i += 1
    else:
      if text[i] == "#":
        in_line_comment = True
        i += 1
      elif text[i] in ('"', "'"):
        # Detect triple-quoted strings (""" or ''').
        if text[i : i + 3] in ('"""', "'''"):
          in_string = True
          string_delim = text[i : i + 3]
          res.append(text[i])
          res.append(text[i + 1])
          res.append(text[i + 2])
          i += 3
        else:
          in_string = True
          string_delim = text[i]
          res.append(text[i])
          i += 1
      elif text[i].isspace():
        i += 1
      else:
        res.append(text[i])
        i += 1
  return "".join(res)


# ---------------------------------------------------------------------------
# Puzzle banner parsing
# ---------------------------------------------------------------------------


def parse_puzzle_requirements(
  puzzle_text: str,
) -> tuple[list[str], dict[str, int], list[tuple[str, str]]]:
  """Parse markers from the puzzle banner."""
  expected_path: list[str] = []
  const_counts: dict[str, int] = {}
  cfg_edges: list[tuple[str, str]] = []

  for line in puzzle_text.splitlines():
    if "//@ EXEC_PATH:" in line:
      path_part = line.split("//@ EXEC_PATH:", 1)[1].strip()
      expected_path = [x.strip() for x in path_part.split("->") if x.strip()]
    elif "//@ <FILL_CONST>:" in line:
      parts = line.split("//@ <FILL_CONST>:", 1)[1].strip().split()
      if len(parts) != 2:
        fail(
          CheckResult.FAIL_PARSE,
          f"Malformed //@ <FILL_CONST> marker: expected 2 tokens, got {len(parts)} in '{line}'",
        )
      val = parts[0]
      try:
        cnt = int(parts[1])
      except ValueError:
        fail(
          CheckResult.FAIL_PARSE,
          f"Malformed //@ <FILL_CONST> marker: count '{parts[1]}' is not an integer in '{line}'",
        )
      const_counts[val] = cnt
    elif "//@ CFG_EDGE:" in line:
      edge_part = line.split("//@ CFG_EDGE:", 1)[1].strip()
      if "->" not in edge_part:
        fail(
          CheckResult.FAIL_PARSE,
          f"Malformed //@ CFG_EDGE marker: missing '->' in '{line}'",
        )
      parts = edge_part.split("->", 1)
      from_node, to_node = parts[0].strip(), parts[1].strip()
      if not from_node or not to_node:
        fail(
          CheckResult.FAIL_PARSE,
          f"Malformed //@ CFG_EDGE marker: empty node name in '{line}'",
        )
      cfg_edges.append((from_node, to_node))

  return expected_path, const_counts, cfg_edges


# ---------------------------------------------------------------------------
# Mask-set inference (mirrors inferMaskSetFromPuzzle in puzzle_common.hpp)
# ---------------------------------------------------------------------------


def infer_mask_set_from_puzzle(
  sol_leaf,
  sol_src: bytes,
  puzzle_text: str,
  defined_funcs: set[str],
) -> set[int] | None:
  """Infer which statement indices were masked in the puzzle by comparing renders.

  Returns the set of masked position indices, or ``None`` if the structure of
  the puzzle and the solution are incompatible (re-masking failure).
  """
  maskable, entry_line, exit_line = get_python_maskable_statements(sol_leaf, sol_src)
  if not entry_line or not exit_line:
    return None
  local_names = collect_python_leaf_locals(sol_leaf)

  full_repls: list = []
  plain_repls: list = []
  dummy_budget: dict = {}

  for stmt in maskable:
    stmt_offsets = get_byte_offsets(
      sol_src, stmt.lineno, stmt.col_offset, stmt.end_lineno, stmt.end_col_offset
    )
    sentinel = (stmt_offsets[0], stmt_offsets[0], "\x01")
    full_repls.append(sentinel)
    plain_repls.append(sentinel)
    is_body = stmt.lineno > entry_line
    collect_python_replacements(
      stmt,
      sol_src,
      is_body,
      full_repls,
      dummy_budget,
      local_names,
      defined_funcs,
    )

  full_rendered = apply_replacements(sol_src, full_repls).decode("utf-8")
  plain_rendered = apply_replacements(sol_src, plain_repls).decode("utf-8")

  full_stripped = strip_comments_and_whitespace(full_rendered)
  plain_stripped = strip_comments_and_whitespace(plain_rendered)

  full_parts = full_stripped.split("\x01")
  plain_parts = plain_stripped.split("\x01")

  stripped_puzzle = strip_comments_and_whitespace(puzzle_text)

  n_positions = len(plain_parts) - 1
  mask_set: set[int] = set()
  pos = 0

  if not plain_parts:
    return None
  prefix = plain_parts[0]
  if not stripped_puzzle.startswith(prefix):
    return None
  pos += len(prefix)

  for i in range(n_positions):
    masked_seg = full_parts[i + 1] if i + 1 < len(full_parts) else ""
    plain_seg = plain_parts[i + 1] if i + 1 < len(plain_parts) else ""

    if (
      pos + len(masked_seg) <= len(stripped_puzzle)
      and stripped_puzzle[pos : pos + len(masked_seg)] == masked_seg
    ):
      mask_set.add(i)
      pos += len(masked_seg)
    elif (
      pos + len(plain_seg) <= len(stripped_puzzle)
      and stripped_puzzle[pos : pos + len(plain_seg)] == plain_seg
    ):
      pos += len(plain_seg)
    else:
      return None

  return mask_set


def check_cfg(func_node, src: bytes, cfg_edges: list[tuple[str, str]]) -> None:
  """Verify that the solution's CFG matches the declared edges exactly."""
  if not cfg_edges:
    return

  actual_edges = build_python_cfg(func_node, src)
  declared_edges = set(cfg_edges)
  if declared_edges != actual_edges:
    unexpected = actual_edges - declared_edges
    missing = declared_edges - actual_edges
    msg_parts = ["CFG topology mismatch."]
    for f, t in sorted(unexpected):
      msg_parts.append(f"  unexpected edge: {f} -> {t}")
    for f, t in sorted(missing):
      msg_parts.append(f"  missing edge:    {f} -> {t}")
    fail(CheckResult.FAIL_CFG, "\n".join(msg_parts))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_python_solution(py_path: str) -> tuple[list[str], int]:
  """Run the Python solution and collect its trace and exit code.

  The run is capped at 5s; a timeout fails the PATH/OUTPUT stages with
  FAIL_TIMEOUT.
  """
  env = dict(os.environ)
  env["DUMP_TRACE"] = "1"
  try:
    r_run = subprocess.run(
      [sys.executable, py_path], capture_output=True, text=True, timeout=5, env=env
    )
  except subprocess.TimeoutExpired:
    fail(
      CheckResult.FAIL_TIMEOUT,
      "Solution timed out after 5s while running.",
    )

  trace = []
  for line in r_run.stdout.splitlines():
    if line.startswith("^"):
      trace.append(line[1:].rstrip(":"))
  return trace, r_run.returncode


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_path(trace: list[str], expected_path: list[str]) -> None:
  """Verify the execution trace matches the expected path exactly."""
  if trace != expected_path:
    exp_str = " -> ".join(expected_path)
    act_str = " -> ".join(trace)
    fail(
      CheckResult.FAIL_PATH,
      f"Execution path mismatch.\n  Expected: {exp_str}\n  Actual:   {act_str}",
    )


def check_output(exit_code: int) -> None:
  """Verify exit code is 0 (check_chksum passed); exit FAIL_OUTPUT otherwise."""
  if exit_code != 0:
    fail(
      CheckResult.FAIL_OUTPUT,
      f"Solution output is incorrect (check_chksum mismatch; exit code {exit_code}).",
    )


def check_remasking(
  sol_leaf,
  sol_src: bytes,
  puzzle_text: str,
  mask_set: set[int],
  defined_funcs: set[str],
) -> dict[str, int]:
  """Re-mask the solution at *mask_set* and verify it matches the puzzle skeleton."""
  maskable, entry_line, exit_line = get_python_maskable_statements(sol_leaf, sol_src)
  local_names = collect_python_leaf_locals(sol_leaf)

  remasked_repls: list = []
  actual_counts: dict[str, int] = {}

  for idx, stmt in enumerate(maskable):
    if idx in mask_set:
      is_body = stmt.lineno > entry_line
      collect_python_replacements(
        stmt,
        sol_src,
        is_body,
        remasked_repls,
        actual_counts,
        local_names,
        defined_funcs,
      )

  remasked_text = apply_replacements(sol_src, remasked_repls).decode("utf-8")
  if strip_comments_and_whitespace(remasked_text) != strip_comments_and_whitespace(
    puzzle_text
  ):
    fail(
      CheckResult.FAIL_REMASKING,
      "Solution structural integrity check failed.\n"
      "  You may have changed code outside the <FILL_XXX> marks, or introduced\n"
      "  unauthorized variables / statements / basic blocks.",
    )

  return actual_counts


def check_fill_const_budget(
  actual_counts: dict[str, int], expected_counts: dict[str, int]
) -> None:
  """Verify the FILL_CONST multiset matches the puzzle budget exactly."""
  if not expected_counts:
    return

  for val, expected_cnt in expected_counts.items():
    actual_cnt = actual_counts.get(val, 0)
    if actual_cnt != expected_cnt:
      fail(
        CheckResult.FAIL_FILL_CONST,
        f"<FILL_CONST> count mismatch for '{val}'. "
        f"Expected {expected_cnt}, got {actual_cnt}.",
      )

  for val, actual_cnt in actual_counts.items():
    if val not in expected_counts:
      fail(
        CheckResult.FAIL_FILL_CONST,
        f"Off-budget constant in a <FILL_CONST> position: '{val}' (count: {actual_cnt}).",
      )


# ---------------------------------------------------------------------------
# Full validation pipeline
# ---------------------------------------------------------------------------


def check(puzzle: str, solution: str) -> None:
  """Validate *solution* against *puzzle*.

  Runs the eight stages in strict order.  Raises CheckFailure on the first
  failing stage; returns normally on success.
  """

  # --- Existence checks (pre-parse guard) ---
  for label, path in [("Puzzle", puzzle), ("Solution", solution)]:
    if not os.path.exists(path):
      fail(CheckResult.FAIL_BASICS, f"{label} file '{path}' does not exist.")

  # -------------------------------------------------------------------------
  # Stage 1 - FAIL_BASICS: parse puzzle requirements.
  # -------------------------------------------------------------------------
  try:
    with open(puzzle, "r") as f:
      puzzle_text = f.read()
  except (OSError, UnicodeError) as e:
    fail(CheckResult.FAIL_BASICS, f"Puzzle file '{puzzle}' cannot be read: {e}")

  expected_path, const_counts, cfg_edges = parse_puzzle_requirements(puzzle_text)
  if not expected_path:
    fail(
      CheckResult.FAIL_BASICS,
      "Puzzle is missing a '//@ EXEC_PATH:' marker; cannot validate.",
    )
  if not cfg_edges:
    fail(
      CheckResult.FAIL_BASICS,
      "Puzzle is missing '//@ CFG_EDGE:' markers; cannot validate.",
    )

  # -------------------------------------------------------------------------
  # Stage 1 - FAIL_BASICS: Check for unfilled <FILL_XXX> marks.
  # -------------------------------------------------------------------------
  try:
    with open(solution, "rb") as f:
      sol_src_raw = f.read()
  except OSError as e:
    fail(CheckResult.FAIL_BASICS, f"Solution file '{solution}' cannot be read: {e}")

  sol_src = strip_refractir_prefix(sol_src_raw)
  try:
    sol_src_str = sol_src.decode("utf-8")
  except UnicodeDecodeError as e:
    fail(CheckResult.FAIL_BASICS, f"Solution file '{solution}' is not valid UTF-8: {e}")

  stripped_sol = strip_comments_and_whitespace(sol_src_str)
  has_fill_marks = False
  for mark in [
    "<FILL_VAR>",
    "<FILL_CONST>",
    "<FILL_OP>",
    "<FILL_TYPE>",
    "<FILL_LABEL>",
    "<FILL_FUNC>",
    "<FILL_FIELD>",
    "<FILL_CTRL>",
  ]:
    if mark in stripped_sol:
      has_fill_marks = True
      break
  if has_fill_marks:
    fail(CheckResult.FAIL_BASICS, "Solution still contains unfilled <FILL_XXX> marks.")

  # -------------------------------------------------------------------------
  # Stage 2 - FAIL_PARSE: Parse the solution source.
  # -------------------------------------------------------------------------
  try:
    sol_tree = ast.parse(sol_src)
  except Exception as e:
    fail(
      CheckResult.FAIL_PARSE,
      f"Could not parse solution Python code (syntax error: {e}).",
    )
  sol_leaf, _leaf_name = find_python_leaf_function(sol_tree, sol_src)
  if not sol_leaf:
    fail(CheckResult.FAIL_PARSE, "Could not find leaf function in solution.")

  defined_funcs = set()
  for node in ast.walk(sol_tree):
    if isinstance(node, ast.FunctionDef):
      defined_funcs.add(node.name)

  # -------------------------------------------------------------------------
  # Stage 3 - FAIL_REMASKING: Re-mask and compare.
  # Pure static check - no execution required.
  # -------------------------------------------------------------------------
  mask_set = infer_mask_set_from_puzzle(
    sol_leaf,
    sol_src,
    puzzle_text,
    defined_funcs,
  )
  if mask_set is None:
    fail(
      CheckResult.FAIL_REMASKING,
      "Solution structural integrity check failed.\n"
      "  Structure outside <FILL_XXX> slots differs from the puzzle.",
    )

  # Re-mask and compare; also yields actual_counts for Stage 8.
  actual_counts = check_remasking(
    sol_leaf,
    sol_src,
    puzzle_text,
    mask_set,
    defined_funcs,
  )

  # -------------------------------------------------------------------------
  # Stage 4 - FAIL_COMPILE: syntax compile checking.
  # -------------------------------------------------------------------------
  try:
    compile(sol_src, solution, "exec")
  except Exception as e:
    fail(
      CheckResult.FAIL_COMPILE,
      f"Solution is not valid Python code (compilation error: {e}).",
    )

  # -------------------------------------------------------------------------
  # Stage 5 - FAIL_CFG: CFG topology check.
  # -------------------------------------------------------------------------
  check_cfg(sol_leaf, sol_src, cfg_edges)

  # -------------------------------------------------------------------------
  # Stage 6 - FAIL_TIMEOUT (raised by run_python_solution)
  # Stage 7 - FAIL_PATH  +  Stage 8 - FAIL_OUTPUT
  # -------------------------------------------------------------------------
  # Run the stripped source (same text used for parsing and CFG analysis);
  # executing the original file could see a different program if it still
  # carried the refractir_ prefix.
  with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as tf:
    tf.write(sol_src)
    sol_path = tf.name
  try:
    trace, exit_code = run_python_solution(sol_path)
  finally:
    os.unlink(sol_path)
  check_path(trace, expected_path)
  check_output(exit_code)

  # -------------------------------------------------------------------------
  # Stage 9 - FAIL_FILL_CONST
  # -------------------------------------------------------------------------
  check_fill_const_budget(actual_counts, const_counts)


def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="Puzzle Checker - validates a candidate solution against a Python codoku puzzle.",
  )
  p.add_argument("puzzle", help="Puzzle file path (.py with <FILL_XXX> marks).")
  p.add_argument("solution", help="Candidate solution file path (.py).")
  return p


def main() -> int:
  parser = build_arg_parser()
  args = parser.parse_args()
  try:
    check(args.puzzle, args.solution)
  except CheckFailure as exc:
    print(f"[{exc.result.value}] {exc.message}", file=sys.stderr)
    return 1
  print("[PASS] Solution is valid!")
  return 0


if __name__ == "__main__":
  sys.exit(main())

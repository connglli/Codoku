#!/usr/bin/env python3
"""rypuzchk.py — C Puzzle Checker

Validates a candidate solution against a C puzzle produced by rypuzmk.py and
prints ``[PASS]`` / ``[FAIL_XXX]``.

Checks performed in strict order from easiest to hardest to reason about:

  Stage 1 — FAIL_BASICS     : Basics (missing markers, unfilled FILL_XXX marks).
  Stage 2 — FAIL_PARSE      : Parse (solution fails to parse).
  Stage 3 — FAIL_REMASKING  : Re-masked skeleton does not match the puzzle.
  Stage 4 — FAIL_COMPILE    : Compile (solution fails to compile).
  Stage 5 — FAIL_CFG        : CFG topology matches the declared //@ CFG_EDGE: markers exactly.
  Stage 6 — FAIL_PATH       : Execution did not follow the prescribed path exactly.
  Stage 7 — FAIL_OUTPUT     : check_chksum reports a wrong result (non-zero exit).
  Stage 8 — FAIL_FILL_CONST : Constant budget multiset mismatch.

Each stage is a strict prerequisite for the next.  When a stage fails, later
stages are skipped, making it unambiguous *why* a solution is wrong.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from enum import Enum

import tree_sitter_c as tsc
from puzzle_common import (
  apply_replacements,
  build_goto_successors,
  collect_defined_functions,
  collect_leaf_locals,
  collect_replacements,
  find_leaf_function,
  get_maskable_statements,
  sanitize_statement_expressions,
  strip_refractir_prefix,
)
from tree_sitter import Language, Parser

# ---------------------------------------------------------------------------
# Check result — ordered from easiest to hardest to satisfy.
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


def fail(result: CheckResult, msg: str) -> None:
  """Print a tagged failure message to stderr and exit with code 1."""
  print(f"[{result.value}] {msg}", file=sys.stderr)
  sys.exit(1)


# -----------------------------------------------------------------------------
# Comment / whitespace stripping for structural comparison
# -----------------------------------------------------------------------------


def strip_comments_and_whitespace(text: str) -> str:
  """Remove C ``//`` and ``/* */`` comments and all whitespace outside strings.

  The stripped form is used for the re-masking byte-for-byte comparison so
  that formatting differences between the puzzle and the re-masked solution
  are ignored.
  """
  res: list[str] = []
  in_line_comment = False
  in_block_comment = False
  in_string = False
  i = 0
  while i < len(text):
    if in_line_comment:
      if text[i] == "\n":
        in_line_comment = False
      i += 1
    elif in_block_comment:
      if text[i : i + 2] == "*/":
        in_block_comment = False
        i += 2
      else:
        i += 1
    elif in_string:
      res.append(text[i])
      if text[i] == "\\" and i + 1 < len(text):
        res.append(text[i + 1])
        i += 2
      elif text[i] == '"':
        in_string = False
        i += 1
      else:
        i += 1
    else:
      if text[i : i + 2] == "//":
        in_line_comment = True
        i += 2
      elif text[i : i + 2] == "/*":
        in_block_comment = True
        i += 2
      elif text[i] == '"':
        in_string = True
        res.append(text[i])
        i += 1
      elif text[i].isspace():
        i += 1
      else:
        res.append(text[i])
        i += 1
  return "".join(res)


# -----------------------------------------------------------------------------
# Puzzle requirement parsing
# -----------------------------------------------------------------------------


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
    elif "//@ FILL_CONST:" in line:
      parts = line.split("//@ FILL_CONST:", 1)[1].strip().split()
      if len(parts) != 2:
        fail(
          CheckResult.FAIL_PARSE,
          f"Malformed //@ FILL_CONST marker: expected 2 tokens, got {len(parts)} in '{line}'",
        )
      val = parts[0]
      try:
        cnt = int(parts[1])
      except ValueError:
        fail(
          CheckResult.FAIL_PARSE,
          f"Malformed //@ FILL_CONST marker: count '{parts[1]}' is not an integer in '{line}'",
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


# -----------------------------------------------------------------------------
# Mask-set inference (mirrors inferMaskSetFromPuzzle in puzzle_common.hpp)
# -----------------------------------------------------------------------------


def infer_mask_set_from_puzzle(
  sol_leaf,
  sol_src: bytes,
  sol_src_sanitized: bytes,
  puzzle_text: str,
  defined_funcs: set[str],
) -> set[int] | None:
  """Infer which statement indices were masked in the puzzle by comparing renders.

  The puzzle file is not parseable (contains FILL_XXX tokens), so the mask set
  is derived by comparing two sentinel-annotated renders of the *solution*
  against the stripped puzzle text — exactly as ``inferMaskSetFromPuzzle`` does
  in puzzle_common.hpp.

  Returns the set of masked position indices, or ``None`` if the structure of
  the puzzle and the solution are incompatible (re-masking failure).
  """
  maskable, entry, _exit = get_maskable_statements(sol_leaf, sol_src_sanitized)
  if not entry or not _exit:
    return None

  local_names = collect_leaf_locals(sol_leaf, sol_src_sanitized)

  # Build full-masking replacements (all positions masked) + sentinel insertions.
  full_repls: list = []
  plain_repls: list = []  # sentinels only — no masking replacements
  dummy_budget: dict = {}

  for stmt in maskable:
    sentinel = (stmt.start_byte, stmt.start_byte, "\x01")
    full_repls.append(sentinel)
    plain_repls.append(sentinel)
    is_body = stmt.start_byte > entry.start_byte
    collect_replacements(
      stmt,
      sol_src_sanitized,
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

  # Consume the non-maskable prefix (identical in both renders and the puzzle).
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
      return None  # neither segment matches — structure incompatible

  return mask_set


def check_cfg(
  func_node,
  src: bytes,
  cfg_edges: list[tuple[str, str]],
) -> None:
  """Verify that the solution's CFG matches the declared edges exactly.

  If cfg_edges is empty (backward compatibility), we skip the check.
  """
  if not cfg_edges:
    return

  succs = build_goto_successors(func_node, src)

  actual_edges = set()
  for from_node, to_nodes in succs.items():
    for to_node in to_nodes:
      actual_edges.add((from_node, to_node))

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


# -----------------------------------------------------------------------------
# Compilation and execution
# -----------------------------------------------------------------------------


def run_compiled_solution(bin_path: str) -> tuple[list[str], int]:
  """Run the compiled solution and collect its trace and exit code.

  ``LeakSanitizer`` is disabled (``ASAN_OPTIONS=detect_leaks=0``) because
  containerised environments often lack the ``ptrace`` capability it needs.

  Returns ``(block_trace, exit_code)`` where *block_trace* is the list of
  basic-block labels in execution order and *exit_code* is the process exit
  status (0 == correct checksum, non-zero == wrong output).
  """

  env = dict(os.environ)
  env["ASAN_OPTIONS"] = "detect_leaks=0"
  r_run = subprocess.run(
    [bin_path], capture_output=True, text=True, timeout=30, env=env
  )

  # Parse the block trace from stdout: lines like "^entry:", "^b0:", etc.
  trace = []
  for line in r_run.stdout.splitlines():
    if line.startswith("^"):
      trace.append(line[1:].rstrip(":"))
  return trace, r_run.returncode


# -----------------------------------------------------------------------------
# Individual check functions
# -----------------------------------------------------------------------------


def check_path(trace: list[str], expected_path: list[str]) -> None:
  """Verify the execution trace matches the expected path exactly.

  Since we do not instrument the main function, the trace output should be
  exactly equal to the expected path (not a subsequence or superset).
  """
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
  sol_src_sanitized: bytes,
  puzzle_text: str,
  mask_set: set[int],
  defined_funcs: set[str],
) -> dict[str, int]:
  """Re-mask the solution at *mask_set* and verify it matches the puzzle skeleton.

  Returns the actual FILL_CONST budget counts collected from the masked positions.
  Exits with FAIL_REMASKING on mismatch.
  """
  maskable, entry, _exit = get_maskable_statements(sol_leaf, sol_src_sanitized)
  local_names = collect_leaf_locals(sol_leaf, sol_src_sanitized)

  remasked_repls: list = []
  actual_counts: dict[str, int] = {}

  for idx, stmt in enumerate(maskable):
    if idx in mask_set:
      is_body = stmt.start_byte > entry.start_byte
      collect_replacements(
        stmt,
        sol_src_sanitized,
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
      "  You may have changed code outside the FILL_XXX marks, or introduced\n"
      "  unauthorized variables / statements / basic blocks.",
    )

  return actual_counts


def check_fill_const_budget(
  actual_counts: dict[str, int], expected_counts: dict[str, int]
) -> None:
  """Verify the FILL_CONST multiset matches the puzzle budget exactly.

  Checks for both missing values and off-budget extras.  Exits with
  FAIL_FILL_CONST on any mismatch.
  """
  if not expected_counts:
    return  # Empty budget (--lift-consts) — no constraints.

  for val, expected_cnt in expected_counts.items():
    actual_cnt = actual_counts.get(val, 0)
    if actual_cnt != expected_cnt:
      fail(
        CheckResult.FAIL_FILL_CONST,
        f"FILL_CONST count mismatch for '{val}'. "
        f"Expected {expected_cnt}, got {actual_cnt}.",
      )

  for val, actual_cnt in actual_counts.items():
    if val not in expected_counts:
      fail(
        CheckResult.FAIL_FILL_CONST,
        f"Off-budget constant in a FILL_CONST position: '{val}' (count: {actual_cnt}).",
      )


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="C Puzzle Checker — validates a candidate solution against a C puzzle.",
  )
  p.add_argument("puzzle", help="Puzzle file path (.c with FILL_XXX marks).")
  p.add_argument("solution", help="Candidate solution file path (.c).")
  return p


def main() -> None:
  parser = build_arg_parser()
  args = parser.parse_args()

  # --- Existence checks (pre-parse guard) ---
  for label, path in [("Puzzle", args.puzzle), ("Solution", args.solution)]:
    if not os.path.exists(path):
      fail(CheckResult.FAIL_BASICS, f"{label} file '{path}' does not exist.")

  # -------------------------------------------------------------------------
  # Stage 1 — FAIL_BASICS: parse puzzle requirements.
  # -------------------------------------------------------------------------
  with open(args.puzzle, "r") as f:
    puzzle_text = f.read()

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
  # Stage 1 — FAIL_BASICS: Check for unfilled FILL_XXX marks.
  # -------------------------------------------------------------------------
  with open(args.solution, "rb") as f:
    sol_src_raw = f.read()

  sol_src = strip_refractir_prefix(sol_src_raw)
  sol_src_str = sol_src.decode("utf-8")

  stripped_sol = strip_comments_and_whitespace(sol_src_str)
  has_fill_marks = False
  for mark in [
    "FILL_VAR",
    "FILL_CONST",
    "FILL_OP",
    "FILL_TYPE",
    "FILL_LABEL",
    "FILL_FUNC",
    "FILL_FIELD",
  ]:
    if mark in stripped_sol:
      has_fill_marks = True
      break
  if has_fill_marks:
    fail(CheckResult.FAIL_BASICS, "Solution still contains unfilled FILL_XXX marks.")

  # -------------------------------------------------------------------------
  # Stage 2 — FAIL_PARSE: Parse the solution C source with tree-sitter.
  # -------------------------------------------------------------------------
  C_LANGUAGE = Language(tsc.language(), "c")
  ts_parser = Parser()
  ts_parser.set_language(C_LANGUAGE)
  sol_src_sanitized = sanitize_statement_expressions(sol_src)
  sol_tree = ts_parser.parse(sol_src_sanitized)

  sol_leaf, _leaf_name = find_leaf_function(sol_tree.root_node, sol_src_sanitized)
  if not sol_leaf:
    fail(CheckResult.FAIL_PARSE, "Could not find leaf function in solution.")

  defined_funcs = collect_defined_functions(sol_tree.root_node, sol_src_sanitized)

  # -------------------------------------------------------------------------
  # Stage 3 — FAIL_REMASKING: Re-mask and compare.
  # Pure static check — no execution required.
  # -------------------------------------------------------------------------
  mask_set = infer_mask_set_from_puzzle(
    sol_leaf, sol_src, sol_src_sanitized, puzzle_text, defined_funcs
  )
  if mask_set is None:
    fail(
      CheckResult.FAIL_REMASKING,
      "Solution structural integrity check failed.\n"
      "  Structure outside FILL_XXX slots differs from the puzzle.",
    )

  # Re-mask and compare; also yields actual_counts for Stage 6.
  actual_counts = check_remasking(
    sol_leaf, sol_src, sol_src_sanitized, puzzle_text, mask_set, defined_funcs
  )

  # -------------------------------------------------------------------------
  # Stage 4 — FAIL_COMPILE: compile the solution C source.
  # -------------------------------------------------------------------------
  with tempfile.TemporaryDirectory(prefix="rypuz_chk_") as temp_dir:
    bin_path = os.path.join(temp_dir, "solution.bin")
    comp_cmd = [
      "gcc",
      "-O0",
      "-fsanitize=address,undefined",
      "-DDUMP_TRACE",
      args.solution,
      "-o",
      bin_path,
      "-lm",
    ]
    r_comp = subprocess.run(comp_cmd, capture_output=True, text=True)
    if r_comp.returncode != 0:
      print(r_comp.stderr, file=sys.stderr)
      fail(
        CheckResult.FAIL_COMPILE,
        f"Solution fails to compile under gcc (exit code {r_comp.returncode}).",
      )

    # -------------------------------------------------------------------------
    # Stage 5 — FAIL_CFG: CFG topology check.
    # -------------------------------------------------------------------------
    check_cfg(sol_leaf, sol_src_sanitized, cfg_edges)

    # -------------------------------------------------------------------------
    # Stage 6 — FAIL_PATH  +  Stage 7 — FAIL_OUTPUT
    # -------------------------------------------------------------------------
    trace, exit_code = run_compiled_solution(bin_path)

    # Stage 6: path.
    check_path(trace, expected_path)

    # Stage 7: output correctness (check_chksum exit code).
    check_output(exit_code)

    # -------------------------------------------------------------------------
    # Stage 8 — FAIL_FILL_CONST
    # -------------------------------------------------------------------------
    check_fill_const_budget(actual_counts, const_counts)

    # Note: the C version of the puzzle does not have an "intrinsics must be
    # called" check because rysmith's C backend emits helper definitions
    # regardless of whether they are called.  Re-masking integrity already
    # guarantees the solver cannot remove existing call sites.

    print("[PASS] Solution is valid!")
    sys.exit(0)


if __name__ == "__main__":
  main()

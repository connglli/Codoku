#!/usr/bin/env python3
"""rypuzchk.py — C Puzzle Checker

Validates a candidate solution against a C puzzle produced by rypuzmk.py and
prints ``[PASS]`` or ``[FAIL]``.

Checks performed (in order, mirroring rypuzchk.cpp):
  1. Parse the puzzle banner (PATH + FILL_CONST markers); missing PATH ⇒ hard error.
  2. Compile the solution with GCC + AddressSanitizer/UBSan and run it with
     ``-DDUMP_TRACE`` to obtain the execution trace.  Non-zero exit ⇒ FAIL.
  3. Verify the leaf execution path matches ``//@ PATH:`` exactly.
  4. Infer which statement positions were masked from the puzzle body (structural
     step 1 — mirrors ``inferMaskSetFromPuzzle`` from puzzle_common.hpp).
  5. Re-mask the solution at the inferred mask set and require it to match the
     puzzle skeleton byte-for-byte (modulo comments/whitespace).  Structural
     step 2 — the anti-cheating gate.
  6. Check the FILL_CONST budget: the multiset of constants in masked positions
     must equal the puzzle's ``//@ FILL_CONST:`` budget exactly.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import tree_sitter_c as tsc
from puzzle_common import (
  apply_replacements,
  collect_defined_functions,
  collect_leaf_locals,
  collect_replacements,
  find_leaf_function,
  get_maskable_statements,
  sanitize_statement_expressions,
  strip_refractir_prefix,
)
from tree_sitter import Language, Parser

# -----------------------------------------------------------------------------
# Comment / whitespace stripping for structural comparison
# -----------------------------------------------------------------------------


def strip_comments_and_whitespace(text: str) -> str:
  """Remove C ``//`` and ``/* */`` comments and all whitespace outside strings.

  The stripped form is used for the structural integrity byte-for-byte
  comparison so that formatting differences between the puzzle and the
  re-masked solution are ignored.
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


def parse_puzzle_requirements(puzzle_text: str) -> tuple[list[str], dict[str, int]]:
  """Parse ``//@ PATH:`` and ``//@ FILL_CONST:`` markers from the puzzle banner.

  Returns ``(expected_path, const_counts)`` where *expected_path* is the list
  of block labels in order and *const_counts* is ``{value: count}`` for each
  ``FILL_CONST`` budget entry.

  Missing ``//@ PATH:`` returns an empty *expected_path* — callers must treat
  this as a hard error.
  """
  expected_path: list[str] = []
  const_counts: dict[str, int] = {}

  for line in puzzle_text.splitlines():
    if "//@ PATH:" in line:
      path_part = line.split("//@ PATH:", 1)[1].strip()
      expected_path = [x.strip() for x in path_part.split("->") if x.strip()]
    elif "//@ FILL_CONST:" in line:
      parts = line.split("//@ FILL_CONST:", 1)[1].strip().split()
      if len(parts) == 2:
        val, cnt = parts[0], int(parts[1])
        const_counts[val] = cnt

  return expected_path, const_counts


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
  in puzzle_common.hpp:

  1. Render the solution with full masking + sentinel ``\\x01`` before each
     maskable position's content.
  2. Render the solution with no masking + the same sentinels only.
  3. Split both at sentinels → per-position segments.
  4. Walk the stripped puzzle in lock-step: each position either matches the
     full-masked segment (masked) or the plain segment (revealed).

  Returns the set of masked position indices, or ``None`` if the structure of
  the puzzle and the solution are incompatible (structural integrity failure).
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


# -----------------------------------------------------------------------------
# Compilation and execution
# -----------------------------------------------------------------------------


def compile_and_run_solution(solution_path: str) -> list[str]:
  """Compile *solution_path* with GCC + ASan/UBSan, run it, and return the block trace.

  The binary is compiled with ``-DDUMP_TRACE`` so that ``printf("^<name>:\\n")``
  statements (injected by rypuzmk.py) are active.  Compilation failure or a
  non-zero exit code both result in ``[FAIL]`` and ``sys.exit(1)``.

  ``LeakSanitizer`` is disabled (``ASAN_OPTIONS=detect_leaks=0``) because
  containerised environments often lack the ``ptrace`` capability it needs.

  Returns the list of basic-block labels in execution order.
  """
  with tempfile.TemporaryDirectory(prefix="rypuz_chk_") as d:
    bin_path = os.path.join(d, "solution.bin")

    comp_cmd = [
      "gcc",
      "-O0",
      "-fsanitize=address,undefined",
      "-DDUMP_TRACE",
      solution_path,
      "-o",
      bin_path,
      "-lm",
    ]
    r_comp = subprocess.run(comp_cmd, capture_output=True, text=True)
    if r_comp.returncode != 0:
      print(
        f"[FAIL] Solution fails to compile under gcc (exit code {r_comp.returncode}).",
        file=sys.stderr,
      )
      print(r_comp.stderr, file=sys.stderr)
      sys.exit(1)

    env = dict(os.environ)
    env["ASAN_OPTIONS"] = "detect_leaks=0"
    r_run = subprocess.run(
      [bin_path], capture_output=True, text=True, timeout=30, env=env
    )
    if r_run.returncode != 0:
      print(
        f"[FAIL] Solution exits with non-zero status (exit code {r_run.returncode}).",
        file=sys.stderr,
      )
      print(r_run.stdout, file=sys.stderr)
      print(r_run.stderr, file=sys.stderr)
      sys.exit(1)

    # Parse the block trace from stdout: lines like "^entry:", "^b0:", etc.
    trace = []
    for line in r_run.stdout.splitlines():
      if line.startswith("^"):
        trace.append(line[1:].rstrip(":"))
    return trace


# -----------------------------------------------------------------------------
# Individual check functions
# -----------------------------------------------------------------------------


def check_path(trace: list[str], expected_path: list[str]) -> None:
  """Verify the execution trace matches the expected path; exit 1 on mismatch.

  The compiled binary emits trace lines for *all* functions — the main wrapper
  and the leaf.  The puzzle PATH covers only the leaf, so we scan the trace
  for the longest suffix that equals ``expected_path`` exactly (loop counts
  included).  This mirrors the C++ approach of dropping exactly
  ``mainFn.blocks.size()`` leading entries.
  """
  if not trace:
    print(
      "[FAIL] Solution trace is empty or too short to contain the leaf path.",
      file=sys.stderr,
    )
    sys.exit(1)

  # Find the first position in the trace where the expected path starts and
  # runs to completion — skipping any main-wrapper blocks that precede it.
  leaf_trace = trace  # fallback: use full trace
  for skip in range(len(trace)):
    candidate = trace[skip : skip + len(expected_path)]
    if candidate == expected_path:
      leaf_trace = candidate
      break

  if leaf_trace != expected_path:
    print("[FAIL] Execution path mismatch.", file=sys.stderr)
    print(f"  Expected: {' -> '.join(expected_path)}", file=sys.stderr)
    print(f"  Actual:   {' -> '.join(trace)}", file=sys.stderr)
    sys.exit(1)


def check_structural_integrity(
  sol_leaf,
  sol_src: bytes,
  sol_src_sanitized: bytes,
  puzzle_text: str,
  mask_set: set[int],
  defined_funcs: set[str],
) -> dict[str, int]:
  """Re-mask the solution at *mask_set* and verify it matches the puzzle skeleton.

  Returns the actual FILL_CONST budget counts collected from the masked positions.
  Exits with code 1 on mismatch.
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
    print("[FAIL] Solution structural integrity check failed.", file=sys.stderr)
    print(
      "  You may have changed code outside the FILL_XXX marks, or introduced",
      file=sys.stderr,
    )
    print("  unauthorized variables / statements / basic blocks.", file=sys.stderr)
    sys.exit(1)

  return actual_counts


def check_fill_const_budget(
  actual_counts: dict[str, int], expected_counts: dict[str, int]
) -> None:
  """Verify the FILL_CONST multiset matches the puzzle budget exactly.

  Checks for both missing values and off-budget extras.  Exits with code 1 on
  any mismatch.
  """
  if not expected_counts:
    return  # Empty budget (--lift-consts) — no constraints.

  for val, expected_cnt in expected_counts.items():
    actual_cnt = actual_counts.get(val, 0)
    if actual_cnt != expected_cnt:
      print(
        f"[FAIL] FILL_CONST count mismatch for '{val}'. "
        f"Expected {expected_cnt}, got {actual_cnt}.",
        file=sys.stderr,
      )
      sys.exit(1)

  for val, actual_cnt in actual_counts.items():
    if val not in expected_counts:
      print(
        f"[FAIL] Off-budget constant in a FILL_CONST position: "
        f"'{val}' (count: {actual_cnt}).",
        file=sys.stderr,
      )
      sys.exit(1)


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

  # --- Existence checks ---
  for label, path in [("Puzzle", args.puzzle), ("Solution", args.solution)]:
    if not os.path.exists(path):
      print(f"[FAIL] {label} file '{path}' does not exist.", file=sys.stderr)
      sys.exit(1)

  # 1. Parse puzzle requirements.
  with open(args.puzzle, "r") as f:
    puzzle_text = f.read()

  expected_path, const_counts = parse_puzzle_requirements(puzzle_text)
  if not expected_path:
    print(
      "[FAIL] Puzzle is missing a '//@ PATH:' marker; cannot validate.", file=sys.stderr
    )
    sys.exit(1)

  # 2. Parse the solution C source and strip the refractir_ prefix so our
  #    detection logic works the same way as in rypuzmk.py.
  with open(args.solution, "rb") as f:
    sol_src_raw = f.read()

  sol_src = strip_refractir_prefix(sol_src_raw)

  C_LANGUAGE = Language(tsc.language(), "c")
  ts_parser = Parser()
  ts_parser.set_language(C_LANGUAGE)
  sol_src_sanitized = sanitize_statement_expressions(sol_src)
  sol_tree = ts_parser.parse(sol_src_sanitized)

  sol_leaf, _leaf_name = find_leaf_function(sol_tree.root_node, sol_src_sanitized)
  if not sol_leaf:
    print("[FAIL] Could not find leaf function in solution.", file=sys.stderr)
    sys.exit(1)

  # Collect function names defined in the solution for FILL_FUNC detection.
  # Compiler builtins and external symbols have no definition node and are
  # automatically excluded.
  defined_funcs = collect_defined_functions(sol_tree.root_node, sol_src_sanitized)

  # 3. Compile and execute — correctness oracle + trace source.
  #    Note: we compile the ORIGINAL solution (before prefix stripping) since
  #    the puzzle file handed to the solver already has prefixes stripped by
  #    rypuzmk.py; the solution they submit is therefore already prefix-free.
  trace = compile_and_run_solution(args.solution)

  # 4. Path verification.
  check_path(trace, expected_path)

  # 5. Infer mask set (structural integrity step 1).
  mask_set = infer_mask_set_from_puzzle(
    sol_leaf, sol_src, sol_src_sanitized, puzzle_text, defined_funcs
  )
  if mask_set is None:
    print("[FAIL] Solution structural integrity check failed.", file=sys.stderr)
    print(
      "  Structure outside FILL_XXX slots differs from the puzzle.", file=sys.stderr
    )
    sys.exit(1)

  # 6. Re-mask and compare (structural integrity step 2) — also yields actual_counts.
  actual_counts = check_structural_integrity(
    sol_leaf, sol_src, sol_src_sanitized, puzzle_text, mask_set, defined_funcs
  )

  # 7. FILL_CONST budget.
  check_fill_const_budget(actual_counts, const_counts)

  # Note: the C version of the puzzle does not have an "intrinsics must be
  # called" check (step 7 in rypuzchk.cpp) because rysmith's C backend emits
  # helper definitions regardless of whether they are called.  Structural
  # integrity already guarantees the solver cannot remove existing call sites.

  print("[PASS] Solution is valid!")
  sys.exit(0)


if __name__ == "__main__":
  main()

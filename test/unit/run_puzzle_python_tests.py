"""End-to-end tests for Python puzzle tooling (rypuzmk / rypuzchk with python target).

Includes unit tests for collect_python_replacements that verify the --no-ub-guards
masking behaviour:  old helper calls (_iadd, _fadd, _sdiv, …) must NOT appear in the
FILL_OP handler list (they are never emitted by the updated backend), and the new
inline forms (BinOp //, IfExp for truncating div, etc.) must be masked correctly.
"""

import os
import re
import subprocess
import sys
import tempfile

GREEN = "\033[32m"
RED = "\033[31m"
GRAY = "\033[90m"
NC = "\033[0m"

results = []


def run(cmd, **kw):
  print(f"  {GRAY}[RUN>]{NC} " + " ".join(cmd))
  return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)


def check(name, ok, detail=""):
  results.append((name, ok, detail))
  color = GREEN if ok else RED
  tag = "PASS" if ok else "FAIL"
  print(f"  [{color}{tag}{NC}] {name}" + (f" — {detail}" if detail and not ok else ""))


def make_puzzle(rypuzmk, rysmith, seed, outdir, *options):
  """Generate a puzzle + ground truth for `seed`; return (puzzle_path, gt_path)."""
  puzzle = os.path.join(outdir, f"p{seed}.py")
  gt = os.path.join(outdir, f"p{seed}.gt.py")
  r = run(
    [
      rypuzmk,
      "--seed",
      str(seed),
      "--rysmith",
      rysmith,
      "--target",
      "python",
      "-o",
      puzzle,
      "--keep-ground-truth",
    ]
    + list(options)
  )
  ok = r.returncode == 0 and os.path.exists(puzzle) and os.path.exists(gt)
  return (puzzle, gt) if ok else (None, None)


def chk(rypuzchk, puzzle, solution):
  """Run rypuzchk; return (passed, combined_output)."""
  r = run([rypuzchk, puzzle, solution])
  return r.returncode == 0, (r.stdout + r.stderr)


def split_header_body(text):
  """Split a Python file into its leading comment banner and the code body."""
  m = re.search(r"\ndef\s+func_", text)
  i = m.start() + 1 if m else 0
  return text[:i], text[i:]


def first_body_stmt_line(gt_text):
  """Index of the first indented statement line inside a basic block."""
  lines = gt_text.splitlines()
  in_body = False
  for idx, ln in enumerate(lines):
    s = ln.strip()
    if re.match(r"^\s*#\s*\^(\w+)", ln) or re.match(r"^\s*#\s*exit:", ln):
      in_body = True
      continue
    if (
      in_body
      and s
      and not s.startswith("#")
      and not s.startswith("def ")
      and not s.startswith("return ")
      and not s.startswith("global ")
      and not s.endswith(":")
      and "=" in s
    ):
      return idx
  return -1


def find_body_line(gt_text, pattern, last=False):
  """Index of the first (or last) stripped line matching `pattern`."""
  rx = re.compile(pattern)
  lines = gt_text.splitlines()
  found = -1
  for idx, ln in enumerate(lines):
    s = ln.strip()
    if not s.startswith("#") and rx.search(s):
      found = idx
      if not last:
        return idx
  return found


def dup_line(gt_text, idx):
  """Return `gt_text` with line `idx` duplicated in place (a structural insert)."""
  lines = gt_text.splitlines(keepends=True)
  return "".join(lines[:idx] + [lines[idx], lines[idx]] + lines[idx + 1 :])


# ---------------------------------------------------------------------------
# Unit tests for collect_python_replacements (--no-ub-guards behaviour)
# ---------------------------------------------------------------------------
# These tests exercise puzzle_common.collect_python_replacements directly,
# bypassing rysmith and the end-to-end pipeline.  They target invariants that
# were introduced when --no-ub-guards became the default for Python puzzles:
#   * Old UB-guard helpers (_iadd, _isub, …) must NOT appear in the FILL_OP
#     function list — they are never emitted by the updated backend and dead
#     code in the list only adds confusion.
#   * The floor-division operator `//` must be masked as a two-character
#     FILL_OP token (the old code only matched the first '/' and left a
#     trailing '/' visible).
#   * The inline truncating-div IfExp pattern emitted by --no-ub-guards must
#     have its `if` / `else` keywords masked and the nested `//` fully hidden.


def _collect_ops_on(src_str: str, stmt_src: str) -> list[str]:
  """Return all FILL_XXX tags produced by masking *stmt_src* inside *src_str*.

  *src_str* is a minimal Python module that contains *stmt_src* as a body
  statement in a function named ``func_test``.  We parse it, locate the
  function, collect maskable statements, and call collect_python_replacements.
  The returned list contains one tag string per replacement, in source order.
  """
  import ast
  import os
  import sys

  # Ensure puzzle_common is importable from the project root.
  puzzle_dir = os.path.join(os.path.dirname(__file__), "..", "..", "puzzle", "target")
  if puzzle_dir not in sys.path:
    sys.path.insert(0, puzzle_dir)
  from puzzle_common import (
    collect_python_replacements,
    find_python_leaf_function,
    get_python_maskable_statements,
  )

  src_bytes = src_str.encode("utf-8")
  tree = ast.parse(src_bytes)
  leaf, _ = find_python_leaf_function(tree, src_bytes)
  assert leaf is not None, "No leaf function found in test source"
  maskable, entry_line, exit_line = get_python_maskable_statements(leaf, src_bytes)
  assert maskable, f"No maskable statements found; source:\n{src_str}"
  replacements = []
  for stmt in maskable:
    collect_python_replacements(
      stmt, src_bytes, True, replacements, {}, {"x", "y", "a", "b", "acc"}, set()
    )
  # Build tag list in source order.
  sorted_repls = sorted(replacements, key=lambda r: r[0])
  return [tag for (_, _, tag) in sorted_repls]


# Minimal module wrapper: defines a leaf function with ^entry / ^exit comments.
_MODULE_TMPL = """\
def func_test(x, y):
    # ^entry
    acc = 0
    # ^b0
    {stmt}
    # ^exit
    return acc
"""


def test_floor_div_two_char_fill_op() -> bool:
  """// must be masked as a single two-character FILL_OP, not '/' + leftover '/'.

  Under --no-ub-guards the Python backend emits  ``a // b`` for truncating
  integer division.  The BinOp operator-detection code used to search for the
  single-byte pattern b'/' which matches only the first '/' of '//'.  That
  left a stray '/' character visible in the puzzle, making FILL_OP/ (with
  trailing slash) appear instead of FILL_OP.

  This test verifies that '// ' is matched as the *two*-character sequence
  b'//' so the whole operator is replaced.
  """
  src = _MODULE_TMPL.format(stmt="acc = (x // y)")
  tags = _collect_ops_on(src, "acc = (x // y)")
  # We expect exactly one FILL_OP for the '//' operator.
  # With the old single-'/' search the operator token was only 1 char wide and
  # the result would contain an extra '/' visible in the puzzle body.
  return tags.count("FILL_OP") >= 1 and "FILL_OP/" not in "".join(
    # Re-apply to get the masked text and check no stray '/' follows FILL_OP.
    _apply_mask_and_text(src, "acc = (x // y)")
  )


def _apply_mask_and_text(src_str: str, _stmt: str) -> str:
  """Helper: return the masked source text for the first maskable statement."""
  import ast
  import os
  import sys

  puzzle_dir = os.path.join(os.path.dirname(__file__), "..", "..", "puzzle", "target")
  if puzzle_dir not in sys.path:
    sys.path.insert(0, puzzle_dir)
  from puzzle_common import (
    apply_replacements,
    collect_python_replacements,
    find_python_leaf_function,
    get_python_maskable_statements,
  )

  src_bytes = src_str.encode("utf-8")
  tree = ast.parse(src_bytes)
  leaf, _ = find_python_leaf_function(tree, src_bytes)
  maskable, _el, _xl = get_python_maskable_statements(leaf, src_bytes)
  replacements = []
  for stmt in maskable:
    collect_python_replacements(
      stmt, src_bytes, True, replacements, {}, {"x", "y", "a", "b", "acc"}, set()
    )
  return apply_replacements(src_bytes, replacements).decode("utf-8")


def test_iadd_not_in_fill_op_handler() -> bool:
  """_iadd must NOT be masked as FILL_OP (dead entry in the handler list).

  The --no-ub-guards Python backend no longer emits _iadd / _isub / _imul.
  Those names must be absent from the FILL_OP function-call handler so the
  list accurately reflects what the backend can emit.  A call to _iadd in a
  hypothetical old-format puzzle must not receive FILL_OP treatment from the
  updated masker.

  We test this by checking that the FILL_OP handler tuple in puzzle_common
  does NOT contain '_iadd'.
  """
  import inspect
  import os
  import sys

  puzzle_dir = os.path.join(os.path.dirname(__file__), "..", "..", "puzzle", "target")
  if puzzle_dir not in sys.path:
    sys.path.insert(0, puzzle_dir)
  import puzzle_common

  src = inspect.getsource(puzzle_common.collect_python_replacements)
  # The old list checked for '_iadd' by name inside a func_name-in-(...) test;
  # after cleanup that name must be absent.
  return '"_iadd"' not in src and "'_iadd'" not in src


def test_fadd_not_in_fill_op_handler() -> bool:
  """_fadd must NOT be masked as FILL_OP (dead entry in the handler list).

  Same rationale as test_iadd_not_in_fill_op_handler: under --no-ub-guards
  the float helpers _fadd / _fsub / _fmul / _fdiv / _ffmod are inlined as
  Python arithmetic expressions (_f32(a op b)) and must no longer appear in
  the FILL_OP handler.
  """
  import inspect
  import os
  import sys

  puzzle_dir = os.path.join(os.path.dirname(__file__), "..", "..", "puzzle", "target")
  if puzzle_dir not in sys.path:
    sys.path.insert(0, puzzle_dir)
  import puzzle_common

  src = inspect.getsource(puzzle_common.collect_python_replacements)
  return '"_fadd"' not in src and "'_fadd'" not in src


def test_sdiv_not_in_fill_op_handler() -> bool:
  """_sdiv must NOT be masked as FILL_OP (replaced by inline truncating div).

  Under --no-ub-guards, _sdiv(a, b, n) is replaced by the inline IfExp
  ``(a // b if (a < 0) == (b < 0) else -((-a) // b))``.  The _sdiv name
  must therefore be absent from the FILL_OP handler list.
  """
  import inspect
  import os
  import sys

  puzzle_dir = os.path.join(os.path.dirname(__file__), "..", "..", "puzzle", "target")
  if puzzle_dir not in sys.path:
    sys.path.insert(0, puzzle_dir)
  import puzzle_common

  src = inspect.getsource(puzzle_common.collect_python_replacements)
  return '"_sdiv"' not in src and "'_sdiv'" not in src


def test_trunc_div_ifexp_fully_masked() -> bool:
  """The truncating-div IfExp pattern must mask 'if'/'else' and both '//' operators.

  Under --no-ub-guards, integer division emits:
    (a // b if (a < 0) == (b < 0) else -((-a) // b))

  collect_python_replacements must mask:
    - the 'if' keyword             → FILL_OP
    - the 'else' keyword           → FILL_OP
    - both '//' occurrences        → FILL_OP  (two-character token)

  If '//' is still matched as single-char '/', the resulting masked text
  will contain 'FILL_OP/' (stray slash) which is incorrect.
  """
  stmt = "acc = (a // b if (a < 0) == (b < 0) else -((-a) // b))"
  src = _MODULE_TMPL.format(stmt=stmt)
  masked = _apply_mask_and_text(src, stmt)
  # Neither '//' should survive: every occurrence of '//' must be gone.
  # If the two-char fix is in place, the body contains no literal '//'.
  body_start = masked.find("# ^b0")
  body_end = masked.find("# ^exit")
  body = masked[body_start:body_end] if body_start >= 0 and body_end >= 0 else masked
  return "//" not in body


def _unit_tests_pass() -> bool:
  """Run all unit tests; return True if every one passes."""
  tests = [
    ("floor-div // masked as two-char FILL_OP", test_floor_div_two_char_fill_op),
    ("_iadd absent from FILL_OP handler", test_iadd_not_in_fill_op_handler),
    ("_fadd absent from FILL_OP handler", test_fadd_not_in_fill_op_handler),
    ("_sdiv absent from FILL_OP handler", test_sdiv_not_in_fill_op_handler),
    (
      "truncating-div IfExp fully masked (no stray //)",
      test_trunc_div_ifexp_fully_masked,
    ),
    ("while True: literal not masked as FILL_CONST", test_while_true_not_masked),
  ]
  all_passed = True
  for name, fn in tests:
    try:
      ok = fn()
    except Exception as exc:
      ok = False
      check(f"[unit] {name}", False, str(exc))
      all_passed = False
      continue
    check(f"[unit] {name}", ok)
    if not ok:
      all_passed = False
  return all_passed


def test_while_true_not_masked() -> bool:
  """The `True` in `while True:` must NOT be masked as FILL_CONST.

  `while True:` is the structured-lowering canonical form for an infinite
  loop (equivalent to C's ``for (;;)``).  The ``True`` literal is
  *structural* — it is always there and is not a solver-chosen constant.
  Masking it produces a nonsensical ``while FILL_CONST:`` that prevents the
  solver from producing a valid Python program (any constant other than True
  or non-zero would change loop semantics).

  The bug: ``collect_python_replacements`` checks boolean sentinels only when
  ``is_body=False`` (initialiser context).  For body statements (is_body=True)
  the guard is skipped and ``True`` falls through to the FILL_CONST branch.
  The fix: treat ``True`` / ``False`` as unconditional sentinels regardless of
  ``is_body`` — they are never legitimate fill targets.
  """
  # Construct a minimal program with while True using the correct ^entry / ^b0 / ^exit
  # block structure that get_python_maskable_statements expects.
  src = """\
def func_test(x, y):
    # ^entry
    acc = 0
    # ^b0
    acc = 1
    while True:
        # ^b1
        acc = (acc + x)
        if (acc > y):
            break
    # ^exit
    return acc
"""
  masked = _apply_mask_and_text(src, "")
  # "while True" must survive intact in the masked output.
  # If True is masked it becomes "while FILL_CONST".
  body_start = masked.find("# ^b0")
  body_end = masked.find("# ^exit")
  body = masked[body_start:body_end] if body_start >= 0 and body_end >= 0 else masked
  return "while True:" in body and "while FILL_CONST" not in body


def main():
  if len(sys.argv) < 4:
    print("usage: run_puzzle_python_tests.py <rypuzmk> <rypuzchk> <rysmith>")
    return 2
  rypuzmk, rypuzchk, rysmith = sys.argv[1:4]

  # (0) unit tests for --no-ub-guards masking behaviour (no rysmith needed).
  _unit_tests_pass()

  with tempfile.TemporaryDirectory(prefix="rypuztest_py_") as outdir:
    # (1) round-trip across a few seeds
    seeds = [42, 500, 9999]
    base_puzzle, base_gt = None, None
    rt_ok = True
    for s in seeds:
      puzzle, gt = make_puzzle(rypuzmk, rysmith, s, outdir)
      if not puzzle:
        rt_ok = False
        check(f"generate python seed {s}", False, "rypuzmk failed")
        continue
      passed, out = chk(rypuzchk, puzzle, gt)
      check(
        f"python ground truth round-trips (seed {s})",
        passed,
        out.strip().splitlines()[-1:],
      )
      rt_ok = rt_ok and passed
      if s == 42:
        base_puzzle, base_gt = puzzle, gt

    if not rt_ok or not base_puzzle:
      print("Round-trip failed; skipping structural checks.")
      return 1

    base_gt_text = open(base_gt).read()
    base_puzzle_text = open(base_puzzle).read()

    # (2) Verify markers exist in the puzzle header
    header, body_puzzle = split_header_body(base_puzzle_text)
    check("banner has #//@ EXEC_PATH marker", "#//@ EXEC_PATH:" in header)
    check("banner has #//@ CFG_EDGE marker", "#//@ CFG_EDGE:" in header)
    check("banner has #//@ FILL_CONST markers", "#//@ FILL_CONST:" in header)

    # (3) require/assume are dropped from both puzzle and ground truth
    _, body_gt = split_header_body(base_gt_text)
    check("puzzle body has no require/assume", "require" not in body_puzzle)
    check("ground truth has no require/assume", "require" not in body_gt)

    # (4) A solution that still contains FILL marks is rejected
    passed, out = chk(rypuzchk, base_puzzle, base_puzzle)
    check(
      "puzzle-as-solution rejected with FILL marks",
      not passed and "[FAIL_BASICS]" in out,
      out.strip().splitlines()[-1:],
    )

    # (5) Structural integrity: deleting or duplicating a statement is rejected
    # Duplicate a statement
    stmt_idx = first_body_stmt_line(base_gt_text)
    if stmt_idx > 0:
      mut_text = dup_line(base_gt_text, stmt_idx)
      mut_path = os.path.join(outdir, "mut_dup.py")
      with open(mut_path, "w") as f:
        f.write(mut_text)
      passed, out = chk(rypuzchk, base_puzzle, mut_path)
      check(
        "duplicated statement rejected with FAIL_REMASKING",
        not passed and any(t in out for t in ["[FAIL_REMASKING]", "[FAIL_PARSE]"]),
        out.strip().splitlines()[-1:],
      )

      # Delete a statement
      mut_lines = base_gt_text.splitlines(keepends=True)
      del mut_lines[stmt_idx]
      mut_text = "".join(mut_lines)
      mut_path = os.path.join(outdir, "mut_delete.py")
      with open(mut_path, "w") as f:
        f.write(mut_text)
      passed, out = chk(rypuzchk, base_puzzle, mut_path)
      check(
        "deleted statement rejected with FAIL_REMASKING",
        not passed and any(t in out for t in ["[FAIL_REMASKING]", "[FAIL_PARSE]"]),
        out.strip().splitlines()[-1:],
      )
    else:
      check("found a masked statement to duplicate", False, "no candidate statement")

    # (6) FILL_CONST budget: changing a budgeted constant to an off-budget value is rejected
    m = re.findall(r"(\b\d+\b)", base_gt_text)
    budgeted_nums = []
    for num_str in m:
      val = int(num_str)
      if f"#//@ FILL_CONST: {val} " in header:
        budgeted_nums.append(num_str)

    if budgeted_nums:
      target_num = budgeted_nums[0]
      perturbed_val = int(target_num) + 13
      # Only replace the target number inside the program body, not in the header comments.
      hdr_part, body_part = split_header_body(base_gt_text)
      # Find first occurrence of target_num in body
      body_perturbed = re.sub(
        r"\b" + target_num + r"\b", str(perturbed_val), body_part, count=1
      )
      mut_text = hdr_part + body_perturbed
      mut_path = os.path.join(outdir, "mut_offbudget.py")
      with open(mut_path, "w") as f:
        f.write(mut_text)

      passed, out = chk(rypuzchk, base_puzzle, mut_path)
      check(
        "perturbed constant rejected with FAIL_OUTPUT, FAIL_PATH or FAIL_FILL_CONST",
        not passed
        and any(
          t in out for t in ["[FAIL_OUTPUT]", "[FAIL_PATH]", "[FAIL_FILL_CONST]"]
        ),
        out.strip().splitlines()[-1:],
      )
    else:
      check(
        "found a budgeted constant to perturb", False, "no budgeted constant in body"
      )

    # (7) Syntax errors and compilation errors inside mask are rejected
    if stmt_idx > 0:
      mut_lines = base_gt_text.splitlines(keepends=True)
      mut_lines[stmt_idx] = mut_lines[stmt_idx].replace(
        "(", "((", 1
      )  # unbalanced parenthesis
      mut_text = "".join(mut_lines)
      mut_path = os.path.join(outdir, "mut_parse_error.py")
      with open(mut_path, "w") as f:
        f.write(mut_text)
      passed, out = chk(rypuzchk, base_puzzle, mut_path)
      check(
        "syntax error inside mask rejected with FAIL_PARSE",
        not passed and "[FAIL_PARSE]" in out,
        out.strip().splitlines()[-1:],
      )

      # Indentation/compilation error
      mut_lines = base_gt_text.splitlines(keepends=True)
      mut_lines[stmt_idx] = " " * 100 + mut_lines[stmt_idx].strip() + "\n"
      mut_text = "".join(mut_lines)
      mut_path = os.path.join(outdir, "mut_compile_error.py")
      with open(mut_path, "w") as f:
        f.write(mut_text)
      passed, out = chk(rypuzchk, base_puzzle, mut_path)
      # In Python, indentation issues are caught during parsing/compilation stage
      check(
        "compilation/indent error rejected with FAIL_PARSE/FAIL_COMPILE",
        not passed and any(t in out for t in ["[FAIL_PARSE]", "[FAIL_COMPILE]"]),
        out.strip().splitlines()[-1:],
      )

    # (8) FAIL_CFG: altered CFG destination is rejected
    # In python, loop structure or helper calls can be changed. Let's see if we can find a while loop
    # or control keyword to swap.
    # We can also test break/continue swap since Python loops are structured.
    if "break" in base_gt_text:
      swapped = base_gt_text.replace("break", "continue", 1)
      swap_path = os.path.join(outdir, "mut_swap_break.py")
      with open(swap_path, "w") as f:
        f.write(swapped)
      passed, out = chk(rypuzchk, base_puzzle, swap_path)
      check(
        "swapped break->continue rejected (FAIL_PATH or FAIL_CFG)",
        not passed
        and any(t in out for t in ["[FAIL_PATH]", "[FAIL_CFG]", "[FAIL_REMASKING]"]),
        out.strip().splitlines()[-1:],
      )

  # Summarize
  failed = [name for name, ok, _ in results if not ok]
  print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
  if failed:
    print(f"Failed checks: {failed}")
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())

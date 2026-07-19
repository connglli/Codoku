"""End-to-end tests for Python puzzle tooling (rypuzmk / rypuzchk with python target)."""

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


def main():
  if len(sys.argv) < 4:
    print("usage: run_puzzle_python_tests.py <rypuzmk> <rypuzchk> <rysmith>")
    return 2
  rypuzmk, rypuzchk, rysmith = sys.argv[1:4]

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

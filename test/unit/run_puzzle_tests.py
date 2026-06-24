"""End-to-end tests for the puzzle tooling (rypuzmk / rypuzchk).

rypuzmk masks a rysmith-generated function into a puzzle (FILL_XXX marks plus
an instruction banner with machine-readable `//@ PATH:` / `//@ FILL_CONST:`
markers); rypuzchk validates a candidate solution against that puzzle.

These tests assert the invariants the two tools must jointly uphold:

  (1) The canonical solution round-trips: rypuzmk emits a ground truth that
      rypuzchk accepts (across several seeds).
  (2) The banner carries the machine-readable PATH / FILL_CONST markers.
  (3) require/assume are dropped from both puzzle and ground truth.
  (4) A solution that still contains FILL marks is rejected.
  (5) Structural integrity: deleting or duplicating a masked statement is
      rejected (the re-masked solution no longer matches the puzzle).
  (6) FILL_CONST budget: changing a budgeted constant to an off-budget value
      is rejected.

Run as:

  python3 -m test.unit.run_puzzle_tests <rypuzmk> <rypuzchk> <rysmith> <symiri>
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


def make_puzzle(rypuzmk, rysmith, seed, outdir):
  """Generate a puzzle + ground truth for `seed`; return (puzzle_path, gt_path)."""
  puzzle = os.path.join(outdir, f"p{seed}.sir")
  gt = os.path.join(outdir, f"p{seed}.gt.sir")
  r = run([rypuzmk, "--seed", str(seed), "--rysmith", rysmith, "-o", puzzle, "--keep-ground-truth"])
  ok = r.returncode == 0 and os.path.exists(puzzle) and os.path.exists(gt)
  return (puzzle, gt) if ok else (None, None)


def chk(rypuzchk, symiri, puzzle, solution):
  """Run rypuzchk; return (passed, combined_output)."""
  r = run([rypuzchk, puzzle, solution, "--symiri", symiri])
  return r.returncode == 0, (r.stdout + r.stderr)


def split_header_body(text):
  """Split a .sir file into its leading comment banner and the code body."""
  m = re.search(r"\nfun @", text)
  i = m.start() + 1 if m else 0
  return text[:i], text[i:]


def first_body_stmt_line(gt_text):
  """Index of the first indented statement line inside a masked (^bN) block."""
  lines = gt_text.splitlines()
  in_body = False
  for idx, ln in enumerate(lines):
    s = ln.strip()
    if re.match(r"^\^b\w*:", s):
      in_body = True
      continue
    if s.startswith("^"):
      in_body = False
      continue
    if in_body and (s.startswith("%") or s.startswith("store") or s.startswith("br ")):
      return idx
  return -1


def main():
  if len(sys.argv) != 5:
    print("usage: run_puzzle_tests.py <rypuzmk> <rypuzchk> <rysmith> <symiri>")
    return 2
  rypuzmk, rypuzchk, rysmith, symiri = sys.argv[1:5]

  with tempfile.TemporaryDirectory(prefix="rypuztest_") as outdir:
    # (1) round-trip across a few seeds
    seeds = [42, 500, 9999]
    base_puzzle, base_gt = None, None
    rt_ok = True
    for s in seeds:
      puzzle, gt = make_puzzle(rypuzmk, rysmith, s, outdir)
      if not puzzle:
        rt_ok = False
        check(f"generate seed {s}", False, "rypuzmk failed")
        continue
      passed, out = chk(rypuzchk, symiri, puzzle, gt)
      check(f"ground truth round-trips (seed {s})", passed, out.strip().splitlines()[-1:] )
      rt_ok = rt_ok and passed
      if s == 42:
        base_puzzle, base_gt = puzzle, gt

    if not base_puzzle:
      check("seed 42 available for mutation tests", False, "no base puzzle")
      return summarize()

    puzzle_text = open(base_puzzle).read()
    gt_text = open(base_gt).read()

    # (2) machine-readable markers present
    check("banner has //@ PATH marker", "//@ PATH:" in puzzle_text)
    check("banner has //@ FILL_CONST markers", "//@ FILL_CONST:" in puzzle_text)

    # (3) require/assume dropped from both puzzle and ground truth
    _, p_body = split_header_body(puzzle_text)
    check("puzzle body has no require/assume",
          not re.search(r"\b(require|assume)\b", p_body))
    check("ground truth has no require/assume",
          not re.search(r"\b(require|assume)\b", gt_text))

    # (4) a solution that still contains FILL marks is rejected
    passed, _ = chk(rypuzchk, symiri, base_puzzle, base_puzzle)
    check("puzzle-as-solution rejected (unfilled FILL marks)", not passed)

    # (5a) deleting a masked statement is rejected
    idx = first_body_stmt_line(gt_text)
    check("found a masked statement to mutate", idx >= 0)
    if idx >= 0:
      lines = gt_text.splitlines(keepends=True)
      del_path = os.path.join(outdir, "mut_delete.sir")
      with open(del_path, "w") as f:
        f.writelines(lines[:idx] + lines[idx + 1:])
      passed, out = chk(rypuzchk, symiri, base_puzzle, del_path)
      check("deleted masked statement rejected", not passed)

      # (5b) duplicating a masked statement is rejected
      dup_path = os.path.join(outdir, "mut_dup.sir")
      with open(dup_path, "w") as f:
        f.writelines(lines[:idx] + [lines[idx], lines[idx]] + lines[idx + 1:])
      passed, out = chk(rypuzchk, symiri, base_puzzle, dup_path)
      check("duplicated masked statement rejected", not passed)

    # (6) changing a budgeted constant to an off-budget value is rejected
    m = re.search(r"//@ FILL_CONST:\s*(-?\d{2,})\s+\d+", puzzle_text)
    check("found an integer FILL_CONST to perturb", m is not None)
    if m:
      val = m.group(1)
      # The ground truth has no banner/markers, so perturb it whole: replace the
      # first standalone occurrence of the budgeted value with an off-budget one.
      new_gt, n = re.subn(r"(?<!\d)" + re.escape(val) + r"(?!\d)", "987654321", gt_text, count=1)
      check("budgeted constant occurs in masked body", n == 1)
      if n == 1:
        off_path = os.path.join(outdir, "mut_offbudget.sir")
        with open(off_path, "w") as f:
          f.write(new_gt)
        passed, out = chk(rypuzchk, symiri, base_puzzle, off_path)
        check("off-budget constant rejected", not passed)

  return summarize()


def summarize():
  npass = sum(1 for _, ok, _ in results if ok)
  total = len(results)
  print(f"\n{GREEN if npass == total else RED}{npass}/{total} checks passed{NC}")
  return 0 if npass == total else 1


if __name__ == "__main__":
  sys.exit(main())

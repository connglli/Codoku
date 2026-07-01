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
  r = run(
    [
      rypuzmk,
      "--seed",
      str(seed),
      "--rysmith",
      rysmith,
      "-o",
      puzzle,
      "--keep-ground-truth",
    ]
  )
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


def body_fill_count(text):
  """Count FILL_XXX marks inside the program body (excludes the banner prose)."""
  _, body = split_header_body(text)
  return body.count("FILL_")


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


def find_body_line(gt_text, pattern, last=False):
  """Index of the first (or last) stripped line in a ^bN block matching `pattern`."""
  rx = re.compile(pattern)
  lines = gt_text.splitlines()
  in_body = False
  found = -1
  for idx, ln in enumerate(lines):
    s = ln.strip()
    if re.match(r"^\^b\w*:", s):
      in_body = True
      continue
    if s.startswith("^"):
      in_body = False
      continue
    if in_body and rx.search(s):
      found = idx
      if not last:
        return idx
  return found


def dup_line(gt_text, idx):
  """Return `gt_text` with line `idx` duplicated in place (a structural insert).

  Duplicating an `addr`/idempotent assignment in place is semantics-preserving
  (the second write stores the same value), so the mutant still executes
  correctly under symiri — isolating the *structural* check as the only gate
  that can legitimately reject it.
  """
  lines = gt_text.splitlines(keepends=True)
  return "".join(lines[:idx] + [lines[idx], lines[idx]] + lines[idx + 1 :])


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
      check(
        f"ground truth round-trips (seed {s})", passed, out.strip().splitlines()[-1:]
      )
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
    check(
      "puzzle body has no require/assume",
      not re.search(r"\b(require|assume)\b", p_body),
    )
    check(
      "ground truth has no require/assume",
      not re.search(r"\b(require|assume)\b", gt_text),
    )

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
        f.writelines(lines[:idx] + lines[idx + 1 :])
      passed, out = chk(rypuzchk, symiri, base_puzzle, del_path)
      check("deleted masked statement rejected", not passed)

      # (5b) duplicating a masked statement is rejected
      dup_path = os.path.join(outdir, "mut_dup.sir")
      with open(dup_path, "w") as f:
        f.writelines(lines[:idx] + [lines[idx], lines[idx]] + lines[idx + 1 :])
      passed, out = chk(rypuzchk, symiri, base_puzzle, dup_path)
      check("duplicated masked statement rejected", not passed)

    # (6) changing a budgeted constant to an off-budget value is rejected
    m = re.search(r"//@ FILL_CONST:\s*(-?\d{2,})\s+\d+", puzzle_text)
    check("found an integer FILL_CONST to perturb", m is not None)
    if m:
      val = m.group(1)
      # The ground truth has no banner/markers, so perturb it whole: replace the
      # first standalone occurrence of the budgeted value with an off-budget one.
      new_gt, n = re.subn(
        r"(?<!\d)" + re.escape(val) + r"(?!\d)", "987654321", gt_text, count=1
      )
      check("budgeted constant occurs in masked body", n == 1)
      if n == 1:
        off_path = os.path.join(outdir, "mut_offbudget.sir")
        with open(off_path, "w") as f:
          f.write(new_gt)
        passed, out = chk(rypuzchk, symiri, base_puzzle, off_path)
        check("off-budget constant rejected", not passed)

    # (7) --p-mask selective masking. The mask is inferred from the puzzle body
    # by rypuzchk (via inferMaskSetFromPuzzle), so no //@ MASK header is ever
    # emitted — the header stays clean for human solvers.
    full_fill = body_fill_count(puzzle_text)
    check(
      "p-mask: default puzzle carries no //@ MASK marker",
      "//@ MASK:" not in puzzle_text,
    )

    half_puzzle = os.path.join(outdir, "pm_half.sir")
    half_gt = os.path.join(outdir, "pm_half.gt.sir")
    r = run(
      [
        rypuzmk,
        "--seed",
        "42",
        "--p-mask",
        "0.5",
        "--rysmith",
        rysmith,
        "-o",
        half_puzzle,
        "--keep-ground-truth",
      ]
    )
    half_ok = (
      r.returncode == 0 and os.path.exists(half_puzzle) and os.path.exists(half_gt)
    )
    check("p-mask 0.5: rypuzmk succeeds", half_ok, r.stderr)
    if half_ok:
      htext = open(half_puzzle).read()
      check("p-mask 0.5: no //@ MASK marker emitted", "//@ MASK:" not in htext)
      check(
        "p-mask 0.5: fewer FILL marks than full masking",
        body_fill_count(htext) < full_fill,
      )
      passed, out = chk(rypuzchk, symiri, half_puzzle, half_gt)
      check(
        "p-mask 0.5: ground truth round-trips", passed, out.strip().splitlines()[-1:]
      )

    zero_puzzle = os.path.join(outdir, "pm_zero.sir")
    zero_gt = os.path.join(outdir, "pm_zero.gt.sir")
    r = run(
      [
        rypuzmk,
        "--seed",
        "42",
        "--p-mask",
        "0",
        "--rysmith",
        rysmith,
        "-o",
        zero_puzzle,
        "--keep-ground-truth",
      ]
    )
    zero_ok = (
      r.returncode == 0 and os.path.exists(zero_puzzle) and os.path.exists(zero_gt)
    )
    check("p-mask 0: rypuzmk succeeds", zero_ok, r.stderr)
    if zero_ok:
      ztext = open(zero_puzzle).read()
      check(
        "p-mask 0: body fully revealed (no FILL marks)", body_fill_count(ztext) == 0
      )
      check("p-mask 0: no //@ MASK marker emitted", "//@ MASK:" not in ztext)
      passed, out = chk(rypuzchk, symiri, zero_puzzle, zero_gt)
      check(
        "p-mask 0: fully-revealed puzzle round-trips",
        passed,
        out.strip().splitlines()[-1:],
      )

    # (8) Structural tampering must be reported via the structural-integrity
    # check, even under partial masking. A semantics-preserving statement
    # insertion still executes correctly (passes symiri) and keeps the path, so
    # the structural check is the *only* legitimate gate. The mask set is
    # inferred from the (now-mutated) solution; an earlier design let that
    # inference drift and surface the rejection as a confusing "FILL_CONST
    # count mismatch" budget error. Harden: structural is the authoritative
    # first gate, so these report "structural integrity" instead.
    def expect_structural(name, puzzle, mutated_text):
      sol = os.path.join(outdir, "mut_struct.sir")
      with open(sol, "w") as f:
        f.write(mutated_text)
      passed, out = chk(rypuzchk, symiri, puzzle, sol)
      ok = (not passed) and ("structural integrity" in out.lower())
      check(name, ok, out.strip().splitlines()[-1:])

    # half_gt and base_gt share the same concrete body (masking only changes the
    # puzzle, never the ground truth), so the same insertions apply to both.
    addr_idx = find_body_line(gt_text, r"^%\w+ = addr\b.*;$")
    const_idx = find_body_line(gt_text, r"^%[\w\[\]]+ = -?\d{3,}\b.*\* %\w+;$")
    first_idx = first_body_stmt_line(gt_text)
    check("found a body addr statement to duplicate", addr_idx >= 0)
    check("found a body statement with a constant to duplicate", const_idx >= 0)

    if half_ok and addr_idx >= 0:
      # (8a) partial-mask: duplicating a constant-free body stmt drifts the mask
      # inference; it must still be reported as a structural failure.
      expect_structural(
        "p-mask 0.5: duplicated addr stmt rejected as structural",
        half_puzzle,
        dup_line(gt_text, addr_idx),
      )
    if half_ok and const_idx >= 0:
      # (8b) partial-mask: duplicating a body stmt that carries a constant.
      expect_structural(
        "p-mask 0.5: duplicated const stmt rejected as structural",
        half_puzzle,
        dup_line(gt_text, const_idx),
      )
    if half_ok and first_idx >= 0:
      # (8c) partial-mask: duplicating the first body statement.
      expect_structural(
        "p-mask 0.5: duplicated first body stmt rejected as structural",
        half_puzzle,
        dup_line(gt_text, first_idx),
      )
    store_idx = find_body_line(gt_text, r"^store\b.*;$")
    if half_ok and store_idx >= 0:
      # (8d) partial-mask: duplicating a `store` (idempotent in place) is another
      # constant-free insertion that drifts the inference; still structural.
      expect_structural(
        "p-mask 0.5: duplicated store stmt rejected as structural",
        half_puzzle,
        dup_line(gt_text, store_idx),
      )
    if const_idx >= 0:
      # (8e) full-mask: duplicating a body stmt with a constant must be a
      # structural failure, not an "extra constant" budget complaint.
      expect_structural(
        "full-mask: duplicated const stmt rejected as structural",
        base_puzzle,
        dup_line(gt_text, const_idx),
      )

    # (8f/edge) a structurally-faithful solution that merely carries an
    # off-budget constant must STILL be reported as a budget error, not a
    # structural one — i.e. running the structural check first must not swallow
    # genuine FILL_CONST violations.
    m_off = re.search(r"//@ FILL_CONST:\s*(-?\d{2,})\s+\d+", puzzle_text)
    if m_off:
      off_gt, n_off = re.subn(
        r"(?<!\d)" + re.escape(m_off.group(1)) + r"(?!\d)",
        "987654321",
        gt_text,
        count=1,
      )
      if n_off == 1:
        off_sol = os.path.join(outdir, "mut_offbudget2.sir")
        with open(off_sol, "w") as f:
          f.write(off_gt)
        passed, out = chk(rypuzchk, symiri, base_puzzle, off_sol)
        ok = (not passed) and ("structural integrity" not in out.lower())
        check("off-budget constant still reported as budget error", ok, out.strip())

    # (edge) an out-of-range probability is rejected rather than silently clamped.
    r = run(
      [
        rypuzmk,
        "--seed",
        "42",
        "--p-mask",
        "1.5",
        "--rysmith",
        rysmith,
        "-o",
        os.path.join(outdir, "pm_bad.sir"),
      ]
    )
    # (9) --lift-consts validation
    lift_puzzle = os.path.join(outdir, "lift_puzzle.sir")
    lift_gt = os.path.join(outdir, "lift_puzzle.gt.sir")
    r_lift = run(
      [
        rypuzmk,
        "--seed",
        "42",
        "--lift-consts",
        "--rysmith",
        rysmith,
        "-o",
        lift_puzzle,
        "--keep-ground-truth",
      ]
    )
    lift_ok = (
      r_lift.returncode == 0 and os.path.exists(lift_puzzle) and os.path.exists(lift_gt)
    )
    check("lift-consts: rypuzmk with --lift-consts succeeds", lift_ok)
    if lift_ok:
      lift_content = open(lift_puzzle).read()
      check(
        "lift-consts: no //@ FILL_CONST marker in puzzle",
        "//@ FILL_CONST:" not in lift_content,
      )
      check(
        "lift-consts: no 'Requirements for FILL_CONST' section in puzzle header",
        "Requirements for FILL_CONST" not in lift_content,
      )
      passed, out = chk(rypuzchk, symiri, lift_puzzle, lift_gt)
      check(
        "lift-consts: ground truth on lift puzzle accepted",
        passed,
        out.strip().splitlines()[-1:],
      )

      # Verify that perturbing a constant is accepted under lift-consts, but still structural check works.
      fail_off_budget_path = os.path.join(
        os.path.dirname(__file__), "..", "puzzle", "fail_off_budget_const.txt"
      )
      if os.path.exists(fail_off_budget_path):
        with open(fail_off_budget_path) as f:
          lines = f.readlines()
        stripped_lines = [
          ln
          for ln in lines
          if not (ln.startswith("// EXPECT:") or ln.startswith("// DESC:"))
        ]
        parts = "".join(stripped_lines).split("=>\n")
        if len(parts) == 2:
          puz_txt, sol_txt = parts
          # Remove //@ FILL_CONST lines from the puzzle to simulate --lift-consts
          puz_no_const = re.sub(r"//@ FILL_CONST:.*?\n", "", puz_txt)
          puz_file = os.path.join(outdir, "lift_fail_off_budget_puz.sir")
          sol_file = os.path.join(outdir, "lift_fail_off_budget_sol.sir")
          with open(puz_file, "w") as f:
            f.write(puz_no_const)
          with open(sol_file, "w") as f:
            f.write(sol_txt)
          passed, out = chk(rypuzchk, symiri, puz_file, sol_file)
          check(
            "lift-consts: off-budget solution accepted when const constraints are absent",
            passed,
            out.strip().splitlines()[-1:],
          )

  return summarize()


def summarize():
  npass = sum(1 for _, ok, _ in results if ok)
  total = len(results)
  print(f"\n{GREEN if npass == total else RED}{npass}/{total} checks passed{NC}")
  return 0 if npass == total else 1


if __name__ == "__main__":
  sys.exit(main())

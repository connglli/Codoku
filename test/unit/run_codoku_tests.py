"""End-to-end tests for the codoku wrapper CLI.

codoku is the puzzle generation & checking wrapper for the Python target.
It must:

  (1) generate a puzzle with `codoku create` or with a bare `codoku`
      (delegation), deterministically for a given --seed, writing
      puzzle.py into the working directory;
  (2) post-process the generated banner: `./tools/rypuzchk ...` is
      rewritten to `codoku check puzzle.py solution.py`;
  (3) validate a solution with `codoku check <puzzle> [<solution>]`,
      including the default puzzle.py / solution.py names;
  (4) reject unknown flags (it does not inherit rypuzmk-tgt's or
      rypuzchk-tgt's options).

Run as:

  python3 -m test.unit.run_codoku_tests <codoku> <rypuzmk-tgt> <rypuzchk-tgt> <rysmith>
"""

import os
import shutil
import subprocess
import sys
import tempfile

GREEN = "\033[32m"
RED = "\033[31m"
GRAY = "\033[90m"
NC = "\033[0m"

results = []


def run(cmd, cwd=None, **kw):
  print(f"  {GRAY}[RUN>]{NC} " + " ".join(cmd))
  return subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=cwd, **kw)


def check(name, ok, detail=""):
  results.append((name, ok, detail))
  color = GREEN if ok else RED
  tag = "PASS" if ok else "FAIL"
  print(f"  [{color}{tag}{NC}] {name}" + (f" — {detail}" if detail and not ok else ""))


def setup_tools(codoku, rypuzmk, rypuzchk, rysmith, outdir):
  """Mirror the image layout: all tools next to the codoku wrapper."""
  for src, name in [
    (codoku, "codoku"),
    (rypuzmk, "rypuzmk-tgt"),
    (rypuzchk, "rypuzchk-tgt"),
    (rysmith, "rysmith"),
  ]:
    os.symlink(os.path.abspath(src), os.path.join(outdir, name))


def generate_ground_truth(rypuzmk, rysmith, seed, workdir):
  """Generate the unmasked ground truth for `seed` via rypuzmk-tgt."""
  gt = os.path.join(workdir, f"gt{seed}.py")
  r = run(
    [
      os.path.join(workdir, "rypuzmk-tgt"),
      "--seed",
      str(seed),
      "--rysmith",
      os.path.join(workdir, "rysmith"),
      "--target",
      "python",
      "-o",
      gt,
      "--keep-ground-truth",
    ],
    cwd=workdir,
  )
  if r.returncode != 0:
    return None
  gt_sol = os.path.splitext(gt)[0] + ".gt.py"
  if not os.path.exists(gt_sol):
    return None
  return gt_sol


def banner_rewritten(puzzle_text: str) -> bool:
  return (
    "codoku check puzzle.py solution.py" in puzzle_text
    and "./tools/rypuzchk" not in puzzle_text
  )


def main():
  if len(sys.argv) < 5:
    print("usage: run_codoku_tests.py <codoku> <rypuzmk-tgt> <rypuzchk-tgt> <rysmith>")
    return 2
  codoku, rypuzmk, rypuzchk, rysmith = sys.argv[1:5]

  with tempfile.TemporaryDirectory(prefix="codoku_test_") as workdir:
    setup_tools(codoku, rypuzmk, rypuzchk, rysmith, workdir)
    codoku_bin = os.path.join(workdir, "codoku")

    # (1) Bare `codoku --seed N` (delegation) generates puzzle.py and
    #     rewrites the banner.
    r = run([codoku_bin, "--seed", "42"], cwd=workdir)
    puzzle = os.path.join(workdir, "puzzle.py")
    banner_ok = False
    if r.returncode == 0 and os.path.exists(puzzle):
      with open(puzzle) as f:
        banner_ok = banner_rewritten(f.read())
    check(
      "delegated generate + banner rewrite",
      r.returncode == 0 and banner_ok,
      r.stdout + r.stderr,
    )

    # (2) `codoku create --seed N` is deterministic: identical puzzle.
    create_dir = os.path.join(workdir, "create_dir")
    os.makedirs(create_dir)
    setup_tools(codoku, rypuzmk, rypuzchk, rysmith, create_dir)
    r = run(
      [os.path.join(create_dir, "codoku"), "create", "--seed", "42"], cwd=create_dir
    )
    create_puzzle = os.path.join(create_dir, "puzzle.py")
    identical = False
    if r.returncode == 0 and os.path.exists(create_puzzle):
      with open(puzzle) as f1, open(create_puzzle) as f2:
        identical = f1.read() == f2.read()
    check(
      "create subcommand is deterministic (same seed)", r.returncode == 0 and identical
    )

    # (3) `codoku check` with explicit names passes on the ground truth.
    gt_sol = generate_ground_truth(rypuzmk, rysmith, 42, workdir)
    if gt_sol is None:
      check(
        "explicit check passes on ground truth", False, "ground-truth generation failed"
      )
    else:
      r = run([codoku_bin, "check", "puzzle.py", os.path.basename(gt_sol)], cwd=workdir)
      check(
        "explicit check passes on ground truth",
        r.returncode == 0 and "[PASS]" in (r.stdout + r.stderr),
        r.stdout + r.stderr,
      )

    # (4) `codoku check` with default names (puzzle.py / solution.py).
    if gt_sol is None:
      check(
        "default check names pass on ground truth",
        False,
        "ground-truth generation failed",
      )
    else:
      shutil.copy(gt_sol, os.path.join(workdir, "solution.py"))
      r = run([codoku_bin, "check"], cwd=workdir)
      check(
        "default check names pass on ground truth",
        r.returncode == 0 and "[PASS]" in (r.stdout + r.stderr),
        r.stdout + r.stderr,
      )

    # (5) Unknown flags are rejected (no rypuzmk option inheritance).
    r = run([codoku_bin, "--bogus"], cwd=workdir)
    check("unknown flag rejected", r.returncode == 2, r.stdout + r.stderr)

    # (6) Non-python --target is rejected.
    r = run([codoku_bin, "--target", "c"], cwd=workdir)
    check("non-python target rejected", r.returncode == 2, r.stdout + r.stderr)

    # (7) Too many `check` positionals are rejected.
    r = run([codoku_bin, "check", "a.py", "b.py", "c.py"], cwd=workdir)
    check("too many check positionals rejected", r.returncode == 2, r.stdout + r.stderr)

    # (8) Help exits 0.
    r = run([codoku_bin, "--help"], cwd=workdir)
    check("top-level --help exits 0", r.returncode == 0, r.stdout + r.stderr)
    r = run([codoku_bin, "create", "--help"], cwd=workdir)
    check("create --help exits 0", r.returncode == 0, r.stdout + r.stderr)

  n_fail = sum(1 for _, ok, _ in results if not ok)
  print(f"\n{len(results) - n_fail}/{len(results)} codoku tests passed")
  return 1 if n_fail else 0


if __name__ == "__main__":
  sys.exit(main())

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


def generate_ground_truth(rypuzmk, rysmith, seed, workdir, cfg):
  """Generate the unmasked ground truth for `seed` via rypuzmk-tgt.

  `cfg` must be the same difficulty config codoku used for the puzzle.
  """
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
      "-B",
      str(cfg["n_bbls"]),
      "-S",
      str(cfg["n_stmts"]),
      "-L",
      str(cfg["min_loop_iter"]),
      "-P",
      str(cfg["p_mask"]),
    ]
    + (["-C"] if cfg["lift_consts"] else []),
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


def import_codoku(codoku_path):
  """Import puzzle/codoku/codoku.py as a module (for direct unit tests)."""
  import importlib.util

  real = os.path.realpath(codoku_path)
  spec = importlib.util.spec_from_file_location("_codoku_mod", real)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def unit_tests_pass(codoku_mod) -> bool:
  """Direct tests on pick_config / difficulty_score invariants."""
  ok = True

  # (U1) Every sampled config respects its level's score cap.
  for level in codoku_mod.DIFFICULTIES:
    spec = codoku_mod.DIFFICULTIES[level]
    for seed in range(1, 51):
      cfg = codoku_mod.pick_config(level, seed)
      score = codoku_mod.difficulty_score(cfg)
      if score > spec["cap"]:
        check(f"unit: {level} score <= cap", False, f"seed {seed} score {score}")
        ok = False
        break
    else:
      check(f"unit: {level} score stays under cap", True)

  # (U2) Budget is lifted only for easy.
  lift = {d: codoku_mod.DIFFICULTIES[d]["lift_consts"] for d in codoku_mod.DIFFICULTIES}
  if lift != {"easy": True, "medium": False, "hard": False}:
    check("unit: budget lifted only for easy", False, str(lift))
    ok = False
  else:
    check("unit: budget lifted only for easy", True)

  # (U3) Sampled dimensions stay within their ranges.
  for level in codoku_mod.DIFFICULTIES:
    spec = codoku_mod.DIFFICULTIES[level]
    bad = False
    for seed in range(1, 51):
      cfg = codoku_mod.pick_config(level, seed)
      lo, hi = spec["ranges"]["n_bbls"]
      if not (lo <= cfg["n_bbls"] <= hi):
        bad = True
      lo, hi = spec["ranges"]["p_mask"]
      if not (lo <= cfg["p_mask"] <= hi):
        bad = True
    check(f"unit: {level} dimensions within ranges", not bad)
    ok = ok and not bad

  return ok


def main():
  if len(sys.argv) < 5:
    print("usage: run_codoku_tests.py <codoku> <rypuzmk-tgt> <rypuzchk-tgt> <rysmith>")
    return 2
  codoku, rypuzmk, rypuzchk, rysmith = sys.argv[1:5]

  codoku_mod = import_codoku(codoku)
  unit_tests_pass(codoku_mod)

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

    # (2b) `codoku create -o <dir>` writes the puzzle into that directory.
    out_cwd = os.path.join(workdir, "out_cwd")
    os.makedirs(out_cwd)
    setup_tools(codoku, rypuzmk, rypuzchk, rysmith, out_cwd)
    outdir = os.path.join(workdir, "outdir")
    r = run([codoku_bin, "create", "--seed", "42", "-o", outdir], cwd=out_cwd)
    files = set(os.listdir(outdir)) if os.path.isdir(outdir) else set()
    cwd_leftovers = [
      n
      for n in ("puzzle.py", "INSTRUCTION.md", "puzzle.gt.py")
      if os.path.exists(os.path.join(out_cwd, n))
    ]
    ok_outdir = r.returncode == 0 and {"puzzle.py", "INSTRUCTION.md"}.issubset(files)
    check(
      "create -o writes puzzle + INSTRUCTION into outdir",
      ok_outdir,
      r.stdout + r.stderr,
    )
    check(
      "create -o leaves nothing in the cwd",
      not cwd_leftovers,
      f"leftover files: {cwd_leftovers}",
    )

    # (2c) The ground truth is moved into <outdir>/oracle/.
    oracle_gt = os.path.join(outdir, "oracle", "puzzle.gt.py")
    check(
      "ground truth moved into oracle/",
      os.path.exists(oracle_gt) and "puzzle.gt.py" not in files,
      f"outdir files: {files}",
    )

    # (2d) `--difficulty easy` generates a puzzle and reports the score.
    easy_dir = os.path.join(workdir, "easy_dir")
    os.makedirs(easy_dir)
    setup_tools(codoku, rypuzmk, rypuzchk, rysmith, easy_dir)
    r = run([codoku_bin, "create", "--difficulty", "easy", "--seed", "9"], cwd=easy_dir)
    check(
      "create --difficulty easy generates a puzzle",
      r.returncode == 0 and os.path.exists(os.path.join(easy_dir, "puzzle.py")),
      r.stdout + r.stderr,
    )
    check(
      "create reports the difficulty score",
      "difficulty=easy" in r.stdout and "score=" in r.stdout,
      r.stdout + r.stderr,
    )

    # (2e) Same difficulty + seed is deterministic.
    easy2 = os.path.join(workdir, "easy_dir2")
    os.makedirs(easy2)
    setup_tools(codoku, rypuzmk, rypuzchk, rysmith, easy2)
    r = run([codoku_bin, "create", "--difficulty", "easy", "--seed", "9"], cwd=easy2)
    identical_difficulty = False
    with (
      open(os.path.join(easy_dir, "puzzle.py")) as f1,
      open(os.path.join(easy2, "puzzle.py")) as f2,
    ):
      identical_difficulty = f1.read() == f2.read()
    check(
      "difficulty + seed is deterministic", r.returncode == 0 and identical_difficulty
    )

    # (2f) Invalid difficulty is rejected.
    r = run([codoku_bin, "create", "--difficulty", "bogus", "--seed", "9"], cwd=workdir)
    check("invalid difficulty rejected", r.returncode == 2, r.stdout + r.stderr)

    # (3) `codoku check` with explicit names passes on the ground truth.
    gt_cfg = codoku_mod.pick_config("medium", 42)
    gt_sol = generate_ground_truth(rypuzmk, rysmith, 42, workdir, gt_cfg)
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

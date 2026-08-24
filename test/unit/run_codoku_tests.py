"""End-to-end tests for the rysmith-based codoku (./codokus/).

The new codoku drives rysmith directly (no rypuzmk/rypuzchk binaries) and
uses the vendored puzzle_common.py / checker.py with angle-bracketed
<FILL_XXX> mask tokens.  It must:

  (1) generate a puzzle with `codoku create` (or bare `codoku`), deterministically
      for a given --seed, writing puzzle.py + INSTRUCTION.md + oracle/ into the
      output directory;
  (2) use <FILL_XXX> mask tokens and the `codoku check` validation command;
  (3) validate a solution with `codoku check`, accepting the ground truth;
  (4) reject unknown profiles/flags;
  (5) keep generated masks/budget consistent (re-mask self-check passes).

Run as:

  python3 -m test.unit.run_codoku_tests <codoku.py> <rysmith>
"""

import importlib.util
import json
import os
import re
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
  return subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=cwd, **kw)


def run_codoku(bin_path, args, cwd):
  """Run the codoku script with the current interpreter."""
  return run([sys.executable, bin_path] + args, cwd=cwd)


def check(name, ok, detail=""):
  results.append((name, ok, detail))
  color = GREEN if ok else RED
  tag = "PASS" if ok else "FAIL"
  print(f"  [{color}{tag}{NC}] {name}" + (f" — {detail}" if detail and not ok else ""))


def import_codoku(codoku_path):
  """Import codoku_creator as a module (for direct unit tests)."""
  real = os.path.realpath(codoku_path)
  module_dir = os.path.dirname(real)
  if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
  creator = os.path.join(module_dir, "codoku_creator.py")
  spec = importlib.util.spec_from_file_location("_codoku_gen_mod", creator)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def import_codoku_checker(codoku_path):
  """Import codoku_checker as a module (for direct unit tests)."""
  module_dir = os.path.dirname(os.path.realpath(codoku_path))
  if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
  checker = os.path.join(module_dir, "codoku_checker.py")
  spec = importlib.util.spec_from_file_location("_codoku_chk_mod", checker)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def unit_tests_pass(mod) -> bool:
  ok = True

  # Profiles validate and sample within ranges.
  for name in mod.PROFILES:
    prof = mod.PROFILES[name]
    prof.validate(name)
    rng = __import__("random").Random(1)
    for _ in range(20):
      cfg = mod.sample_config(prof, rng)
      lo, hi = prof.n_bbls.minimum, prof.n_bbls.maximum
      if not lo <= cfg.n_bbls <= hi:
        check(f"unit: {name} bbls within range", False)
        ok = False
        break
    else:
      check(f"unit: {name} samples within ranges", True)

  # easy is the integer-scalar profile (no fp/vec/intrinsics, no pointers);
  # hard is full-featured.
  easy_feats = set(mod.PROFILES["easy"].features)
  easy_ptr = mod.PROFILES["easy"].max_ptr_depth
  if not ({"--no-fp", "--no-vec"} <= easy_feats) or (
    easy_ptr.minimum,
    easy_ptr.maximum,
  ) != (0, 0):
    check(
      "unit: easy uses reduced feature set", False, str(easy_feats) + f" ptr={easy_ptr}"
    )
    ok = False
  else:
    check("unit: easy uses reduced feature set", True)

  return ok


def checker_unit_tests_pass(chk_mod) -> bool:
  ok = True

  # A solution that hangs must fail with FAIL_TIMEOUT (5s cap).
  hang_dir = tempfile.mkdtemp(prefix="codoku_hang_")
  try:
    hang_path = os.path.join(hang_dir, "hang.py")
    with open(hang_path, "w") as f:
      f.write("while True:\n    pass\n")
    try:
      chk_mod.run_python_solution(hang_path)
      check("unit: hanging solution times out", False, "no timeout raised")
      ok = False
    except chk_mod.CheckFailure as exc:
      check(
        "unit: hanging solution times out",
        exc.result == chk_mod.CheckResult.FAIL_TIMEOUT,
        str(exc.message),
      )
      ok = ok and exc.result == chk_mod.CheckResult.FAIL_TIMEOUT
  finally:
    shutil.rmtree(hang_dir, ignore_errors=True)

  return ok


def main():
  if len(sys.argv) < 3:
    print("usage: run_codoku_tests.py <codoku.py> <rysmith>")
    return 2
  codoku_src, rysmith = sys.argv[1:3]

  mod = import_codoku(codoku_src)
  unit_tests_pass(mod)
  checker_unit_tests_pass(import_codoku_checker(codoku_src))

  with tempfile.TemporaryDirectory(prefix="codoku_gen_") as workdir:
    # Mirror the image layout: codoku + vendored modules + rysmith in one dir.
    src_dir = os.path.dirname(os.path.realpath(codoku_src))
    for f in (
      "codoku.py",
      "codoku_creator.py",
      "codoku_checker.py",
      "codoku_common.py",
      "codoku_complexity.py",
    ):
      shutil.copy(os.path.join(src_dir, f), os.path.join(workdir, f))
    os.symlink(os.path.abspath(rysmith), os.path.join(workdir, "rysmith"))

    codoku_bin = os.path.join(workdir, "codoku.py")

    # (1) Bare `codoku --seed N` generates the puzzle set.
    r = run_codoku(codoku_bin, ["--seed", "42"], workdir)
    puzzle = os.path.join(workdir, "puzzle.py")
    oracle_gt = os.path.join(workdir, "oracle", "puzzle.gt.py")
    manifest = os.path.join(workdir, "oracle", "metadata.json")
    ok_files = (
      r.returncode == 0
      and os.path.exists(puzzle)
      and os.path.exists(oracle_gt)
      and os.path.exists(os.path.join(workdir, "INSTRUCTION.md"))
      and os.path.exists(manifest)
    )
    check("generate produces puzzle set", ok_files, r.stdout + r.stderr)

    # (2) The puzzle uses <FILL_XXX> tokens and the codoku check command.
    with open(puzzle) as f:
      ptext = f.read()
    check(
      "puzzle uses <FILL_XXX> tokens",
      "<FILL_VAR>" in ptext or "<FILL_CONST>" in ptext or "<FILL_OP>" in ptext,
    )
    check(
      "puzzle banner points at codoku check",
      "codoku check puzzle.py solution.py" in ptext and "./tools/rypuzchk" not in ptext,
    )
    with open(os.path.join(workdir, "INSTRUCTION.md")) as f:
      itext = f.read()
    bare_fill_re = re.compile(r"(?<!<)FILL_[A-Z_]+")
    check(
      "no bare FILL_XXX in puzzle or INSTRUCTION",
      not bare_fill_re.search(ptext) and not bare_fill_re.search(itext),
      bare_fill_re.search(ptext) or bare_fill_re.search(itext),
    )

    # (3) Ground truth passes the vendored checker.
    shutil.copy(oracle_gt, os.path.join(workdir, "solution.py"))
    r = run_codoku(codoku_bin, ["check"], workdir)
    check(
      "ground truth passes check",
      r.returncode == 0 and "[PASS]" in (r.stdout + r.stderr),
      r.stdout + r.stderr,
    )

    # (4) Determinism: same seed reproduces the identical puzzle.
    r2_dir = os.path.join(workdir, "r2")
    os.makedirs(r2_dir)
    for f in (
      "codoku.py",
      "codoku_creator.py",
      "codoku_checker.py",
      "codoku_common.py",
      "codoku_complexity.py",
    ):
      shutil.copy(os.path.join(workdir, f), os.path.join(r2_dir, f))
    os.symlink(os.path.abspath(rysmith), os.path.join(r2_dir, "rysmith"))
    r2 = run_codoku(
      os.path.join(r2_dir, "codoku.py"), ["create", "--seed", "42"], r2_dir
    )
    with open(puzzle) as f1, open(os.path.join(r2_dir, "puzzle.py")) as f2:
      deterministic = r2.returncode == 0 and f1.read() == f2.read()
    check("same seed is deterministic", deterministic, r2.stdout + r2.stderr)

    # (5) The manifest records realized metrics and complexity estimate.
    with open(manifest) as f:
      meta = json.load(f)
    metrics = meta["realized_metrics"]
    complexity = meta.get("complexity_estimate", {})
    ok_metrics = (
      meta["profile"] == "medium"
      and metrics["exec_path_length"] > 0
      and metrics["cfg_nodes"] > 0
      and metrics["total_masks"] > 0
      and "static_struct" in complexity
      and "size" not in complexity
      and "static_structure" not in complexity
    )
    check("manifest records realized metrics", ok_metrics, str(meta))

    # (6) Invalid profile is rejected.
    r = run_codoku(codoku_bin, ["create", "--profile", "bogus"], workdir)
    check("invalid profile rejected", r.returncode == 2, r.stdout + r.stderr)

  n_fail = sum(1 for _, ok, _ in results if not ok)
  print(f"\n{len(results) - n_fail}/{len(results)} codoku tests passed")
  return 1 if n_fail else 0


if __name__ == "__main__":
  sys.exit(main())

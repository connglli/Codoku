#!/usr/bin/env python3
"""Tests for the agent-free codoku baselines in ./codokus/solvers/.

Run directly:

  python3 test/unit/run_solver_tests.py

The suite is deliberately independent of test/unit/run_codoku_tests.py:
solver work does not need puzzle generation, so it stays fast.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GREEN = "\033[32m"
RED = "\033[31m"
NC = "\033[0m"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  results.append((name, ok, detail))
  color = GREEN if ok else RED
  tag = "PASS" if ok else "FAIL"
  print(f"[{color}{tag}{NC}] {name}" + (f" ({detail})" if detail and not ok else ""))


TWO_BLOCK_PUZZLE = """\
#//@ CFG_EDGE: entry -> b0
#//@ CFG_EDGE: b0 -> exit
#//@ EXEC_PATH: entry -> b0 -> exit
#//@ <FILL_CONST>: 5 1


def _in_check_chksum(expected, actual):
    if expected != actual:
        raise RuntimeError("mismatch")
    return actual


def codoku():
    x = <FILL_CONST>
    # ^entry
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^entry:")
    # ^b0
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^b0:")
    y = <FILL_VAR> <FILL_OP> x
    # ^exit
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^exit:")
    return y


def main():
    r = codoku()
    r = _in_check_chksum(10, r)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
"""


def write_puzzle(path: str, text: str) -> None:
  Path(path).write_text(text)


def import_solvers():
  return (
    importlib.import_module("codokus.solvers.common"),
    importlib.import_module("codokus.solvers.rnd_solver"),
    importlib.import_module("codokus.solvers.smt_solver"),
    importlib.import_module("codokus.solvers.tpl_solver"),
  )


def test_cell_inference(common_mod) -> None:
  with tempfile.TemporaryDirectory(prefix="codoku_solver_cells_") as tmpdir:
    puzzle = os.path.join(tmpdir, "puzzle.py")
    write_puzzle(puzzle, TWO_BLOCK_PUZZLE)
    model = common_mod.PuzzleModel.from_file(puzzle)
    kinds = [cell.kind for cell in model.cells]
    kinds_ok = (
      kinds.count("CONST") == 1 and kinds.count("VAR") == 1 and kinds.count("OP") == 1
    )
    var_cell = next(c for c in model.cells if c.kind == "VAR")
    op_cell = next(c for c in model.cells if c.kind == "OP")
    const_cell = next(c for c in model.cells if c.kind == "CONST")
    domain_ok = (
      "x" in var_cell.domain
      and "+" in op_cell.domain
      and const_cell.domain == ["5"]
      and const_cell.live
    )
    check("cells: type-aware inference", kinds_ok and domain_ok, str(kinds))


def test_random(common_mod, random_mod) -> None:
  with tempfile.TemporaryDirectory(prefix="codoku_solver_random_") as tmpdir:
    puzzle = os.path.join(tmpdir, "puzzle.py")
    write_puzzle(puzzle, TWO_BLOCK_PUZZLE)
    model = common_mod.PuzzleModel.from_file(puzzle)
    solver = random_mod.RandomBaseline(model, common_mod.BaselineOptions(seed=7))
    result = solver.run(time_limit=10.0, max_attempts=100)
    check(
      "random: solves a one-constant puzzle",
      result.status == "solved" and result.attempts >= 1,
      f"{result.status} attempts={result.attempts} {result.message}",
    )


def test_full_smt(common_mod, smt_mod) -> None:
  if shutil.which("z3") is None:
    check("full-smt: skipped (z3 unavailable)", True)
    return
  with tempfile.TemporaryDirectory(prefix="codoku_solver_smt_") as tmpdir:
    puzzle = os.path.join(tmpdir, "puzzle.py")
    write_puzzle(puzzle, TWO_BLOCK_PUZZLE)
    model = common_mod.PuzzleModel.from_file(puzzle)
    solver = smt_mod.FullSMTBaseline(
      model, common_mod.BaselineOptions(seed=7, solver_timeout=10.0)
    )
    result = solver.run(time_limit=60.0, max_attempts=50)
    check(
      "full-smt: solves a one-constant puzzle",
      result.status == "solved" and result.attempts >= 1,
      f"{result.status} {result.message}",
    )


def test_template_smt(common_mod, template_mod) -> None:
  if shutil.which("z3") is None:
    check("template-smt: skipped (z3 unavailable)", True)
    return
  with tempfile.TemporaryDirectory(prefix="codoku_solver_template_") as tmpdir:
    puzzle = os.path.join(tmpdir, "puzzle.py")
    write_puzzle(puzzle, TWO_BLOCK_PUZZLE)
    model = common_mod.PuzzleModel.from_file(puzzle)
    solver = template_mod.TemplateSMTBaseline(
      model, common_mod.BaselineOptions(seed=7, solver_timeout=10.0)
    )
    result = solver.run(time_limit=30.0, max_attempts=30)
    check(
      "template-smt: solves op+var+const puzzle",
      result.status == "solved",
      f"{result.status} attempts={result.attempts} {result.message}",
    )


def test_solution_memory(common_mod, random_mod) -> None:
  with tempfile.TemporaryDirectory(prefix="codoku_solver_memory_") as tmpdir:
    path = os.path.join(tmpdir, "solutions.sha256")
    memory = common_mod.SolutionMemory(path, flush_every=2, max_entries=10)
    memory.add_text("alpha")
    memory.add_text("beta")
    check(
      "memory: hashes are deduplicated and persisted",
      memory.seen_text("alpha")
      and memory.seen_text("beta")
      and not memory.seen_text("gamma"),
      f"len={len(memory)}",
    )
    memory.close()
    reopened = common_mod.SolutionMemory(path, max_entries=10)
    persisted = reopened.seen_text("alpha") and reopened.seen_text("beta")
    check("memory: reloads hashes from disk", persisted)
    reopened.close()

    puzzle = os.path.join(tmpdir, "puzzle.py")
    write_puzzle(puzzle, TWO_BLOCK_PUZZLE)
    model = common_mod.PuzzleModel.from_file(puzzle)

    class RepeatSolver(common_mod.Baseline):
      name = "repeat"

      def __init__(self, model, options):
        super().__init__(model, options)
        self.values = model.sample_values(self.rng)

      def propose(self, remaining):
        del remaining
        return self.model.assemble(self.values)

    repeat = RepeatSolver(
      model,
      common_mod.BaselineOptions(seed=5, memory_max_entries=10),
    )
    result = repeat.run(time_limit=10.0, max_attempts=5)
    check(
      "memory: the wrapper checks each hash once",
      result.status == "exhausted"
      and result.attempts == 1
      and "duplicate" in result.message,
      f"{result.status} attempts={result.attempts} {result.message}",
    )


LIVEDEAD_PUZZLE = """\
#//@ CFG_EDGE: entry -> b0
#//@ CFG_EDGE: b0 -> b1
#//@ CFG_EDGE: b0 -> exit
#//@ EXEC_PATH: entry -> b0 -> exit
#//@ <FILL_CONST>: 5 1 0
#//@ <FILL_CONST>: 7 0 1


def codoku():
    a = <FILL_CONST>
    # ^entry
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^entry:")
    # ^b0
    if 0:
        # ^b1
        if __import__("os").environ.get("DUMP_TRACE"):
            print("^b1:")
        b = <FILL_CONST>
    # ^exit
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^exit:")
    return a


def main():
    r = codoku()
    return 0
"""


def test_livedead_budget(common_mod) -> None:
  with tempfile.TemporaryDirectory(prefix="codoku_solver_livedead_") as tmpdir:
    puzzle = os.path.join(tmpdir, "puzzle.py")
    write_puzzle(puzzle, LIVEDEAD_PUZZLE)
    model = common_mod.PuzzleModel.from_file(puzzle)
    consts = [cell for cell in model.cells if cell.kind == "CONST"]
    regions_ok = (
      len(consts) == 2
      and consts[0].live
      and not consts[1].live
      and consts[0].domain == ["5"]
      and consts[1].domain == ["7"]
    )
    samples_ok = all(
      model.sample_values(__import__("random").Random(seed)) == ["5", "7"]
      for seed in range(5)
    )
    check(
      "budget: live/dead regions constrain constant sampling",
      regions_ok and samples_ok,
      f"{[(cell.live, cell.domain) for cell in consts]}",
    )


THREE_OPERAND_PUZZLE = """\
#//@ CFG_EDGE: entry -> b0
#//@ CFG_EDGE: b0 -> exit
#//@ EXEC_PATH: entry -> b0 -> exit
#//@ <FILL_CONST>: 5 1


def _in_check_chksum(expected, actual):
    if expected != actual:
        raise RuntimeError("mismatch")
    return actual


def codoku():
    x = <FILL_CONST>
    # ^entry
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^entry:")
    # ^b0
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^b0:")
    y = 1 <FILL_OP> x <FILL_OP> 0
    # ^exit
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^exit:")
    return y


def main():
    r = codoku()
    r = _in_check_chksum(1, r)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
"""


def test_ternary_chain(common_mod, smt_mod, template_mod) -> None:
  if shutil.which("z3") is None:
    check("ternary: skipped (z3 unavailable)", True)
    return
  with tempfile.TemporaryDirectory(prefix="codoku_solver_ternary_") as tmpdir:
    puzzle = os.path.join(tmpdir, "puzzle.py")
    write_puzzle(puzzle, THREE_OPERAND_PUZZLE)
    model = common_mod.PuzzleModel.from_file(puzzle)
    op_domains = [cell.domain for cell in model.cells if cell.kind == "OP"]
    domain_ok = (
      len(op_domains) == 2
      and any("if" in d for d in op_domains)
      and any("else" in d for d in op_domains)
    )
    check("ternary: both masked keywords enter the domains", domain_ok, str(op_domains))
    solver = template_mod.TemplateSMTBaseline(
      model, common_mod.BaselineOptions(seed=3, solver_timeout=10.0)
    )
    result = solver.run(time_limit=30.0, max_attempts=30)
    check(
      "ternary: template-SMT solves a three-operand chain",
      result.status == "solved",
      f"{result.status} attempts={result.attempts} {result.message}",
    )
    solver = smt_mod.FullSMTBaseline(
      model, common_mod.BaselineOptions(seed=3, solver_timeout=10.0)
    )
    result = solver.run(time_limit=30.0, max_attempts=50)
    check(
      "ternary: full-SMT solves a three-operand chain",
      result.status == "solved",
      f"{result.status} attempts={result.attempts} {result.message}",
    )


BRANCH_PUZZLE = """\
#//@ CFG_EDGE: entry -> b0
#//@ CFG_EDGE: b0 -> b1
#//@ CFG_EDGE: b0 -> b2
#//@ CFG_EDGE: b1 -> exit
#//@ CFG_EDGE: b2 -> exit
#//@ EXEC_PATH: entry -> b0 -> b1 -> exit
#//@ <FILL_CONST>: 5 1


def _in_check_chksum(expected, actual):
    if expected != actual:
        raise RuntimeError("mismatch")
    return actual


def codoku():
    x = <FILL_CONST>
    # ^entry
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^entry:")
    # ^b0
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^b0:")
    if x > 3:
        # ^b1
        if __import__("os").environ.get("DUMP_TRACE"):
            print("^b1:")
        y = 1
    else:
        # ^b2
        if __import__("os").environ.get("DUMP_TRACE"):
            print("^b2:")
        y = 2
    # ^exit
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^exit:")
    return y


def main():
    r = codoku()
    r = _in_check_chksum(1, r)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
"""


def test_branch_path(common_mod, smt_mod) -> None:
  if shutil.which("z3") is None:
    check("branch: skipped (z3 unavailable)", True)
    return
  with tempfile.TemporaryDirectory(prefix="codoku_solver_branch_") as tmpdir:
    puzzle = os.path.join(tmpdir, "puzzle.py")
    write_puzzle(puzzle, BRANCH_PUZZLE)
    model = common_mod.PuzzleModel.from_file(puzzle)
    solver = smt_mod.FullSMTBaseline(
      model, common_mod.BaselineOptions(seed=1, solver_timeout=10.0)
    )
    result = solver.run(time_limit=30.0, max_attempts=10)
    check(
      "branch: full-SMT follows the declared arm",
      result.status == "solved",
      f"{result.status} attempts={result.attempts} {result.message}",
    )


POINTER_PUZZLE = """\
#//@ CFG_EDGE: entry -> b0
#//@ CFG_EDGE: b0 -> exit
#//@ EXEC_PATH: entry -> b0 -> exit
#//@ <FILL_CONST>: 5 1


def _Ptr(buf, off, stride, lo, hi, frame=None):
    return [buf, off, stride, lo, hi, frame]


def _rd(buf, off):
    return buf[off]


def _padd(p, n):
    return _Ptr(p[0], p[1] + n * p[2], p[2], p[3], p[4], p[5])


def _prel(p, q):
    return p[1] - q[1]


def _store(p, v):
    p[0][p[1]] = v


def _in_check_chksum(expected, actual):
    if expected != actual:
        raise RuntimeError("mismatch")
    return actual


def codoku():
    a = [0, 0, 0, 0]
    p = _Ptr(a, 0, 2, 0, 4, None)
    # ^entry
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^entry:")
    # ^b0
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^b0:")
    p0 = [p]
    q = _padd(_rd(p0, 0), 1)
    _store(q, <FILL_CONST>)
    # ^exit
    if __import__("os").environ.get("DUMP_TRACE"):
        print("^exit:")
    return _rd(a, 2) + _prel(q, p)


def main():
    r = codoku()
    r = _in_check_chksum(7, r)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
"""


def test_pointer_arithmetic(common_mod, smt_mod, template_mod) -> None:
  if shutil.which("z3") is None:
    check("pointers: skipped (z3 unavailable)", True)
    return
  with tempfile.TemporaryDirectory(prefix="codoku_solver_pointer_") as tmpdir:
    puzzle = os.path.join(tmpdir, "puzzle.py")
    write_puzzle(puzzle, POINTER_PUZZLE)
    model = common_mod.PuzzleModel.from_file(puzzle)
    solver = smt_mod.FullSMTBaseline(
      model, common_mod.BaselineOptions(seed=1, solver_timeout=10.0)
    )
    result = solver.run(time_limit=30.0, max_attempts=30)
    check(
      "pointers: full-SMT traces offsets and strides",
      result.status == "solved",
      f"{result.status} attempts={result.attempts} {result.message}",
    )
    solver = template_mod.TemplateSMTBaseline(
      model, common_mod.BaselineOptions(seed=1, solver_timeout=10.0)
    )
    result = solver.run(time_limit=30.0, max_attempts=30)
    check(
      "pointers: template-SMT traces offsets and strides",
      result.status == "solved",
      f"{result.status} attempts={result.attempts} {result.message}",
    )


def test_codoku_solve_cli() -> None:
  with tempfile.TemporaryDirectory(prefix="codoku_solver_cli_") as tmpdir:
    puzzle = os.path.join(tmpdir, "puzzle.py")
    write_puzzle(puzzle, TWO_BLOCK_PUZZLE)
    run = subprocess.run(
      [
        sys.executable,
        str(ROOT / "codokus" / "codoku.py"),
        "solve",
        "--solver",
        "random",
        "--time-limit",
        "10",
        "--max-attempts",
        "100",
        "--seed",
        "7",
        puzzle,
      ],
      capture_output=True,
      text=True,
      timeout=30,
    )
    check(
      "cli: codoku solve runs the random solver",
      run.returncode == 0 and "solved" in (run.stdout + run.stderr),
      run.stdout + run.stderr,
    )


def test_wrapper_budget(common_mod, random_mod) -> None:
  with tempfile.TemporaryDirectory(prefix="codoku_solver_budget_") as tmpdir:
    puzzle = os.path.join(tmpdir, "puzzle.py")
    text = TWO_BLOCK_PUZZLE.replace(
      "_in_check_chksum(10, r)", "_in_check_chksum(11, r)"
    )
    write_puzzle(puzzle, text)
    model = common_mod.PuzzleModel.from_file(puzzle)
    solver = random_mod.RandomBaseline(model, common_mod.BaselineOptions(seed=3))
    result = solver.run(time_limit=10.0, max_attempts=2)
    check(
      "wrapper: respects the attempt budget",
      result.status == "exhausted" and result.attempts == 2,
      f"{result.status} attempts={result.attempts}",
    )


def main() -> int:
  common_mod, random_mod, smt_mod, template_mod = import_solvers()
  test_cell_inference(common_mod)
  test_random(common_mod, random_mod)
  test_full_smt(common_mod, smt_mod)
  test_template_smt(common_mod, template_mod)
  test_solution_memory(common_mod, random_mod)
  test_livedead_budget(common_mod)
  test_ternary_chain(common_mod, smt_mod, template_mod)
  test_branch_path(common_mod, smt_mod)
  test_pointer_arithmetic(common_mod, smt_mod, template_mod)
  test_codoku_solve_cli()
  test_wrapper_budget(common_mod, random_mod)
  failures = [name for name, ok, _ in results if not ok]
  print(f"\n{len(results) - len(failures)}/{len(results)} solver tests passed")
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())

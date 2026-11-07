"""Whole-puzzle SMT baseline.

Every masked cell - identifiers, operators, constants, functions, control
keywords, and goto labels - is a variable in one SMT query.  The query
encodes domain membership, the constant multiset, and a trace-guided
symbolic execution of the declared execution path.  The shared wrapper
checks each model and blocks the exact cell assignment on failure, so the
solver can search for a different filling (a CEGIS-style loop) without an
LLM in the loop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .common import Baseline, BaselineOptions, PuzzleModel, SolverTimeout
from .symbolic import SymbolicEncoder


class FullSMTBaseline(Baseline):
  name = "full-smt"

  def __init__(
    self, model: PuzzleModel, options: BaselineOptions | None = None
  ) -> None:
    super().__init__(model, options)
    self.blocked: list[dict[int, str]] = []
    self.last_choices: dict[int, str] = {}
    self.retries = 0

  def propose(self, remaining: float) -> str | None:
    timeout_ms = max(100, int(min(remaining, self.options.solver_timeout) * 1000))
    base_limit = self.options.smt_domain_limit
    domain_limit = 0 if base_limit <= 0 else max(1, base_limit >> min(self.retries, 3))
    encoder = SymbolicEncoder(
      self.model,
      symbolic_kinds={
        "VAR",
        "CONST",
        "OP",
        "FUNC",
        "CTRL",
        "FLAG_GO",
        "FLAG_CTRL",
      },
      blocked=self.blocked,
      timeout_ms=timeout_ms,
      smt_solver=self.options.smt_solver,
      deadline=time.monotonic() + timeout_ms / 1000.0,
      domain_limit=domain_limit,
    )
    try:
      values = encoder.solve()
    except SolverTimeout:
      self.retries += 1
      raise
    if values is None:
      return None
    self.last_choices = dict(values)
    ordered = [values[cell.index] for cell in self.model.cells]
    return self.model.assemble(ordered)

  def on_failure(self, check_result: str, message: str) -> None:
    del check_result, message
    if self.last_choices:
      self.blocked.append(dict(self.last_choices))


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("puzzle", help="puzzle.py to solve")
  parser.add_argument("--time-limit", type=float, default=60.0)
  parser.add_argument("--max-attempts", type=int, default=100)
  parser.add_argument("--solver-timeout", type=float, default=10.0)
  parser.add_argument("--smt-solver", default="z3", help="z3 or cvc5")
  parser.add_argument(
    "--smt-domain-limit",
    type=int,
    default=0,
    help="cap each symbolic non-constant cell domain; 0 is uncapped",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--output", default="")
  parser.add_argument("--solution-memory", default="")
  parser.add_argument("--memory-flush-every", type=int, default=1000)
  parser.add_argument("--memory-max-entries", type=int, default=100000)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  options = BaselineOptions(
    seed=args.seed,
    smt_solver=args.smt_solver,
    solver_timeout=args.solver_timeout,
    memory_path=args.solution_memory or None,
    memory_flush_every=args.memory_flush_every,
    memory_max_entries=args.memory_max_entries,
  )
  model = PuzzleModel.from_file(args.puzzle, options)
  solver = FullSMTBaseline(model, options)
  result = solver.run(time_limit=args.time_limit, max_attempts=args.max_attempts)
  print(
    f"[{result.status.upper()}] attempts={result.attempts} "
    f"elapsed={result.elapsed:.2f}s {result.message}"
  )
  if result.solution is not None and args.output:
    Path(args.output).write_text(result.solution)
  return 0 if result.status == "solved" else 1


if __name__ == "__main__":
  sys.exit(main())

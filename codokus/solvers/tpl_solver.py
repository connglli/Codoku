"""Template-enumeration + SMT baseline.

The outer loop enumerates every non-constant cell (variables, operators,
functions, control keywords, and labels).  Each enumeration yields a
template whose only remaining holes are constants; that template is
encoded as one SMT query over the constant pool.  Templates that are
syntactically or semantically outside the fragment are skipped, and the
shared wrapper checks every concrete candidate.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from .common import (
  Baseline,
  BaselineOptions,
  PuzzleModel,
  SolverTimeout,
  UnsupportedPuzzle,
  enumerate_templates,
)
from .symbolic import SymbolicEncoder


class TemplateSMTBaseline(Baseline):
  name = "template-smt"

  def __init__(
    self, model: PuzzleModel, options: BaselineOptions | None = None
  ) -> None:
    super().__init__(model, options)
    self._templates = enumerate_templates(
      model.cells,
      self.rng,
      limit=self.options.max_enumeration,
    )
    self._template_count = 0
    self._const_cells = [cell for cell in model.cells if cell.is_constant]
    self._current_template: dict[int, str] | None = None
    self._current_blocked: list[dict[int, str]] = []
    self._last_values: dict[int, str] | None = None

  def propose(self, remaining: float) -> str | None:
    if self._const_cells and not self.model.harness_examples:
      raise UnsupportedPuzzle("no harness example could be parsed")
    if any(
      not re.fullmatch(r"-?\d+", value)
      for cell in self._const_cells
      for value in cell.domain
    ):
      raise UnsupportedPuzzle("floating-point constant")
    global_deadline = time.monotonic() + remaining
    while time.monotonic() < global_deadline:
      if self._current_template is None:
        try:
          self._current_template = next(self._templates)
        except StopIteration:
          return None
        self._current_blocked = []
        self._template_count += 1
      template = self._current_template
      if not self._const_cells:
        self._last_values = None
        return self.model.assemble(
          [template.get(cell.index, cell.domain[0]) for cell in self.model.cells]
        )
      query_deadline = min(
        global_deadline,
        time.monotonic() + self.options.solver_timeout,
      )
      timeout_ms = max(100, int((query_deadline - time.monotonic()) * 1000))
      try:
        encoder = SymbolicEncoder(
          self.model,
          symbolic_kinds={"CONST"},
          fixed=template,
          blocked=self._current_blocked,
          timeout_ms=timeout_ms,
          smt_solver=self.options.smt_solver,
          deadline=query_deadline,
        )
      except UnsupportedPuzzle:
        self._current_template = None
        continue
      try:
        values = encoder.solve()
      except UnsupportedPuzzle:
        self._current_template = None
        continue
      except SolverTimeout:
        self._current_template = None
        continue
      if values is None:
        self._current_template = None
        continue
      self._last_values = dict(values)
      ordered = [values[cell.index] for cell in self.model.cells]
      return self.model.assemble(ordered)
    raise SolverTimeout("template search exhausted the time budget")

  def on_failure(self, check_result: str, message: str) -> None:
    del check_result, message
    if not self._const_cells:
      self._current_template = None
      return
    if self._last_values:
      self._current_blocked.append(dict(self._last_values))


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("puzzle", help="puzzle.py to solve")
  parser.add_argument("--time-limit", type=float, default=60.0)
  parser.add_argument("--max-attempts", type=int, default=100)
  parser.add_argument("--solver-timeout", type=float, default=5.0)
  parser.add_argument("--smt-solver", default="z3", help="z3 or cvc5")
  parser.add_argument(
    "--smt-domain-limit",
    type=int,
    default=0,
    help="cap symbolic domains; 0 is uncapped",
  )
  parser.add_argument("--max-enumeration", type=int, default=200000)
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
    smt_domain_limit=args.smt_domain_limit,
    solver_timeout=args.solver_timeout,
    max_enumeration=args.max_enumeration,
    memory_path=args.solution_memory or None,
    memory_flush_every=args.memory_flush_every,
    memory_max_entries=args.memory_max_entries,
  )
  model = PuzzleModel.from_file(args.puzzle, options)
  solver = TemplateSMTBaseline(model, options)
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

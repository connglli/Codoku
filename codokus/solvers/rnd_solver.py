"""Type-aware random-search baseline.

Each call to ``propose`` draws exactly one filling from the cell domains
(never a raw token soup): variables come from the visible locals, functions
from visible defs, operators from the construct's operator vocabulary,
labels from the declared CFG nodes, and constants from the live/dead
budget.  The shared ``Baseline`` wrapper keeps drawing candidates until the
checker accepts one or the budget is exhausted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import Baseline, BaselineOptions, PuzzleModel


class RandomBaseline(Baseline):
  name = "random"

  def __init__(
    self, model: PuzzleModel, options: BaselineOptions | None = None
  ) -> None:
    super().__init__(model, options)

  def propose(self, remaining: float) -> str | None:
    del remaining
    return self.model.assemble(self.model.sample_values(self.rng))


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("puzzle", help="puzzle.py to solve")
  parser.add_argument("--time-limit", type=float, default=60.0)
  parser.add_argument("--max-attempts", type=int, default=1000)
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
    memory_path=args.solution_memory or None,
    memory_flush_every=args.memory_flush_every,
    memory_max_entries=args.memory_max_entries,
  )
  model = PuzzleModel.from_file(args.puzzle, options)
  solver = RandomBaseline(model, options)
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

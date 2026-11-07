"""Agent-free baselines for Codoku puzzles.

The package exposes three solvers that share one runner:

* ``rnd_solver.RandomBaseline`` samples a type-aware filling.
* ``smt_solver.FullSMTBaseline`` encodes every cell as one SMT query.
* ``tpl_solver.TemplateSMTBaseline`` enumerates non-constant cells
  and solves the resulting constants-only template with SMT.

``common.Baseline`` owns the time/attempt budget and the checker loop, so
all three solvers differ only in ``propose()``.
"""

from .common import Baseline, BaselineOptions, PuzzleModel, RunResult

__all__ = ["Baseline", "BaselineOptions", "PuzzleModel", "RunResult"]

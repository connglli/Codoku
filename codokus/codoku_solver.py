#!/usr/bin/env python3
"""Run the agent-free Codoku baselines over one or more puzzles.

Examples:

  codoku solve --solver random puzzle.py
  codoku solve --solver all cases/ --time-limit 600 --output-dir runs/

Each run is bounded by wall-clock and attempt budgets.  The optional
solution memory stores SHA256 digests on disk so candidates generated in a
previous run are not checked again; it flushes every ``--memory-flush-every``
additions (1000 by default) and keeps at most ``--memory-max-entries``
digests in RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solvers.common import BaselineOptions, PuzzleModel  # noqa: E402
from solvers.rnd_solver import RandomBaseline  # noqa: E402
from solvers.smt_solver import FullSMTBaseline  # noqa: E402
from solvers.tpl_solver import TemplateSMTBaseline  # noqa: E402

SOLVERS = {
  "random": RandomBaseline,
  "full-smt": FullSMTBaseline,
  "template-smt": TemplateSMTBaseline,
}


def find_puzzles(paths: list[str]) -> list[Path]:
  found: list[Path] = []
  for raw in paths:
    path = Path(raw)
    if path.is_dir():
      found.extend(sorted(path.rglob("puzzle.py")))
    elif path.is_file():
      found.append(path)
    else:
      raise FileNotFoundError(raw)
  unique: list[Path] = []
  seen: set[Path] = set()
  for path in found:
    resolved = path.resolve()
    if resolved not in seen:
      seen.add(resolved)
      unique.append(path)
  return unique


def memory_path(directory: Path | None, puzzle: Path, solver: str) -> str | None:
  if directory is None:
    return None
  digest = hashlib.sha256(puzzle.read_bytes()).hexdigest()[:16]
  return str(directory / f"{puzzle.stem}-{digest}-{solver}.sha256")


def run_one(
  puzzle: Path,
  solver_name: str,
  *,
  time_limit: float,
  max_attempts: int,
  solver_timeout: float,
  smt_solver: str,
  smt_domain_limit: int,
  max_enumeration: int,
  seed: int,
  memory_dir: Path | None,
  output_dir: Path | None,
  memory_flush_every: int,
  memory_max_entries: int,
) -> dict:
  options = BaselineOptions(
    seed=seed,
    smt_solver=smt_solver,
    smt_domain_limit=smt_domain_limit,
    solver_timeout=solver_timeout,
    max_enumeration=max_enumeration,
    memory_path=memory_path(memory_dir, puzzle, solver_name),
    memory_flush_every=memory_flush_every,
    memory_max_entries=memory_max_entries,
  )
  model = PuzzleModel.from_file(puzzle, options)
  solver_class = SOLVERS[solver_name]
  solver = solver_class(model, options)
  started = time.monotonic()
  result = solver.run(time_limit=time_limit, max_attempts=max_attempts)
  elapsed = time.monotonic() - started
  record = {
    "solver": solver_name,
    "puzzle": str(puzzle),
    "status": result.status,
    "attempts": result.attempts,
    "elapsed_seconds": round(elapsed, 3),
    "check_result": result.check_result,
    "message": result.message,
    "cells": len(model.cells),
    "examples": model.n_examples,
    "domain_log10": round(model.domain_log10(), 3),
    "time_limit": time_limit,
    "max_attempts": max_attempts,
    "solver_timeout": solver_timeout,
    "smt_solver": smt_solver,
    "smt_domain_limit": smt_domain_limit,
    "max_enumeration": max_enumeration,
    "seed": seed,
  }
  if output_dir is not None:
    tag = hashlib.sha256(str(puzzle.resolve()).encode()).hexdigest()[:8]
    run_dir = output_dir / f"{puzzle.stem}-{tag}-{solver_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if result.solution is not None:
      solution_path = run_dir / "solution.py"
      solution_path.write_text(result.solution)
      record["solution"] = str(solution_path)
    (run_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
  return record


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="codoku solve",
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    "--solver",
    choices=[*sorted(SOLVERS), "all"],
    default="random",
    help="baseline to run, or 'all' (default: random)",
  )
  parser.add_argument("puzzles", nargs="+", help="puzzle.py files or directories")
  parser.add_argument(
    "--time-limit",
    type=float,
    default=60.0,
    help="wall-clock budget per puzzle and solver, in seconds",
  )
  parser.add_argument(
    "--max-attempts",
    type=int,
    default=1000,
    help="candidate budget per puzzle and solver",
  )
  parser.add_argument(
    "--solver-timeout",
    type=float,
    default=5.0,
    help="per-SMT-query deadline, in seconds",
  )
  parser.add_argument(
    "--smt-solver", default="z3", help="external SMT binary: z3 or cvc5"
  )
  parser.add_argument(
    "--smt-domain-limit",
    type=int,
    default=0,
    help="cap each symbolic non-constant cell domain; 0 is uncapped",
  )
  parser.add_argument(
    "--max-enumeration",
    type=int,
    default=200000,
    help="template-enumeration cap before the SMT stage",
  )
  parser.add_argument("--seed", type=int, default=0, help="sampling seed")
  parser.add_argument(
    "--output-dir",
    default="",
    help="write each solution.py and result.json under this directory",
  )
  parser.add_argument(
    "--solution-memory-dir",
    default="",
    help="persist SHA256 candidate memory files under this directory",
  )
  parser.add_argument(
    "--memory-flush-every",
    type=int,
    default=1000,
    help="flush this many new hashes to disk at a time",
  )
  parser.add_argument(
    "--memory-max-entries",
    type=int,
    default=100000,
    help="maximum hashes kept in RAM",
  )
  parser.add_argument(
    "--json",
    action="store_true",
    help="print one JSON record per puzzle instead of a summary line",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  puzzles = find_puzzles(args.puzzles)
  if not puzzles:
    print("no puzzle.py files found", file=sys.stderr)
    return 2
  solvers = (
    ["random", "full-smt", "template-smt"] if args.solver == "all" else [args.solver]
  )
  output_dir = Path(args.output_dir) if args.output_dir else None
  memory_dir = Path(args.solution_memory_dir) if args.solution_memory_dir else None
  records = []
  for puzzle in puzzles:
    for solver_name in solvers:
      try:
        record = run_one(
          puzzle,
          solver_name,
          time_limit=args.time_limit,
          max_attempts=args.max_attempts,
          solver_timeout=args.solver_timeout,
          smt_solver=args.smt_solver,
          smt_domain_limit=args.smt_domain_limit,
          max_enumeration=args.max_enumeration,
          seed=args.seed,
          memory_dir=memory_dir,
          output_dir=output_dir,
          memory_flush_every=args.memory_flush_every,
          memory_max_entries=args.memory_max_entries,
        )
      except Exception as error:  # noqa: BLE001
        record = {
          "solver": solver_name,
          "puzzle": str(puzzle),
          "status": "error",
          "attempts": 0,
          "elapsed_seconds": 0.0,
          "check_result": None,
          "message": f"{type(error).__name__}: {error}",
          "cells": 0,
          "examples": 0,
          "domain_log10": 0.0,
        }
      records.append(record)
      if args.json:
        print(json.dumps(record))
      else:
        print(
          f"{record['solver']:<12} {record['status']:<10} "
          f"attempts={record['attempts']:<5} "
          f"{record['elapsed_seconds']:>8.2f}s  {puzzle}"
        )
  if not args.json:
    solved = sum(1 for record in records if record["status"] == "solved")
    print(f"\n{solved}/{len(records)} runs solved")
  return 0 if all(record["status"] == "solved" for record in records) else 1


if __name__ == "__main__":
  sys.exit(main())

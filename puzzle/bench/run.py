#!/usr/bin/env python3
"""
run.py — Push-button benchmark runner for RefractIR puzzle solving.

Usage:
    python puzzle/bench/run.py -n 100 -m claude-opus-4-8 [-o output] [-a claude] [-j 1] [rypuzmk opts]

Workflow:
    1. Generate N puzzles (skipped if the output directory already has them).
    2. Detect / build the Docker image (rypuz:latest).
    3. Start J parallel containers, each solving one puzzle.
    4. Analyze results and emit a summary table + result.json.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Add current directory to path so we can import analyzers decoupled
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyzers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = Path(__file__).resolve().parent

# Image for puzzle generation
REFRACTIR_IMAGE = "refractir:latest"
REFRACTIR_DOCKERFILE = REPO_ROOT / "Dockerfile"

# Image for running agents
BENCH_IMAGE = "rypuz:latest"
BENCH_DOCKERFILE = BENCH_DIR / "Dockerfile"
BENCH_ENTRYPOINT = BENCH_DIR / "entrypoint.sh"
BENCH_SYSTEM_PROMPT = BENCH_DIR / "system.md"
BENCH_ENV_FILE = BENCH_DIR / "rypuzbench.env"
BENCH_DOCKER_MEMORY = "8g"
BENCH_DOCKER_CPUS = "2"

AGENTS = {
  "claude": BENCH_DIR / "claude.sh",
  "opencode": BENCH_DIR / "opencode.sh",
  # Future: "codex": BENCH_DIR / "codex.sh",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str, *, level: str = "INFO") -> None:
  ts = datetime.datetime.now().strftime("%H:%M:%S")
  print(f"[{ts}] [{level}] {msg}", flush=True)


def fatal(msg: str) -> None:
  log(msg, level="FATAL")
  sys.exit(1)


def sha1_file(path: Path) -> str:
  h = hashlib.sha1()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
      h.update(chunk)
  return h.hexdigest()


def run_cmd(
  cmd: list[str],
  *,
  cwd: Path | None = None,
  timeout: int | None = None,
  capture: bool = True,
) -> subprocess.CompletedProcess[str]:
  """Run a command, returning CompletedProcess.  Raises on non-zero exit."""
  return subprocess.run(
    cmd,
    cwd=cwd,
    capture_output=capture,
    text=True,
    timeout=timeout,
  )


# ---------------------------------------------------------------------------
# Docker images and containers
# ---------------------------------------------------------------------------


def docker_image_exists(image_name: str) -> bool:
  result = run_cmd(["docker", "image", "inspect", image_name])
  return result.returncode == 0


def build_docker_image(image_name: str, dockerfile_path: Path) -> None:
  if docker_image_exists(image_name):
    log(f"Docker image '{image_name}' already exists, skipping build.")
    return

  log(f"Building Docker image '{image_name}' (this may take a while)...")
  uid = os.getuid()
  gid = os.getgid()
  cmd = [
    "docker",
    "build",
    "-t",
    image_name,
    "--build-arg",
    f"UID={uid}",
    "--build-arg",
    f"GID={gid}",
    "-f",
    str(dockerfile_path),
    str(REPO_ROOT),
  ]
  log(f"  $ {' '.join(cmd)}")
  result = run_cmd(cmd, capture=False)
  if result.returncode != 0:
    fatal(f"Docker build for {image_name} failed. See output above.")
  log(f"Docker image {image_name} built successfully.")


# ---------------------------------------------------------------------------
# Phase 1: Generate puzzles
# ---------------------------------------------------------------------------


def run_in_generation_container(
  args: list[str],
  *,
  output_dir: Path | None = None,
  capture: bool = True,
  timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
  """Run a command inside the refractir:latest container with standard mounts."""
  cmd = [
    "docker",
    "run",
    "--rm",
    "-v",
    f"{REPO_ROOT.resolve()}:/workspace",
  ]
  if output_dir is not None:
    cmd += ["-v", f"{output_dir.resolve()}:/output"]
  cmd += [REFRACTIR_IMAGE] + args

  log(f"  $ {' '.join(cmd)}")

  return run_cmd(cmd, capture=capture, timeout=timeout)


def generate_puzzles(
  output_dir: Path,
  n: int,
  rypuzmk_opts: list[str],
) -> None:
  """Generate n puzzles under output_dir/puz-NNNN/ + output_dir/oracles/."""
  oracles = output_dir / "oracles"
  oracles.mkdir(parents=True, exist_ok=True)

  # Check which puzzles already exist and skip them.
  start_from = 0
  for i in range(1, n + 1):
    puz_dir = output_dir / f"puz-{i:04d}"
    puzzle_file = puz_dir / "puzzle.sir"
    gt_file = oracles / f"sol-{i:04d}.sir"
    hash_file = oracles / f"puz-{i:04d}.hash"
    if puzzle_file.exists() and gt_file.exists() and hash_file.exists():
      start_from = i
    else:
      break
  else:
    start_from = n  # All puzzles exist

  if start_from >= n:
    log(f"All {n} puzzles already generated, skipping.")
    return

  # Build the refractir image if not skipped
  build_docker_image(REFRACTIR_IMAGE, REFRACTIR_DOCKERFILE)

  # Compile tools inside the container
  log("Compiling tools (rypuzmk, rypuzchk, symiri) inside the Docker container...")
  result = run_in_generation_container(
    ["make", "rypuzmk", "rypuzchk", "symiri"],
    capture=False,
  )
  if result.returncode != 0:
    fatal("Failed to compile tools inside the Docker container.")
  log("Tools compiled successfully inside the Docker container.")

  # Start generating puzzles from the last existing index + 1
  log(f"Generating puzzles {start_from + 1}..{n} ({n - start_from} remaining)")

  for i in range(start_from + 1, n + 1):
    puz_dir = output_dir / f"puz-{i:04d}"
    puz_dir.mkdir(parents=True, exist_ok=True)
    puzzle_file = puz_dir / "puzzle.sir"
    gt_file = oracles / f"sol-{i:04d}.sir"
    hash_file = oracles / f"puz-{i:04d}.hash"

    # Build the rypuzmk command arguments for the container.
    # Use --seed <i> for reproducibility unless the user already passed --seed.
    container_args = [
      "./rypuzmk",
      "-o",
      "/output/puzzle.sir",
      "--keep-ground-truth",
    ] + rypuzmk_opts

    result = run_in_generation_container(container_args, output_dir=puz_dir)
    if result.returncode != 0:
      log(
        f"rypuzmk failed for puzzle {i}: {result.stderr.strip()}",
        level="ERROR",
      )
      # Try next seed on failure (rypuzmk retries internally too)
      continue

    # rypuzmk writes ground-truth to <output>.gt.sir
    gt_raw = puzzle_file.with_suffix(".gt.sir")
    if gt_raw.exists():
      shutil.move(str(gt_raw), str(gt_file))
    else:
      log(f"Warning: ground-truth not found for puzzle {i}", level="WARN")

    # Hash the puzzle file
    if puzzle_file.exists():
      hash_file.write_text(sha1_file(puzzle_file) + "\n")

    if i % 10 == 0 or i == n:
      log(f"  Generated {i}/{n} puzzles")

  log("Puzzle generation complete.")


# ---------------------------------------------------------------------------
# Phase 2: Run agents in containers
# ---------------------------------------------------------------------------


class BenchmarkRunner:
  """Manages parallel agent runs and tracks results."""

  def __init__(
    self,
    output_dir: Path,
    n: int,
    agent: str,
    model: str,
    parallelism: int,
    max_turns: int,
    timeout: int,
    max_budget_usd: int,
  ):
    self.output_dir = output_dir
    self.n = n
    self.agent = agent
    self.model = model
    self.parallelism = parallelism
    self.max_turns = max_turns
    self.timeout = timeout
    self.max_budget_usd = max_budget_usd
    self._containers: list[str] = []
    self._stop = False

  def run_all(self) -> list[dict[str, Any]]:
    """Run the agent on all puzzles, return per-puzzle analysis dicts."""
    # Build the Docker image for the agent if it doesn't exist
    build_docker_image(BENCH_IMAGE, BENCH_DOCKERFILE)

    # Determine which puzzles haven't been solved yet
    pending: list[int] = []
    already: list[int] = []
    for i in range(1, self.n + 1):
      analysis_file = self.output_dir / f"puz-{i:04d}" / "analyses.json"
      if analysis_file.exists():
        already.append(i)
      else:
        pending.append(i)

    if already:
      log(f"Skipping {len(already)} already-analyzed puzzles")
    if not pending:
      log("All puzzles already analyzed, skipping agent runs.")
      return self._collect_all_analyses()

    log(
      f"Running {self.agent} (model={self.model}) on {len(pending)} puzzles "
      f"with parallelism={self.parallelism}"
    )

    results: list[dict[str, Any]] = []

    # Install signal handler for graceful shutdown
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _shutdown(signum: int, frame: Any) -> None:
      log("Received shutdown signal, stopping containers...", level="WARN")
      self._stop = True
      self._kill_all_containers()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
      with ThreadPoolExecutor(max_workers=self.parallelism) as pool:
        futures = {pool.submit(self._run_single, i): i for i in pending}
        for future in as_completed(futures):
          idx = futures[future]
          if self._stop:
            break
          try:
            analysis = future.result()
            results.append(analysis)
            status = analysis.get("verdict", "UNKNOWN")
            log(f"  puz-{idx:04d}: {status}")
          except Exception as e:
            log(f"  puz-{idx:04d}: EXCEPTION: {e}", level="ERROR")
            results.append({"puzzle_id": idx, "verdict": "ERROR", "error": str(e)})
    finally:
      signal.signal(signal.SIGINT, original_sigint)
      signal.signal(signal.SIGTERM, original_sigterm)

    return self._collect_all_analyses()

  def _run_single(self, puzzle_idx: int) -> dict[str, Any]:
    """Run the agent on a single puzzle in a Docker container."""
    if self._stop:
      return {"puzzle_id": puzzle_idx, "verdict": "CANCELLED"}

    puz_dir = self.output_dir / f"puz-{puzzle_idx:04d}"
    (puz_dir / "workspace").mkdir(exist_ok=True)

    agent_script = AGENTS[self.agent]
    container_puzzle_dir = "/workspace/puzzle"

    # Agent script and system prompt are bind-mounted read-only
    # The puzzle directory is bind-mounted read-write
    container_name = f"rypuz-{puzzle_idx:04d}-{int(time.time())}"

    cmd = [
      "docker",
      "run",
      "--rm",
      "--name",
      container_name,
      # Mount puzzle dir
      "-v",
      f"{puz_dir.resolve()}:{container_puzzle_dir}",
      # Mount common script (read-only)
      "-v",
      f"{(BENCH_DIR / 'common.sh').resolve()}:/opt/rypuz/common.sh:ro",
      # Mount agent script (read-only)
      "-v",
      f"{agent_script.resolve()}:/opt/rypuz/agent.sh:ro",
      # Mount global env file (read-only)
      "-v",
      f"{BENCH_ENV_FILE.resolve()}:/opt/rypuz/rypuzbench.env:ro",
      # Resource limits
      "--memory",
      BENCH_DOCKER_MEMORY,
      "--cpus",
      BENCH_DOCKER_CPUS,
      # No network access for the solver itself (security)
      # NOTE: disabled because agents needs API access
      # "--network",
      # "none",
      BENCH_IMAGE,
      # entrypoint args: agent-script puzzle-dir [agent-args...]
      "/opt/rypuz/agent.sh",
      container_puzzle_dir,
      "--model",
      self.model,
      "--timeout",
      str(self.timeout),
      "--max-turns",
      str(self.max_turns),
      "--max-budget-usd",
      str(self.max_budget_usd),
    ]

    self._containers.append(container_name)

    start_time = time.time()
    elapsed = 0.0
    log(f"  Starting container for puz-{puzzle_idx:04d}: {container_name}")

    try:
      result = run_cmd(
        cmd,
        timeout=self.timeout + 60 if self.timeout > 0 else None,
      )
      elapsed = time.time() - start_time

      # Save container stdout/stderr as log
      log_file = puz_dir / "container.log"
      with open(log_file, "w") as f:
        f.write(f"=== STDOUT ===\n{result.stdout}\n")
        f.write(f"=== STDERR ===\n{result.stderr}\n")
        f.write(f"=== EXIT CODE: {result.returncode} ===\n")

    except subprocess.TimeoutExpired:
      elapsed = time.time() - start_time
      log(
        f"  Container for puz-{puzzle_idx:04d} timed out after {elapsed:.0f}s",
        level="WARN",
      )
      # Kill the container
      run_cmd(["docker", "kill", container_name])

    finally:
      if container_name in self._containers:
        self._containers.remove(container_name)

    # Analyze the result
    analysis = self._analyze_puzzle(puzzle_idx, elapsed)

    # Save analysis
    analysis_file = puz_dir / "analyses.json"
    with open(analysis_file, "w") as f:
      json.dump(analysis, f, indent=2)

    return analysis

  def _analyze_puzzle(self, puzzle_idx: int, elapsed: float) -> dict[str, Any]:
    """Analyze a single puzzle result after the agent finishes."""
    puz_dir = self.output_dir / f"puz-{puzzle_idx:04d}"
    oracles = self.output_dir / "oracles"
    puzzle_file = puz_dir / "puzzle.sir"
    solution_file = puz_dir / "solution.sir"
    trajectory_file = puz_dir / "trajectory.json"
    hash_file = oracles / f"puz-{puzzle_idx:04d}.hash"

    log(f"  Analyzing puzzle {puzzle_idx:04d} (elapsed={elapsed:.1f}s)")

    analysis: dict[str, Any] = {
      "puzzle_id": puzzle_idx,
      "elapsed_seconds": round(elapsed, 2),
    }

    # 1. Check if solution.sir exists
    has_solution = solution_file.exists()
    analysis["has_solution"] = has_solution
    log(f"  - has_solution={has_solution}")

    if not has_solution:
      analysis["verdict"] = "NO_SOLUTION"
      analysis.update(analyzers.parse_trajectory(self.agent, trajectory_file))
      return analysis

    # 2. Check if puzzle.sir was modified (cheat detection)
    expected_hash = hash_file.read_text().strip()
    if puzzle_file.exists():
      actual_hash = sha1_file(puzzle_file)
      puzzle_modified = actual_hash != expected_hash
    else:
      actual_hash = "<puzzle.sir deleted>"
      puzzle_modified = (
        True  # If puzzle.sir is missing, treat it as modified (cheating)
      )

    analysis["puzzle_modified"] = puzzle_modified
    log(
      f"  - puzzle_modified={puzzle_modified} (expected={expected_hash}, actual={actual_hash})"
    )

    if puzzle_modified:
      analysis["verdict"] = "CHEAT"
      analysis.update(analyzers.parse_trajectory(self.agent, trajectory_file))
      return analysis

    # 3. Run rypuzchk to validate the solution
    rypuzchk = REPO_ROOT / "rypuzchk"
    if not rypuzchk.exists():
      analysis["verdict"] = "ERROR"
      analysis["error"] = "rypuzchk not found"
      return analysis

    try:
      container_args = [
        "./rypuzchk",
        f"/output/puz-{puzzle_idx:04d}/puzzle.sir",
        f"/output/puz-{puzzle_idx:04d}/solution.sir",
        "--symiri",
        "./symiri",
      ]
      chk_result = run_in_generation_container(
        container_args,
        output_dir=self.output_dir,
        timeout=60,
      )
      chk_stdout = chk_result.stdout + chk_result.stderr
      passed = "[PASS]" in chk_stdout
    except subprocess.TimeoutExpired:
      passed = False
      chk_stdout = "rypuzchk timed out"
    except Exception as e:
      passed = False
      chk_stdout = str(e)

    analysis["checker_output"] = chk_stdout.strip()

    # 4. Verdict
    if passed:
      analysis["verdict"] = "PASS"
    else:
      analysis["verdict"] = "FAIL"
    log(f"  - checker_passed={passed} ({chk_stdout.strip()})")

    # 5. Parse trajectory for token/cost/round stats (decoupled via analyzers)
    analysis.update(analyzers.parse_trajectory(self.agent, trajectory_file))

    return analysis

  def _collect_all_analyses(self) -> list[dict[str, Any]]:
    """Collect all analyses.json from puzzle directories."""
    results = []
    for i in range(1, self.n + 1):
      analysis_file = self.output_dir / f"puz-{i:04d}" / "analyses.json"
      if analysis_file.exists():
        with open(analysis_file) as f:
          results.append(json.load(f))
      else:
        results.append({"puzzle_id": i, "verdict": "PENDING"})
    return results

  def _kill_all_containers(self) -> None:
    """Kill all tracked running containers."""
    for name in list(self._containers):
      log(f"  Killing container: {name}", level="WARN")
      run_cmd(["docker", "kill", name])


# ---------------------------------------------------------------------------
# Phase 3: Summary
# ---------------------------------------------------------------------------


def compute_summary(
  results: list[dict[str, Any]],
  output_dir: Path,
  metadata: dict[str, Any],
) -> dict[str, Any]:
  """Compute aggregate statistics and save result.json."""
  total = len(results)

  counts: dict[str, int] = {}
  for r in results:
    v = r.get("verdict", "UNKNOWN")
    counts[v] = counts.get(v, 0) + 1

  passed = counts.get("PASS", 0)
  failed = counts.get("FAIL", 0)
  cheated = counts.get("CHEAT", 0)
  no_solution = counts.get("NO_SOLUTION", 0)
  errors = counts.get("ERROR", 0)

  # Aggregate numeric stats
  rounds_list = [
    r.get("num_rounds", 0) for r in results if r.get("verdict") != "PENDING"
  ]
  tokens_list = [
    r.get("total_tokens", 0) for r in results if r.get("verdict") != "PENDING"
  ]
  cost_list = [r.get("cost_usd", 0.0) for r in results if r.get("verdict") != "PENDING"]
  elapsed_list = [
    r.get("elapsed_seconds", 0.0) for r in results if r.get("verdict") != "PENDING"
  ]

  def safe_avg(lst: list[float | int]) -> float:
    return sum(lst) / len(lst) if lst else 0.0

  summary = {
    "total_puzzles": total,
    "pass": passed,
    "fail": failed,
    "cheat": cheated,
    "no_solution": no_solution,
    "error": errors,
    "pass_rate": round(passed / total * 100, 1) if total else 0.0,
    "avg_rounds": round(safe_avg(rounds_list), 1),
    "avg_total_tokens": round(safe_avg(tokens_list), 0),
    "avg_cost_usd": round(safe_avg(cost_list), 4),
    "avg_elapsed_seconds": round(safe_avg(elapsed_list), 1),
    "total_cost_usd": round(sum(cost_list), 4),
    "total_elapsed_seconds": round(sum(elapsed_list), 1),
    "verdicts": counts,
    "per_puzzle": results,
    "metadata": metadata,
  }

  # Save
  result_file = output_dir / "result.json"
  with open(result_file, "w") as f:
    json.dump(summary, f, indent=2)

  return summary


def print_summary_table(summary: dict[str, Any]) -> None:
  """Print a human-readable summary table."""
  meta = summary.get("metadata", {})
  opts = meta.get("options", {})
  n = summary["total_puzzles"]

  print()
  print("=" * 68)
  print(
    f"  ReifyPuzzle Benchmark — {opts.get('agent', '?')} / {opts.get('model', '?')}"
  )
  print("=" * 68)
  print()
  print(f"  {'Metric':<30} {'Value':>15}")
  print(f"  {'-' * 30} {'-' * 15}")
  print(f"  {'Total puzzles':<30} {n:>15}")
  print(f"  {'Passed':<30} {summary['pass']:>15}")
  print(f"  {'Failed':<30} {summary['fail']:>15}")
  print(f"  {' Cheated':<30} {summary['cheat']:>15}")
  print(f"  {' No solution':<30} {summary['no_solution']:>15}")
  print(f"  {' Errors':<30} {summary['error']:>15}")
  print(f"  {'Pass rate':<30} {summary['pass_rate']:>14.1f}%")
  print(f"  {'-' * 30} {'-' * 15}")
  print(f"  {'Avg rounds':<30} {summary['avg_rounds']:>15.1f}")
  print(f"  {'Avg tokens':<30} {summary['avg_total_tokens']:>15,.0f}")
  avg_cost_str = f"${summary['avg_cost_usd']:.4f}"
  print(f"  {'Avg cost (USD)':<30} {avg_cost_str:>15}")
  print(f"  {'Avg time (s)':<30} {summary['avg_elapsed_seconds']:>15.1f}")
  print(f"  {'-' * 30} {'-' * 15}")
  total_cost_str = f"${summary['total_cost_usd']:.4f}"
  print(f"  {'Total cost (USD)':<30} {total_cost_str:>15}")
  print(f"  {'Total time (s)':<30} {summary['total_elapsed_seconds']:>15.1f}")
  print()
  print(f"  Results saved to: {summary['metadata'].get('output_dir', '?')}/result.json")
  print("=" * 68)
  print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> tuple[argparse.Namespace, list[str]]:
  parser = argparse.ArgumentParser(
    prog="run.py",
    description="RefractIR puzzle benchmark runner",
    epilog=textwrap.dedent("""\
            Difficulty options forwarded to rypuzmk:
              -L, --min-loop-iter N   Minimum loop iterations (default: 2)
              -B, --n-bbls N          Number of basic blocks (default: 5)
              -S, --n-stmts N         Statements per block (default: 3)
              --p-mask P              Masking probability [0,1] (default: 1.0)
              --lift-consts           Remove FILL_CONST budget constraints
              --seed S                Base seed for puzzle generation

            These are passed through to rypuzmk verbatim.
        """),
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    "-n",
    type=int,
    required=True,
    help="Number of puzzles to generate",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Output directory (default: $PWD/benchmark)",
  )
  parser.add_argument(
    "-a",
    "--agent",
    type=str,
    default="claude",
    choices=list(AGENTS.keys()),
    help="Agent to use (default: claude)",
  )
  parser.add_argument(
    "-m",
    "--model",
    type=str,
    required=True,
    help="Model name to pass to the agent",
  )
  parser.add_argument(
    "-j",
    "--parallelism",
    type=int,
    default=1,
    help="Number of parallel agent instances (default: 1)",
  )
  parser.add_argument(
    "--max-turns",
    type=int,
    default=0,
    help="Maximum agent conversation turns (default: 0 = unlimited)",
  )
  parser.add_argument(
    "--timeout",
    type=int,
    default=0,
    help="Per-puzzle timeout in seconds (default: 0 = unlimited)",
  )
  parser.add_argument(
    "--max-budget-usd",
    type=int,
    default=0,
    help="Maximum USD budget for the agent run (default: 0 = unlimited)",
  )
  parser.add_argument(
    "--generate-only",
    action="store_true",
    help="Only generate puzzles, do not run agents",
  )
  parser.add_argument(
    "--analyze-only",
    action="store_true",
    help="Only analyze existing results, do not generate or run",
  )

  args, remaining = parser.parse_known_args()
  return args, remaining


def main() -> None:
  args, rypuzmk_opts = parse_args()

  # Resolve output directory
  output_dir = Path(args.output) if args.output else Path.cwd() / "benchmark"
  output_dir = output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  start_time = time.time()

  # Save metadata
  metadata = {
    "command": " ".join(sys.argv),
    "options": vars(args),
    "output_dir": str(output_dir),
    "start_time": datetime.datetime.now().isoformat(),
  }

  # --- Phase 1: Generate puzzles ---
  if not args.analyze_only:
    log("Phase 1: Generating puzzles")
    generate_puzzles(output_dir, args.n, rypuzmk_opts)
  else:
    log("Phase 1: Skipped (--analyze-only)")

  if args.generate_only:
    log("Done (--generate-only).  Puzzles are in: " + str(output_dir))
    # Save metadata
    metadata["end_time"] = datetime.datetime.now().isoformat()
    metadata["total_seconds"] = round(time.time() - start_time, 2)
    with open(output_dir / "metadata.json", "w") as f:
      json.dump(metadata, f, indent=2)
    return

  # --- Phase 2: Run agents ---
  if not args.analyze_only:
    log("Phase 2: Running agents")
    runner = BenchmarkRunner(
      output_dir=output_dir,
      n=args.n,
      agent=args.agent,
      model=args.model,
      parallelism=args.parallelism,
      max_turns=args.max_turns,
      timeout=args.timeout,
      max_budget_usd=args.max_budget_usd,
    )
    results = runner.run_all()
  else:
    log("Phase 2: Skipped (--analyze-only)")
    # Re-analyze existing results
    results = []
    for i in range(1, args.n + 1):
      analysis_file = output_dir / f"puz-{i:04d}" / "analyses.json"
      if analysis_file.exists():
        with open(analysis_file) as f:
          results.append(json.load(f))
      else:
        results.append({"puzzle_id": i, "verdict": "PENDING"})

  # --- Phase 3: Summary ---
  log("Phase 3: Computing summary")
  metadata["end_time"] = datetime.datetime.now().isoformat()
  metadata["total_seconds"] = round(time.time() - start_time, 2)

  # Save metadata
  with open(output_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

  summary = compute_summary(results, output_dir, metadata)
  print_summary_table(summary)


if __name__ == "__main__":
  main()

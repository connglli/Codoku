"""codoku_creator.py - rysmith-based Python codoku puzzle creation.

Python-specific: generation drives ``rysmith --target python`` directly.
Masking and analysis reuse codoku_common.py with the angle-bracketed
<FILL_XXX> mask tokens.

Generation profiles control generator inputs; reported complexity values are
heuristic estimates based on the generated puzzle's realized structure, path,
masks, and constraints - not calibrated measures of solving difficulty.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import secrets
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, NamedTuple

from codoku_common import (
  ConstBudget,
  apply_replacements,
  build_python_cfg,
  collect_python_leaf_locals,
  collect_python_replacements,
  find_python_block_comments,
  find_python_leaf_function,
  get_line_indent,
  get_python_maskable_statements,
  mentions_go_flag,
  merge_const_split,
  stmt_is_on_live_path,
  strip_refractir_prefix,
  swap_preamble,
)
from codoku_complexity import (
  DISABLEABLE_MASKS,
  ComplexityEstimate,
  PuzzleMetrics,
  analyze_puzzle,
  estimate_complexity,
  filter_disabled_masks,
)

DEFAULT_MAX_ATTEMPTS = 20
RYSMITH_ATTEMPTS = 100

INSTRUCTION_TEMPLATE = """\
# Python Puzzle Solver

You are an expert Python programmer. A Python puzzle is a masked function where specific code elements have been replaced with `<FILL_XXX>` placeholders. Your job is to fill in these placeholders so the function produces the correct output along a prescribed execution path.

## Task

The puzzle file is at `puzzle.py`.
Save the complete solution to `solution.py`.
Use `./scratch/` for any intermediate files (scripts, notes, attempts, thoughts, etc.).
That said, avoid using `/tmp` or similar directories, as they may be cleaned up automatically.
That also said, avoid generating the solution file before you solve the puzzle successfully.

## How to Read the Puzzle

1. Read the puzzle file. Pay attention to:
   - The **CFG** (control-flow graph, `#//@ CFG_EDGE: ...`) at the top - shows which basic blocks exist and how they connect
   - The **execution path** (`#//@ EXEC_PATH: ...`) - the exact sequence of basic blocks that must execute
   {{BUDGET_READ}}
   - The **mask marks**: `<FILL_VAR>`, `<FILL_CONST>`, `<FILL_OP>`, `<FILL_TYPE>`, `<FILL_LABEL>`, `<FILL_FUNC>`, `<FILL_FIELD>`, `<FILL_CTRL>`

2. The function body uses comment-marked basic blocks and control keywords. The EXEC_PATH tells you which basic blocks are executed in sequence.

## How to Fill in the Blanks

- `<FILL_VAR>` → a local variable or parameter name (possibly with `[idx]` subscript)
- {{CONST_FILL}}
- `<FILL_OP>` → a Python operator (`+`, `-`, `*`, `/`, `%`, `//`, `&`, `|`, `^`, `<<`, `>>`, `~`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `if`, `else`, `not`, `and`, `or`)
- `<FILL_CTRL>` → a control keyword (`break` or `continue`)
- `<FILL_LABEL>` → the destination of a goto flag (`_go_<FILL_LABEL>` such as `_go_b3`)
- `_<FILL_CTRL>_<FILL_LABEL>` → a break/continue flag: the kind and the destination are both hidden, so fill in both (e.g. `_break_exit`, `_continue_b4`)
- `<FILL_FUNC>` → a function call name (e.g., an intrinsic helper like `_in_abs`, or any non-internal function defined in the file)
- `<FILL_FIELD>` → not used
- `<FILL_TYPE>` → not used (Python is dynamically typed)

## Verification

Use the checker to verify your solution:
```bash
codoku check puzzle.py solution.py
```

`[PASS]` means your solution is valid. `[FAIL]` means something is wrong - read the error message.

You can also run the solution manually (the puzzle prints trace statements when DUMP_TRACE is set):
```bash
DUMP_TRACE=1 python solution.py
```

## Rules

- Replace ONLY the `<FILL_XXX>` marks. Do NOT change any other code.
- Do NOT add new variables, statements, or basic blocks.
- Do NOT remove any code.
{{BUDGET_RULE}}
- Save the complete solution file (the full program with blanks filled) - not just the changes.

## Strategy Tips

- Read the CFG and EXEC_PATH carefully - they tell you the control flow.
- Map out all local variables and their types from the declarations at the top of the function.
- Trace the execution path block by block, reasoning about what each statement must compute.
{{BUDGET_TIP}}
- Use the checker (`codoku check`) for the definitive pass/fail verdict.
- If the checker fails with a path mismatch, the control flow transitions are wrong - revisit `<FILL_CTRL>` (for control keywords) marks.
- If the checker fails with a structural integrity error, you changed something outside the blanks.
{{CHECK_ERR}}
"""

BUDGET_READ = (
  "- The **<FILL_CONST> budget** "
  "(`#//@ <FILL_CONST>: <value> <live> <dead>` lines) - constants you must use"
)
NO_BUDGET_READ = (
  "- The **<FILL_CONST> marks** - fill each with an integer or float literal "
  "that keeps the function correct"
)
CONST_FILL_BUDGET = (
  "`<FILL_CONST>` → an integer or float literal (must match the budget "
  "exactly - right value, right type, right live/dead split; `2` and `2.0` "
  "are distinct)"
)
CONST_FILL_FREE = (
  "`<FILL_CONST>` → an integer or float literal (choose any value that keeps "
  "the function correct; `0`, `1`, `0.0`, and `1.0` stay visible and do not "
  "fill a mark)"
)
BUDGET_RULE = (
  "- The `<FILL_CONST>` budget must be matched exactly: each value at its "
  "exact live and dead counts, no extras, and with the same type (integer "
  "vs float - `2` is not `2.0`). A constant parked in the wrong region "
  "fails even when the totals add up.\n"
  "- `<FILL_CONST>` is never `0`, `1`, `0.0`, or `1.0`: those literals stay "
  "visible in the puzzle and are not masked anywhere. Filling a mark with "
  "them fails re-masking (`FAIL_REMASKING`)."
)
BUDGET_TIP = (
  "- For each `<FILL_CONST>`, use the budget "
  "(`#//@ <FILL_CONST>: <value> <live> <dead>` lines) to constrain your choices."
)
CHECK_ERR = (
  "- If the checker fails with a <FILL_CONST> budget error, you used the "
  "wrong constant value, type, or count.\n"
  "- If the checker fails with a `FAIL_REMASKING` error after filling "
  "`0`, `1`, `0.0`, or `1.0` into a `<FILL_CONST>` mark, replace them with "
  "other budgeted values."
)


def render_instruction(has_budget: bool) -> str:
  """Render INSTRUCTION.md; budget lines are only shown when a budget exists."""
  return (
    INSTRUCTION_TEMPLATE.replace(
      "{{BUDGET_READ}}", BUDGET_READ if has_budget else NO_BUDGET_READ
    )
    .replace("{{CONST_FILL}}", CONST_FILL_BUDGET if has_budget else CONST_FILL_FREE)
    .replace("{{BUDGET_RULE}}", BUDGET_RULE if has_budget else "")
    .replace("{{BUDGET_TIP}}", BUDGET_TIP if has_budget else "")
    .replace("{{CHECK_ERR}}", CHECK_ERR if has_budget else "")
  )


def write_instruction(path: Path, has_budget: bool) -> None:
  path.write_text(render_instruction(has_budget))


# ---------------------------------------------------------------------------
# Banner templates (Python-adapted: `//` comment lines become `#`,
# machine markers become `#//@`).
# ---------------------------------------------------------------------------

PUZZLE_HEADER_TEMPLATE = """\
#
# {{LEAF_NAME}}() is a function of the following CFG:
#
{{CFG}}//
# ------------------------------------------------
# Task
# ------------------------------------------------
#
# Replace all occurrences of <FILL_XXX> with appropriate code to make
# the function return the expected value for the test case in main
# following the below execution path:
#
#//@ EXEC_PATH: {{PATH}}
{{DISABLED_MASKS}}#
# ------------------------------------------------
# Validation
# ------------------------------------------------
#
# Use the following command to verify your solution:
#
#   codoku check [this_puzzle_file].py [your_solution].py
#
# ------------------------------------------------
# General Requirements
# ------------------------------------------------
#
# 1. Each <FILL_XXX> mark must be filled out with a corresponding element.
# 2. You have access to all common command line tools and SMT solvers.
# 3. Do NOT change any code except for the <FILL_XXX> marks.
# 4. Do NOT introduce any new code, variables, or basic blocks.
# 5. <FILL_CONST> is never `0`, `1`, `0.0`, or `1.0` (those literals stay
# visible). Do NOT fill a mark with them; that fails with FAIL_REMASKING.
#
{{BUDGET_SECTION}}//
"""

BUDGET_SECTION_TEMPLATE = """\
# ------------------------------------------------
# Requirements for <FILL_CONST>
# ------------------------------------------------
#
# The lines below list every constant the <FILL_CONST> marks must carry, as
# "<value> <live> <dead>" triples. <live> counts the slots in blocks on the
# execution path (plus the declarations before `# ^entry`, which always run);
# <dead> counts the slots in blocks off the path. Across your whole solution
# each <value> must appear in <FILL_CONST> positions exactly <live> times on
# the path and <dead> times off it -- no more, no fewer -- and no other
# constant may appear in any <FILL_CONST> position. The value must match
# exactly, including its type: `2` (integer) and `2.0` (float) are distinct.
# Constants already shown in the fixed (entry/exit) code do not count toward
# this budget. `0`, `1`, `0.0`, and `1.0` stay visible and never take a
# <FILL_CONST> position, so they are not in the budget.
#
{{FILL_CONST}}//
"""


# ---------------------------------------------------------------------------
# Profiles and ranges
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntRange:
  minimum: int
  maximum: int

  def sample(self, rng: random.Random) -> int:
    return rng.randint(self.minimum, self.maximum)

  def validate(self, name: str) -> None:
    if self.minimum > self.maximum:
      raise ValueError(f"{name}: minimum {self.minimum} exceeds maximum {self.maximum}")


@dataclass(frozen=True)
class FloatRange:
  minimum: float
  maximum: float

  def sample(self, rng: random.Random) -> float:
    return rng.uniform(self.minimum, self.maximum)

  def validate(self, name: str) -> None:
    if self.minimum > self.maximum:
      raise ValueError(f"{name}: minimum {self.minimum} exceeds maximum {self.maximum}")


@dataclass(frozen=True)
class GeneratorConfig:
  """Inputs passed to rysmith (see `rysmith --help`).

  `features` carries the boolean rysmith toggles (e.g. `--no-fp`); the
  numeric knobs are dedicated fields.
  """

  n_bbls: int
  n_stmts: int
  min_loop_iter: int
  p_mask: float
  max_ptr_depth: int
  p_backedge: float
  p_branch: float
  n_vars: int
  n_params: int
  lift_consts: bool
  features: tuple[str, ...] = ()
  disabled_masks: frozenset[str] = frozenset()

  def validate(self) -> None:
    if self.n_bbls < 1:
      raise ValueError("n_bbls must be at least 1")
    if self.n_stmts < 1:
      raise ValueError("n_stmts must be at least 1")
    if self.min_loop_iter < 0:
      raise ValueError("min_loop_iter must be at least 0")
    if not 0.0 <= self.p_mask <= 1.0:
      raise ValueError("p_mask must be in [0, 1]")
    if self.max_ptr_depth < 0:
      raise ValueError("max_ptr_depth must be at least 0")
    if not 0.0 <= self.p_backedge <= 1.0:
      raise ValueError("p_backedge must be in [0, 1]")
    if not 0.0 <= self.p_branch <= 1.0:
      raise ValueError("p_branch must be in [0, 1]")
    if self.n_vars < 1:
      raise ValueError("n_vars must be at least 1")
    if self.n_params < 1:
      raise ValueError("n_params must be at least 1")


@dataclass(frozen=True)
class GenerationProfile:
  """Distribution of generator inputs plus realized-metric bounds.

  `acceptance` is an inclusive [min, max] interval per PuzzleMetrics metric
  (flattened names); a candidate outside any interval is rejected.
  """

  n_bbls: IntRange
  n_stmts: IntRange
  min_loop_iter: IntRange
  p_mask: FloatRange
  max_ptr_depth: IntRange
  p_backedge: FloatRange
  p_branch: FloatRange
  n_vars: IntRange
  n_params: IntRange
  lift_consts: bool
  features: tuple[str, ...] = ()
  disabled_masks: frozenset[str] = frozenset()
  acceptance: Mapping[str, tuple[float, float]] = field(default_factory=dict)

  def validate(self, name: str) -> None:
    self.n_bbls.validate(f"{name}.n_bbls")
    self.n_stmts.validate(f"{name}.n_stmts")
    self.min_loop_iter.validate(f"{name}.min_loop_iter")
    self.p_mask.validate(f"{name}.p_mask")
    self.max_ptr_depth.validate(f"{name}.max_ptr_depth")
    self.p_backedge.validate(f"{name}.p_backedge")
    self.p_branch.validate(f"{name}.p_branch")
    self.n_vars.validate(f"{name}.n_vars")
    self.n_params.validate(f"{name}.n_params")
    if not 0.0 <= self.p_mask.minimum <= 1.0:
      raise ValueError(f"{name}.p_mask minimum must be in [0, 1]")
    if not 0.0 <= self.p_mask.maximum <= 1.0:
      raise ValueError(f"{name}.p_mask maximum must be in [0, 1]")
    if not 0.0 <= self.p_backedge.minimum <= 1.0:
      raise ValueError(f"{name}.p_backedge minimum must be in [0, 1]")
    if not 0.0 <= self.p_backedge.maximum <= 1.0:
      raise ValueError(f"{name}.p_backedge maximum must be in [0, 1]")
    if not 0.0 <= self.p_branch.minimum <= 1.0:
      raise ValueError(f"{name}.p_branch minimum must be in [0, 1]")
    if not 0.0 <= self.p_branch.maximum <= 1.0:
      raise ValueError(f"{name}.p_branch maximum must be in [0, 1]")
    if self.max_ptr_depth.minimum < 0:
      raise ValueError(f"{name}.max_ptr_depth minimum must be at least 0")
    unknown_masks = set(self.disabled_masks) - DISABLEABLE_MASKS
    if unknown_masks:
      raise ValueError(
        f"{name}.disabled_masks: mask kinds cannot be disabled {sorted(unknown_masks)}"
      )
    for metric, bounds in self.acceptance.items():
      low, high = bounds
      if low > high:
        raise ValueError(
          f"{name}.acceptance.{metric}: minimum {low} exceeds maximum {high}"
        )


PROFILES: dict[str, GenerationProfile] = {
  # ---- by size ----
  # A small function with a short path, where part of the code stays visible and constants need no maching.
  "small": GenerationProfile(
    n_bbls=IntRange(2, 4),
    n_stmts=IntRange(2, 3),
    min_loop_iter=IntRange(0, 1),
    p_mask=FloatRange(0.5, 0.7),
    max_ptr_depth=IntRange(0, 0),
    p_backedge=FloatRange(0.1, 0.3),
    p_branch=FloatRange(0.3, 0.5),
    n_vars=IntRange(6, 10),
    n_params=IntRange(2, 3),
    lift_consts=False,
    features=(
      "--no-fp",
      "--no-vec",
      "--no-ptrarith",
      "--no-intrinsics",
    ),
    acceptance={
      "exec_path_length": (3, 15),
      "total_masks": (10, 90),
      "mask_<fill_ctrl>": (0, 4),
      "cyclomatic": (2, 6),
    },
  ),
  # A loop-driven function where most statement is masked and constants must match a budget.
  "medium": GenerationProfile(
    n_bbls=IntRange(3, 6),
    n_stmts=IntRange(2, 4),
    min_loop_iter=IntRange(1, 2),
    p_mask=FloatRange(0.8, 1.0),
    max_ptr_depth=IntRange(1, 1),
    p_backedge=FloatRange(0.2, 0.4),
    p_branch=FloatRange(0.4, 0.6),
    n_vars=IntRange(10, 16),
    n_params=IntRange(3, 4),
    lift_consts=False,
    features=(
      "--no-vec",
      "--no-ptrarith",
    ),
    acceptance={
      "exec_path_length": (6, 20),
      "total_masks": (50, 400),
      "mask_<fill_ctrl>": (0, 6),
      "cyclomatic": (2, 10),
    },
  ),
  # A large branching function with deep loops, nothing visible but the skeleton, and a tight constant budget.
  "large": GenerationProfile(
    n_bbls=IntRange(6, 10),
    n_stmts=IntRange(3, 5),
    min_loop_iter=IntRange(2, 4),
    p_mask=FloatRange(1.0, 1.0),
    max_ptr_depth=IntRange(2, 2),
    p_backedge=FloatRange(0.3, 0.5),
    p_branch=FloatRange(0.5, 0.7),
    n_vars=IntRange(14, 20),
    n_params=IntRange(4, 5),
    lift_consts=False,
    features=(),
    acceptance={
      "exec_path_length": (10, 2147483647),
      "total_masks": (150, 2147483647),
      "mask_<fill_ctrl>": (0, 2147483647),
      "cyclomatic": (3, 2147483647),
    },
  ),
  # ---- by types ----
  "dataflow_reconstruction": GenerationProfile(
    n_bbls=IntRange(6, 10),
    n_stmts=IntRange(3, 5),
    min_loop_iter=IntRange(1, 2),
    p_mask=FloatRange(1.0, 1.0),
    max_ptr_depth=IntRange(1, 1),
    p_backedge=FloatRange(0.20, 0.40),
    p_branch=FloatRange(0.50, 0.70),
    n_vars=IntRange(10, 14),
    n_params=IntRange(3, 5),
    lift_consts=True,
    # We consider constants, variables, and control flows.
    disabled_masks=frozenset({"<FILL_OP>", "<FILL_FUNC>"}),
    features=(),
    acceptance={
      "exec_path_length": (10, 30),
      "unique_path_blocks": (6, 12),
      "repeated_block_visits": (2, 20),
      "n_loops": (1, 2147483647),
      "cyclomatic": (4, 14),
      "total_masks": (1, 2147483647),
    },
  ),
  "arithmetic_reconstruction": GenerationProfile(
    n_bbls=IntRange(6, 10),
    n_stmts=IntRange(3, 5),
    min_loop_iter=IntRange(1, 2),
    p_mask=FloatRange(1.0, 1.0),
    max_ptr_depth=IntRange(1, 1),
    p_backedge=FloatRange(0.20, 0.40),
    p_branch=FloatRange(0.50, 0.70),
    n_vars=IntRange(10, 14),
    n_params=IntRange(3, 5),
    lift_consts=True,
    # We consider constants, operators, and control flows.
    disabled_masks=frozenset({"<FILL_VAR>"}),
    features=(),
    acceptance={
      "exec_path_length": (10, 30),
      "unique_path_blocks": (6, 12),
      "repeated_block_visits": (2, 20),
      "n_loops": (1, 2147483647),
      "cyclomatic": (4, 14),
      "total_masks": (1, 2147483647),
    },
  ),
  "operation_routing": GenerationProfile(
    n_bbls=IntRange(6, 10),
    n_stmts=IntRange(3, 5),
    min_loop_iter=IntRange(1, 2),
    p_mask=FloatRange(1.0, 1.0),
    max_ptr_depth=IntRange(1, 1),
    p_backedge=FloatRange(0.20, 0.40),
    p_branch=FloatRange(0.50, 0.70),
    n_vars=IntRange(10, 14),
    n_params=IntRange(3, 5),
    # We consider variables, operators, and control flows.
    disabled_masks=frozenset({"<FILL_CONST>"}),
    lift_consts=True,
    features=(),
    acceptance={
      "exec_path_length": (10, 30),
      "unique_path_blocks": (6, 12),
      "repeated_block_visits": (2, 20),
      "n_loops": (1, 2147483647),
      "cyclomatic": (4, 14),
      "total_masks": (1, 2147483647),
    },
  ),
  # ---- by style ----
  # Visible arithmetic anchors + exact constant allocation.
  # Retain enough fixed computation to restrict alternative completions.
  "anchored": GenerationProfile(
    n_bbls=IntRange(5, 8),
    n_stmts=IntRange(3, 5),
    min_loop_iter=IntRange(1, 2),
    p_mask=FloatRange(0.60, 0.75),
    max_ptr_depth=IntRange(0, 0),
    p_backedge=FloatRange(0.20, 0.35),
    p_branch=FloatRange(0.40, 0.60),
    n_vars=IntRange(6, 10),
    n_params=IntRange(3, 4),
    lift_consts=False,
    features=(
      "--no-fp",
      "--no-vec",
      "--no-ptrarith",
      "--no-intrinsics",
    ),
    acceptance={
      "exec_path_length": (8, 20),
      "total_masks": (50, 250),
      "cyclomatic": (3, 10),
    },
  ),
  # Repeated state updates, with some visible statements preserved.
  # The supplied 21-block-visit example is a useful target.
  "recurrent": GenerationProfile(
    n_bbls=IntRange(5, 8),
    n_stmts=IntRange(3, 5),
    min_loop_iter=IntRange(2, 3),
    p_mask=FloatRange(0.65, 0.85),
    max_ptr_depth=IntRange(0, 0),
    p_backedge=FloatRange(0.30, 0.45),
    p_branch=FloatRange(0.40, 0.60),
    n_vars=IntRange(6, 10),
    n_params=IntRange(3, 4),
    lift_consts=False,
    features=(
      "--no-vec",
      "--no-ptrarith",
    ),
    acceptance={
      "exec_path_length": (12, 26),
      "total_masks": (70, 350),
      "cyclomatic": (3, 12),
    },
  ),
  # Short execution with mixed features and argument-level memory puzzles.
  # Assumes pointer/memory operation names remain visible.
  # Avoid full masking and deeper pointer chains for now.
  "saturated": GenerationProfile(
    n_bbls=IntRange(6, 10),
    n_stmts=IntRange(3, 5),
    min_loop_iter=IntRange(1, 2),
    p_mask=FloatRange(0.70, 0.90),
    max_ptr_depth=IntRange(1, 1),
    p_backedge=FloatRange(0.20, 0.40),
    p_branch=FloatRange(0.50, 0.70),
    n_vars=IntRange(8, 12),
    n_params=IntRange(3, 4),
    lift_consts=False,
    features=(),
    acceptance={
      "exec_path_length": (10, 24),
      "total_masks": (100, 500),
      "cyclomatic": (4, 14),
    },
  ),
}

# Profile aliases
PROFILES["easy"] = PROFILES["small"]
PROFILES["hard"] = PROFILES["large"]


# ---------------------------------------------------------------------------
# Candidate and profile acceptance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedCandidate:
  directory: Path
  config: GeneratorConfig
  generator_seed: int
  metrics: PuzzleMetrics
  complexity: ComplexityEstimate


def profile_accepts(
  profile: GenerationProfile, metrics: PuzzleMetrics
) -> tuple[bool, list[str]]:
  values = metrics.flattened()
  failures: list[str] = []
  for metric_name, (minimum, maximum) in profile.acceptance.items():
    if metric_name not in values:
      raise ValueError(f"profile references unknown metric {metric_name!r}")
    value = values[metric_name]
    if not minimum <= value <= maximum:
      failures.append(f"{metric_name}={value} is outside [{minimum}, {maximum}]")
  return not failures, failures


# ---------------------------------------------------------------------------
# Candidate generation via rysmith
# ---------------------------------------------------------------------------


def build_rysmith_command(
  config: GeneratorConfig, seed: int, outdir: Path, rysmith_path: Path
) -> list[str]:
  max_loop_iter = config.min_loop_iter + 2
  cmd = [
    str(rysmith_path),
    "-n",
    "1",
    "--n-inits",
    "1",
    "--no-crc32",
    "--emit-main",
    "--target",
    "python",
    "--n-bbls",
    str(config.n_bbls),
    "--n-stmts",
    str(config.n_stmts),
    "--min-loop-iter",
    str(config.min_loop_iter),
    "--max-loop-iter",
    str(max_loop_iter),
    "--max-ptr-depth",
    str(config.max_ptr_depth),
    "--p-backedge",
    str(config.p_backedge),
    "--p-branch",
    str(config.p_branch),
    "--n-vars",
    str(config.n_vars),
    "--n-params",
    str(config.n_params),
    "--seed",
    str(seed),
    "-o",
    str(outdir),
  ]
  cmd.extend(config.features)
  return cmd


def run_rysmith(
  config: GeneratorConfig, seed: int, outdir: Path, rysmith_path: Path
) -> tuple[Path, Path] | None:
  """Run rysmith once; return (py_path, sir_path) or None on failure.

  rysmith can fail per seed (solver/constraint rejection); the caller retries
  with fresh seeds.
  """
  cmd = build_rysmith_command(config, seed, outdir, rysmith_path)
  try:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
  except subprocess.TimeoutExpired:
    return None
  if r.returncode != 0:
    return None

  py_stems = {f.stem for f in outdir.iterdir() if f.suffix == ".py"}
  sir_stems = {f.stem for f in outdir.iterdir() if f.suffix == ".sir"}
  common = py_stems & sir_stems
  if not common:
    return None
  # rysmith writes one init per stem (func_<id>_0<letter>), so a stem names
  # exactly one .py and one .sir.  Sort deterministically and take the
  # highest init letter rather than relying on directory-iteration order.
  stem = sorted(common)[-1]
  return outdir / (stem + ".py"), outdir / (stem + ".sir")


def extract_path_from_sir(sir_path: Path) -> str:
  """Read the `// PATH:` comment from the rysmith .sir output."""
  for line in sir_path.read_text().splitlines():
    if line.startswith("// PATH:"):
      return line.split("PATH:", 1)[1].strip()
  raise RuntimeError("rysmith banner is missing its '// PATH:' comment")


def extract_cfg_from_sir(sir_path: Path) -> list[tuple[str, str]]:
  """Read the `// CFG:` adjacency comments from the rysmith .sir output."""
  edges: list[tuple[str, str]] = []
  in_cfg = False
  for line in sir_path.read_text().splitlines():
    if line.startswith("// CFG:"):
      in_cfg = True
      continue
    if not in_cfg:
      continue
    if not line.startswith("//   "):
      break
    body = line[5:].strip()
    if "->" not in body:
      continue
    lhs, rhs = body.split("->", 1)
    for dst in rhs.split():
      edges.append((lhs.strip(), dst))
  if not in_cfg:
    raise RuntimeError("rysmith banner is missing its '// CFG:' comment")
  return edges


# ---------------------------------------------------------------------------
# Masking and puzzle assembly
# ---------------------------------------------------------------------------


def build_trace_replacements(leaf_node: ast.FunctionDef, src: bytes) -> list:
  """Insert DUMP_TRACE prints after each block comment in the leaf function."""
  replacements = []
  comments = find_python_block_comments(src)
  comments = [c for c in comments if leaf_node.lineno <= c[3] <= leaf_node.end_lineno]
  for start, end, label, line in comments:
    indent = get_line_indent(src, start)
    ins_text = f'\n{indent}if __import__("os").environ.get("DUMP_TRACE"):\n{indent}    print("^{label}:")'
    replacements.append((end, end, ins_text))
  return replacements


class MaskedPuzzle(NamedTuple):
  """Masking output: puzzle and ground-truth bodies, the masked statement
  set, and the constant budget as a (live, dead) split per value."""

  puzzle_body: str
  gt_body: str
  mask_set: set[int]
  budget_counts: ConstBudget


def mask_puzzle(
  src: bytes,
  leaf_node: ast.FunctionDef,
  entry_line: int,
  maskable: list,
  local_names: set[str],
  defined_funcs: set[str],
  p_mask: float,
  seed: int | None,
  live_blocks: frozenset[str],
  disabled_masks: frozenset[str] = frozenset(),
) -> MaskedPuzzle:
  """Return a MaskedPuzzle (puzzle_body, gt_body, mask_set, budget_counts).

  puzzle_body has both the DUMP_TRACE instrumentation and the <FILL_XXX> masks;
  gt_body has only the instrumentation.  Kinds in *disabled_masks* stay
  visible; their constructs are never masked.  budget_counts maps each
  constant to its (live, dead) slot split: slots in blocks on the
  prescribed path (plus pre-entry declarations) are live, the rest dead.
  """
  mask_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
  rng = random.Random(mask_seed)

  mask_set: set[int] = set()
  if p_mask > 0.0:
    for _ in range(100):
      mask_set = set()
      for idx in range(len(maskable)):
        if rng.random() < p_mask:
          mask_set.add(idx)
      if mask_set:
        break
    if not mask_set and p_mask > 1e-9:
      raise RuntimeError("failed to build a non-empty mask set after 100 attempts")

  # Goto flags of any spelling are always masked: one visible occurrence
  # would reveal every other <FILL_LABEL> slot's target.
  for idx, stmt in enumerate(maskable):
    if mentions_go_flag(stmt):
      mask_set.add(idx)

  trace_repls = build_trace_replacements(leaf_node, src)
  comments = find_python_block_comments(src)
  live_counts: dict[str, int] = {}
  dead_counts: dict[str, int] = {}
  mask_repls: list = []
  const_disabled = "<FILL_CONST>" in disabled_masks
  for idx, stmt in enumerate(maskable):
    if idx in mask_set:
      is_body = stmt.lineno > entry_line
      stmt_repls: list = []
      stmt_budget: dict[str, int] = {}
      collect_python_replacements(
        stmt, src, is_body, stmt_repls, stmt_budget, local_names, defined_funcs
      )
      # Disabled kinds stay visible: drop their spans so only enabled
      # kinds are masked (goto-flag spans never match a known kind).
      mask_repls.extend(filter_disabled_masks(stmt_repls, disabled_masks))
      if not const_disabled:
        slot_counts = (
          live_counts
          if stmt_is_on_live_path(comments, stmt.lineno, live_blocks)
          else dead_counts
        )
        for val, cnt in stmt_budget.items():
          slot_counts[val] = slot_counts.get(val, 0) + cnt

  puzzle_body = apply_replacements(src, trace_repls + mask_repls).decode("utf-8")
  gt_body = apply_replacements(src, trace_repls).decode("utf-8")
  budget_counts = merge_const_split(live_counts, dead_counts)
  return MaskedPuzzle(puzzle_body, gt_body, mask_set, budget_counts)


def render_header(
  leaf_name: str,
  cfg_edges: list,
  path_str: str,
  budget_counts: ConstBudget,
  lift_consts: bool,
  disabled_masks: frozenset[str] = frozenset(),
) -> str:
  fill_const_lines = "".join(
    f"#//@ <FILL_CONST>: {val} {live} {dead}\n"
    for val in sorted(budget_counts)
    for live, dead in [budget_counts[val]]
  )
  # No <FILL_CONST> marks when constants are disabled, so the budget section
  # (a prose banner over an empty list) is suppressed like lift_consts.
  if lift_consts or "<FILL_CONST>" in disabled_masks:
    budget_section = ""
  else:
    budget_section = BUDGET_SECTION_TEMPLATE.replace("{{FILL_CONST}}", fill_const_lines)

  # The disabled kinds ride in the banner so the checker re-masks with the
  # same vocabulary: without it the checker would mask kinds the puzzle left
  # visible and fail the structural comparison.
  disabled_masks_str = (
    f"#\n# In this task, the following masks are disabled:\n#\n#//@ DISABLED_MASKS: {' '.join(sorted(disabled_masks))}\n"
    if disabled_masks
    else ""
  )

  cfg_edges_str = "".join(f"#//@ CFG_EDGE: {f} -> {t}\n" for f, t in sorted(cfg_edges))
  header = (
    PUZZLE_HEADER_TEMPLATE.replace("{{LEAF_NAME}}", leaf_name)
    .replace("{{CFG}}", cfg_edges_str if cfg_edges_str else "#   [unknown CFG]\n")
    .replace("{{PATH}}", path_str if path_str else "[unknown]")
    .replace("{{DISABLED_MASKS}}", disabled_masks_str)
    .replace("{{BUDGET_SECTION}}", budget_section)
  )
  lines = []
  for line in header.splitlines():
    if line.startswith("//@"):
      lines.append("#//@" + line[3:])
    elif line.startswith("//"):
      lines.append("#" + line[2:])
    else:
      lines.append(line)
  return "\n".join(lines) + "\n"


def self_check(
  gt_body: str,
  puzzle_body: str,
  mask_set: set[int],
  defined_funcs: set[str],
  cfg_edges: list,
  disabled_masks: frozenset[str] = frozenset(),
) -> bool:
  """Re-mask the ground truth and verify it reproduces the puzzle exactly."""
  import difflib

  gt_bytes = gt_body.encode("utf-8")
  try:
    gt_tree = ast.parse(gt_bytes)
  except Exception as e:
    print(
      f"Error: self-check failed: ground truth has syntax errors: {e}", file=sys.stderr
    )
    return False
  gt_leaf, _ = find_python_leaf_function(gt_tree, gt_bytes)
  if not gt_leaf:
    print(
      "Error: self-check failed: no leaf function in ground truth.", file=sys.stderr
    )
    return False
  gt_maskable, entry_line, exit_line = get_python_maskable_statements(gt_leaf, gt_bytes)
  if not entry_line or not exit_line:
    print(
      "Error: self-check failed: ground truth missing entry/exit comments.",
      file=sys.stderr,
    )
    return False
  local_names = collect_python_leaf_locals(gt_leaf)

  # self-check verifies structure only; the budget is unused.
  remasked_repls: list = []
  for idx, stmt in enumerate(gt_maskable):
    if idx in mask_set:
      is_body = stmt.lineno > entry_line
      stmt_repls: list = []
      collect_python_replacements(
        stmt, gt_bytes, is_body, stmt_repls, {}, local_names, defined_funcs
      )
      remasked_repls.extend(filter_disabled_masks(stmt_repls, disabled_masks))
  remasked = apply_replacements(gt_bytes, remasked_repls).decode("utf-8")
  if remasked != puzzle_body:
    print(
      "\n".join(
        difflib.unified_diff(
          remasked.splitlines(), puzzle_body.splitlines(), lineterm=""
        )
      ),
      file=sys.stderr,
    )
    print(
      "Error: self-check failed: ground truth does not re-mask to the puzzle.",
      file=sys.stderr,
    )
    return False

  if cfg_edges:
    actual_edges = build_python_cfg(gt_leaf, gt_bytes)
    if set(cfg_edges) != actual_edges:
      print(
        "Error: self-check failed: declared CFG edges do not match ground truth.\n"
        f"  Declared: {set(cfg_edges)}\n  Actual:   {actual_edges}",
        file=sys.stderr,
      )
      return False
  return True


# ---------------------------------------------------------------------------
# Candidate loop, installation, and CLI
# ---------------------------------------------------------------------------


def sample_config(profile: GenerationProfile, rng: random.Random) -> GeneratorConfig:
  config = GeneratorConfig(
    n_bbls=profile.n_bbls.sample(rng),
    n_stmts=profile.n_stmts.sample(rng),
    min_loop_iter=profile.min_loop_iter.sample(rng),
    p_mask=round(profile.p_mask.sample(rng), 4),
    max_ptr_depth=profile.max_ptr_depth.sample(rng),
    p_backedge=round(profile.p_backedge.sample(rng), 4),
    p_branch=round(profile.p_branch.sample(rng), 4),
    n_vars=profile.n_vars.sample(rng),
    n_params=profile.n_params.sample(rng),
    lift_consts=profile.lift_consts,
    features=profile.features,
    disabled_masks=profile.disabled_masks,
  )
  config.validate()
  return config


def child_seed(rng: random.Random) -> int:
  # rysmith parses --seed as uint32_t; keep child seeds in range.
  return rng.randrange(0, 2**32)


def replace_file(source: Path, destination: Path) -> None:
  destination.parent.mkdir(parents=True, exist_ok=True)
  if destination.exists():
    if destination.is_dir():
      shutil.rmtree(destination)
    else:
      destination.unlink()
  shutil.move(str(source), str(destination))


def install_candidate(
  candidate: GeneratedCandidate,
  outdir: Path,
  profile_name: str,
  master_seed: int,
  accepted_attempt: int,
) -> None:
  puzzle_source = candidate.directory / "puzzle.py"
  ground_truth_source = candidate.directory / "puzzle.gt.py"

  puzzle_destination = outdir / "puzzle.py"
  replace_file(puzzle_source, puzzle_destination)
  postprocess_puzzle(puzzle_destination)

  oracle_destination = outdir / "oracle" / "puzzle.gt.py"
  if ground_truth_source.exists():
    replace_file(ground_truth_source, oracle_destination)
  elif oracle_destination.exists():
    oracle_destination.unlink()

  write_instruction(outdir / "INSTRUCTION.md", not candidate.config.lift_consts)

  metrics_data = asdict(candidate.metrics)
  metrics_data["masks_by_kind"] = dict(candidate.metrics.masks_by_kind)
  config_data = asdict(candidate.config)
  config_data["disabled_masks"] = sorted(candidate.config.disabled_masks)
  manifest = {
    "target": "python",
    "profile": profile_name,
    "master_seed": master_seed,
    "generator_seed": candidate.generator_seed,
    "accepted_attempt": accepted_attempt,
    "generator_config": config_data,
    "realized_metrics": metrics_data,
    "complexity_estimate": asdict(candidate.complexity),
  }
  manifest_path = outdir / "oracle" / "metadata.json"
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def postprocess_puzzle(path: Path) -> None:
  """Rewrite the generated puzzle banner's validation command."""
  text = path.read_text()
  puzzle_name = path.name
  text = text.replace("./tools/rypuzchk", "codoku check")
  text = text.replace("[this_puzzle_file].py", puzzle_name)
  text = text.replace("[your_solution].py", "solution.py")
  path.write_text(text)


def generate_candidate(
  candidate_dir: Path,
  config: GeneratorConfig,
  generator_seed: int,
  rysmith_path: Path,
) -> GeneratedCandidate | None:
  """Run rysmith + masking in candidate_dir; return the analyzed candidate."""
  pair = None
  used_seed = generator_seed
  for attempt in range(RYSMITH_ATTEMPTS):
    for f in candidate_dir.iterdir():
      if f.is_file():
        f.unlink()
    used_seed = generator_seed + attempt
    pair = run_rysmith(config, used_seed, candidate_dir, rysmith_path=rysmith_path)
    if pair is not None:
      break
  if pair is None:
    return None
  py_path, sir_path = pair

  src_raw = py_path.read_bytes()
  src = strip_refractir_prefix(swap_preamble(src_raw))
  try:
    tree = ast.parse(src)
  except Exception as e:
    raise RuntimeError(f"rysmith output is not valid Python: {e}")

  leaf_node, leaf_name = find_python_leaf_function(tree, src)
  if leaf_node is None:
    raise RuntimeError("no leaf function in rysmith output")
  maskable, entry_line, exit_line = get_python_maskable_statements(leaf_node, src)
  if not entry_line or not exit_line:
    raise RuntimeError("leaf function missing entry/exit comments")

  local_names = collect_python_leaf_locals(leaf_node)
  defined_funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

  # rysmith's generated banner is trusted as written: the puzzle banner
  # embeds the adjacency and PATH comments verbatim.
  cfg_edges = extract_cfg_from_sir(sir_path)
  path_str = extract_path_from_sir(sir_path)
  live_blocks = frozenset(
    block.strip() for block in path_str.split("->") if block.strip()
  )

  masked = mask_puzzle(
    src,
    leaf_node,
    entry_line,
    maskable,
    local_names,
    defined_funcs,
    config.p_mask,
    used_seed,
    live_blocks,
    config.disabled_masks,
  )

  if not self_check(
    masked.gt_body,
    masked.puzzle_body,
    masked.mask_set,
    defined_funcs,
    cfg_edges,
    config.disabled_masks,
  ):
    raise RuntimeError("self-check failed")

  header = render_header(
    leaf_name,
    cfg_edges,
    path_str,
    masked.budget_counts,
    config.lift_consts,
    config.disabled_masks,
  )
  (candidate_dir / "puzzle.py").write_text(header + masked.puzzle_body)
  (candidate_dir / "puzzle.gt.py").write_text(masked.gt_body)

  metrics = analyze_puzzle(
    candidate_dir / "puzzle.py", gt_path=candidate_dir / "puzzle.gt.py"
  )
  complexity = estimate_complexity(metrics)
  return GeneratedCandidate(
    directory=candidate_dir,
    config=config,
    generator_seed=used_seed,
    metrics=metrics,
    complexity=complexity,
  )


def generate(args: argparse.Namespace, rysmith_path: Path) -> int:
  rysmith = Path(rysmith_path).resolve()
  if not rysmith.exists():
    print(f"codoku: error: rysmith not found at {rysmith}", file=sys.stderr)
    return 2

  profile = PROFILES[args.profile]
  profile.validate(args.profile)

  outdir = Path(args.outdir).resolve()
  outdir.mkdir(parents=True, exist_ok=True)

  master_seed = args.seed if args.seed is not None else secrets.randbits(63)
  master_rng = random.Random(master_seed)

  rejections: list[str] = []
  for attempt in range(1, args.max_attempts + 1):
    config = sample_config(profile, master_rng)
    generator_seed = child_seed(master_rng)

    with tempfile.TemporaryDirectory(prefix=".codoku-candidate-", dir=outdir) as tmp:
      candidate_dir = Path(tmp)
      try:
        candidate = generate_candidate(
          candidate_dir, config, generator_seed, rysmith_path=rysmith
        )
      except RuntimeError as e:
        message = f"attempt {attempt}: {e}"
        rejections.append(message)
        print(message, file=sys.stderr)
        continue

      if candidate is None:
        message = f"attempt {attempt}: rysmith failed"
        rejections.append(message)
        print(message, file=sys.stderr)
        continue

      accepted, failures = profile_accepts(profile, candidate.metrics)
      if not accepted:
        reason = "; ".join(failures)
        message = f"attempt {attempt}: rejected: {reason}"
        rejections.append(message)
        print(message, file=sys.stderr)
        continue

      install_candidate(candidate, outdir, args.profile, master_seed, attempt)
      print(
        f"accepted candidate {attempt}: profile={args.profile} "
        f"seed={master_seed} "
        f"path_length={candidate.metrics.exec_path_length} "
        f"masks={candidate.metrics.total_masks} "
        f"estimated_sol_space_log10={candidate.metrics.sol_space_log10:.2f} "
        f"estimated_complexity={candidate.complexity.total}"
      )
      return 0

  print(
    f"codoku: error: failed to generate an acceptable puzzle after {args.max_attempts} attempts",
    file=sys.stderr,
  )
  for rejection in rejections[-5:]:
    print(f"  {rejection}", file=sys.stderr)
  return 1

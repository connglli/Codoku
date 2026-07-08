#!/usr/bin/env python3
"""rypuzmk.py — C Puzzle Maker

Generates a fill-in-the-blanks C puzzle from a rysmith-generated concrete C
program.  Parallels the RefractIR rypuzmk.cpp but targets the C backend
output of rysmith (--target c).

Puzzle format:
  - A machine-readable instruction banner (//@ PATH: … and //@ FILL_CONST: …)
  - The masked C source with FILL_XXX placeholders in body statements
  - An optional unmasked ground-truth copy (--keep-ground-truth)
"""

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

import tree_sitter_c as tsc
from puzzle_common import (
  apply_replacements,
  collect_defined_functions,
  collect_leaf_locals,
  collect_replacements,
  find_leaf_function,
  get_maskable_statements,
  sanitize_statement_expressions,
  strip_refractir_prefix,
)
from tree_sitter import Language, Parser

PWD = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Puzzle Header Templates
# -----------------------------------------------------------------------------

# Lines beginning with `//@` are machine-readable markers parsed by rypuzchk.py;
# everything else is for the human/agent solver.  Keep `//@` markers stable.
PUZZLE_HEADER_TEMPLATE = """\
//
//
// {{LEAF_NAME}}() is a function of the following CFG:
//
{{CFG}}//
// ------------------------------------------------
// Task
// ------------------------------------------------
//
// Replace all occurrences of FILL_XXX with appropriate code to make
// the function return the expected value for the test case in main
// following the below execution path:
//
//@ PATH: {{PATH}}
//
// ------------------------------------------------
// Validation
// ------------------------------------------------
//
// Use the following command to verify your solution:
//
//   ./tools/rypuzchk.py [this_puzzle_file].c [your_solution].c
//
// ------------------------------------------------
// General Requirements
// ------------------------------------------------
//
// 1. Each FILL_XXX mark must be filled out with a corresponding element.
// 2. You have access to all common command line tools.
// 3. Do NOT change any code except for the FILL_XXX marks.
// 4. Do NOT introduce any new code, variables, or basic blocks.
//
{{BUDGET_SECTION}}//
"""

BUDGET_SECTION_TEMPLATE = """\
// ------------------------------------------------
// Requirements for FILL_CONST
// ------------------------------------------------
//
// The lines below list every constant the FILL_CONST marks must carry, as
// "<value> <count>" pairs. Across your whole solution each <value> must appear
// in FILL_CONST positions exactly <count> times -- no more, no fewer -- and no
// other constant may appear in any FILL_CONST position. Constants already shown
// in the fixed (entry/exit) code do not count toward this budget.
//
{{FILL_CONST}}//
"""


# -----------------------------------------------------------------------------
# Trace Instrumentation Helper
# -----------------------------------------------------------------------------


def _build_trace_replacements(leaf_node, src: bytes) -> list:
  """Build ``(ins_point, ins_point, text)`` insertions for DUMP_TRACE blocks.

  Inserts ``#ifdef DUMP_TRACE / printf("^<name>:\\n"); / #endif`` immediately
  after each label's colon in the leaf function body.  After prefix stripping,
  all labels are bare names (``entry``, ``exit``, ``b0``, …) that match the
  SIR PATH format directly.
  """
  replacements = []
  body = leaf_node.child_by_field_name("body")
  if not body:
    return replacements
  for child in body.children:
    if child.type == "labeled_statement":
      label_id = child.child_by_field_name("label")
      if label_id:
        lbl_name = src[label_id.start_byte : label_id.end_byte].decode("utf-8")
        # After prefix stripping all leaf labels are ours; instrument all.
        stmt_child = child.children[2]
        ins_point = stmt_child.end_byte
        ins_text = f'\n#ifdef DUMP_TRACE\n  printf("^{lbl_name}:\\n");\n#endif'
        replacements.append((ins_point, ins_point, ins_text))
  return replacements


# -----------------------------------------------------------------------------
# Self-Check: Ground-Truth Re-Masks to Puzzle
# -----------------------------------------------------------------------------


def self_check_puzzle(
  puzzle_body: str, gt_body: str, mask_set: set, ts_parser, defined_funcs: set[str]
) -> bool:
  """Re-mask the unmasked ground-truth and verify it matches the puzzle body.

  ``gt_body`` already contains the DUMP_TRACE instrumentation (it is the
  original source with only trace blocks inserted, no masking).  The self-check
  re-parses ``gt_body``, derives masking replacements from its AST byte offsets,
  and applies *only* those (no trace re-insertion — that would double-insert the
  blocks).  The result must equal ``puzzle_body`` exactly.

  Returns True on success, False on failure.
  """
  gt_bytes = gt_body.encode("utf-8")
  gt_sanitized = sanitize_statement_expressions(gt_bytes)
  gt_tree = ts_parser.parse(gt_sanitized)
  gt_leaf, _ = find_leaf_function(gt_tree.root_node, gt_sanitized)
  if not gt_leaf:
    print(
      "Error: self-check failed: regenerated ground truth has no leaf function.",
      file=sys.stderr,
    )
    return False

  gt_maskable, gt_entry, gt_exit = get_maskable_statements(gt_leaf, gt_sanitized)
  if not gt_entry or not gt_exit:
    print(
      "Error: self-check failed: ground truth is missing entry/exit labels.",
      file=sys.stderr,
    )
    return False

  local_names = collect_leaf_locals(gt_leaf, gt_sanitized)
  # Re-use the defined_funcs from the original parse — the ground truth is the
  # same source with only DUMP_TRACE blocks inserted, so the function set is
  # identical; computing it from gt_tree would give the same result.

  # Derive masking replacements from gt_body's own byte offsets.
  # Do NOT call _build_trace_replacements here — gt_body already has those
  # blocks; re-inserting them would produce double trace output and wrong offsets.
  remasked_repls: list = []
  gt_budget: dict = {}
  for idx, stmt in enumerate(gt_maskable):
    if idx in mask_set:
      is_body = stmt.start_byte > gt_entry.start_byte
      collect_replacements(
        stmt,
        gt_sanitized,
        is_body,
        remasked_repls,
        gt_budget,
        local_names,
        defined_funcs,
      )

  remasked = apply_replacements(gt_bytes, remasked_repls).decode("utf-8")
  if remasked != puzzle_body:
    print(
      "Error: self-check failed: ground truth does not re-mask to the puzzle "
      "(printer/AST round-trip mismatch).",
      file=sys.stderr,
    )
    return False

  return True


# -----------------------------------------------------------------------------
# Metadata Extraction from SIR Companion File
# -----------------------------------------------------------------------------


def extract_metadata_from_sir(sir_path: str) -> tuple[str, str, str]:
  """Extract PATH, CFG adjacency lines, and solved checksum from a .sir file.

  rysmith emits ``// PATH: …`` and ``//   a -> b`` comments as part of its
  SIR output; those carry the execution path and CFG that we embed in the
  puzzle banner.  The SIR path already uses bare block names (``entry``,
  ``b0``, ``exit``) without the ``refractir_`` prefix, so no further
  transformation is needed.

  Returns ``(path_str, cfg_str, checksum_str)``.
  """
  path_val = ""
  cfg_lines = []
  checksum_val = ""

  if not os.path.exists(sir_path):
    return path_val, "", checksum_val

  with open(sir_path, "r") as f:
    for line in f:
      if line.startswith("// PATH:"):
        path_val = line.split("PATH:", 1)[1].strip()
      elif line.startswith("//   ") and "->" in line:
        cfg_lines.append(line.strip())
      elif line.startswith("// SOLVED:"):
        m = re.search(r"ret=(-?\d+)", line)
        if m:
          checksum_val = m.group(1)

  cfg_str = "\n".join(cfg_lines) + "\n" if cfg_lines else ""
  return path_val, cfg_str, checksum_val


# -----------------------------------------------------------------------------
# Main Processing Pipeline
# -----------------------------------------------------------------------------


def run_rysmith_loop(args, run_seed: int, tmp_dir: str) -> tuple[str, str]:
  """Run rysmith repeatedly (up to 100 times) until a C+SIR pair is produced.

  Seeds are incremented on each failed attempt.  Returns ``(c_path, sir_path)``.
  Exits with code 1 after 100 failures.
  """
  attempt = 0
  while attempt < 100:
    # Clean the temp dir before each attempt.
    for item in os.listdir(tmp_dir):
      item_path = os.path.join(tmp_dir, item)
      if os.path.isdir(item_path):
        shutil.rmtree(item_path)
      else:
        os.remove(item_path)

    cmd = [
      args.rysmith,
      "-n",
      "1",
      "--no-crc32",
      "--emit-main",
      "--min-loop-iter",
      str(args.min_loop_iter),
      "--n-bbls",
      str(args.n_bbls),
      "--n-stmts",
      str(args.n_stmts),
      "-o",
      tmp_dir,
      "--seed",
      str(run_seed),
      "--target",
      "c",
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    c_path = sir_path = None
    for f in os.listdir(tmp_dir):
      if f.endswith(".c"):
        c_path = os.path.join(tmp_dir, f)
      elif f.endswith(".sir"):
        sir_path = os.path.join(tmp_dir, f)

    if c_path and sir_path:
      return c_path, sir_path

    run_seed += 1
    attempt += 1

  print(
    "Error: rysmith did not generate a C+SIR pair after 100 attempts.", file=sys.stderr
  )
  sys.exit(1)


def create_puzzle(c_path: str, sir_path: str | None, args) -> None:
  """Parse *c_path*, apply masks, and write the puzzle (and optionally the ground truth).

  This is the main pipeline function called whether the source came from
  rysmith generation or was supplied directly via ``--input``.
  """
  with open(c_path, "rb") as f:
    src_raw = f.read()

  # Strip the refractir_ prefix from all identifiers to keep the puzzle concise.
  src = strip_refractir_prefix(src_raw)

  # Build the tree-sitter parser once and reuse it in the self-check.
  C_LANGUAGE = Language(tsc.language(), "c")
  ts_parser = Parser()
  ts_parser.set_language(C_LANGUAGE)

  src_sanitized = sanitize_statement_expressions(src)
  tree = ts_parser.parse(src_sanitized)

  leaf_node, leaf_name = find_leaf_function(tree.root_node, src_sanitized)
  if not leaf_node:
    print("Error: Could not find leaf function in C program.", file=sys.stderr)
    sys.exit(1)

  # Validate leaf shape: must have entry and exit labels.
  maskable, entry_node, exit_node = get_maskable_statements(leaf_node, src_sanitized)
  if not entry_node or not exit_node:
    print("Error: Leaf function is missing 'entry' or 'exit' labels.", file=sys.stderr)
    sys.exit(1)

  # Collect local variable names for FILL_VAR detection.
  local_names = collect_leaf_locals(leaf_node, src_sanitized)

  # Collect all function names defined in this file for FILL_FUNC detection.
  # Only these are eligible for masking; compiler builtins and external symbols
  # are automatically excluded because they have no function_definition node.
  defined_funcs = collect_defined_functions(tree.root_node, src_sanitized)

  # Extract PATH, CFG, and checksum from the companion .sir file (if available).
  path_str = cfg_str = ""
  if sir_path:
    path_str, cfg_str, _ = extract_metadata_from_sir(sir_path)

  # --- Selective masking (--p-mask) ---
  # With p == 1.0 every maskable statement is masked.  With p < 1.0 each
  # position is independently masked with probability p; we retry up to 100
  # times to ensure at least one position is masked when p > 0.
  mask_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
  rng = random.Random(mask_seed)

  mask_set: set[int] = set()
  if args.p_mask > 0.0:
    for _attempt in range(100):
      for idx in range(len(maskable)):
        if rng.random() < args.p_mask:
          mask_set.add(idx)
      if mask_set:
        break
    if not mask_set and args.p_mask > 1e-9:
      print(
        "Error: Failed to generate a non-empty mask set after 100 attempts.",
        file=sys.stderr,
      )
      sys.exit(1)

  # --- Build replacements ---
  # DUMP_TRACE instrumentation is always included (both in puzzle and ground truth).
  trace_replacements = _build_trace_replacements(leaf_node, src_sanitized)

  budget_counts: dict[str, int] = {}
  mask_replacements = []
  for idx, stmt in enumerate(maskable):
    if idx in mask_set:
      is_body = stmt.start_byte > entry_node.start_byte
      collect_replacements(
        stmt,
        src_sanitized,
        is_body,
        mask_replacements,
        budget_counts,
        local_names,
        defined_funcs,
      )

  puzzle_replacements = trace_replacements + mask_replacements
  puzzle_body = apply_replacements(src, puzzle_replacements).decode("utf-8")

  # Ground truth: only trace instrumentation, no masking.
  gt_body = apply_replacements(src, trace_replacements).decode("utf-8")

  # --- Self-check ---
  # Re-masking the ground truth must reproduce the puzzle body exactly.
  if not self_check_puzzle(puzzle_body, gt_body, mask_set, ts_parser, defined_funcs):
    sys.exit(1)

  # --- Build FILL_CONST budget lines ---
  fill_const_lines = "".join(
    f"//@ FILL_CONST: {val} {cnt}\n"
    for val in sorted(budget_counts)
    for cnt in [budget_counts[val]]
  )

  if args.lift_consts:
    budget_section = ""
  else:
    budget_section = BUDGET_SECTION_TEMPLATE.replace("{{FILL_CONST}}", fill_const_lines)

  # --- Render header ---
  header = (
    PUZZLE_HEADER_TEMPLATE.replace("{{LEAF_NAME}}", leaf_name)
    .replace("{{CFG}}", cfg_str if cfg_str else "//   [unknown CFG]\n")
    .replace("{{PATH}}", path_str if path_str else "[unknown]")
    .replace("{{BUDGET_SECTION}}", budget_section)
  )

  # --- Write output ---
  if args.output:
    with open(args.output, "w") as f:
      f.write(header + puzzle_body)
    if args.keep_ground_truth:
      # Derive ground-truth path by replacing the final extension with .gt.c.
      base, ext = os.path.splitext(args.output)
      gt_path = base + ".gt" + ext if ext else args.output + ".gt.c"
      with open(gt_path, "w") as f:
        f.write(gt_body)
  else:
    sys.stdout.write(header + puzzle_body)


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="C Puzzle Creator — masks a rysmith-generated C function into a fill-in-the-blanks puzzle.",
  )
  p.add_argument(
    "input",
    nargs="?",
    help="Optional concrete .c (or .sir with a sibling .c) file to mask instead of generating one.",
  )
  p.add_argument("-o", "--output", help="Output puzzle file path (default: stdout).")

  # Difficulty knobs forwarded to rysmith.
  p.add_argument(
    "-L",
    "--min-loop-iter",
    type=int,
    default=2,
    help="Minimum loop iterations constraint for rysmith (default: 2).",
  )
  p.add_argument(
    "-B",
    "--n-bbls",
    type=int,
    default=5,
    help="Number of basic blocks for rysmith (default: 5).",
  )
  p.add_argument(
    "-S",
    "--n-stmts",
    type=int,
    default=3,
    help="Number of statements per block on path for rysmith (default: 3).",
  )
  p.add_argument(
    "-P",
    "--p-mask",
    type=float,
    default=1.0,
    help="Probability in [0,1] that each maskable statement is masked (default: 1.0).",
  )
  p.add_argument(
    "-C",
    "--lift-consts",
    action="store_true",
    help="Omit the FILL_CONST budget section (no magic-number constraints).",
  )

  # Other options.
  p.add_argument(
    "--keep-ground-truth",
    action="store_true",
    help="Save the unmasked ground-truth C file as <output>.gt.c.",
  )
  p.add_argument(
    "--rysmith",
    default=f"{PWD}/rysmith",
    help="Path to the rysmith binary (default: ./rysmith).",
  )
  p.add_argument(
    "--seed",
    type=int,
    help="Master seed: drives rysmith generation and --p-mask coin flips.",
  )
  return p


def main() -> None:
  parser = build_arg_parser()
  args = parser.parse_args()

  if args.p_mask < 0.0 or args.p_mask > 1.0:
    print("Error: --p-mask must be in [0, 1].", file=sys.stderr)
    sys.exit(1)

  if args.keep_ground_truth and not args.output:
    print(
      "Error: --keep-ground-truth requires --output to be specified.", file=sys.stderr
    )
    sys.exit(1)

  if args.input:
    # Use a caller-supplied file instead of generating one.
    if args.input.endswith(".sir"):
      sir_path = args.input
      c_path = args.input[:-4] + ".c"
      if not os.path.exists(c_path):
        # Translate the SIR to C via symirc.
        subprocess.run(["./symirc", sir_path, "-o", c_path], check=True)
    elif args.input.endswith(".c"):
      c_path = args.input
      pot_sir = args.input[:-2] + ".sir"
      sir_path = pot_sir if os.path.exists(pot_sir) else None
    else:
      print(f"Error: Unknown input file extension for '{args.input}'.", file=sys.stderr)
      sys.exit(1)
    create_puzzle(c_path, sir_path, args)
  else:
    # Generate a fresh C program via rysmith.
    run_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    with tempfile.TemporaryDirectory(prefix=f"rypuz_tmp_{run_seed}_") as tmp_dir:
      c_path, sir_path = run_rysmith_loop(args, run_seed, tmp_dir)
      create_puzzle(c_path, sir_path, args)


if __name__ == "__main__":
  main()

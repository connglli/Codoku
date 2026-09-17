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

import ast
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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
  print(f"  [{color}{tag}{NC}] {name}" + (f" ({detail})" if detail and not ok else ""))


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


def import_codoku_common(codoku_path):
  """Import codoku_common as a module (for direct unit tests)."""
  module_dir = os.path.dirname(os.path.realpath(codoku_path))
  if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
  cfile = os.path.join(module_dir, "codoku_common.py")
  spec = importlib.util.spec_from_file_location("_codoku_common_mod", cfile)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def import_codoku_complexity(codoku_path):
  """Import codoku_complexity as a module (for direct unit tests)."""
  module_dir = os.path.dirname(os.path.realpath(codoku_path))
  if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
  cfile = os.path.join(module_dir, "codoku_complexity.py")
  spec = importlib.util.spec_from_file_location("_codoku_cplx_mod", cfile)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def complexity_unit_tests_pass(cmod) -> bool:
  ok = True

  # Halstead vocabulary on `def f(a, b): c = a + b; return c`:
  # operators {FunctionDef, arguments, Assign, BinOp, Add, Return},
  # operands {a x2, b x2, c x2}.
  leaf = ast.parse("def f(a, b):\n    c = a + b\n    return c\n").body[0]
  h = cmod.compute_halstead(leaf)
  good = (
    (h.operators, h.operands, h.total_operators, h.total_operands) == (6, 3, 6, 6)
    and h.vocabulary == 9
    and h.length == 12
    and h.volume == round(12 * math.log2(9), 2)
    and h.difficulty == 6.0
    and h.effort == round(6.0 * h.volume, 2)
  )
  check("unit: halstead counts (no time/defects)", good, str(h))
  ok = ok and good

  # DepDegree: params -> stmt -> return; the two param uses open one pair
  # edge, plus one edge for c.
  d = cmod.compute_depdegree(leaf)
  good = (
    (d.nodes, d.edges) == (3, 2)
    and d.avg_degree == round(2 * 2 / 3, 2)
    and d.max_degree == 2
    and d.density == round(2 / (3 * 2), 4)
  )
  check("unit: depdegree counts", good, str(d))
  ok = ok and good

  # McCabe: straight line is 1, if/else diamond is 2, empty is 0.
  m_line = cmod.compute_mccabe({"a", "b", "c"}, [("a", "b"), ("b", "c")])
  m_diamond = cmod.compute_mccabe(
    {"entry", "b0", "b1", "exit"},
    [("entry", "b0"), ("entry", "b1"), ("b0", "exit"), ("b1", "exit")],
  )
  m_empty = cmod.compute_mccabe(set(), [])
  good = (
    (m_line.nodes, m_line.edges, m_line.cyclomatic) == (3, 2, 1)
    and (m_diamond.nodes, m_diamond.edges, m_diamond.cyclomatic) == (4, 4, 2)
    and (m_empty.nodes, m_empty.edges, m_empty.cyclomatic) == (0, 0, 0)
    and cmod.McCabeMetrics.zero() == m_empty
  )
  check("unit: mccabe counts", good, f"{m_line} {m_diamond} {m_empty}")
  ok = ok and good

  # Estimated time (E/18) and defects (V/3000) are deliberately absent.
  import dataclasses

  hal_fields = {f.name for f in dataclasses.fields(cmod.HalsteadMetrics)}
  est_fields = {f.name for f in dataclasses.fields(cmod.ComplexityEstimate)}
  good = not any(
    "time" in f or "bug" in f or "defect" in f for f in hal_fields | est_fields
  )
  check("unit: no time/defect heuristics", good, str(sorted(hal_fields | est_fields)))
  ok = ok and good

  # Vocab + dataflow are static terms inside static_struct (no separate axes).
  metrics = cmod.PuzzleMetrics(
    cfg_nodes=2,
    cfg_edges=1,
    cyclomatic=1,
    n_loops=0,
    exec_path_length=2,
    unique_path_blocks=2,
    repeated_block_visits=0,
    max_block_visits=1,
    loop_iterations_total=0,
    loop_iterations_avg=0.0,
    total_masks=0,
    masks_by_kind={},
    sol_space_log10=0.0,
    const_budget_entries=0,
    const_budget_total=0,
    source_lines=5,
    non_comment_source_lines=3,
    hal_operators=6,
    hal_operands=3,
    hal_total_operators=6,
    hal_total_operands=6,
    hal_vocabulary=9,
    hal_length=12,
    hal_volume=38.04,
    hal_difficulty=6.0,
    hal_effort=228.24,
    dep_nodes=3,
    dep_edges=2,
    dep_avg_degree=1.33,
    dep_max_degree=2,
    dep_density=0.3333,
  )
  # Vocab + dataflow fold into static_struct (no separate axes); total is
  # the four-axis sum.
  est = cmod.estimate_complexity(metrics)
  parts = est.static_struct + est.dynamic_trace + est.masking + est.constraints
  good = (
    not hasattr(est, "vocab")
    and not hasattr(est, "dataflow")
    and abs(
      est.static_struct - (2 + 1 + 2 + 0.15 + 0.01 * 38.04 + 0.5 * 6.0 + 2.0 * 1.33 + 2)
    )
    < 0.01
    and abs(est.total - parts) < 1e-9
  )
  check("unit: vocab/dataflow folded into static", good, str(est))
  ok = ok and good

  # Unparsable input measures zero instead of raising.
  tmpdir = tempfile.mkdtemp(prefix="codoku_cplx_edge_")
  try:
    bad = os.path.join(tmpdir, "puzzle.py")
    with open(bad, "w") as f:
      f.write("def broken(:\n  <FILL_VAR> ???\n")
    from pathlib import Path

    edge_metrics = cmod.analyze_puzzle(Path(bad))
    edge_est = cmod.estimate_complexity(edge_metrics)
    edge_parts = (
      edge_est.static_struct
      + edge_est.dynamic_trace
      + edge_est.masking
      + edge_est.constraints
    )
    good = (
      edge_metrics.hal_volume == 0.0
      and edge_metrics.dep_nodes == 0
      and abs(edge_est.total - edge_parts) < 1e-9
    )
    check("unit: unparsable puzzle measures zero", good, str(edge_metrics))
    ok = ok and good
  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

  # Vocab/dataflow come from the ground truth when available: the explicit
  # gt_path, else the oracle sibling.  Markers still come from the puzzle.
  gt_body = "def func_tiny(a, b):\n    c = a + b\n    d = c * 2\n    return d\n"
  masked_body = (
    "#//@ CFG_EDGE: entry -> exit\n"
    "#//@ EXEC_PATH: entry -> exit\n"
    "def func_tiny(a, b):\n"
    "    <FILL_VAR> = a + <FILL_CONST>\n"
    "    <FILL_VAR> = <FILL_VAR> <FILL_OP> <FILL_CONST>\n"
    "    return <FILL_VAR>\n"
  )
  gt_dir = tempfile.mkdtemp(prefix="codoku_cplx_gt_")
  try:
    from pathlib import Path

    outdir = Path(gt_dir) / "out"
    (outdir / "oracle").mkdir(parents=True)
    puzzle_path = outdir / "puzzle.py"
    puzzle_path.write_text(masked_body)
    gt_path = outdir / "oracle" / "puzzle.gt.py"
    gt_path.write_text(gt_body)
    met_gt = cmod.analyze_puzzle(gt_path)
    met_explicit = cmod.analyze_puzzle(puzzle_path, gt_path=gt_path)
    met_found = cmod.analyze_puzzle(puzzle_path)
    good = (
      met_explicit.hal_volume == met_gt.hal_volume
      and (met_explicit.dep_nodes, met_explicit.dep_edges)
      == (met_gt.dep_nodes, met_gt.dep_edges)
      and met_found.hal_volume == met_gt.hal_volume
      and met_found.dep_max_degree == met_gt.dep_max_degree
      and met_found.cfg_nodes == 2
      and met_found.exec_path_length == 2
      and met_found.total_masks > 0
    )
    check("unit: vocab/dataflow measured on GT", good, str(met_found))
    ok = ok and good

    # The vocab source is visible, not silent: text names the GT file (or
    # says fallback), --json carries it, and --ground-truth names it.
    import io
    import json as json_lib
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
      rc_text = cmod.main([str(puzzle_path)])
    text_out = buf.getvalue()
    buf = io.StringIO()
    with redirect_stdout(buf):
      rc_json = cmod.main([str(puzzle_path), "--json"])
    payload = json_lib.loads(buf.getvalue())
    elsewhere = outdir / "elsewhere_gt.py"
    elsewhere.write_text(gt_body)
    buf = io.StringIO()
    with redirect_stdout(buf):
      rc_flag = cmod.main([str(puzzle_path), "--ground-truth", str(elsewhere)])
    # The flag wins even with different content: bare puzzle, no sibling.
    bare_dir = Path(gt_dir) / "bare"
    bare_dir.mkdir()
    lonely = bare_dir / "lonely.py"
    lonely.write_text(masked_body)
    other_gt = bare_dir / "other.py"
    other_gt.write_text(gt_body.replace("c * 2", "c * 2 + a - b"))
    met_flagged = cmod.analyze_puzzle(lonely, gt_path=other_gt)
    met_other = cmod.analyze_puzzle(other_gt)
    good = (
      rc_text == 0
      and str(gt_path) in text_out
      and rc_json == 0
      and payload.get("vocab_source") == str(gt_path)
      and rc_flag == 0
      and cmod.resolve_vocab_source(puzzle_path) == gt_path
      and cmod.resolve_vocab_source(lonely) is None
      and met_flagged.hal_volume == met_other.hal_volume
      and met_flagged.hal_volume != met_gt.hal_volume
    )
    check("unit: vocab source visible + --ground-truth", good, text_out[-200:])
    ok = ok and good

    # Review fixes: trace scaffolding is not puzzle logic, subscript
    # AugAssign redefines the base, and an unusable explicit GT is an error.
    trace_if = (
      '    if __import__("os").environ.get("DUMP_TRACE"):\n        print("^b0:")\n'
    )
    traced_src = "def f(a, b):\n" + trace_if + "    c = a + b\n    return c\n"
    traced_leaf = ast.parse(traced_src).body[0]
    h_traced = cmod.compute_halstead(traced_leaf)
    d_traced = cmod.compute_depdegree(traced_leaf)
    hal_ok = (
      h_traced.operators,
      h_traced.operands,
      h_traced.total_operators,
      h_traced.total_operands,
    ) == (6, 3, 6, 6)
    dep_ok = (d_traced.nodes, d_traced.edges) == (3, 2)
    good = hal_ok and dep_ok
    check("unit: trace scaffolding excluded", good, f"{h_traced} {d_traced}")
    ok = ok and good

    collector = cmod._DepCollector()
    collector.visit(ast.parse("a[i] += v\n").body[0])
    (a_defs, a_uses), *_ = collector.stmts
    good = a_defs == {"a"} and {"i", "v", "a"} <= a_uses
    check("unit: augassign subscript redefines base", good, f"{a_defs} {a_uses}")
    ok = ok and good

    from contextlib import redirect_stderr

    err = io.StringIO()
    with redirect_stderr(err):
      rc_bad = cmod.main([str(puzzle_path), "--ground-truth", str(outdir / "nope.py")])
    good = rc_bad == 2 and "ground-truth" in err.getvalue().lower()
    check("unit: unusable --ground-truth errors", good, err.getvalue()[-200:])
    ok = ok and good

    # A masked ternary's `else` keyword is `<FILL_OP>`, so the
    # placeholder-demasked text loses the ternary and cannot parse.  The
    # vocabulary measurement falls back to the ground-truth tree instead
    # of collapsing the solution space to -inf.
    invalid_demask = (
      "def func_t(a, b):\n"
      "    c = a if a <FILL_OP> 1 <FILL_OP> b\n"
      "    return <FILL_VAR>\n"
    )
    puzzle_bad = outdir / "puzzleBad.py"
    gt_bad = outdir / "oracle" / "puzzleBad.gt.py"
    puzzle_bad.write_text(invalid_demask)
    gt_bad.write_text(gt_body.replace("c = a + b", "c = a if a < 1 else b"))
    met_bad = cmod.analyze_puzzle(puzzle_bad, gt_path=gt_bad)
    good = met_bad.sol_space_log10 != float("-inf") and met_bad.total_masks > 0
    check(
      "unit: masked ternary keeps the sol space finite",
      good,
      str(met_bad.sol_space_log10),
    )
    ok = ok and good
  finally:
    shutil.rmtree(gt_dir, ignore_errors=True)

  return ok


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

  # Banner extraction is fail-closed: a .sir missing the trusted
  # comments raises instead of degrading the puzzle banner to [unknown],
  # and a present `// CFG:` header with no edges is legitimate.
  from pathlib import Path

  with tempfile.TemporaryDirectory(prefix="codoku_banner_") as banner_dir:
    bfd = Path(banner_dir)
    cfg_only = bfd / "cfg_only.sir"
    cfg_only.write_text("// CFG:\n//   entry -> exit\n")
    bare = bfd / "bare.sir"
    bare.write_text("def func_t():\n    return 0\n")

    def raises(fn, arg):
      try:
        fn(arg)
      except RuntimeError:
        return True
      return False

    closed = (
      raises(mod.extract_path_from_sir, cfg_only)
      and raises(mod.extract_path_from_sir, bare)
      and raises(mod.extract_cfg_from_sir, bare)
      and mod.extract_cfg_from_sir(cfg_only) == [("entry", "exit")]
    )
    check("unit: banner extraction is fail-closed", closed)

  return ok


def goto_flag_unit_tests_pass(ccommon, chk_mod) -> bool:
  """Structured-goto flags (_go_/_brk_/_cnt_) must all be masked by target-hiding."""
  ok = True

  # (a) Target decoding covers every flag kind and rejects ordinary names.
  cases = {
    "_go_exit": "exit",
    "_go_b6": "b6",
    "_brk_exit": "exit",
    "_brk_b2": "b2",
    "_break_exit": "exit",
    "_cnt_b0": "b0",
    "_continue_b0": "b0",
    "acc": None,
    "_brk": None,
    "_cnt_": "",
  }
  ok_a = True
  for name, want in cases.items():
    got = ccommon.goto_flag_target(name)
    if got != want:
      check(f"unit: goto_flag_target({name})", False, f"got {got!r}, want {want!r}")
      ok_a = False
      ok = False
  if ok_a:
    check("unit: goto_flag_target decodes all flag kinds", True)

  # (b) The maskable-statement predicate fires for every flag spelling.
  mod_src = (
    "def func_test(x):\n"
    "    _brk_exit = False\n"
    "    _cnt_b0 = False\n"
    "    acc = 0\n"
    "    # ^entry\n"
    "    acc = (x + 1)\n"
    "    while True:\n"
    "        # ^b0\n"
    "        acc = (acc + x)\n"
    "        if (acc > 10):\n"
    "            _brk_exit = True\n"
    "            break\n"
    "        _cnt_b0 = True\n"
    "        break\n"
    "    if _brk_exit:\n"
    "        _brk_exit = False\n"
    "        acc = (acc + 1)\n"
    "    if _cnt_b0:\n"
    "        _cnt_b0 = False\n"
    "        acc = (acc + 2)\n"
    "    # ^exit\n"
    "    return acc\n"
  )
  mod_bytes = mod_src.encode("utf-8")
  tree = ast.parse(mod_bytes)
  leaf, _ = ccommon.find_python_leaf_function(tree, mod_bytes)
  if leaf is None:
    check("unit: flag fixture has a leaf function", False, "no leaf found")
    return False
  maskable, entry_line, _ = ccommon.get_python_maskable_statements(leaf, mod_bytes)
  sees_brk = any(ccommon.mentions_go_flag(s) for s in maskable)
  if not sees_brk:
    check("unit: maskable scan sees _brk_/_cnt_ flags", False, "no flagged stmt")
    ok = False
  else:
    check("unit: maskable scan sees _brk_/_cnt_ flags", True)
  plain = ast.parse(b"def func_test(x):\n    y = (x + 1)\n    return y\n")
  if ccommon.mentions_go_flag(plain):
    check("unit: plain code has no goto flags", False, "false positive")
    ok = False
  else:
    check("unit: plain code has no goto flags", True)

  # (c) Masking hides every flag target; _brk_/_cnt_ also hide the kind.
  local_names = ccommon.collect_python_leaf_locals(leaf)
  defined_funcs = set()
  repls: list = []
  for stmt in maskable:
    ccommon.collect_python_replacements(
      stmt, mod_bytes, stmt.lineno > entry_line, repls, local_names, defined_funcs
    )
  masked = ccommon.apply_replacements(mod_bytes, repls).decode("utf-8")
  leaked = sorted(
    set(re.findall(r"_(?:go|brk|cnt)_[A-Za-z0-9]+", masked))
    - {"_go_<FILL_LABEL>", "_<FILL_CTRL>_<FILL_LABEL>"}
  )
  if leaked:
    check("unit: no concrete flag target survives masking", False, str(leaked))
    ok = False
  else:
    check("unit: no concrete flag target survives masking", True)
  if "_<FILL_CTRL>_<FILL_LABEL>" not in masked:
    check("unit: brk/cnt flags hide kind and target", False, "missing compound form")
    ok = False
  else:
    check("unit: brk/cnt flags hide kind and target", True)

  # (c1) Full-word spellings mask identically (the checker accepts both).
  ok_words = True
  for spelled in ("_break_exit", "_continue_b0"):
    node = ast.parse(spelled, mode="eval").body
    src_bytes = spelled.encode("utf-8")
    word_repls: list = []
    ccommon.collect_python_replacements(node, src_bytes, True, word_repls, set(), set())
    word_masked = ccommon.apply_replacements(src_bytes, word_repls).decode("utf-8")
    if word_masked != "_<FILL_CTRL>_<FILL_LABEL>":
      check(f"unit: {spelled} masks to compound token", False, word_masked)
      ok_words = False
      ok = False
  if ok_words:
    check("unit: full-word flag spellings mask identically", True)

  # (c2) _go_ keeps its kind prefix (a general goto pairs with no keyword).
  go_src = (
    "def func_g(x):\n"
    "    _go_b1 = False\n"
    "    # ^entry\n"
    "    _go_b1 = True\n"
    "    # ^b0\n"
    "    y = (x + 1)\n"
    "    # ^b1\n"
    "    y = (y + 2)\n"
    "    # ^exit\n"
    "    return y\n"
  ).encode("utf-8")
  go_tree = ast.parse(go_src)
  go_leaf, _ = ccommon.find_python_leaf_function(go_tree, go_src)
  go_maskable, go_entry, _ = ccommon.get_python_maskable_statements(go_leaf, go_src)
  go_locals = ccommon.collect_python_leaf_locals(go_leaf)
  go_repls: list = []
  for stmt in go_maskable:
    ccommon.collect_python_replacements(
      stmt, go_src, stmt.lineno > go_entry, go_repls, go_locals, set()
    )
  go_masked = ccommon.apply_replacements(go_src, go_repls).decode("utf-8")
  if "_go_<FILL_LABEL>" not in go_masked or "_<FILL_CTRL>_<FILL_LABEL>" in go_masked:
    check("unit: go flags keep their kind prefix", False, go_masked)
    ok = False
  else:
    check("unit: go flags keep their kind prefix", True)

  # (c3) Exit-block plumbing: guard test, reset and keyword all maskable.
  exit_src = (
    "def func_e(x):\n"
    "    # ^entry\n"
    "    y = (x + 1)\n"
    "    while True:\n"
    "        # ^b0\n"
    "        y = (y + x)\n"
    "        if (y > 10):\n"
    "            # ^exit\n"
    "            y = (y + 1)\n"
    "            return y\n"
    "        if _cnt_b0:\n"
    "            _cnt_b0 = False\n"
    "            continue\n"
    "    return y\n"
  ).encode("utf-8")
  exit_tree = ast.parse(exit_src)
  exit_leaf, _ = ccommon.find_python_leaf_function(exit_tree, exit_src)
  exit_maskable, exit_entry, _ = ccommon.get_python_maskable_statements(
    exit_leaf, exit_src
  )
  exit_locals = ccommon.collect_python_leaf_locals(exit_leaf)
  exit_repls: list = []
  for stmt in exit_maskable:
    ccommon.collect_python_replacements(
      stmt, exit_src, stmt.lineno > exit_entry, exit_repls, exit_locals, set()
    )
  exit_masked = ccommon.apply_replacements(exit_src, exit_repls).decode("utf-8")
  exit_leaks = sorted(
    set(re.findall(r"_(?:go|brk|cnt)_[A-Za-z0-9]+", exit_masked))
    - {"_go_<FILL_LABEL>", "_<FILL_CTRL>_<FILL_LABEL>"}
  )
  if exit_leaks or exit_masked.count("_<FILL_CTRL>_<FILL_LABEL>") != 2:
    check("unit: exit-guard reset masked with its guard", False, exit_masked)
    ok = False
  else:
    check("unit: exit-guard reset masked with its guard", True)
  if "<FILL_CTRL>" not in exit_masked:
    check("unit: exit-guard keyword masked", False, exit_masked)
    ok = False
  else:
    check("unit: exit-guard keyword masked", True)

  # (c4) Flag setter in the exit region: an `else:` setter outside body
  # blocks must mask too, or it answers the dispatch guard.
  setter_src = (
    "def func_s(x):\n"
    "    # ^entry\n"
    "    y = (x + 1)\n"
    "    while True:\n"
    "        # ^b0\n"
    "        y = (y + x)\n"
    "        if (y > 10):\n"
    "            # ^exit\n"
    "            y = (y + 1)\n"
    "            return y\n"
    "        else:\n"
    "            _cnt_b0 = True\n"
    "            break\n"
    "    return y\n"
  ).encode("utf-8")
  setter_tree = ast.parse(setter_src)
  setter_leaf, _ = ccommon.find_python_leaf_function(setter_tree, setter_src)
  setter_maskable, setter_entry, _ = ccommon.get_python_maskable_statements(
    setter_leaf, setter_src
  )
  setter_locals = ccommon.collect_python_leaf_locals(setter_leaf)
  setter_repls: list = []
  for stmt in setter_maskable:
    ccommon.collect_python_replacements(
      stmt,
      setter_src,
      stmt.lineno > setter_entry,
      setter_repls,
      setter_locals,
      set(),
    )
  setter_masked = ccommon.apply_replacements(setter_src, setter_repls).decode("utf-8")
  setter_leaks = sorted(
    set(re.findall(r"_(?:go|brk|cnt)_[A-Za-z0-9]+", setter_masked))
    - {"_go_<FILL_LABEL>", "_<FILL_CTRL>_<FILL_LABEL>"}
  )
  if setter_leaks or "_<FILL_CTRL>_<FILL_LABEL>" not in setter_masked:
    check("unit: exit-region setter masked", False, setter_masked)
    ok = False
  else:
    check("unit: exit-region setter masked", True)

  # (e) Dispatch transfer equality: a guard must land where its flag names.
  transfer_src = (
    "def func_d(x):\n"
    "    # ^entry\n"
    "    y = (x + 1)\n"
    "    while True:\n"
    "        # ^b0\n"
    "        y = (y + x)\n"
    "        if _brk_exit:\n"
    "            _brk_exit = False\n"
    "            break\n"
    "        if not _cnt_b0:\n"
    "            y = (y + 2)\n"
    "        else:\n"
    "            _cnt_b0 = False\n"
    "            continue\n"
    "        if _go_exit:\n"
    "            break\n"
    "    # ^exit\n"
    "    return y\n"
  ).encode("utf-8")
  transfer_tree = ast.parse(transfer_src)
  transfer_leaf, _ = ccommon.find_python_leaf_function(transfer_tree, transfer_src)
  dispatches = ccommon.iter_flag_dispatches(transfer_leaf, transfer_src)
  got = sorted((f, e, s) for f, e, s, _ in dispatches)
  want = sorted(
    [
      ("_brk_exit", "exit", "exit"),
      ("_cnt_b0", "b0", "b0"),
      ("_go_exit", "exit", "exit"),
    ]
  )
  if got != want:
    check("unit: dispatch transfers follow the loop model", False, str(got))
    ok = False
  else:
    check("unit: dispatch transfers follow the loop model", True)

  # (e2) End to end: a mistargeted dispatch fails check_cfg with FAIL_CFG
  # while the well-targeted original passes. The fixtures below are real
  # lowered shapes (see MULTI_LEVEL_SRC / SELF_LOOP_SRC), so their flag
  # targets are declared by the SIR-projected edges.
  multi_src = MULTI_LEVEL_SRC.encode("utf-8")
  multi_leaf, _ = ccommon.find_python_leaf_function(ast.parse(multi_src), multi_src)
  actual_edges = sorted(ccommon.build_python_cfg(multi_leaf, multi_src))
  bad_transfer = MULTI_LEVEL_SRC.replace("_brk_exit", "_brk_b0").encode("utf-8")
  bad_leaf, _ = ccommon.find_python_leaf_function(ast.parse(bad_transfer), bad_transfer)
  bad_edges = sorted(ccommon.build_python_cfg(bad_leaf, bad_transfer))
  for src, edges, want_ok in (
    (multi_src, actual_edges, True),
    (bad_transfer, bad_edges, False),
  ):
    t = ast.parse(src)
    node = t.body[0]
    label = "accepts true flag target" if want_ok else "rejects mistargeted flag"
    try:
      chk_mod.check_cfg(node, src, edges)
      passed = want_ok
    except chk_mod.CheckFailure as exc:
      passed = (not want_ok) and exc.result == chk_mod.CheckResult.FAIL_CFG
      if not want_ok and "claims target" not in str(exc.message):
        passed = False
    except Exception as exc:  # noqa: BLE001 - any other error is a failure
      check(f"unit: checker {label}", False, str(exc))
      ok = False
      continue
    check(f"unit: checker {label}", passed)
    ok = ok and passed
  self_src = SELF_LOOP_SRC.encode("utf-8")
  good_src, good_edges = self_src, SELF_LOOP_EDGES
  bad_src = SELF_LOOP_SRC.replace("_brk_exit", "_brk_nope").encode("utf-8")
  for src, edges, want_ok in (
    (good_src, good_edges, True),
    (bad_src, good_edges, False),
  ):
    t = ast.parse(src)
    node = t.body[0]
    label = (
      "accepts declared flag targets" if want_ok else "rejects unknown flag target"
    )
    try:
      chk_mod.check_cfg(node, src, edges)
      passed = want_ok
    except chk_mod.CheckFailure as exc:
      passed = (not want_ok) and exc.result == chk_mod.CheckResult.FAIL_CFG
    except Exception as exc:  # noqa: BLE001 - any other error is a failure
      check(f"unit: checker {label}", False, str(exc))
      ok = False
      continue
    check(f"unit: checker {label}", passed)
    ok = ok and passed

  return ok


MULTI_LEVEL_SRC = """\
def func_493ad2_0(pa0, pa1):
    _brk_exit = False
    _cnt_entry = False
    v0 = 16325
    v1 = 9223372036854775807
    v2 = 2147483646
    v__chk = 0
    while True:
        # ^entry
        v2 = (-1310722 + _cast_int(pa0, 32))
        v2 = ((536608768 - 536915912) + _cast_int(v0, 32))
        if (((_cast_int(536870912, 64) - 1) - (0 if (v0) <= (0) else 1073741829))) < ((_cast_int(1610612715, 64) + (-2147483639 // v1 if (-2147483639 < 0) == (v1 < 0) else -(--2147483639 // v1)))):
            while True:
                # ^b0
                v2 = (-1300407837 if (pa0) == (0) else 1301556872)
                v1 = ((_cast_int(1073747998, 64) - -1073747999) + _cast_int(v0, 64))
                # ^b1
                v2 = ((_cast_int(v0, 32) + _cast_int(pa1, 32)) - _cast_int(v1, 32))
                v1 = (_cast_int(16326, 64) - _cast_int(v0, 64))
                if (((pa0 - (-524286 // pa0 if (-524286 < 0) == (pa0 < 0) else -(--524286 // pa0))) - (0 * pa1))) >= (((_cast_int(524287, 20) - (-1 & pa0)) - (-524287 ^ pa0))):
                    # ^b2
                    v2 = ((_cast_int(pa1, 32) - -551647243) - -264371443)
                    v0 = (((_cast_int(v2, 16) - -15143) - -32566) + -9771)
                    v1 = (_cast_int(pa0, 64) - 4517238977076481430)
                    v2 = (((((_cast_int(pa0, 32) + -2084803915) + 1569701865) + 302539934) - -774380226) + -1630638020)
                    if ((((v2 + (1714810848 << v2)) + -213193905) + (85069559 * v2))) >= (((v2 - v2) - (1277168334 if (pa0) >= (227482) else v2))):
                        # ^b3
                        v0 = (_cast_int(v2, 16) - 19976)
                        v2 = (((((_cast_int(pa0, 32) - -1906059922) - 1312907534) - -1141817475) - 1738823502) + 2011160631)
                        v0 = ((_cast_int(pa0, 16) - -16435) - 261)
                        v1 = ((((_cast_int(v0, 64) + 5885115908715253784) - 5825851685144504661) + 549766622887299018) - 5896690118441534040)
                        if (((((v1 - -8567497505432704657) + 6840167212111924233) + v1) - -7955833527755308626)) != ((((v1 + (-1109064034639631911 ^ v1)) + (-2819723110373104754 & v1)) - (v1 * v1))):
                            # ^b4
                            v1 = (((((_cast_int(pa0, 64) - 4621200110528184748) + 1466042252710198724) + -5731660007434343443) + -3189424937736278487) + 5184697247205444140)
                            v2 = (((_cast_int(pa1, 32) - 2042850073) + 1657856511) + -1301767476)
                            v0 = (_cast_int(pa1, 16) + -31966)
                            v2 = (((_cast_int(v0, 32) - -425826940) - -758129501) - 1849844542)
                            _brk_exit = True
                            break
                        else:
                            _brk_exit = True
                            break
                else:
                    _cnt_entry = True
                    break
            if _brk_exit:
                _brk_exit = False
                break
            if _brk_exit:
                _brk_exit = False
                break
            if _cnt_entry:
                _cnt_entry = False
                continue
        else:
            break
    # ^exit
    v__chk = 0
    v__chk = (_cast_int(v0, 32) + v__chk)
    v__chk = (_cast_int(v1, 32) + v__chk)
    v__chk = (v__chk + v2)
    v__chk = (_cast_int(pa0, 32) + v__chk)
    v__chk = (_cast_int(pa1, 32) + v__chk)
    return v__chk

"""
MULTI_LEVEL_EDGES = [
  ("b0", "b1"),
  ("b1", "b2"),
  ("b1", "entry"),
  ("b2", "b0"),
  ("b2", "b3"),
  ("b3", "b4"),
  ("b3", "exit"),
  ("b4", "exit"),
  ("entry", "b0"),
  ("entry", "exit"),
]

GO_CHAIN_SRC = """\
def func_277737_0(pa0, pa1):
    _go_b4 = False
    v0 = 1390183470
    v1 = -3376710
    a0 = [25, _PAD, _PAD, _PAD, _PAD, _PAD, _PAD, _PAD]
    v__chk = 0
    while True:
        # ^entry
        a0[0] = (_cast_int(2147483647, 64) - _cast_int(v0, 64))
        v0 = _cast_int(pa1, 32)
        if (((_cast_int(-1048575, 24) + 4194304) - (~v1))) >= ((_cast_int(-230978, 24) + (3376709 ^ v1))):
            # ^b0
            v0 = ((-1074790383 + 1083178992) + _cast_int(v1, 32))
            v1 = (_cast_int(4194303, 24) - (7832113 if (pa1) == (0) else -64))
            if (((_cast_int(-1, 16) + 0) - (-1 * pa0))) == (((pa0 - (~pa0)) + 32766)):
                # ^b1
                v0 = ((_cast_int(pa1, 32) - -1000007290) - 180518944)
                a0[0] = (_cast_int(pa0, 64) + 4779544505478012505)
                a0[0] = (((_cast_int(v0, 64) + 5899122092922097481) - -1804417916858868272) + -6012549381708219711)
                a0[0] = (((((_cast_int(pa0, 64) + -4271742685527139969) + -5479604327715065043) - -5494721611445278071) + -8742055666264067646) + -3668418034450071883)
                if not (((((pa0 - 16650) - -10913) - (pa0 * pa0))) >= ((((((pa0 - (32658 if (pa0) < (27043) else 12537)) - (-29464 | pa0)) - -8312) + (30364 * pa0)) - 5588))):
                    _go_b4 = True
            else:
                _go_b4 = True
        if not _go_b4 and not _go_b4:
            # ^b2
            a0[0] = ((_cast_int(-1, 64) + _cast_int(pa1, 64)) + _cast_int(pa1, 64))
            a0[0] = ((_cast_int(0, 64) - -1688389) + _cast_int(v1, 64))
            # ^b3
            v1 = ((_cast_int(-3145728, 24) - _cast_int(pa0, 24)) + -5242880)
            v0 = (8388608 + _cast_int(v1, 32))
        _go_b4 = False
        _go_b4 = False
        # ^b4
        v1 = ((_cast_int(-4194306, 24) + 0) - (-1 if (pa1) >= (0) else -1))
        a0[0] = (_cast_int(4194304, 64) - _cast_int(v0, 64))
        if (((~v0) - (-2 * v0))) < (((-2145386496 | v0) + (2141192193 ^ v0))):
            # ^exit
            v__chk = 0
            v__chk = (v__chk + v0)
            v__chk = (_cast_int(v1, 32) + v__chk)
            v__chk = (_cast_int(_rd(a0, 0), 32) + v__chk)
            v__chk = (_cast_int(pa0, 32) + v__chk)
            v__chk = (_cast_int(pa1, 32) + v__chk)
            return v__chk

"""
GO_CHAIN_EDGES = [
  ("b0", "b1"),
  ("b0", "b4"),
  ("b1", "b2"),
  ("b1", "b4"),
  ("b2", "b3"),
  ("b3", "b4"),
  ("b4", "entry"),
  ("b4", "exit"),
  ("entry", "b0"),
  ("entry", "b2"),
]

ROTATE_CONDLOOP_SRC = """\
def func_493ad2_0(pa0, pa1):
    v0 = -115
    v1 = -6162980551422932163
    v2 = -623431742
    v__chk = 0
    # ^entry
    v2 = (-1572864 + _cast_int(pa0, 32))
    v2 = ((-134217615 - 0) + _cast_int(v0, 32))
    while (((_cast_int(8, 64) - -1073741823) - (2147483645 if (v0) <= (0) else 0))) < ((_cast_int(-1073741813, 64) + (-2147483631 // v1 if (-2147483631 < 0) == (v1 < 0) else -(--2147483631 // v1)))):
        # ^b0
        v2 = (-1300407837 if (pa0) == (0) else 1301556872)
        v1 = ((_cast_int(1073741825, 64) - 1073741708) + _cast_int(v0, 64))
        if ((pa1 + (0 * pa0))) == (((pa1 + 262028) - (-109 ^ pa1))):
            # ^b1
            v1 = (((((_cast_int(v0, 64) - 3618436768347745732) + -1063236922891631481) + -118010701503943815) - -6025136141538511466) + 8406188734100277877)
            v2 = (((_cast_int(v0, 32) - 1756591453) - 1611936888) - -551647243)
            v0 = (((_cast_int(v2, 16) - -15143) - -32566) + -9771)
            v1 = (_cast_int(pa0, 64) - 4517238977076481430)
            # ^b2
            v2 = (((((_cast_int(pa0, 32) + -2084803915) + 1569701865) + 302539934) - -774380226) + -1630638020)
            v1 = (((_cast_int(pa0, 64) + 8822109915019866080) + -8223997732859483313) + -2176851898327607292)
            v1 = (((_cast_int(pa0, 64) - 7796409652611158611) - 9103323340232698618) + -4386925660137850002)
            v0 = (((_cast_int(pa0, 16) - -24711) + -16731) - 635)
            # ^b3
            v1 = (((((_cast_int(v0, 64) + -4309271748649342592) - 530665322524323784) - 7945833385774970504) - 5878441208365549374) - 8623276992307221721)
            v2 = (((_cast_int(pa0, 32) - -1743428950) - -2131161064) + 1356436797)
            v0 = ((((_cast_int(pa1, 16) - -18197) - 20067) + 18539) - 32348)
            v2 = ((((_cast_int(pa1, 32) + 926130707) - -1707575981) - -214359569) + -96858234)
            break
        # ^entry
        v2 = (-1572864 + _cast_int(pa0, 32))
        v2 = ((-134217615 - 0) + _cast_int(v0, 32))
    # ^exit
    v__chk = 0
    v__chk = (_cast_int(v0, 32) + v__chk)
    v__chk = (_cast_int(v1, 32) + v__chk)
    v__chk = (v__chk + v2)
    v__chk = (_cast_int(pa0, 32) + v__chk)
    v__chk = (_cast_int(pa1, 32) + v__chk)
    return v__chk

"""
ROTATE_CONDLOOP_EDGES = [
  ("b0", "b1"),
  ("b0", "entry"),
  ("b1", "b2"),
  ("b2", "b3"),
  ("b3", "exit"),
  ("entry", "b0"),
  ("entry", "exit"),
]


# SIR-projected CFG edges for each lowered shape above: what the fixed
# flag-aware extractor must reconstruct (multi-level continue, jump-join
# chain, rotated/condition loop).

SELF_LOOP_SRC = (
  "def func_t(pa):\n"
  "    _brk_exit = False\n"
  "    _cnt_entry = False\n"
  "    # ^entry\n"
  "    v = (pa + 1)\n"
  "    while True:\n"
  "        # ^b0\n"
  "        v = (v + 2)\n"
  "        if (v > 100):\n"
  "            # ^exit\n"
  "            v = (v + 3)\n"
  "            return v\n"
  "        _cnt_entry = False\n"
)
SELF_LOOP_EDGES = [("entry", "b0"), ("b0", "exit"), ("b0", "b0")]


def sir_extractor_tests_pass(ccommon, chk_mod) -> bool:
  """CFG_EDGE/EXEC_PATH are SIR-projected: the extractor (used at puzzle
  time as an exact-equality gate and by the checker) must reconstruct the
  lowered shapes' true SIR edges - never the mechanical break edges."""
  ok = True

  cases = [
    ("multi-level continue out two levels", MULTI_LEVEL_SRC, MULTI_LEVEL_EDGES),
    ("jump-join chain", GO_CHAIN_SRC, GO_CHAIN_EDGES),
    ("rotated/condition loop", ROTATE_CONDLOOP_SRC, ROTATE_CONDLOOP_EDGES),
    ("self-loop edge kept", SELF_LOOP_SRC, SELF_LOOP_EDGES),
  ]
  for name, src_text, want_edges in cases:
    src = src_text.encode("utf-8")
    tree = ast.parse(src)
    leaf, _ = ccommon.find_python_leaf_function(tree, src)
    got = sorted(ccommon.build_python_cfg(leaf, src))
    good = set(got) == set(want_edges)
    if not good:
      missing = sorted(set(want_edges) - set(got))
      extra = sorted(set(got) - set(want_edges))
      check(
        f"unit: extractor reconstructs {name}",
        False,
        f"missing={missing} extra={extra}",
      )
      ok = False
    else:
      check(f"unit: extractor reconstructs {name}", True)

  # A fill whose flag target renames into a *declared* but wrong block
  # must fail FAIL_CFG: the extracted CFG re-derives every flag's edge
  # from its SetFlag site per solution.
  declared = MULTI_LEVEL_EDGES
  mutated = MULTI_LEVEL_SRC.replace("_cnt_entry", "_cnt_b0")
  go_mutated = GO_CHAIN_SRC.replace("_go_b4", "_go_b3")
  for name, src_text in (
    ("unrenamed", MULTI_LEVEL_SRC),
    ("_cnt_ target renamed into a declared block", mutated),
    ("_go_ target renamed into a declared block", go_mutated),
  ):
    src = src_text.encode("utf-8")
    leaf, _ = ccommon.find_python_leaf_function(ast.parse(src), src)
    try:
      chk_mod.check_cfg(leaf, src, declared)
      check(
        f"unit: check_cfg {name}",
        name == "unrenamed",
        "" if name == "unrenamed" else "no CFG failure raised",
      )
      ok = ok and name == "unrenamed"
    except chk_mod.CheckFailure as exc:
      wrong = name == "unrenamed" or exc.result != chk_mod.CheckResult.FAIL_CFG
      check(f"unit: check_cfg {name}", not wrong, str(exc.result))
      ok = ok and not wrong
    except Exception as exc:  # noqa: BLE001
      check(f"unit: check_cfg {name}", False, str(exc))
      ok = False

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

  # Flag validation runs even when no CFG edges are declared: a puzzle
  # with no declared edges holds no goto flags, so any flag is unknown.
  flag_src = (
    "def func_t(pa):\n"
    "    _brk_exit = False\n"
    "    # ^entry\n"
    "    _brk_exit = True\n"
    "    return pa\n"
  ).encode("utf-8")
  flag_leaf = ast.parse(flag_src).body[0]
  flag_rejected = False
  flag_cause = None
  try:
    chk_mod.check_cfg(flag_leaf, flag_src, [])
  except chk_mod.CheckFailure as exc:
    flag_rejected = exc.result == chk_mod.CheckResult.FAIL_CFG
    flag_cause = exc.message
  check(
    "unit: flag rejected with empty declared set",
    flag_rejected,
    flag_cause or "no CFG failure raised",
  )
  ok = ok and flag_rejected

  return ok


def disabled_masks_unit_tests(ccommon, mod) -> bool:
  """Test that disabled_masks filters specific mask kinds from puzzle output.

  Goto-flag compound tokens (_go_<FILL_LABEL>, _<FILL_CTRL>_<FILL_LABEL>)
  never match a known mask kind, so disabling any kind preserves them.
  """
  ok = True

  src = (
    "def func_d(x):\n"
    "    # ^entry\n"
    "    y = (x + 1)\n"
    "    while True:\n"
    "        # ^b0\n"
    "        y = helper(y, 2)\n"
    "        if (y > 10):\n"
    "            # ^exit\n"
    "            return y\n"
    "        if _go_exit:\n"
    "            _go_exit = False\n"
    "            break\n"
    "        if _brk_b0:\n"
    "            _brk_b0 = False\n"
    "            break\n"
    "        if _cnt_b0:\n"
    "            _cnt_b0 = False\n"
    "            continue\n"
    "    return y\n"
  ).encode("utf-8")
  tree = ast.parse(src)
  leaf, _ = ccommon.find_python_leaf_function(tree, src)
  if leaf is None:
    check("disabled_masks: fixture has a leaf function", False, "no leaf found")
    return False
  maskable, entry_line, _ = ccommon.get_python_maskable_statements(leaf, src)
  local_names = ccommon.collect_python_leaf_locals(leaf)
  defined_funcs = {"helper"}

  all_repls: list = []
  for stmt in maskable:
    ccommon.collect_python_replacements(
      stmt,
      src,
      stmt.lineno > entry_line,
      all_repls,
      local_names,
      defined_funcs,
    )
  all_masked = ccommon.apply_replacements(src, all_repls).decode("utf-8")

  baseline_ok = (
    "<FILL_VAR>" in all_masked
    and "<FILL_CONST>" in all_masked
    and "<FILL_OP>" in all_masked
    and "<FILL_FUNC>" in all_masked
    and "<FILL_CTRL>" in all_masked
    and "_<FILL_CTRL>_<FILL_LABEL>" in all_masked
    and "_go_<FILL_LABEL>" in all_masked
  )
  check("disabled_masks: baseline has all mask kinds", baseline_ok, all_masked)
  ok = ok and baseline_ok

  def masked_with(disabled: frozenset[str]) -> str:
    repls: list = []
    for stmt in maskable:
      stmt_repls: list = []
      ccommon.collect_python_replacements(
        stmt,
        src,
        stmt.lineno > entry_line,
        stmt_repls,
        local_names,
        defined_funcs,
      )
      stmt_repls = ccommon.filter_disabled_masks(stmt_repls, disabled)
      repls.extend(stmt_repls)
    return ccommon.apply_replacements(src, repls).decode("utf-8")

  # Test 1: Disable <FILL_VAR>.
  m = masked_with(frozenset({"<FILL_VAR>"}))
  t1 = "<FILL_VAR>" not in m and "<FILL_CONST>" in m and "<FILL_OP>" in m
  check("disabled_masks: <FILL_VAR> suppressed", t1, m)
  ok = ok and t1

  # Test 2: Disable <FILL_CONST>.
  m = masked_with(frozenset({"<FILL_CONST>"}))
  t2 = "<FILL_CONST>" not in m and "<FILL_VAR>" in m and "<FILL_OP>" in m
  check("disabled_masks: <FILL_CONST> suppressed", t2, m)
  ok = ok and t2

  # Test 3: Disable <FILL_CTRL> -- standalone suppressed, compound preserved.
  m = masked_with(frozenset({"<FILL_CTRL>"}))
  standalone = m.replace("_<FILL_CTRL>_<FILL_LABEL>", "")
  t3 = (
    "<FILL_CTRL>" not in standalone
    and "_<FILL_CTRL>_<FILL_LABEL>" in m
    and "<FILL_VAR>" in m
  )
  check("disabled_masks: <FILL_CTRL> suppressed, compound preserved", t3, m)
  ok = ok and t3

  # Test 4: Disable <FILL_OP>.
  m = masked_with(frozenset({"<FILL_OP>"}))
  t4 = "<FILL_OP>" not in m and "<FILL_VAR>" in m and "<FILL_CONST>" in m
  check("disabled_masks: <FILL_OP> suppressed", t4, m)
  ok = ok and t4

  # Test 5: Disable all five normal kinds -- only goto-flag compounds survive.
  all_five = frozenset(
    {"<FILL_VAR>", "<FILL_CONST>", "<FILL_OP>", "<FILL_FUNC>", "<FILL_CTRL>"}
  )
  m = masked_with(all_five)
  t5 = (
    "<FILL_VAR>" not in m
    and "<FILL_CONST>" not in m
    and "<FILL_OP>" not in m
    and "<FILL_FUNC>" not in m
    and "<FILL_CTRL>" not in m.replace("_<FILL_CTRL>_<FILL_LABEL>", "")
    and "_<FILL_CTRL>_<FILL_LABEL>" in m
    and "_go_<FILL_LABEL>" in m
  )
  check("disabled_masks: all five suppressed, goto flags survive", t5, m)
  ok = ok and t5

  # Test 6 (edge case): Profile validation rejects unknown and
  # non-disableable mask kinds: kind names are pinned so the vocabulary
  # cannot silently change.
  for kind in (
    "<FILL_BOGUS>",
    "<FILL_TYPE>",
    "<FILL_LABEL>",
    "<FILL_CTRL>",
    "<FILL_FIELD>",
  ):
    try:
      bad = mod.GenerationProfile(
        n_bbls=mod.IntRange(2, 4),
        n_stmts=mod.IntRange(2, 3),
        min_loop_iter=mod.IntRange(0, 1),
        p_mask_lhs_vars=mod.FloatRange(0.05, 0.15),
        p_mask_rhs_vars=mod.FloatRange(0.5, 0.7),
        p_mask_ops=mod.FloatRange(0.5, 0.7),
        p_mask_funcs=mod.FloatRange(0.5, 0.7),
        p_mask_consts=mod.FloatRange(0.5, 0.7),
        max_ptr_depth=mod.IntRange(0, 0),
        p_backedge=mod.FloatRange(0.1, 0.3),
        p_branch=mod.FloatRange(0.3, 0.5),
        n_vars=mod.IntRange(6, 10),
        n_params=mod.IntRange(2, 3),
        n_examples=mod.IntRange(3, 3),
        chksum_every=mod.IntRange(3, 3),
        lift_consts=False,
        features=("--no-fp", "--no-vec", "--no-ptrarith", "--no-intrinsics"),
        disabled_masks=frozenset({kind}),
      )
      bad.validate("bad")
      check(f"disabled_masks: {kind} rejected", False, "no ValueError")
      ok = False
    except ValueError:
      check(f"disabled_masks: {kind} rejected", True)
    except Exception as exc:
      check(f"disabled_masks: {kind} rejected", False, str(exc))
      ok = False

  return ok


def trusted_layout_unit_tests_pass(ccommon) -> bool:
  """_Ptr geometry args (off/stride/lo/hi) and _cast_int widths come from
  the frontend layout and IR types, not the answer space: both stay
  visible and never enter the constant budget. The _Ptr buffer and the
  _cast_int value mask normally."""
  ok = True
  src = (
    "def func_t(v1):\n"
    "    # ^entry\n"
    "    y = (v1 + 1)\n"
    "    while True:\n"
    "        # ^b0\n"
    "        p0 = _Ptr(v1, 0, 2, 0, 2, _frame)\n"
    "        v2 = _cast_int(v1 + 5, 32)\n"
    "        break\n"
    "    # ^exit\n"
    "    return y\n"
  ).encode("utf-8")
  tree = ast.parse(src)
  leaf, _ = ccommon.find_python_leaf_function(tree, src)
  maskable, entry_line, _ = ccommon.get_python_maskable_statements(leaf, src)
  local_names = ccommon.collect_python_leaf_locals(leaf)
  repls: list = []
  const_values: dict = {}
  for stmt in maskable:
    ccommon.collect_python_replacements(
      stmt,
      src,
      stmt.lineno > entry_line,
      repls,
      local_names,
      set(),
      const_values=const_values,
    )
  masked = ccommon.apply_replacements(src, repls).decode("utf-8")

  good = "_Ptr(<FILL_VAR>, 0, 2, 0, 2, _frame)" in masked
  check("trusted: _Ptr geometry stays visible", good, masked)
  ok = ok and good

  good = "_cast_int(<FILL_VAR> <FILL_OP> <FILL_CONST>, 32)" in masked
  check("trusted: _cast_int width stays visible", good, masked)
  ok = ok and good

  good = set(const_values.values()) == {"5"}
  check("trusted: geometry args never enter the budget", good, str(const_values))
  ok = ok and good

  return ok


def budget_split_unit_tests_pass(ccommon, chk_mod) -> bool:
  """The budget is a (live, dead) split per value: live slots sit in blocks
  the prescribed run reaches (plus pre-entry declarations, which always
  run), dead slots off it. Totals that add up but park a constant across
  the boundary fail."""
  ok = True
  src = (
    "def func_t(pa):\n"
    "    _brk_exit = False\n"
    "    v1 = 7\n"
    "    while True:\n"
    "        # ^entry\n"
    "        if (pa > 0):\n"
    "            # ^b0\n"
    "            v1 = 5\n"
    "            _brk_exit = True\n"
    "            break\n"
    "        else:\n"
    "            break\n"
    "    # ^exit\n"
    "    return v1\n"
  ).encode("utf-8")
  tree = ast.parse(src)
  leaf, _ = ccommon.find_python_leaf_function(tree, src)
  maskable, entry_line, _ = ccommon.get_python_maskable_statements(leaf, src)
  local_names = ccommon.collect_python_leaf_locals(leaf)
  comments = ccommon.find_python_block_comments(src)
  live_blocks = frozenset({"entry", "exit"})

  # Every cell of every statement is in the render, so the budget counts
  # each const cell at its statement's region.
  live_counts: dict = {}
  dead_counts: dict = {}
  repls: list = []
  cells, _lhs_spans, const_values = ccommon.collect_canonical_cells(
    maskable, entry_line, src, local_names, set(), frozenset()
  )
  for start, end, token, stmt_index, _is_lhs in cells:
    repls.append((start, end, token))
    if token != "<FILL_CONST>":
      continue
    slot = (
      live_counts
      if ccommon.stmt_is_on_live_path(
        comments, maskable[stmt_index].lineno, live_blocks
      )
      else dead_counts
    )
    val = const_values[(start, end)]
    slot[val] = slot.get(val, 0) + 1
  split = ccommon.merge_const_split(live_counts, dead_counts)

  good = (
    split == {"7": (1, 0), "5": (0, 1)}
    and live_counts == {"7": 1}
    and dead_counts == {"5": 1}
  )
  check("split: pre-entry slots live, off-path slots dead", good, str(split))
  ok = ok and good

  banner = (
    "#//@ CFG_EDGE: entry -> b0\n"
    "#//@ CFG_EDGE: b0 -> exit\n"
    "#//@ EXEC_PATH: entry -> exit\n"
    "#//@ <FILL_CONST>: 7 1 0\n"
    "#//@ <FILL_CONST>: 5 0 1\n"
  )
  puzzle = banner + ccommon.apply_replacements(src, repls).decode("utf-8")
  inferred = chk_mod.infer_masked_cells(leaf, src, puzzle, set())
  good = inferred is not None
  check("split: mask set infers from the split puzzle", good, str(inferred))
  ok = ok and good
  if inferred is None:
    return ok

  actual = chk_mod.check_remasking(leaf, src, puzzle, inferred, frozenset(live_blocks))
  check("split: re-masking counts the true split", actual == split, str(actual))

  req = chk_mod.parse_puzzle_requirements(puzzle)
  truth_ok = True
  detail = ""
  try:
    chk_mod.check_fill_const_budget(actual, req.const_budget)
  except chk_mod.CheckFailure as exc:
    truth_ok = False
    detail = str(exc)
  check("split: the true split passes", truth_ok, detail)
  ok = ok and truth_ok

  parked_ok = False
  try:
    chk_mod.check_fill_const_budget(actual, {"7": (0, 1), "5": (1, 0)})
  except chk_mod.CheckFailure as exc:
    parked_ok = exc.result == chk_mod.CheckResult.FAIL_FILL_CONST
  except Exception as exc:  # noqa: BLE001
    check("split: parked across the boundary fails on equal totals", False, str(exc))
    return ok
  check("split: parked across the boundary fails on equal totals", parked_ok)
  ok = ok and parked_ok

  malformed_ok = False
  try:
    chk_mod.parse_puzzle_requirements("#//@ <FILL_CONST>: 7\n")
  except chk_mod.CheckFailure as exc:
    malformed_ok = exc.result == chk_mod.CheckResult.FAIL_PARSE
  check("split: malformed budget lines fail closed at parse", malformed_ok)
  ok = ok and malformed_ok

  return ok


def livedead_budget_option_unit_tests_pass(ccommon, chk_mod, mod) -> bool:
  """livedead_const_budget option in profiles:
  - Defaults to False on profiles and configs.
  - When False, render_header produces 2-token budget markers and pairs prose.
  - When True, render_header produces 3-token budget markers and triples prose.
  - When 2-token budget is in the puzzle, checker parses and validates
    regardless of whether constants sit in live or dead blocks.
  - Total count mismatch and off-budget constants are rejected under 2-token budget.
  """
  ok = True

  # 1. Option defaults to False on profiles and GeneratorConfig
  small_prof = mod.PROFILES["small"]
  good1 = (
    hasattr(small_prof, "livedead_const_budget")
    and small_prof.livedead_const_budget is False
  )
  if good1:
    cfg = mod.sample_config(small_prof, random.Random(0))
    good1 = hasattr(cfg, "livedead_const_budget") and cfg.livedead_const_budget is False
  check("livedead_option: defaults to False on profile and sampled config", good1)
  ok = ok and good1

  # 2. Header rendering with livedead_const_budget=False outputs 2-token lines
  budget_counts = {"7": (2, 1), "5": (0, 3)}
  hdr_false = ""
  try:
    hdr_false = mod.render_header(
      "test_leaf",
      [],
      "entry -> exit",
      budget_counts,
      lift_consts=False,
      livedead_const_budget=False,
    )
  except TypeError:
    pass
  good2 = (
    "#//@ <FILL_CONST>: 7 3\n" in hdr_false
    and "#//@ <FILL_CONST>: 5 3\n" in hdr_false
    and '"<value> <count>" pairs' in hdr_false
    and '"<value> <live> <dead>" triples' not in hdr_false
  )
  check("livedead_option: render_header produces 2-token lines when False", good2)
  ok = ok and good2

  # 3. Header rendering with livedead_const_budget=True outputs 3-token lines
  hdr_true = ""
  try:
    hdr_true = mod.render_header(
      "test_leaf",
      [],
      "entry -> exit",
      budget_counts,
      lift_consts=False,
      livedead_const_budget=True,
    )
  except TypeError:
    pass
  good3 = (
    "#//@ <FILL_CONST>: 7 2 1\n" in hdr_true
    and "#//@ <FILL_CONST>: 5 0 3\n" in hdr_true
    and '"<value> <live> <dead>" triples' in hdr_true
    and '"<value> <count>" pairs' not in hdr_true
  )
  check("livedead_option: render_header produces 3-token lines when True", good3)
  ok = ok and good3

  # 4. Checker parses 2-token budget and allows constant placement across regions
  banner_2tok = (
    "#//@ CFG_EDGE: entry -> exit\n"
    "#//@ EXEC_PATH: entry -> exit\n"
    "#//@ <FILL_CONST>: 7 1\n"
    "#//@ <FILL_CONST>: 5 1\n"
  )
  req_2tok = None
  try:
    req_2tok = chk_mod.parse_puzzle_requirements(banner_2tok)
  except Exception:
    pass
  good4 = req_2tok is not None and req_2tok.const_budget == {"7": 1, "5": 1}
  if good4:
    try:
      # Actual: 7 is dead (0, 1), 5 is live (1, 0).
      chk_mod.check_fill_const_budget({"7": (0, 1), "5": (1, 0)}, req_2tok.const_budget)
      # Actual: 7 is live (1, 0), 5 is dead (0, 1).
      chk_mod.check_fill_const_budget({"7": (1, 0), "5": (0, 1)}, req_2tok.const_budget)
    except chk_mod.CheckFailure:
      good4 = False
  check(
    "livedead_option: checker parses 2-token budget and accepts any valid live/dead distribution",
    good4,
  )
  ok = ok and good4

  # 5. Checker rejects count mismatch and off-budget on 2-token budget
  good5 = True
  if req_2tok is not None:
    mismatch_fail = False
    try:
      chk_mod.check_fill_const_budget({"7": (2, 0), "5": (1, 0)}, req_2tok.const_budget)
    except chk_mod.CheckFailure as exc:
      mismatch_fail = (
        exc.result == chk_mod.CheckResult.FAIL_FILL_CONST
        and "Expected 1, got 2" in str(exc)
      )
    offbudget_fail = False
    try:
      chk_mod.check_fill_const_budget(
        {"7": (1, 0), "5": (1, 0), "99": (1, 0)}, req_2tok.const_budget
      )
    except chk_mod.CheckFailure as exc:
      offbudget_fail = (
        exc.result == chk_mod.CheckResult.FAIL_FILL_CONST
        and "Off-budget constant in a <FILL_CONST> position: '99' (count: 1)"
        in str(exc)
      )
    good5 = mismatch_fail and offbudget_fail
  else:
    good5 = False
  check(
    "livedead_option: checker rejects count mismatch and off-budget on 2-token budget",
    good5,
  )
  ok = ok and good5

  # 6. Edge case: mixing 2-token and 3-token lines in the same banner fails closed
  mixed_banner = (
    "#//@ CFG_EDGE: a -> b\n"
    "#//@ EXEC_PATH: a -> b\n"
    "#//@ <FILL_CONST>: 7 1\n"
    "#//@ <FILL_CONST>: 5 1 0\n"
  )
  mixed_ok = False
  try:
    chk_mod.parse_puzzle_requirements(mixed_banner)
  except chk_mod.CheckFailure as exc:
    mixed_ok = (
      exc.result == chk_mod.CheckResult.FAIL_PARSE
      and "mixed 2-token and 3-token" in str(exc)
    )
  check(
    "livedead_option: mixed 2-token and 3-token markers fail closed at parse", mixed_ok
  )
  ok = ok and mixed_ok

  # 7. render_instruction correctly formats 2-token vs 3-token vs free budget
  inst_false = mod.render_instruction(has_budget=True, livedead_const_budget=False)
  inst_true = mod.render_instruction(has_budget=True, livedead_const_budget=True)
  inst_free = mod.render_instruction(has_budget=False)
  good7 = (
    "<value> <count>" in inst_false
    and "<value> <live> <dead>" not in inst_false
    and "parked in the wrong region" not in inst_false
    and "<value> <live> <dead>" in inst_true
    and "parked in the wrong region" in inst_true
    and "choose any value" in inst_free
  )
  check("livedead_option: render_instruction formats 2-token and 3-token modes", good7)
  ok = ok and good7

  # 8. analyze_puzzle extracts 2-token budget metrics and rejects extra tokens
  with tempfile.TemporaryDirectory() as tmpdir:
    p_file = Path(tmpdir) / "puzzle.py"
    p_file.write_text(
      "#//@ CFG_EDGE: entry -> exit\n"
      "#//@ EXEC_PATH: entry -> exit\n"
      "#//@ <FILL_CONST>: 7 2\n"
      "#//@ <FILL_CONST>: 5 1\n"
      "#//@ <FILL_CONST>: 9 1 2 3\n"
      "def leaf():\n"
      "  return 0\n"
    )
    metrics_2tok = mod.analyze_puzzle(p_file)
    good8 = (
      metrics_2tok.const_budget_entries == 2 and metrics_2tok.const_budget_total == 3
    )
  check(
    "livedead_option: analyze_puzzle counts 2-token budget and ignores malformed tokens",
    good8,
  )
  ok = ok and good8

  return ok


def sentinel_unit_tests_pass(ccommon, chk_mod) -> bool:
  """0, 1, 0.0, and 1.0 are structural sentinels (counters, identity
  elements): they stay visible in every zone, never become <FILL_CONST>
  cells, and never enter the budget, so filling a mark with one fails
  re-masking."""
  ok = True
  src = (
    "def func_t(pa):\n"
    "    v1 = 0\n"
    "    v2 = 1\n"
    "    v3 = 7\n"
    "    # ^entry\n"
    "    v4 = pa\n"
    "    # ^b0\n"
    "    v5 = v3 * 0\n"
    "    v6 = v3 + 1\n"
    "    v7 = v3 + 1.0\n"
    "    v8 = v3 * 0.0\n"
    "    v9 = v3 - 2\n"
    "    # ^exit\n"
    "    return v4\n"
  ).encode("utf-8")
  tree = ast.parse(src)
  leaf, _ = ccommon.find_python_leaf_function(tree, src)
  maskable, entry_line, _ = ccommon.get_python_maskable_statements(leaf, src)
  local_names = ccommon.collect_python_leaf_locals(leaf)
  repls: list = []
  const_values: dict = {}
  for stmt in maskable:
    ccommon.collect_python_replacements(
      stmt,
      src,
      stmt.lineno > entry_line,
      repls,
      local_names,
      set(),
      const_values=const_values,
    )
  masked = ccommon.apply_replacements(src, repls).decode("utf-8")

  good = "<FILL_OP> 0\n" in masked
  check("sentinel: body 0 stays visible", good, masked)
  ok = ok and good

  good = "<FILL_OP> 1\n" in masked
  check("sentinel: body 1 stays visible", good, masked)
  ok = ok and good

  good = "<FILL_OP> 1.0\n" in masked and "<FILL_OP> 0.0\n" in masked
  check("sentinel: float sentinels stay visible", good, masked)
  ok = ok and good

  good = set(const_values.values()) == {"7", "2"}
  check("sentinel: 0/1 never enter the budget", good, str(const_values))
  ok = ok and good

  banner = (
    "#//@ CFG_EDGE: entry -> b0\n"
    "#//@ CFG_EDGE: b0 -> exit\n"
    "#//@ EXEC_PATH: entry -> b0 -> exit\n"
    "#//@ <FILL_CONST>: 7 1 0\n"
    "#//@ <FILL_CONST>: 2 1 0\n"
  )
  puzzle = banner + masked
  filled_zero = (banner + src.decode("utf-8")).replace("v9 = v3 - 2", "v9 = v3 - 0")
  filled_tree = ast.parse(filled_zero)
  filled_leaf, _ = ccommon.find_python_leaf_function(filled_tree, filled_zero)
  inferred_zero = chk_mod.infer_masked_cells(
    filled_leaf, filled_zero.encode("utf-8"), puzzle, set()
  )
  good = inferred_zero is None
  check("sentinel: filling a mark with 0 fails re-masking", good, str(inferred_zero))
  ok = ok and good

  # Declarations keep their sentinels verbatim even when they sit in the
  # mask set, and the end-to-end re-masked split matches the budget: only
  # non-sentinel values appear, at their true live and dead counts.
  good = "v1 = 0" in masked and "v2 = 1" in masked and "v3 = <FILL_CONST>" in masked
  check("sentinel: declaration sentinels remain visible", good, masked)
  ok = ok and good

  inferred = chk_mod.infer_masked_cells(leaf, src, puzzle, set())
  split_ok = False
  detail = ""
  if inferred is not None:
    try:
      split = chk_mod.check_remasking(
        leaf, src, puzzle, inferred, frozenset({"entry", "b0", "exit"})
      )
      split_ok = split == {"7": (1, 0), "2": (1, 0)}
      detail = str(split)
    except chk_mod.CheckFailure as exc:
      detail = str(exc)
  check("sentinel: end-to-end split excludes 0/1", split_ok, detail)
  ok = ok and split_ok

  return ok


def fine_grained_unit_tests_pass(ccommon, chk_mod, mod) -> bool:
  """Every cell wears its kind's probability: ops/consts/funcs mask on
  p_mask_ops/consts/funcs, variables split by side on p_mask_lhs_vars and
  p_mask_rhs_vars, and ctrl/goto cells mask always. The checker infers the
  masked cells from the puzzle text itself."""
  ok = True
  src = (
    "def func_t(pa):\n"
    "    v1 = 7\n"
    "    while True:\n"
    "        # ^entry\n"
    "        if (pa > 0):\n"
    "            # ^b0\n"
    "            v2 = v1 + 5\n"
    "            v3 = v1 * 2\n"
    "            break\n"
    "    # ^exit\n"
    "    return v2\n"
  ).encode("utf-8")
  tree = ast.parse(src)
  leaf, _ = ccommon.find_python_leaf_function(tree, src)
  maskable, entry_line, _ = ccommon.get_python_maskable_statements(leaf, src)
  local_names = ccommon.collect_python_leaf_locals(leaf)

  def collect():
    """Canonical cells per statement, with statement indices and const values."""
    with_stmt = []
    const_values = {}
    for idx, stmt in enumerate(maskable):
      repls: list = []
      ccommon.collect_python_replacements(
        stmt,
        src,
        stmt.lineno > entry_line,
        repls,
        local_names,
        set(),
        const_values=const_values,
      )
      for start, end, token in ccommon.canonical_cells(repls):
        with_stmt.append((start, end, token, idx))
    return with_stmt, const_values

  stmt_cells, const_values = collect()

  # (1) A partially masked body: the lhs var and the const of one statement,
  # everything else visible. The checker infers exactly those masked cells.
  chosen = []
  for start, end, token, idx in stmt_cells:
    line = src[start:end].decode("utf-8")
    if token == "<FILL_VAR>" and src[end:].decode("utf-8").lstrip().startswith(
      "= v1 + 5"
    ):
      chosen.append((start, end, token, idx))
    elif token == "<FILL_CONST>" and (line == "5" or line == "7"):
      chosen.append((start, end, token, idx))
  masked_cells = [(s, e, t) for s, e, t, _ in chosen]
  banner = (
    "#//@ CFG_EDGE: entry -> b0\n"
    "#//@ CFG_EDGE: b0 -> exit\n"
    "#//@ EXEC_PATH: entry -> b0 -> exit\n"
    "#//@ <FILL_CONST>: 7 1 0\n"
    "#//@ <FILL_CONST>: 5 1 0\n"
  )
  puzzle = banner + ccommon.apply_replacements(src, masked_cells).decode("utf-8")
  inferred = chk_mod.infer_masked_cells(leaf, src, puzzle, set(), frozenset())
  good = inferred is not None and inferred.cells == chosen
  check("fine: checker infers the masked cells", good, str(inferred))
  ok = ok and good

  # (2) Changed fixed code has no consistent masked set: FAIL_REMASKING.
  tampered = puzzle.replace("return v2", "return v3", 1)
  inferred = chk_mod.infer_masked_cells(leaf, src, tampered, set(), frozenset())
  good = inferred is None
  check("fine: tampered fixed code fails re-masking", good, str(inferred))
  ok = ok and good

  # (3) The budget is the multiset of masked const cells, split by their
  # statement's region; visible const cells never enter it.
  actual = chk_mod.check_remasking(
    leaf,
    src,
    puzzle,
    chk_mod.InferredMasks(chosen, const_values),
    frozenset({"entry", "b0", "exit"}),
  )
  good = actual == {"7": (1, 0), "5": (1, 0)}
  check("fine: budget counts only masked const cells", good, str(actual))
  ok = ok and good

  # (4) Zero probabilities still mask the forced cells: ctrl keywords and
  # goto tokens. Nothing else shows a mask.
  flag_src = (
    "def func_g(x):\n"
    "    _brk_exit = False\n"
    "    v1 = 5\n"
    "    while True:\n"
    "        # ^entry\n"
    "        v2 = v1 + x\n"
    "        if (v2 > 3):\n"
    "            # ^b0\n"
    "            _brk_exit = True\n"
    "            break\n"
    "    # ^exit\n"
    "    return v2\n"
  ).encode("utf-8")
  flag_tree = ast.parse(flag_src)
  flag_leaf, _ = ccommon.find_python_leaf_function(flag_tree, flag_src)
  flag_maskable, flag_entry, _ = ccommon.get_python_maskable_statements(
    flag_leaf, flag_src
  )
  flag_locals = ccommon.collect_python_leaf_locals(flag_leaf)
  zero_masked = mod.mask_puzzle(
    flag_src,
    flag_leaf,
    flag_entry,
    flag_maskable,
    flag_locals,
    set(),
    mod.MaskProbs(0.0, 0.0, 0.0, 0.0, 0.0),
    42,
    frozenset(),
    frozenset(),
  )
  body = zero_masked.puzzle_body
  good = (
    "<FILL_CTRL>" in body
    and "_<FILL_CTRL>_<FILL_LABEL>" in body
    and "<FILL_VAR>" not in body
    and "<FILL_OP>" not in body
    and "<FILL_CONST>" not in body
  )
  check("fine: zero probs leave only forced cells masked", good, body)
  ok = ok and good

  # (5) lhs and rhs variables roll separately: full lhs probability masks
  # only assignment targets; rhs vars, ops, and consts stay visible.
  lhs_masked = mod.mask_puzzle(
    src,
    leaf,
    entry_line,
    maskable,
    local_names,
    set(),
    mod.MaskProbs(1.0, 0.0, 0.0, 0.0, 0.0),
    42,
    frozenset(),
    frozenset(),
  )
  good = (
    "<FILL_VAR> = v1 + 5" in lhs_masked.puzzle_body
    and "<FILL_VAR> = v1 * 2" in lhs_masked.puzzle_body
    and "v1 + 5" in lhs_masked.puzzle_body
    and "return v2" in lhs_masked.puzzle_body
  )
  check("fine: lhs and rhs variables roll separately", good, lhs_masked.puzzle_body)
  ok = ok and good

  # (5b) rhs, ops, and consts roll independently on their own knobs.
  rhs_masked = mod.mask_puzzle(
    src,
    leaf,
    entry_line,
    maskable,
    local_names,
    set(),
    mod.MaskProbs(0.0, 1.0, 0.0, 0.0, 0.0),
    42,
    frozenset(),
    frozenset(),
  )
  good = (
    "v2 = <FILL_VAR> + 5" in rhs_masked.puzzle_body
    and "v3 = <FILL_VAR> * 2" in rhs_masked.puzzle_body
    and "<FILL_VAR> = " not in rhs_masked.puzzle_body
  )
  check("fine: rhs roll masks only operand variables", good, rhs_masked.puzzle_body)
  ok = ok and good

  ops_masked = mod.mask_puzzle(
    src,
    leaf,
    entry_line,
    maskable,
    local_names,
    set(),
    mod.MaskProbs(0.0, 0.0, 1.0, 0.0, 0.0),
    42,
    frozenset(),
    frozenset(),
  )
  good = (
    "v1 <FILL_OP> 5" in ops_masked.puzzle_body
    and "v1 <FILL_OP> 2" in ops_masked.puzzle_body
    and "<FILL_VAR>" not in ops_masked.puzzle_body
    and "<FILL_CONST>" not in ops_masked.puzzle_body
  )
  check("fine: ops roll masks only operators", good, ops_masked.puzzle_body)
  ok = ok and good

  consts_masked = mod.mask_puzzle(
    src,
    leaf,
    entry_line,
    maskable,
    local_names,
    set(),
    mod.MaskProbs(0.0, 0.0, 0.0, 0.0, 1.0),
    42,
    frozenset(),
    frozenset(),
  )
  good = (
    "v1 + <FILL_CONST>" in consts_masked.puzzle_body
    and "v1 * <FILL_CONST>" in consts_masked.puzzle_body
    and "<FILL_VAR>" not in consts_masked.puzzle_body
    and "<FILL_OP>" not in consts_masked.puzzle_body
  )
  check("fine: consts roll masks only constants", good, consts_masked.puzzle_body)
  ok = ok and good

  func_src = (
    "def func_f(pa):\n"
    "    v1 = 7\n"
    "    while True:\n"
    "        # ^entry\n"
    "        if (pa > 0):\n"
    "            # ^b0\n"
    "            v2 = helper(v1)\n"
    "            break\n"
    "    # ^exit\n"
    "    return v2\n"
    "def helper(x):\n"
    "    return x + 1\n"
  ).encode("utf-8")
  func_tree = ast.parse(func_src)
  func_leaf, _ = ccommon.find_python_leaf_function(func_tree, func_src)
  func_maskable, func_entry, _ = ccommon.get_python_maskable_statements(
    func_leaf, func_src
  )
  func_locals = ccommon.collect_python_leaf_locals(func_leaf)
  funcs_masked = mod.mask_puzzle(
    func_src,
    func_leaf,
    func_entry,
    func_maskable,
    func_locals,
    {"helper"},
    mod.MaskProbs(0.0, 0.0, 0.0, 1.0, 0.0),
    42,
    frozenset(),
    frozenset(),
  )
  good = (
    "v2 = <FILL_FUNC>(v1)" in funcs_masked.puzzle_body
    and "<FILL_VAR>" not in funcs_masked.puzzle_body
  )
  check("fine: funcs roll masks only function calls", good, funcs_masked.puzzle_body)
  ok = ok and good

  # (6) The piece walk is insensitive to whitespace between cells: a
  # solution reformatted with blank lines still re-masks to the puzzle,
  # with the same masked cells and the same budget.
  reformatted = src.replace(b"    v2 = v1 + 5\n", b"    v2 = v1 + 5\n\n", 1)
  fmt_tree = ast.parse(reformatted)
  fmt_leaf, _ = ccommon.find_python_leaf_function(fmt_tree, reformatted)
  re_inferred = chk_mod.infer_masked_cells(fmt_leaf, reformatted, puzzle, set())
  fmt_ok = re_inferred is not None
  if fmt_ok:
    fmt_split = chk_mod.check_remasking(
      fmt_leaf,
      reformatted,
      puzzle,
      re_inferred,
      frozenset({"entry", "b0", "exit"}),
    )
    fmt_ok = fmt_split == {"7": (1, 0), "5": (1, 0)}
  check("fine: reformatted solution still checks", fmt_ok, str(re_inferred))
  ok = ok and fmt_ok

  # (6b) Cell spans are byte offsets: a multibyte character before a masked
  # cell must not skew the inference walk.
  uni_src = src.replace(
    b"        # ^entry\n",
    "        # caf\u00e9 note\n        # ^entry\n".encode("utf-8"),
    1,
  )
  uni_tree = ast.parse(uni_src)
  uni_leaf, _ = ccommon.find_python_leaf_function(uni_tree, uni_src)
  uni_maskable, uni_entry, _ = ccommon.get_python_maskable_statements(uni_leaf, uni_src)
  uni_locals = ccommon.collect_python_leaf_locals(uni_leaf)
  uni_cells, _uni_lhs, _uni_cvals = ccommon.collect_canonical_cells(
    uni_maskable, uni_entry, uni_src, uni_locals, set(), frozenset()
  )
  uni_tgt = [(s, e, t) for s, e, t, _si, _lhs in uni_cells if uni_src[s:e] == b"5"]
  uni_puzzle = banner + ccommon.apply_replacements(uni_src, uni_tgt).decode("utf-8")
  uni_inferred = chk_mod.infer_masked_cells(uni_leaf, uni_src, uni_puzzle, set())
  uni_ok = uni_inferred is not None and len(uni_inferred.cells) == 1
  check("fine: multibyte text before a cell still infers", uni_ok, str(uni_inferred))
  ok = ok and uni_ok

  # (7) The five knobs carry defaults (lhs 0.1, rest 0.75), reject values
  # outside [0, 1], and sample within each profile's five ranges.
  cfg = mod.GeneratorConfig(
    n_bbls=2,
    n_stmts=2,
    min_loop_iter=1,
    max_ptr_depth=0,
    p_backedge=0.5,
    p_branch=0.5,
    n_vars=4,
    n_params=2,
    n_examples=1,
    chksum_every=1,
    lift_consts=False,
  )
  good = (
    cfg.p_mask_lhs_vars == 0.1
    and cfg.p_mask_rhs_vars == 0.75
    and cfg.p_mask_ops == 0.75
    and cfg.p_mask_funcs == 0.75
    and cfg.p_mask_consts == 0.75
  )
  check("fine: five knobs carry defaults", good, str(cfg))
  ok = ok and good

  bad_cfg = mod.GeneratorConfig(
    n_bbls=2,
    n_stmts=2,
    min_loop_iter=1,
    p_mask_lhs_vars=1.5,
    max_ptr_depth=0,
    p_backedge=0.5,
    p_branch=0.5,
    n_vars=4,
    n_params=2,
    n_examples=1,
    chksum_every=1,
    lift_consts=False,
  )
  bad_ok = False
  try:
    bad_cfg.validate()
  except ValueError:
    bad_ok = True
  check("fine: out-of-range knob rejected", bad_ok, str(bad_cfg))
  ok = ok and bad_ok

  for name, prof in mod.PROFILES.items():
    prof.validate(name)
    rng = __import__("random").Random(2)
    for _ in range(20):
      sampled_cfg = mod.sample_config(prof, rng)
      sampled = (
        prof.p_mask_ops.minimum <= sampled_cfg.p_mask_ops <= prof.p_mask_ops.maximum
        and prof.p_mask_lhs_vars.minimum
        <= sampled_cfg.p_mask_lhs_vars
        <= prof.p_mask_lhs_vars.maximum
        and prof.p_mask_rhs_vars.minimum
        <= sampled_cfg.p_mask_rhs_vars
        <= prof.p_mask_rhs_vars.maximum
        and prof.p_mask_funcs.minimum
        <= sampled_cfg.p_mask_funcs
        <= prof.p_mask_funcs.maximum
        and prof.p_mask_consts.minimum
        <= sampled_cfg.p_mask_consts
        <= prof.p_mask_consts.maximum
      )
      if not sampled:
        check(f"fine: {name} samples the five within ranges", False, str(sampled_cfg))
        ok = False
        break
    else:
      check(f"fine: {name} samples the five within ranges", True)

  # (8) AugAssign targets are left-hand side: they roll on p_mask_lhs_vars.
  aug_src = (
    "def func_t(pa):\n"
    "    v1 = 7\n"
    "    while True:\n"
    "        # ^entry\n"
    "        if (pa > 0):\n"
    "            # ^b0\n"
    "            v2 = v1 + 5\n"
    "            v2 += 3\n"
    "            break\n"
    "    # ^exit\n"
    "    return v2\n"
  ).encode("utf-8")
  aug_tree = ast.parse(aug_src)
  aug_leaf, _ = ccommon.find_python_leaf_function(aug_tree, aug_src)
  aug_maskable, aug_entry, _ = ccommon.get_python_maskable_statements(aug_leaf, aug_src)
  aug_locals = ccommon.collect_python_leaf_locals(aug_leaf)
  good = any(isinstance(s, ast.AugAssign) for s in aug_maskable)
  check("fine: augassign statement is maskable", good, str(aug_maskable))
  ok = ok and good

  aug_stmt = next((s for s in aug_maskable if isinstance(s, ast.AugAssign)), None)
  if aug_stmt is None:
    check("fine: augassign target is lhs", False, "no AugAssign statement found")
    ok = False
    return ok
  aug_repls: list = []
  aug_lhs: set = set()
  ccommon.collect_python_replacements(
    aug_stmt, aug_src, True, aug_repls, aug_locals, set(), lhs_spans=aug_lhs
  )
  aug_target = [(s, e) for s, e, t in aug_repls if aug_src[s:e] == b"v2"]
  good = len(aug_target) == 1 and aug_target[0] in aug_lhs
  check("fine: augassign target is lhs", good, str(aug_repls))
  ok = ok and good

  aug_lhs_masked = mod.mask_puzzle(
    aug_src,
    aug_leaf,
    aug_entry,
    aug_maskable,
    aug_locals,
    set(),
    mod.MaskProbs(1.0, 0.0, 0.0, 0.0, 0.0),
    42,
    frozenset(),
    frozenset(),
  )
  good = "<FILL_VAR> += 3" in aug_lhs_masked.puzzle_body
  check("fine: lhs roll masks the augassign target", good, aug_lhs_masked.puzzle_body)
  ok = ok and good

  aug_rhs_masked = mod.mask_puzzle(
    aug_src,
    aug_leaf,
    aug_entry,
    aug_maskable,
    aug_locals,
    set(),
    mod.MaskProbs(0.0, 1.0, 0.0, 0.0, 0.0),
    42,
    frozenset(),
    frozenset(),
  )
  good = "v2 += 3" in aug_rhs_masked.puzzle_body
  check("fine: rhs roll leaves the augassign target", good, aug_rhs_masked.puzzle_body)
  ok = ok and good

  return ok


def checksum_unit_tests_pass(ccommon, chk_mod, mod) -> bool:
  """The exit-block checksum randomizes its operator per chain step and
  recalibrates the expected checksum in main by running the ground truth;
  a sample that trips div/rem by zero or a negative shift count is
  discarded, and all failing samples keep the addition chain."""
  ok = True
  fix_src = (
    "def _cast_int(v, m):\n"
    "    v &= (1 << m) - 1\n"
    "    return v - (1 << m) if v >= 1 << (m - 1) else v\n"
    "\n"
    "def _in_check_chksum(expected, actual):\n"
    "    return 0 if expected == actual else 1\n"
    "\n"
    "def func_t(pa0, pa1, pa2):\n"
    "    v__chk = 0\n"
    "    v__chk = (v__chk + pa0)\n"
    "    v__chk = (v__chk + pa1)\n"
    "    v__chk = (v__chk + pa2)\n"
    "    return v__chk\n"
    "\n"
    "def main():\n"
    "    r = func_t(3, 5, 7)\n"
    "    r = _in_check_chksum(123, r)\n"
    "    return 0\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    import sys\n"
    "    sys.exit(main())\n"
  ).encode("utf-8")

  def run_module(src_bytes: bytes) -> int:
    import subprocess

    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as tf:
      tf.write(src_bytes)
      path = tf.name
    try:
      r = subprocess.run(
        [sys.executable, path], capture_output=True, text=True, timeout=60
      )
    finally:
      os.unlink(path)
    return r.returncode

  # (1) The chain operators are randomized: the rewritten source differs
  # from the addition chain at a fixed seed.
  randomized = mod.randomize_checksum(fix_src, 42)
  good = randomized != fix_src
  check(
    "checksum: the chain operators are randomized", good, randomized.decode("utf-8")
  )
  ok = ok and good

  # (2) The recalibrated expected checksum makes main's check pass.
  returncode = run_module(randomized)
  good = returncode == 0
  check("checksum: main's check passes after recalibration", good, str(returncode))
  ok = ok and good

  # (3) The rewrite is deterministic for a seed.
  again = mod.randomize_checksum(fix_src, 42)
  good = again == randomized
  check("checksum: same seed is deterministic", good, str(again == randomized))
  ok = ok and good

  # (4) All failing samples keep the original addition chain unchanged.
  broken = fix_src.replace(b"    v__chk = 0\n", b"    v__chk = 0 // 0\n", 1)
  kept = mod.randomize_checksum(broken, 42)
  good = kept == broken
  check("checksum: failing samples keep the addition chain", good, kept.decode("utf-8"))
  ok = ok and good

  # (5) The recalibrated literal is bounded and the patched main parses.
  import re

  literal_ok = re.search(
    r"_in_check_chksum\((-?\d{1,1000}), r\)", randomized.decode("utf-8")
  )
  good = literal_ok is not None and len(literal_ok.group(1)) <= 1000
  check("checksum: literal bounded and parseable", good, str(literal_ok))
  ok = ok and good

  # (6) Well-definedness is guarded by construction: `//` and `%` set
  # their divisor's lowest bit (`y | 1` is never zero), and `**`,
  # `<<` and `>>` clamp their second operand into 0..64.
  good = (
    mod._checksum_wrap("//", "pa0") == "// (pa0 | 1)"
    and mod._checksum_wrap("%", "v__chk") == "% (v__chk | 1)"
    and mod._checksum_wrap("<<", "_rd(v0, 0)") == "<< min(max(_rd(v0, 0), 0), 64)"
    and mod._checksum_wrap(">>", "pa2") == ">> min(max(pa2, 0), 64)"
    and mod._checksum_wrap("+", "pa0") == "+ pa0"
    and mod._checksum_wrap("**", "pa0") == "** min(max(pa0, 0), 64)"
  )
  check("checksum: zero and shift guards wrap the chain", good, str(good))
  ok = ok and good

  # (7) Edge case: rysmith's original checksum can be negative, so the
  # recalibration reads and rewrites it alike.
  negative = fix_src.replace(
    b"_in_check_chksum(123, r)", b"_in_check_chksum(-123, r)", 1
  )
  random_negative = mod.randomize_checksum(negative, 42)
  returncode = run_module(random_negative)
  good = returncode == 0
  check("checksum: negative original constant recalibrates", good, str(returncode))
  ok = ok and good

  # (8) The accumulator leads every chain step (chk op x), whatever the
  # original order: mixed-order fixtures ship uniformly.
  swap_src = fix_src.replace(
    b"    v__chk = (v__chk + pa1)\n", b"    v__chk = (pa1 + v__chk)\n", 1
  )
  swapped = mod.randomize_checksum(swap_src, 42)
  swap_tree = ast.parse(swapped)
  swap_leaf, _ = ccommon.find_python_leaf_function(swap_tree, swapped)
  leads = True
  for stmt in ast.walk(swap_leaf):
    if (
      isinstance(stmt, ast.Assign)
      and len(stmt.targets) == 1
      and isinstance(stmt.targets[0], ast.Name)
      and stmt.targets[0].id.startswith("v__")
      and isinstance(stmt.value, ast.BinOp)
    ):
      checked = False
      if (
        isinstance(stmt.value.left, ast.Name)
        and stmt.value.left.id == stmt.targets[0].id
      ):
        checked = True
      if not checked:
        leads = False
  good = leads and run_module(swapped) == 0
  check(
    "checksum: the accumulator leads every chain step", good, swapped.decode("utf-8")
  )
  ok = ok and good

  # (8) Edge case: the clamp keeps the second operand inside 0..64, so a
  # negative exponent yields 1 (never a float) and a huge count stays
  # small.
  good = (
    eval(f"2 {mod._checksum_wrap('**', '-3')}") == 1
    and isinstance(eval(f"2 {mod._checksum_wrap('**', '-3')}"), int)
    and eval(f"1 {mod._checksum_wrap('<<', '10**18')}") == 1 << 64
    and eval(f"1 {mod._checksum_wrap('>>', '-7')}") == 1
    and eval(f"7 {mod._checksum_wrap('**', '2')}") == 49
  )
  check("checksum: clamp bounds the second operand", good, str(good))
  ok = ok and good

  return ok


def multi_example_tests_pass(ccommon, chk_mod) -> bool:
  """The @main harness replays the leaf once per example (rysmith
  --n-examples 5): each replay carries its own `_in_check_chksum(<expected>,
  r)` call, and the prescribed run traces the leaf path once per example."""
  ok = True

  def program(n_examples: int) -> bytes:
    replay = ""
    for i in range(n_examples):
      arg = 3 - 8 * i
      replay += f"    r = func_t({arg})\n"
      # The leaf returns (pa + 1) + 2 on the prescribed path.
      replay += f"    r = _in_check_chksum({arg + 3}, r)\n"
    return (
      "def _cast_int(v, m):\n"
      "    v &= (1 << m) - 1\n"
      "    return v - (1 << m) if v >= 1 << (m - 1) else v\n"
      "\n"
      "def _in_check_chksum(expected, actual):\n"
      "    if expected != actual:\n"
      "        raise ValueError('@check_chksum mismatch')\n"
      "    return actual\n"
      "\n"
      "def func_t(pa):\n"
      "    # ^entry\n"
      '    if __import__("os").environ.get("DUMP_TRACE"):\n'
      '        print("^entry:")\n'
      "    v = (pa + 1)\n"
      "    # ^b0\n"
      '    if __import__("os").environ.get("DUMP_TRACE"):\n'
      '        print("^b0:")\n'
      "    v = (v + 2)\n"
      "    # ^exit\n"
      '    if __import__("os").environ.get("DUMP_TRACE"):\n'
      '        print("^exit:")\n'
      "    return v\n"
      "\n"
      "def main():\n" + replay + "    return 0\n"
      "\n"
      'if __name__ == "__main__":\n'
      "    import sys\n"
      "    sys.exit(main())\n"
    ).encode("utf-8")

  # (1) The harness example count is one per checksum call, not the
  # helper's definition, and one per replay in the puzzle text.
  count = chk_mod.count_harness_examples(program(3).decode("utf-8"))
  good = count == 3
  check("multi: example count is one per checksum call", good, str(count))
  ok = ok and good

  # (2) A multi-example ground truth passes the full checker: the
  # prescribed trace is the path once per example, and every replay's
  # checksum output must pass.
  def e2e(n_examples: int, tmpdir: str) -> tuple[bool, str]:
    banner = (
      "#//@ CFG_EDGE: entry -> b0\n"
      "#//@ CFG_EDGE: b0 -> exit\n"
      "#//@ EXEC_PATH: entry -> b0 -> exit\n"
    )
    try:
      puzzle_path = os.path.join(tmpdir, "puzzle.py")
      solution_path = os.path.join(tmpdir, "solution.py")
      with open(puzzle_path, "w") as f:
        f.write(banner + program(n_examples).decode("utf-8"))
      with open(solution_path, "wb") as f:
        f.write(program(n_examples))
      chk_mod.check(puzzle_path, solution_path)
      return True, ""
    except chk_mod.CheckFailure as exc:
      return False, f"{exc.result}: {exc.message}"

  with tempfile.TemporaryDirectory(prefix="codoku_multi2_") as tmpdir:
    passed, detail = e2e(2, tmpdir)
    check("multi: two-example ground truth passes the full checker", passed, detail)
    ok = ok and passed

  with tempfile.TemporaryDirectory(prefix="codoku_multi3_") as tmpdir:
    passed, detail = e2e(3, tmpdir)
    check("multi: three-example ground truth passes the full checker", passed, detail)
    ok = ok and passed

  # (3) Edge case: a masked threshold cell decides the path a replay
  # takes. The ground truth's fill keeps every replay on the prescribed
  # path, while a fill that departs one replay fails FAIL_PATH.
  def branching_program(fill: str) -> bytes:
    replay = (
      "    r = func_t(3)\n"
      "    r = _in_check_chksum(1, r)\n"  # truth: (pa + 1) - 3 = 1 on b2
      "    r = func_t(-5)\n"
      "    r = _in_check_chksum(-7, r)\n"
    )
    return (
      "def _cast_int(v, m):\n"
      "    v &= (1 << m) - 1\n"
      "    return v - (1 << m) if v >= 1 << (m - 1) else v\n"
      "\n"
      "def _in_check_chksum(expected, actual):\n"
      "    if expected != actual:\n"
      "        raise ValueError('@check_chksum mismatch')\n"
      "    return actual\n"
      "\n"
      "def func_t(pa):\n"
      "    # ^entry\n"
      '    if __import__("os").environ.get("DUMP_TRACE"):\n'
      '        print("^entry:")\n'
      "    v = (pa + 1)\n"
      "    # ^b0\n"
      '    if __import__("os").environ.get("DUMP_TRACE"):\n'
      '        print("^b0:")\n'
      "    if (v > " + fill + "):\n"
      "        # ^b1\n"
      '        if __import__("os").environ.get("DUMP_TRACE"):\n'
      '            print("^b1:")\n'
      "        v = (v + 2)\n"
      "    else:\n"
      "        # ^b2\n"
      '        if __import__("os").environ.get("DUMP_TRACE"):\n'
      '            print("^b2:")\n'
      "        v = (v - 3)\n"
      "    # ^exit\n"
      '    if __import__("os").environ.get("DUMP_TRACE"):\n'
      '        print("^exit:")\n'
      "    return v\n"
      "\n"
      "def main():\n" + replay + "    return 0\n"
      "\n"
      'if __name__ == "__main__":\n'
      "    import sys\n"
      "    sys.exit(main())\n"
    ).encode("utf-8")

  branch_banner = (
    "#//@ CFG_EDGE: entry -> b0\n"
    "#//@ CFG_EDGE: b0 -> b1\n"
    "#//@ CFG_EDGE: b0 -> b2\n"
    "#//@ CFG_EDGE: b1 -> exit\n"
    "#//@ CFG_EDGE: b2 -> exit\n"
    "#//@ EXEC_PATH: entry -> b0 -> b2 -> exit\n"
  )
  with tempfile.TemporaryDirectory(prefix="codoku_multictrl_") as tmpdir:
    puzzle_path = os.path.join(tmpdir, "puzzle.py")
    with open(puzzle_path, "w") as f:
      f.write(branch_banner + branching_program("<FILL_CONST>").decode("utf-8"))
    verdicts = []
    for name, fill in (("ground truth", "5"), ("departing fill", "3")):
      solution_path = os.path.join(tmpdir, f"{name}.py")
      with open(solution_path, "wb") as f:
        f.write(branching_program(fill))
      try:
        chk_mod.check(puzzle_path, solution_path)
        verdicts.append((name, "PASS", ""))
      except chk_mod.CheckFailure as exc:
        verdicts.append((name, str(exc.result), exc.message))
    good = verdicts[0][1] == "PASS" and verdicts[1][1] == str(
      chk_mod.CheckResult.FAIL_PATH
    )
    check("multi: departing fill fails, ground truth passes", good, str(verdicts))
    ok = ok and good

  return ok


def multi_example_checksum_tests_pass(ccommon, chk_mod, mod) -> bool:
  """The harness carries one `_in_check_chksum` per example: the checksum
  randomization must recalibrate every expectation from a run of the
  ground truth, so a valid solution passes every example's check."""
  ok = True

  def run_module(src_bytes: bytes) -> int:
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as tf:
      tf.write(src_bytes)
      path = tf.name
    try:
      r = subprocess.run(
        [sys.executable, path], capture_output=True, text=True, timeout=60
      )
      return r.returncode
    finally:
      os.unlink(path)

  def multi_fixture(n_checks: int) -> bytes:
    replay = ""
    for i in range(n_checks):
      a0, a1, a2 = 3 - 5 * i, 5 + i, 7 - 2 * i
      replay += f"    r = func_t({a0}, {a1}, {a2})\n"
      # The originals are stale guesses, as in the generated flow: the
      # recalibration re-derives each from a run.
      replay += f"    r = _in_check_chksum({11 * i + 123}, r)\n"
    return (
      "def _cast_int(v, m):\n"
      "    v &= (1 << m) - 1\n"
      "    return v - (1 << m) if v >= 1 << (m - 1) else v\n"
      "\n"
      "def _in_check_chksum(expected, actual):\n"
      "    if expected != actual:\n"
      "        raise ValueError('@check_chksum mismatch')\n"
      "    return actual\n"
      "\n"
      "def func_t(pa0, pa1, pa2):\n"
      "    v__chk = 0\n"
      "    v__chk = (v__chk + pa0)\n"
      "    v__chk = (v__chk + pa1)\n"
      "    v__chk = (v__chk + pa2)\n"
      "    return v__chk\n"
      "\n"
      "def main():\n" + replay + "    return 0\n"
      "\n"
      'if __name__ == "__main__":\n'
      "    import sys\n"
      "    sys.exit(main())\n"
    ).encode("utf-8")

  for n_checks in (2, 3):
    randomized = mod.randomize_checksum(multi_fixture(n_checks), 42)
    returncode = run_module(randomized)
    good = returncode == 0
    check(
      f"checksum: every example's expectation recalibrates ({n_checks} checks)",
      good,
      f"rc={returncode}\n{randomized.decode('utf-8')}",
    )
    ok = ok and good
    calls = re.findall(r"_in_check_chksum\(-?\d+\s*,\s*r\)", randomized.decode("utf-8"))
    good = len(calls) == n_checks
    check(
      f"checksum: every replay's check call survives ({n_checks} checks)",
      good,
      str(calls),
    )
    ok = ok and good

  # (b) Edge case: a leaf the recalibration cannot run keeps the addition
  # chain, whose expectations rysmith calibrated for every example.
  broken = multi_fixture(3).replace(b"    v__chk = 0\n", b"    v__chk = 0 // 0\n", 1)
  kept = mod.randomize_checksum(broken, 42)
  good = kept == broken
  check(
    "checksum: failing samples keep the addition chain (3 checks)",
    good,
    str(kept == broken),
  )
  ok = ok and good

  return ok


def checksum_seed_tests_pass(ccommon, chk_mod, mod) -> bool:
  """The exit block seeds the checksum accumulator with a very large random
  constant instead of zero: zero collapses the first chain steps (`0 * x`
  stays zero), and the seeding travels inside one recalibration bundle."""
  ok = True

  def run_module(src_bytes: bytes) -> int:
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as tf:
      tf.write(src_bytes)
      path = tf.name
    try:
      r = subprocess.run(
        [sys.executable, path], capture_output=True, text=True, timeout=60
      )
      return r.returncode
    finally:
      os.unlink(path)

  fix_src = (
    "def _cast_int(v, m):\n"
    "    v &= (1 << m) - 1\n"
    "    return v - (1 << m) if v >= 1 << (m - 1) else v\n"
    "\n"
    "def _in_check_chksum(expected, actual):\n"
    "    if expected != actual:\n"
    "        raise ValueError('@check_chksum mismatch')\n"
    "    return actual\n"
    "\n"
    "def func_t(pa0, pa1, pa2):\n"
    "    v__chk = 0\n"
    "    v__chk = (v__chk + pa0)\n"
    "    v__chk = (v__chk + pa1)\n"
    "    v__chk = (v__chk + pa2)\n"
    "    return v__chk\n"
    "\n"
    "def main():\n"
    "    r = func_t(3, 5, 7)\n"
    "    r = _in_check_chksum(123, r)\n"
    "    return 0\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    import sys\n"
    "    sys.exit(main())\n"
  ).encode("utf-8")

  # (1) The exit block seeds a very large constant instead of zero: the
  # extracted function's initializer carries the seed.
  randomized = mod.randomize_checksum(fix_src, 42)
  init_match = re.search(
    r"def _in_global_chksum\([^)]*\):\n    y = (\d+)\n", randomized.decode("utf-8")
  )
  good = init_match is not None and int(init_match.group(1)) >= 2**63
  check(
    "checksum: the exit block seeds a very large constant",
    good,
    randomized.decode("utf-8"),
  )
  ok = ok and good

  # (2) The accumulator's let-declaration keeps its zero: only the exit
  # block's initialiser carries the seed.
  decl_fixture = (
    "def _cast_int(v, m):\n"
    "    v &= (1 << m) - 1\n"
    "    return v - (1 << m) if v >= 1 << (m - 1) else v\n"
    "\n"
    "def _in_check_chksum(expected, actual):\n"
    "    if expected != actual:\n"
    "        raise ValueError('@check_chksum mismatch')\n"
    "    return actual\n"
    "\n"
    "def func_t(pa0):\n"
    "    v__chk = 0\n"
    "    # ^entry\n"
    "    v1 = (pa0 + 1)\n"
    "    # ^exit\n"
    "    v__chk = 0\n"
    "    v__chk = (v__chk + v1)\n"
    "    v__chk = (v__chk + pa0)\n"
    "    return v__chk\n"
    "\n"
    "def main():\n"
    "    r = func_t(7)\n"
    "    r = _in_check_chksum(123, r)\n"
    "    return 0\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    import sys\n"
    "    sys.exit(main())\n"
  ).encode("utf-8")
  randomized = mod.randomize_checksum(decl_fixture, 42)
  text = randomized.decode("utf-8")
  init_match = re.search(r"def _in_global_chksum\([^)]*\):\n    y = (\d+)\n", text)
  good = (
    "    v__chk = 0\n    # ^entry\n" in text
    and init_match is not None
    and int(init_match.group(1)) >= 2**63
  )
  check("checksum: the declaration keeps its zero, the exit block seeds", good, text)
  ok = ok and good

  # (3) The multi-example recalibration covers the seed: every example's
  # expectation re-derives from a run of the seeded chain.
  multi_fixture = (
    "def _cast_int(v, m):\n"
    "    v &= (1 << m) - 1\n"
    "    return v - (1 << m) if v >= 1 << (m - 1) else v\n"
    "\n"
    "def _in_check_chksum(expected, actual):\n"
    "    if expected != actual:\n"
    "        raise ValueError('@check_chksum mismatch')\n"
    "    return actual\n"
    "\n"
    "def func_t(pa0, pa1, pa2):\n"
    "    v__chk = 0\n"
    "    v__chk = (v__chk + pa0)\n"
    "    v__chk = (v__chk + pa1)\n"
    "    v__chk = (v__chk + pa2)\n"
    "    return v__chk\n"
    "\n"
    "def main():\n"
    "    r = func_t(3, 5, 7)\n"
    "    r = _in_check_chksum(123, r)\n"
    "    r = func_t(-2, 6, 5)\n"
    "    r = _in_check_chksum(134, r)\n"
    "    r = func_t(-7, 7, 3)\n"
    "    r = _in_check_chksum(145, r)\n"
    "    return 0\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    import sys\n"
    "    sys.exit(main())\n"
  ).encode("utf-8")
  randomized = mod.randomize_checksum(multi_fixture, 42)
  init_match = re.search(
    r"def _in_global_chksum\([^)]*\):\n    y = (\d+)\n",
    randomized.decode("utf-8"),
  )
  good = (
    init_match is not None
    and int(init_match.group(1)) >= 2**63
    and run_module(randomized) == 0
  )
  check(
    "checksum: the seed recalibrates every example (3 checks)",
    good,
    f"rc={run_module(randomized)}\n{randomized.decode('utf-8')}",
  )
  ok = ok and good

  # (4) The seed varies with the master seed.
  a_init_match = re.search(
    r"def _in_global_chksum\([^)]*\):\n    y = (\d+)\n",
    mod.randomize_checksum(fix_src, 42).decode("utf-8"),
  )
  b_init_match = re.search(
    r"def _in_global_chksum\([^)]*\):\n    y = (\d+)\n",
    mod.randomize_checksum(fix_src, 43).decode("utf-8"),
  )
  good = (
    a_init_match is not None
    and b_init_match is not None
    and a_init_match.group(1) != b_init_match.group(1)
    and int(a_init_match.group(1)) >= 2**63
    and int(b_init_match.group(1)) >= 2**63
  )
  check(
    "checksum: the seed varies with the master seed",
    good,
    f"{a_init_match and a_init_match.group(1)} vs "
    f"{b_init_match and b_init_match.group(1)}",
  )
  ok = ok and good

  # (5) Edge case: a non-constant exit initialiser seeds no span, so only
  # the operator rewrite applies, and the recalibration still holds.
  no_init = (
    "def _cast_int(v, m):\n"
    "    v &= (1 << m) - 1\n"
    "    return v - (1 << m) if v >= 1 << (m - 1) else v\n"
    "\n"
    "def _in_check_chksum(expected, actual):\n"
    "    if expected != actual:\n"
    "        raise ValueError('@check_chksum mismatch')\n"
    "    return actual\n"
    "\n"
    "def func_t(pa0, pa1):\n"
    "    v__chk = (0 // 1)\n"
    "    v__chk = (v__chk + pa0)\n"
    "    v__chk = (v__chk + pa1)\n"
    "    return v__chk\n"
    "\n"
    "def main():\n"
    "    r = func_t(3, 5)\n"
    "    r = _in_check_chksum(123, r)\n"
    "    return 0\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    import sys\n"
    "    sys.exit(main())\n"
  ).encode("utf-8")
  randomized = mod.randomize_checksum(no_init, 42)
  good = (
    "    v__chk = (0 // 1)\n" in randomized.decode("utf-8")
    and randomized != no_init
    and run_module(randomized) == 0
  )
  check(
    "checksum: a non-constant exit initialiser keeps only the operator rewrite",
    good,
    str(randomized.decode("utf-8")),
  )
  ok = ok and good

  return ok


def checksum_coeff_tests_pass(ccommon, chk_mod, mod) -> bool:
  """Every chain step lifts a random coefficient into its other operand
  (`chk op coeff*x`): more randomization breadth per step, and the
  wrapped operators keep their guards around the lifted operand."""
  ok = True

  def run_module(src_bytes: bytes) -> int:
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as tf:
      tf.write(src_bytes)
      path = tf.name
    try:
      r = subprocess.run(
        [sys.executable, path], capture_output=True, text=True, timeout=60
      )
      return r.returncode
    finally:
      os.unlink(path)

  fix_src = (
    "def _cast_int(v, m):\n"
    "    v &= (1 << m) - 1\n"
    "    return v - (1 << m) if v >= 1 << (m - 1) else v\n"
    "\n"
    "def _in_check_chksum(expected, actual):\n"
    "    if expected != actual:\n"
    "        raise ValueError('@check_chksum mismatch')\n"
    "    return actual\n"
    "\n"
    "def func_t(pa0, pa1, pa2):\n"
    "    v__chk = 0\n"
    "    v__chk = (v__chk + pa0)\n"
    "    v__chk = (v__chk + pa1)\n"
    "    v__chk = (v__chk + pa2)\n"
    "    return v__chk\n"
    "\n"
    "def main():\n"
    "    r = func_t(3, 5, 7)\n"
    "    r = _in_check_chksum(123, r)\n"
    "    return 0\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    import sys\n"
    "    sys.exit(main())\n"
  ).encode("utf-8")

  # (1) Every chain step lifts a coefficient: each step's operand is
  # `<coeff> * (cN)` inside the extracted function.
  randomized = mod.randomize_checksum(fix_src, 42)
  coeff_matches = re.findall(r"(\d+) \* \(c\d+\)", randomized.decode("utf-8"))
  good = len(coeff_matches) == 3
  check("checksum: every chain step lifts a coefficient", good, str(coeff_matches))
  ok = ok and good

  # (2) Each coefficient stays in its declared range.
  good = all(2 <= int(c) < 10**4 for c in coeff_matches)
  check("checksum: the coefficient stays in range", good, str(coeff_matches))
  ok = ok and good

  # (3) The lifted chain recalibrates: main's check passes.
  good = coeff_matches and run_module(randomized) == 0
  check(
    "checksum: the lifted chain recalibrates and passes",
    good,
    str(run_module(randomized)),
  )
  ok = ok and good

  # (4) The coefficients vary with the master seed.
  a_coeffs = re.findall(
    r"(\d+) \* \(c\d+\)", mod.randomize_checksum(fix_src, 42).decode("utf-8")
  )
  b_coeffs = re.findall(
    r"(\d+) \* \(c\d+\)", mod.randomize_checksum(fix_src, 43).decode("utf-8")
  )
  good = a_coeffs and b_coeffs and a_coeffs != b_coeffs
  check(
    "checksum: the coefficient varies with the master seed",
    good,
    f"{a_coeffs} vs {b_coeffs}",
  )
  ok = ok and good

  # (5) Edge case: the wrapped operators keep the lifted operand inside
  # their guard shapes, so the guards stay well-defined.
  lifted = "3 * (pa0)"
  good = (
    mod._checksum_wrap("//", lifted) == "// (3 * (pa0) | 1)"
    and mod._checksum_wrap("%", lifted) == "% (3 * (pa0) | 1)"
    and mod._checksum_wrap("**", lifted) == "** min(max(3 * (pa0), 0), 64)"
    and mod._checksum_wrap("<<", lifted) == "<< min(max(3 * (pa0), 0), 64)"
    and mod._checksum_wrap("+", lifted) == "+ 3 * (pa0)"
  )
  check("checksum: wrap guards wrap the lifted operand", good, str(good))
  ok = ok and good

  return ok


def checksum_replay_distinct_tests_pass(ccommon, chk_mod, mod) -> bool:
  """Every harness replay carries its own checksum: one replay whose value
  collapses onto a neighbour's shrinks the n-examples strap below its
  example count, so the creator rejects such a case instead of shipping
  fewer anchors than examples.  The printable budget and the fallback
  follow the same gate."""
  ok = True

  def run_module(src_bytes: bytes) -> int:
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as tf:
      tf.write(src_bytes)
      path = tf.name
    try:
      r = subprocess.run(
        [sys.executable, path], capture_output=True, text=True, timeout=60
      )
      return r.returncode
    finally:
      os.unlink(path)

  def harness(chain_operand: str, replays: list[tuple[int, int]]) -> bytes:
    replay = ""
    for i, (a0, a1) in enumerate(replays):
      replay += f"    r = func_t({a0}, {a1})\n"
      # The originals are stale guesses, as in the generated flow: the
      # recalibration re-derives each from a run.
      replay += f"    r = _in_check_chksum({11 * i + 123}, r)\n"
    return (
      "def _cast_int(v, m):\n"
      "    v &= (1 << m) - 1\n"
      "    return v - (1 << m) if v >= 1 << (m - 1) else v\n"
      "\n"
      "def _in_check_chksum(expected, actual):\n"
      "    if expected != actual:\n"
      "        raise ValueError('@check_chksum mismatch')\n"
      "    return actual\n"
      "\n"
      "def func_t(pa0, pa1):\n"
      "    v__chk = 0\n"
      f"    v__chk = (v__chk + {chain_operand})\n"
      "    return v__chk\n"
      "\n"
      "def main():\n" + replay + "    return 0\n"
      "\n"
      'if __name__ == "__main__":\n'
      "    import sys\n"
      "    sys.exit(main())\n"
    ).encode("utf-8")

  # (1) A chain that ignores the differing parameter coordinate collapses
  # every replay onto one checksum: the case is rejected, never kept.
  duplicate = harness("pa1", [(3, 5), (7, 5)])
  got = mod.randomize_checksum(duplicate, 42)
  good = got is None
  check(
    "checksum: a case whose replay checksums duplicate is rejected",
    good,
    f"returned {type(got).__name__}",
  )
  ok = ok and good

  # (2) A chain that follows the differing coordinate spans distinct
  # values: the sampled rewrite applies, and the recalibrated literals
  # stay pairwise distinct.
  varying = harness("pa0", [(3, 5), (7, 5)])
  got = mod.randomize_checksum(varying, 42)
  literals = re.findall(
    r"_in_check_chksum\((-?\d+), r\)", got.decode("utf-8") if got else ""
  )
  rc = run_module(got) if got is not None else -1
  distinct_literals = len(literals) == 2 and literals[0] != literals[1]
  good = (got is not None) and (rc == 0) and distinct_literals
  check(
    "checksum: a distinct chain recalibrates pairwise-distinct literals",
    good,
    f"rc={rc} literals={literals}",
  )
  ok = ok and good

  # (3) Two replays that share the leaf inputs land on the same value
  # under any chain: with three examples the case is rejected too.
  identical = harness("pa0", [(3, 5), (3, 5), (7, 5)])
  got = mod.randomize_checksum(identical, 42)
  good = got is None
  check(
    "checksum: identical replays among three reject the case",
    good,
    f"returned {type(got).__name__}",
  )
  ok = ok and good

  # (4) The printable budget gates the fallback too: with every sampled
  # value overflowing the limit, an original whose own values overflow
  # rejects the case where it used to keep the chain.
  overflow_src = harness("pa0", [(10**32, 5)])
  saved_ops = mod.CHECKSUM_OPS
  mod.CHECKSUM_OPS = (b"**",)
  try:
    got = mod.randomize_checksum(overflow_src, 42)
  finally:
    mod.CHECKSUM_OPS = saved_ops
  good = got is None
  check(
    "checksum: an overflow-only fallback rejects the case",
    good,
    f"returned {type(got).__name__}",
  )
  ok = ok and good

  # (5) Edge case: a leaf the calibration cannot run keeps the original
  # addition chain, whose literals rysmith calibrated.
  broken = harness("pa1", [(3, 5), (7, 5)]).replace(
    b"    v__chk = 0\n", b"    v__chk = 0 // 0\n", 1
  )
  kept = mod.randomize_checksum(broken, 42)
  good = kept == broken
  check(
    "checksum: a failing calibration keeps the addition chain",
    good,
    str(kept == broken),
  )
  ok = ok and good

  return ok


GLOBAL_CHKSUM_SRC = (
  "class RefractIRTrap(Exception):\n"
  "    pass\n"
  "\n"
  "_UNDEF = ['undef']\n"
  "_PAD = ['pad']\n"
  "\n"
  "def _cast_int(v, m):\n"
  "    v &= (1 << m) - 1\n"
  "    return v - (1 << m) if v >= 1 << (m - 1) else v\n"
  "\n"
  "def _rd(buf, off):\n"
  "    v = buf[off]\n"
  "    if v is _UNDEF:\n"
  "        raise RefractIRTrap('read of undef value')\n"
  "    if v is _PAD:\n"
  "        raise RefractIRTrap('access to the interior of a value')\n"
  "    return v\n"
  "\n"
  "def _in_check_chksum(expected, actual):\n"
  "    if expected != actual:\n"
  "        raise RefractIRTrap('@check_chksum mismatch')\n"
  "    return actual\n"
  "\n"
  "def func_t(pa0):\n"
  "    v0 = 5\n"
  "    v1 = -3\n"
  "    t0 = [_UNDEF, _PAD]\n"
  "    v__chk = 0\n"
  "    while True:\n"
  "        # ^entry\n"
  "        v1 = (v1 + pa0)\n"
  "        # ^b0\n"
  "        v1 = (v1 + 2)\n"
  "        # ^b1\n"
  "        v9 = (v0 + 1)\n"
  "        v1 = (v1 - 1)\n"
  "        # ^b2\n"
  "        t0[0] = (v1 + v9)\n"
  "        # ^b3\n"
  "        v1 = (v1 - 2)\n"
  "        # ^exit\n"
  "        v__chk = (v__chk + _cast_int(v0, 32))\n"
  "        v__chk = (v__chk + _cast_int(v1, 32))\n"
  "        v__chk = (v__chk + _cast_int(v9, 32))\n"
  "        v__chk = (v__chk + _rd(t0, 0))\n"
  "        v__chk = (v__chk + pa0)\n"
  "        return v__chk\n"
  "\n"
  "def main():\n"
  "    r = func_t(1)\n"
  "    r = _in_check_chksum(14, r)\n"
  "    r = func_t(2)\n"
  "    r = _in_check_chksum(17, r)\n"
  "    return 0\n"
  "\n"
  'if __name__ == "__main__":\n'
  "    import sys\n"
  "    sys.exit(main())\n"
).encode("utf-8")

GLOBAL_CHKSUM_OPERANDS = [
  "_cast_int(v0, 32)",
  "_cast_int(v1, 32)",
  "_cast_int(v9, 32)",
  "_rd(t0, 0)",
  "pa0",
]


def global_chksum_tests_pass(ccommon, chk_mod, mod) -> bool:
  """The global checksum reuses the exit checksum's arithmetic: mid-path
  `_g[0] = _cast_int(_g[0] <op> coeffs * (operand), 64)` chain steps at
  every K-th visited block (K is the profile-owned `chksum_every` knob),
  and main checks the accumulated total once with the existing
  `_in_check_chksum`. The scaffold stays visible: the step statements are
  skipped by the maskable scan, their operands draw from the exit-chain
  operands' normalization, and the fold reuses `_checksum_wrap`,
  `CHECKSUM_OPS`, and the coefficient range the exit chain samples."""
  ok = True

  def run_module(src_bytes: bytes) -> int:
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as tf:
      tf.write(src_bytes)
      path = tf.name
    try:
      r = subprocess.run(
        [sys.executable, path], capture_output=True, text=True, timeout=60
      )
      return r.returncode
    finally:
      os.unlink(path)

  # (1) The checksum call statements are creator-shaped scaffold: the
  # maskable scan skips them, so the call's operands stay visible while
  # every other statement still masks.
  fix_src = GLOBAL_CHKSUM_SRC
  call_line = (
    b"        _g[0] = _cast_int(_in_global_chksum(_g[0], _cast_int(v0, 32), pa0), 64)\n"
  )
  instr_src = fix_src.replace(b"        # ^b1\n", call_line + b"        # ^b1\n", 1)
  instr_tree = ast.parse(instr_src)
  instr_leaf, _ = ccommon.find_python_leaf_function(instr_tree, instr_src)
  instr_maskable, instr_entry, _ = ccommon.get_python_maskable_statements(
    instr_leaf, instr_src
  )
  chk_stmts = [s for s in instr_maskable if ccommon._is_global_chksum_call(s)]
  instr_repls: list = []
  for stmt in instr_maskable:
    ccommon.collect_python_replacements(
      stmt,
      instr_src,
      stmt.lineno > instr_entry,
      instr_repls,
      ccommon.collect_python_leaf_locals(instr_leaf),
      set(),
    )
  instr_masked = ccommon.apply_replacements(instr_src, instr_repls).decode("utf-8")
  good = (
    not chk_stmts
    and len(instr_maskable) > 3
    and "_g[0] = _cast_int(_in_global_chksum(_g[0], _cast_int(v0, 32), pa0), 64)"
    in instr_masked
    and "<FILL_VAR>" in instr_masked
  )
  check(
    "gchk: the maskable scan skips checksum calls",
    good,
    f"chk_stmts={len(chk_stmts)}\n{instr_masked[-1200:]}",
  )
  ok = ok and good

  # (3) The stride is a profile-owned knob: profiles carry a bounded
  # chksum_every range (minimum at least 1), a direct config must state
  # its stride, sample_config lifts the sampled stride, and a zero range
  # is rejected fail-loudly.
  stride_required = False
  try:
    mod.GeneratorConfig(
      n_bbls=2,
      n_stmts=2,
      min_loop_iter=1,
      max_ptr_depth=0,
      p_backedge=0.5,
      p_branch=0.5,
      n_vars=4,
      n_params=2,
      n_examples=3,
    )
  except TypeError:
    stride_required = True
  check("gchk: a direct config requires its stride", stride_required)
  ok = ok and stride_required

  from dataclasses import fields as dc_fields
  from dataclasses import replace as dc_replace

  missing = [
    name
    for name, prof in mod.PROFILES.items()
    if "chksum_every" not in {f.name for f in dc_fields(prof)}
  ]
  underbounded = [
    name for name, prof in mod.PROFILES.items() if prof.chksum_every.minimum < 1
  ]
  bad_strides = []
  for name, prof in mod.PROFILES.items():
    prof.validate(name)
    rng = random.Random(3)
    for _ in range(20):
      sampled_cfg = mod.sample_config(prof, rng)
      stride = getattr(sampled_cfg, "chksum_every", None)
      if not (prof.chksum_every.minimum <= stride <= prof.chksum_every.maximum):
        bad_strides.append(f"{name}:{stride}")
        break
  good = not missing and not underbounded and not bad_strides
  check(
    "gchk: profiles own the bounded chksum_every stride",
    good,
    f"missing={missing} underbounded={underbounded} bad={bad_strides}",
  )
  ok = ok and good

  zero_ok = False
  try:
    dc_replace(mod.PROFILES["medium"], chksum_every=mod.IntRange(0, 0)).validate(
      "medium-bad-stride"
    )
  except ValueError:
    zero_ok = True
  except Exception as exc:  # noqa: BLE001
    check("gchk: a zero stride range is rejected", False, repr(exc))
    return ok
  check("gchk: a zero stride range is rejected", zero_ok)
  ok = ok and zero_ok

  # (4) Extraction: the exit checksum computation becomes ONE function.
  # `randomize_checksum` splices `_in_global_chksum` (a very large seed plus
  # one guarded, coefficient-lifted fold per positional parameter, folded
  # into the `_g` box) and rewrites the leaf's exit block to call it with
  # the same operand texts the inline chain read. Insertion then calls it
  # at every K-th visited mid-path block where ALL operands are available.
  try:
    operands = mod.collect_global_chk_operands(fix_src)
  except Exception as exc:  # noqa: BLE001
    operands = []
    check("gchk: exit-chain operands extract", False, str(exc))
    return ok
  good = operands == GLOBAL_CHKSUM_OPERANDS
  check(
    "gchk: exit-chain operands extract in chain order",
    good,
    f"got {operands}, want {GLOBAL_CHKSUM_OPERANDS}",
  )
  ok = ok and good

  try:
    extracted = mod.randomize_checksum(fix_src, 42)
  except Exception as exc:  # noqa: BLE001
    check("gchk: exit checksum extracted into a function", False, str(exc))
    return ok
  if extracted is None:
    check("gchk: exit checksum extracted into a function", False, "returned None")
    return ok
  text = extracted.decode("utf-8")
  seed_match = re.search(r"def _in_global_chksum\([^)]*\):\n    y = (\d+)", text)
  good = (
    text.count("def _in_global_chksum") == 1
    and text.count("_g = [0]") == 1
    and seed_match is not None
    and int(seed_match.group(1)) >= 2**63
    # The exit block calls the function (first parameter: the checksum to
    # compute, a fresh 0 there) with the same operand texts; the leaf's
    # per-example checksum stays in v__chk, separate from `_g`.
    and "        v__chk = _in_global_chksum(0, "
    + "_cast_int(v0, 32), _cast_int(v1, 32), "
    "_cast_int(v9, 32), _rd(t0, 0), pa0)\n"
    "        return v__chk\n"
    in text
    and "    return addend + y" in text
    and "_g[0] = _cast_int(_g[0] + y, 64)" not in text
  )
  fold_steps = re.findall(
    r"    y = \(y \S+ \d+ \* \(c\d+\)"
    r"|    y = \(y \S+ \(\d+ \* \(c\d+\) \| 1\)\)"
    r"|    y = \(y \S+ min\(max\(\d+ \* \(c\d+\), 0\), 64\)\)\)",
    text,
  )
  good = good and len(fold_steps) == 5
  check(
    "gchk: the exit checksum extracts into one function",
    good,
    f"fold_steps={len(fold_steps)}\n{text[-1400:]}",
  )
  ok = ok and good
  rc = run_module(extracted)
  good = rc == 0
  check("gchk: the extracted module runs clean", good, f"rc={rc}")
  ok = ok and good

  # K=2 chooses b0 and b2, where not every operand is available (v9 unbound
  # at b0; t0's slot still `_UNDEF` at both): no mid-path call, but the end
  # anchor still checks the exit calls' accumulated total.
  try:
    instrumented = mod.insert_global_chksum(
      extracted, "entry -> b0 -> b1 -> b2 -> b3 -> exit", 2, operands
    )
  except Exception as exc:  # noqa: BLE001
    check("gchk: instrumented module built", False, str(exc))
    return ok
  if instrumented is None:
    check("gchk: instrumented module built", False, "returned None")
    return ok
  itext = instrumented.decode("utf-8")
  call_sites = re.findall(r"_g\[0\] = _cast_int\(_in_global_chksum\(_g\[0\], ", itext)
  global_anchor = re.findall(r"_in_check_chksum\(-?\d+, _g\[0\]\)", itext)
  harness_anchors = chk_mod.count_harness_examples(itext)
  good = (
    len(call_sites) == 0
    and len(global_anchor) == 1
    and harness_anchors == 2
    and len(re.findall(r"    r = _in_check_chksum\(-?\d+, r\)", itext)) == 2
  )
  check(
    "gchk: no qualifying block keeps the end-of-main check alone",
    good,
    f"calls={len(call_sites)} anchor={global_anchor} harness={harness_anchors}",
  )
  ok = ok and good

  # (5) Behavior: the instrumented module (no qualifying block under K=2,
  # so only the end-of-main anchor) exits 0 and the sampling is
  # deterministic. A module whose checksum randomization kept the
  # addition chain degrades untouched.
  anchor_match = re.search(r"_in_check_chksum\((-?\d+), _g\[0\]\)", itext)
  rc = run_module(instrumented)
  good = anchor_match is not None and rc == 0
  check(
    "gchk: the instrumented module runs clean",
    good,
    f"rc={rc} anchor={anchor_match and anchor_match.group(1)}",
  )
  ok = ok and good

  again = mod.insert_global_chksum(
    extracted, "entry -> b0 -> b1 -> b2 -> b3 -> exit", 2, operands
  )
  good = again == instrumented
  check("gchk: repeated insertion is deterministic", good, str(again == instrumented))
  ok = ok and good

  degrades = mod.insert_global_chksum(
    fix_src, "entry -> b0 -> b1 -> b2 -> b3 -> exit", 2, operands
  )
  good = degrades == fix_src
  check(
    "gchk: a module without the checksum function degrades",
    good,
    (degrades or b"")[:400].decode("utf-8", errors="ignore"),
  )
  ok = ok and good

  # (6) Edge cases: a K=1 stride reaches b3, where all operands are
  # finally available (t0's slot concrete after its b2 store); a zero
  # stride and a missing harness anchor fail closed; leaf extraction
  # keeps helper/builtin callees out of the operand leaves.
  every_visit = mod.insert_global_chksum(
    extracted, "entry -> b0 -> b1 -> b2 -> b3 -> exit", 1, operands
  )
  every_text = every_visit.decode("utf-8") if every_visit else ""
  every_calls = len(
    re.findall(r"_g\[0\] = _cast_int\(_in_global_chksum\(_g\[0\], ", every_text)
  )
  good = (
    every_calls == 1
    and every_text.count("_rd(t0, 0)") == 2
    and len(re.findall(r"_in_check_chksum\(-?\d+, _g\[0\]\)", every_text)) == 1
  )
  check(
    "gchk: a K=1 stride calls the function at every visited block",
    good,
    f"calls={every_calls}",
  )
  ok = ok and good

  for stride in (0, -1):
    zero_raise = False
    try:
      mod.insert_global_chksum(extracted, "entry -> b0 -> exit", stride, operands)
    except RuntimeError:
      zero_raise = True
    except Exception as exc:  # noqa: BLE001
      check(f"gchk: a stride of {stride} fails closed", False, repr(exc))
      ok = False
      break
    check(f"gchk: a stride of {stride} fails closed", zero_raise)
    ok = ok and zero_raise

  missing_anchor = extracted.replace(b"    r = _in_check_chksum(", b"    r = _in_nope(")
  anchor_raise = False
  try:
    mod.insert_global_chksum(
      missing_anchor, "entry -> b0 -> b1 -> b2 -> b3 -> exit", 2, operands
    )
  except RuntimeError:
    anchor_raise = True
  except Exception as exc:  # noqa: BLE001
    check("gchk: a missing harness anchor fails closed", False, repr(exc))
    ok = False
  check("gchk: a missing harness anchor fails closed", anchor_raise)
  ok = ok and anchor_raise

  good = (
    mod._operand_leaves("_cast_int(_rd(v0, 0), 32)") == (set(), {("v0", 0)})
    and mod._operand_leaves("_rd(vec1.lanes, 2)") == (set(), {("vec1", 2)})
    and mod._operand_leaves("_in_max(pa0, pa1)") == ({"pa0", "pa1"}, set())
    and mod._operand_leaves("v9") == ({"v9"}, set())
    and mod._operand_leaves("0 // 1") == (set(), set())
    and mod._operand_leaves("_load(p0)") is None
    and mod._operand_leaves("_rd(t0, pa0)") is None
  )
  check("gchk: operand leaves keep helpers out and slots concrete", good, str(good))
  ok = ok and good

  # (7) Shape and indentation edge cases: an empty step list fails loud
  # instead of rendering a dangling signature, the shaped signature carries
  # one positional parameter per step, and the end-of-main anchor indents
  # from the last harness check's own line rather than a hardcoded
  # 4-space body.
  empty_raise = False
  try:
    mod._exit_chksum_function(7, [])
  except RuntimeError:
    empty_raise = True
  except Exception as exc:  # noqa: BLE001
    check("gchk: an empty step list fails loud", False, repr(exc))
    return ok
  check("gchk: an empty step list fails loud", empty_raise)
  ok = ok and empty_raise

  shaped = mod._exit_chksum_function(7, [("+", 3, "pa0")])
  good = (
    "def _in_global_chksum(addend, c0):\n    y = 7\n    y = (y + 3 * (c0))" in shaped
    and "    return addend + y" in shaped
  )
  check("gchk: the shaped signature carries one parameter per step", good, shaped)
  ok = ok and good

  anchors4 = re.findall(r"\n[ ]*_in_check_chksum\(-?\d+, _g\[0\]\)", itext)
  good = len(anchors4) == 1 and anchors4[0].startswith("\n    _")
  check("gchk: the shipped harness keeps its 4-space anchor", good, str(anchors4))
  ok = ok and good

  # A main harness indented by 6 spaces carries its end anchor at 6 spaces
  # and the instrumented module runs clean.
  indented = (
    GLOBAL_CHKSUM_SRC.replace(b"    r = func_t", b"      r = func_t")
    .replace(b"    r = _in_check_chksum", b"      r = _in_check_chksum")
    .replace(b"    return 0\n", b"      return 0\n")
  )
  try:
    indented_extracted = mod.randomize_checksum(indented, 42)
  except Exception as exc:  # noqa: BLE001
    check("gchk: a 6-space main harness extracts", False, str(exc))
    return ok
  if indented_extracted is None:
    check("gchk: a 6-space main harness extracts", False, "returned None")
    return ok
  indented_instrumented = mod.insert_global_chksum(
    indented_extracted, "entry -> b0 -> b1 -> b2 -> b3 -> exit", 2, operands
  )
  iitext = indented_instrumented.decode("utf-8") if indented_instrumented else ""
  anchors6 = re.findall(r"\n[ ]*_in_check_chksum\(-?\d+, _g\[0\]\)", iitext)
  good = len(anchors6) == 1 and anchors6[0].startswith("\n      _")
  check("gchk: the end anchor indents like the last harness check", good, str(anchors6))
  ok = ok and good
  rc = run_module(indented_instrumented) if indented_instrumented else 1
  good = rc == 0
  check("gchk: a 6-space harness runs clean", good, f"rc={rc}")
  ok = ok and good

  return ok


def example_count_tests_pass(ccommon, chk_mod, mod) -> bool:
  """rysmith's example count is a generator-input knob owned by the profile:
  like the other volume knobs it samples from an IntRange, the sampled
  config lifts it, the rysmith command reads it from the config, and a
  range outside the per-example letter budget is rejected fail-loudly."""
  ok = True
  from dataclasses import fields, replace

  # (1) Every profile owns the count as an IntRange inside the per-example
  # letter budget: tuning a profile's range is free, only bounds are fixed.
  bad_profiles = [
    name
    for name, p in mod.PROFILES.items()
    if not (
      getattr(p, "n_examples", None) is not None
      and 1 <= p.n_examples.minimum
      and p.n_examples.maximum <= 26
    )
  ]
  check(
    "example count: profiles own a bounded n_examples range",
    not bad_profiles,
    str(bad_profiles),
  )
  ok = ok and not bad_profiles

  # (2) The config field sits right below n_params, like the profile's.
  field_names = [f.name for f in fields(mod.GeneratorConfig)]
  good = field_names.index("n_examples") == field_names.index("n_params") + 1
  check(
    "example count: the config field sits below n_params",
    good,
    str(field_names),
  )
  ok = ok and good

  # (3) The config field is required: a direct config must state its ask.
  ask_required = False
  try:
    mod.GeneratorConfig(
      n_bbls=3,
      n_stmts=2,
      min_loop_iter=1,
      max_ptr_depth=0,
      p_backedge=0.4,
      p_branch=0.5,
      n_vars=6,
      n_params=2,
      chksum_every=1,
      lift_consts=False,
    )
  except TypeError:
    ask_required = True
  check("example count: a direct config requires its ask", ask_required)
  ok = ok and ask_required

  # (4) The sampled config lifts the profile's sampled count: whatever the
  # profile tunes, the config draws from its range.
  config = mod.sample_config(mod.PROFILES["medium"], random.Random(42))
  profile_range = getattr(mod.PROFILES["medium"], "n_examples", None)
  profile_lo = getattr(profile_range, "minimum", None)
  profile_hi = getattr(profile_range, "maximum", None)
  count = getattr(config, "n_examples", None)
  good = (
    None not in (count, profile_lo, profile_hi) and profile_lo <= count <= profile_hi
  )
  check("example count: sample_config lifts the profile's count", good, str(count))
  ok = ok and good

  # (5) The rysmith command reads --n-examples from the config.
  cmd = mod.build_rysmith_command(config, 42, ".", "rysmith")
  good = cmd[cmd.index("--n-examples") + 1] == str(getattr(config, "n_examples", "?5"))
  check(
    "example count: the rysmith command reads the config",
    good,
    str(cmd[cmd.index("--n-examples") + 1]),
  )
  ok = ok and good

  # (6) Edge: the range bounds to the per-example letter budget rysmith
  # draws from (a..z), so a profile outside [1, 26] is rejected.
  for bad in (0, 27):
    raised = None
    try:
      replace(mod.PROFILES["medium"], n_examples=mod.IntRange(bad, bad)).validate(
        "medium-bad"
      )
    except Exception as exc:  # noqa: BLE001
      raised = exc
    good = isinstance(raised, ValueError)
    check(
      f"example count: a profile with n_examples [{bad}, {bad}] is rejected",
      good,
      repr(raised),
    )
    ok = ok and good

  return ok


def examples_complexity_tests_pass(cmod, mod) -> bool:
  """The example count is a realized property of the puzzle file: the
  metrics measure the harness's anchor count, the trace axis scales with
  it, and the profile's own example-count range answers to the shipped
  harness."""
  ok = True

  def harness_file(n_examples: int) -> str:
    replay = ""
    for i in range(n_examples):
      replay += f"    r = func_t({3 - i})\n"
      replay += f"    r = _in_check_chksum({11 * i + 123}, r)\n"
    return (
      "#//@ EXEC_PATH: entry -> exit\n"
      "def _cast_int(v, m):\n"
      "    v &= (1 << m) - 1\n"
      "    return v - (1 << m) if v >= 1 << (m - 1) else v\n"
      "\n"
      "def _in_check_chksum(expected, actual):\n"
      "    if expected != actual:\n"
      "        raise ValueError('@check_chksum mismatch')\n"
      "    return actual\n"
      "\n"
      "def func_t(pa):\n"
      "    return (pa + 1)\n"
      "\n"
      "def main():\n" + replay + "    return 0\n"
      "\n"
      'if __name__ == "__main__":\n'
      "    import sys\n"
      "    sys.exit(main())\n"
    )

  profile = mod.GenerationProfile(
    n_bbls=mod.IntRange(2, 4),
    n_stmts=mod.IntRange(2, 3),
    min_loop_iter=mod.IntRange(0, 1),
    p_mask_lhs_vars=mod.FloatRange(0.1, 0.1),
    p_mask_rhs_vars=mod.FloatRange(0.75, 0.75),
    p_mask_ops=mod.FloatRange(0.75, 0.75),
    p_mask_funcs=mod.FloatRange(1.0, 1.0),
    p_mask_consts=mod.FloatRange(0.75, 0.75),
    max_ptr_depth=mod.IntRange(0, 0),
    p_backedge=mod.FloatRange(0.1, 0.3),
    p_branch=mod.FloatRange(0.3, 0.5),
    n_vars=mod.IntRange(6, 10),
    n_params=mod.IntRange(2, 3),
    n_examples=mod.IntRange(3, 5),
    chksum_every=mod.IntRange(3, 3),
  )

  tmpdir = tempfile.mkdtemp(prefix="codoku_examples_")
  try:
    outdir = Path(tmpdir) / "out"
    outdir.mkdir(parents=True)
    (outdir / "puzzle.py").write_text(harness_file(3))
    (Path(tmpdir) / "in_range.py").write_text(harness_file(5))
    (Path(tmpdir) / "out_of_range.py").write_text(harness_file(2))
    analyzed = cmod.analyze_puzzle(outdir / "puzzle.py")
    in_range = cmod.analyze_puzzle(Path(tmpdir) / "in_range.py")
    out_of_range = cmod.analyze_puzzle(Path(tmpdir) / "out_of_range.py")

    # (1) The metrics measure the harness anchor count.
    good = getattr(analyzed, "n_examples", None) == 3
    check("examples: metrics measure the harness anchor count", good, str(good))
    ok = ok and good

    # (2) flattened() carries n_examples.
    good = analyzed.flattened().get("n_examples") == 3
    check("examples: flattened carries n_examples", good, str(good))
    ok = ok and good

    # (3) The trace axis multiplies the per-example dynamics by the count.
    est = cmod.estimate_complexity(analyzed)
    per_example = 0.6 * analyzed.exec_path_length + 0.5 * analyzed.loop_iterations_avg
    good = est.dynamic_trace == round(
      getattr(analyzed, "n_examples", 0) * per_example, 2
    )
    check("examples: the trace axis counts examples", good, str(est))
    ok = ok and good

    # (4) The realized example count answers only the acceptance range a
    # profile lists: the knob bounds the ask to rysmith, and rysmith's own
    # realization is approximate, so an unlisted profile gates nothing.
    accepted_hi, _ = mod.profile_accepts(profile, in_range)
    accepted_lo, silent_failures = mod.profile_accepts(profile, out_of_range)
    good = accepted_hi and accepted_lo and silent_failures == []
    check(
      "examples: an unlisted count gates nothing",
      good,
      f"hi={accepted_hi} lo={accepted_lo}{silent_failures}",
    )
    ok = ok and good

    # (5) A listed count gates once with the flat failure message.
    from dataclasses import replace as dc_replace

    listed = dc_replace(profile, acceptance={"n_examples": (3, 5)})
    listed_hi, listed_hi_failures = mod.profile_accepts(listed, in_range)
    listed_lo, listed_failures = mod.profile_accepts(listed, out_of_range)
    good = (
      listed_hi
      and not listed_lo
      and not listed_hi_failures
      and len(listed_failures) == 1
      and listed_failures[0] == "n_examples=2 is outside [3, 5]"
    )
    check(
      "examples: a listed count ships one drift message",
      good,
      str(listed_failures),
    )
    ok = ok and good
  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

  return ok


def main():
  if len(sys.argv) < 3:
    print("usage: run_codoku_tests.py <codoku.py> <rysmith>")
    return 2
  codoku_src, rysmith = sys.argv[1:3]

  mod = import_codoku(codoku_src)
  chk_mod = import_codoku_checker(codoku_src)
  unit_tests_pass(mod)
  checker_unit_tests_pass(chk_mod)
  ccommon = import_codoku_common(codoku_src)
  goto_flag_unit_tests_pass(ccommon, chk_mod)
  sir_extractor_tests_pass(ccommon, chk_mod)
  complexity_unit_tests_pass(import_codoku_complexity(codoku_src))
  examples_complexity_tests_pass(import_codoku_complexity(codoku_src), mod)
  disabled_masks_unit_tests(ccommon, mod)
  trusted_layout_unit_tests_pass(ccommon)
  budget_split_unit_tests_pass(ccommon, chk_mod)
  livedead_budget_option_unit_tests_pass(ccommon, chk_mod, mod)
  sentinel_unit_tests_pass(ccommon, chk_mod)
  fine_grained_unit_tests_pass(ccommon, chk_mod, mod)
  checksum_unit_tests_pass(ccommon, chk_mod, mod)
  checksum_seed_tests_pass(ccommon, chk_mod, mod)
  checksum_coeff_tests_pass(ccommon, chk_mod, mod)
  multi_example_tests_pass(ccommon, chk_mod)
  multi_example_checksum_tests_pass(ccommon, chk_mod, mod)
  checksum_replay_distinct_tests_pass(ccommon, chk_mod, mod)
  global_chksum_tests_pass(ccommon, chk_mod, mod)
  example_count_tests_pass(ccommon, chk_mod, mod)

  with tempfile.TemporaryDirectory(prefix="codoku_gen_") as workdir:
    # Mirror the image layout: codoku + vendored modules + rysmith in one dir.
    src_dir = os.path.dirname(os.path.realpath(codoku_src))
    for f in (
      "codoku.py",
      "codoku_creator.py",
      "codoku_checker.py",
      "codoku_common.py",
      "codoku_preamble.py",
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

    # (1b) The shipped puzzle's exit checksum lives in the extracted
    # function: it seeds a very large constant instead of zero, and every
    # fold step lifts a random coefficient into its positional parameter.
    seed_match = re.search(
      r"def _in_global_chksum\([^)]*\):\n    y = (\d{10,})\n", ptext
    )
    check(
      "generate seeds the exit checksum",
      seed_match is not None and int(seed_match.group(1)) >= 2**63,
      seed_match and seed_match.group(1),
    )
    check(
      "generate lifts chain coefficients",
      re.search(r"    y = \(y [^\n]*\d+ \* \(c0\)", ptext) is not None,
      re.search(r"def _in_global_chksum\([^)]*\):\n(?:.*\n){3}", ptext),
    )

    # (1c) The profile owns the example count: the @main harness carries one
    # checksum anchor per profile example, and every shipped anchor stays
    # distinct from its neighbours.
    with open(manifest) as f:
      profile_examples = json.load(f)["generator_config"]["n_examples"]
    anchors = re.findall(r"_in_check_chksum\(-?\d+\s*,\s*r\)", ptext)
    check(
      "generate carries one checksum anchor per profile example",
      len(anchors) == profile_examples and profile_examples >= 1,
      f"anchors={len(anchors)} profile={profile_examples}",
    )
    check(
      "generate keeps every anchor distinct",
      len(set(anchors)) == len(anchors),
      str(anchors),
    )

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
      "codoku_preamble.py",
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
      and metrics["hal_volume"] > 0
      and metrics["dep_nodes"] > 0
      and "static_struct" in complexity
      and "vocab" not in complexity
      and "dataflow" not in complexity
      and "size" not in complexity
      and "static_structure" not in complexity
    )
    check("manifest records realized metrics", ok_metrics, str(meta))

    # (5b) The manifest records the realized example count, answering to the
    # profile's own range, and the trace axis scales with it.
    profile_examples = meta["generator_config"]["n_examples"]
    good = metrics.get("n_examples") == profile_examples and profile_examples >= 1
    check("generate records the realized example count", good, str(meta))

    # (6) Invalid profile is rejected.
    r = run_codoku(codoku_bin, ["create", "--profile", "bogus"], workdir)
    check("invalid profile rejected", r.returncode == 2, r.stdout + r.stderr)

    # (7) Creation gates: a candidate whose shape makes the extractor
    # diverge from the SIR CFG (or whose ground-truth trace diverges from
    # the SIR PATH) must be rejected, never emitted as a puzzle.
    from pathlib import Path

    cfg = mod.GeneratorConfig(
      n_bbls=5,
      n_stmts=2,
      min_loop_iter=1,
      p_mask_lhs_vars=1.0,
      p_mask_rhs_vars=1.0,
      p_mask_ops=1.0,
      p_mask_funcs=1.0,
      p_mask_consts=1.0,
      max_ptr_depth=0,
      p_backedge=0.4,
      p_branch=0.5,
      n_vars=6,
      n_params=2,
      n_examples=5,
      chksum_every=3,
      lift_consts=False,
      features=("--no-fp", "--no-vec", "--no-ptrarith", "--no-intrinsics"),
    )
    gate_dir = tempfile.mkdtemp(prefix="codoku_gates_")
    try:
      candidate = None
      error = None
      cand_dir = Path(gate_dir) / "cand"
      cand_dir.mkdir(parents=True, exist_ok=True)
      try:
        candidate = mod.generate_candidate(
          cand_dir, cfg, 31, rysmith_path=Path(rysmith).resolve()
        )
      except Exception as exc:  # noqa: BLE001
        error = exc
      if candidate is None:
        check("creation: seed-shape candidate created", False, str(error))
      else:
        check("creation: seed-shape candidate created", True)

        # The banner must reconstruct the trusted companion .sir banner:
        # CFG edges == the .sir CFG, and the path == the `// PATH:` walk.
        sirs = sorted(cand_dir.glob("func_*.sir"))
        sir_cfg = mod.extract_cfg_from_sir(sirs[-1])
        sir_path_str = mod.extract_path_from_sir(sirs[-1])

        ptext = (cand_dir / "puzzle.py").read_text()
        exp_path, _, declared, _ = chk_mod.parse_puzzle_requirements(ptext)
        path_blocks = [b.strip() for b in sir_path_str.split("->") if b.strip()]
        try:
          gt_path = cand_dir / "puzzle.gt.py"
          trace, rc = chk_mod.run_dumps_trace(gt_path, timeout=60)
          # The harness replays the leaf once per example, so the
          # ground-truth trace is the prescribed path once per example.
          n_examples = chk_mod.count_harness_examples(ptext)
          trace_ok = rc == 0 and trace == path_blocks * n_examples
        except RuntimeError:
          trace_ok = False

        good = (
          set(declared) == set(sir_cfg)
          and exp_path == path_blocks
          and all(e in set(sir_cfg) for e in zip(path_blocks, path_blocks[1:]))
          and trace_ok
        )
        check(
          "creation marks carry the SIR CFG + PATH trace",
          good,
          f"cfg_mismatch={set(declared) != set(sir_cfg)} "
          f"path_mismatch={exp_path != path_blocks} "
          f"trace_mismatch={not trace_ok}",
        )
    finally:
      shutil.rmtree(gate_dir, ignore_errors=True)

  n_fail = sum(1 for _, ok, _ in results if not ok)
  print(f"\n{len(results) - n_fail}/{len(results)} codoku tests passed")
  return 1 if n_fail else 0


if __name__ == "__main__":
  sys.exit(main())

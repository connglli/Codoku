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
import re
import shutil
import subprocess
import sys
import tempfile

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
  print(f"  [{color}{tag}{NC}] {name}" + (f" — {detail}" if detail and not ok else ""))


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
      stmt, mod_bytes, stmt.lineno > entry_line, repls, {}, local_names, defined_funcs
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
    ccommon.collect_python_replacements(
      node, src_bytes, True, word_repls, {}, set(), set()
    )
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
      stmt, go_src, stmt.lineno > go_entry, go_repls, {}, go_locals, set()
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
      stmt, exit_src, stmt.lineno > exit_entry, exit_repls, {}, exit_locals, set()
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
      {},
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
  want = sorted([("_brk_exit", "exit", "exit"), ("_cnt_b0", "b0", "b0")])
  if got != want:
    check("unit: dispatch transfers follow the loop model", False, str(got))
    ok = False
  else:
    check("unit: dispatch transfers follow the loop model", True)

  # (e2) End to end: a mistargeted dispatch fails check_cfg with FAIL_CFG
  # while the well-targeted original passes.
  actual_edges = sorted(ccommon.build_python_cfg(transfer_leaf, transfer_src))
  bad_transfer = transfer_src.replace(b"_brk_exit", b"_brk_b0")
  for src, want_ok in ((transfer_src, True), (bad_transfer, False)):
    t = ast.parse(src)
    node = t.body[0]
    label = "accepts true flag target" if want_ok else "rejects mistargeted flag"
    try:
      chk_mod.check_cfg(node, src, actual_edges)
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
  good_src = (
    b"def func_t(x):\n"
    b"    # ^entry\n"
    b"    _brk_exit = False\n"
    b"    # ^b0\n"
    b"    _cnt_b0 = True\n"
    b"    # ^exit\n"
    b"    return x\n"
  )
  good_edges = [("entry", "b0"), ("b0", "exit")]
  bad_src = good_src.replace(b"_brk_exit", b"_brk_nope")
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
  goto_flag_unit_tests_pass(import_codoku_common(codoku_src), chk_mod)
  complexity_unit_tests_pass(import_codoku_complexity(codoku_src))

  with tempfile.TemporaryDirectory(prefix="codoku_gen_") as workdir:
    # Mirror the image layout: codoku + vendored modules + rysmith in one dir.
    src_dir = os.path.dirname(os.path.realpath(codoku_src))
    for f in (
      "codoku.py",
      "codoku_creator.py",
      "codoku_checker.py",
      "codoku_common.py",
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

    # (6) Invalid profile is rejected.
    r = run_codoku(codoku_bin, ["create", "--profile", "bogus"], workdir)
    check("invalid profile rejected", r.returncode == 2, r.stdout + r.stderr)

  n_fail = sum(1 for _, ok, _ in results if not ok)
  print(f"\n{len(results) - n_fail}/{len(results)} codoku tests passed")
  return 1 if n_fail else 0


if __name__ == "__main__":
  sys.exit(main())

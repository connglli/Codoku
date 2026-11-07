"""SMT-LIB encoding shared by the two solver baselines.

The encoder turns a puzzle into one quantifier-free SMT query whose
variables are the cell choices.  Values are Ints (booleans are 0/1),
mutable lists are flat addresses into one ``(Array Int Int)``, and the
masked program is interpreted along the declared execution path.  The
full-cell baseline keeps every cell symbolic; the template baseline fixes
all non-constant cells and keeps only constants symbolic.

Nonlinear operations on two symbolic values, division, exponentiation, and
bitwise operations use uninterpreted functions so the query stays in a
decidable fragment.  The checker filters models that exploit an
uninterpreted result, and the full-cell baseline blocks those exact cell
choices before asking for another model.  Floats and callees with control
flow are outside the fragment and end the baseline with ``unsupported``.
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field

from .common import (
  Cell,
  PuzzleModel,
  SolverTimeout,
  UnsupportedPuzzle,
  run_smt2,
)

UNDEF = 900000000001
PAD = 900000000002
BASE_START = 100000
MAX_QUERY_BYTES = 256 * 1024 * 1024
MAX_EXPR_BYTES = 8 * 1024 * 1024
SIGN64 = 9223372036854775808
MOD64 = 18446744073709551616

ALL_OPERATOR_ORDER = (
  "**",
  "<<",
  ">>",
  "//",
  "+",
  "-",
  "*",
  "%",
  "&",
  "|",
  "^",
  "==",
  "!=",
  "<=",
  ">=",
  "<",
  ">",
  "is not",
  "is",
  "not",
  "~",
  "in",
  "not in",
  "if",
  "else",
)

BINARY_SYMBOLS = {
  "+",
  "-",
  "*",
  "//",
  "%",
  "**",
  "<<",
  ">>",
  "&",
  "|",
  "^",
}
COMPARISON_SYMBOLS = {"==", "!=", "<", "<=", ">", ">=", "is", "is not"}
UNARY_SYMBOLS = {"+", "-", "~", "not"}


def _limited_domain(domain: list[str], limit: int) -> list[str]:
  """Keep common operators plus a representative spread of the rest."""
  if limit <= 0 or len(domain) <= limit:
    return list(domain)
  priority = {
    "+",
    "-",
    "*",
    "//",
    "%",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "is",
    "is not",
    "not",
    "if",
    "else",
  }
  chosen: list[str] = [value for value in domain if value in priority][:limit]
  seen = set(chosen)
  remaining = [value for value in domain if value not in seen]
  missing = limit - len(chosen)
  if missing > 0 and remaining:
    step = len(remaining) / missing
    for index in range(missing):
      value = remaining[min(len(remaining) - 1, int(index * step))]
      if value not in seen:
        chosen.append(value)
        seen.add(value)
  return chosen


def _int(value: int) -> str:
  return str(value)


def _bool(value: bool) -> str:
  return "1" if value else "0"


def _neg(expr: str) -> str:
  return f"(- {expr})"


def _add(a: str, b: str) -> str:
  return f"(+ {a} {b})"


def _sub(a: str, b: str) -> str:
  return f"(- {a} {b})"


def _mul(a: str, b: str) -> str:
  return f"(* {a} {b})"


def _eq(a: str, b: str) -> str:
  return f"(= {a} {b})"


def _ne(a: str, b: str) -> str:
  return f"(not (= {a} {b}))"


def _lt(a: str, b: str) -> str:
  return f"(< {a} {b})"


def _le(a: str, b: str) -> str:
  return f"(<= {a} {b})"


def _gt(a: str, b: str) -> str:
  return f"(> {a} {b})"


def _ge(a: str, b: str) -> str:
  return f"(>= {a} {b})"


def _and(items: list[str]) -> str:
  if not items:
    return "true"
  if len(items) == 1:
    return items[0]
  result = f"(and {' '.join(items)})"
  if len(result) > MAX_EXPR_BYTES:
    raise SolverTimeout("SMT expression exceeded the size guard")
  return result


def _or(items: list[str]) -> str:
  if not items:
    return "false"
  if len(items) == 1:
    return items[0]
  result = f"(or {' '.join(items)})"
  if len(result) > MAX_EXPR_BYTES:
    raise SolverTimeout("SMT expression exceeded the size guard")
  return result


def _not(expr: str) -> str:
  return f"(not {expr})"


def _ite(cond: str, then: str, other: str) -> str:
  if then == other:
    return then
  result = f"(ite {cond} {then} {other})"
  if len(result) > MAX_EXPR_BYTES:
    raise SolverTimeout("SMT expression exceeded the size guard")
  return result


def _truthy(expr: str) -> str:
  return _ne(expr, "0")


def _int_of(cond: str) -> str:
  return _ite(cond, "1", "0")


LITERAL_INT_RE = __import__("re").compile(r"^-?\d+$")


def _literal(expr: str) -> int | None:
  if LITERAL_INT_RE.match(expr):
    return int(expr)
  return None


def _div(a: str, b: str) -> str:
  """Python floor division, including negative divisors."""
  divisor = _literal(b)
  if divisor == 0:
    return "0"
  quotient = f"(div {a} {b})"
  remainder = f"(mod {a} {b})"
  if divisor is not None and divisor > 0:
    return quotient
  adjust = _and([_ne(remainder, "0"), _lt(b, "0")])
  return _ite(adjust, _sub(quotient, "1"), quotient)


def _mod(a: str, b: str) -> str:
  """Python remainder; the sign follows the divisor."""
  if _literal(b) == 0:
    return "0"
  q = _div(a, b)
  return _sub(a, _mul(q, b))


def _pow(a: str, b: str) -> str:
  base = _literal(a)
  exponent = _literal(b)
  if base is not None and exponent is not None and 0 <= exponent <= 4096:
    if abs(base) > 1 and exponent * abs(base).bit_length() > 8192:
      return f"(to_int (^ {a} {b}))"
    return str(base**exponent)
  return f"(to_int (^ {a} {b}))"


def _shift(symbol: str, a: str, b: str) -> str:
  left = _literal(a)
  count = _literal(b)
  if left is not None and count is not None and 0 <= count < 4096:
    if symbol == "shl":
      return str(left << count)
    return str(left >> count)
  factor = f"(to_int (^ 2 {b}))"
  if symbol == "shl":
    return _mul(a, factor)
  return _div(a, factor)


def _bitwise(symbol: str, a: str, b: str) -> str:
  left, right = _literal(a), _literal(b)
  if left is not None and right is not None:
    if symbol == "band":
      return str(left & right)
    if symbol == "bor":
      return str(left | right)
    if symbol == "bxor":
      return str(left ^ right)
  return f"({symbol}_uf {a} {b})"


def _bitnot(a: str) -> str:
  value = _literal(a)
  if value is not None:
    return str(~value)
  return f"(bnot_uf {a})"


def _pow2(bits: int) -> int:
  return 1 << bits


def _cast_int(expr: str, bits: int) -> str:
  if bits <= 0:
    return expr
  span = _pow2(bits)
  unsigned = f"(mod {expr} {span})"
  return _ite(
    _ge(unsigned, str(span // 2)),
    _sub(unsigned, str(span)),
    unsigned,
  )


class SmtBuilder:
  """Declarations, assertions, and the final query text."""

  def __init__(self, timeout_ms: int = 10000) -> None:
    self.decls: list[str] = [
      "(declare-fun band_uf (Int Int) Int)",
      "(declare-fun bor_uf (Int Int) Int)",
      "(declare-fun bxor_uf (Int Int) Int)",
      "(declare-fun bnot_uf (Int) Int)",
    ]
    self.assertions: list[str] = []
    self.bytes = 0
    self.timeout_ms = timeout_ms
    self._counter = 0

  def fresh(self, prefix: str, sort: str = "Int") -> str:
    name = f"{prefix}_{self._counter}"
    self._counter += 1
    self.decls.append(f"(declare-const {name} {sort})")
    return name

  def add(self, assertion: str) -> None:
    if len(assertion) > MAX_EXPR_BYTES:
      raise SolverTimeout("SMT assertion exceeded the size guard")
    self.assertions.append(assertion)
    self.bytes += len(assertion) + 1
    if self.bytes > MAX_QUERY_BYTES:
      raise SolverTimeout(f"SMT query exceeded the {MAX_QUERY_BYTES} byte size guard")

  def query(self, values: list[str]) -> str:
    body = [
      "(set-logic ALL)",
      f"(set-option :timeout {self.timeout_ms})",
      *self.decls,
      *[f"(assert {item})" for item in self.assertions],
      "(check-sat)",
    ]
    if values:
      body.append(f"(get-value ({' '.join(values)}))")
    return "\n".join(body) + "\n"


@dataclass
class _ListLiteral:
  elements: list[str]


@dataclass
class _ObjectLiteral:
  fields: dict[str, str]


@dataclass
class _State:
  mem: str
  ptr_base: str
  ptr_off: str
  ptr_stride: str
  ptr_lo: str
  ptr_hi: str
  ptr_frame: str
  ptr_fields: dict[int, tuple[str, str, str, str, str, str]] = field(
    default_factory=dict
  )
  next_ptr: int = 0
  scalars: dict[str, str] = field(default_factory=dict)
  lists: dict[str, str] = field(default_factory=dict)
  objects: dict[str, dict[str, str]] = field(default_factory=dict)
  returns: str | None = None


class SymbolicEncoder:
  """One SMT query over a chosen subset of the cells."""

  def __init__(
    self,
    model: PuzzleModel,
    *,
    symbolic_kinds: set[str],
    fixed: dict[int, str] | None = None,
    blocked: list[dict[int, str]] | None = None,
    timeout_ms: int = 10000,
    smt_solver: str = "z3",
    deadline: float | None = None,
    domain_limit: int = 0,
  ) -> None:
    self.model = model
    self.deadline = deadline
    self.domain_limit = domain_limit
    self._items: dict[str, list[tuple[str, ast.AST]]] | None = None
    self.symbolic_kinds = set(symbolic_kinds)
    self.fixed = fixed or {}
    self.blocked = blocked or []
    self.smt_solver = smt_solver
    self.builder = SmtBuilder(timeout_ms)
    self.cell_var: dict[int, str] = {}
    self.cell_choices: dict[int, list[str]] = {}
    self.cell_probe_span: dict[tuple[int, int], Cell] = dict(model.cells_by_probe_span)
    self._base = BASE_START
    self._inline_depth = 0
    self._functions = {
      node.name: node
      for node in ast.walk(model.tree)
      if isinstance(node, ast.FunctionDef)
    }
    self._const_cells: list[Cell] = []
    try:
      self._declare_cells()
      self._add_domain_constraints()
      self._add_blocking()
      self._encode_examples()
    except MemoryError as error:
      raise SolverTimeout("symbolic encoding exhausted memory") from error
    except RecursionError as error:
      raise UnsupportedPuzzle("expression nesting is too deep") from error

  def _check_deadline(self) -> None:
    if self.deadline is not None and time.monotonic() > self.deadline:
      raise SolverTimeout("symbolic encoding exceeded its deadline")

  def _bind(self, expr: str, prefix: str, sort: str = "Int") -> str:
    name = self.builder.fresh(prefix, sort)
    self.builder.add(_eq(name, expr))
    return name

  def _maybe_bind(self, expr: str, prefix: str, sort: str = "Int") -> str:
    if len(expr) <= 256 or re.fullmatch(r"-?\d+|[A-Za-z_][A-Za-z0-9_]*", expr):
      return expr
    return self._bind(expr, prefix, sort)

  # -- cell declarations ---------------------------------------------------

  def _declare_cells(self) -> None:
    for cell in self.model.cells:
      if cell.is_constant:
        self._const_cells.append(cell)
        if "CONST" in self.symbolic_kinds and cell.index not in self.fixed:
          var = self.builder.fresh(f"const_{cell.index}")
          self.cell_var[cell.index] = var
          self.cell_choices[cell.index] = list(cell.domain)
        else:
          self.cell_choices[cell.index] = [self.fixed.get(cell.index, cell.domain[0])]
        continue
      if cell.kind in self.symbolic_kinds and cell.index not in self.fixed:
        var = self.builder.fresh(f"cell_{cell.index}")
        self.cell_var[cell.index] = var
        domain = self._op_symbols(cell) if cell.kind == "OP" else cell.domain
        self.cell_choices[cell.index] = _limited_domain(domain, self.domain_limit)
      else:
        value = self.fixed.get(cell.index, cell.domain[0])
        self.cell_choices[cell.index] = [value]

  def _op_symbols(self, cell: Cell) -> list[str]:
    """Syntactically valid operators for this cell's construct."""
    if cell.context == "unary":
      symbols = set(UNARY_SYMBOLS)
    elif cell.context == "comparison":
      symbols = set(COMPARISON_SYMBOLS)
    elif cell.context == "ternary_if":
      symbols = {"if"}
    elif cell.context == "ternary_else":
      symbols = {"else"}
    else:
      symbols = set(BINARY_SYMBOLS) | set(COMPARISON_SYMBOLS)
      if cell.index in self.model.ternary_if_cells:
        symbols.add("if")
      if cell.index in self.model.ternary_else_cells:
        symbols.add("else")
    return [symbol for symbol in ALL_OPERATOR_ORDER if symbol in symbols]

  def _add_domain_constraints(self) -> None:
    for cell in self.model.cells:
      if cell.index not in self.cell_var:
        if cell.is_constant:
          pass
        continue
      var = self.cell_var[cell.index]
      if cell.is_constant:
        if not self.model.const_budget:
          self.builder.add(_ge(var, "0"))
          self.builder.add(_ne(var, "0"))
          self.builder.add(_ne(var, "1"))
          continue
        values = self.cell_choices[cell.index]
        if any(not re.fullmatch(r"-?\d+", value) for value in values):
          raise UnsupportedPuzzle("floating-point constant")
        self.builder.add(_or([_eq(var, value) for value in values]))
      else:
        choices = self.cell_choices[cell.index]
        self.builder.add(_ge(var, "0"))
        self.builder.add(_lt(var, str(len(choices))))
    if "CONST" not in self.symbolic_kinds:
      return
    self._add_constant_counts()

  def _add_constant_counts(self) -> None:
    if not self.model.const_budget:
      return
    if self.model.const_split:
      groups = {
        "live": [cell for cell in self._const_cells if cell.live],
        "dead": [cell for cell in self._const_cells if not cell.live],
      }
      for region, cells in groups.items():
        counts: dict[str, int] = {}
        for cell in cells:
          if cell.index not in self.cell_var:
            continue
          for value in self.cell_choices[cell.index]:
            counts.setdefault(value, 0)
        for value in (
          self.model.const_live_pool if region == "live" else self.model.const_dead_pool
        ):
          counts[value] = counts.get(value, 0) + 1
        for value, count in counts.items():
          occurrences = [
            _ite(_eq(self.cell_var[cell.index], value), "1", "0")
            for cell in cells
            if cell.index in self.cell_var and value in self.cell_choices[cell.index]
          ]
          if occurrences:
            self.builder.add(_eq(f"(+ {' '.join(occurrences)})", str(count)))
    else:
      values = sorted(set(self.model.const_total_pool))
      for value in values:
        occurrences = [
          _ite(_eq(self.cell_var[cell.index], value), "1", "0")
          for cell in self._const_cells
          if cell.index in self.cell_var and value in self.cell_choices[cell.index]
        ]
        if occurrences:
          self.builder.add(
            _eq(
              f"(+ {' '.join(occurrences)})",
              str(self.model.const_total_pool.count(value)),
            )
          )

  def _add_blocking(self) -> None:
    for assignment in self.blocked:
      conjuncts: list[str] = []
      for index, value in assignment.items():
        if index not in self.cell_var:
          continue
        cell = self.model.cells[index]
        if cell.is_constant:
          conjuncts.append(_eq(self.cell_var[index], value))
        else:
          choices = self.cell_choices[index]
          if value in choices:
            conjuncts.append(_eq(self.cell_var[index], str(choices.index(value))))
      if conjuncts:
        self.builder.add(_not(_and(conjuncts)))

  # -- query ---------------------------------------------------------------

  def _encode_examples(self) -> None:
    examples = self.model.harness_examples
    if not examples:
      raise UnsupportedPuzzle("no harness example could be parsed")
    for example in examples:
      self._check_deadline()
      if any(isinstance(value, float) for value in example.args):
        raise UnsupportedPuzzle("floating-point example input")
      if isinstance(example.expected, float):
        raise UnsupportedPuzzle("floating-point example output")
      state = self._fresh_state()
      self._bind_parameters(state, example.args)
      self._execute_pre_entry(state)
      self._execute_path(state)
      if state.returns is None:
        raise UnsupportedPuzzle("leaf did not return on the prescribed path")
      self.builder.add(_eq(state.returns, str(int(example.expected))))

  def _fresh_state(self) -> _State:
    self._base = BASE_START
    zero = "((as const (Array Int Int)) 0)"
    return _State(
      mem=f"((as const (Array Int Int)) {UNDEF})",
      ptr_base=zero,
      ptr_off=zero,
      ptr_stride=zero,
      ptr_lo=zero,
      ptr_hi=zero,
      ptr_frame=zero,
    )

  def _bind_parameters(self, state: _State, args: list[int | float]) -> None:
    for arg, value in zip(self.model.leaf.args.args, args):
      state.scalars[arg.arg] = str(int(value))

  # -- program execution ---------------------------------------------------

  def _execute_pre_entry(self, state: _State) -> None:
    body = self.model.leaf.body
    if len(body) == 1 and isinstance(body[0], ast.Try):
      body = body[0].body
    for stmt in body:
      if getattr(stmt, "lineno", 0) < self.model.entry_line:
        self._exec_stmt(stmt, state)

  def _execute_path(self, state: _State) -> None:
    self._check_deadline()
    if self._items is None:
      self._items = self._collect_block_items()
    items = self._items
    path = self.model.expected_path
    for index, label in enumerate(path):
      next_label = path[index + 1] if index + 1 < len(path) else None
      for kind, node in items.get(label, []):
        if kind == "stmt":
          self._exec_stmt(node, state)
        elif kind == "if":
          if self._exec_if(node, state, label, next_label):
            break
        elif kind == "while":
          self._exec_while(node, state, label, next_label)
          break

  def _collect_block_items(self) -> dict[str, list[tuple[str, ast.AST]]]:
    items: dict[str, list[tuple[str, ast.AST]]] = {}
    body = self.model.leaf.body
    if len(body) == 1 and isinstance(body[0], ast.Try):
      body = body[0].body

    def add(label: str | None, kind: str, node: ast.AST) -> None:
      if label is None:
        return
      items.setdefault(label, []).append((kind, node))

    def arm_has_comment(stmts: list[ast.stmt], start_line: int) -> bool:
      if not stmts:
        return False
      end_line = max(getattr(stmt, "end_lineno", stmt.lineno) for stmt in stmts)
      return any(
        start_line < comment_line <= end_line
        for _, _, _, comment_line in self.model.comments
      )

    def walk(stmts: list[ast.stmt]) -> None:
      for stmt in stmts:
        if _is_dump_trace(stmt):
          continue
        if isinstance(
          stmt,
          (ast.Assign, ast.AugAssign, ast.Expr, ast.Return, ast.Break, ast.Continue),
        ):
          add(
            _label_for(self.model, stmt.lineno),
            "stmt",
            stmt,
          )
        elif isinstance(stmt, ast.If):
          label = _label_for(self.model, stmt.lineno)
          add(label, "if", stmt)
          if arm_has_comment(stmt.body, stmt.lineno):
            walk(stmt.body)
          if arm_has_comment(stmt.orelse, stmt.lineno):
            walk(stmt.orelse)
        elif isinstance(stmt, ast.While):
          add(_loop_label(self.model, stmt), "while", stmt)
          if arm_has_comment(stmt.body, stmt.lineno):
            walk(stmt.body)
          if arm_has_comment(stmt.orelse, stmt.lineno):
            walk(stmt.orelse)
        elif isinstance(stmt, ast.Try):
          walk(stmt.body)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
          continue

    walk(body)
    for label, entries in items.items():
      entries.sort(
        key=lambda item: (
          getattr(item[1], "lineno", 0),
          getattr(item[1], "col_offset", 0),
        )
      )
    return items

  def _exec_if(
    self,
    node: ast.If,
    state: _State,
    label: str,
    next_label: str | None,
  ) -> bool:
    test = self._eval(node.test, state)
    body_label = self._arm_label(node.body, node.lineno)
    else_label = self._arm_label(node.orelse, node.lineno)
    body_dest = self._inline_destination(node.body, label)
    else_dest = self._inline_destination(node.orelse, label)
    if body_label is not None and body_label == next_label:
      self.builder.add(_truthy(test))
      return True
    if else_label is not None and else_label == next_label:
      self.builder.add(_not(_truthy(test)))
      return True
    if body_dest is not None and body_dest == next_label:
      self.builder.add(_truthy(test))
      self._exec_inline(node.body, state, label, next_label)
      return True
    if else_dest is not None and else_dest == next_label:
      self.builder.add(_not(_truthy(test)))
      self._exec_inline(node.orelse, state, label, next_label)
      return True

    if body_label is None and else_label is None:
      if self._inline_can_reach(node.orelse, next_label, label):
        self.builder.add(_not(_truthy(test)))
        self._exec_inline(node.orelse, state, label, next_label)
        return True
      if self._inline_can_reach(node.body, next_label, label):
        self.builder.add(_truthy(test))
        self._exec_inline(node.body, state, label, next_label)
        return True
      if body_dest is None and else_dest is None:
        if _has_state_effect(node.body) or _has_state_effect(node.orelse):
          raise UnsupportedPuzzle(
            f"inline branch effects are not path-resolved at line "
            f"{getattr(node, 'lineno', '?')}"
          )
      self.builder.add(_not(_truthy(test)))
      if else_dest is not None:
        raise UnsupportedPuzzle(
          f"inline else transfers away from the path at line "
          f"{getattr(node, 'lineno', '?')}"
        )
      self._exec_inline(node.orelse, state, label, next_label)
      return False

    if body_label is None:
      if else_label is not None:
        self.builder.add(_truthy(test))
        if body_dest is not None:
          raise UnsupportedPuzzle(
            f"inline body transfers away from the path at line "
            f"{getattr(node, 'lineno', '?')}"
          )
        self._exec_inline(node.body, state, label, next_label)
        return False
      return False

    if else_label is None:
      if body_dest is None and else_dest is None:
        if _has_state_effect(node.body) or _has_state_effect(node.orelse):
          raise UnsupportedPuzzle(
            f"inline branch effects are not path-resolved at line "
            f"{getattr(node, 'lineno', '?')}"
          )
      self.builder.add(_not(_truthy(test)))
      if else_dest is not None:
        raise UnsupportedPuzzle(
          f"inline else transfers away from the path at line "
          f"{getattr(node, 'lineno', '?')}"
        )
      self._exec_inline(node.orelse, state, label, next_label)
      return False

    if next_label is None:
      self.builder.add(_truthy(test))
      return True
    raise UnsupportedPuzzle(
      f"cannot resolve branch at line {getattr(node, 'lineno', '?')}"
    )

  def _exec_while(
    self,
    node: ast.While,
    state: _State,
    label: str,
    next_label: str | None,
  ) -> None:
    del label
    if isinstance(node.test, ast.Constant) and node.test.value is True:
      return
    test = self._eval(node.test, state)
    body_label = self._arm_label(node.body, node.lineno)
    if body_label is None:
      raise UnsupportedPuzzle(
        f"loop at line {getattr(node, 'lineno', '?')} has no block marker"
      )
    if body_label == next_label:
      self.builder.add(_truthy(test))
    else:
      self.builder.add(_not(_truthy(test)))

  def _exec_inline(
    self,
    stmts: list[ast.stmt],
    state: _State,
    label: str,
    next_label: str | None,
  ) -> None:
    for stmt in stmts:
      if _is_dump_trace(stmt):
        continue
      if isinstance(stmt, ast.If):
        if self._exec_if(stmt, state, label, next_label):
          break
      elif isinstance(stmt, ast.While):
        self._exec_while(stmt, state, label, next_label)
      else:
        self._exec_stmt(stmt, state)

  def _exec_stmt(self, stmt: ast.stmt, state: _State) -> None:
    self._check_deadline()
    if _is_global_chksum_stmt(stmt):
      return
    if isinstance(stmt, ast.Assign):
      value = self._eval(stmt.value, state)
      for target in stmt.targets:
        self._assign(target, value, state)
    elif isinstance(stmt, ast.AugAssign):
      current = self._read_target(stmt.target, state)
      value = self._apply_symbol(
        _binop_symbol(stmt.op), current, self._eval(stmt.value, state)
      )
      self._assign(stmt.target, value, state)
    elif isinstance(stmt, ast.Expr):
      self._eval(stmt.value, state)
    elif isinstance(stmt, ast.Return):
      state.returns = self._eval(stmt.value, state) if stmt.value is not None else "0"
    elif isinstance(stmt, (ast.Break, ast.Continue, ast.Pass)):
      return
    else:
      raise UnsupportedPuzzle(
        f"unsupported statement {type(stmt).__name__} at line "
        f"{getattr(stmt, 'lineno', '?')}"
      )

  def _assign(self, target: ast.AST, value, state: _State) -> None:
    cell = self._cell_for(target)
    if cell is not None:
      if isinstance(value, _ListLiteral):
        value = self._alloc_list(value.elements, state)
      if isinstance(value, _ObjectLiteral):
        self._assign_object(cell, value, state)
        return
      self._write_selected(cell, value, state)
      return
    if isinstance(value, _ListLiteral):
      value = self._alloc_list(value.elements, state)
    if isinstance(value, _ObjectLiteral):
      if not isinstance(target, ast.Name):
        raise UnsupportedPuzzle("object assigned to a non-name target")
      state.objects[target.id] = value.fields
      return
    if isinstance(target, ast.Name):
      state.scalars[target.id] = self._maybe_bind(value, "val")
      return
    if isinstance(target, ast.Subscript):
      address = self._address_of(target.value, state)
      index = self._eval(target.slice, state)
      state.mem = self._bind(
        f"(store {state.mem} {_add(address, index)} {self._maybe_bind(value, 'val')})",
        "mem",
        "(Array Int Int)",
      )
      return
    raise UnsupportedPuzzle("unsupported assignment target")

  def _assign_object(self, cell: Cell, value: _ObjectLiteral, state: _State) -> None:
    if len(self.cell_choices[cell.index]) != 1:
      raise UnsupportedPuzzle("symbolic variable choosing an object")
    name = self.cell_choices[cell.index][0]
    state.objects[name] = value.fields

  def _read_target(self, target: ast.AST, state: _State) -> str:
    cell = self._cell_for(target)
    if cell is not None:
      return self._read_selected(cell, state)
    if isinstance(target, ast.Name):
      return self._name_value(target.id, state)
    if isinstance(target, ast.Subscript):
      address = self._address_of(target.value, state)
      index = self._eval(target.slice, state)
      return f"(select {state.mem} {_add(address, index)})"
    raise UnsupportedPuzzle("unsupported read target")

  # -- memory --------------------------------------------------------------

  def _alloc_list(self, elements: list[str], state: _State) -> str:
    base = str(self._base)
    self._base += 1
    mem = state.mem
    for index, element in enumerate(elements):
      value = self._maybe_bind(element, "elem")
      mem = f"(store {mem} {_add(base, str(index))} {value})"
    state.mem = self._maybe_bind(mem, "mem", "(Array Int Int)")
    return base

  # -- name-cell reads and writes -----------------------------------------

  def _read_selected(self, cell: Cell, state: _State) -> str:
    self._check_deadline()
    choices = self.cell_choices[cell.index]
    if cell.index not in self.cell_var:
      return self._read_name(choices[0], state)
    var = self.cell_var[cell.index]
    result = self._read_name(choices[-1], state)
    for index in range(len(choices) - 2, -1, -1):
      self._check_deadline()
      result = _ite(
        _eq(var, str(index)),
        self._read_name(choices[index], state),
        result,
      )
    return self._maybe_bind(result, "sel")

  def _write_selected(self, cell: Cell, value: str, state: _State) -> None:
    self._check_deadline()
    choices = self.cell_choices[cell.index]
    if cell.index not in self.cell_var:
      self._write_name(choices[0], value, state)
      return
    var = self.cell_var[cell.index]
    if any("[" in choice for choice in choices):
      for index, name in enumerate(choices):
        self._check_deadline()
        if "[" not in name:
          continue
        base, subscript = name.split("[", 1)
        idx = subscript.rstrip("]")
        state.mem = _ite(
          _eq(var, str(index)),
          f"(store {state.mem} {_add(self._list_address(base, state), idx)} {value})",
          state.mem,
        )
      plain = [name for name in choices if "[" not in name]
      for name in plain:
        old = self._plain_value(name, state)
        new = _ite(
          _eq(var, str(_choice_index(cell, name))),
          self._maybe_bind(value, "val"),
          old,
        )
        self._set_plain(name, self._maybe_bind(new, "var"), state)
    else:
      for index, name in enumerate(choices):
        self._check_deadline()
        old = self._plain_value(name, state)
        new = _ite(_eq(var, str(index)), self._maybe_bind(value, "val"), old)
        self._set_plain(name, self._maybe_bind(new, "var"), state)

  def _read_name(self, name: str, state: _State) -> str:
    if "[" in name:
      base, subscript = name.split("[", 1)
      index = subscript.rstrip("]")
      return f"(select {state.mem} {_add(self._list_address(base, state), index)})"
    return self._name_value(name, state)

  def _write_name(self, name: str, value: str, state: _State) -> None:
    if "[" in name:
      base, subscript = name.split("[", 1)
      index = subscript.rstrip("]")
      state.mem = self._bind(
        f"(store {state.mem} {_add(self._list_address(base, state), index)} {value})",
        "mem",
        "(Array Int Int)",
      )
      return
    self._set_plain(name, value, state)

  def _plain_value(self, name: str, state: _State) -> str:
    if name in state.lists:
      return state.lists[name]
    if name in state.scalars:
      return state.scalars[name]
    return "0"

  def _set_plain(self, name: str, value: str, state: _State) -> None:
    if name in state.lists:
      state.lists[name] = value
    else:
      state.scalars[name] = value

  def _name_value(self, name: str, state: _State) -> str:
    if name == "_UNDEF":
      return str(UNDEF)
    if name == "_PAD":
      return str(PAD)
    if name in state.scalars:
      return state.scalars[name]
    if name in state.lists:
      return state.lists[name]
    if name in state.objects:
      raise UnsupportedPuzzle(f"vector object {name!r} used as a scalar")
    # A name bound only on an arm the path did not take is uninitialized
    # at this point.  Keep the query total and let the checker reject any
    # candidate that relies on the placeholder.
    return "0"

  def _list_address(self, name: str, state: _State) -> str:
    if name in state.lists:
      return state.lists[name]
    if name in state.scalars:
      return state.scalars[name]
    return "0"

  # -- expressions ---------------------------------------------------------

  def _eval(self, node: ast.AST, state: _State):
    self._check_deadline()
    cell = self._cell_for(node)
    if cell is not None and cell.is_constant:
      return self.cell_var.get(cell.index, self.cell_choices[cell.index][0])
    if isinstance(node, ast.Constant):
      if isinstance(node.value, bool):
        return _bool(node.value)
      if isinstance(node.value, int):
        return str(node.value)
      if isinstance(node.value, float):
        raise UnsupportedPuzzle("floating-point constant")
      raise UnsupportedPuzzle(f"unsupported literal {node.value!r}")
    if isinstance(node, ast.Name):
      cell = self._cell_for(node)
      if cell is not None:
        return self._read_selected(cell, state)
      return self._name_value(node.id, state)
    if isinstance(node, ast.BinOp):
      return self._eval_binop(node, state)
    if isinstance(node, ast.UnaryOp):
      return self._eval_unary(node, state)
    if isinstance(node, ast.Compare):
      return self._eval_compare(node, state)
    if isinstance(node, ast.BoolOp):
      return self._eval_boolop(node, state)
    if isinstance(node, ast.IfExp):
      test = self._eval(node.test, state)
      body = self._eval(node.body, state)
      other = self._eval(node.orelse, state)
      return _ite(_truthy(test), body, other)
    if isinstance(node, ast.Call):
      return self._eval_call(node, state)
    if isinstance(node, ast.Subscript):
      address = self._address_of(node.value, state)
      index = self._eval(node.slice, state)
      return f"(select {state.mem} {_add(address, index)})"
    if isinstance(node, ast.Attribute):
      if node.attr == "lanes" and isinstance(node.value, ast.Name):
        fields = state.objects.get(node.value.id)
        if fields is None:
          raise UnsupportedPuzzle(f"unknown vector {node.value.id!r}")
        return fields["lanes"]
      raise UnsupportedPuzzle(f"unsupported attribute .{node.attr}")
    if isinstance(node, ast.List):
      return _ListLiteral([self._eval(element, state) for element in node.elts])
    if isinstance(node, ast.Tuple):
      return [self._eval(element, state) for element in node.elts]
    raise UnsupportedPuzzle(
      f"unsupported expression {type(node).__name__} at line "
      f"{getattr(node, 'lineno', '?')}"
    )

  def _address_of(self, node: ast.AST, state: _State) -> str:
    if isinstance(node, ast.Attribute) and node.attr == "lanes":
      return self._eval(node, state)
    if isinstance(node, ast.Name):
      cell = self._cell_for(node)
      if cell is not None:
        return self._read_selected(cell, state)
      return self._name_value(node.id, state)
    return self._eval(node, state)

  def _eval_binop(self, node: ast.BinOp, state: _State) -> str:
    left_cell = self._op_cell(node.left, node.right)
    inner_cell = None
    if isinstance(node.left, ast.BinOp):
      inner_cell = self._op_cell(node.left.left, node.left.right)
    if inner_cell is not None and self._is_ternary_chain(node, inner_cell):
      return self._eval_ternary_chain(node, state)
    left = self._eval(node.left, state)
    right = self._eval(node.right, state)
    if left_cell is None:
      return self._apply_symbol(_binop_symbol(node.op), left, right)
    return self._apply_binop(left_cell, left, right)

  def _eval_ternary_chain(self, node: ast.BinOp, state: _State) -> str:
    inner = node.left
    assert isinstance(inner, ast.BinOp)
    inner_cell = self._op_cell(inner.left, inner.right)
    outer_cell = self._op_cell(node.left, node.right)
    if inner_cell is None or outer_cell is None:
      raise UnsupportedPuzzle("ternary chain without cell operators")
    body = self._eval(inner.left, state)
    test = self._eval(inner.right, state)
    other = self._eval(node.right, state)
    inner_choices = self.cell_choices[inner_cell.index]
    outer_choices = self.cell_choices[outer_cell.index]
    result = "0"
    for i in range(len(inner_choices) - 1, -1, -1):
      for j in range(len(outer_choices) - 1, -1, -1):
        self._check_deadline()
        if inner_choices[i] == "if" and outer_choices[j] == "else":
          branch = _ite(_truthy(test), body, other)
        elif inner_choices[i] in BINARY_SYMBOLS and outer_choices[j] in BINARY_SYMBOLS:
          branch = self._apply_symbol(
            outer_choices[j],
            self._apply_symbol(inner_choices[i], body, test),
            other,
          )
        else:
          branch = "0"
        cond = _and(
          [
            _eq(self.cell_var[inner_cell.index], str(i)),
            _eq(self.cell_var[outer_cell.index], str(j)),
          ]
        )
        result = _ite(cond, branch, result)
    return result

  def _is_ternary_chain(self, node: ast.BinOp, inner_cell: Cell) -> bool:
    outer_cell = self._op_cell(node.left, node.right)
    if outer_cell is None:
      return False
    return (
      "else" in self.cell_choices[outer_cell.index]
      and "if" in self.cell_choices[inner_cell.index]
    )

  def _apply_binop(self, cell: Cell | None, left: str, right: str) -> str:
    self._check_deadline()
    if cell is None:
      raise UnsupportedPuzzle("operator cell was not inferred")
    if cell.index not in self.cell_var:
      return self._apply_symbol(self.cell_choices[cell.index][0], left, right)
    choices = self.cell_choices[cell.index]
    result = "0"
    for index in range(len(choices) - 1, -1, -1):
      self._check_deadline()
      branch = self._apply_symbol(choices[index], left, right)
      result = _ite(_eq(self.cell_var[cell.index], str(index)), branch, result)
    return self._maybe_bind(result, "op")

  def _apply_symbol(self, symbol: str, left: str, right: str) -> str:
    if symbol == "+":
      return _add(left, right)
    if symbol == "-":
      return _sub(left, right)
    if symbol == "*":
      return _mul(left, right)
    if symbol == "/":
      raise UnsupportedPuzzle("true division is outside the integer fragment")
    if symbol == "//":
      return _div(left, right)
    if symbol == "%":
      return _mod(left, right)
    if symbol == "**":
      return _pow(left, right)
    if symbol == "<<":
      return _shift("shl", left, right)
    if symbol == ">>":
      return _shift("shr", left, right)
    if symbol == "&":
      return _bitwise("band", left, right)
    if symbol == "|":
      return _bitwise("bor", left, right)
    if symbol == "^":
      return _bitwise("bxor", left, right)
    if symbol in COMPARISON_SYMBOLS:
      return self._apply_comparison(symbol, left, right)
    raise UnsupportedPuzzle(f"operator {symbol!r} is outside the fragment")

  def _apply_comparison(self, symbol: str, left: str, right: str) -> str:
    if symbol == "==" or symbol == "is":
      return _int_of(_eq(left, right))
    if symbol == "!=" or symbol == "is not":
      return _int_of(_ne(left, right))
    if symbol == "<":
      return _int_of(_lt(left, right))
    if symbol == "<=":
      return _int_of(_le(left, right))
    if symbol == ">":
      return _int_of(_gt(left, right))
    if symbol == ">=":
      return _int_of(_ge(left, right))
    raise UnsupportedPuzzle(f"comparison {symbol!r} is outside the fragment")

  def _eval_unary(self, node: ast.UnaryOp, state: _State) -> str:
    operand = self._eval(node.operand, state)
    cell = self._cell_for(node)
    if cell is None:
      span = _unary_span(self.model, node)
      cell = self.cell_probe_span.get(span) if span else None
    if cell is None:
      return self._apply_unary(_unary_symbol(node.op), operand)
    if cell.index not in self.cell_var:
      return self._apply_unary(self.cell_choices[cell.index][0], operand)
    choices = self.cell_choices[cell.index]
    result = "0"
    for index in range(len(choices) - 1, -1, -1):
      self._check_deadline()
      branch = self._apply_unary(choices[index], operand)
      result = _ite(_eq(self.cell_var[cell.index], str(index)), branch, result)
    return self._maybe_bind(result, "un")

  def _apply_unary(self, symbol: str, operand: str) -> str:
    if symbol == "+":
      return operand
    if symbol == "-":
      return _neg(operand)
    if symbol == "~":
      return _bitnot(operand)
    if symbol == "not":
      return _int_of(_eq(operand, "0"))
    raise UnsupportedPuzzle(f"unary operator {symbol!r} is outside the fragment")

  def _eval_compare(self, node: ast.Compare, state: _State) -> str:
    left = self._eval(node.left, state)
    parts: list[str] = []
    previous = node.left
    current = left
    for op, comparator in zip(node.ops, node.comparators):
      right = self._eval(comparator, state)
      span = _compare_op_span_between(self.model.probe_bytes, previous, comparator)
      cell = self.cell_probe_span.get(span) if span else None
      if cell is None:
        symbol = _compare_symbol(op)
        parts.append(self._apply_comparison(symbol, current, right))
      else:
        parts.append(self._apply_binop(cell, current, right))
      previous = comparator
      current = right
    return _int_of(_and([_truthy(part) for part in parts]))

  def _eval_boolop(self, node: ast.BoolOp, state: _State) -> str:
    values = [self._eval(value, state) for value in node.values]
    if isinstance(node.op, ast.And):
      result = values[-1]
      for value in reversed(values[:-1]):
        result = _ite(_truthy(value), result, value)
      return result
    result = values[-1]
    for value in reversed(values[:-1]):
      result = _ite(_truthy(value), value, result)
    return result

  def _alloc_ptr(
    self,
    state: _State,
    base: str,
    off: str,
    stride: str,
    lo: str,
    hi: str,
    frame: str,
  ) -> str:
    pid = state.next_ptr
    state.next_ptr += 1
    fields = (base, off, stride, lo, hi, frame)
    state.ptr_fields[pid] = fields
    for name, value in zip(
      ("ptr_base", "ptr_off", "ptr_stride", "ptr_lo", "ptr_hi", "ptr_frame"),
      fields,
    ):
      array = getattr(state, name)
      setattr(
        state,
        name,
        self._bind(f"(store {array} {pid} {value})", name, "(Array Int Int)"),
      )
    return str(pid)

  def _ptr_fields(
    self, state: _State, pointer: str
  ) -> tuple[str, str, str, str, str, str]:
    pid = _literal(pointer)
    if pid is not None and pid in state.ptr_fields:
      return state.ptr_fields[pid]
    return tuple(
      f"(select {getattr(state, name)} {pointer})"
      for name in (
        "ptr_base",
        "ptr_off",
        "ptr_stride",
        "ptr_lo",
        "ptr_hi",
        "ptr_frame",
      )
    )

  def _ptr_address(self, state: _State, pointer: str) -> str:
    base, off, _, _, _, _ = self._ptr_fields(state, pointer)
    return _add(base, off)

  def _eval_ptr_call(self, node: ast.Call, state: _State) -> str:
    def argument(index: int) -> str:
      if index >= len(node.args):
        return "0"
      value = node.args[index]
      if isinstance(value, ast.Constant) and value.value is None:
        return "0"
      return self._eval(value, state)

    base = argument(0)
    if isinstance(base, _ListLiteral):
      base = self._alloc_list(base.elements, state)
    elif isinstance(base, _ObjectLiteral):
      raise UnsupportedPuzzle("_Ptr of an object literal")
    return self._alloc_ptr(
      state,
      base,
      argument(1),
      argument(2),
      argument(3),
      argument(4),
      argument(5),
    )

  def _eval_call(self, node: ast.Call, state: _State):
    if isinstance(node.func, ast.Name):
      name = node.func.id
      cell = self._cell_for(node.func)
      if cell is not None:
        if len(self.cell_choices[cell.index]) != 1:
          raise UnsupportedPuzzle("symbolic function choice")
        name = self.cell_choices[cell.index][0]
      if name == "_Ptr":
        return self._eval_ptr_call(node, state)
      args = [self._eval(arg, state) for arg in node.args]
      if name == "_cast_int":
        try:
          width = int(args[1])
        except ValueError as error:
          raise UnsupportedPuzzle("symbolic _cast_int width") from error
        return _cast_int(args[0], width)
      if name in {"_rd", "_vrd"}:
        if name == "_vrd":
          try:
            count = int(args[2])
            stride = int(args[3])
          except ValueError as error:
            raise UnsupportedPuzzle("symbolic _vrd extent") from error
          return _ListLiteral(
            [
              f"(select {state.mem} {_add(args[0], str(k * stride))})"
              for k in range(count)
            ]
          )
        return f"(select {state.mem} {_add(args[0], args[1])})"
      if name == "_load":
        return f"(select {state.mem} {self._ptr_address(state, args[0])})"
      if name == "_store":
        value = self._maybe_bind(args[1], "val")
        state.mem = self._bind(
          f"(store {state.mem} {self._ptr_address(state, args[0])} {value})",
          "mem",
          "(Array Int Int)",
        )
        return args[1]
      if name == "_derive":
        base, _, _, _, _, frame = self._ptr_fields(state, args[0])
        return self._alloc_ptr(state, base, args[1], args[2], args[3], args[4], frame)
      if name in {
        "_idx",
        "_need_int",
        "_live",
        "_chk_ptr",
        "_chk_deref",
        "_chk_geom",
      }:
        return args[0] if args else "0"
      if name == "_padd":
        base, off, stride, lo, hi, frame = self._ptr_fields(state, args[0])
        return self._alloc_ptr(
          state,
          base,
          _add(off, _mul(args[1], stride)),
          stride,
          lo,
          hi,
          frame,
        )
      if name == "_pfield":
        base, off, _, _, _, frame = self._ptr_fields(state, args[0])
        return self._alloc_ptr(
          state,
          base,
          _add(off, args[1]),
          args[2],
          off,
          _add(off, args[3]),
          frame,
        )
      if name == "_pdiff":
        _, off_p, stride_p, _, _, _ = self._ptr_fields(state, args[0])
        _, off_q, _, _, _, _ = self._ptr_fields(state, args[1])
        return _div(_sub(off_p, off_q), stride_p)
      if name == "_peq":
        base_p, off_p, _, _, _, _ = self._ptr_fields(state, args[0])
        base_q, off_q, _, _, _, _ = self._ptr_fields(state, args[1])
        return _int_of(_and([_eq(base_p, base_q), _eq(off_p, off_q)]))
      if name == "_prel":
        _, off_p, _, _, _, _ = self._ptr_fields(state, args[0])
        _, off_q, _, _, _, _ = self._ptr_fields(state, args[1])
        return _sub(off_p, off_q)
      if name == "_pidx":
        base, off, _, _, _, frame = self._ptr_fields(state, args[0])
        stride = args[3]
        return self._alloc_ptr(
          state,
          base,
          _add(off, _mul(args[1], stride)),
          stride,
          off,
          _add(off, _mul(args[2], stride)),
          frame,
        )
      if name == "min":
        return _ite(_le(args[0], args[1]), args[0], args[1])
      if name == "max":
        return _ite(_ge(args[0], args[1]), args[0], args[1])
      if name == "abs":
        return _ite(_lt(args[0], "0"), _neg(args[0]), args[0])
      if name == "int":
        return args[0]
      if name.startswith("_vec_"):
        lanes = args[0]
        if not isinstance(lanes, _ListLiteral):
          raise UnsupportedPuzzle("vector constructor without a list literal")
        base = self._alloc_list(lanes.elements, state)
        return _ObjectLiteral({"lanes": base})
      if name in self._functions:
        return self._inline_call(name, args, state)
      raise UnsupportedPuzzle(f"call to unsupported function {name!r}")
    if isinstance(node.func, ast.Attribute):
      raise UnsupportedPuzzle("attribute calls are outside the fragment")
    raise UnsupportedPuzzle("indirect calls are outside the fragment")

  def _inline_call(self, name: str, args: list, state: _State) -> str:
    if self._inline_depth >= 8:
      raise UnsupportedPuzzle("function inlining is too deep")
    function = self._functions[name]
    if any(
      isinstance(node, (ast.If, ast.While, ast.For, ast.Try))
      for node in ast.walk(function)
    ):
      raise UnsupportedPuzzle(f"call to control-flow function {name!r}")
    child = _State(
      mem=state.mem,
      ptr_base=state.ptr_base,
      ptr_off=state.ptr_off,
      ptr_stride=state.ptr_stride,
      ptr_lo=state.ptr_lo,
      ptr_hi=state.ptr_hi,
      ptr_frame=state.ptr_frame,
      ptr_fields=dict(state.ptr_fields),
      next_ptr=state.next_ptr,
    )
    for parameter, value in zip(function.args.args, args):
      if isinstance(value, _ListLiteral):
        value = self._alloc_list(value.elements, child)
      child.scalars[parameter.arg] = value
    self._inline_depth += 1
    try:
      for stmt in function.body:
        self._exec_stmt(stmt, child)
    finally:
      self._inline_depth -= 1
    state.mem = child.mem
    state.ptr_base = child.ptr_base
    state.ptr_off = child.ptr_off
    state.ptr_stride = child.ptr_stride
    state.ptr_lo = child.ptr_lo
    state.ptr_hi = child.ptr_hi
    state.ptr_frame = child.ptr_frame
    state.ptr_fields = child.ptr_fields
    state.next_ptr = child.next_ptr
    if child.returns is None:
      raise UnsupportedPuzzle(f"function {name!r} has no return")
    return child.returns

  def _cell_for(self, node: ast.AST) -> Cell | None:
    if not hasattr(node, "lineno"):
      return None
    try:
      span = (
        _byte_offset(
          self.model.probe_bytes,
          node.lineno,
          node.col_offset,
        ),
        _byte_offset(
          self.model.probe_bytes,
          node.end_lineno,
          node.end_col_offset,
        ),
      )
    except AttributeError:
      return None
    return self.cell_probe_span.get(span)

  def _op_cell(self, left: ast.AST, right: ast.AST) -> Cell | None:
    span = _binop_span(self.model.probe_bytes, left, right)
    if span is None:
      return None
    return self.cell_probe_span.get(span)

  def _arm_label(self, stmts: list[ast.stmt], start_line: int) -> str | None:
    if not stmts:
      return None
    end_line = max(getattr(stmt, "end_lineno", stmt.lineno) for stmt in stmts)
    for _, _, label, line in self.model.comments:
      if start_line < line <= end_line:
        return label
    return None

  def _inline_destination(self, stmts: list[ast.stmt], label: str) -> str | None:
    loop = _innermost_loop(self.model.leaf, stmts[0].lineno) if stmts else None
    for stmt in stmts:
      keyword = _first_transfer(stmt)
      if keyword is None:
        continue
      if isinstance(keyword, ast.Break) and loop is not None:
        return _exit_label(self.model, loop)
      if isinstance(keyword, ast.Continue) and loop is not None:
        return _loop_label(self.model, loop)
      if isinstance(keyword, ast.Return):
        return "exit"
    return None

  def _inline_can_reach(
    self, stmts: list[ast.stmt], next_label: str | None, label: str
  ) -> bool:
    if next_label is None:
      return False
    if self._inline_destination(stmts, label) == next_label:
      return True
    for stmt in stmts:
      if isinstance(stmt, ast.If):
        if self._arm_label(stmt.body, stmt.lineno) == next_label:
          return True
        if self._arm_label(stmt.orelse, stmt.lineno) == next_label:
          return True
    return False

  # -- solving -------------------------------------------------------------

  def solve(self) -> dict[int, str] | None:
    names = [
      self.cell_var[cell.index]
      for cell in self.model.cells
      if cell.index in self.cell_var
    ]
    query = self.builder.query(names)
    if self.smt_solver:
      stdout, stderr = run_smt2(
        query,
        smt_solver=self.smt_solver,
        timeout=max(0.1, self.builder.timeout_ms / 1000.0),
      )
    else:
      raise UnsupportedPuzzle("no SMT solver configured")
    first = stdout.splitlines()[0].strip() if stdout.splitlines() else ""
    if first == "unsat":
      return None
    if first != "sat":
      raise SolverTimeout(
        f"{self.smt_solver} did not decide the query: {first or stderr.strip()}"
      )
    values: dict[int, str] = {}
    for cell in self.model.cells:
      if cell.index not in self.cell_var:
        values[cell.index] = self.cell_choices[cell.index][0]
        continue
      var = self.cell_var[cell.index]
      match = re.search(rf"\(\s*{re.escape(var)}\s+(-?\d+)\s*\)", stdout)
      if match is not None:
        number = int(match.group(1))
      else:
        negated = re.search(rf"\(\s*{re.escape(var)}\s+\(-\s+(\d+)\)\s*\)", stdout)
        if negated is None:
          raise SolverTimeout(f"model omitted {var}")
        number = -int(negated.group(1))
      if cell.is_constant:
        values[cell.index] = str(number)
      else:
        choices = self.cell_choices[cell.index]
        if not 0 <= number < len(choices):
          raise SolverTimeout(f"model returned out-of-range index for {var}")
        values[cell.index] = choices[number]
    return values


def _byte_offset(src: bytes, lineno: int, col: int) -> int:
  return sum(len(line) for line in src.splitlines(keepends=True)[: lineno - 1]) + col


def _binop_span(src: bytes, left: ast.AST, right: ast.AST) -> tuple[int, int] | None:
  left_end = _byte_offset(src, left.end_lineno, left.end_col_offset)
  right_start = _byte_offset(src, right.lineno, right.col_offset)
  slice_ = src[left_end:right_start]
  for symbol in (
    b"**",
    b"<<",
    b">>",
    b"//",
    b"+",
    b"-",
    b"*",
    b"/",
    b"%",
    b"&",
    b"|",
    b"^",
  ):
    index = slice_.find(symbol)
    if index != -1:
      return left_end + index, left_end + index + len(symbol)
  return None


def _compare_op_span_between(
  src: bytes, previous: ast.AST, comparator: ast.AST
) -> tuple[int, int] | None:
  previous_end = _byte_offset(src, previous.end_lineno, previous.end_col_offset)
  start = _byte_offset(src, comparator.lineno, comparator.col_offset)
  slice_ = src[previous_end:start]
  for symbol in (b"==", b"!=", b"<=", b">=", b"<", b">", b"is not", b"is"):
    index = slice_.find(symbol)
    if index != -1:
      return previous_end + index, previous_end + index + len(symbol)
  return None


def _unary_span(model: PuzzleModel, node: ast.UnaryOp) -> tuple[int, int] | None:
  start = _byte_offset(model.probe_bytes, node.lineno, node.col_offset)
  operand_start = _byte_offset(
    model.probe_bytes, node.operand.lineno, node.operand.col_offset
  )
  slice_ = model.probe_bytes[start:operand_start]
  for symbol in (b"not", b"~", b"+", b"-"):
    index = slice_.find(symbol)
    if index != -1:
      return start + index, start + index + len(symbol)
  return None


def _binop_symbol(op: ast.AST) -> str:
  mapping = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
  }
  return mapping[type(op)]


def _unary_symbol(op: ast.AST) -> str:
  mapping = {
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Invert: "~",
    ast.Not: "not",
  }
  return mapping[type(op)]


def _compare_symbol(op: ast.AST) -> str:
  mapping = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Is: "is",
    ast.IsNot: "is not",
  }
  return mapping[type(op)]


def _label_for(model: PuzzleModel, lineno: int) -> str | None:
  from codoku_common import block_label_for_line

  return block_label_for_line(model.comments, lineno)


def _loop_label(model: PuzzleModel, loop: ast.While) -> str:
  from codoku_common import loop_header_label

  return loop_header_label(model.comments, loop)


def _exit_label(model: PuzzleModel, loop: ast.While) -> str:
  for _, _, label, line in model.comments:
    if line > loop.end_lineno:
      return label
  return "exit"


def _innermost_loop(leaf: ast.FunctionDef, lineno: int) -> ast.While | None:
  found: ast.While | None = None
  for node in ast.walk(leaf):
    if isinstance(node, ast.While) and node.lineno <= lineno <= node.end_lineno:
      if found is None or node.lineno > found.lineno:
        found = node
  return found


def _first_transfer(node: ast.AST) -> ast.AST | None:
  if isinstance(node, (ast.Break, ast.Continue, ast.Return)):
    return node
  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.While)):
    return None
  for child in ast.iter_child_nodes(node):
    found = _first_transfer(child)
    if found is not None:
      return found
  return None


def _has_state_effect(stmts: list[ast.stmt]) -> bool:
  """True when an inline arm assigns or calls something other than a transfer."""
  for stmt in stmts:
    if _is_dump_trace(stmt):
      continue
    if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.Expr)):
      if not _is_global_chksum_stmt(stmt):
        return True
    if isinstance(stmt, ast.If):
      if _has_state_effect(stmt.body) or _has_state_effect(stmt.orelse):
        return True
  return False


def _is_global_chksum_stmt(node: ast.stmt) -> bool:
  """True only for the mid-path ``_g`` checksum instrumentation.

  The exit block's ``v__chk = _in_global_chksum(...)`` is real program
  content: it computes the returned value, so the encoder must inline it.
  """
  if not isinstance(node, (ast.Assign, ast.AugAssign)):
    return False
  targets = node.targets if isinstance(node, ast.Assign) else [node.target]
  if not any(_is_global_accumulator(target) for target in targets):
    return False
  return any(
    isinstance(item, ast.Call)
    and isinstance(item.func, ast.Name)
    and item.func.id == "_in_global_chksum"
    for item in ast.walk(node)
  )


def _is_global_accumulator(target: ast.AST) -> bool:
  base = target
  while isinstance(base, (ast.Subscript, ast.Attribute)):
    base = base.value
  return isinstance(base, ast.Name) and base.id == "_g"


def _is_dump_trace(node: ast.stmt) -> bool:
  return isinstance(node, ast.If) and any(
    isinstance(item, ast.Constant) and item.value == "DUMP_TRACE"
    for item in ast.walk(node.test)
  )


def _choice_index(cell: Cell, name: str) -> int:
  return cell.domain.index(name)

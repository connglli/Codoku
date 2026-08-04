# CFG reducibility and structured control flow

RefractIR is a goto-style CFG IR: a function body is a set of basic blocks joined by explicit `br` terminators, with no structural restrictions. That suits symbolic execution and constraint generation, but not every translation target can express arbitrary control flow. Python has neither `goto` nor labeled `break`. Rather than degrade such a target to an unreadable dispatch encoding, `symirc` restricts it to reducible CFGs and reconstructs genuine `while` / `if` control flow from the dominator tree.

This document specifies the analyses and the transform that make that possible. They live in [include/analysis](../include/analysis) and [src/analysis](../src/analysis) as plain build-from-CFG structs, usable outside the pass pipeline; only the reducibility check has a diagnostic pass surface.

| Stage | Files | Dump flag |
|---|---|---|
| Dominator tree | `dominators.{hpp,cpp}` | `--dump-domtree` |
| Reducibility check | `reducibility.{hpp,cpp}` | (diagnostics; `--require-reducible`) |
| Loop nesting forest | `loop_info.{hpp,cpp}` | `--dump-loops` |
| Control-tree builder | `structurizer.{hpp,cpp}` | `--dump-control-tree` |
| Structured lowering | `structured_lowering.{hpp,cpp}` | `--dump-lowered-tree` |

All five run inside `symirc`. Which of them a backend consumes is stated in [symirc.md](./symirc.md): Python and structured C take the lowered control tree of §5, while structured WASM takes the unlowered tree of §4 directly, because its native multi-level `br` expresses every transfer.

## 1. Reducibility

Fix a depth-first ordering of the CFG and let `rpo(b)` be a block's reverse-postorder number. An edge `u -> v` is *retreating* iff `rpo(v) <= rpo(u)`. A retreating edge is a *back edge* iff its target dominates its source. A CFG is **reducible** iff every retreating edge is a back edge.

Equivalently, every cycle has a unique entry block, its *header*, that dominates every block in the cycle. Irreducible control flow is a branch that enters a loop from the side, past its header. No `while` / `break` / `continue` expresses that without duplicating code, which is why the structuring targets reject it.

The dominance-based test is used rather than the classic T1/T2 interval collapse, because the dominator tree is needed anyway for loops and structuring, and because it names the exact offending edge:

```text
  10 |   br %x < 10, ^a, ^exit;
     |   ^
     |   error: Irreducible control flow: branch from ^b to ^a re-enters a loop whose header does not dominate ^b
   5 | ^a:
     | ^
     | note: ^a is reached both from above and by this retreating edge, so it is not a unique loop header
```

The check runs as a pass, `ReducibilityCheck`, registered by the driver when the target cannot express irreducible control flow (`--target python`), when a control-tree dump is requested, or unconditionally under `--require-reducible`. Rejection is a static error, exit code 4, reported before anything is emitted.

Reducibility is a property of the CFG shape, not of the program's behaviour: the same loop written with a single header block is accepted. `rysmith` and `rylink` generate reducible control flow on request (`--require-reducible`), so the reify pipeline can target the structuring backends.

## 2. Dominator tree (`DomTree`)

Built with the Cooper-Harvey-Kennedy iterative algorithm: immediate dominators are intersected over the reverse postorder until a fixpoint. Near-linear on the small CFGs RefractIR produces, with no auxiliary forests.

* `idom[b]`, the immediate dominator per block, with `idom[entry] == entry`.
* `children[b]`, the dominator-tree children, ordered by RPO number so traversals are deterministic.
* `rpoNumber[b]`, the block's reverse-postorder position, which doubles as the retreating-edge classifier.
* `dominates(a, b)`, a reflexive dominance query by idom-chain walk.

A block unreachable from the entry holds the sentinel `kNone` in both `idom` and `rpoNumber` and is excluded from everything downstream; the reachability analysis diagnoses it separately.

`--dump-domtree` prints one section per function in a stable label-based format:

```text
domtree @sum:
  ^head: (root)
  ^body: idom=^head
  ^done: idom=^head
```

## 3. Loop nesting forest (`LoopInfo`)

Natural loops are discovered from back edges, and all back edges targeting the same header merge into a single `Loop`. That merge is the only normalization structured emission needs: the header is unique by construction on a reducible CFG, an extra latch is just an extra `continue` site, and each exit edge lowers independently to a `break`.

No preheaders or dedicated exit blocks are synthesized, and the block list is never mutated. Loops nest into a forest (`parent`, `children`, `depth`), with `innermostLoop[b]` mapping each block to its innermost containing loop.

On an irreducible CFG, `LoopInfo` silently ignores the irreducible cycles, since only true back edges form loops. Run `ReducibilityCheck` first when that matters.

`--dump-loops` prints, per function:

```text
loops @sum:
  loop 0: header=^head depth=1 parent=none
    latches: ^body
    blocks: ^head ^body
    exits: ^head->^done
```

## 4. Control-tree builder (`Structurizer`)

Reconstructs a structured control tree from a reducible CFG using the dominator-tree translation of Ramsey, *Beyond Relooper* (ICFP 2022). The method is total on reducible CFGs, with no node splitting and no CFG mutation, and the tree's nesting equals the emitted statement nesting.

* A block's dominator-tree children that are merge nodes (two or more forward in-edges) or loop headers become an RPO-ordered follower sequence after the block's own content.
* A child with a single forward predecessor inlines into the branch arm that reaches it.
* A loop header wraps its natural-loop body in a `Loop` node, and its dominated out-of-loop blocks follow the loop.
* Each CFG edge then classifies against the context stack as inline, fall-through, `continue` (a back edge), `break` of one or more levels (a loop exit), or a forward jump to a non-immediate follower. Reducibility plus dominance guarantee every edge resolves.

The resulting `ControlTree` is target-neutral: a transfer node says what must happen, not how a backend spells it.

| Node | Meaning |
|---|---|
| `Seq` | Ordered children at one nesting level |
| `BlockStmts` | One block's straight-line instructions, terminator excluded |
| `If` | A conditional terminator with then/else subtrees |
| `Loop` | A natural loop, `while True` until a `Break` leaves it |
| `Break{target, levels}` | Leave `levels >= 1` enclosing loops, resume at a pending join |
| `Continue{header, levels}` | Back edge, after first leaving `levels >= 0` inner loops |
| `FallThrough` | Fall through to the next pending join |
| `JumpJoin{target, levels}` | Forward jump that must skip intermediate join subtrees |
| `Return`, `Trap` | A `ret` or `unreachable` terminator |

A target with labeled break, or WASM's `br N`, lowers `Break{levels=2}` natively and consumes this tree unchanged. A target without one runs structured lowering first.

`--dump-control-tree`, which implies `--require-reducible`, prints an indented rendering:

```text
control-tree @sum:
  loop 0 header=^head
    block ^head
    if ^head
      then:
        block ^body
        continue ^head levels=0
      else:
        break ^done levels=1
  block ^done
  return ^done
```

## 5. Structured lowering (`StructuredLowering`)

Rewrites a control tree for a target that has only single-level `break` and `continue` and no `goto`: the Python backend, and the C backend under `--structured-lowering`. A multi-level transfer becomes a one-shot guard flag plus cascaded single-level breaks, and the cost is pay-as-you-go, since the common shapes emit no flags at all.

* `Break{levels=L>1}` sets a flag and breaks, then an `if <flag>: break` cascade follows each of the `L-1` enclosing loops. The final cascade resets the flag.
* `Continue{levels=k>0}` uses the same cascade, ending in `if <flag>: <flag> = False; continue` inside the target loop.
* `JumpJoin` sets a flag, adding break and cascades when loops are crossed. The skipped join subtrees are wrapped in `if not <flag>:` guards, and the flag resets where the target subtree is reached.
* `FallThrough` nodes and a tail-position `continue levels=0` are dropped, and an emptied `If` arm is removed, negating the condition when only the then-arm emptied.
* A `while True` loop whose header has no instructions and whose header `If` has a single-level-break arm peepholes into a condition loop, `while cond:` or `while not cond:`.
* A `while True` loop whose body ends with a single-level `if cond: break` peepholes into a `DoWhile` node, `do { body } while (cond);` in structured C, but only when the rest of the body has no continue site bound to the loop. C's `continue` inside a do-while evaluates the condition instead of re-entering the body unconditionally. A target without do-while re-expands the node to the exact pre-peephole form.
* A header-test loop whose header carries instructions rotates, which is classic loop inversion: `loop { H; if cond: R else break }` becomes `H; while cond: { R; H }`, duplicating H's statements once so the loop condition is visible instead of an infinite loop with a break. The same continue-site veto applies, because a `continue` in R would skip the trailing H and re-test the condition early.

A loop with scattered mid-body exits or live continue sites stays `while True`. No single-condition form expresses it without deeper restructuring.

The lowered tree contains no `FallThrough` and no `JumpJoin`, every `Break` has `levels == 1`, and every `Continue` has `levels == 0`, which is exactly what a single-level-break language expresses directly. A flag is `False` except between its set and its final cascade or reset, so re-entering the region is safe; emitters declare and initialize the flags at function entry.

`ret` needs no flags, since Python and C both allow `return` anywhere, so a `Return` node is emitted in place.

`--dump-lowered-tree`, which implies `--require-reducible`, shows the tree the Python backend prints from:

```text
lowered-tree @sum:
  while 0 header=^head
    block ^body
  block ^done
  return ^done
```

## 6. Worked example

Every dump in §§2 to 5 is this function's:

```sir
fun @sum(%n: i32) : i32 {
  sym %?step : value i32;
  let mut %i: i32 = 0;
  let mut %acc: i32 = 0;
^head:
  br %i < %n, ^body, ^done;
^body:
  %acc = %acc + %?step;
  %i = %i + 1;
  br ^head;
^done:
  ret %acc;
}
```

`symirc --target python` emits, after the semantics preamble:

```python
def refractir_sum(n):
    i = 0
    acc = 0
    while (i) < (n):
        # ^body
        acc = _iadd(acc, sum__step(), 32)
        i = _iadd(i, 1, 32)
    return acc
```

The loop header's `br` became the `while` condition through the condition-loop peephole, the back edge disappeared into the loop structure, the block label survived as a comment, and no guard flag was needed.

## 7. Testing

Fixtures live in [test/reducibility](../test/reducibility) and run under `make test-frontend`:

```bash
python3 -m test.lib.run_reducibility_tests test/reducibility ./symirc
```

Each `.sir` fixture passes its dump flag through a `// COMPILER_ARGS:` line, a sibling `<name>.sir.expected` pins the dump byte for byte, and a `// EXPECT: FAIL:StaticError` fixture pins an irreducibility diagnostic.

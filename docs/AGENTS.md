# The documentation standard

How to write the prose that stays: the header comments that explain a design, the notes under [.agents/notes/](../.agents/notes/), and the commit messages that record a change. The documents we write are [README.md](../README.md) and [AGENTS.md](../AGENTS.md), the Codoku project; the documents under `docs/` belong to the RefractIR dependency and serve as reference material to read, not files to edit. [AGENTS.md](../AGENTS.md) owns the engineering rules; this file owns the writing.

## Where a fact belongs

* [SPEC_v0.2.3.md](./SPEC_v0.2.3.md) holds the normative language definition and the roadmap for its release line.
* [undefined.md](./undefined.md) holds the strict UB rules and how each tool enforces them.
* [float.md](./float.md) holds the finite-only floating-point model and the bit-exact text serialization invariant.
* [intrinsics.md](./intrinsics.md) holds the intrinsic signatures, SMT encodings, UB conditions, and per-backend lowering rules.
* [reducibility.md](./reducibility.md) holds the dominator, reducibility, loop, and control tree analyses behind structured lowering.
* [symirc.md](./symirc.md), [symiri.md](./symiri.md), [symirsolve.md](./symirsolve.md), and [reify.md](./reify.md) hold the usage and behavior of each tool.
* Earlier SPEC versions record their releases.
* [CHANGELOG.md](../CHANGELOG.md) holds the release history, newest first.
* [README.md](../README.md) holds the Codoku overview and the build and usage entry points.
* [Makefile](../Makefile) holds the build and test targets.
* [.agents/notes/](../.agents/notes/) holds ephemeral rationale and working records; its README owns the subdirectory meanings and file names.
* Source files and tests hold the live implementation, type definitions, and invariants: code comments explain why rather than what, and test suites define executable contracts for components.

## Writing rules

Write for a compiler engineer who has not read the code. Prefer plain English to metaphor. For the specification, it needs to be precise and complete.

Document the current state, not the change history. Avoid "previously", "now" and "no longer", and avoid citing commits, branches or review threads in durable prose; name the live mechanism instead. Change stories go in the commit that made them, and their reasoning in a note.

State the contract: what holds, under which conditions, and what happens when they do not.

One rule per paragraph. Emphasis marks the clause that changes behaviour and nothing else. Numbers carry their provenance, so "about 70% of seeds solve over the full type lattice" is a claim and "yields are good" is not.

## Use natural writing

Use [natural writing](https://github.com/flutter/flutter/blob/fdf8a01bd014798113aa59ac5b4fd3c30573d9eb/.agents/agents/reidbaker-agent/skills/natural-writing/SKILL.md). For example, avoid:

* Puffery ("a testament to", "a pivotal moment").
* Dangling commentary ("highlighting", "reflecting", "showcasing").
* Promotional verbs ("boasts", "features", "leverages", "ensures").
* Copula inflation ("serves as" where "is" is meant).
* Negative parallelism ("not only X but also Y").
* Three adjectives chosen for rhythm.
* False ranges, where "from X to Y" spans nothing.
* Elegant variation: repeat the name instead of finding a synonym.
* Weasel attribution ("it is generally accepted").
* A closing paragraph that speculates about future work; end on the last fact.
* Overused punctuation (curly/smart quotes and en- and em-dashes).

## Formatting

One physical line per paragraph. Use editor soft-wrap. Code blocks, tables, and list structure keep their formatting; code comments stay under the linter's column limit.

Sentence case headers. Plain `*` bullets, no emoji (unless in the top-level README.md). Straight quotes and apostrophes. No em- or en-dashes anywhere; commas and parentheses do the same work.

Code blocks carry SIR, C, WAT, or Python text, a shell invocation, or a pipeline sketch, and have to be true. An SIR snippet is something a reader can paste into `symiri` or `symirc`.

Link with relative paths that resolve, as [SPEC_v0.2.3.md](./SPEC_v0.2.3.md), never a bare filename.

## The slop checklist

Hunt these in any document you touch. Four are mechanical, so grep first: the temporal words ("now", "currently", "previously", "no longer", "used to"), a distinctive phrase from any rule you stated, `**` runs beyond a term's first mention, and a number with no measurement behind it.

* The same rule stated in two homes. Keep one, link the other.
* Narrated history in durable prose: "previously", "was renamed", a commit or branch cited as if it were a fact about the code.
* A war story told inline where one sentence of rule and a link to an implementation note would do.
* Status annotations in prose or diagrams: "implemented", "planned", "future work". Status rots; the repo layout and the spec's roadmap carry it.
* A hand-maintained inventory of rules, tests, options, or files that the tree already carries. Name where it lives instead.
* Reasoning transcripts: derivations, alternatives weighed, proof of the obvious branch, a walkthrough of a test. Keep the contract, move the rationale to a design note.
* The same rationale repeated beside each of several sibling functions. State it once, at the thing they share.
* Paragraph walls: one paragraph carrying several rules and a parenthetical aside. Split it, or demote the detail to the document that owns it.
* Emphasis inflation. Bold, CAPS and "critically" everywhere mean nothing stands out.
* A table built for two rows, or a bulleted list of one.
* Numbers with no provenance, and percentages whose denominator is not stated.

Two failures survive every grep, so read the diff for them: a paragraph that grew a second subject, and a passage explaining how you got there rather than what holds.

## Length

Relocate first, condense second, accept the length third. Review is the check; there is no budget script.

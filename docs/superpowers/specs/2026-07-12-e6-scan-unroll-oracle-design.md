# E6 — Scan-Unrolling Property Oracle — Design

**Status:** approved 2026-07-12. Feeds `writing-plans`.

## Goal

Add the third, strongest E6 metamorphic law — deferred from
`docs/superpowers/specs/2026-07-11-e6-property-oracles-design.md` — to the property-oracle
harness: **a scan over axis `l = L` must equal the hand-unrolled `L`-statement program with
explicit indices.** This is a *complete* semantic characterization of `evalScan` against the far
simpler non-scan evaluator, and is precisely the oracle that would have caught the RC5/RC6-era
silent-wrong-scan bugs (`papers/restructure_suggestions.md` E6.3) mechanically. It is the
regression net under which Spike 5's `finalizeScans` rewrite and E4's plan compiler become
low-anxiety changes.

Non-goal: proving correctness. As with the scan-free harness, this is bounded-exhaustive property
testing, not a proof.

## Background facts (verified in the codebase)

- Scans are not a distinct surface construct: `TLProgram.stmts` is a flat `List Stmt`, and
  statements carrying `.iterAt`/`.iterNext` `LHSSlot`s are auto-grouped into `ScanStmt.scan` nodes
  by `finalizeScans` (`LeanNCD/DSL/Pipeline/Structural.lean:404-519`) during compilation. A
  generator can therefore emit ordinary flat `Stmt` lists — exactly the convention the existing
  hand-written examples (`test/Eval/ScanTest.lean`, `test/Eval/Portfolio/RecurrenceTest.lean`)
  already use — with no new AST support needed.
- `evalScan` (`LeanNCD/Eval/Scan.lean:91-135`) allocates each state tensor at full shape, fills
  boundary cells from `base` stmts, then walks the cartesian product of `[0 … L_a−2]` over every
  advancing axis in **reverse-lexicographic order** (`cartesianList`, last axis slowest), writing
  only fully-advanced cells; boundary cells keep their zero-default unless an explicit base stmt
  pins them.
- `writeSliceAtMulti` (`Eval/Scan.lean:60-64`) writes a non-iteration slice into a full state
  tensor at a set of `(position, index)` iteration coordinates. No inverse (slice-*out*) helper
  exists yet — this design adds one.
- `IdxExpr.const : Int → IdxExpr` already exists (`LeanNCD/DSL/Ast.lean:27`) — a literal-index
  read needs no new AST support.
- The oracle must go through the full `TLProgram.eval` entry (compile → schedule → finalizeScans →
  eval), same requirement as the scan-free harness, so the reordering/scheduling and scan-grouping
  machinery are exercised, not bypassed.
- The existing scan-free harness (`test/Eval/PropertyOracle/{Compare,Transforms,Gen,Oracle}.lean`,
  `test/Eval/PropertyOracleTest.lean`) is complete and committed (reordering + materialization laws
  pass over its bounded generator). This spike is purely additive — no edits to those files.

## Scope

**In:** a curated family of six scan templates (below), a mechanical unroll transform (two
focused functions: 1-axis and 2-axis), a slice-extraction comparator, and a build-failing entry —
plus test-the-tester checks, following the same shape as the scan-free harness.

**Out (this spike):** general N-axis scans beyond 2-D; shifted/affine reads *of the scan axis
itself* within a recurrence (every existing scan example in the codebase reads the scan axis
plainly at the current index; this is a real restriction, not an oversight, and is called out
explicitly rather than silently assumed); full combinatorial (non-curated) scan generation.

## Why a curated template family, not bounded-exhaustive combinatorics

The scan-free generator (Task 3 of the prior spike) built a combinatorial cross-product because
scan-free well-formedness is comparatively loose (any read of a declared tensor with matching
arity is valid). Scan well-formedness is materially tighter — causality (`readsIterAhead`),
matching base/recur names per component, "every state recurrence must advance over the
component's full axis set" (`Structural.lean:481-490`), heterogeneous-coupling rejection — so a
combinatorial generator risks silently encoding the wrong invariants, which is a correctness risk
in the *generator*, not just extra code. A curated family — six templates, each varied over `L`
and small coefficients, chosen to be **orthogonal** (linear vs. nonlinear, single vs. coupled
state, sum vs. tropical aggregator, self-read vs. external-axis-read, 1-D vs. 2-D) — gets
meaningful, deliberately-chosen coverage at much lower risk, and each template is a direct,
mechanical generalization of an existing hand-verified point example
(`ScanTest.lean`/`RecurrenceTest.lean`), not a hand-derived one-off.

## Component 1: Scan template generator

Six templates, each instantiated over small parameters (not single points):

| # | Shape | Varies over | Axes | Orthogonal dimension covered |
|---|-------|-------------|------|-------------------------------|
| 1 | Linear self-scan: `S[j,l+1] := S[j,l]·A[j]` | `L∈{2,3}`, `A∈{pos,neg}` | 1 | baseline recurrence |
| 2 | Nonlin self-scan: `S[j,l+1] := relu(S[j,l]·A[j])` | `L∈{2,3}`, `A∈{pos,neg}` | 1 | pointwise nonlin / clamping |
| 3 | Coupled 2-state: `G[l+1]:=G[l]+H[l]; H[l+1]:=G[l]` | `L∈{2,3}` | 1 | multi-state coupling |
| 4 | State + external read: `S[l+1] := S[l] + X[l]` (X indexed by the scan axis) | `L∈{2,3}` | 1 | external-tensor read on the scan axis |
| 5 | Tropical aggregator: `M[j,l+1] := maxreduce(M[j,l]·W[j,k])` and a `minreduce` variant | `L∈{2,3}` | 1 | combine operator (RC5/RC7-shaped; the historical `KG-scanagg` bug class) |
| 6 | 2-D grid-DP: `G[r,0]:=Z[r]; G[r+1,c+1]:=G[r,c]+A[r,c]` | fixed `Lr=Lc=2` (grid size grows as `Lr·Lc`, kept minimal) | 2 | multi-axis |

Deterministic non-degenerate inputs, same `data[flat] = flat+1`-style convention as the existing
scan-free generator. Total: roughly 16–20 `(program, inputs)` instances — small enough to keep
`lake build` fast.

**Interface (produced):**

```lean
structure ScanCase where
  prog   : TLProgram
  inputs : Std.HashMap String DenseTensor
  axes   : List AxisSpec        -- the scan axis/axes, in slot-position order
  Ls     : List Nat             -- per-axis length, matching `axes` order
  base   : List Stmt            -- the base (iterAt) statements, as authored
  recur  : List Stmt            -- the recurrence (iterNext) statements, as authored

def enumScanCases : List ScanCase
```

Unlike the scan-free `enumPrograms : List (TLProgram × inputs)`, each `ScanCase` also carries its
own `axes`/`Ls`/`base`/`recur` directly — because the generator *authors* these statements itself
(the same convention `ScanTest.lean` uses), it already knows the grouping that `finalizeScans`
would otherwise have to re-derive at compile time. The unroll transform (Component 2) consumes
this structure directly rather than reimplementing `finalizeScans`'s classification logic.

**Contract tests (fire on build):** every `ScanCase.prog` evaluates to `.ok` through the full
`TLProgram.eval` pipeline (well-formedness gate, same shape as the scan-free harness's contract
test 2); at least one instance of every one of the six templates is present.

## Component 2: Unroll transform

Two focused functions — not one fully-generic N-axis walker, matching the 1-D/2-D scope above:

```lean
def unrollScan1D (c : ScanCase) : TLProgram   -- templates 1-5
def unrollScan2D (c : ScanCase) : TLProgram   -- template 6
```

Each mirrors `evalScan`'s own algorithm structurally — allocate conceptually, fill boundary from
`base`, walk the grid — but instead of computing `DenseTensor`s, **emits `Stmt` values**: one
scan-free `.assign` per (state name × grid cell), named e.g. `Su_<state>_<k>` (1-D) or
`Su_<state>_<r>_<c>` (2-D). Substitution rule applied to each recur statement's RHS when building
the leaf at grid coordinate `k` (predecessor `k-1`, or the grid analogue):

- a read of the **state itself** at the scan axis (`.axis l`) → rewritten to a read of the
  **previously-unrolled leaf tensor** at that coordinate, with the scan-axis position dropped from
  the read's index list (the leaf tensor has no scan axis — it's exactly one slice)
- a read of **any other (external) tensor** at the scan axis (`.axis l`) → rewritten to
  `.read name [.const k]` (literal-index read; `IdxExpr.const` already exists, no new AST needed)
- all non-scan-axis reads/indices pass through unchanged
- grid cells covered by neither `base` nor `recur` (e.g. template 6's `r=0, c>0`) still get an
  emitted leaf — a trivial `.assign` with an empty-terms RHS (the aggregator's identity element,
  i.e. literal zero under `.sum`), matching `evalScan`'s zero-default boundary semantics
  structurally rather than by assumption

**Independence-of-traversal-order requirement:** the grid-cell enumeration order used by
`unrollScan1D`/`unrollScan2D` must be genuinely different from `evalScan`'s own `cartesianList`
(reverse-lexicographic, last axis slowest) — e.g. forward, row-major. If the unroller used the
identical traversal, this would stop being an independently-derived check and become a tautology.
This is a design-time invariant enforced by how the emitter is written (row-major by construction),
not a runtime check — flagged with a one-line comment at the call site so a future refactor
doesn't accidentally converge the two orders.

The emitted `TLProgram` is entirely scan-free (no `.iterAt`/`.iterNext` slots), so it runs through
the exact same `TLProgram.eval` full-pipeline entry as everything else — no second evaluator path.

## Component 3: Slice comparator

New helper — the inverse of `Scan.lean`'s existing `writeSliceAtMulti`:

```lean
def sliceTensorAtMulti (iters : List (Nat × Nat)) (full : DenseTensor) : DenseTensor
```

Given the scan-evaluated full state tensor and a set of `(position, index)` iteration
coordinates, extracts the corresponding non-iteration slice (iterating
`DenseTensor.allCoords` of the slice shape and reading `full.get!` at each reconstructed
full coordinate — the same coordinate bookkeeping `writeSliceAtMulti` does, in reverse).

**Self-check (test-the-tester, before this is trusted in the oracle):** write a slice into a
zero tensor with `writeSliceAtMulti`, read it back with `sliceTensorAtMulti`, confirm it returns
the original slice exactly. This is new code with no existing counterpart to lean on (unlike
`denseEq`/`producedNames`, which the prior spike's tasks could reuse as-is), so it gets its own
round-trip guard rather than being trusted on the strength of the oracle alone.

Leaf-vs-slice comparison itself reuses `LeanNCD.PropertyOracle.denseEq` from `Compare.lean`
(bit-exact shape + `Float` data), consistent with the rest of the harness.

## Component 4: Oracle runner + entry

```lean
def checkScanLaw (c : ScanCase) : Option String   -- none if OK, else a counterexample message
def runAllScans  : Option String                  -- first failure across enumScanCases, else none
```

Structurally mirrors the existing `Oracle.lean`'s `checkLaws`/`runAll`, but for this law: eval the
scan case through the full pipeline, eval its unrolled companion, and for each grid coordinate
compare the sliced-out scan cell (`sliceTensorAtMulti`) against the corresponding unrolled leaf
(`denseEq`). On the first violation: `throwError` with the minimal offending `ScanCase` (its
`Repr`), both programs, and both outputs.

**Kept in separate files from the existing harness** — `ScanGen.lean`, `ScanUnroll.lean`,
`ScanOracle.lean`, and a new `PropertyOracleScanTest.lean` entry (registered separately in
`lakefile.toml`'s `Tests` globs) — so a violation is unambiguously reported as a **SCAN-UNROLL**
failure distinct from the existing REORDERING/MATERIALIZATION messages, and so this spike is
purely additive: zero edits to the already-committed, already-passing scan-free harness.

## Testing the tester

- **Sanity-pass:** one hand-built scan per template, borrowing the existing hand-verified point
  values from `ScanTest.lean`/`RecurrenceTest.lean` (e.g. RC6 for the 2-D template), must pass the
  unroll law — doubling as a regression check that the new harness agrees with already-trusted
  examples.
- **Has-teeth:** a deliberately wrong unroll (e.g. drop the `+1` boundary offset, or swap
  `maxreduce`→`sum` in template 5's emitted leaf) must be *caught* by the comparator.
- **`sliceTensorAtMulti` round-trip:** see Component 3 — the extraction helper is verified against
  its own inverse before being trusted inside the oracle.

## Risks / notes

- **Boundary-cell zero-emission** (template 6) must match `evalScan`'s zero-default exactly (the
  aggregator's identity element under `.sum` is `0`) — if a future aggregator is added to the
  template set, its identity element must be used instead, not a hardcoded `0`.
- **Traversal-order independence** is a hand-maintained invariant (row-major emitter vs.
  reverse-lex `evalScan`), not a runtime-checked one — worth a comment, not a test, since there's
  no meaningful runtime assertion that would catch an accidental convergence short of re-deriving
  `evalScan`'s own order inside the check (which would then need its own independence guarantee).
- **Scope restriction on scan-axis reads** (plain `.axis` only, no shift/affine of the scan axis
  itself) matches every existing example in the codebase today; if a future scan feature
  introduces shifted scan-axis reads, this harness will need a template addition, not a rework.

## File layout

- `test/Eval/PropertyOracle/ScanGen.lean` — `ScanCase`, `enumScanCases` (the six templates).
- `test/Eval/PropertyOracle/ScanUnroll.lean` — `unrollScan1D`, `unrollScan2D`,
  `sliceTensorAtMulti`.
- `test/Eval/PropertyOracle/ScanOracle.lean` — `checkScanLaw`, `runAllScans`.
- `test/Eval/PropertyOracleScanTest.lean` — the `run_cmd` entry + test-the-tester checks;
  registered in `lakefile.toml` `Tests` globs.

(Exact module split is a plan-time detail; the responsibilities above are the fixed boundaries.)

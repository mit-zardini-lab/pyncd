# Multi-axis scans (KG-2dscan) — design

**Status:** approved design, pre-implementation. **Date:** 2026-07-08.

## Problem

A 2-D / nested recurrence such as the grid-DP / PixelRNN step

```
axis r : ℕ = 2, c : ℕ = 2
G[r, 0]       := Z[r]
G[r +1, c +1] := G[r, c] + A[r, c]
```

silently collapses to a 1-D scan and produces the wrong answer (RC6: returns
`[[0,1],[0,1]]`, correct grid-DP is `[[0,0],[0,1]]`). No error is raised — it is the
KG-2dscan soundness bug (§14/§18 of `docs/test_portfolio.md`).

**Root cause.** The scan machinery assumes a statement advances along exactly one axis.
`Stmt.iterInfo` ([Structural.lean:341](../../LeanNCD/DSL/Pipeline/Structural.lean)) uses
`findSome?` and keeps only the *first* advancing slot; `ScanStmt.scan` carries a single
`AxisSpec`; `iterSlotPos`/`evalScan` iterate one axis. For `G[r +1, c +1]` the second
advancing slot (`c +1`) is dropped, and the base case `G[r, 0]` (whose `0` literal sits in the
`c` position) is force-adopted onto the step's first axis `r`, so `r` becomes both a free axis
and the iteration axis — the base collapses to a scalar and the other boundary is overwritten.

## Decisions (resolved)

1. **Scope: general n-D.** Carry a *list* of iteration slots everywhere and nest the loops
   lexicographically. 2-D grid-DP is the driving case; 3-D+ falls out for free.
2. **Boundaries: zero-default.** The step writes only fully-advanced cells (every advancing
   index ≥ 1). Any boundary cell (some index = 0) keeps its value from the zero-allocated state;
   explicit base statements override specific boundary slices. This matches how single-axis
   scans already zero-allocate state and makes RC6-as-written yield the correct `[[0,0],[0,1]]`.
3. **Br/execution layer: full support** (not fail-loud). The Br step builder is already
   axis-count-agnostic (see §4), so multi-axis lowers correctly once the upstream node is
   well-formed.

## Design

### 1. Data model
- `ScanStmt.scan : String → List AxisSpec → List Stmt → List Stmt → Bool → ScanStmt`
  (was a single `AxisSpec`). Axis list is in step-slot order.
- `Stmt.iterInfo : Stmt → List (UID × AxisSpec × Bool × Nat)` — returns **all** advancing
  slots, each with its slot **position** (the new `Nat`), used for positional base recovery.

### 2. Compile — `finalizeScans` ([Structural.lean:381](../../LeanNCD/DSL/Pipeline/Structural.lean))
- **Collect all iteration slots** per statement (list, not first).
- **Positional base-axis recovery** (generalize `adoptBaseIterAxis`): the E1 parser emits a
  placeholder axis for a base's `iterAt` slot. Recover the real axis by *position* — align a
  base's slots to the same-name step's slots; a base literal-`0` slot at position *p* adopts the
  step's `iterNext` axis at position *p*. So `G[r,0]` is recognized as pinning **c = 0** with `r`
  free.
- **Group by connected component** of iteration axes: statements sharing any advancing axis are
  coupled into one scan node carrying that component's axis list. Strictly generalizes today's
  single-UID grouping (RC1's coupled `G/H` over one axis = the degenerate 1-axis component).
- **Causality**: reject a look-ahead read (`shift a n`, `n > 0`) on **any** iteration axis
  (generalize `readsIterAhead`).
- **ScanAffine**: force `isAffine = false` for multi-axis (Prop 8.7 parallel-prefix stays
  1-D-only in v1).

### 3. Eval — `evalScan` ([Scan.lean:69](../../LeanNCD/Eval/Scan.lean))
- Per-axis length `L_a`; state shape has one dim per axis at its slot position (already true via
  `stateShape`).
- **Nested loop**: iterate the cartesian product `∏_a [0 … L_a − 2]`; seed *all* iteration axes
  to the current tuple; evaluate the step slice; write at `(l_a + 1)` in every axis position.
  Lexicographic order in slot order is dependency-safe (each written cell reads strictly-earlier
  cells).
- **Zero-default boundaries**: state zero-allocated; base statements initialize their boundary
  slices first; uncovered boundary cells stay 0.
- Generalize `iterSlotPos` (→ list of `(UID × pos)`) and `writeSliceAt` (→ insert at multiple
  positions).

### 4. Br / Lowering — full support
- The scan `ax` field is destructured as `_` in every `Lowering.lean` consumer; the Br step is
  built from the representative statement's slots (`repStmt` → `retainedOutputSpecs` →
  `degree`/`reindexings`), with `iterAt`/`iterNext` mapped to ordinary axes via `slotAxisIdx?`.
  The `.scan` BrOp is a label, orthogonal to arity; the topo-sort's scan self-edge handling keys
  on tensor *names*. So the Br/Acset layer is already axis-count-agnostic.
- Work required: propagate the `List AxisSpec` type through the ~10 mechanical `_` matches; the
  correctly-structured multi-axis node then lowers correctly for free. `AcsetCodec` roundtrips
  unchanged.
- **Verification**: a compile-level test (`TLProgram.compile` on a 2-D scan) asserting a
  well-formed `BrProgram` — step count, `.scan` op, `degree` containing *both* iteration axes —
  and that the existing Lowering proofs (`reindexing_wellFormed`, `buildStep_*`) still build.

### 5. Scope / limitations (v1)
- Step writes at `+1` per axis and reads at the current iteration point (grid-DP / PixelRNN
  shape). Look-ahead on any axis stays rejected by the causality check.
- No parallel-prefix for multi-axis.
- Genuinely exotic couplings (mixed-arity tensors that cannot be component-grouped) fail loud
  rather than silently misbehave.

## Success criteria / tests
- **RC6** → correct `[[0,0],[0,1]]`; retag `[F]` → `[N]`; update its comment and all KG-2dscan
  doc entries (`test_portfolio.md` §7, §14, §18 §A).
- **RC8** (new) — a 3-D nested scan *or* a 2×2 edit-distance-style DP with a non-trivial
  interior, proving generality and locking the semantics.
- **New compile test** — 2-D scan lowers to a well-formed multi-axis Br step (§4).
- **No regression** — full `lake build Tests` green; 1-D scans (RC1, RC2, RC5, RC7, SS1–3, CM4)
  unchanged.

## Files touched
- `LeanNCD/DSL/Pipeline/Types.lean` — `ScanStmt.scan` axis-list type.
- `LeanNCD/DSL/Pipeline/Structural.lean` — `iterInfo` (list + positions), positional base
  recovery, component grouping, causality.
- `LeanNCD/Eval/Scan.lean` — `iterSlotPos`, `writeSliceAt`, `evalScan` nested loop.
- `LeanNCD/DSL/Pipeline/Lowering.lean` — mechanical `List AxisSpec` propagation.
- `LeanNCD/Bridge/AcsetCodec.lean` — verify roundtrip (expected: no change).
- `test/Eval/Portfolio/RecurrenceTest.lean` (RC6 flip, RC8) + a compile test + `test_portfolio.md`.

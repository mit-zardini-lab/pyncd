# Final fix report — multi-axis-scans branch

Two fixes from the whole-branch review, applied on top of HEAD `2ad4359`.

## Fix 1 (Important) — fail-loud guard for heterogeneous coupled multi-axis scans

**Problem.** In `finalizeScans` (`LeanNCD/DSL/Pipeline/Structural.lean`), the scan node's
`axes` list is derived from the HEAD state-recurrence only (`stateRecur.head?`). If a connected
component couples two state tensors that advance over DIFFERENT axis sets (e.g. `H` over `{c}`
coupled with `G` over `{r,c}`, sharing `c`), the non-head axes are dropped and `evalScan`
silently mis-addresses — a silent-wrong path the design (§5) requires to FAIL LOUD.

**Guard code** (added in the `for comp in comps` loop, just before the existing
`missingBaseCase`/`causalityViolation` validation loop):

```lean
    -- FAIL LOUD (design §5): every state recurrence in a component MUST advance over the
    -- component's FULL axis set. A heterogeneous coupling — e.g. `H` advancing over `{c}` coupled
    -- (via shared `c`) with `G` advancing over `{r,c}` — would drop the non-head axes when `axes`
    -- is taken from the head alone, and `evalScan` would silently mis-address the shorter tensor.
    -- Compare axis-UID SETS (order-independent) against the component's unioned axis set `comp`.
    for r in stateRecur do
      let radv := ((r.iterInfo.filter (·.2.2.1)).map (·.1)).eraseDups
      unless radv.length == comp.length && comp.all (fun u => radv.contains u) do
        throw (CompileError.inconsistentScanAxes
          s!"{r.lhsName}: coupled scan statements advance over different axis sets (each must advance over the component's full axis set)")
```

**CompileError constructor reused: `inconsistentScanAxes`** (`LeanNCD/Exec/Uid.lean:31`,
doc: "coupled scan outputs disagree on shared axis order"). This is the existing, appropriately
named fail-loud constructor for exactly this family — it is already thrown by the Br step builder
(`Lowering.lean:515`) when coupled scan outputs disagree on axes. Our case (coupled statements
advancing over different axis *sets*) is the finalizeScans-level analogue, so reusing it keeps
one error identity for "coupled scan axis disagreement". No new constructor added, so nothing
ripples into the Acset error encoding.

**Set comparison is order-independent** as required: `radv` (this statement's advancing-axis
UIDs, deduped) is compared to `comp` (the component's full unioned axis set from `comps`, already
deduped) by equal length + `comp ⊆ radv` — equivalent to set equality for two deduped lists, and
it uses `comp` (not the head's axes) as the reference so it is robust to source order.

**Does NOT fire for the required cases** — evidence: focused `lake build Eval.Portfolio.RecurrenceTest`
shows RC6 (single tensor over `{r,c}`) and RC8 (single tensor over `{a,b,d}`) still green; the
full `lake build Tests` is green, which includes RC1 (coupled `G`/`H` both single-axis over `l`,
in `Eval.EvalExamplesTest`). For all of these every state recurrence's advancing set equals the
component's full set, so the guard is inert.

**Live reject-test ADDED** — RC9 in `RecurrenceTest.lean` (a `run_cmd do` mirroring the RC4
reject-test style). The heterogeneous coupling IS constructible in the surface DSL:

```
axis r : ℕ = 2, c : ℕ = 2
H[j, 0]       := X[j]
H[j, c +1]    := H[j, c]        -- advances over {c}
G[r, 0]       := Z[r]
G[r +1, c +1] := G[r, c] + A[r, c]   -- advances over {r,c}; shares c ⇒ same component
```

Prototyped via `lean_run_code`: this throws
`inconsistentScanAxes "H: coupled scan statements advance over different axis sets ..."`.
RC9 asserts the `.inconsistentScanAxes` constructor fires (and `throwError`s on any other
constructor or on success).

## Fix 2 (Minor, coverage) — multi-axis scan with a tropical aggregator

**New test RC10** (chained into the `#lspec group` with `$`), exercising the previously-untested
KG-scanagg × KG-2dscan interaction (RC5/RC7 are 1-D tropical; RC6/RC8 are multi-axis sum):

```
axis r : ℕ = 2, c : ℕ = 2
G[r, 0]       := Z[r]
G[r +1, c +1] := maxreduce(G[r, c] · W[r, c, k])
```
with `Z = [2,5]`, `W` shape `[2,2,2]` with `W[0,0,:] = [1,3]` (all other cells 0, unused).

**Hand-derivation (zero-default boundaries).** Axes `r,c` size 2, so the step iterates the
cartesian product `∏ [0..L-2] = {0} × {0}` — a single tuple `(r,c)=(0,0)`. The step writes only
the fully-advanced cell `G[1,1]`:

- `G[1,1] = maxₖ(G[0,0] · W[0,0,k])`. `maxreduce` selects tropical `(×, max, −∞)` (Contract.lean
  `Combine.max`): within the term `prod = 1·G[0,0]·W[0,0,k]`, then `max` over contracted `k`.
  `G[0,0] = 2`, so `= max(2·1, 2·3) = max(2, 6) = 6` (NOT the ℝ-sum `2·1+2·3 = 8`).
- Boundary/base cells: base `G[r,0]` fills the `c=0` column ⇒ `G[0,0]=Z[0]=2`, `G[1,0]=Z[1]=5`.
  `G[0,1]` has `r=0` (a boundary index), is not fully-advanced, and no base pins it ⇒ zero-default `0`.

Row-major `[r][c]` ⇒ `[[2,0],[5,6]]` = flat `[2,0,5,6]`. Asserted with `evalEqB` against
`tl [2,2] [2,0, 5,6]`.

Prototyped via `lean_run_code` before asserting: `G shape=[2,2] data=[2,0,5,6]` — matches the
derivation exactly. No `mul prod 0.0` tropical-padding path is reached (all reads in-range), so
the deferred Contract.lean issue (finding #2) is untouched and irrelevant here.

The interaction is CORRECT, not broken — no concern to report.

## Test evidence

Focused (both fixes; RC9 `run_cmd` passes silently at elaboration, RC10 green):

```
$ lake build Eval.Portfolio.RecurrenceTest
ℹ [8508/8508] Built Eval.Portfolio.RecurrenceTest (3.0s)
info: test/Eval/Portfolio/RecurrenceTest.lean:64:0: §7 — Recurrence & scans:
  ✓ ∃: RC2 rnn
  ✓ ∃: RC3 prefix-sum
  ✓ ∃: RC5 maxreduce-in-scan (KG-scanagg, fixed)
  ✓ ∃: RC6 2d-scan (KG-2dscan, fixed)
  ✓ ∃: RC7 minreduce-in-scan
  ✓ ∃: RC8 3d-scan
  ✓ ∃: RC10 multi-axis maxreduce (KG-scanagg × KG-2dscan)
Build completed successfully (8508 jobs).
```

Full suite (tail):

```
$ lake build Tests
...
ℹ [8601/8602] Built Bridge.AgreementTest (3.7s)
info: ... 'LeanNCD.realize_fromThreadedComposed_agree' depends on axioms: [propext, Classical.choice, Quot.sound]
Build completed successfully (8602 jobs).
```

The only warnings are pre-existing `linter.unusedSimpArgs` notes in `LeanNCD/Bridge/Agreement.lean`,
unrelated to these changes.

## Files changed

- `LeanNCD/DSL/Pipeline/Structural.lean` — Fix 1 fail-loud guard (10 lines) in the `for comp in comps` loop.
- `test/Eval/Portfolio/RecurrenceTest.lean` — RC9 reject-test (`run_cmd`) + RC10 tropical multi-axis test.

## Self-review

- Guard is surgical: one added loop, reuses an existing error constructor, no new constructor,
  no Acset ripple. Uses `comp` (component union) as reference per the requirement, order-independent.
- The guard iterates only `stateRecur` (genuine state recurrences); per-step intermediates have
  empty `iterInfo` (`isInter` requires it) so they are never mis-flagged.
- After the guard passes, all state recurrences share the same advancing set = `comp`, which also
  retroactively justifies the head-only `axes` derivation on the following line.
- Both new tests were prototyped with `lean_run_code` before being committed, so the asserted
  values are observed, not assumed.

## Concerns

None. All required non-firing cases (RC1/RC6/RC8) verified green; the tropical multi-axis
interaction is correct; deferred Contract.lean padding issue left untouched.

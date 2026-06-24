# LeanNCD affine shape inference (maximal padded extent)

This document describes the affine index arithmetic extension implemented in `leanncd/LeanNCD/Eval/Shape.lean`.

**Implementation branch:** `wm/affine-size-solver-ml-padded`

## Files touched

This implementation touched the following repository files:

- `leanncd/LeanNCD/Eval/Shape.lean`
- `leanncd/test/Eval/AffineShapeSolverTest.lean`
- `leanncd/lakefile.toml`
- `docs/lean_affine_shape_solver_max_padded_semantics.md`

## Semantics contract

The extension preserves existing padded-read evaluation semantics:

- Out-of-range reads are still legal and zero-padded at evaluation time.
- Shape inference remains **maximal padded extent** inference.
- The solver improves what can be inferred; it does **not** introduce ML `valid/same/full` window semantics.

For a read position

```text
X[c0 + Σ c_u * axis_u]
```

against input dimension `d`, shape inference uses:

```text
max reachable source index = d - 1
```

which yields:

```text
c0 + Σ c_u * (size_u - 1) = d - 1
Σ c_u * size_u = d - c0 + Σ c_u - 1
```

Current scope in this branch supports affine forms with:

- signed variable coefficients (handled via upper-envelope projection for max-index constraints),
- integer constant offsets (including negative constants, e.g. `-p`),
- strictly positive inferred axis sizes.

## Padded semantics vs conventional ML semantics

This implementation is explicitly for **padded semantics** (the existing LeanNCD evaluation model), not conventional ML operator semantics.

### What padded semantics means here

- Read expressions are evaluated with permissive boundary behavior (out-of-range reads are padded).
- Shape inference uses maximal reachable source index equalities.
- Output extents are inferred by the maximal padded-extent rule.

### What conventional ML semantics usually means

In most ML frameworks, convolution/pooling semantics are operator-level and include explicit window geometry rules:

- stride/dilation/padding interpreted as operator parameters,
- output shape formulas derived from mode conventions (`valid`, `same`, `full`, explicit left/right padding),
- framework-specific rounding conventions for strided windows.

Typical meaning of these modes (1D intuition, generalized per axis in ND):

- `valid`: no implicit padding; only fully in-bounds windows contribute.
  - stride=1, dilation=1 formula: `out = in - kernel + 1`.
- `same`: choose padding so output extent is approximately input extent (exactly equal for stride=1 in common frameworks).
  - common strided convention: `out = ceil(in / stride)`.
  - required padding is then solved from kernel/dilation/stride and split left/right per framework policy.
- `full`: include all overlaps, including edge-partial windows that require outside-input padding.
  - stride=1, dilation=1 formula: `out = in + kernel - 1`.

With dilation `d`, effective kernel size is:

```text
kernel_eff = d * (kernel - 1) + 1
```

and formulas above replace `kernel` with `kernel_eff`. For stride > 1, framework rounding rules (`floor` vs `ceil` and left/right padding split) become part of the semantic contract.

Worked 1D example (`in = 9`, `kernel = 4`, `stride = 2`, `dilation = 1`):

- `valid`: `out = floor((9 - 4)/2) + 1 = 3`
- `same` (common convention): `out = ceil(9/2) = 5`
- `full`: `out = floor((9 + 4 - 2)/2) + 1 = 6`
- existing padded semantics in this branch: `out = 4`

For the affine read form:

```lean
Y[o] := W[k] · X[2 * o + k - 1]
```

under this branch's padded semantics, maximal-extent inference solves:

- `W[k]` with `W:[4]` ⇒ `k = 4`
- `X[2*o + k - 1]` with `X:[9]` and `k=4`:
  - maximal-index rule gives `-1 + 2*(o - 1) + (4 - 1) = 9 - 1`
  - so `2*(o - 1) = 6`, hence `o = 4`

so this setup separates all modes: `valid:[3]`, `padded:[4]`, `same:[5]`, `full:[6]`.

### Why they differ

An affine read like:

```lean
Y[o] := W[k] · X[o + k - 2]
```

can look conv-like in both systems, but:

- under padded semantics, inference follows maximal padded extent from source-index reach,
- under conventional ML semantics, output extent is derived from operator geometry/mode conventions.

So two systems can compute similar pointwise values for overlapping coordinates, while still disagreeing on inferred output shape/domain.

### Practical implication

The solver extension in this branch:

- **does** improve affine-size inference under padded semantics,
- **does not** add `valid/same/full` operator semantics,
- **does not** reinterpret DSL affine reads as framework conv/pool operators.

### Phase 2F relationship to ML semantics

Phase 2F is the explicit path for ML-style semantics, and it is intentionally separate from existing affine-read semantics.

In-scope for Phase 2F:

- add explicit operator-level constructs/modes for ML window geometry (e.g. `valid`/`same`/`full` or explicit asymmetric padding),
- define shape formulas from operator parameters (stride, dilation, kernel, padding) rather than maximal padded reach equalities,
- keep padded-affine and ML-operator semantics side-by-side without implicit reinterpretation.

Out-of-scope for Phase 2F:

- changing current padded evaluation behavior for existing affine reads,
- silently mapping current affine expressions onto ML operator modes without explicit syntax/mode selection.

## Development plan used

This is the exact implementation plan followed.

1. Preserve the existing one-unknown fast path and conflict semantics.
2. Add deferred affine-constraint collection for read positions with 2+ unknown axes.
3. Normalize affine coefficients by UID before solving.
4. Convert deferred constraints to a linear equality system over axis sizes.
5. Solve exactly with rational elimination (RREF-like Gaussian elimination).
6. Reject underdetermined, inconsistent, non-integral, and non-positive solutions.
7. Integrate solver into the existing fixpoint loop so fast-path and solver can unlock each other.
8. Add broad tests:
   - direct shape-inference solver tests,
   - ML-shaped padded-window tests,
   - failure-mode tests,
   - end-to-end `TLProgram.eval` checks.
9. Add a new semantics+implementation document in `docs/` (without editing `papers/leanncd.md`).

## What changed in code

## 1. New internal data structures

Added in `LeanNCD/Eval/Shape.lean`:

```lean
structure SizeConstraint where
  coeffs : List (Int × UID)
  rhs : Int
  source : String
```

- `coeffs`: left-hand linear coefficients over axis-size variables.
- `rhs`: right-hand constant.
- `source`: human-readable read-location context for diagnostics.

Additional internal helper structs:

```lean
private structure AffinePosition where
  coeffs : List (Int × UID)
  const : Int
  dim : Nat
  source : String

private structure RatRow where
  coeffs : Array Rat
  rhs : Rat
  sources : List String := []
```

`AffinePosition` is the normalized read-position form before conversion to `SizeConstraint`.

## 2. Affine normalization and preprocessing

New helpers:

- `normalizeCoeffs`: merges duplicate UID terms and removes zero coefficients.
- `upperEnvelopeCoeffs`: projects signed rows to positive-coefficient envelope for maximal-index constraints.
- `mkConstraint`: creates:
  ```text
  Σ c_u * size_u = d - c0 + Σ c_u - 1
  ```
- `substituteKnownSizes`: applies currently-known sizes into deferred constraints.
- `canonicalizeConstraint`: row gcd/sign normalization and stable coefficient ordering.

Why this matters:

- keeps rows minimal before elimination,
- avoids duplicated columns for same UID,
- lets known-size inference reduce solver complexity each iteration.

## 3. Solver integration strategy (fixpoint)

`inferAxisSizes` now runs this loop:

1. Run existing one-unknown fast path.
2. Collect multi-unknown affine constraints for unresolved positions.
3. Substitute already-known sizes.
4. Solve remaining linear system exactly (RREF over `Rat`).
5. Insert solved sizes and iterate.
6. Stop when no new bindings are produced.

The original fast path remains the first mechanism for common simple cases.

## 4. Exact solver algorithm details

`solveSizeConstraints` executes:

1. Collect variable order (`List UID`) from all constraints.
2. Build dense rational rows (`RatRow`) with that fixed column order.
3. Perform elimination:
   - pivot search per column,
   - row swap into pivot position,
   - pivot normalization to leading `1`,
   - eliminate that column in all other rows.
4. Post-check:
   - zero row with nonzero RHS => inconsistent.
5. Rank check:
   - fewer pivots than variables => underdetermined.
6. Extract solution from pivot rows and validate:
   - denominator must be `1` (integral),
   - numerator must be `> 0` (strictly positive size).

The fallback solver performs exact elimination over rational rows and rejects:

- `affine size system underdetermined`
- `affine size system inconsistent`
- `affine size system non-integral`
- `affine size system non-positive`

In Phase 2A+, signed coefficients are supported through upper-envelope projection.

## 5. Preserved behavior and compatibility notes

- Existing one-unknown affine inference behavior is preserved.
- Bare-axis conflict detection remains (`name[a]` vs incompatible tensor dimension).
- Fully-known affine reads are **not** upgraded into strict consistency checks (important for padded semantics compatibility).

Reason for the third point: evaluation allows padded out-of-range reads; converting fully-known affine reads into strict equalities would accidentally tighten semantics and reject valid padded programs.

### Known gap: fully-known multi-term read bounds (Issue H)

When all axes referenced in a multi-term affine read are already sized — either because they were resolved in a prior fixpoint iteration or because they were provided via the seed (explicit `axis a = n` declarations) — the read is fully known and its maximum index is computed. If that maximum index meets or exceeds the tensor dimension, a non-fatal `"padded-access warning"` is emitted in the returned `List String`.

**Current limitation**: this warning only fires for seeded axes. When axes are inferred by the solver, the equality-based constraint derivation (`rhs = dim + Σcoef − 1`) produces tight-fit sizes by construction: the inferred maximum index equals `dim − 1`, which is always strictly less than `dim`. The warning therefore cannot fire for solver-inferred axes even when the padded-access intent is present.

**Practical consequence**: out-of-range reads in fully solver-inferred programs are valid under padded semantics and produce zero, but they are silently undiagnosed. Users who rely on padding effects (e.g. zero-fill at borders) will not see any diagnostic.

**Future resolution**: switching the multi-term constraint from an equality to an inequality (`lhs ≤ rhs`) would allow solver-inferred axes to exceed the tight-fit bound, enabling the warning to fire. This requires extending the solver with inequality handling, which is left for a future pass.

**Surfacing**: warnings are emitted to stderr via `dbg_trace` during evaluation. They are also returned from `inferAxisSizes` as a `List String` for callers that want structured access.

## Examples

## A. Solvable multi-equation system

```lean
tlprog!{
  Y[i, j] := X[i + j] + U[i + 2 * j]
}
```

With:

- `X : [7]`
- `U : [9]`

inference solves:

- `i + j = 8`
- `i + 2j = 11`

so:

- `i = 5`
- `j = 3`
- `Y : [5, 3]`

## B. ML-shaped padded window form (still maximal padded extent)

```lean
tlprog!{
  Y[h] := W[k] · X[2 * h + k - 1]
}
```

With:

- `X : [8]`
- `W : [3]`

inference gives `h = 4`, so `Y : [4]`.

This is maximal padded-extent inference, not framework conv mode inference.

## C. Known underdetermined case

```lean
tlprog!{
  Y[i, j] := X[i + j]
}
```

With only `X` dimensional information, two unknown sizes from one equation remain underdetermined and inference errors accordingly.

## D. Failure-mode examples covered

- Inconsistent system (conflicting equations from different reads).
- Non-integral system (fractional inferred size).
- Non-positive inferred size.
- Underdetermined systems with explicit unconstrained UID diagnostics.

## Tests added

New file:

- `leanncd/test/Eval/AffineShapeSolverTest.lean`

Coverage includes:

- successful multi-equation solving,
- coefficient normalization and deferred-solving path,
- ML-shaped affine padded-window program,
- underdetermined system,
- inconsistent system,
- non-integral solution rejection,
- non-positive solution rejection,
- signed-affine success under upper-envelope constraints,
- deterministic unconstrained-UID reporting for underdetermined systems,
- duplicate-term normalization stability,
- factor-order invariance,
- 2D padded-window family inference,
- 3D padded-window family inference,
- mixed-path (fast-path + deferred solver) inference,
- redundant-equality regression (Phase 2E pruning path),
- end-to-end `TLProgram.eval` checks.

Test suite registration updated in:

- `leanncd/lakefile.toml`

Targeted test/build command used during development:

```bash
cd leanncd
lake build Eval.AffineShapeSolverTest Eval.ShapeTest Eval.EvalExamplesTest
```

## Detailed implementation notes

## Error surfaces and insertion semantics

- Size binding still funnels through shared conflict logic (`insertSolvedSize`), so old and new inference paths report consistent conflict failures.
- Solver-introduced failures are explicit and do not silently default to `0` or skip solving.

## Why exact rational elimination

- Avoids floating-point instability.
- Detects true integrality failures cleanly.
- Keeps implementation small and executable (no heavy symbolic dependency).

## Why nonnegative coefficients were used in phase 1

- Matches immediate target forms (`i + j`, `i + 2*j`, `s*h + d*k - p`).
- Avoids changing semantics around lower envelopes and signed-variable max extraction.
- Leaves phase-2 extension path open for signed-variable support with explicit upper-envelope handling.

## Complexity profile

Let:

- `P` = number of affine read positions,
- `V` = number of unresolved UIDs in deferred constraints,
- `E` = number of deferred constraints.

Per fixpoint iteration:

- normalization/substitution is roughly linear in total term count,
- elimination is cubic in variables in dense worst case (`O(V^3)`), which is acceptable for expected DSL sizes.

## Development-phase notes

Two practical implementation adjustments made during rollout:

1. Replaced unavailable API usage (`Array.mkArray`) with `Array.replicate` for Lean 4.30 compatibility.
2. Switched `value.den`/`value.num` field notation to `Rat.den value` / `Rat.num value` to satisfy elaboration in this environment.

Both are implementation portability details; they do not change solver semantics.

## Future extension plan (phase 2+)

## Prioritized roadmap

Recommended priority order:

1. **Phase 2A (implemented in this branch):** signed-coefficient affine support while preserving maximal padded-extent semantics.
2. **Phase 2B (implemented in this branch):** solver robustness/canonicalization (stable normalization, deterministic rows/messages).
3. **Phase 2C (implemented in this branch):** diagnostic scaffolding foundation used by later Phase 3 enhancements.
4. **Phase 2D (implemented in this branch):** expanded regression corpus (2D/ND padded-window families and mixed-path cases).
5. **Phase 2E (implemented in this branch):** performance pass (constraint pruning and elimination optimizations).
6. **Phase 2F (optional design track):** explicit non-padded ML operator modes as separate language constructs.
7. **Phase 3A (implemented in this branch):** diagnostic payload + ML hint extraction.
8. **Phase 3B (implemented in this branch):** root-cause explainability details.
9. **Phase 3C (implemented in this branch):** source-provenance tracing.
10. **Phase 3D (implemented in this branch):** actionable remediation guidance.

Suggested execution cadence:

- Sprint 1: Phase 2A
- Sprint 2: Phase 2B + 2C
- Sprint 3: Phase 2D
- Sprint 4: Phase 2E
- Parallel design track after Sprint 2 starts: Phase 2F

## Detailed plan: Phase 2A (signed coefficients, padded semantics preserved)

Goal: support signed-variable affine reads (e.g. `j - i + 3`, `2*i - j + 1`) without changing padded evaluation semantics.

### Scope and non-goals

In-scope:

- signed coefficients in read-affine forms,
- shape inference from maximal reachable index under padded semantics,
- preserving v1 behavior for existing nonnegative forms.

Out-of-scope:

- `valid/same/full` convolution-mode semantics,
- strict in-bounds validation of every read point,
- changing gather/evaluation padding rules.

### Design rule (core semantic guardrail)

For signed affine expression:

```text
e = c0 + Σ a_u * axis_u
```

derive maximal reachable index using upper envelope only:

```text
max(e) = c0 + Σ max(a_u, 0) * (size_u - 1)
```

then enforce:

```text
max(e) = d - 1
```

This preserves padded semantics: negative-coefficient contribution affects lower tail, not maximal-reach constraint.

### Work breakdown

1. **Representation updates**
   - Extend/annotate normalization path to retain signed coefficients.
   - Add helper to split each row into positive/negative coefficient parts.

2. **Constraint extraction updates**
   - Replace v1 positivity rejection with upper-envelope coefficient projection for multi-unknown reads.
   - Preserve current one-unknown fast path for pure nonnegative rows.
   - Route signed one-unknown rows to deferred solver path when needed.

3. **Known-size substitution updates**
   - Apply substitution before envelope projection where required for stable simplification.
   - Ensure canonical row generation remains deterministic.

4. **Solver integration**
   - Keep exact elimination machinery unchanged.
   - Feed solver projected constraints only (no lower-envelope constraints in Phase 2A).

5. **Diagnostics**
   - Replace current v1-scope rejection with actionable signed-row diagnostics only when system is still unsolved/underdetermined.
   - Keep existing failure classes: underdetermined/inconsistent/non-integral/non-positive.

6. **Regression and new tests**
   - Add success tests for signed forms solvable under upper-envelope constraints.
   - Add ambiguity tests where signed rows remain underdetermined.
   - Add compatibility tests proving v1 nonnegative cases are unchanged.
   - Add end-to-end `TLProgram.eval` examples with signed-affine reads.

### Risks and mitigations

- **Risk:** accidental semantic tightening (treating signed reads as strict in-bounds constraints).  
  **Mitigation:** never add lower-bound validity constraints in solver path.

- **Risk:** user surprise from signed rows dropping variables out of upper envelope.  
  **Mitigation:** explicit underdetermined diagnostics listing unconstrained UIDs.

- **Risk:** regression in current nonnegative fast path.  
  **Mitigation:** keep fast path gated and preserve prior tests.

### Acceptance criteria

Phase 2A is complete when:

1. Signed-affine examples in scope infer correct shapes under upper-envelope rule.
2. Existing v1 tests and behavior remain unchanged.
3. No evaluation semantics changes (padding behavior unchanged).
4. Failure messaging distinguishes underdetermined vs inconsistent vs arithmetic invalidity.

## Phase 2A implementation update

This branch now implements the core Phase 2A semantics.

Code changes:

- Added upper-envelope projection helper:
  - `upperEnvelopeCoeffs : List (Int × UID) → List (Int × UID)`
- Updated inference loop in `inferAxisSizes` to:
  - compute unknowns from upper-envelope coefficients,
  - solve one-unknown equations from upper-envelope rows,
  - defer multi-unknown constraints from upper-envelope rows.
- Removed the v1 runtime rejection for signed coefficients in unknown-bearing rows.

Implementation details:

- `upperEnvelopeCoeffs` is applied per read position before unknown partitioning, so the one-unknown and deferred paths both see the same projected row.
- For signed rows, negative coefficients are intentionally dropped from max-index constraints (they affect lower envelope only under padded semantics).
- The one-unknown path still uses:
  - `numer := (d - 1) - c0 - other`,
  - `size := numer / coef + 1`,
  but now `other` is computed over projected coefficients.
- Deferred constraints are constructed with:
  - `mkConstraint { pos with coeffs := maxCoeffs }`
  to ensure the linear system is fully envelope-based.
- Fully-known affine read handling remains unchanged (no strict affine consistency checks added).
- Constraint substitution still routes through `substituteKnownSizes`, so known UIDs are eliminated before solve.

Semantics preserved:

- Evaluation remains padded.
- Inference remains maximal padded-extent.
- No strict lower-bound/in-bounds constraints were added.

Test updates:

- Replaced prior signed-coefficient rejection test with signed-affine success checks.
- Added end-to-end signed-affine `TLProgram.eval` coverage.

## Phase 2B implementation update

This branch now implements the core Phase 2B robustness work.

Code changes:

- Added deterministic coefficient/constraint ordering:
  - sorted coefficient normalization by UID,
  - sorted variable collection from constraints,
  - sorted constraint ordering before elimination.
- Added row canonicalization:
  - gcd normalization across coefficients and RHS,
  - sign normalization (leading coefficient nonnegative),
  - stable re-sorting after canonicalization.
- Hardened underdetermined diagnostics:
  - now reports deterministic unconstrained UID list in ascending UID order.

Implementation details:

- Deterministic ordering primitives:
  - `insertCoeffSorted` / `sortCoeffs` for `(coef, uid)` terms,
  - `insertUIDSorted` / `sortedVarsOfConstraints` for solver columns,
  - `insertConstraintSorted` / `sortConstraints` for row ordering before elimination.
- Canonical row normalization is centralized in:
  - `canonicalizeConstraint : SizeConstraint → SizeConstraint`
  and applied in both:
  - `mkConstraint` (creation-time normalization),
  - `substituteKnownSizes` (post-substitution normalization).
- `canonicalizeConstraint` pipeline:
  1. merge + sort coefficients,
  2. compute `gcd` across `|rhs|` and `|coeffs|`,
  3. divide row by gcd when `g > 1`,
  4. normalize sign so leading coefficient is nonnegative,
  5. re-sort coefficients for stable render/key behavior.
- Solver entry now always canonicalizes and sorts constraints first:
  - `let constraints := sortConstraints (constraints.map canonicalizeConstraint)`
- Underdetermined classification now computes:
  - `pivotCols`,
  - `freeCols = allCols \ pivotCols`,
  - `freeUids` in deterministic order,
  and surfaces them in the error message.
- All of the above is behavior-preserving for solvable systems; it targets reproducibility and debug stability.

Semantics preserved:

- Inference equations still follow maximal padded-extent semantics.
- Signed-coefficient handling remains upper-envelope based.
- Evaluation padding semantics remain unchanged.

Test updates:

- Added deterministic underdetermined diagnostic assertion.
- Added factor-order invariance test.
- Added duplicate-term canonicalization regression.

## Phase 2C implementation update (diagnostic foundation)

This branch now includes the Phase 2C foundation that supports the Phase 3A-3D diagnostics stack.

Implemented foundation elements:

- centralized diagnostic payload modeling and rendering path,
- deterministic hint insertion/ordering utilities,
- stable wiring points in the solver for failure-kind classification.

This foundation is intentionally behavior-preserving for successful solves and is expanded concretely in Phases 3A-3D below.

## Phase 2D implementation update (expanded regression corpus)

This branch now implements Phase 2D with broader regression coverage across multi-axis padded-window families and mixed-path inference.

New regression coverage added in `AffineShapeSolverTest`:

- 2D padded-window family inference:
  - kernel/input coupling across two dimensions,
  - verifies inferred output extents and kernel axis extents.
- ND padded-window family inference (3D):
  - extends the same maximal padded-extent rule to 3 dimensions,
  - verifies stable inferred output/kernel sizes.
- Mixed-path inference case:
  - one-unknown fast-path solving and deferred affine-system solving exercised together in one statement,
  - verifies cooperative convergence and final output shape.

Semantics preserved:

- maximal padded-extent inference remains unchanged,
- padded read evaluation semantics remain unchanged.

## Phase 2E implementation update (constraint pruning + elimination optimization)

This branch now implements Phase 2E performance work in the affine solver.

Implemented:

- Redundant equality pruning before elimination:
  - equivalent canonical rows are merged before matrix elimination,
  - provenance sources are merged deterministically for diagnostics.
- Elimination bookkeeping optimization:
  - pivot tracking uses an array append path (instead of repeated list concatenation),
  - free-column detection uses a hash-set membership check for pivot columns.

Implementation details:

- In `Shape.lean`:
  - added row-equivalence helpers:
    - `sameEquation`
    - `findEquivalentRow?`
  - row construction in `solveSizeConstraints` now de-duplicates equivalent rows while preserving merged source provenance.
  - pivot bookkeeping changed from list concatenation to `Array.push`.
  - free-column derivation now uses `HashSet Nat` for pivot-column membership checks.

Regression coverage:

- Added a Phase 2E regression in `AffineShapeSolverTest` where identical affine equalities are repeated multiple times; inference result remains identical to the non-redundant system.

Semantics preserved:

- no change to maximal padded-extent equations,
- no change to failure-class semantics or diagnostic categories,
- no change to evaluation-time padded read behavior.

## Phase 2F detailed scope (optional design track)

Phase 2F is not implemented in this branch; this section defines the target scope.

Planned work:

1. Language surface:
   - add explicit ML-window operators or mode-annotated forms (rather than reusing plain affine reads).
2. Shape inference:
   - implement operator-geometry formulas for each declared mode (`valid`/`same`/`full` or explicit padding forms).
3. Evaluation semantics:
   - define mode-specific boundary behavior for the new constructs only.
4. Separation guarantees:
   - preserve current affine padded semantics as default and unchanged.
5. Test/documentation:
   - add paired examples showing where ML-mode output extents differ from maximal padded-extent inference.

## Historical plan snapshot: Phase 2B (solver robustness/canonicalization)

Goal: make solver behavior deterministic and diagnostics stable without changing semantics.

Work items:

1. Canonical variable ordering (ascending UID) for matrix construction.
2. Canonical row normalization:
   - merge duplicate coefficients,
   - drop zero terms,
   - gcd normalization,
   - sign normalization.
3. Deterministic constraint ordering before elimination.
4. Deterministic underdetermined diagnostics with explicit unconstrained UID list.
5. Regression tests for order invariance and normalization stability.

Acceptance checks:

- Equivalent row/factor orderings produce identical solved sizes.
- Underdetermined diagnostics surface stable UID ordering.
- Existing eval semantics and prior test behavior stay unchanged.

## Phase 3A implementation update (diagnostic payload + hinting)

This branch now includes Sprint 3A diagnostic infrastructure inside the solver.

Implemented:

- Internal diagnostic model:
  - `SolveFailureKind`
  - `SolveDiagnostic`
- Centralized diagnostic rendering:
  - `renderSolveDiagnostic`
- ML-oriented hint extraction over normalized constraints:
  - `mlHintsOfConstraints`
  - hints currently emitted for:
    - coupled affine systems,
    - multi-axis window constraints,
    - stride/dilation-like coefficients.

Implementation details:

- New internal types in `Shape.lean`:
  - `SolveFailureKind` with variants:
    - `.inconsistent`
    - `.underdetermined`
    - `.nonIntegral`
    - `.nonPositive`
  - `SolveDiagnostic` payload fields:
    - `kind`
    - `unconstrained : List UID`
    - `offendingUid? : Option UID`
    - `detail? : Option String`
    - `sourceRefs : List String`
    - `remediation : List String`
    - `mlHints : List String`
- Message rendering is centralized in:
  - `renderSolveDiagnostic : SolveDiagnostic → String`
  so all solver failure surfaces share one formatting path.
- Hint extraction (`mlHintsOfConstraints`) is deterministic:
  - hints are inserted through `insertStringSortedUnique`,
  - emitted hints depend only on normalized constraints,
  - no dependence on hash-map traversal order.
- Current hint rules:
  - `constraints.length ≥ 2` → `"ml-hint: coupled affine constraints"`
  - any row with `coeffs.length ≥ 2` → `"ml-hint: multi-axis window constraint"`
  - any `|coef| > 1` → `"ml-hint: stride/dilation-like coefficients"`

Solver integration:

- Inconsistent, underdetermined, non-integral, and non-positive solver failures now route through the diagnostic renderer.
- Diagnostics include:
  - deterministic unconstrained UID lists,
  - offending UID for non-integral/non-positive failures,
  - optional detail payload (e.g., rational value),
  - ML hint list.

  Per-failure wiring:

  - Inconsistent row (`0 = nonzero`) → `SolveDiagnostic { kind := .inconsistent, detail? := some "reduced witness: 0 = <rhs>", mlHints }`.
  - Underdetermined (`freeCols ≠ []`) → `SolveDiagnostic { kind := .underdetermined, unconstrained := freeUids, detail? := some "rank=<r>, vars=<v>, uid <u>: in <n> equation(s)", mlHints }`.
  - Non-integral solved value → `SolveDiagnostic { kind := .nonIntegral, offendingUid? := some uid, detail? := some "value=<rat>, reduced row=<r>, col=<c>", mlHints }`.
  - Non-positive solved value → `SolveDiagnostic { kind := .nonPositive, offendingUid? := some uid, detail? := some "value=<rat>, reduced row=<r>, col=<c>", mlHints }`.

  Tests updated:

  - `AffineShapeSolverTest` now asserts:
    - underdetermined diagnostics include ML hint text,
    - non-integral diagnostics include offending UID and stride/dilation hint.

## Phase 3B implementation update (failure root-cause explainability)

This branch now implements Sprint 3B explainability refinements on top of 3A diagnostics.

Implemented:

- Inconsistent systems now report a reduced-system contradiction witness:
  - `reduced witness: 0 = <rhs>`
- Underdetermined systems now report rank/context details:
  - `rank=<r>, vars=<v>, uid <u>: in <n> equation(s)`
- Non-integral and non-positive failures now report reduced-system location:
  - `value=<rat>, reduced row=<r>, col=<c>`
- Added per-kind ML-facing guidance:
  - non-integral: `ml-hint: output extent divisibility mismatch`
  - non-positive: `ml-hint: offset/window yields non-positive extent`

Implementation details:

- Added helper functions in `Shape.lean`:
  - `uidConstraintCount`
  - `renderUnderdeterminedDetail`
- Updated failure construction in `solveSizeConstraints`:
  - inconsistent checks now include reduced witness detail,
  - underdetermined checks now include rank/vars + per-free-UID equation counts,
  - non-integral/non-positive checks now include reduced row/column metadata.
- Preserved deterministic rendering and ordering:
  - all new details are computed from normalized/sorted rows and UID sets.

Tests updated:

- `AffineShapeSolverTest` now additionally asserts:
  - underdetermined diagnostics include `rank=..., vars=...`,
  - inconsistent diagnostics include reduced witness text,
  - non-integral diagnostics include reduced-row detail and divisibility hint,
  - non-positive diagnostics include reduced-row detail and non-positive extent hint.

## Phase 3C implementation update (constraint provenance tracing)

This branch now implements Sprint 3C provenance tracking for solver diagnostics.

Implemented:

- Added source-provenance payload to diagnostics:
  - `sourceRefs : List String` in `SolveDiagnostic`
- Added row-level provenance tracking through elimination:
  - each `RatRow` now carries `sources : List String`
  - row scaling preserves sources
  - row subtraction merges source sets deterministically
- Failure messages now include source-read provenance:
  - appended as `sources: [...]`
  - derived from reduced rows where applicable

Implementation details:

- Added helper functions in `Shape.lean`:
  - `mergeStringLists`
  - `uidConstraintSources`
- Solver wiring:
  - inconsistent failure uses reduced contradictory row sources,
  - underdetermined failure aggregates sources touching unconstrained UIDs,
  - non-integral/non-positive failures use the offending reduced pivot row sources.
- Determinism preserved:
  - source lists are merged via sorted-unique insertion,
  - diagnostics remain stable across row/factor ordering.

Tests updated:

- `AffineShapeSolverTest` now asserts source provenance presence for:
  - underdetermined failures,
  - inconsistent failures,
  - non-integral failures,
  - non-positive failures.

## Phase 3D implementation update (actionable remediation guidance)

This branch now implements Sprint 3D actionability on top of 3C provenance diagnostics.

Implemented:

- Added remediation payload to diagnostics:
  - `remediation : List String` in `SolveDiagnostic`
- Diagnostic rendering now emits actionable guidance:
  - appended as `actions: [...]`
  - deterministic, failure-kind-specific guidance for all solver failure classes

Implementation details:

- Added helper in `Shape.lean`:
  - `remediationOfDiagnostic`
- Guidance is emitted for:
  - inconsistent systems (dimension/offset consistency and equation separation guidance),
  - underdetermined systems (axis binding and independent-constraint guidance),
  - non-integral systems (divisibility/stride-offset adjustment guidance),
  - non-positive systems (window/extent positivity guidance).
- If explicit remediation is provided in a future diagnostic constructor, renderer uses it; otherwise it falls back to `remediationOfDiagnostic`.

Tests updated:

- `AffineShapeSolverTest` now asserts remediation action text for:
  - underdetermined failures,
  - inconsistent failures,
  - non-integral failures,
  - non-positive failures.

Semantics unchanged:

- Inference equations remain maximal padded-extent.
- Evaluation remains zero-padded for out-of-range reads.

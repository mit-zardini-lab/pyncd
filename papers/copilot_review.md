# Copilot Code Review — Lean implementation (`main`)

Date: 2026-06-23  
Scope reviewed: `leanncd/LeanNCD/**/*.lean`, `leanncd/test/**/*.lean`, and context docs in `papers/` + `docs/` (especially `papers/leanncd.md`, `papers/graded_prop.md`, `leanncd/SORRY_INVENTORY.md`).

## Executive assessment

The implementation is thoughtfully structured and heavily documented, with strong progress in the executable DSL/evaluator path. However, the formal bridge that underpins the main categorical claim is still materially incomplete on `main`: several core theorem/instance obligations are still `sorry`, and a few executable bridge paths currently rely on lossy defaults or partial derivations. This is acceptable for an explicit staged milestone, but it is **not yet a complete end-to-end formal realization** of the design thesis.

---

## Critical findings

### 1) Core graded-PROP flagship instance is still signature-only
- **Files:** `leanncd/LeanNCD/Instances/StBr.lean:13-24`, `leanncd/LeanNCD/Core/Weave.lean:29-32`
- **Issue:** `instDGradedStBr` has most semantic fields as `sorry` (`act`, `δ`, `δ0`, `υ`, `α`, `sh_act`, `act_unit_assoc`, `υ_nat`, `dist_coh`, `broadcast_gen`). `weave_unique` is also `sorry`.
- **Why it matters:** The design promise (“prove once, inherit everywhere”) depends on these being real obligations. Until then, propositions specialized to `St→Br` are structurally present but not substantively discharged.
- **Recommendation:** Mark this explicitly as “interface-complete / proof-incomplete” in user-facing docs and gate any claims of formal completeness on closing these specific fields first.

### 2) Bridge from compiled/routed program to `BrMorph` remains unimplemented
- **Files:** `leanncd/LeanNCD/Bridge/Realize.lean:87-90`, `leanncd/LeanNCD/Bridge/SBr.lean:10`, `leanncd/LeanNCD/Bridge/Agreement.lean:9,15`
- **Issue:** `realize`, `realizeSBr`, `fromThreadedComposed`, and `realize_fromThreadedComposed_agree` are still `sorry`.
- **Why it matters:** This is the formal seam between executable pipeline artifacts and categorical morphisms; without it, the two tracks are not formally connected.
- **Recommendation:** Prioritize these before adding new theorem surface area. The project already identifies this as a central gap; sequencing should reflect that.

---

## High-severity findings

### 3) Stated “sorry-free seam” vs actual remaining `St` coherence sorries
- **Files:** `leanncd/LeanNCD/Base/St.lean:267-268`, `leanncd/SORRY_INVENTORY.md` (claims around seam progress)
- **Issue:** `swap_hexagon_fwd` and `swap_hexagon_rev` in `St` are still `sorry`, while surrounding documentation can be read as implying broader closure.
- **Why it matters:** This creates ambiguity for reviewers/users about what is genuinely proved versus staged.
- **Recommendation:** Keep claims narrowly scoped per module/field, and add a concise “remaining proof obligations by file+line” table that matches code exactly.

### 4) Grothendieck split is currently vacuous at relation/data level
- **Files:** `leanncd/LeanNCD/Grothendieck/Split.lean:14-16,41-45,54-56`
- **Issue:** `structuralCongruence` is `True`; `Dat` is `Unit`-constant; `grothendieck_split` is `sorry`.
- **Why it matters:** The file captures the shape but not the intended structural/data theorem content yet; using it as evidence of the split would be premature.
- **Recommendation:** Tag this module clearly as “schema placeholder” (not theorem implementation), and avoid citing it as realized proof content.

### 5) Algebra target and flagship construct are largely deferred
- **Files:** `leanncd/LeanNCD/Algebra/Target.lean:78-86`, `leanncd/LeanNCD/Algebra/Construct.lean:13-21,43`
- **Issue:** `TargetActegory` instance for `Mat ℝ` and `instAlgebraBrMatR` are mostly `sorry`; `construct_correspondence` is `sorry`.
- **Why it matters:** The paper-level construct/algebra story is represented but not discharged; theorem statements exist without implementation-level proof.
- **Recommendation:** Keep the excellent obstruction notes, but add explicit milestone gating in top-level docs (`LeanNCD.lean` or README-style entrypoint) to prevent over-interpretation.

### 6) `realizeStMat` accepts matrix dimension mismatches silently
- **Files:** `leanncd/LeanNCD/Bridge/Realize.lean:21-27`, `leanncd/LeanNCD/DSL/Target.lean:11-15`
- **Issue:** Realization uses `getD` fallbacks and does not enforce `domLen/codLen` or row/column consistency.
- **Why it matters:** Malformed presentation matrices can be turned into incorrect `StMat` values without any failure.
- **Recommendation:** Add explicit validation for declared lengths and matrix/bias dimensions; fail fast on mismatch.

### 7) Predicate equality semantics are currently approximate
- **Files:** `leanncd/LeanNCD/Eval/Gather.lean:21-23,37`, `leanncd/test/Eval/GatherTest.lean`
- **Issue:** `.ieq` is explicitly approximated (non-modular equality behavior), and evaluator tests do not cover it.
- **Why it matters:** Predicate programs relying on intended equality behavior may evaluate incorrectly.
- **Recommendation:** Implement faithful `.ieq` semantics (or reject unsupported form), and add focused tests.

---

## Medium findings

### 8) `realize` domain inference is documented as partial and can misidentify external inputs
- **Files:** `leanncd/LeanNCD/Bridge/Realize.lean:69-79`
- **Issue:** `dom` is taken from first step input weaves as a surrogate; comments acknowledge this is not fully faithful to routing.
- **Why it matters:** Even after replacing `sorry` body, this heuristic can encode wrong external boundary in some DAGs.
- **Recommendation:** Implement the documented routing walk (`step = nExternal`) before treating this path as canonical.

### 9) Pipeline routing fallback can silently route unknown reads to external slot 0
- **File:** `leanncd/LeanNCD/DSL/Pipeline/Lowering.lean:278-281`
- **Issue:** When read name is missing in both `nameToStep` and `extIndex`, route defaults to `Wire nExternal 0` via `.getD 0`.
- **Why it matters:** This can mask upstream dataflow errors and produce incorrect routed graphs.
- **Recommendation:** Replace default with explicit compile error for unresolved reads.

### 10) Several evaluator paths use permissive defaults (`getD`/zeros), which can hide shape/key errors
- **Examples:** `leanncd/LeanNCD/Eval/Tensor.lean:30`, `Eval/Shape.lean:104`, `Eval/Scan.lean:93`
- **Issue:** Missing coordinates/entries often map to defaults (`0`, empty tensors, etc.).
- **Why it matters:** Useful for totality, but risks converting invalid programs into plausible outputs.
- **Recommendation:** For critical boundaries, prefer explicit failure unless default behavior is semantically intentional and tested.

### 11) Serialization path can silently drop unsupported `SizeExpr`
- **Files:** `leanncd/LeanNCD/Acset/Csv.lean:67-73`, `leanncd/LeanNCD/Acset/Io.lean:15,18`
- **Issue:** Unsupported size encodings can become empty-string defaults through `.toOption.getD ""`.
- **Why it matters:** This can produce lossy CSV output without surfacing an error.
- **Recommendation:** Make write path return/propagate `Except` failures instead of defaulting.

### 12) `SizeExpr.toNumeric` is semantically lossy for `.sub`/`.div`
- **Files:** `leanncd/LeanNCD/DSL/SizeExpr.lean:33-43`, `leanncd/LeanNCD/Bridge/Realize.lean:8-10`
- **Issue:** Bridge conversion does not preserve full size-expression semantics.
- **Why it matters:** Realized symbolic size interpretation can diverge from DSL intent.
- **Recommendation:** Either support faithful conversion or reject unsupported forms at the bridge boundary.

### 13) Some formal tests are mostly axiom-presence checks
- **Files:** `test/Bridge/RealizeTest.lean:20`, `test/Bridge/SBrTest.lean:5`, `test/Algebra/ConstructTest.lean:12-14`, `test/Core/WeaveTest.lean:8-10`
- **Issue:** Several tests primarily confirm expected `sorryAx`/axiom status rather than semantics.
- **Why it matters:** Semantic regressions can pass if type signatures still elaborate.
- **Recommendation:** Add more behavior/equational tests that fail on semantic drift.

---

## Positive findings

1. The codebase is unusually well self-documented; obligations are generally honest and localized (`SORRY_INVENTORY.md` is high value).
2. Executable DSL/eval path appears robustly developed and broadly tested (`LeanNCD/Eval/*` + many targeted tests).
3. Architectural separation is strong: categorical core, seam adapter, executable pipeline, bridge, and acset IO are cleanly partitioned.
4. Bridge/evaluator comments clearly call out known fidelity limits instead of hiding them.

---

## Suggested priority order

1. Close `Bridge/Realize` + `Bridge/SBr` + `Bridge/Agreement` `sorry`s (formal executable↔categorical connection).
2. Replace routing fallback-to-slot-0 with hard error.
3. Close `Instances/StBr` core graded obligations used by flagship specialization.
4. Close `St` hexagon obligations and reconcile status docs with code.
5. Upgrade `Grothendieck/Split` from placeholder relation/data functor to intended semantics.

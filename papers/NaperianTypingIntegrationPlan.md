# Naperian Typing Integration Plan

This document turns the strategy in [NaperianTyping.md](./NaperianTyping.md) into a
concrete implementation plan for the current LeanNCD codebase.

It is written against the code as it exists today:

- affine read/write syntax in `LeanNCD/DSL/Ast.lean`,
- affine-LHS reclassification in `LeanNCD/DSL/Pipeline/Structural.lean`,
- routed symbolic `StMatP` generation in `LeanNCD/DSL/Pipeline/Lowering.lean`,
- runtime affine size inference in `LeanNCD/Eval/Shape.lean`,
- evaluation entrypoint in `LeanNCD/Eval/Eval.lean`.

The main update relative to earlier drafts is a stricter separation between:

1. **symbolic Naperian/reindex typing before size solving**, and
2. **concrete finite-point instantiation after size solving**.

That separation is required because the current axis size story is symbolic (`Numeric`,
`SizeExpr`) during compilation, while the affine solver learns concrete extents only when
runtime input tensor shapes are available.

## 1. Current pipeline and staging constraint

Today the compile/eval split is already:

```text
assignUIDs
resolveDecls
checkReadRanks
checkDtypes
unifyAxes
lowerArith
finalizeScans
splitNonlins
schedule
route
inferAxisSizes
evalPlain / evalScan / scatter evaluation
```

Two facts drive the integration strategy:

1. `route` produces **symbolic** routed shapes using `SizeExpr.var a.name`, so compile-time
   reindexing and degree construction do not require solved extents.
2. `inferAxisSizes` runs in `evalScheduled`, using concrete input tensor shapes plus affine
   reads to solve `UID -> Nat`.

Therefore, the implementation must **not** collapse all Naperian work into a single pass.
Instead it should split into:

- a **symbolic typed reindexing/law-checking layer** before `inferAxisSizes`, and
- a **concrete finite-point layer** after `inferAxisSizes`.

## 2. Recommended future pass order

The recommended staged pipeline is:

```text
assignUIDs
resolveDecls
checkReadRanks
checkDtypes
unifyAxes
lowerArith
finalizeScans
splitNonlins
schedule
elaborateAffineReindexings      -- new
checkNaperianSymbolic           -- new
route                           -- refactored to consume symbolic typed reindexings
inferAxisSizes                  -- existing runtime/value-level affine solver
instantiateConcreteNaperian     -- new
evalPlain / evalScan / scatter evaluation
```

Interpretation:

- `elaborateAffineReindexings` and `checkNaperianSymbolic` are **compile-time** and work
  over canonical UIDs plus symbolic shape terms.
- `inferAxisSizes` remains the existing **runtime/value-level** solver and is not replaced
  by dependent types.
- `instantiateConcreteNaperian` is where solved extents become finite point data.

## 3. File-by-file implementation plan

### 3.1 New file: `LeanNCD/Core/Naperian.lean`

Create the core symbolic API here:

- `NaperianAxis`
- `NaperianFamily`
- `BroadcastJoin`
- `ReindexAction`
- `PointwiseLift`
- `AxisReduce`
- `ev_naperian`
- `naperian_jointly_monic` (statement only at first)

This file should stay **symbolic**:

- shape objects remain `StObj`,
- extents may remain symbolic,
- no assumption that every axis already has a concrete `Fintype`.

The most important early responsibility of this file is to make the intended laws explicit,
not to force concrete enumeration too early.

### 3.2 New file: `LeanNCD/Instances/StNaperian.lean`

Add the `StObj`-specific instance strategy here.

Use an auxiliary class such as:

```lean
class AxisPointData (a : Axis) where
  El : Type
  finite : Fintype El
  -- bridge from concrete points to `I_D ⟶ a`
```

and then build:

- point data for one axis,
- product/append point data for `StObj := List Axis`,
- `NaperianAxis StObj`.

Important staging rule:

- this file may define the **shape** of the instance and the pointwise laws,
- but any path that needs actual finite point sets for runtime evaluation must be supplied
  only after extents are concretely known.

So the instance should be written to support **symbolic compilation first**, with concrete
point realization deferred to the post-solver stage.

### 3.3 `LeanNCD/DSL/Pipeline/Structural.lean`

Keep:

- `assignUIDs`
- `resolveDecls`
- `checkReadRanks`
- `checkDtypes`
- `unifyAxes`
- `lowerArith`
- `finalizeScans`

No size solver should move here.

`lowerArith` should remain responsible for:

- affine-LHS classification to `Stmt.scatter`,
- overlap-policy checks that do not need concrete inferred extents.

It should **not** try to instantiate concrete Naperian point sets.

### 3.4 `LeanNCD/DSL/Pipeline/Lowering.lean`

This file becomes the home of the new symbolic affine/Naperian staging.

#### Step 1: add `elaborateAffineReindexings`

This pass should:

- normalize every `IdxExpr` into affine-row form,
- assign canonical UID column order,
- attach one symbolic `St` row per coordinate position,
- detect any non-affine cases before routing.

It can reuse/refactor existing ideas from:

- `idxAxes`
- `idxToRow`
- `idxAffineForm` (currently in `Eval/Shape.lean`)

but should produce a compile-time artifact instead of recomputing ad hoc later.

#### Step 2: add `checkNaperianSymbolic`

This pass should enforce the symbolic invariants that do **not** need concrete extents:

- each `IdxExpr` lowers to exactly one affine row over canonical degree axes,
- source tensor rank equals row count,
- degree construction is canonical under UID equality,
- pointwise ops preserve degree,
- reduction removes only contracted axes,
- scan arithmetic stays separated from ordinary static reindexing.

This is the best location for early `BroadcastJoin` / `ReindexAction` integration.

#### Step 3: refactor `route`

`route` should consume pre-elaborated symbolic reindexings rather than discovering them
itself. It remains the pass that packages:

- `degree`,
- `inputWeaves`,
- `outputWeaves`,
- symbolic `StMatP`,
- final `ThreadedComposed` routing.

In other words: `route` should become the **consumer** of symbolic affine/Naperian data,
not its primary synthesizer.

### 3.5 `LeanNCD/DSL/Target.lean`

Keep the existing computable `StMatP` representation for compatibility, but introduce the
dependent strengthening in parallel.

Recommended direction:

- keep `structure StMatP` for the current executable path,
- add a typed form alongside it, e.g. `StMatP' (dom cod : StObj)`,
- add conversion/bridge lemmas between the symbolic executable form and the typed core form.

This preserves the current back-end while allowing the law-level API to become properly
typed.

### 3.6 `LeanNCD/Eval/Shape.lean`

This file remains the home of the affine size solver.

Do **not** move `inferAxisSizes` earlier in the compile pipeline.

Its responsibilities remain:

- collect affine read positions,
- solve `UID -> Nat` using padded maximal-extent semantics,
- validate underdetermined/inconsistent/non-integral/non-positive cases,
- emit warnings for padded-access cases.

The change here is interface-level, not semantic:

- accept input that is already symbolically normalized where possible,
- return a size environment suitable for concrete Naperian instantiation.

In particular, dependent typing should constrain the solver's **input format**, but should
not replace the solver's value-level job.

### 3.7 New file: `LeanNCD/Eval/NaperianRuntime.lean` (or similar)

Add a post-solver realization layer here.

Responsibilities:

- consume routed symbolic shapes/reindexings,
- consume solved `UID -> Nat`,
- build concrete point-enumeration data for shapes used at runtime,
- provide the bridge from symbolic `NaperianAxis` structure to executable finite
  coordinate loops.

This is where `instantiateConcreteNaperian` should live.

It is deliberately separated from `Core/Naperian.lean` because it depends on runtime size
knowledge and should not contaminate the symbolic layer.

### 3.8 `LeanNCD/Eval/Eval.lean`

Keep the current high-level structure:

- compile,
- solve sizes,
- evaluate.

The only staging change is to insert:

```text
inferAxisSizes
instantiateConcreteNaperian
evalPlain / evalScan / scatter evaluation
```

between routing and concrete execution.

### 3.9 Minimal changes elsewhere

Touch as little as possible in:

- `LeanNCD/Core/Graded.lean`
- `LeanNCD/Base/ColoredPROP.lean`
- `LeanNCD/Instances/StBr.lean`
- `LeanNCD.lean`

Early milestones should add APIs and bridge lemmas rather than force broad rewrites.

## 4. Proof impact and per-field analysis

### 4.1 What gets easier immediately

The strongest immediate benefit is around **reindex soundness**:

- typed domain/codomain rank,
- composition/identity laws for reindexings,
- clearer proof targets for `route`-level invariants.

This follows from moving affine rows into a typed/symbolic core before evaluation.

### 4.2 What does not get solved automatically

The following remain substantial:

- `act` on `Br` morphisms,
- quotient interaction for `BrMorph`,
- `broadcast_gen`,
- `weave_unique`,
- coherence fields that still require extensionality and the action itself.

Naperian typing gives a better *proof shape* for these, but does not discharge them alone.

### 4.3 Why the solver split matters for proofs

If concrete finite-point data were forced too early, proofs would become entangled with the
runtime size environment. Keeping symbolic typing before the solver avoids this:

- symbolic proofs talk about `IdxExpr`, `St`, `degree`, and coherence,
- runtime proofs talk about the instantiated `UID -> Nat` environment and concrete point
  enumeration.

That separation is cleaner both logically and implementation-wise.

## 5. Milestone sequencing

### Milestone 0 — Baseline confirmation

- confirm current compile/eval behavior,
- keep `inferAxisSizes` exactly where it is,
- add tests documenting the symbolic-vs-concrete split.

### Milestone 1 — Core symbolic API

- add `Core/Naperian.lean`,
- state the mixin classes and key laws,
- no runtime point enumeration yet.

### Milestone 2 — Symbolic affine elaboration

- add `elaborateAffineReindexings`,
- move affine-row normalization out of ad hoc helpers and into the compile pipeline,
- keep `route` behavior unchanged by using a bridging adapter first.

### Milestone 3 — Symbolic Naperian checks

- add `checkNaperianSymbolic`,
- enforce degree/reindex/pointwise/reduction/scan invariants,
- keep all checks independent of concrete extents.

### Milestone 4 — Refactor `route`

- make `route` consume the symbolic affine artifacts,
- introduce the typed `StMat` bridge,
- preserve the existing executable `ThreadedComposed` output.

### Milestone 5 — Concrete point instantiation

- add the runtime instantiation layer,
- consume `inferAxisSizes` output,
- produce finite point sets and coordinate enumeration data.

### Milestone 6 — Proof tightening

- connect symbolic laws to concrete runtime instantiation,
- pursue `δ`/`δ0`/`υ`/`α` proof simplifications,
- add law-level tests alongside existing executable tests.

### Milestone 7 — Optional dependent migration

- migrate more of the executable path from loose `StMatP` records to typed structures,
- only as touched by later work,
- avoid a flag day rewrite.

### Minimum viable integration

The minimum useful version is:

1. `Core/Naperian.lean` with symbolic classes,
2. symbolic affine elaboration + `checkNaperianSymbolic`,
3. runtime post-solver concrete Naperian instantiation hook.

That already captures the solver-ordering insight without forcing a full back-end rewrite.

## 6. Risks and mitigations

### 6.1 Symbolic-size blocker

Risk:

- `Axis.size` / `SizeExpr` do not determine a `Fintype` during compilation.

Mitigation:

- keep symbolic Naperian checks pre-solver,
- instantiate finite point data only post-solver.

### 6.2 Route/solver duplication

Risk:

- affine normalization logic becomes duplicated between compile and eval code.

Mitigation:

- extract shared affine normalization helpers,
- let compile and eval share the normalization artifact even if they use it differently.

### 6.3 Over-eager dependent migration

Risk:

- trying to migrate the whole executable path at once stalls progress.

Mitigation:

- keep the current executable `StMatP` form alive,
- add typed bridges in parallel,
- migrate incrementally.

### 6.4 Quotient and instance complexity

Risk:

- `Br` quotient interaction and instance diamonds complicate the core proof story.

Mitigation:

- keep the first milestones focused on the symbolic `St` side and routing invariants,
- defer full `Br` action integration until the symbolic layer is stable.

### 6.5 Tensor-order convention for `α`

Risk:

- getting the `Q ⊗ P` vs `P ⊗ Q` order wrong in action associativity.

Mitigation:

- encode the convention explicitly in the API and tests from the start,
- reuse the `alphaElEquiv` convention described in `NaperianTyping.md`.

## 7. Summary

The updated implementation plan is:

- **do symbolic Naperian typing before the affine solver**,
- **keep the affine solver as the runtime/value-level source of concrete extents**,
- **instantiate concrete finite Naperian point data only after solving extents**.

This lets the codebase gain the benefits of typed reindexing and law-level structure
without breaking the current padded-semantics evaluation model or forcing premature
runtime-size assumptions into compile-time code.

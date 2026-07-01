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

> **Prerequisite / sequencing.** Do not start this in parallel with the active multi-output
> `BrBase` + `compile_wellFormed` effort (`docs/superpowers/plans/2026-06-26-multioutput-impl-plan.md`).
> This plan touches the categorical core and, near-term, *adds* surface area before removing
> sorries. Begin with the [minimum viable integration](#minimum-viable-integration) once the
> well-formedness proof lands. The projected proof savings in
> [NaperianTyping.md §6](./NaperianTyping.md) are a **hypothesis to validate at Milestone 2**,
> and are gated on the [Milestone 1.5 circularity spike](#milestone-15--circularity-spike-s1-gate-for-track-b-spike)
> succeeding — not a committed deliverable.

## 0. Code audit findings (2026-07-01) — read this first

A ground-truth audit of the LeanNCD **code** (not the design docs) reshapes this plan:

1. **`act` does not exist — it is a bare `sorry`.** The instance
   `instDGradedStBr : DGradedColoredPROP StObj BrObj` (`LeanNCD/Instances/StBr.lean:13`) has exactly
   one real field (`sh`); the other **10** — `act`, `δ`, `δ0`, `υ`, `α`, `sh_act`,
   `act_unit_assoc`, `υ_nat`, `dist_coh`, `broadcast_gen` — are all `:= sorry`. Naperian typing (per
   [NaperianTyping.md](./NaperianTyping.md)) is a reinterpretation layer *on top of* the graded
   action `act`. **The action it sits on is the single largest unbuilt piece**, and no coherence-iso
   benefit is realizable until `act` exists. Defining `act` is a prerequisite, not a Naperian
   deliverable.
2. **The `Br`/`BrMorph` SMC quotient is DONE and sorry-free** (`LeanNCD/Base/Br.lean:390`):
   `swap_swap`/`tensorHom_comp` discharge via `Quotient.sound`. The keystone re-presentation concern
   is resolved. So `act` is *well-posed* — it must be a `Quotient.lift` over `Br.Hom` respecting the
   ~20 `Rel` constructors plus functor laws — but that is still substantial new proof work.
3. **`brCancelPoint` (free-strict-SMC normal form) is an isolated `sorry`, off the executable path.**
   It gates only `weave_unique` (`Core/Weave.lean:32`, also `sorry`), which has **no consumers**.
   `BrNF` (the NbE route) is in progress and explicitly not load-bearing; slice extraction uses
   `St` elementality (proved). So `brCancelPoint`/`weave_unique` are **not** on the critical path for
   the executable compiler or for the symbolic Naperian layer.
4. **The evaluation-side action `actV` has a recorded *impossibility*, not just a gap.** Milestone H
   of `SORRY_INVENTORY.md` records that a faithful `actV` in `FGModuleCat ℝ` is mathematically
   impossible (it needs an `R`-semimodule carrier, not vector spaces). Naperian does not fix this;
   Track A (below) is unaffected by it, but any eval-side actegory work inherits it.
5. **The multi-output `compile_wellFormed` surface is DISJOINT from Naperian.** Every `WellFormed`
   conjunct (`wf_typeMatch`, `wf_dom`, `wf_topo`, `topo_bound`, output-count) is symbolic axis-list /
   wire-dispatch / `List.length` bookkeeping; `realize` merely *consumes* `WellFormed`. Naperian's
   only possible contribution is to the shape-*order*-matching sorries (`buildStep_output_fixedAxes`,
   `wf_typeMatch`, `internal_pointwise`) — and *only* if weaves are re-typed from `List`-with-`.getD`
   to representable/`Fin`-indexed functions. The hardest multi-output blockers (`topo_bound`'s
   false-as-stated modeling gap; `stepPiece`/`finalPiece` pool reconciliation) are untouched by
   better shape types.
6. **Naperian is greenfield and Mathlib does not supply the base classes.** No `Naperian`/`El`/
   `ev_p`-as-Naperian scaffolding exists. Mathlib has no Haskell-style `Representable`/`Log` or
   `Foldable` class (verified against the pinned `v4.30.0` toolchain) — these must be defined locally.

### 0.1 Two tracks

The findings split this plan into two tracks with very different risk/value profiles:

- **Track A — symbolic typed reindexing layer (feasible now, modest real value, low risk).** The
  `elaborateAffineReindexings` + `checkNaperianSymbolic` passes and the typed `StMatP` reindexing
  core (§2, §3, §5 M2–M4). Orthogonal to `act`, to the coherence tower, and to the multi-output
  work. It hardens the compiler (typed reindex rank/codomain, canonical degree, early law checks).
  Its wins are the **"reindex soundness"** items in §4.1 — **not** the 50–70% coherence figure.
- **Track B — categorical / representability payoff (gated, expensive, speculative).** The
  `NaperianAxis`/`NaperianFamily`/`naperian_jointly_monic` layer meant to shrink the coherence
  sorries. **Blocked on `act` (finding 1)** and on the circularity spike. The projected 50–70%
  reduction lives here and is unvalidated. Do not begin Track B until (a) multi-output lands, (b)
  `act` is defined or scoped by spike S0, and (c) spikes S0/S1 pass.

### 0.2 Spikes to run before committing

| Spike | Question | Exit criterion | Detail |
| --- | --- | --- | --- |
| **S0 — `act` definability** | Can `act` be a `Quotient.lift` over `Br.Hom` respecting every `Rel` constructor, and how large is that proof? | A skeleton `act` with the `Rel`-respecting obligations enumerated + a size estimate; or a named blocker. | §5 Milestone 0.5 |
| **S1 — `jointly_monic` circularity** | Is there an acyclic proof of `naperian_jointly_monic` (not routed through `broadcast_gen`)? | Acyclic proof/skeleton, or the explicit extra assumption it needs. | §5 Milestone 1.5 |
| **S2 — dependent-weave re-typing** | Would re-typing weaves as `Fin`-indexed/representable make `wf_typeMatch`/`buildStep_output_fixedAxes` definitional, and what does it cost the executable path? | A spike branch re-typing ONE weave/shape and re-proving one shape-match lemma; a cost/benefit note. | §5 Milestone 4.5 |
| **S3 — `actV` impossibility** | Confirm the `FGModuleCat ℝ` obstruction; is any eval-side actegory work in scope? | Written confirmation + a decision on semimodule redesign scope. | §6.6 |

**S0 is pivotal.** If `act` turns out far larger than Track B's projected savings, Track B is
net-negative and the effort should stop at Track A + the [minimum viable
integration](#minimum-viable-integration).

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

**Foundational constraint — split finiteness out of the symbolic class.** As written in
[NaperianTyping.md §5.2](./NaperianTyping.md), `NaperianAxis` carries `finite : ∀ P, Fintype (El P)`
as a *class field*. That field **cannot be constructed at the symbolic layer**: `Axis.size` is
`Numeric`/`SizeExpr` and does not determine a `Fintype`. So a single monolithic `NaperianAxis`
instance for `StObj` is impossible before size solving — the plan and the paper's class definition
are in direct tension unless the class is split. The required resolution:

- **`NaperianAxis` (symbolic, constructible now):** carries only the index-category structure —
  `El` as an abstract point *type*, `mapEl`, `point_hom`, and the strong-monoidal coherence
  equivalences. **No `Fintype` field.**
- **`AxisPointData` / a separate `FiniteNaperian` class (concrete, post-solver):** supplies
  `Fintype El` and the enumerable coordinate data, instantiated only after `inferAxisSizes` gives
  concrete extents.

```lean
-- concrete layer only; NOT a field of the symbolic NaperianAxis
class AxisPointData (a : Axis) where
  El : Type
  finite : Fintype El
  -- bridge from concrete points to `I_D ⟶ a`
```

Then build: point data for one axis → product/append point data for `StObj := List Axis` → the
concrete finite instance. The symbolic `NaperianAxis StObj` instance is what compilation uses;
the concrete finite instance is realized in the post-solver stage (§3.7). **Action item:** before
Milestone 1, adjust the `NaperianAxis` class in `NaperianTyping.md §5.2` to remove the `finite`
field (move it to the concrete layer), so the symbolic instance is actually constructible.

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

### 4.1 What gets easier immediately (Track A — real, ungated)

The strongest immediate benefit is around **reindex soundness**, and it does **not** depend on
`act` or on any coherence lemma:

- typed domain/codomain rank,
- composition/identity laws for reindexings,
- clearer proof targets for `route`-level invariants.

This follows from moving affine rows into a typed/symbolic core before evaluation. These are the
*only* wins this plan can promise without the Track-B prerequisites landing.

**Possible (not automatic) Track-A win on the multi-output shape-match sorries.** Per the audit
(finding 5), `buildStep_output_fixedAxes`, `wf_typeMatch`, and `internal_pointwise` are all "two
weaves have the same `fixedAxesP` (same axes, same order) ⇒ same `ArrayType`", proved today with
`List.getD … default` positional bookkeeping. If weaves were re-typed as `Fin`-indexed/representable
functions, these become near-definitional. That is a genuine but **localized** simplification, it
requires re-typing the executable weave representation, and it is **not** what unblocks multi-output
(see §4.2). Validate it with **spike S2** before assuming it.

### 4.2 What does NOT get solved (and what Naperian is orthogonal to)

Substantial and **gating Track B** (audit findings 1–2):

- **`act` on `Br` morphisms is unbuilt (a `sorry`)** — it must be defined (via `Quotient.lift`
  respecting all `Rel` constructors + functor laws) before *any* coherence-iso benefit exists. This
  is the largest single piece and is a prerequisite, not a payoff. Scope it with **spike S0**.
- `broadcast_gen`, `weave_unique`, and the coherence fields `δ`/`δ0`/`υ`/`α`/`act_unit_assoc`/
  `υ_nat`/`dist_coh` — all still `sorry`, all downstream of `act`.

Naperian gives a better *proof shape* for these, but does not discharge them, and cannot even be
stated usefully until `act` exists.

Explicitly **orthogonal** to Naperian typing (audit findings 3–5) — do not expect Naperian to touch
these:

- The multi-output blockers `topo_bound` (false-as-stated modeling gap: cycles + scan self-reads)
  and `stepPiece`/`finalPiece` pool reconciliation — `Wire`-list membership/order, not shape types.
- `brCancelPoint`/`weave_unique` — isolated, off the executable path, no consumers.
- The `actV` (`FGModuleCat ℝ`) impossibility — an eval-side semantic-algebra redesign (spike S3).

### 4.3 Why the solver split matters for proofs

If concrete finite-point data were forced too early, proofs would become entangled with the
runtime size environment. Keeping symbolic typing before the solver avoids this:

- symbolic proofs talk about `IdxExpr`, `St`, `degree`, and coherence,
- runtime proofs talk about the instantiated `UID -> Nat` environment and concrete point
  enumeration.

That separation is cleaner both logically and implementation-wise.

## 5. Milestone sequencing

**Top-level order (revised per the §0 findings):**

1. **Finish the multi-output `compile_wellFormed` effort first, to completion.** It is orthogonal
   to everything here (audit finding 5) and in-flight; do not interleave. Accept that if spike S2
   later succeeds, a few shape-match lemmas may be re-typed — a bounded, acceptable "prove-twice"
   risk that is cheaper than blocking real work on a speculative refactor.
2. **Track A (M0 → M2 → M3 → M4)** — the symbolic typed-reindexing layer. Deliverable and valuable
   regardless of Track B. Run spike **S2** (M4.5) here to decide the weave re-typing.
3. **Spikes S0 (M0.5) and S1 (M1.5)** — gate Track B. Run S0 early; it is pivotal.
4. **Track B (M6 coherence work)** — only if S0 and S1 pass and `act` is defined.

Milestones are tagged **[A]** / **[B]** / **[spike]** below.

### Milestone 0 — Baseline confirmation **[A]**

- confirm current compile/eval behavior,
- keep `inferAxisSizes` exactly where it is,
- add tests documenting the symbolic-vs-concrete split.

### Milestone 0.5 — `act`-definability spike (S0, GATE for Track B) **[spike]**

**Run before committing to any Track-B milestone.** `act` is currently a bare `sorry`
(`Instances/StBr.lean:15`); the `Br`/`BrMorph` quotient it must lift over is done and sorry-free
(`Base/Br.lean:390`), so the task is well-posed but unmeasured.

- Sketch `act` as `Quotient.lift` over `Br.Hom`; enumerate the `Rel` constructors it must respect
  (congruence, category/bifunctor laws, interchange, braid involution/naturality/hexagons, the
  cast-crossing unit/assoc coherences, CD comonoid laws) and the functor laws (`map_id`/`map_comp`).
- Estimate the proof size and identify any constructor that looks intractable.

**Exit criterion:** a skeleton `act` with the obligations listed and a size estimate, **or** a named
blocker. **If `act` dwarfs Track B's projected savings, stop at Track A + the [minimum viable
integration](#minimum-viable-integration).**

### Milestone 1 — Core symbolic API **[A/B boundary]**

- add `Core/Naperian.lean`,
- state the mixin classes and key laws (statements only; no `act` dependence yet),
- no runtime point enumeration yet.

### Milestone 1.5 — circularity spike (S1, GATE for Track B) **[spike]**

**This is a gate, not optional.** Before building any typeclass scaffolding on top of it,
establish an **acyclic** proof route for `naperian_jointly_monic` (lifted `P`-indexed families
are separated by their `ev_p` coordinate evaluations). The hazard: proving it *from*
`broadcast_gen` while also using it to prove `broadcast_gen` proves neither.

Concretely, in this spike:

- attempt a **direct** proof over `St` evaluations that does **not** reference `broadcast_gen`
  (e.g. via the finite point enumeration of `El` and function extensionality on coordinates), OR
- if no direct proof is found, write down the **explicit extra assumption** it requires and
  confirm that assumption does not itself depend on `broadcast_gen`.

**Exit criterion:** either an acyclic proof/skeleton of `naperian_jointly_monic`, or a precise
statement of the assumption it needs. **If neither is achievable, stop at the
[minimum viable integration](#minimum-viable-integration)** — the projected proof savings in
[NaperianTyping.md §6](./NaperianTyping.md) do not hold, and Milestones 2+ should not be funded on
the expectation of them.

### Milestone 2 — Symbolic affine elaboration **[A]**

- add `elaborateAffineReindexings`,
- move affine-row normalization out of ad hoc helpers and into the compile pipeline,
- keep `route` behavior unchanged by using a bridging adapter first.

### Milestone 3 — Symbolic Naperian checks **[A]**

- add `checkNaperianSymbolic`,
- enforce degree/reindex/pointwise/reduction/scan invariants,
- keep all checks independent of concrete extents.

### Milestone 4 — Refactor `route` **[A]**

- make `route` consume the symbolic affine artifacts,
- introduce the typed `StMat` bridge,
- preserve the existing executable `ThreadedComposed` output.

### Milestone 4.5 — dependent-weave re-typing spike (S2) **[spike]**

Decides whether the *one* place Naperian could touch the multi-output proofs is worth it (§4.1).

- On a throwaway branch, re-type ONE weave/shape from `List WeaveSlotP` (accessed via `.getD …
  default`) to a `Fin`-indexed / representable form.
- Re-prove one shape-match lemma against it — pick `buildStep_output_fixedAxes` or the
  `weaveToArrayType_congr` step of `wf_typeMatch`.
- Measure: did the `.getD`-default bookkeeping and length side-conditions actually disappear, and
  what did the executable path (eval, `StMatP`, tests) pay for the re-typing?

**Exit criterion:** a cost/benefit note. If the executable-path cost outweighs the proof savings,
keep `List`-based weaves and close those lemmas by hand (as the multi-output effort already does).

### Milestone 5 — Concrete point instantiation **[A]**

- add the runtime instantiation layer,
- consume `inferAxisSizes` output,
- produce finite point sets and coordinate enumeration data.

### Milestone 6 — Proof tightening **[B — gated on S0/S1 and on `act` being defined]**

- **Prerequisite: `act` is implemented** (not this plan's deliverable; see M0.5/S0). Without it the
  items below cannot even be stated.
- connect symbolic laws to concrete runtime instantiation,
- pursue `δ`/`δ0`/`υ`/`α` proof simplifications,
- add law-level tests alongside existing executable tests.

### Milestone 7 — Optional dependent migration **[A, opportunistic]**

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

- the `Br`/`BrMorph` quotient itself is done and sorry-free (audit finding 2), so this is now
  primarily about defining `act` over it — see 6.7 and spike S0,
- keep the first milestones focused on the symbolic `St` side and routing invariants,
- defer full `Br` action integration until the symbolic layer is stable.

### 6.5 Tensor-order convention for `α`

Risk:

- getting the `Q ⊗ P` vs `P ⊗ Q` order wrong in action associativity.

Mitigation:

- encode the convention explicitly in the API and tests from the start,
- reuse the `alphaElEquiv` convention described in `NaperianTyping.md` (the `Q ⊗ P` order is
  confirmed correct against `graded_prop.md` Def 3.1).

### 6.6 Eval-side `actV` impossibility (recorded obstruction)

Risk:

- Milestone H of `SORRY_INVENTORY.md` records that a **faithful `actV` in `FGModuleCat ℝ` is
  mathematically impossible** — it needs an `R`-semimodule carrier, not vector spaces. Any
  Naperian work that reaches the *evaluation-side* actegory inherits this.

Mitigation:

- Track A (symbolic reindexing) does **not** touch `actV` and is unaffected — keep the eval-side
  actegory out of scope unless a semimodule redesign is explicitly funded (spike S3);
- treat this as a separate semantic-algebra decision, not a Naperian-typing task.

### 6.7 The gating risk: `act` is unbuilt (Track B has no floor without it)

Risk:

- The entire Track-B payoff assumes `act` exists; today it is a `sorry` (audit finding 1). If S0
  shows `act` is very large, the projected coherence savings are net-negative — you would spend
  more building `act` than Naperian typing saves.

Mitigation:

- run spike **S0 (M0.5) first** and treat it as a hard gate;
- ship Track A regardless (it needs no `act`);
- do not quote the 50–70% figure as a plan deliverable until S0 + S1 have passed and `act` is
  defined.

### 6.8 "Prove-twice" risk on multi-output shape lemmas

Risk:

- Finishing multi-output first means proving `wf_typeMatch`/`buildStep_output_fixedAxes` by hand;
  a later successful S2 weave re-typing could make some of them near-definitional, redoing work.

Mitigation:

- accept it — the rework is bounded and localized, and is far cheaper than blocking the in-flight
  verified-compiler work on a speculative refactor. Let S2's cost/benefit note decide whether the
  re-typing ever happens.

## 7. Summary

After the 2026-07-01 code audit (§0), the plan is:

- **Finish the multi-output `compile_wellFormed` effort first.** It is orthogonal to everything
  here; Naperian offers it almost nothing (at most the one shape-match simplification, spike S2).
- **Ship Track A** — the symbolic typed-reindexing layer — as the real, ungated deliverable:
  - do symbolic Naperian typing before the affine solver,
  - keep the affine solver as the runtime/value-level source of concrete extents,
  - instantiate concrete finite Naperian point data only after solving extents.
  This hardens the compiler (typed reindex rank/codomain, canonical degree, early law checks)
  without breaking the padded-semantics evaluation model.
- **Gate Track B on spikes.** The coherence/representability payoff (the projected 50–70% sorry
  reduction) sits on top of `act`, which is currently a `sorry`. Run spike **S0** (`act`
  definability) and **S1** (`jointly_monic` circularity) before committing; if either fails, stop
  at Track A + the minimum viable integration. Do not treat the 50–70% figure as a deliverable.

In one line: **Track A is worth doing on its own merits; Track B is a bet on `act` that must be
priced by spike S0 before it is funded — and neither should delay the multi-output work.**

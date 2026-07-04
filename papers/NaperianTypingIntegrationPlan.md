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

> **Prerequisite / sequencing — UPDATED 2026-07-03: prerequisite MET.** The multi-output `BrBase`
> + `compile_wellFormed` effort (`docs/superpowers/plans/2026-06-26-multioutput-impl-plan.md`) is
> **complete**, and the follow-on §8.2 acset agreement work
> (`docs/superpowers/plans/2026-07-01-acset-agreement-impl-plan.md`, Tasks A–E) has also landed:
> `compile_wellFormed`, `realize`, and `realize_fromThreadedComposed_agree` are all sorry-free.
> **Track A of this plan can start now.** Track B remains gated — `act` and all ten coherence
> fields on `instDGradedStBr` are unchanged, still bare `sorry` (`LeanNCD/Instances/StBr.lean`),
> so spike S0 has not been run and nothing in the projected proof savings of
> [NaperianTyping.md §6](./NaperianTyping.md) should be treated as validated. Run **S0** before
> committing to any Track-B milestone.

## 0. Code audit findings (2026-07-01) — read this first

> **Re-audit, 2026-07-03.** Findings 1–4 and 6 below were re-checked against the current tree and
> are **unchanged**: `act` and the ten `instDGradedStBr` coherence fields are still bare `sorry`
> (`LeanNCD/Instances/StBr.lean`); the `Br`/`BrMorph` quotient is still sorry-free; `brCancelPoint`
> is still an isolated, off-critical-path `sorry`; the `actV` impossibility still stands; Mathlib
> still has no `Representable`/`Foldable` base classes. **Finding 5 is superseded — see the update
> appended to it below.**
>
> **Also 2026-07-03 (after spike S0):** the `sh_act` *class field* in `Core/Graded.lean` was relaxed
> from a strict `=` to a canonical iso `≅` (the object-level blocker S0 surfaced — see §0.2 result
> box). The `instDGradedStBr.sh_act` *instance* field is still `:= sorry` (finding 1's count is
> unchanged), but its target type is now the iso; `lake build` is green.

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
   to representable/`Fin`-indexed functions.

   **UPDATE (2026-07-03): this entire surface is now DONE, by hand, without Naperian.**
   `compile_wellFormed`, `wf_typeMatch`, `internal_pointwise`, and `buildStep_output_fixedAxes` are
   all proved (zero `sorry` in `Bridge/Agreement.lean`, `DSL/Pipeline/RouteSpec.lean`). The
   `topo_bound` modeling gap (cycles + scan self-reads) was fixed by restating the theorem to take
   `routeCore` success (`hrc : routableInOrder`) rather than by re-typing weaves — confirming the
   original prediction that Naperian/representable re-typing was orthogonal to what actually
   unblocked multi-output. **Spike S2 (dependent-weave re-typing, Milestone 4.5) is now MOOT for
   this surface** — the lemmas it would have simplified are already closed by hand. S2 could still
   be run speculatively for its own sake (a cleaner shape representation going forward), but it no
   longer has a blocked-proof payoff to justify it.
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
- **Track B — categorical / representability payoff (gates now cleared; expensive).** The
  `NaperianAxis`/`NaperianFamily`/`naperian_jointly_monic` layer meant to shrink the coherence
  sorries. **Gate status (2026-07-03):** (a) multi-output has landed; (b) `act` scoped by S0 and its
  `sh_act` blocker resolved; (c) **both S0 and S1 pass** — S1 gave `jointly_monic` a Lean-verified
  acyclic route (no `broadcast_gen`; residual reroutes to the proved `St.elemental`). The projected
  50–70% reduction is no longer *blocked*, though it is still *unmeasured* until `act` + the coherence
  proofs are actually built. Track B is now a funded cost/benefit decision, not a feasibility gamble.

### 0.2 Spikes to run before committing

| Spike | Question | Exit criterion | Detail |
| --- | --- | --- | --- |
| **S0 — `act` definability** ✅ RUN 2026-07-03 | Can `act` be a `Quotient.lift` over `Br.Hom` respecting every `Rel` constructor, and how large is that proof? | ✅ DONE — morphism level tractable (20/21 `Rel` cases reuse proved `BrMorph` laws, ~400–650 lines for `act` alone). The object-level blocker it surfaced (strict `sh_act` unsatisfiable for ≥2-array bundles) is **RESOLVED**: `sh_act` relaxed `=` → `≅` in `Core/Graded.lean` (build green, zero proof consumers). `act` now has no design gate. Full report: [`leanncd/docs/superpowers/plans/2026-07-03-s0-act-definability-spike.md`](../leanncd/docs/superpowers/plans/2026-07-03-s0-act-definability-spike.md). | §5 Milestone 0.5 |
| **S1 — `jointly_monic` circularity** ✅ RUN 2026-07-03 | Is there an acyclic proof of `naperian_jointly_monic` (not routed through `broadcast_gen`)? | ✅ POSITIVE — acyclic route Lean-verified: `jointly_monic ⟸ representability (lifted object is the El P-power, legs = ev_p)` via `IsLimit.hom_ext`, axioms `[propext, Classical.choice, Quot.sound]`, no `broadcast_gen`, off `brCancelPoint`. **Residual probe refined the cost:** closing it for real is a *commitment-sized build* (reindex-half of `act` + concrete finite-`El` representability), NOT a cheap spike — `El P` isn't enumerable symbolically and a faithful `evSlice` is part of `act`. Still non-circular. Report: [`leanncd/docs/superpowers/plans/2026-07-03-s1-jointly-monic-spike.md`](../leanncd/docs/superpowers/plans/2026-07-03-s1-jointly-monic-spike.md). | §5 Milestone 1.5 |
| **S2 — dependent-weave re-typing** | Would re-typing weaves as `Fin`-indexed/representable make `wf_typeMatch`/`buildStep_output_fixedAxes` definitional, and what does it cost the executable path? | A spike branch re-typing ONE weave/shape and re-proving one shape-match lemma; a cost/benefit note. | §5 Milestone 4.5 |
| **S3 — `actV` impossibility** | Confirm the `FGModuleCat ℝ` obstruction; is any eval-side actegory work in scope? | Written confirmation + a decision on semimodule redesign scope. | §6.6 |

**S0 is pivotal.** If `act` turns out far larger than Track B's projected savings, Track B is
net-negative and the effort should stop at Track A + the [minimum viable
integration](#minimum-viable-integration).

> **S0 RESULT (2026-07-03).** The literal S0 question is answered **yes, tractably** — `act.map` is a
> well-posed `Quotient.lift`, and 20 of 21 `Rel` constructors discharge against `BrMorph` theorems
> that are *already proved* (the free-strict-SMC laws that made `Br : ColoredPROP` sorry-free). The
> `gen` leaf (batch-lift of one `BrBase`) is a bounded definition, not a `Rel` obligation. Estimate:
> **~400–650 lines for `act` alone** (functor + well-definedness + reindex + `map_id`/`map_comp`),
> excluding the `δ`/`δ0`/`υ`/`α`/`broadcast_gen` tower that sits on top of it.
>
> **The object-level blocker S0 surfaced is now RESOLVED (option 1 implemented).** The strict
> `sh_act` field (`sh*(X ⊛ P) = sh*(X) ⊗ P`) was unsatisfiable for ≥2-array bundles under per-array
> broadcast (the concatenated `sh_star` carries `|X|` copies of `P`, not one; a strict `=` admits no
> iso freedom). It has been **relaxed from `=` to a canonical braiding iso `≅`** in
> `Core/Graded.lean` — `lake build` green, and the re-trace found **zero proof consumers** (the field
> was referenced only by its declaration, doc comment, and the still-`sorry` instance). `act` now has
> **no remaining design gate**. (**Update: S1 has since also passed — see the S1 RESULT box below.**)
> Whether to build `act` is now a cost/benefit call (~400–650 lines + the coherence tower vs. Track
> B's projected savings), not a feasibility one.
>
> **S1 RESULT (2026-07-03).** `naperian_jointly_monic` has a **Lean-verified acyclic route**:
> `jointly_monic ⟸ representability (lifted object is the El P-power, legs = ev_p)` via
> `IsLimit.hom_ext`, axioms `[propext, Classical.choice, Quot.sound]`, **no `broadcast_gen`**. The
> reduction reroutes the obligation (it does not eliminate it) to St-slice separation, which reduces
> to the proved `St.elemental` — **not** `brCancelPoint`. The circularity hazard is avoided:
> `jointly_monic ← representability`, never `← broadcast_gen`. **Residual (refined by the residual
> probe — a commitment-sized build, NOT a cheap spike):** a faithful `evSlice` is the reindex-half of
> `act`, and `El P` is not enumerable symbolically (`Fintype (BrMorph [] P)` has no instance), so
> closing `jointly_monic` needs the `act` build + the concrete finite-`El` representability
> (`tabulate`/`lookup`) — real work, but still non-circular. Report:
> `leanncd/docs/superpowers/plans/2026-07-03-s1-jointly-monic-spike.md`. **Both Track-B gates are now
> cleared.**

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

**Multi-output shape-match sorries — RESOLVED (2026-07-03), moot for Track A.** Per the original
audit (finding 5), `buildStep_output_fixedAxes`, `wf_typeMatch`, and `internal_pointwise` were
candidates for Naperian re-typing (they are "two weaves have the same `fixedAxesP` ⇒ same
`ArrayType`"). They are now proved directly, with `List.getD … default` positional bookkeeping
intact — no re-typing was needed. Spike S2 (weave re-typing) is therefore optional, not a pending
decision blocking anything.

### 4.2 What does NOT get solved (and what Naperian is orthogonal to)

Substantial and **gating Track B** (audit findings 1–2):

- **`act` on `Br` morphisms is unbuilt (a `sorry`)** — it must be defined (via `Quotient.lift`
  respecting all `Rel` constructors + functor laws) before *any* coherence-iso benefit exists. This
  is the largest single piece and is a prerequisite, not a payoff. Scope it with **spike S0**.
- `broadcast_gen`, `weave_unique`, and the coherence fields `δ`/`δ0`/`υ`/`α`/`act_unit_assoc`/
  `υ_nat`/`dist_coh` — all still `sorry`, all downstream of `act`.

Naperian gives a better *proof shape* for these, but does not discharge them, and cannot even be
stated usefully until `act` exists.

Explicitly **orthogonal** to Naperian typing (audit findings 3–4, and historically 5) —
**confirmed, not just predicted:**

- The multi-output blockers `topo_bound` (was a false-as-stated modeling gap: cycles + scan
  self-reads) and `stepPiece`/`finalPiece` pool reconciliation — `Wire`-list membership/order, not
  shape types. **RESOLVED 2026-07-03** by restating `topo_bound` over `routeCore` success, exactly
  as predicted: better shape types would not have helped, and didn't need to.
- `brCancelPoint`/`weave_unique` — isolated, off the executable path, no consumers. Unchanged.
- The `actV` (`FGModuleCat ℝ`) impossibility — an eval-side semantic-algebra redesign (spike S3).
  Unchanged.

### 4.3 Why the solver split matters for proofs

If concrete finite-point data were forced too early, proofs would become entangled with the
runtime size environment. Keeping symbolic typing before the solver avoids this:

- symbolic proofs talk about `IdxExpr`, `St`, `degree`, and coherence,
- runtime proofs talk about the instantiated `UID -> Nat` environment and concrete point
  enumeration.

That separation is cleaner both logically and implementation-wise.

## 5. Milestone sequencing

**Top-level order (revised 2026-07-03 — step 1 is now DONE):**

1. ~~Finish the multi-output `compile_wellFormed` effort first, to completion.~~ **DONE.** The
   multi-output effort and the §8.2 acset agreement (Tasks A–E) are both complete and sorry-free;
   see the 2026-07-03 audit update in §0 above. `topo_bound` was fixed by restating it over
   `routeCore` success, not by re-typing weaves, so spike S2's original "prove-twice" justification
   for this surface no longer applies.
2. **Track A (M0 → M2 → M3 → M4) — start now.** The symbolic typed-reindexing layer. Deliverable
   and valuable regardless of Track B. S2 (M4.5) is now optional/speculative (see the finding-5
   update in §0) rather than a decision this surface is waiting on.
3. **Spikes S0 (M0.5) and S1 (M1.5)** — gate Track B. **Both DONE (2026-07-03).** S0: the `Rel`-lift
   is tractable (~400–650 lines) and its object-level `sh_act` blocker is **resolved** (`=` → `≅` in
   `Core/Graded.lean`, build green). S1: **POSITIVE** — `jointly_monic` has a Lean-verified acyclic
   route (via representability / `IsLimit.hom_ext`, no `broadcast_gen`); the residual reroutes to
   St-slice separation ⇒ the proved `St.elemental`, needing concrete `act` to close.
4. **Track B (M6 coherence work)** — both feasibility gates cleared. The next concrete step is a
   **commitment-sized build, not a further probe** (the residual probe confirmed there is no cheaper
   de-risking left): build `act.obj`/`act.map`/`ev_p` for `Br` *and* the concrete finite-`El`
   representability (`tabulate`/`lookup`), which together close the S1 residual (`jointly_monic`) and
   underpin the coherence tower. It is a funded cost/benefit call, not blocked.

Milestones are tagged **[A]** / **[B]** / **[spike]** below.

### Milestone 0 — Baseline confirmation **[A]**

- confirm current compile/eval behavior,
- keep `inferAxisSizes` exactly where it is,
- add tests documenting the symbolic-vs-concrete split.

### Milestone 0.5 — `act`-definability spike (S0, GATE for Track B) **[spike — ✅ DONE 2026-07-03]**

**Full report:** [`leanncd/docs/superpowers/plans/2026-07-03-s0-act-definability-spike.md`](../leanncd/docs/superpowers/plans/2026-07-03-s0-act-definability-spike.md).

**Verdict: SPLIT.**

- **Morphism level (the literal question) — TRACTABLE.** `act.map` is a well-posed `Quotient.lift`
  factored as `act.map (f,η) = [f,P] ; [Y,η]`; well-definedness reduces to `Rel f g → liftAt P f =
  liftAt P g` on the batch lift. Of the 21 `Rel` constructors, **20 map to already-proved `BrMorph`
  theorems** (`id_comp`/`assoc`/`tensor_comp`/`braid_braid`/`braid_natural`/the `*_heq` cast lemmas/
  the `copyW_*` comonoid laws). The `gen` leaf is a bounded `BrBase` construction, not a `Rel`
  obligation. Confirmed technical tax: `List.map` is **not** defeq-distributive over `++` (checked
  via `lean_run_code`), so every `tensor`/`braid` case carries a `List.map_append` cast — friction,
  not a wall (singleton `copyW`/`delW` objects stay defeq-clean). **Estimate: ~400–650 lines for
  `act` alone**, excluding `δ`/`δ0`/`υ`/`α`/`broadcast_gen`.
- **Object level — was a NAMED BLOCKER, now RESOLVED (option 1).** Strict `sh_act`
  (`sh*(X ⊛ P) = sh*(X) ⊗ P`, a `=`) was **unsatisfiable for ≥2-array bundles** under per-array
  broadcast: the concatenated `sh_star` carries `|X|` copies of `P`, not one, and a strict `=`
  admits no iso freedom. **Fixed by relaxing `sh_act` to a canonical braiding iso `≅`** in
  `Core/Graded.lean` (keeps the per-array `List.map` action, which makes `δ`/`δ0` near-definitional).
  Build green; blast radius was zero proof consumers, so it was a one-field change.

**Consequence for the gate:** the `Rel` proof was never what made `act` expensive or risky — the
`sh_act` design question was, and it is now decided and implemented. `act` is a bounded, mechanical
build; Track B is fundable pending S1. `brCancelPoint` / the `Br` free-SMC quotient are confirmed
off this path.

### Milestone 1 — Core symbolic API **[A/B boundary]**

- add `Core/Naperian.lean`,
- state the mixin classes and key laws (statements only; no `act` dependence yet),
- no runtime point enumeration yet.

### Milestone 1.5 — circularity spike (S1, GATE for Track B) **[spike — ✅ DONE 2026-07-03, POSITIVE]**

**Full report:** [`leanncd/docs/superpowers/plans/2026-07-03-s1-jointly-monic-spike.md`](../leanncd/docs/superpowers/plans/2026-07-03-s1-jointly-monic-spike.md).

**Verdict: acyclic route established, `broadcast_gen` not needed.** The Lean-verified reduction (in
the real project, axioms `[propext, Classical.choice, Quot.sound]`) is:

```lean
theorem naperian_jointly_monic_of_repr (X : C) (P : Dᵒᵖ)
    (hrepr : Limits.IsLimit (evFan X P))          -- lifted object is the El P-power, legs = ev_p
    {W} (f g : W ⟶ act.obj (X, P))
    (h : ∀ p : ElP P, f ≫ ev_p p X = g ≫ ev_p p X) : f = g := by
  apply hrepr.hom_ext; rintro ⟨p⟩; exact h p
```

**Honest nuance:** `IsLimit.hom_ext` gives `jointly_monic` *from* the power structure, but
constructing `IsLimit (evFan)` needs its `uniq` field, which *is* `jointly_monic`. So the reduction
**reroutes** the obligation rather than eliminating it — the point is *where*: to representability /
**St-slice separation** (which reduces to the proved `St.elemental`, `Base/St.lean:364`), and
verifiably **away from `broadcast_gen`**. That is precisely the cycle S1 tests for, and it is avoided:
`jointly_monic ← representability`, never `← broadcast_gen`.

**Residual (a commitment-sized build, NOT a spike — refined by the 2026-07-03 residual probe).** A
follow-on probe checked whether the residual could be de-risked cheaply. It cannot: (i) a faithful
`evSlice` morphism is the reindex-half of `act` (needs the `degree`/weave/`reindexings` apparatus),
so even *stating* the concrete lemma requires part of the ~400–650-line `act` build; (ii) `El P` is
not finitely enumerable symbolically (`Fintype (BrMorph [] P)` has no instance), so the clean product
route (`X ⊛ P ≅ ∏_{El P} X`) is a **concrete / post-solver** fact needing the `tabulate`/`lookup`
representability construction; (iii) the `St.elemental` reduction is clean only for single-`BrBase`
morphisms — a general `BrMorph` composite inducts into the same `trans`-case wall as `brCancelPoint`,
and the escape is precisely the representability route. **Net:** the residual stays non-circular (no
`broadcast_gen`, and off `brCancelPoint` via representability), but it is real work sized with the
`act` + concrete finite-`El` build — there is no cheaper probe that retires it. The projected savings
are **not** blocked by a hidden cycle; they are gated on committing to that build.

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

### Milestone 4.5 — dependent-weave re-typing spike (S2) **[spike, now OPTIONAL — 2026-07-03]**

**Status update:** the lemmas this spike targeted (`buildStep_output_fixedAxes`, `wf_typeMatch`,
`internal_pointwise`) are already proved by hand — see the finding-5 update in §0. This spike no
longer unblocks anything; it would only be worth running if a cleaner weave representation is
wanted for its own sake going forward.

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

### 6.8 "Prove-twice" risk on multi-output shape lemmas — RESOLVED, risk did not materialize

Multi-output finished (2026-07-03): `wf_typeMatch`/`buildStep_output_fixedAxes` are proved by hand,
`List`-based. Nothing has re-typed the weaves, so there was no rework to redo. If S2 is ever run
for other reasons and succeeds, revisit whether re-proving these is worth it then — but it is not
a live risk today.

## 7. Summary

After the 2026-07-03 re-audit (§0), the plan is:

- **Multi-output `compile_wellFormed` is DONE** (as is the follow-on §8.2 acset agreement). It
  turned out orthogonal to Naperian, as predicted; Naperian offered it nothing in the end — the
  shape-match lemmas were closed by hand, and spike S2 is now optional rather than load-bearing.
- **Ship Track A now** — the symbolic typed-reindexing layer — as the real, ungated deliverable:
  - do symbolic Naperian typing before the affine solver,
  - keep the affine solver as the runtime/value-level source of concrete extents,
  - instantiate concrete finite Naperian point data only after solving extents.
  This hardens the compiler (typed reindex rank/codomain, canonical degree, early law checks)
  without breaking the padded-semantics evaluation model.
- **Track B gates are now CLEARED (2026-07-03).** Both spikes have run: **S0** — `act` is a tractable
  ~400–650-line build (20/21 `Rel` cases reuse proved `BrMorph` laws) and its `sh_act` blocker is
  resolved (`=` → `≅`); **S1 — POSITIVE** — `jointly_monic` has a Lean-verified acyclic route (via
  representability / `IsLimit.hom_ext`, no `broadcast_gen`; residual reroutes to the proved
  `St.elemental`). The projected coherence reduction is no longer blocked — but still **unmeasured**
  until `act` + the coherence proofs are built, so don't quote the percentage as a delivered result.
  The 2026-07-03 residual probe confirmed there is **no cheaper de-risking left**: closing the S1
  residual (`jointly_monic`) needs the reindex-half of `act` + the concrete finite-`El`
  representability (`El P` isn't enumerable symbolically), i.e. a **commitment-sized build, not a
  further spike**. Next concrete step for Track B is therefore that build itself.

In one line: **Multi-output is done and orthogonal to Naperian; Track A can start today; Track B's
feasibility gates (S0, S1) have both passed — it is now a funded cost/benefit decision, not a bet.**

## 8. Track A — final status (2026-07-03)

Track A landed on `main` (merge `b0bc15a` + follow-ups). Routing-layer invariants proved (all
additive, axiom-clean, `route`/`compile_wellFormed` untouched): `StMatP.wellFormed`/`validate` +
`routeCore_reindexings_wellFormed`; `routeCore_reindexings_domLen` (domain rank = degree);
`stepDegAxesMulti_uid_nodup` (canonical degree); `buildStep_inputWeaves_allFixed` (input weaves
full-rank); M2 `elaborateAffineReindexings` artifact + faithful bridges + one affine primitive (§6.2
dedup); M4 route consumes the artifact + typed `StMatP'` (structural well-formedness); M3
`buildStep_output_reducesOnlyContracted`; #1 `buildStep_reindexings_codLen_eq_inputRank` (codomain
rank = input weave rank, conditional on `readArityOk`).

### 8.1 Remaining items — definitive resolution (NOT soundly closeable as routing-layer theorems)

Investigated against the actual pipeline; each is **upstream or structural**, and forcing a
routing-layer proof would be unsound or vacuous. Their honest status:

- **`readArityOk` from `checkReadRanks`/`env` — NOT PROVABLE (design gap, cf. `topo_bound`).** Two
  independent reasons in-code: (i) `resolveDecls` puts only *declared* tensors in `env`, so a
  produced-but-undeclared **intermediate** is neither in `env` nor external and `checkReadRanks`
  never checks its read arity — arity=producer-rank is simply unenforced; (ii) even for declared
  producers, `checkReadRanks` pins arity to `decl.axisCount`, but the producer publishes
  `dedupByUid(LHS)` — a **repeated-LHS** program (`Y[i,i]`: decl 2, published 1) makes `readArityOk`
  *false*. So #1's internal-read case correctly stays a hypothesis (`readArityOk`); closing it needs
  a frontend/design decision (reject `Y[i,i]`-style reads; require intermediates declared), not a
  proof.
- **pointwise-preserves-degree — UPSTREAM (`splitNonlins`), conditional only.** A pointwise op's
  step has `contracted = ∅` (reads = LHS axes) *by `splitNonlins` construction*, and only for
  well-formed programs. Routing has no witness of this; it could at best be a conditional theorem
  whose entire content is the hypothesis "this step's reads are its LHS axes" — thin, and genuinely
  a `splitNonlins`-level fact, not a routing one.
- **scan-separation — STRUCTURAL, not a crisp theorem.** "Scan iteration arithmetic stays separate
  from static reindexing" is an architectural property of how `finalizeScans`/the scan generator
  handle the iteration axis (it is not a reindexing), not a stateable routing invariant.

**Conclusion:** the routing-layer content of Track A is complete and proved. The three residual
items are upstream/structural and are *closed* here as documented design assumptions/gaps — not as
theorems, because sound theorems do not exist for them at this layer. Any further progress on them
is frontend/`splitNonlins`/`checkReadRanks`-level work with its own design decisions.

# leanncd restructuring suggestions

A global review of `leanncd/` (~9,200 lines of Lean across 49 modules) after the recent wave of
feature fixes (`Factor.unaryFn` inline transcendentals, new `Nonlin` activations, `l2normalize`,
the friendly `/` operator, multi-axis scans, scan aggregators, the multi-output `schedule` fix,
and the scan-projection guard). Each fix was locally surgical; collectively they exposed
structural duplication that is now worth paying down. This document catalogs concrete
restructuring opportunities, organized into independently plannable spikes, ordered so that
earlier spikes reduce the cost and risk of later ones. Part I (Spikes 1–8) stays close to the
existing architecture; Part II (end of document) collects exploratory, higher-risk redesign
directions that deliberately go beyond it.

**Method.** Three parallel deep-reads (DSL pipeline, Eval interpreter, proof track + bridge),
each claim cross-checked against the code; the highest-leverage claims (dead code, exact
duplicates, hardcoded semiring constants) were re-verified independently. Everything below
carries `file:line` references against HEAD as of 2026-07-09.

## Table of Contents

- [0. Constraints — what must NOT be restructured](#0-constraints--what-must-not-be-restructured)
- [Wave 1 — mechanical deletions and corrections (near-zero risk)](#wave-1--mechanical-deletions-and-corrections-near-zero-risk)
  - [Spike 1: dead code, stale docs, housekeeping](#spike-1-dead-code-stale-docs-housekeeping)
- [Wave 2 — structural unification of the executable tracks](#wave-2--structural-unification-of-the-executable-tracks)
  - [Spike 2: one home for AST accessors and read/axis traversals](#spike-2-one-home-for-ast-accessors-and-readaxis-traversals)
  - [Spike 3: make the Nonlin wildcard hazards unrepresentable](#spike-3-make-the-nonlin-wildcard-hazards-unrepresentable)
  - [Spike 4: Eval interpreter unification](#spike-4-eval-interpreter-unification)
  - [Spike 5 (cross-cutting, after 2–4): decompose `finalizeScans`](#spike-5-cross-cutting-after-24-decompose-finalizescans)
- [Wave 3 — proof track: consolidate lemma bases, prune the sorry surface, simplify types](#wave-3--proof-track-consolidate-lemma-bases-prune-the-sorry-surface-simplify-types)
  - [Spike 6: proof-infrastructure consolidation (RouteSpec + AcsetCodec, ~600 lines)](#spike-6-proof-infrastructure-consolidation-routespec--acsetcodec-600-lines)
  - [Spike 7: prune the sorry surface to the honest open core (38 → ~14)](#spike-7-prune-the-sorry-surface-to-the-honest-open-core-38--14)
  - [Spike 8: proof-adjacent type cleanups (the noncomputable/computable seam)](#spike-8-proof-adjacent-type-cleanups-the-noncomputablecomputable-seam)
- [Suggested spike ordering and dependencies](#suggested-spike-ordering-and-dependencies)
- [Part II — Exploratory redesigns (creative, higher-risk, prototype-first)](#part-ii--exploratory-redesigns-creative-higher-risk-prototype-first)
  - [How the explorations relate to the Part I spikes](#how-the-explorations-relate-to-the-part-i-spikes)
  - [E1. One traversal to rule the collectors (van Laarhoven)](#e1-one-traversal-to-rule-the-collectors-van-laarhoven)
  - [E2. A typed core IR the pipeline narrows into ("trees that grow")](#e2-a-typed-core-ir-the-pipeline-narrows-into-trees-that-grow)
  - [E3. Evaluate over an arbitrary semiring](#e3-evaluate-over-an-arbitrary-semiring)
  - [E4. EvalPlan — make the evaluator run the same affine maps the route proves things about](#e4-evalplan--make-the-evaluator-run-the-same-affine-maps-the-route-proves-things-about)
  - [E5. An executable denotation for the routed artifact](#e5-an-executable-denotation-for-the-routed-artifact)
  - [E6. Property-based oracles (the tests the portfolio can't express)](#e6-property-based-oracles-the-tests-the-portfolio-cant-express)
  - [E7. Accumulate compile diagnostics (the Validation applicative)](#e7-accumulate-compile-diagnostics-the-validation-applicative)
  - [E8. Open registration of unary functions](#e8-open-registration-of-unary-functions)
  - [E9. Datatype-generic acset codecs](#e9-datatype-generic-acset-codecs)
  - [E10. Stage the interpreter into a code generator](#e10-stage-the-interpreter-into-a-code-generator)
  - [E11. Investigation: do UIDs earn their keep?](#e11-investigation-do-uids-earn-their-keep)
  - [E12. Named simp sets for the proof domains (tiny)](#e12-named-simp-sets-for-the-proof-domains-tiny)
  - [How the exploratory ideas compose](#how-the-exploratory-ideas-compose)

---

**Headline numbers if all recommended spikes land:**

| Metric | Now | After |
|--------|-----|-------|
| Net lines deleted | — | ≈1,100–1,300 |
| `sorry`s in the default build | 38 | ≈14 (the honest open surface) |
| Sites to touch when adding one new nonlin | 9 | ≈5 |
| Hand-maintained "MUST match X" mirror pairs | 3 | 0 |
| Byte-identical duplicate functions | 6+ | 0 |

---

## 0. Constraints — what must NOT be restructured

These boundaries came out of the review as *deliberate architecture*, and several look
superficially like duplication. Any implementation plan should treat them as fixed:

1. **The three parallel type families are not redundant.** Math tower (`Base/`,
   noncomputable, dependent), presentation (`DSL/Target.lean` `*P` types, computable,
   `ToExpr`), and acset (`Acset/`, flat relational rows byte-faithful to Python) each exist
   for a distinct reason, documented at `LeanNCD.lean:58-74`. Do not merge them or try to
   derive one from another. The conversions (`realize*`, `fromThreadedComposed`, CSV codec)
   are genuine boundary crossings.
2. **`SBrInstance` stays a list-of-rows.** The relational shape mirrors Python's
   `acset/instances.py` byte-for-byte; replacing rows with maps would break interop and
   invalidate the round-trip proofs' statements.
3. **RouteSpec unfolds `Lowering` definitionally.** `RouteSpec.lean`'s ~30 theorems unfold
   `buildStep`, `stepDegAxesMulti`, `dedupByUid`, `inputReadFactors`, `routeCore` by
   definition. Any change to those bodies — even semantically neutral — costs proof repair.
   This is the binding sequencing constraint: Spike 6 (which makes RouteSpec robust to such
   changes) should precede or accompany anything that touches `buildStep`. Conversely,
   `liveFix`, `topoSort`, and `stepMkWeave` are *outside* the proved surface (RouteSpec
   starts from `ScheduledProgram`) and are free to change.
4. **`brOpIdx`/`brOpOfIdx` stay an explicit table.** The index is a *serialization format*
   (acset labels via `natToUnary`). Deriving it from constructor order (`BrOp.toCtorIdx`)
   would make a constructor reorder silently change the wire format. Keep the table;
   optionally add a `#guard` totality check (`(List.range 15).map (brOpIdx ∘ brOpOfIdx) ==
   List.range 15`).
5. **Two guards that share an error constructor are different checks.** `finalizeScans`'
   `inconsistentScanAxes` (`Structural.lean:489-492`, axis *sets*) and `buildStep`'s
   `outputAxesConsistent` (`Lowering.lean:476-486`, axis *order*) must not be merged; give
   them distinct messages or a cross-reference comment.
6. **`cartesianList` (`Eval/Scan.lean:20-22`) is not a duplicate of
   `DenseTensor.allCoords`** — its iteration order (last axis slowest) is load-bearing for
   scan causality. Add a cross-reference comment so nobody "deduplicates" it.

---

## Wave 1 — mechanical deletions and corrections (near-zero risk)

### Spike 1: dead code, stale docs, housekeeping

> **✅ DONE — merged to `main` 2026-07-10** (commits `1cf92d3..3bb7e13`; full `lake build` green,
> 8,604 jobs; whole-branch review clean). All eight sub-items **1a–1h** landed, including the
> optional behavior-changing **1h** (fail-loud `cyclicDataflow` reject, added test-first).
> Reference drift found and corrected during execution: the `_scratch_nf` doc citation was at
> `BrNF.lean:34` (not `Br.lean:288`); the `Shape.lean` "Task 6" placeholder was at `:515`; and
> `St.lean` was indeed **not** sorry-free (`swap_hexagon_fwd/rev`, `St.lean:267-268`), so the
> `SORRY_INVENTORY.md` claim was corrected — the full sorry recount stays deferred to Spike 7.
> The bullets below are retained as the record of what was changed.

All verified dead by grep and/or reasoning from the current code; each bullet is
independently landable.

**1a. `liveFix`/`liveStep` are provably no-ops — delete.**
`schedule` (`DSL/Pipeline/Lowering.lean:155-167`) now seeds liveness with *every* produced
name (`produced.eraseDups`, the KG-multiout fix). Every `ScanStmt` has nonempty `writes`, and
each stmt's writes are contained in `produced ⊆ live`, so the filter
`stmts.filter ((·.writes).any (live.contains ·))` keeps every statement — it is the identity.
Delete `liveStep` (:137-139), `liveFix` (:141-145 — also removes a `partial def`), and the
seeding lines (:156-158). `schedule` reduces to `topoSort` + `explicitSizes` + `liveExtNames`
(the last still does real work: it drops never-read external names via `orderedReads`).
Rewrite the "Dead-code elimination by backward reachability" doc block (:65-69) to say
"ordering only; no statement pruning (KG-multiout)". Pinned by
`test/DSL/Pipeline/LoweringTest.lean:37-48` ("No DCE for unread top-level statements"),
which passes identically.

**1b. Dead trio `stepContracted`/`stepDegAxes`/`stepMkWeave`
(`Lowering.lean:292-303`) — delete.** `stepMkWeave` has zero call sites in `LeanNCD/` and
`test/`; the other two are called only by it. All three were superseded by the multi-stmt
`ScanStmt.stepDegAxesMulti`/`slotWeave` (:398-467). One stale doc-comment mention at :400.

**1c. Dead surface syntax `tl_shape`/`elabTLShape`** (`Syntax.lean:27,62-63`,
`Elab.lean:42-44`) — the comment itself says "kept for future use; no longer referenced by
any `tl_decl` rule". Delete, or keep with a conscious decision recorded.

**1d. Speculative always-empty fields `extraStmts`/`auxStmts`**
(`Pipeline/Types.lean:30,45`; constructed as `#[]` at `Structural.lean:137,327`). They encode
"§12.4 said there would be intermediates and there aren't". Deleting them simplifies two
records but touches ~15 test-record literals (`StructuralTest.lean`, `MaxReduceTest.lean`) —
mechanical, compiler-guided.

**1e. Root scratch files.** Eight untracked `_scratch_*.lean` at the repo root are
unreferenced by `lakefile.toml` and imported by nothing. Move to a `spikes/` directory (or
delete). One dependency first: `Br.lean:288`'s `brCancelPoint` doc cites "the old
`_scratch_nf.lean`" as the assemble-step reference — inline that skeleton into `BrNF`'s
header (or the doc) before removing the file.

**1f. Merge the empty mixin stubs.** `Mixins/Route.lean` (11 lines) and
`Mixins/Symmetry.lean` (13 lines) are one-class stubs; merge into a single
`Mixins/Stubs.lean` (a `Mixins.StubsTest` already exists in the test globs, suggesting this
was the original intent). Optional, same spirit: fold `Bridge/SBr.lean` (22 lines, one def)
into `Agreement.lean`.

**1g. Stale documentation that now misleads** (each verified):

- `Bridge/Agreement.lean:316-338` — a long "FALSE AS STATED" warning block describing the
  *old* `topo_bound` statement; the theorem at :339 is now proved (the Phase-A
  `routableInOrder` guard discharged it). Also :323 claims "the pipeline never rejects cyclic
  dataflow" — `routeCore` now rejects cycles with `cyclicDataflow`. Trim to a short note.
- `Props/Generic.lean:52-53` — doc says "SIGNATURE — body is `sorry`" but
  `scan_catamorphism` is proved.
- `SORRY_INVENTORY.md` (last updated 2026-07-03) — claims `St.lean` is fully sorry-free
  (false: `swap_hexagon_fwd/rev` at `St.lean:267-268` were added as fields afterward), and
  has no entries for `BrNF.lean`'s 6 sorries. Refresh against the census in Spike 7.
- `Structural.lean:147` references `readsOf` (an Eval-layer name) from the DSL layer.
- `Eval/Shape.lean:518` — "affine slots handled in Task 6 — 0 placeholder" is stale.

**1h. Optional fail-loud quick win (behavior-changing, flag separately):** on the eval-only
path (`compileToScheduled`, no `route`), a cyclic program sails through `schedule` (the
`topoSortFuel` cycle fallback silently returns source order, `Lowering.lean:126`) and dies
later with a generic "unknown tensor". `topoSort` already detects the cycle; make `schedule`
report `cyclicDataflow` instead of falling back. Add a reject test.

---

## Wave 2 — structural unification of the executable tracks

### Spike 2: one home for AST accessors and read/axis traversals

The single largest duplication cluster. Root cause: `Eval/` imports only `DSL/Ast.lean`
(deliberate layering), so pure-AST helpers that live in `Pipeline/Structural.lean` were
re-declared in Eval — and then again inside function bodies. Fix the placement, not the
layering: move the primitives *down* into `Ast.lean` (or a new `DSL/Collect.lean`), where
both layers can import them.

**2a. The read-projection family — 6 functions, one primitive.** Each pattern-matches
`Factor` with identical arms (`.read`/`.unaryFn` → keep name+exprs, `.iverson` → drop) and
differs only in projection:

| Function | Location | Projection |
|----------|----------|------------|
| `Stmt.readNames` | `Structural.lean:116` | name |
| `stmtReads` | `Structural.lean:154` | (name, arity) |
| `Stmt.rhsReads` | `Structural.lean:361` | index exprs |
| `Stmt.readFactors` | `Lowering.lean:197` | (name, exprs) |
| `Eval.Stmt.readsOf` | `Eval/Shape.lean:15` | verbatim duplicate of the row above |
| `readNames` (RHS-level) | `Eval/Contract.lean:47` | name |

Add to `Ast.lean`: `Factor.read? : Factor → Option (String × List IdxExpr)` (the ONE place a
new `Factor` constructor gets classified), `RHSExpr.readFactors`, `Stmt.readFactors`. The six
become one-line projections; `Eval.Stmt.readsOf` is deleted outright. Ordering is identical
in all six (terms in order, factors filterMapped in order) — zero behavior change.

**2b. The axis-collector family — two parallel towers.** The private `specs*` family
(`Structural.lean:26-60`, AxisSpec-valued) and the `*AxisUIDs` family
(`Eval/Contract.lean:7-44`, UID-valued) are the same structural walks; each UID function is
its Structural twin post-composed with `(·.uid)`. Additionally `idxAxes`
(`Lowering.lean:228`) is a public verbatim copy of the private `specsIdx`
(`Structural.lean:26`). Move one AxisSpec-valued family into the AST layer (`IdxExpr.axes`,
`BoolExpr.axes`, `Factor.axes`, `LHSSlot.axes`) and derive the UID versions by `.map (·.uid)`.
**Preserve the one semantic distinction explicitly:** `specsRHS` includes the nonlin mask's
axes; `readAxisUIDs` deliberately does not (per-term contraction scoping must not see mask
axes). Two named entry points — e.g. `RHSExpr.axes (includeNonlinMask := true/false)` or
`axesWithMask` vs `readAxes` — not a silent flag default. A `CollectAxes` typeclass would be
overkill; plain shared functions suffice.

**2c. Accessor twins created by the layering split** — move to `Ast.lean` and delete the
duplicates (all verified byte-identical or projection-trivial):

| Keep (new home: Ast.lean) | Delete |
|---------------------------|--------|
| `Stmt.lhsName` | `stmtName` (`Eval/Scan.lean:67`) |
| `Stmt.slots` | `Eval.Stmt.lhsSlots` (`Eval/Shape.lean:9`), `stmtSlots` (`Eval/Scan.lean:73`) |
| `Stmt.nonlinOf` | `Stmt.nonlin` (`Lowering.lean:218`) |
| `LHSSlot.axisUID?` | `lhsAxisUID?` (`Eval/Shape.lean:501`), the `slotUID` closure (`Structural.lean:451-453`), inline matches at `Structural.lean:180-183` and `Lowering.lean:207` |
| `LHSSlot.outIdx` | `lhsSlotIdx` (`Eval/Scatter.lean:6`), `slotOutIdx` (`Eval/Shape.lean:375` — its own comment apologizes for the import-cycle copy) |
| `Decl.name` | `declName` (`Eval/Contract.lean:129`) |
| `outputShape` (one copy) | `stateShape` (`Eval/Scan.lean:81` — byte-identical body) |

**2d. The most dangerous mirror: scatter extent sizing.** `scatterOutShape`
(`Eval/Eval.lean:12-27`, `Except`, fail-loud) and `scatterOutDim` (`Eval/Shape.lean:386-395`,
`Option`) implement the same 5-case per-slot extent formula, with a comment begging "MUST
match … kept in sync by mirroring". The extent convention (`.const n ⇒ n+1`, `.scale c a ⇒
c·size`) is deliberately not derivable from `idxAffineForm` (upsample stride semantics), so
it deserves exactly one home: `LHSSlot.outExtent (sz : UID → Option Nat) : Option Nat`, with
both callers derived (`scatterOutShape` = `mapM` + `getDM (throw …)`). Deletes the mirror and
its sync obligation (~25 lines).

**2e. `Stmt.iterInfo`'s anonymous 4-tuple** (`Structural.lean:354`,
`List (UID × AxisSpec × Bool × Nat)`) forces `t.2.2.1`-style accessors at 6+ sites and its
first component is redundant (`= axis.uid`). Replace with
`structure IterSlot where axis : AxisSpec; isRecur : Bool; pos : Nat`; `Eval/Scan.lean:10`'s
`iterSlotPositions` becomes a projection of it.

*Estimated effect: ~10 duplicate functions deleted (~150–200 lines), the "9-site mirroring"
pattern that every recent feature had to hand-replicate drops to one site
(`Factor.read?`/`Factor.axes`). Pinned by StructuralTest, LoweringTest,
AffineShapeSolverTest, ScatterTest, ShapeTest, ScanTest, EvalExamplesTest. Do this spike
first in Wave 2 — Spikes 3 and 4 both shrink substantially once it lands.*

### Spike 3: make the Nonlin wildcard hazards unrepresentable

Two real bugs this cycle came from the same type-design flaw: `Nonlin`'s 9 flat constructors
force every match site to re-enumerate the pointwise-vs-rowwise split, and wildcard fallbacks
silently misclassify new variants (`specsNonlin` swallowed `l2normalize`'s mask;
`evalPlain` initially demanded a marked axis for `sigmoid` — both documented in-code with
warning comments that this spike deletes).

**3a. Restructure the AST:**

```lean
inductive PointwiseFn | relu | sigmoid | tanh | gelu | leakyrelu
inductive RowwiseFn   | softmax | normalize | l2normalize
inductive Nonlin
  | identity
  | pointwise : PointwiseFn → Nonlin
  | rowwise   : RowwiseFn → Option BoolExpr → Nonlin
```

A new pointwise fn *cannot* forget mask handling (no mask exists, by type); a new rowwise fn
automatically flows through mask collection, norm-axis lookup, and nonlin-split at every site
via the existing `.rowwise` arms. Eleven match sites collapse (survey verified —
`Traverse.lean:36-45` 9→3 arms; `specsNonlin` `Structural.lean:38-40` becomes
`| .rowwise _ (some m) => specsBool m | _ => []` with a *provably safe* wildcard, deleting
the 8-line hazard warning in the module docstring; `buildStep`'s op mapping
`Lowering.lean:537-546` 9→3 arms + two tiny `toBrOp` tables; `applyNonlin`
`Eval/Nonlin.lean:97-108` 9→3; the `evalPlain`/`evalStmtSliceSeeded` axis-resolution matches
become exhaustive 3-arm matches with no wildcard, deleting the second hazard comment).
Equality-only sites (`== .identity` at `Structural.lean:247,516`, `Lowering.lean:34`) are
unchanged. `BrOp` stays flat (Python bridge); only the mapping changes. Surface syntax
unchanged. **Budget RouteSpec repair** for the one `buildStep` arm this touches (limited
exposure; see Constraint 3 — cheaper if Spike 6a has landed first). Pinned by NonlinTest,
FF5–FF8 (unmarked activations), LoweringTest's masked-softmax split, EvalExamplesTest's
masked attention.

**3b. Elab/Syntax keyword tables.** Two mechanical generator patterns replace 16 copy-pasted
productions/arms:

- `tl_unary_kw` category: 5 one-token rules + one `tl_factor` production + one elab arm with
  a `String → UnaryOp` table (replaces `Syntax.lean:126-130` + `Elab.lean:164-173`). Adding
  a unary fn becomes a 2-line change.
- `tl_nonlin_kw` category: 8 keyword tokens + one
  `tl_nonlin_kw (atomic("(" "where") tl_bool_expr ")")?` production + one elab arm. Bonus: a
  mask on a pointwise keyword becomes a clear *elaboration* error ("relu takes no
  where-mask") instead of today's inscrutable parse failure. **Verify** the
  `atomic("(" "where")` lookahead still disambiguates `softmax(sum)` (documented at
  `Syntax.lean:143-145`) — run `ParseLayer34Test` and `CompileExamplesTest` before trusting
  it.
- Cleanup: `Elab.lean` repeats `x.getId.eraseMacroScopes.getString!` 19 times — extract
  `private def identStr`.

**3c. Optional follow-up: merge `Factor.unaryFn` into `Factor.read`.** Exactly one consumer
inspects the op (`gather`, `Eval/Gather.lean:75-78`); nine sites carry a `.unaryFn` arm
identical to their `.read` arm. `| read : (fn : Option UnaryOp) → String → List IdxExpr →
Factor` makes "a unaryFn *is* a read" true in the AST. **Do this only after Spike 2a** — once
`Factor.read?` exists the nine duplicated arms are already gone and this shrinks to a small
honesty refactor (Traverse/Gather/Elab); evaluate then whether it still pays.

### Spike 4: Eval interpreter unification

**4a. `evalAssignWith` = `evalAssignSeeded` with an empty seed.**
(`Eval/Contract.lean:73-103` vs `:162-195`) — ~80% token-identical; every delta (frees
filter, baseCoord init, termContr filter, outShape) reduces to the unseeded form when
`seed = {}`. Delete `evalAssignWith`; keep one function. **Two real payoffs beyond line
count:** (i) the unseeded version has a fail-loud unsized-free-axis check (:82-84) that the
seeded version *lacks* — today a scan slice with an unsized free axis silently yields an
empty tensor via `getD 0`; unification ports the check to the scan path (a bug-shaped hole
closed). (ii) Pass `Combine` as a struct instead of three loose `(mul, combine, unit0)`
params (the struct already exists at :109-112, defined *below* its would-be first use — move
it up). Pinned by ContractTest, ScanTest, MaxReduceTest, EvalExamplesTest.

**4b. Add the ⊗-unit to `Combine` now.** `prod := 1.0` — the within-term combine's unit — is
hardcoded at `Contract.lean:95,187` and `Scatter.lean:42`. The planned general-semiring
notation (`agg(min, +)`, portfolio doc §19.2) needs `unit1 = 0.0` for an additive combine, so
today's code would be *wrong*, not just inelegant. Adding `unit1 : Float := 1.0` to `Combine`
and threading it to those three sites is a 10-line change that unblocks the semiring work and
documents the invariant.

**4c. Unify the contract-then-nonlin dispatch.** `evalPlain`'s assign arm
(`Eval/Eval.lean:33-51`) and `evalStmtSliceSeeded` (`Eval/Scan.lean:37-53`) duplicate the
Combine selection and the 7-line axisPos-resolution match. **Semantic divergence found:** the
scan path inlines only the agg match, skipping `combineFor`'s predicate→`Combine.bool` branch
(`Contract.lean:137-143`) — a predicate state inside a scan would contract with `Combine.real`,
unlike the identical statement outside a scan. Unify into one
`evalAssignStmt (decls … seed) (s : Stmt)` (thread `decls` through `evalScan` — one new
parameter), with `evalPlain` passing `seed = {}`. Fixes the divergence and deletes ~17 lines.
Error-message prefixes change; no test pins them.

**4d. Move nonlin axis resolution into `Nonlin.lean`.** `applyNonlin` takes
`axisPos/axisUids` that 6 of 9 variants ignore, forcing both callers to pre-resolve with the
hazard-prone match. Give `Nonlin` a `normMask? : Nonlin → Option (Option BoolExpr)`
classifier and make `applyNonlin` `Except`-returning, owning the axis lookup. Callers shrink
to one line; the classification lives in exactly one place, adjacent to the constructor list.
(If Spike 3a lands first, this is nearly free — classification is then just a 3-arm match.)

**4e. Split `Eval/Shape.lean` (521 lines, five jobs).** New layout:
`Eval/Slots.lean` (AST/slot vocabulary: `lhsSlots`, `readsOf` deletion per Spike 2,
`LHSSlot.outExtent` per 2d, `outputShape`), `Eval/SizeSolve.lean` (constraint types,
diagnostic rendering, the exact-RREF solver — :26-371), `Eval/SizeInfer.lean` (fixpoint
driver + scatter shapes — :373-498). Pure file moves. **Caution:**
`AffineShapeSolverTest.lean` pins exact substrings of rendered diagnostics ("rank=1, vars=2",
"ml-hint: …") — moving files is safe, editing `renderSolveDiagnostic` is not.

**4f. Decompose `evalScan`** (`Eval/Scan.lean:91-135`, 45 lines, four phases already numbered
in comments) into `allocStates` / `applyBase` / `runRecurStep` + an `iterWritePositions`
helper for the `(position, index+δ)` bookkeeping that appears twice (δ=0 base, δ=1 recur).
Behavior-neutral; preserve the `work` vs `stepEnv` insert ordering (:130-132) exactly.

**4g. Replace the stringly-typed scatter reduce.** `opts.reduce : Option String` is matched
as `some "sum" | some "max" | _ => overwrite` (`Eval/Scatter.lean:53-56`) — an unrecognized
string silently overwrites, violating the fail-loud convention. Introduce
`inductive ReduceOp | sum | max | overwrite`, parse/validate once at the boundary.

**4h. Structured `EvalError` (largest item — do last).** `abbrev EvalError := String`
(`Eval/Tensor.lean:4`) vs the structured `CompileError`. Tests show the cost:
`AffineShapeSolverTest` needs 5–8 chained `.contains` substring assertions per case, and
`SolveDiagnostic` — already a structured value — is eagerly flattened to a string at every
throw site (`Shape.lean:330,340,353,366`). Proposal: an inductive with
`unknownTensor / unsizedAxis / sizeConflict / solveFailure (SolveDiagnostic) / domainError
(UnaryOp, Float) / normAxisError / unsupported String`, plus a `ToString` that reuses
`renderSolveDiagnostic`. Migration plan: keep `ToString` output byte-identical first (tests
pass unchanged), migrate assertions to field matches second. ~20 throw sites across 6 files;
`SolveDiagnostic` and friends need un-`private`-ing.

**4i. Small decoupling wins:** `Eval/Eval.lean` imports all of `DSL.Compile` solely for the
`TLProgram.eval` wrapper (:78-83) — move the wrapper out (e.g. `Eval/Entry.lean`) so the
interpreter depends only on `DSL.Ast` + `Pipeline.Types`. And `evalScheduled` drops solver
warnings into `dbg_trace` (`Eval/Eval.lean:64`) — return them instead
(`AffineShapeSolverTest:412` already asserts on a returned list elsewhere).

### Spike 5 (cross-cutting, after 2–4): decompose `finalizeScans`

`Structural.lean:406-521` — one 115-line function doing seven jobs (base-axis adoption,
scanPre partition, dep fixpoint, connected components, cross-component guard, per-component
validation ×4 + node assembly, plain classification), with five local closures. It absorbed
four behavioral fixes this cycle (multi-axis, heterogeneous-coupling reject, scan-projection
reject, aggregators) and is where the next scan feature will land too. Extract:

```lean
def Stmt.scanAxisSet : Stmt → List UID                    -- the axSet closure
def scanDeps (stmts) : HashMap String (List UID)          -- reads-only fixpoint (dep)
def scanComponents (iterStmts) : List (List UID)          -- CC by repeated merge
def mkScanNode (dep nonPre comp) : Except CompileError ScanStmt   -- classify + validate + build
```

`finalizeScans` becomes ~15 lines of plumbing; the four validation loops live inside
`mkScanNode` next to the classification predicates they guard. Uses Spike 2's
`LHSSlot.axisUID?` and `IterSlot`. Well pinned: StructuralTest (scan grouping, coupled scans,
missing base, causality), ScanTest, RecurrenceTest (RC4/RC9), KnownGap/RejectTest (UF5), the
3-D scan test. Behavior-dense — this is the one Wave-2 spike that warrants its own detailed
plan with a test-first checklist.

Optional adjacent cleanup: `ScanStmt.scan`'s positional
`String → List AxisSpec → List Stmt → List Stmt → Bool` constructor produces `.scan _ _ b r _`
patterns at ~15 sites; a `ScanNode` structure argument would name the fields. Medium churn,
readability-only — fold into this spike or skip.

---

## Wave 3 — proof track: consolidate lemma bases, prune the sorry surface, simplify types

Three spikes, each an independent lens on the proof track: **Spike 6** consolidates proof
*scripts* (no statement changes), **Spike 7** prunes the *sorry surface* to its honest open
core, and **Spike 8** simplifies the *types* the proofs are stated over (the
noncomputable/computable seam). None of the three depends on Wave 2; see the ordering table
at the end for how they interleave with it.

### Spike 6: proof-infrastructure consolidation (RouteSpec + AcsetCodec, ~600 lines)

No statement changes anywhere in this spike — pure proof-engineering. It also *future-proofs*
the proofs against Wave-2 pipeline changes, which is why 6a is worth doing early if Spike 3
is planned.

**6a. Factor `ScanStmt.toBrBaseP` out of `buildStep`.** Five RouteSpec lemmas
(`buildStep_outputWeaves`:344, `_reindexings`:405, `_degree`:424, `_inputWeaves`:613,
`_wires_mapM`:716) each redo the same 3-way `cases sc` + `bind_pure_pair_ok` dance because
`buildStep` (`Lowering.lean:498-564`) interleaves guards with record construction. Reshape as
`guards; pure (sc.toBrBaseP, ← wires)`; the five collapse to one
`buildStep_ok_eq : buildStep … = .ok (b, w) → b = sc.toBrBaseP ∧ …` plus `rfl` projections.
Future `BrBaseP` field additions then need zero new extraction lemmas — this is what makes
Spike 3a's `buildStep` touch cheap. ≈ −150 lines.

**6b. One `routeCore` elimination lemma.** Six proofs (RouteSpec :87-162, :469-513, :692-712)
re-unfold `routeCore`, case on `routableInOrder`, case on the `mapM`. Prove
`routeCore_ok_elim` once (+ a generic `routeCore_forall_steps` for the three
per-step-property lifts). ≈ −120 lines.

**6c. Shared `Util/ListMapM.lean`.** RouteSpec's private `mapM_ok_length/getD/mem` (:19-83)
are re-exported for AcsetCodec (:897-906); `map_eq_range_map_getD` is duplicated *verbatim*
in `Realize.lean:173-178`; `find?_unique` lives in AcsetCodec. One shared util module.

**6d. AcsetCodec (1,593 lines): split + two combinators.** Split into `Codec.lean` (defs,
≈330 lines), `Isolation.lean` (generic list lemmas), `RoundTrip.lean` (the theorem). Then two
proof schemas that account for ~700 lines collapse:

- *Keyed-table isolation*: `filterSlot_flatMap_off` (:601-626) and `filterSlot_samples_off`
  (:1050-1081) are the same proof modulo row type; one polymorphic `filter_tagged_groups`
  lemma covers both plus the degree-slot case (:637-665). ≈ −120 lines.
- *find?-uniqueness-by-injective-key*: `lookupSize_from` (:752), `lookupName_from` (:784),
  `from_equation_find` (:1370), `from_inputRow_find` (:1383), and two ~45-line blocks inside
  `decodeReindexing_from` all follow membership-witness + injectivity + `find?_unique`. One
  `find?_of_injective_key` combinator per-site cost drops to ~5 lines. ≈ −150–200 lines.

### Spike 7: prune the sorry surface to the honest open core (38 → ~14)

Full census performed (verified by grep + reading each site). The Bridge track
(Realize/Agreement/AcsetCodec/SBr/RouteSpec) and the whole executable track are **zero-sorry**
today; all 38 are math-tower. Three actions:

**7a. Delete the two impossible `Algebra` instances (−17 sorries, ~30 min).**
`Algebra/Target.lean:79-86` (8 fields) is false-as-stated by a *recorded mathematical
obstruction* (file comment :70-77: no faithful dimension-adding `actV` over `FGModuleCat ℝ`
with symbolic sizes; `δ_V` forces `f(P)=f(P)² ⇒ 1`). `Algebra/Construct.lean:14-21,43`
(8 fields + `construct_correspondence`) is gated on it and quantifies over sorry data. Keep
the *classes* (`TargetActegory`, `Algebra`, `ParaAlgebra` — all sorry-free) and the proved
`semiring_choice_split`; keep the obstruction notes as docs; delete the instances. They
remain instantiable at Milestone I with concrete `Nat` sizes — which is the actual plan of
record anyway.

**7b. Park `BrNF` outside the default build (−6 sorries, −247 built lines).** Verified:
imported by nothing except the root `LeanNCD.lean:101`. More importantly, a **model-adequacy
finding supersedes its plan**: `Hom` gained CD generators `copyW`/`delW` (`Br.lean:61-62`)
with comonoid laws in `Rel` (:125-134), and `brCancelPoint` quantifies over *all* raw terms —
but `BrNF`'s `NData` wiring is a *bijection* (`OutPort ≃ InPort`), which provably cannot
interpret one-in-two-out `copyW` (`BrNF.lean:243-244` admits exactly this). **BrNF as
designed can never complete `brCancelPoint`.** The honest route is a gs-monoidal/cospan
(non-bijective) model — a larger development than BrNF's header estimates, which predate the
CD extension. Actions: drop the import, move the file to `spikes/`, and record the
cospan-model requirement in `Br.lean:298`'s doc (the sorry-free wiring combinators and the
`@[simp]`-projection technique in BrNF survive in the file for eventual reuse). Optionally
demote `instance : Elemental BrObj` (`Br.lean:425`) to a scoped instance/def so no default
instance carries a transitive sorry — `weave_unique`/`weave_subsingleton` already take
`[Elemental C]` explicitly. Consumer chain verified safe: `brCancelPoint → Elemental BrObj →
weave_unique (itself sorry, double-gated) → weave_subsingleton → nothing`; the executable
target uses the *proved* `St.elemental`.

**7c. Restate or gate `grothendieck_split` (−1 false-as-stated sorry).** With the stub
`structuralCongruence := fun _ _ _ _ => True` (`Grothendieck/Split.lean:15`), `Cˢʰᵃʳᵖ` is the
thin quotient and the claimed equivalence `C ≌ ∫Dat` is false for any non-thin `C`.
Parameterize the theorem over a real `HomRel`+`Congruence` hypothesis, or delete until the
real relation exists.

**What remains after 7a–7c (~14, the honest open surface):**

| Item | Status |
|------|--------|
| `St.swap_hexagon_fwd/rev` (`St.lean:267-268`) | **The one recommended proof spike.** Provable as stated, same genre as the proved `tensorHom_assoc`. Known friction (recorded from a prior attempt): the associator casts are defeq-not-syntactic — needs the `rw`/`erw` cast-bridge idiom (cf. `Br_swap_hexagon_fwd`'s `cast_quot_id`), *not* simp; budget a focused Br-hexagon-sized effort, not a quick win. Closing them makes `St` fully sorry-free; the Adapter consumers already exist. |
| `brCancelPoint` (`Br.lean:300`) | True; parked pending the cospan model (7b). Not load-bearing. |
| `weave_unique` (`Core/Weave.lean:32`) | Double-gated (`broadcast_gen`, `brCancelPoint`); park. |
| `instDGradedStBr` 10 fields (`Instances/StBr.lean:15-24`) | The genuine long pole: `act` is de-risked (spike S0, ~400–650 lines), `sh_act` redesigned to `≅`, `broadcast_gen` deep. Staged work, unaffected by this doc. |

### Spike 8: proof-adjacent type cleanups (the noncomputable/computable seam)

A fourth, separate proof-track spike — findings that are neither sorry-count reductions
(Spike 7) nor proof-script consolidation (Spike 6), but simplifications of the *types* the
proofs are stated over. Grouped here because they're small, mostly independent of each other
and of Spikes 6/7, and each closes a documented "this is a known wart" note rather than fixing
a bug.

**8a. `StMatP'` (`DSL/Target.lean:38-52`) is nearly dead.** Only `toStMatP` + one theorem
consume it; no pipeline stage does. Either wire `elaborateReindexings` through it (making
`StMatP.wellFormed` structural by construction, deleting `StMatP.validate`) or delete it
outright. Low cost either way; resolve it rather than leave it half-used.

**8b. `realizeStMat` could take a `wellFormed` hypothesis instead of `getD 0` fuzz.**
Currently (`Bridge/Realize.lean:25-28`) it defaults out-of-range entries silently. Now that
`routeCore_reindexings_wellFormed` (`RouteSpec.lean:469`) supplies well-formedness upstream,
`realizeStMat` could accept it as a hypothesis instead — but only worth doing if a future
proof actually needs entry-level faithfulness; otherwise the fuzz is harmless and this is
churn without payoff. Lowest-priority item in this spike.

**8c. `Axis.size : Numeric → SizeExpr` swap — the concrete next step on the
`leanncd-numeric-noncomputable` open item.** `SizeExpr` already backs the entire DSL/Eval
track; the remaining `Numeric` surface is small and, critically, *inert*: `Base/Numeric.lean:9`
(the def), `Base/St.lean:10` (`Axis.size`), `Base/Br.lean:8` (`DType.nat`),
`DSL/SizeExpr.lean:37-43` (the lossy `toNumeric` bridge — `.sub`/`.div` are silently dropped, a
documented lossiness), `Bridge/Realize.lean:11` (`realizeAxis`, the sole `toNumeric`
consumer). Nothing in any `St`/`Br` proof pattern-matches on a `Numeric` *value* — the `StMat`
laws touch only `coeffs`/`bias` (`Coeff = MvPolynomial String ℤ`, genuinely used via
`Matrix.mul_assoc`/ring reasoning) — so swapping `Axis.size`/`DType.nat` to `SizeExpr` is
safe: it makes `StObj`/`BrObj`/`WeaveShape` *objects* computable (morphisms stay noncomputable
via `Coeff`), deletes `SizeExpr.toNumeric` outright, and turns `realizeAxis` into the identity
on sizes — killing the documented sub/div lossiness for good. Touch set: `Base/St.lean` (1
field, laws unaffected), `Base/Br.lean` (1 constructor), `Bridge/Realize.lean` (2 defs
simplify), `Base/Numeric.lean` (keep only `Coeff`), `Base.NumericTest`/`Bridge.RealizeTest`.
Estimated < 1 day. **Does not** fix the `Algebra/Target` obstruction (Spike 7a) — that needs
concrete `Nat` sizes, a separate design decision. **Explicitly not recommended as a
follow-on:** a full `Coeff` computability swap (needed to make `StMat` itself computable)
would need a computable multivariate-polynomial normal form and re-proving the St laws over
it; the presentation layer (`StMatP`) already covers the computable need, so that would be
effort spent on a track that already has a computable twin.

**8d. Optional, very low value:** `Base/Category.lean` (31 lines) has one importer besides
tests (`ColoredPROP.lean`) and could merge into it. Only worth doing opportunistically.

---

## Suggested spike ordering and dependencies

```text
Wave 1   Spike 1  ✅ DONE 2026-07-10 (dead code/docs/housekeeping) — merged 1cf92d3..3bb7e13
Wave 3a  Spike 7a/7b/7c (sorry pruning)                          — anytime, zero deps
         Spike 8  (proof-adjacent type cleanups: 8a/8c independent, 8b/8d opportunistic)
Wave 2   Spike 2  (AST accessors/traversals)                     — first structural spike
         Spike 6a (toBrBaseP)                                    — before/with Spike 3
         Spike 3  (Nonlin + Elab tables; then optional 3c)       — after 2 (and ideally 6a)
         Spike 4  (Eval unification; 4a→4b→4c→4d, then 4e–4i)    — after 2; 4d cheaper after 3
         Spike 5  (finalizeScans decomposition)                  — after 2
Wave 3b  Spike 6b/6c/6d (RouteSpec/AcsetCodec consolidation)     — independent of Wave 2
         St hexagons proof spike                                 — independent, budget separately
```

Every spike is verified by `lake build` (full suite, currently 8,602 jobs) — the test
portfolio pins numeric behavior, reject paths, and the two historical wildcard bugs
(FF5–FF8), so mechanical refactors here are unusually safe. The two spikes that need genuine
test-first care are Spike 5 (`finalizeScans` is behavior-dense) and Spike 3b's `atomic`
lookahead interaction (run the parse-layer tests early).

---

## Part II — Exploratory redesigns (creative, higher-risk, prototype-first)

Part I optimizes the architecture as it stands. This part asks a different question: *if the
recent fixes are symptoms, what would the design look like that doesn't produce them?* Each
idea below names the design pattern or functional-programming pearl it draws from, sketches
what it would concretely look like in this codebase, and ends with a verdict
(**spike-worthy** — could be planned today; **prototype-first** — build a throwaway proof of
concept before committing; **recorded-only** — worth knowing about, not worth doing now).
None of these are prerequisites for Part I, but they are not free-floating either: several
*replace*, *fold into*, *depend on*, or *should precede* a specific Part I spike. The per-idea
writeups note this inline; [How the explorations relate to the Part I spikes](#how-the-explorations-relate-to-the-part-i-spikes)
below consolidates it into one map so the interaction is decidable at planning time.

| # | Idea | Pattern provenance | Verdict |
|---|------|--------------------|---------|
| E1 | One traversal to rule the collectors | van Laarhoven traversals (Kmett lenses) | prototype-first |
| E2 | A typed core IR the pipeline narrows into | "Trees that grow" (GHC), typestate | spike-worthy (light form) |
| E3 | Evaluate over an arbitrary semiring | algebra-parameterized interpreters | spike-worthy |
| E4 | EvalPlan: run the route's own affine maps | stride-based einsum; "one artifact, two consumers" | prototype-first |
| E5 | Executable denotation of the routed artifact | denotational semantics as executable spec | spike-worthy |
| E6 | Property-based oracles for the pipeline | QuickCheck / metamorphic testing | spike-worthy |
| E7 | Accumulate compile diagnostics | the Validation applicative | spike-worthy |
| E8 | Open registration of unary functions | environment extensions, command macros | recorded-only |
| E9 | Datatype-generic acset codecs | custom `deriving` handlers | recorded-only |
| E10 | Stage the interpreter into a code generator | Futamura projections | recorded-only |
| E11 | Do UIDs earn their keep? | locally-nameless discipline | investigation |
| E12 | Named simp sets for the proof domains | mathlib attribute hygiene | spike-worthy (tiny) |

### How the explorations relate to the Part I spikes

Each exploration sits in one of five relationships to the concrete spikes. The distinction
matters for planning: two must be *decided at a specific spike* (you cannot do the spike its
plain way and add the exploration later without wasted work), two are worth doing *before* the
spikes they protect or reshape, and the rest are genuinely independent — pull them in whenever.

| Idea | Relationship to Part I | When to act |
| ---- | ---------------------- | ----------- |
| **E1** (one traversal) | **Replaces** Spike 2a/2b if the prototype succeeds; makes `TermTraversable` its `Id` case | *At Spike 2* — prototype E1 first; commit to the shared-functions form (2a/2b) only if it fails. Deciding this after 2a/2b lands wastes that work. |
| **E12** (named simp sets) | **Folds into** Spike 6 | *During Spike 6* — register the lemmas as you consolidate them, not as a separate pass. |
| **E3** (semiring) | **Depends on** Spike 4a/4b — it is their payoff (`unit1` threaded, one evaluator body) | *After Spike 4.* Not a competitor to Spike 4; do not attempt before it. |
| **E10** (codegen) | **Depends on** E4 (its `EvalPlan` is ~80% of the specialization) | *After E4*, and only once the Python-target work is actually scheduled. |
| **E6** (property oracles) | **Precedes** — the regression net that makes behavior-dense refactors low-anxiety | *Before Spikes 4 and 5, and before E4.* Highest value landed early; the scan-unrolling oracle is precisely what de-risks Spike 5's `finalizeScans` rewrite and E4's plan compiler. |
| **E11** (UID investigation) | **Precedes/reshapes** Spike 2 — a "yes, delete UIDs" outcome changes what 2b/2c relocate | *Before committing Spike 2* — one focused day. If UIDs survive, Spike 2 proceeds unchanged; if not, 2b/2c shrink or move. |
| **E4** (EvalPlan) | **Overlaps** Spike 4 — rewrites the same `Contract`/`Gather`/`Scatter` internals Spike 4 unifies | *Decide at Spike 4.* If E4 is on the roadmap, weigh how much of Spike 4's internal unification (4c/4d/4f) it would supersede before investing there; keep 4a/4b regardless (E4 builds on them, as E3 does). |
| **E2 light** (typed core IR) | **Overlaps** the Wave-2 executable spikes (3–5) — `CoreStmt` makes unwritable the invariants 3/4/5 only document | *After Spikes 3–5*, as their typed successor — or fold the `Recurrence` structure into Spike 5 if attempted concurrently. |
| **E5** (executable denotation) | **Independent** — new executable spec; complements proof-track Spikes 6/7, replaces nothing | Anytime; cheap to start incrementally. |
| **E7** (Validation applicative) | **Independent** — a *Compile*-layer change (`resolveDecls`/`checkReadRanks`/`checkDtypes`), distinct from Spike 4h's *Eval*-layer `EvalError` | Anytime; self-contained. |
| **E8** (open unary registration) | **Independent, recorded-only** — Spike 3b already shrinks the syntax/elab share | Deferred; revisit only if the function inventory starts growing monotonically. |
| **E9** (generic codecs) | **Independent, recorded-only** — Spike 6d's combinators already capture most of the line savings | Deferred; revisit only if the acset table inventory grows. |

**So: should the explorations come first?** Mostly no — do the Part I spikes on their own
schedule and pull each exploration in at the point above. The two exceptions are **E6** and
**E11**: E6 is the safety net worth having in place before the behavior-dense spikes (4, 5) and
E4 touch anything, and E11 is a one-day question worth answering before Spike 2 invests in
relocating UID-keyed accessors. The remaining sequencing among the explorations themselves
(E6 → E3 → E4 → E5) is covered in [How the exploratory ideas compose](#how-the-exploratory-ideas-compose).

Concretely, here is one sequencing that satisfies every dependency above — it extends the
Part I [spike ordering](#suggested-spike-ordering-and-dependencies) by front-loading the two
exploration *enablers* (E6, E11) and slotting the rest in at their decision points, while the
proof track runs in parallel with the executable track throughout:

```text
Phase 0  — zero-dep, start now, run in parallel
  Spike 1        dead code / stale docs / housekeeping
  Spike 7a/7b/7c sorry-surface pruning (38 → ~14)
  Spike 8a/8c    proof-adjacent type cleanups
  E6             property-based oracles      ← regression net; MUST land before Spikes 4/5 + E4
  E11            UID investigation (~1 day)  ← MUST resolve before the Spike 2 commit

Phase 1  — foundational structural spike (executable track)
  E1             prototype the one-traversal encoding   ← decides the shape of Spike 2
  Spike 2        AST accessors/traversals — as 2a/2b, OR replaced by E1 if the prototype lands
                 (scope informed by E11's outcome)
  Spike 6a       factor toBrBaseP out of buildStep (proof track; before Spike 3)

Phase 2  — executable-track unification (after Spike 2; E6 net in place)
  Spike 3        Nonlin + Elab tables (after 6a); optional 3c after Spike 2a
  Spike 4        Eval unification: 4a → 4b first …
  E4             ← decision point: if pursued, prototype on 4a/4b; it supersedes parts of
                    4c/4d/4f — then run 4c–4i, skipping whatever E4 replaces
  E3             evaluate over a semiring (after 4a/4b — this is their payoff)
  Spike 5        finalizeScans decomposition (E6's scan-unrolling oracle guards this rewrite)

Phase 3  — proof consolidation (independent of Waves above; run in parallel)
  Spike 6b/6c/6d RouteSpec + AcsetCodec consolidation
  E12            named simp sets            ← folded into Spike 6
  St hexagons    swap_hexagon_fwd/rev proof spike (budget separately)

Phase 4  — typed successor + executable denotation (after Spikes 3–5)
  E2 (light)     CoreStmt boundary type (subsumes invariants Spikes 3–5 only document)
  E5             executable denotation / agreement test (natural once E4 lands)

Opportunistic / deferred (pull in on demand, no phase)
  E7             accumulate compile diagnostics (Validation) — anytime, self-contained
  Spike 8b/8d    lowest-priority type cleanups
  E8, E9         open unary registry / generic codecs — recorded-only; revisit if inventories grow
  E10            interpreter → codegen — after E4, and only when the Python target is scheduled
```

The critical path runs Phase 0's E6/E11 enablers → Spike 2 → Spike 4 → E4/E3 → E5; everything
in Phase 3 and the "deferred" block is off that path and can be scheduled to fill gaps. If only
one exploration thread is resourced, run **E6 → E3 → E4 → E5** (the highest-value path from
[How the exploratory ideas compose](#how-the-exploratory-ideas-compose)) and leave the spikes
otherwise untouched.

### E1. One traversal to rule the collectors (van Laarhoven)

Spike 2 deduplicates the collectors by sharing functions. The pearl-level fix is stronger:
almost every function in `Traverse.lean`, the `specs*` family, and the `*AxisUIDs` family is
the *same* traversal instantiated at a different applicative functor:

```lean
def Factor.traverseAxes [Applicative f] (g : AxisSpec → f AxisSpec) : Factor → f Factor
-- rename/remap  = traverseAxes at Id        (subsumes Factor.mapUID and friends)
-- collect specs = traverseAxes at Const (List AxisSpec)   (subsumes specsFactor…)
-- collect uids  = traverseAxes at Const (List UID)        (subsumes termAxisUIDs…)
```

One definition per AST node (`IdxExpr`, `BoolExpr`, `Factor`, `LHSSlot`, `Stmt`), and every
mapper/collector — present and future — is a two-line instantiation. A `Const` applicative is
~5 lines (Lean core doesn't ship one). The catch discovered in review: the rebuild tower maps
`UData → UData` (`Traverse.lean`) while the collectors return `AxisSpec` — the two views of
"an axis occurrence" must be reconciled into one canonical occurrence type before the
traversal can subsume both. That reconciliation is itself clarifying (it answers "what IS an
axis occurrence?" once), but it's why this is prototype-first rather than a drop-in
replacement for Spike 2. If the prototype works, it *replaces* Spikes 2a/2b entirely and
makes `TermTraversable` (`Exec/Traversable.lean`) its `Id` special case.

*References:* Gibbons & Oliveira, [The Essence of the Iterator Pattern](http://www.cs.ox.ac.uk/jeremy.gibbons/publications/iterator.pdf) (JFP 2009) — traversal as applicative-functor iteration, the `Const`-vs-`Id` instantiation trick used above; van Laarhoven, [CPS-based functional references](https://www.twanvl.nl/blog/haskell/cps-functional-references) (2009); Kmett, [`lens`](https://hackage.haskell.org/package/lens) (the library that industrialized the encoding).

### E2. A typed core IR the pipeline narrows into ("trees that grow")

The scan-projection bug hunt surfaced a class of hazard Part I only documents, never removes:
**phase invariants that live in comments.** "Classification must happen before `splitNonlins`
because `splitStmt` reuses slots verbatim"; "after `lowerArith` every affine LHS is a
scatter"; "post-split, every statement is either pure contraction or pure nonlin" — each is
an invariant some later phase silently relies on, and each is exactly the kind of fact a type
can carry. Two escalation levels:

- **Light (spike-worthy): one extra type at the one boundary that has already bitten.**
  After `splitNonlins`, introduce `CoreStmt` — no `nonlin` field on contractions; nonlin
  application is its own constructor; scans carry an explicit
  `structure Recurrence where state : String; init : Stmt; step : Stmt` instead of parallel
  base/recur lists paired by name-matching. `eval`/`route` consume only `CoreStmt`. Payoffs:
  `buildStep`'s op selection becomes total by construction (no `.identity`-with-agg fallback
  chain), `evalScan`'s `stateNames` name-matching (`Eval/Scan.lean:99,126-133`) disappears,
  and the splitNonlins slot-reuse subtlety becomes unwritable rather than documented.
- **Heavy (recorded-only): phase-indexed programs.** `Prog (φ : Phase)` with type families
  narrowing field types per phase, GHC's "trees that grow" transplanted. Honest cost: Lean's
  `deriving` (`DecidableEq`, `Repr`, `ToExpr` — all load-bearing here for elaboration-time
  embedding) does not play well with indexed families, and every test constructing record
  literals pays churn. The light form captures most of the value; escalate only if a third
  phase-invariant bug appears.

*References:* Najd & Peyton Jones, [Trees that Grow](https://simon.peytonjones.org/trees-that-grow/) (JUCS 2017; [arXiv](https://arxiv.org/abs/1610.04799)) — the type-family-per-phase idiom from GHC's own AST; Strom & Yemini, [Typestate: A Programming Language Concept for Enhancing Software Reliability](https://www.cs.cmu.edu/~aldrich/papers/classic/tse12-typestate.pdf) (IEEE TSE 1986) — invariants a type carries across phases.

### E3. Evaluate over an arbitrary semiring

`Combine {mul, combine, unit0}` plus Spike 4b's `unit1` is already a semiring struct with an
apology. Name it, finish it, and parameterize the *whole* interpreter (contract, scan,
scatter — `evalScatter` currently hardcodes ℝ and ignores `rhs.agg` entirely,
`Scatter.lean:40-47`):

```lean
structure EvalSemiring where
  add : Float → Float → Float := (· + ·);  zero : Float := 0
  mul : Float → Float → Float := (· * ·);  one  : Float := 1
```

Instances: `real`, `maxTimes`/`minTimes` (exist today as `Combine.max/min`), `bool` (exists),
plus the two that unlock known gaps for free: **`minPlus`** (`add := min, mul := (+), zero :=
+∞, one := 0`) closes KG-min's open half — shortest path, Viterbi, DTW — with *zero* new
evaluator code once the surface `agg(⊕,⊗)` notation (already designed, portfolio §19.2)
selects the instance; and **`logSumExp`** (with a max-shift for stability) gives the
log-domain HMM forward algorithm and numerically-stable marginalization. One interface, three
textbook algorithms. Mathlib connection worth recording: `Tropical` exists in mathlib, so if
the semiring interface is ever lifted from `Float` to a typeclass, the tropical instances can
be *certified* against mathlib's, giving the Eval track its first proof surface that isn't
shape-only. Sequencing: requires Spike 4a/4b (one evaluator body, `unit1` threaded) — this is
their payoff, not their competitor.

*Reference:* Dolan, [Fun with Semirings: a functional pearl on the abuse of linear algebra](https://stedolan.net/research/semirings.pdf) (ICFP 2013; [ACM](https://dl.acm.org/doi/10.1145/2500365.2500613)) — one semiring interface yielding transitive closure, shortest paths, and dataflow; the direct provenance for the `minPlus`/`logSumExp` instances above.

### E4. EvalPlan — make the evaluator run the same affine maps the route proves things about

The deepest structural observation from the review. The evaluator's per-point protocol —
build a `HashMap UID Int` coordinate, `evalIdx` each `IdxExpr` against it, silently `getD 0`
on a missing UID (`Gather.lean:8-12`) — re-implements, in dictionary form, exactly what the
presentation layer already represents as integer matrices: `idxDensify` (`Ast.lean:47`)
turns an `IdxExpr` into a dense coefficient row, and `elaborateReindexings` assembles those
rows into `StMatP`s that `RouteSpec` proves well-formed. The evaluator never touches them.

Proposal: a plan-compilation pass `Stmt → EvalPlan`, where each factor carries a densified
affine map (coefficient rows over `frees ++ termContr` + bias — i.e., an `StMatP` in all but
name) from the loop coordinate vector (a flat `Array Int`) to the source offset. Gather
becomes dot-product-and-index; the per-cell `HashMap` copy (`Contract.lean:94,186` — a full
map clone per contraction cell) disappears; scan seeds become one extra bias contribution
composed into the plan.

Three distinct payoffs: (1) **correctness** — the `getD 0` silent-default hazard (any
seeding/scoping bug reads plane 0 instead of failing) is structurally eliminated, since plan
compilation resolves every axis once, fail-loud; (2) **performance** — per-point cost drops
from hash-map allocation to integer arithmetic, asymptotically removing the dominant
interpreter overhead; (3) **architecture** — eval and route come to consume *the same
artifact*, which is precisely what makes E5's agreement statement natural. Cost: a genuine
rewrite of `Contract`/`Gather`/`Scatter` internals (~300 lines) with numeric-equivalence risk
— prototype on the scan-free fragment and diff against the current evaluator over the whole
portfolio before committing.

*Reference:* the [`numpy.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) Einstein-summation convention — the stride/coefficient-row contraction spec this plan representation mirrors.

### E5. An executable denotation for the routed artifact

`LeanNCD.lean:87-94` names the principal open seam: the categorical meaning of a runnable
artifact — `realize` and the §8 agreement — is deferred proof work. There is a cheaper
attack that changes the epistemic situation *now*: give the routed presentation an
**executable** semantics,

```lean
def evalBrBaseP    : BrBaseP → List DenseTensor → Except EvalError DenseTensor
def evalThreaded   : ThreadedComposed → (ext : List DenseTensor) → Except EvalError (List DenseTensor)
```

and state the agreement as a *test* long before it's a theorem: for every scan-free portfolio
program `p`, `evalThreaded (route p) ext ≈ TLProgram.eval p env`. (Scan-free because the
routed artifact deliberately collapses scan bodies — `LeanNCD.lean:84-86`; the fragment still
covers most of the portfolio.) This catches a whole class of bug RouteSpec cannot: RouteSpec
proves weave *shapes* are right, not that the wiring *means* the right contraction. It also
gives the eventual `realize`-agreement proof its computational skeleton — the denotation is
the function the proof will characterize. Cheap to start (`evalBrBaseP` for `contract`/
`relu`-class ops is a small wrapper over existing Eval pieces), incremental thereafter.

*Reference:* [Denotational semantics](https://en.wikipedia.org/wiki/Denotational_semantics) — giving a program a meaning as a mathematical (here, *executable*) function; the agreement test characterizes `route` against that denotation before the eventual `realize` proof does.

### E6. Property-based oracles (the tests the portfolio can't express)

The portfolio pins ~130 hand-computed examples — excellent, but every one is a point. The
pipeline has *laws*, and Lean has a QuickCheck (`Plausible` — confirmed present in this
project's `lake-manifest.json` via mathlib). Three oracles stand out, in increasing order of
power:

1. **Reordering invariance:** independent top-level statements can be permuted without
   changing `eval` output (pins `topoSort`+`schedule` semantics, which Spike 1a is about to
   touch).
2. **Materialization invariance:** `Y := A + B·C` ≡ `T1 := A; T2 := B·C; Y := T1 + T2` —
   the per-term-scoping law, generalized from EC13/EC15's two points to all small programs.
3. **Scan-unrolling oracle (the strong one):** a scan over `axis l = L` must equal the
   hand-unrolled `L`-statement program with explicit indices. This is a *complete* semantic
   characterization of `evalScan` against the far simpler non-scan evaluator — precisely the
   oracle that would have caught the RC5/RC6-era silent-wrong-scan bugs mechanically, and the
   regression net under which Spike 5's `finalizeScans` rewrite and E4's plan compiler become
   low-anxiety changes.

The generator is the real work (small well-formed `TLProgram`s: 1–3 axes of size ≤3, 1–4
statements), and it need not round-trip through surface syntax — generate `Stmt` values
directly. Spike-worthy independent of everything else; highest value if landed *before*
Spikes 4/5/E4.

*References:* Claessen & Hughes, [QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs](https://www.cs.tufts.edu/~nr/cs257/archive/john-hughes/quick.pdf) (ICFP 2000) — the property/generator model (`Plausible` is its Lean port); Chen et al., [Metamorphic Testing: A Review of Challenges and Opportunities](https://dl.acm.org/doi/10.1145/3143561) (ACM Computing Surveys 2018; [tech report](https://www.cs.hku.hk/data/techreps/document/TR-2017-04.pdf)) — the invariance-relation oracles (reordering, materialization, scan-unrolling) above.

### E7. Accumulate compile diagnostics (the Validation applicative)

`FreshM = EStateM CompileError Nat` fails on the *first* error. For a human-facing DSL this
is a UX tax: fix one undeclared name, recompile, find the next. The classic fix is the
Validation applicative — same structure as `Except`, but `<*>` accumulates errors instead of
short-circuiting. The pipeline is already `mapM`-shaped in exactly the phases where
accumulation helps (`resolveDecls`, `checkReadRanks`, `checkDtypes` — per-statement,
independent checks); phases with genuine data dependence (`unifyAxes` onward) keep fail-fast.
Minimal version, no monad surgery: those three phases collect a `List CompileError` and throw
`CompileError.many : List CompileError → CompileError` at the phase boundary. Small,
self-contained, immediate payoff in every reject-test iteration loop.

*References:* McBride & Paterson, [Applicative Programming with Effects](https://openaccess.city.ac.uk/id/eprint/13222/1/Applicative-final.pdf) (JFP 2008) — the applicative structure whose error-accumulating `<*>` this exploits; the [`validation`](https://hackage.haskell.org/package/validation) package (the canonical accumulating instance).

### E8. Open registration of unary functions

Adding `atan` today touches `UnaryOp`, `Syntax.lean`, `Elab.lean`, and `applyUnaryFn` (4
sites; Spike 3b reduces the syntax/elab share). The open-datatype endgame is a command macro:

```lean
declare_unary_fn atan Float.atan            -- total
declare_unary_fn log  Float.log  (domain := fun v => v > 0)
```

backed by an environment extension (the same mechanism as `@[simp]` registration): `UnaryOp`
becomes a `String` key validated *at elaboration time* against the registry (so fail-loud is
preserved — a typo'd function name is still a compile error), and `applyUnaryFn` becomes a
registry lookup. Why recorded-only: it trades a closed inductive — which the AST's
`DecidableEq/ToExpr` deriving and any future proofs about evaluation lean on — for
extensibility that isn't currently in demand (five functions, stable for the foreseeable
future). Revisit if the function inventory starts growing monotonically. Open question to
resolve first either way: whether `Factor.unaryFn`'s op survives into the routed artifact at
all (the Br layer has no per-factor op today) — if the route is silently lossy for unary fns,
that's a Part-I-grade finding worth its own check.

*Reference:* [Metaprogramming in Lean 4](https://leanprover-community.github.io/lean4-metaprogramming-book/) — environment extensions and command macros (the same mechanism backing `@[simp]` registration) that a `declare_unary_fn` command would use.

### E9. Datatype-generic acset codecs

`AcsetCodec.lean`'s encode/decode pairs and their round-trip lemmas follow one schema per row
type. Lean 4 supports custom `deriving` handlers — a `deriving AcsetRow` that generates
`toRow`/`ofRow` and a `by simp`-discharged round-trip lemma for records of scalars is
feasible metaprogramming. Why recorded-only: Constraint 4 (the wire format is
position-sensitive and append-only by convention) means generated code must be pinned against
format drift anyway, and Spike 6d's combinators already capture most of the line savings
without metaprogram maintenance. The pattern earns its keep only if the table inventory grows
(e.g., when scan bodies get their own acset representation).

*Reference:* [Metaprogramming in Lean 4](https://leanprover-community.github.io/lean4-metaprogramming-book/) — custom `deriving` handlers (the metaprogram that would generate `toRow`/`ofRow` and the round-trip lemma).

### E10. Stage the interpreter into a code generator

`papers/pytorch_compilation.md` already contemplates compiling tensor-logic to Python. The
pearl framing: the evaluator is an interpreter `eval : Program → Env → Tensors`; its first
Futamura projection — specialize it to a fixed program — is a code generator. E4's `EvalPlan`
is 80% of that specialization already (per-statement dense affine maps + a semiring tag is
essentially an einsum call spec): `EvalPlan → String` emitting `torch.einsum`/`scatter_add`
calls is a small pretty-printer, and the Lean evaluator becomes the differential-testing
oracle for the emitted code. Recorded-only until the Python-target work is actually
scheduled, but it strengthens the case for E4: the plan representation serves eval,
route-agreement, *and* codegen.

*Reference:* Futamura, "Partial Evaluation of Computation Process — An Approach to a Compiler-Compiler" (Systems, Computers, Controls 1971; reprinted in *Higher-Order and Symbolic Computation* 12(4):381–391, 1999); overview: [Partial evaluation — Futamura projections](https://en.wikipedia.org/wiki/Partial_evaluation#Futamura_projections) — specializing an interpreter to a fixed program *is* a compiler.

### E11. Investigation: do UIDs earn their keep?

Axis identity is *defined* to be name-based within program scope (§12.1, quoted at
`Structural.lean:12`), yet the pipeline mints `Nat` UIDs (`FreshM`), then runs a coequalizer
(`Exec/Context.lean`) and `unifyAxes` to merge the UIDs that name-identity says were never
distinct. That's two representations of one equivalence relation, plus a phase to reconcile
them. The question worth one focused day: **what breaks if resolved names (plus a fresh-name
supply for `%nl`-style compiler-generated intermediates) are the identity, and UID minting
disappears?** Honest unknowns that the investigation must answer, not assume: whether any
phase genuinely distinguishes same-named occurrences (marks like `freeNorm`/`iterAt` live on
slots, not identities, so plausibly not); whether `Context`'s coequalizer does work beyond
name-unification (declared-vs-used axis reconciliation); and what the bridge/acset layers key
on (`axisUidFor3` suggests UIDs are load-bearing in serialization). Outcome either way is
valuable: either a major simplification (delete a phase and a monad parameter) or a precise
doc note on exactly why UIDs are necessary — which no comment currently states.

*Reference:* Charguéraud, [The Locally Nameless Representation](https://link.springer.com/article/10.1007/s10817-011-9225-2) (Journal of Automated Reasoning 2012) — names for free variables, indices for bound ones; the discipline that would let resolved names be the axis identity.

### E12. Named simp sets for the proof domains (tiny)

Mathlib-style hygiene: register the codec and route projection lemmas in named simp sets
(`@[acset_simp]`, `@[route_simp]`) so downstream proofs say `simp [acset_simp]` instead of
hand-curated lemma lists — the unused-simp-arg lint warnings currently scattered through
`AcsetCodec.lean:873-889` and `Agreement.lean:78-104` are the symptom this treats. Fold into
Spike 6 if adopted.

*Reference:* Lean reference manual, [Simp sets](https://lean-lang.org/doc/reference/latest/The-Simplifier/Simp-sets/) — `register_simp_attr` for named, non-default simp sets (`@[acset_simp]`, `@[route_simp]`).

### How the exploratory ideas compose

E3 + E4 + E5 are one program viewed from three sides: a single **plan representation**
(dense affine maps — the route's own `StMatP` language), evaluated over a **semiring
parameter**, that **denotes** the routed artifact. Landing all three would leave the
codebase with one semantic core serving four consumers — the Float interpreter, tropical/
log-domain evaluation, the executable agreement test, and (eventually, E10) code generation —
where today there are two disconnected coordinate systems and no executable bridge. E6 is the
safety net that makes the migration testable; E1/E2 are the AST-side analogues of the same
"one definition, many instantiations" move. If only one exploratory thread gets pursued,
E6 → E3 → E4 → E5 in that order is the highest-value path.

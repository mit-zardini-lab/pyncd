# `∀ p, WellFormed (compile p)` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ThreadedComposed.WellFormed` provably hold for every program the compiler accepts, then prove the compiler theorem `(compile p).run s = .ok tc s' → tc.WellFormed` — discharging `realize`'s precondition on real input.

**Architecture:** `WellFormed` is currently **false** for compiled programs (verified empirically on the 2-layer net). Two root causes are fixed first as compiler changes, then proved:
1. **Conjunct 2 (type match)** fails because [`route`](../../../LeanNCD/DSL/Pipeline/Lowering.lean#L257) sets every input weave to the step's own output weave (`readFactors.map (fun _ => mkWeave)`), so a producer's output type ≠ a consumer's input type. Fix: `route` derives every tensor's wire type **once** into a `nameToType` map and builds both the producer's output weave and each consumer's input weave from it — making conjunct 2 hold *by construction*.
2. **Conjunct 4 (topological order)** fails for forward-referencing source because `schedule` does no sort (it filters, preserving source order). Fix: add a real topological sort to `schedule` and prove it.

The proof is made tractable by first refactoring `route`'s `do`/`forIn` body into a **pure, structurally-recursive core** (`routeCore`) with equation lemmas, so the four conjuncts are provable by `List` induction rather than `EStateM`/`forIn` reasoning.

**Tech Stack:** Lean 4, Mathlib (pinned), Std `HashMap`, the project's `EStateM CompileError Nat` monad (`FreshM`). LSpec for executable tests; `#guard` for elaboration-time checks; `#print axioms` for sorry-freeness.

## Global Constraints

- **Sorry budget:** every new declaration must be `#print axioms`-clean — `[propext, Classical.choice, Quot.sound]` permitted, **no `sorryAx`**. The final theorem must be sorry-free.
- **No behavioural change to the evaluator.** `TLProgram.eval` uses `compileToScheduled` (pre-`route`); `route`/`schedule` changes must not alter `compileToScheduled`'s observable output for the 11 evaluator examples. `schedule` IS in `compileToScheduled`, so the topo-sort (Phase 2) must keep all `test/Eval/EvalExamplesTest.lean` assertions green.
- **No regression in existing tests.** `lake build` (whole library + `Tests` lib) stays green at every commit. Hand-built `tc` fixtures in `test/Bridge/RealizeTest.lean`, `test/DSL/Pipeline/TargetTest.lean`, `test/DSL/Pipeline/RecurMorphismTest.lean` construct `ThreadedComposed` directly and must be unaffected.
- **Match existing conventions:** `route`/`schedule` live in `LeanNCD/DSL/Pipeline/Lowering.lean`; the theorem and supporting lemmas go in `LeanNCD/Bridge/Agreement.lean` (it already imports `Bridge.Realize` which defines `WellFormed`). New structural lemmas about `route` may live in a new `LeanNCD/DSL/Pipeline/RouteSpec.lean` to keep `Lowering.lean` focused.
- **The theorem statement (verbatim target):**
  ```lean
  theorem compile_wellFormed (p : TLProgram) (s : Nat) (tc : ThreadedComposed) (s' : Nat)
      (h : (TLProgram.compile p).run s = EStateM.Result.ok tc s') : tc.WellFormed
  ```

---

## File Structure

- **Modify** `LeanNCD/DSL/Pipeline/Lowering.lean` — refactor `route` into `routeCore` + thin wrapper (Phase 0); add `nameToType`-based faithful weaves (Phase 1); add topo sort to `schedule` (Phase 2).
- **Create** `LeanNCD/DSL/Pipeline/RouteSpec.lean` — pure structural lemmas characterizing `routeCore`'s output (Phase 3).
- **Modify** `LeanNCD/Bridge/Agreement.lean` — the four per-conjunct lemmas + `compile_wellFormed` (Phase 4).
- **Create** `test/DSL/Pipeline/RouteWeaveTest.lean` — `#guard`/LSpec checks that compiled examples now have faithful weaves and pass a `decide`-able `WellFormed`-surrogate.
- **Modify** `leanncd/SORRY_INVENTORY.md` and `leanncd/realize.md` — record the fix and correct the stale "schedule topologically sorts" claim.

---

## Phase 0 — Refactor `route` to a provable core

The goal: replace `route`'s two-pass `do`/`forIn`/`mut` body with a pure function `routeCore : ScheduledProgram → Except CompileError (List BrBaseP × List (List Wire))` defined by structural recursion (`List.mapM`/`foldl` over `sp.stmts`), so its output is characterizable by equation lemmas. The PASS-1 maps (`nameToStep`, `extIndex`) are already pure folds; only PASS-2 needs lifting out of `forIn`.

### Task 0.1: Extract the per-step builder as a pure function

**Files:**
- Modify: `LeanNCD/DSL/Pipeline/Lowering.lean:218-290` (`route`)

**Interfaces:**
- Produces: `buildStep (nameToStep : Std.HashMap String Nat) (extIndex : Std.HashMap String Nat) (sc : ScanStmt) : Except CompileError (BrBaseP × List Wire)` — the body of the current PASS-2 loop iteration, returning one step + its wires (or a `CompileError.undeclaredName`/`shapeMismatch`).
- Produces: `routeCore (sp : ScheduledProgram) : Except CompileError (List BrBaseP × List (List Wire))` — folds `buildStep` over `sp.stmts`, accumulating; the `nameToStep`/`extIndex`/`nExternal` are computed by the existing pure folds.

- [ ] **Step 1: Add a characterization `#guard` test (the "failing test")**

In `test/DSL/Pipeline/RouteWeaveTest.lean` (create), assert the refactor preserves output on matmul:
```lean
import LeanNCD.DSL.Compile
namespace LeanNCD
-- BEFORE refactor this elaborates against the current `route`; it must still hold AFTER.
#guard (tl!{ Y[i,j] := W[i,k] · X[k,j] }).steps.length == 1
#guard (tl!{ Y[i,j] := W[i,k] · X[k,j] }).routing == [[Wire.external 0, Wire.external 1]]
#guard (tl!{
    H[i,k] := W1[k,d] · X[i,d]
    Y[i,j] := relu(W2[j,k] · H[i,k])
  }).routing == [[Wire.external 0, Wire.external 1],
                 [Wire.external 2, Wire.internal 0 0],
                 [Wire.internal 1 0]]
end LeanNCD
```

- [ ] **Step 2: Run to verify it passes against current `route`**

Run: `lake env lean test/DSL/Pipeline/RouteWeaveTest.lean`
Expected: no errors (these are the current behaviour; they pin it).

- [ ] **Step 3: Extract `buildStep`**

Move the body of the PASS-2 `for sc in sp.stmts` loop (everything from `match sc with | .scanPre …` through computing `step` and `wires`) into a top-level `def buildStep (nameToStep extIndex : Std.HashMap String Nat) (sc : ScanStmt) : Except CompileError (BrBaseP × List Wire) := do …`, returning `(step, wires)`. Replace the loop's `throw` calls with `Except` `throw` (same `CompileError` constructors). Keep `idxToRow`/`Stmt.readFactors`/`Stmt.lhsAxes` exactly as-is.

- [ ] **Step 4: Define `routeCore` and rewrite `route` as a wrapper**

```lean
def routeCore (sp : ScheduledProgram) : Except CompileError (List BrBaseP × List (List Wire)) := do
  let nameToStep : Std.HashMap String Nat :=
    (sp.stmts.zipIdx).foldl (fun m (sc, i) => sc.writes.foldl (fun m nm => m.insert nm i) m) {}
  let extIndex : Std.HashMap String Nat := buildExtIndex sp   -- the PASS-1 numbering, as a pure fold
  let built ← sp.stmts.mapM (buildStep nameToStep extIndex)
  pure (built.map (·.1), built.map (·.2))

def route (sp : ScheduledProgram) : FreshM ThreadedComposed :=
  match routeCore sp with
  | .ok (steps, routing) => pure { steps, routing, nExternal := sp.extNames.card }
  | .error e => throw e
```
Define `buildExtIndex` as the PASS-1 `extIndex` fold written purely (over `sp.stmts.flatMap ScanStmt.reads`, filtered to `∈ extNames`, first-seen numbering).

- [ ] **Step 5: Verify the refactor preserves behaviour**

Run: `lake env lean test/DSL/Pipeline/RouteWeaveTest.lean` (Step 1 guards) and `lake build`
Expected: green; `test/DSL/CompileExamplesTest.lean` and `test/DSL/Pipeline/LoweringTest.lean` still pass.

- [ ] **Step 6: Commit**

```bash
git add LeanNCD/DSL/Pipeline/Lowering.lean test/DSL/Pipeline/RouteWeaveTest.lean
git commit -m "refactor(route): extract pure routeCore/buildStep for provability"
```

---

## Phase 1 — Faithful per-tensor weaves (fixes conjunct 2 by construction)

Make every wire's type derive from a single source. Build `nameToType : String → List AxisP` once (the produced tensor's retained axes, in producer order); the producer's output weave and every consumer's input weave for that tensor both have `targetAxes = nameToType[nm]`. For external reads, derive from the read factor's own axes. Crucially the input weave's `.fixed` slots are listed in **producer order** (= `nameToType[nm]`), decoupled from the consumer's degree order, with `reindexings` rows reordered to match — so a transposed read cannot break the type match.

### Task 1.1: Build `nameToType` and faithful weaves in `buildStep`

**Files:**
- Modify: `LeanNCD/DSL/Pipeline/Lowering.lean` (`buildStep`, `routeCore`)
- Test: `test/DSL/Pipeline/RouteWeaveTest.lean`

**Interfaces:**
- Consumes: `buildStep`, `routeCore` from Phase 0.
- Produces: `tensorAxes (s : Stmt) : List AxisP` — a stmt's retained output axes as `AxisP` (name + `SizeExpr.var name`), in LHS order; this is what a producer publishes and a consumer must receive.
- Produces: `readPosAxis : IdxExpr → AxisP` — one representative axis per read position, for external reads.
- Produces: `nameToType : Std.HashMap String (List AxisP)` threaded through `routeCore`, built in a pre-pass: for each `sc`, for each `nm ∈ sc.writes`, `nameToType[nm] := tensorAxes (sc.repStmt.getD …)`.

- [ ] **Step 1: Write the failing faithfulness test**

In `test/DSL/Pipeline/RouteWeaveTest.lean` add:
```lean
-- Fixed-axis NAME list of a presentation weave (computable surrogate of weaveToArrayType shape).
def fixedNames (w : WeaveShapeP) : List (Option String) :=
  w.filterMap fun s => match s with | .fixed a => some a.name | .tiled => none

-- After the fix: the H-read input weave of step 1 must have targetAxes = H's output axes [i,k],
-- NOT the degenerate [i,j]. The H wire is routing[1][1] = internal 0 0.
#guard
  let tc := tl!{
    H[i,k] := W1[k,d] · X[i,d]
    Y[i,j] := relu(W2[j,k] · H[i,k])
  }
  fixedNames ((tc.steps.getD 1 default).inputWeaves.getD 1 [])
    == fixedNames ((tc.steps.getD 0 default).outputWeaves.getD 0 [])
```

- [ ] **Step 2: Run to verify it FAILS against current code**

Run: `lake env lean test/DSL/Pipeline/RouteWeaveTest.lean`
Expected: `#guard` failure — current input weave fixed-names are `[i,j]`, output `[i,k]`.

- [ ] **Step 3: Implement `nameToType` + faithful INPUT weaves (MINIMAL scope — see note)**

**Scope decision (2026-06-25):** descoped to *only* the change `WellFormed` needs. The reindexing-row
reorder and a fully faithful external-read arity are **deferred to the agreement task** — `WellFormed`
never inspects `reindexings`, and the change below makes conjunct 2 hold by construction.

Define a producer-published-axes helper and a per-position external helper:
```lean
-- a stmt's published (retained) output axes, in LHS order — what a producer emits and a consumer receives.
def tensorAxes (s : Stmt) : List AxisP :=
  s.lhsAxes.map (fun a => AxisP.mk (some a.name) (SizeExpr.var a.name))

-- one representative axis per READ POSITION, for external reads (no producer to publish a type).
def readPosAxis : IdxExpr → AxisP
  | .axis a      => AxisP.mk (some a.name) (SizeExpr.var a.name)
  | .shift a _   => AxisP.mk (some a.name) (SizeExpr.var a.name)
  | .scale _ a   => AxisP.mk (some a.name) (SizeExpr.var a.name)
  | .affine _ xs => match xs.head? with
                    | some (_, a) => AxisP.mk (some a.name) (SizeExpr.var a.name)
                    | none        => AxisP.mk none (SizeExpr.var "_")
  | .const _     => AxisP.mk none (SizeExpr.var "_")
```
In `routeCore`, build `nameToType : Std.HashMap String (List AxisP)` in a pre-pass over `sp.stmts`
(mirror the `nameToStep` fold; for each `sc`, for each `nm ∈ sc.writes`, insert
`nm ↦ tensorAxes (sc.repStmt.getD …)`). Pass it to `buildStep`. In `buildStep`, replace
```lean
let inputWeaves : List WeaveShapeP := readFactors.map (fun _ => mkWeave)
```
with a per-factor weave — **internal reads use the producer's published axes (producer order); external
reads use one `.fixed` per read position**:
```lean
let inputWeaves : List WeaveShapeP :=
  readFactors.map (fun rf =>
    match nameToType[rf.1]? with
    | some axs => axs.map (fun a => WeaveSlotP.fixed a)        -- internal: producer-published order
    | none     => rf.2.map (fun e => WeaveSlotP.fixed (readPosAxis e)))  -- external: one per position
```
**Leave the output weave (`outputWeaves := [ mkWeave ]`) and `reindexings` UNCHANGED.** The output
weave's `.fixed` slots are already `tensorAxes` (degree = `lhsAxes ++ contracted`, so `mkWeave`'s
fixed slots, in order, equal the LHS axes), so an internal consumer's input weave (`nameToType[nm]`)
and the producer's output weave have identical `targetAxes` — conjunct 2 is then definitional.

Record in the report (deferred to the agreement task): reindexing rows are NOT reordered to producer
order (only matters for transposed reads), and external-read weaves are position-representative (an
affine read like `X[i+p, 2*j+r]` publishes `[i, j]` by first-variable-per-position — a faithfulness
approximation, not load-bearing for `WellFormed`).

- [ ] **Step 4: Run faithfulness + example tests**

Run: `lake env lean test/DSL/Pipeline/RouteWeaveTest.lean` and `lake build`
Expected: the Step-1 guard PASSES; `test/DSL/CompileExamplesTest.lean` (which checks output-weave `.tiled` counts and `op` tags) still passes — verify the matmul `outputWeaves.head!.filter (· == .tiled)).length == 1` guard still holds (the summed `k` stays tiled in the output).

- [ ] **Step 5: Commit**

```bash
git add LeanNCD/DSL/Pipeline/Lowering.lean test/DSL/Pipeline/RouteWeaveTest.lean
git commit -m "fix(route): faithful per-tensor weaves via nameToType (conjunct 2 by construction)"
```

---

## Phase 2 — Topological sort in `schedule` (fixes conjunct 4)

Add a real producer-before-consumer sort so `nameToStep[readName] < consumerIndex` holds for every internal read, regardless of source order. The DCE filter already preserves relative order; the sort runs after it.

### Task 2.1: Implement and verify a topological sort over `ScanStmt`s

**Files:**
- Modify: `LeanNCD/DSL/Pipeline/Lowering.lean:104-117` (`schedule`)
- Test: `test/DSL/Pipeline/LoweringTest.lean`, `test/Eval/EvalExamplesTest.lean` (regression)

**Interfaces:**
- Produces: `topoSort (stmts : List ScanStmt) : List ScanStmt` — a Kahn-style sort: repeatedly emit a stmt all of whose internal-read producers are already emitted; an internal read is a read name that some stmt writes (external/unwritten names impose no constraint). Self-loops from `scan`/`scanPre` (a node reading its own write across iterations) are EXCLUDED from the dependency (a scan node's recurrence reads its own state — not a forward dep).
- Produces: `topoSort_spec : ∀ stmts, (topoSort stmts).Perm-of-input ∧ producers-precede-consumers` (stated precisely in Phase 4).

- [ ] **Step 1: Write the failing out-of-order test**

In `test/DSL/Pipeline/LoweringTest.lean` add a program whose source order is NOT topological (consumer before producer) and assert routing is topological after schedule+route:
```lean
-- Y reads H, but H is written by a LATER source statement. After topoSort, H's step must precede Y's.
#guard
  let tc := tl!{
    Y[i,j] := relu(W2[j,k] · H[i,k])
    H[i,k] := W1[k,d] · X[i,d]
  }
  -- find the step that reads an internal wire; that wire's producer index must be < the reader index
  tc.routing.zipIdx.all (fun (ws, i) =>
    ws.all (fun w => match w with | .internal j _ => decide (j < i) | .external _ => true))
```

- [ ] **Step 2: Run to verify it FAILS**

Run: `lake env lean test/DSL/Pipeline/LoweringTest.lean`
Expected: failure — with source order preserved, `H` is step 1, `Y` is step 0, so `routing[0]` contains `internal 1 0` (j=1 ≥ i=0).

- [ ] **Step 3: Implement `topoSort` and call it in `schedule`**

Add `topoSort` (Kahn's algorithm over the internal-producer dependency, excluding scan self-edges; `partial` is acceptable but prefer a fuel-bounded structural version `topoSortFuel (fuel := stmts.length)` so equation lemmas exist for Phase 4). In `schedule`, after computing `liveStmts`, set `let ordered := topoSort liveStmts` and return `stmts := ordered`. Update the `ScheduledProgram.stmts` doc-comment (currently says "reverse-topological order" — wrong) to "topological order: producers precede consumers."

- [ ] **Step 4: Run topo + evaluator regression**

Run: `lake env lean test/DSL/Pipeline/LoweringTest.lean`, then `lake build` exercising `test/Eval/EvalExamplesTest.lean`.
Expected: Step-1 guard passes; all 11 evaluator examples still compute identical numbers (the sort must be a no-op on already-ordered programs — verify the §12.1 examples produce the same step order as before by re-checking `CompileExamplesTest.lean`).

- [ ] **Step 5: Commit**

```bash
git add LeanNCD/DSL/Pipeline/Lowering.lean test/DSL/Pipeline/LoweringTest.lean
git commit -m "feat(schedule): topological sort so producers precede consumers (conjunct 4)"
```

---

## Phase 3 — Structural lemmas characterizing `routeCore`

Pure `List` lemmas (no `EStateM`) that turn `routeCore sp = .ok (steps, routing)` into per-index facts. These are the bridge the conjunct proofs stand on.

### Task 3.1: Length and per-index characterization

**Files:**
- Create: `LeanNCD/DSL/Pipeline/RouteSpec.lean`
- Modify: `LeanNCD/DSL/Pipeline/Lowering.lean` (import RouteSpec where needed, or keep RouteSpec downstream)

**Interfaces:**
- Produces:
  - `routeCore_steps_length : routeCore sp = .ok (steps, routing) → steps.length = sp.stmts.length` (and same for `routing`).
  - `routeCore_getD : routeCore sp = .ok (steps, routing) → i < sp.stmts.length → (buildStep nm ext (sp.stmts.getD i d)) = .ok (steps.getD i default, routing.getD i [])` — each output index is exactly `buildStep` of the corresponding stmt.
  - `buildStep_outputWeaves_length_one : buildStep nm ext sc = .ok (b, w) → b.outputWeaves.length = 1` (direct from the literal `outputWeaves := [ outW ]`).

- [ ] **Step 1: State the lemmas with `sorry` and build**

Write the three signatures above with `:= sorry` in `RouteSpec.lean`; `lake build` to confirm they elaborate (types are well-formed). This is the "failing test."

- [ ] **Step 2: Prove `routeCore_steps_length` / `routeCore_getD`**

Strategy: `routeCore`'s `built ← sp.stmts.mapM (buildStep …)` — use Mathlib's `List.mapM` characterization (`List.mapM_eq_ok`/`List.forall₂` on `Except`), or prove a local helper `mapM_ok_getD`. Then `steps = built.map (·.1)`, so `steps.length = built.length = sp.stmts.length` and `steps.getD i = (built.getD i).1`. Verify each with `lean_goal` while iterating.

- [ ] **Step 3: Prove `buildStep_outputWeaves_length_one`**

Strategy: unfold `buildStep`; `outputWeaves := [ outW ]` is literal, so after the `do`-bind the result `b` always has `b.outputWeaves = [outW]`; `simp`/`rfl` on `.length`.

- [ ] **Step 4: Verify sorry-free**

Run: `lake build`; then `#print axioms routeCore_getD` (and the others) — expect no `sorryAx`.

- [ ] **Step 5: Commit**

```bash
git add LeanNCD/DSL/Pipeline/RouteSpec.lean
git commit -m "feat(routespec): structural characterization lemmas for routeCore output"
```

### Task 3.2: `nameToType` ⊳ weave-consistency lemma (the conjunct-2 engine)

**Files:**
- Modify: `LeanNCD/DSL/Pipeline/RouteSpec.lean`

**Interfaces:**
- Produces:
  - `producer_output_targetAxes : routeCore sp = .ok (steps, routing) → nameToStep nm = some j → (steps.getD j default).outputWeaves.getD 0 []` has `targetAxes`-names `= nameToType nm` (the producer publishes exactly its `nameToType` entry).
  - `consumer_input_targetAxes : routeCore sp = .ok (steps, routing) → routing.getD i [] |>.getD p (.external 0) = .internal j 0 → (steps.getD i default).inputWeaves.getD p []` has `targetAxes`-names `= nameToType (readName)` AND `readName`'s producer is `j`, so `nameToType readName = nameToType (producerWrite j)`.

- [ ] **Step 1: State with `sorry`, build.** (failing test)

- [ ] **Step 2: Prove.** Strategy: both sides are built in `buildStep` from `nameToType[nm]` via the same `readPubAxes`/`tensorAxes` path (Phase 1), so the `.fixed`-slot name lists are equal by `rfl`/`simp` after unfolding `buildStep`. The wire `.internal j 0` arises only when `nameToStep[readName] = some j` (the `route` wiring match), and `nameToStep`/`nameToType` are built from the same `writes` fold, so `nameToType readName` is the producer step's published type.

- [ ] **Step 3: Verify sorry-free + commit.**

```bash
git add LeanNCD/DSL/Pipeline/RouteSpec.lean
git commit -m "feat(routespec): producer/consumer weave-consistency lemmas"
```

---

## Phase 4 — Prove the four conjuncts and assemble the theorem

Each conjunct becomes a lemma about `tc = route's output`; `compile_wellFormed` threads the monad and conjoins them. Order: easiest first (3 → 1 → 4 → 2).

### Task 4.1: `compile`-to-`route` plumbing lemma

**Files:**
- Modify: `LeanNCD/Bridge/Agreement.lean`

**Interfaces:**
- Produces: `compile_eq_route : (compile p).run s = .ok tc s' → ∃ sp s₀, (schedule …).run … = .ok sp s₀ ∧ route sp |>.run … = .ok tc … ∧ routeCore sp = .ok (tc.steps, tc.routing) ∧ tc.nExternal = sp.extNames.card`.

- [ ] **Step 1: State with `sorry`, build.**
- [ ] **Step 2: Prove.** Strategy: `compile` is a `do`-bind chain ending in `route g`. Use `EStateM.run_bind`/`EStateM.run_pure` simp lemmas to peel binds; `route`'s wrapper makes `routeCore sp = .ok (tc.steps, tc.routing)` from `route sp = .ok tc`. This isolates ALL further reasoning to `routeCore` + `sp` (a `ScheduledProgram` from `schedule`), so no later proof touches `EStateM` again.
- [ ] **Step 3: sorry-free + commit.**

### Task 4.2: Conjunct 3 (single output)

**Files:** Modify `LeanNCD/Bridge/Agreement.lean`
**Interfaces:** Produces `wf_singleOutput : routeCore sp = .ok (tc.steps, tc.routing) → ∀ i, i < tc.steps.length → (tc.steps.getD i default).outputWeaves.length = 1`.

- [ ] **Step 1: State with `sorry`, build.**
- [ ] **Step 2: Prove** via `routeCore_getD` + `buildStep_outputWeaves_length_one` (Task 3.1).
- [ ] **Step 3: sorry-free + commit.**

### Task 4.3: Conjunct 1 (`wellFormedDom = true`)

**Files:** Modify `LeanNCD/Bridge/Agreement.lean`
**Interfaces:** Produces `wf_dom : … → tc.wellFormedDom = true`.

- [ ] **Step 1: State with `sorry`, build.**
- [ ] **Step 2: Prove.** Strategy: `wellFormedDom` requires every slot `< nExternal` referenced and all consuming ports agree on `weaveRank`. After Phase 1, an external read's input weave fixed-slots = the read factor's own axes, so two ports consuming the same external `nm` share `nameToType`-absent ⇒ same `idxAxes nm` ⇒ same rank. Referencedness: `nExternal = extNames.card` and `extIndex` numbers exactly the read names `∈ extNames` (`extNames ⊆ reads`, established by `resolveDecls`: `extNames := reads \ produced`). Needs a lemma `extNames_subset_reads` about `resolveDecls`/`schedule`; if `resolveDecls`/`liveFix` reasoning is intractable (the `partial liveFix`), restate `wellFormedDom` reliance as: numbered ext indices are `< extIndex.size ≤ extNames.card`. Verify each branch with `decide` on examples first.
- [ ] **Step 3: sorry-free + commit.**

### Task 4.4: Conjunct 4 (topological — reads ⊆ pool)

**Files:** Modify `LeanNCD/Bridge/Agreement.lean`
**Interfaces:** Produces `wf_topo : … → ∀ i, i < tc.steps.length → ∀ w ∈ tc.routing.getD i [], w ∈ tc.poolAt i`.

- [ ] **Step 1: State with `sorry`, build.**
- [ ] **Step 2: Prove.** Strategy: a wire is `.external k` with `k < nExternal` (from `extIndex` numbering bound, reuse Task 4.3) ⇒ `∈ poolAt i`; or `.internal j 0` with `j = nameToStep[readName]`. `topoSort` (Phase 2) guarantees `nameToStep[readName] < i` for every internal read of step `i` — this is `topoSort_spec`'s producers-precede-consumers, transported through `routeCore`'s `nameToStep`. `poolAt i = externals ++ {internal j 0 : j < i}`, so `j < i ⇒ internal j 0 ∈ poolAt i`. Prove `topoSort_spec` here (or as a Phase-2 lemma): for the fuel-bounded `topoSortFuel`, induct on fuel showing each emitted stmt's internal-read producers are already in the output prefix.
- [ ] **Step 3: sorry-free + commit.**

### Task 4.5: Conjunct 2 (producer ⊳ consumer type match)

**Files:** Modify `LeanNCD/Bridge/Agreement.lean`
**Interfaces:** Produces `wf_typeMatch : … → ∀ i, i < tc.steps.length → (tc.routing.getD i []).map tc.wireType = (tc.steps.getD i default).inputWeaves.map weaveToArrayType`.

- [ ] **Step 1: State with `sorry`, build.**
- [ ] **Step 2: Prove.** Strategy: pointwise (`List.map_congr` over the `routing[i]`/`inputWeaves[i]` pairing — equal length by `routeCore_getD`: both have one entry per read factor). For port `p` with wire `w`:
  - `w = .external k`: `tc.wireType (.external k) = weaveToArrayType (inputWeave at externalPort k)`. After Phase 1, `externalPort k`'s first consuming port publishes the same `readPubAxes`, and conjunct-1's rank-agreement plus Phase-1's name-derivation give equal `weaveToArrayType`. (For the FIRST consuming port this is `rfl`; for others, equal `targetAxes` from `wellFormedDom`.)
  - `w = .internal j 0`: `tc.wireType (.internal j 0) = weaveToArrayType (steps[j].outputWeaves[0])`. By `producer_output_targetAxes` (Task 3.2) this has `targetAxes`-names `= nameToType (producerWrite j)`; by `consumer_input_targetAxes`, `inputWeaves[i][p]` has `targetAxes`-names `= nameToType readName` and `producerWrite j = readName`. Since `weaveToArrayType` depends only on `targetAxes` (and `dtype := .reals` always), equal `targetAxes`-name-and-size lists ⇒ equal `ArrayType`. Sizes are `SizeExpr.var name` on both sides ⇒ equal `toNumeric`. Close with `congr`/`rfl`.
  This is the hard lemma; budget the most time here. If a transpose edge case surfaces (producer axis order ≠ consumer read order), Phase 1's producer-order weave construction must already have handled it — confirm with a transposed-read `#guard` example (`Y[k,i] := H[i,k]`).
- [ ] **Step 3: sorry-free + commit.**

### Task 4.6: Assemble `compile_wellFormed`

**Files:** Modify `LeanNCD/Bridge/Agreement.lean`
**Interfaces:** Produces the Global-Constraints theorem.

- [ ] **Step 1: Write the theorem with `sorry`, build.**
- [ ] **Step 2: Prove** by `compile_eq_route` (Task 4.1) then `⟨wf_dom, wf_typeMatch, wf_singleOutput, wf_topo⟩` (matching `WellFormed`'s conjunct order in [`Realize.lean:171`](../../../LeanNCD/Bridge/Realize.lean#L171)).
- [ ] **Step 3: Verify sorry-free**

Run: `lake build`; `#print axioms compile_wellFormed` — expect `[propext, Classical.choice, Quot.sound]`, no `sorryAx`.

- [ ] **Step 4: Add the corollary that `realize` applies on real input**

```lean
/-- Every compiled program crosses the bridge: the formal morphism exists. -/
noncomputable def realizeCompiled (p : TLProgram) (s : Nat) (tc : ThreadedComposed) (s' : Nat)
    (h : (TLProgram.compile p).run s = .ok tc s') : Σ (dom cod : BrObj), BrMorph dom cod :=
  realize tc (compile_wellFormed p s tc s' h)
```

- [ ] **Step 5: Commit**

```bash
git add LeanNCD/Bridge/Agreement.lean
git commit -m "feat(agreement): prove compile_wellFormed — realize applies to every compiled program"
```

### Task 4.7: Update docs

**Files:** Modify `SORRY_INVENTORY.md`, `realize.md`

- [ ] **Step 1:** In `realize.md` §3.2 / §7, correct the stale "schedule topologically sorts" claim to reflect that the sort was ADDED in this work; move the §6a `WellFormed for compiled tc` item to "done"; note the conjunct-2 `route` weave fix.
- [ ] **Step 2:** In `SORRY_INVENTORY.md` Milestone E2b, record `compile_wellFormed` as proved and the `route` faithful-weave change.
- [ ] **Step 3: Commit**

```bash
git add SORRY_INVENTORY.md realize.md
git commit -m "docs: record compile_wellFormed + route weave fix + topo sort"
```

---

## De-risk result (2026-06-25)

A throwaway spike hand-built the faithful 2-layer `tc` exactly as Phase 1's fixed `route` would emit it (per-tensor weaves in producer order; output weave `tiled`-on-contracted) and **proved `WellFormed` sorry-free** (`#print axioms` ⇒ `[propext, Classical.choice, Quot.sound]`). Findings that de-risk Task 4.5:

- **Conjunct 2 closes by `rfl` per concrete index.** `weaveToArrayType` filters out `.tiled` slots, so a producer output weave `[fixed i, fixed k, tiled]` and a consumer input weave `[fixed i, fixed k]` are **definitionally equal** `ArrayType`s. As long as Phase 1 builds both the producer's output `.fixed` slots and the consumer's input `.fixed` slots from the same `nameToType[nm]` list, the type match is definitional — `interval_cases i <;> rfl`. The central plan bet is validated.
- Conjuncts 1, 3 by `decide`; conjunct 4 by `interval_cases i <;> (intro w hw; fin_cases hw <;> simp [poolAt])`.

**What the spike did NOT cover (still open):** (a) the spike used a hand-built tc already in faithful form — Phase 0/1 must make `route` actually emit this shape; (b) the ∀ p monadic/structural machinery (Phases 0, 3, 4) is untested.

**Transpose canary — RESOLVED (2026-06-25).** A second spike built a `tc` where the consumer reads `H` transposed (producer publishes `[i,k]`, consumer reads `H[k,i]`), and proved both directions:

- **Producer-order weave** (consumer input weave = `nameToType[H]` = `[i,k]`, transpose deferred to the ignored `reindexings`): `WellFormed` holds, conjunct 2 still `rfl`, sorry-free.
- **Read-order weave** (consumer input weave = `[k,i]`): conjunct 2 is **provably false** (`aI ≠ aK`).

**Locked design decision for Task 1.1:** `route` MUST build each internal read's input weave from `nameToType[nm]` (producer order), never from the read-expression order. The transpose lives only in `reindexings` (which `WellFormed` does not inspect). The plan's `readPubAxes` already does this — confirmed correct.

## Self-Review Notes

- **Spec coverage:** conjunct 2 → Phase 1 + Task 4.5; conjunct 4 → Phase 2 + Task 4.4; conjuncts 1,3 → Tasks 4.3/4.2; monad plumbing → Task 4.1; final theorem → Task 4.6. All four conjuncts of [`WellFormed`](../../../LeanNCD/Bridge/Realize.lean#L171) covered.
- **Risk hotspots (flagged, not hidden):**
  1. **Task 4.5** (conjunct 2) is the genuine research-grade proof — the type match across producer/consumer. Phase 1's `nameToType` design is what makes it `rfl`-close-able; if the producer/consumer weaves are NOT literally equal-by-construction, this proof balloons.
  2. **`partial liveFix`** has no equation lemmas. Tasks 4.3/4.4 must avoid reasoning *inside* it — they rely only on `filter` preserving order/membership and on `topoSort` running *after* it. If `extNames ⊆ reads` genuinely needs `resolveDecls` internals, that is a sub-risk to surface early.
  3. **Task 2.1 topo sort** must be a no-op on already-ordered §12.1 programs or the evaluator regression (Phase 2 Step 4) fails. Prefer a *stable* sort.
- **Open assumption to confirm in Task 1.1:** that `tensorAxes` (producer LHS-order axes) and `readPubAxes` (consumer's view) can be made identical lists. The transposed-read `#guard` (`Y[k,i] := H[i,k]`) is the canary.

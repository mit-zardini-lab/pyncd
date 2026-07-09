# Multi-axis (n-D) Scans Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support n-D recurrences (e.g. `G[r+1,c+1] := G[r,c] + A[r,c]`) so grid-DP / PixelRNN scans evaluate correctly instead of silently collapsing to a 1-D scan (fixes KG-2dscan).

**Architecture:** Generalize the single-iteration-slot assumption to a *list* of iteration axes. `finalizeScans` is the gate — while it emits single-axis nodes, everything downstream reproduces current behavior. Task 1 widens the types mechanically (behavior-preserving); Task 2 flips `finalizeScans`/`evalScan` to genuine multi-axis (driven by RC6); Task 3 verifies the Br/execution layer (already arity-agnostic); Task 4 adds a 3-D generality test; Task 5 syncs docs.

**Tech Stack:** Lean 4, `lake`, LSpec test harness. Build/test via `lake build <target>`.

## Global Constraints

- Boundary cells (any advancing index = 0) are **zero-default**: the step writes only fully-advanced cells; explicit base statements override specific boundary slices. (Spec decision 2.)
- Scope is **general n-D**, not 2-D-special-cased. (Spec decision 1.)
- **No regression** to 1-D scans: RC1, RC2, RC5, RC7, SS1–SS3, CM4 must stay green.
- Multi-axis scans are **not** parallel-prefix (ScanAffine); force `isAffine = false` for them.
- Work on branch `multi-axis-scans` (already created). Commit after each task.
- Build the whole suite with `cd leanncd && lake build Tests`; target one file e.g. `lake build Eval.Portfolio.RecurrenceTest`.

---

### Task 1: Widen scan types (mechanical, behavior-preserving)

Change the scan carrier types from a single axis to a list, and update **every** consumer to compile while preserving exact current behavior via first-element (`.head?`) access. No behavior changes; the full suite stays green and RC6 still pins its current wrong output.

**Files:**
- Modify: `LeanNCD/DSL/Pipeline/Types.lean` (`ScanStmt.scan` type)
- Modify: `LeanNCD/DSL/Pipeline/Structural.lean` (`Stmt.iterInfo`; `finalizeScans` node build)
- Modify: `LeanNCD/Eval/Scan.lean` (`evalScan` destructure)
- Modify: `LeanNCD/DSL/Pipeline/Lowering.lean` (~10 `.scan` matches)
- Test: full suite (regression only — this is a refactor)

**Interfaces:**
- Produces: `ScanStmt.scan : String → List AxisSpec → List Stmt → List Stmt → Bool → ScanStmt`
- Produces: `Stmt.iterInfo : Stmt → List (UID × AxisSpec × Bool × Nat)` (each entry: axis uid, axis, isRecur, slot-position)

- [ ] **Step 1: Change `ScanStmt.scan` to carry an axis list**

In `LeanNCD/DSL/Pipeline/Types.lean`, change the constructor:

```lean
  | scan    : String → List AxisSpec → List Stmt → List Stmt → Bool → ScanStmt  -- axis list, final Bool = isAffine
```

- [ ] **Step 2: Generalize `Stmt.iterInfo` to return all iteration slots with positions**

In `LeanNCD/DSL/Pipeline/Structural.lean`, replace `Stmt.iterInfo`:

```lean
/-- All iteration slots of a stmt: `(uid, axis, isRecur, slot-position)` for each `iterAt`/`iterNext`.
    A 1-D scan yields a single-element list; multi-axis scans yield one entry per advancing slot. -/
def Stmt.iterInfo (s : Stmt) : List (UID × AxisSpec × Bool × Nat) :=
  s.slots.zipIdx.filterMap (fun (sl, i) => match sl with
    | .iterAt a _ => some (a.uid, a, false, i)
    | .iterNext a => some (a.uid, a, true, i)
    | _           => none)
```

(If `List.zipIdx` is unavailable in this toolchain, use `(s.slots.enum).filterMap (fun (i, sl) => …)`.)

- [ ] **Step 3: Adapt every `iterInfo` consumer in `finalizeScans` to first-element behavior**

`iterInfo` was `Option`; now it's `List`. Preserve behavior by reading the first element everywhere it was used. In `finalizeScans` ([Structural.lean:381+](../../LeanNCD/DSL/Pipeline/Structural.lean)):

- `recurAxisFor`: `match s.iterInfo with | some (_, a, true) => …` becomes `match s.iterInfo.head? with | some (_, a, true, _) => …`
- `stmts0` map: `match s.iterInfo with | some (_, _, false) => …` becomes `match s.iterInfo.head? with | some (_, _, false, _) => …`
- `dep` seeding: `match s.iterInfo with | some (u, _, _) => …` becomes `match s.iterInfo.head? with | some (u, _, _, _) => …`
- `iterStmts := nonPre.filter (fun s => s.iterInfo.isSome)` becomes `… (fun s => !s.iterInfo.isEmpty)`
- `uids := (iterStmts.filterMap (fun s => s.iterInfo.map (·.1)))` becomes `… (fun s => s.iterInfo.head?.map (·.1))`
- `isBase`/`isState`: `s.iterInfo.map (fun t => t.1 == u && t.2.2 == false)` becomes `s.iterInfo.head?.map (fun t => t.1 == u && t.2.2.1 == false)` (note the extra `.1` — the tuple gained a 4th field, so `isRecur` is now `t.2.2.1`).
- `axis` binding: `s.iterInfo.map (·.2.1)` stays valid (axis is still `.2.1`); the guard `s.iterInfo.map (fun t => t.1 == u)` becomes `s.iterInfo.head?.map (fun t => t.1 == u)`.
- The final node: `ScanStmt.scan repName axis baseStmts recurStmts isAffine` becomes `ScanStmt.scan repName [axis] baseStmts recurStmts isAffine` (wrap the single axis in a list).

- [ ] **Step 4: Adapt `Lowering.lean` `.scan` matches to the list type**

Every `.scan` pattern in `LeanNCD/DSL/Pipeline/Lowering.lean` currently binds the axis as `ax` or `_`. All uses are `_` except `splitScan` (line 52) which forwards `ax`. They already ignore the axis, so just keep the binder name; the type now infers as `List AxisSpec`. Concretely `splitScan`:

```lean
  | .scan nm ax base recur isAff =>
      let base'  ← base.flatMapM splitStmt
      let recur' ← recur.flatMapM splitStmt
      return [ ScanStmt.scan nm ax base' recur' isAff ]   -- ax : List AxisSpec, forwarded unchanged
```

The remaining `.scan _ _ b r _` / `.scan ..` / `.scan _ _ _ _ isAff` matches need no change.

- [ ] **Step 5: Adapt `evalScan` to the list type, first-element behavior**

In `LeanNCD/Eval/Scan.lean`, `evalScan` destructures `.scan _ ax base recur _`. Preserve current single-axis behavior by using the first axis:

```lean
  | .scan _ axes base recur _ => do
      let ax ← match axes.head? with
        | some a => pure a
        | none   => .error "evalScan: scan node has no iteration axis"
      -- … existing body unchanged, using `ax` …
```

- [ ] **Step 6: Build the whole suite; verify green with RC6 still wrong**

Run: `cd leanncd && lake build Tests`
Expected: `Build completed successfully`. In the RecurrenceTest output, `✓ ∃: RC6 2d-scan (KG-2dscan, current wrong)` still passes (RC6 still pins `[[0,1],[0,1]]` — behavior unchanged). No errors from Lowering/AcsetCodec.

- [ ] **Step 7: Commit**

```bash
cd leanncd
git add LeanNCD/DSL/Pipeline/Types.lean LeanNCD/DSL/Pipeline/Structural.lean LeanNCD/Eval/Scan.lean LeanNCD/DSL/Pipeline/Lowering.lean
git commit -m "refactor(scan): widen ScanStmt/iterInfo to carry an axis list (behavior-preserving)"
```

---

### Task 2: Multi-axis compile structuring + evaluation (the core fix)

Flip `finalizeScans` to recognize all advancing axes and group them into one node, with positional base-axis recovery; and generalize `evalScan` to a nested loop with zero-default boundaries. Driven by flipping RC6 to its correct output.

**Files:**
- Modify: `LeanNCD/DSL/Pipeline/Structural.lean` (`adoptBaseIterAxis` → positional; `readsIterAhead` per-axis; `finalizeScans` grouping + node axis list + isAffine)
- Modify: `LeanNCD/Eval/Scan.lean` (`iterSlotPos` → list; `writeSliceAt` → multi-position; `evalScan` nested loop)
- Test: `test/Eval/Portfolio/RecurrenceTest.lean` (RC6 flip)

**Interfaces:**
- Consumes: `Stmt.iterInfo : Stmt → List (UID × AxisSpec × Bool × Nat)` (Task 1)
- Produces: `ScanStmt.scan` nodes whose axis list has one entry per advancing axis (in slot order)

- [ ] **Step 1: Flip RC6 to the correct output (failing test)**

In `test/Eval/Portfolio/RecurrenceTest.lean`, change RC6:

```lean
-- RC6  2-D / nested recurrence (grid-DP / PixelRNN). Base G=0, A=ones(2×2).
--   Only the fully-advanced cell G[1,1] = G[0,0]+A[0,0] = 1 is written; boundary cells (r=0 or
--   c=0) keep their zero-default. Correct grid-DP ⇒ [[0,0],[0,1]]. (KG-2dscan fixed 2026-07-08.)
test "RC6 2d-scan (KG-2dscan, fixed)"
    (evalEqB (tlprog!{ axis r : ℕ = 2, c : ℕ = 2
            G[r, 0]       := Z[r]
            G[r +1, c +1] := G[r, c] + A[r, c] })
      (HashMap.ofList [("Z", tl [2] [0,0]), ("A", tl [2,2] [1,1,1,1])])
      "G" (tl [2,2] [0,0, 0,1])) $
```

(Keep RC7 after it, unchanged.)

- [ ] **Step 2: Build RecurrenceTest; verify RC6 fails**

Run: `cd leanncd && lake build Eval.Portfolio.RecurrenceTest`
Expected: RC6 FAILS — the test reports the current wrong `[[0,1],[0,1]]` against expected `[[0,0],[0,1]]` (an LSpec `✗` on `RC6 2d-scan (KG-2dscan, fixed)`).

- [ ] **Step 3: Positional base-axis recovery in `finalizeScans`**

In `LeanNCD/DSL/Pipeline/Structural.lean`, replace `Stmt.adoptBaseIterAxis` with a positional version that names each base `iterAt` slot from the step's slot at the same position. Add a helper that maps slot-position → step iterNext axis:

```lean
/-- Map slot-position → iteration axis, read from a step (`iterNext`) stmt. -/
def Stmt.stepAxisAt (step : Stmt) : Nat → Option AxisSpec := fun p =>
  step.iterInfo.findSome? (fun (_, a, isRec, i) => if isRec && i == p then some a else none)

/-- Rewrite a base stmt's `iterAt` slots so each adopts the step's iteration axis at the SAME
    slot position (the E1 parser leaves base iter-axes as placeholders). A base `iterAt` whose
    position has no matching step `iterNext` axis is left as-is. -/
def Stmt.adoptBaseIterAxes (step : Stmt) : Stmt → Stmt
  | .assign nm ls r    => .assign  nm (ls.zipIdx.map (fun (sl, p) => match sl with
      | .iterAt _ n => match step.stepAxisAt p with | some a => .iterAt a n | none => sl
      | sl'         => sl')) r
  | .scatter nm ls r o => .scatter nm (ls.zipIdx.map (fun (sl, p) => match sl with
      | .iterAt _ n => match step.stepAxisAt p with | some a => .iterAt a n | none => sl
      | sl'         => sl')) r o
  | s@(.recurMorphism _ _ _) => s
```

Then in `finalizeScans`, replace the base-axis pre-pass. The old code found a single `recurAxisFor` name and called `adoptBaseIterAxis`. New: find the matching step stmt (same name, has an `iterNext`) and adopt positionally.

```lean
  -- Recover each base case's iteration axes from the matching (same-name) step, by slot position.
  let stepFor (nm : String) : Option Stmt :=
    lp.stmts.find? (fun s => s.lhsName == nm && s.iterInfo.any (fun t => t.2.2.1 == true))
  let stmts0 := lp.stmts.map (fun s =>
    -- a base stmt (has iterAt, no iterNext) adopts its step's axes positionally
    if s.iterInfo.any (fun t => t.2.2.1 == false) && !s.iterInfo.any (fun t => t.2.2.1 == true) then
      match stepFor s.lhsName with
      | some step => s.adoptBaseIterAxes step
      | none      => s
    else s)
  let lp := { lp with stmts := stmts0 }
```

- [ ] **Step 4: Component grouping + multi-axis node in `finalizeScans`**

Replace the per-UID node loop with connected-component grouping over iteration axes. Two iter-stmts are coupled iff their axis-sets share a UID. Build components, then one `ScanStmt.scan` per component carrying its axis list (in step-slot order).

```lean
  -- Connected components over iteration-axis UIDs: statements sharing any axis are one scan node.
  let iterStmts := nonPre.filter (fun s => !s.iterInfo.isEmpty)
  -- seed: each stmt's axis-uid set
  let axSet : Stmt → List UID := fun s => (s.iterInfo.map (·.1)).eraseDups
  -- union-find via repeated merge into components (bounded by #stmts passes)
  let mut comps : List (List UID) := (iterStmts.map axSet)
  for _ in List.range (iterStmts.length + 1) do
    comps := comps.foldl (fun acc c =>
      match acc.find? (fun d => (d.inter c).length > 0) with
      | some d => (acc.erase d) ++ [(d ++ c).eraseDups]
      | none   => acc ++ [c]) []
  -- for each component, gather its stmts and axis list (in step slot order)
  let mut nodes : List ScanStmt := []
  for comp in comps do
    let inComp : Stmt → Bool := fun s => (axSet s).any (fun u => comp.contains u)
    let stepStmts  := nonPre.filter (fun s => inComp s && s.iterInfo.any (·.2.2.1))
    let baseStmts  := nonPre.filter (fun s => inComp s && s.iterInfo.all (fun t => !t.2.2.1) && !s.iterInfo.isEmpty)
    -- axis list in slot order, taken from a representative step
    let axes : List AxisSpec := (stepStmts.head?.map (fun st =>
      (st.iterInfo.filter (·.2.2.1)).mergeSort (fun a b => a.2.2.2 ≤ b.2.2.2) |>.map (·.2.1))).getD []
    -- per-step intermediates (non-iter stmts whose dep-set ⊆ comp)
    let isInter : Stmt → Bool := fun s => s.iterInfo.isEmpty &&
      (let d := dep.getD s.lhsName []; !d.isEmpty && d.all (fun u => comp.contains u))
    -- causality: no look-ahead on ANY axis in this component
    for r in stepStmts do
      unless baseStmts.any (fun b => b.lhsName == r.lhsName) do
        throw (CompileError.missingBaseCase r.lhsName)
      for u in comp do
        if readsIterAhead r u then throw (CompileError.causalityViolation r.lhsName)
    let recurStmts := nonPre.filter (fun s => isInter s || (inComp s && s.iterInfo.any (·.2.2.1)))
    let repName := ((recurStmts.head?.orElse (fun _ => baseStmts.head?)).map Stmt.lhsName).getD ""
    -- ScanAffine only for a genuine 1-axis component (multi-axis ⇒ sequential)
    let isAffine : Bool := axes.length ≤ 1 && recurStmts.all (fun s => Stmt.nonlinOf s == Nonlin.identity)
    nodes := nodes ++ [ ScanStmt.scan repName axes baseStmts recurStmts isAffine ]
```

(`List.mergeSort` orders the axes by slot position `t.2.2.2`. If the exact `mergeSort` comparator API differs in this toolchain, sort by `(·.2.2.2)` using the available list-sort. Delete the now-unused old `uids`/per-`u` loop and old `isBase`/`isState`/`axis` bindings.)

- [ ] **Step 5: Generalize `iterSlotPos` to a list**

In `LeanNCD/Eval/Scan.lean`:

```lean
/-- All iteration slots `(uid, position)` of a base/recur stmt's slot list, in slot order. -/
def iterSlotPositions (slots : List LHSSlot) : List (UID × Nat) :=
  slots.zipIdx.filterMap (fun (sl, i) => match sl with
    | .iterAt a _ => some (a.uid, i)
    | .iterNext a => some (a.uid, i)
    | _           => none)
```

- [ ] **Step 6: Generalize `writeSliceAt` to multiple iteration positions**

Replace `writeSliceAt` so it inserts the current iteration index at every iteration position (positions must be inserted low-to-high so earlier insertions don't shift later ones):

```lean
/-- Write a non-iter `slice` into the full state tensor `out`, given the iteration
    `(position, index)` pairs (one per advancing axis). The slice's coords are the out-coords
    with all iteration positions removed; we rebuild the full coord by inserting each iteration
    index at its position (ascending position order). -/
def writeSliceAtMulti (out : DenseTensor) (iters : List (Nat × Nat)) (slice : DenseTensor) : DenseTensor :=
  let sorted := iters.mergeSort (fun a b => a.1 ≤ b.1)   -- ascending by position
  (DenseTensor.allCoords slice.shape).foldl (fun cur scoord =>
    let ocoord := sorted.foldl (fun acc (pos, idx) => acc.insertIdx pos idx) scoord
    cur.set! ocoord (slice.get! scoord)) out
```

(`List.insertIdx pos idx` inserts `idx` at position `pos`. Ascending position order makes each insertion land correctly relative to the already-inserted lower positions.)

- [ ] **Step 7: Rewrite `evalScan` as a nested (cartesian) loop over all axes**

Replace the body of the `.scan` case. Key changes: `Ls` = list of per-axis lengths; state shape unchanged (`stateShape` already gives `L` per iter slot); allocate; fill base at the boundary; iterate the cartesian product of `[0 … L_a − 2]` for each axis; seed all axes; write at `(l_a + 1)` per position.

```lean
  | .scan _ axes base recur _ => do
      if axes.isEmpty then .error "evalScan: scan node has no iteration axis" else
      let axUids := axes.map (·.uid)
      let Ls     := axUids.map (fun u => (sizes[u]?).getD 0)
      let stateNames := (base.map stmtName).eraseDups
      -- 1. allocate each state tensor (zeros) and record its iteration slot positions
      let mut work := env
      let mut iterPosOf : HashMap String (List (UID × Nat)) := {}
      for s in base do
        match s with
        | .assign nm slots _ =>
            work := work.insert nm (DenseTensor.zeros (stateShape sizes slots (Ls.headD 0)))
            iterPosOf := iterPosOf.insert nm (iterSlotPositions slots)
        | _ => throw "evalScan: base stmts must be assigns"
      -- 2. fill boundaries from base stmts (each base pins a subset of axes to 0)
      for s in base do
        let seed : HashMap UID Int := (s.slots.zipIdx.filterMap (fun (sl, _) => match sl with
            | .iterAt a n => some (a.uid, n) | _ => none)).foldl (·.insert · |>.uncurry) {}
        let (nm, slice) ← evalStmtSliceSeeded work sizes seed s
        work := work.insert nm (writeBaseSlice ((work[nm]?).getD (DenseTensor.zeros [])) (iterPosOf.getD nm []) seed slice)
      -- 3. nested loop: for each tuple in ∏ [0 … L_a − 2], run the recur stmts
      let ranges := Ls.map (fun L => List.range (L - 1))
      for tup in cartesianList ranges do          -- tup : List Nat, one per axis, in axes order
        let seed : HashMap UID Int := (axUids.zip tup).foldl (fun m (u, v) => m.insert u (Int.ofNat v)) {}
        let mut stepEnv := work
        for s in recur do
          let (nm, slice) ← evalStmtSliceSeededAll stepEnv sizes seed s
          match iterPosOf[nm]? with
          | some poss =>
              -- state slice: write at (position, index+1) for each iteration axis
              let iters := poss.map (fun (u, p) => (p, ((seed[u]?).getD 0).toNat + 1))
              let updated := writeSliceAtMulti ((work[nm]?).getD (DenseTensor.zeros [])) iters slice
              work := work.insert nm updated
              stepEnv := stepEnv.insert nm updated
          | none => stepEnv := stepEnv.insert nm slice
      return stateNames.filterMap (fun nm => (work[nm]?).map (fun t => (nm, t)))
```

Supporting helpers to add near the top of `Scan.lean`:

```lean
/-- Cartesian product of a list of index ranges → list of tuples (each tuple a `List Nat`). -/
def cartesianList : List (List Nat) → List (List Nat)
  | []      => [[]]
  | r :: rs => (cartesianList rs).flatMap (fun tail => r.map (fun x => x :: tail))
```

`evalStmtSliceSeededAll` generalizes `evalStmtSlice` to seed a *set* of iteration UIDs (not one). Refactor `evalStmtSlice` to take a `seed : HashMap UID Int` and drop all seeded UIDs from the slice axes (the existing `· != iterUID` filter becomes `!seed.contains ·`). `evalStmtSliceSeeded`/`writeBaseSlice` write a base slice into the boundary given its pinned-axis seed (fill along the base's free axes at the pinned index). Keep the nonlin-position logic from the current `evalStmtSlice`, filtering slice axes by `!seed.contains`.

- [ ] **Step 8: Build RecurrenceTest; verify RC6 passes and single-axis unchanged**

Run: `cd leanncd && lake build Eval.Portfolio.RecurrenceTest`
Expected: all green — `✓ ∃: RC2 rnn`, `✓ ∃: RC5 maxreduce-in-scan (KG-scanagg, fixed)`, `✓ ∃: RC6 2d-scan (KG-2dscan, fixed)`, `✓ ∃: RC7 minreduce-in-scan`.

- [ ] **Step 9: Build the whole suite; verify no regression**

Run: `cd leanncd && lake build Tests`
Expected: `Build completed successfully`. 1-D scans elsewhere (SS1–3 in GenerativeTest, CM4 in ClassicalMLTest, RC1 in EvalExamplesTest) still green.

- [ ] **Step 10: Commit**

```bash
cd leanncd
git add LeanNCD/DSL/Pipeline/Structural.lean LeanNCD/Eval/Scan.lean test/Eval/Portfolio/RecurrenceTest.lean
git commit -m "feat(scan): evaluate multi-axis (n-D) recurrences correctly (KG-2dscan)"
```

---

### Task 3: Verify multi-axis lowering through the Br layer

The Br step builder is arity-agnostic (builds from the representative stmt's slots), so a correctly-structured multi-axis node should lower to a well-formed Br step. Add a compile-level test that proves it, and fix any Lowering issue it surfaces.

**Files:**
- Test: `test/Eval/Portfolio/RecurrenceTest.lean` (add a `run_cmd` compile check) or a new `test/DSL/MultiAxisScanCompileTest.lean`
- Modify (only if the test surfaces a gap): `LeanNCD/DSL/Pipeline/Lowering.lean`, `LeanNCD/Bridge/AcsetCodec.lean`

**Interfaces:**
- Consumes: `TLProgram.compile` producing a `BrProgram`; `BrOp.scan`; `brOpIdx`/`brOpOfIdx` (unchanged).

- [ ] **Step 1: Add a compile-level assertion for the 2-D scan (failing-or-passing test)**

Add to the top of `test/Eval/Portfolio/RecurrenceTest.lean` (after imports), a `run_cmd` that compiles the RC6 program and asserts a well-formed scan step. Adjust the accessor names to the actual `BrProgram`/`BrBaseP` fields (inspect via `#check`/`lean_file_outline` on `Target.lean` first):

```lean
run_cmd do
  match TLProgram.compile (tlprog!{ axis r : ℕ = 2, c : ℕ = 2
                                    G[r, 0]       := Z[r]
                                    G[r +1, c +1] := G[r, c] + A[r, c] }) |>.run 0 with
  | .ok prog _ =>
      -- exactly one scan step, op = .scan (not scanAffine — multi-axis is sequential),
      -- and its degree carries BOTH iteration axes (rank-2 output).
      let steps := prog.steps.filter (fun st => st.op == BrOp.scan)
      unless steps.length == 1 do throw (IO.userError s!"expected 1 scan step, got {steps.length}")
      let deg := (steps.head!).degree.length
      unless deg == 2 do throw (IO.userError s!"expected scan-step degree 2 (r,c), got {deg}")
  | .error e _ => throwError s!"multi-axis compile failed: {repr e}"
```

- [ ] **Step 2: Build and run the compile check**

Run: `cd leanncd && lake build Eval.Portfolio.RecurrenceTest`
Expected: PASS (no `run_cmd` error). If it fails with a wrong degree or a Lowering error, that is the gap to fix in Step 3; otherwise skip to Step 4.

- [ ] **Step 3: (Only if Step 2 failed) Fix the Lowering gap**

If the scan step's degree omits the second axis, check `retainedOutputSpecs` / `slotAxisIdx?` in `Lowering.lean` — both should already include every `iterNext` slot. If `AcsetCodec` roundtrip errors, confirm `brOpOfIdx (brOpIdx .scan) = .scan` still holds (`#eval` it). Re-run Step 2 until green.

- [ ] **Step 4: Commit**

```bash
cd leanncd
git add test/Eval/Portfolio/RecurrenceTest.lean
git commit -m "test(scan): verify 2-D scan lowers to a well-formed multi-axis Br step"
```

---

### Task 4: 3-D generality test

Prove the machinery is genuinely n-D (not 2-D-special-cased) with a 3-D nested scan.

**Files:**
- Test: `test/Eval/Portfolio/RecurrenceTest.lean` (add RC8)

- [ ] **Step 1: Add RC8 (failing until the loop truly nests 3 axes — should already pass after Task 2)**

Append RC8 to the `#lspec group` (change RC7's trailing `)` to `) $` and add):

```lean
-- RC8  3-D nested scan (generality of n-D support). Axes a,b,d each size 2. Base S = 0 on the
--   d=0 plane; step adds T=ones. Only the fully-advanced cell G[1,1,1] = G[0,0,0]+T[0,0,0] = 1
--   is written; all boundary cells keep 0. ⇒ a 2×2×2 tensor with a single 1 at [1,1,1].
test "RC8 3d-scan"
    (evalEqB (tlprog!{ axis a : ℕ = 2, b : ℕ = 2, d : ℕ = 2
            G[a, b, 0]        := S[a, b]
            G[a +1, b +1, d +1] := G[a, b, d] + T[a, b, d] })
      (HashMap.ofList [("S", tl [2,2] [0,0,0,0]), ("T", tl [2,2,2] [1,1,1,1,1,1,1,1])])
      "G" (tl [2,2,2] [0,0, 0,0, 0,0, 0,1]))
```

(Flat data order for the expected `[2,2,2]`: only index `[1,1,1]` — the last element — is `1`.)

- [ ] **Step 2: Build; verify RC8 passes**

Run: `cd leanncd && lake build Eval.Portfolio.RecurrenceTest`
Expected: `✓ ∃: RC8 3d-scan` (plus all prior RC tests green).

- [ ] **Step 3: Commit**

```bash
cd leanncd
git add test/Eval/Portfolio/RecurrenceTest.lean
git commit -m "test(scan): 3-D nested scan proves n-D generality"
```

---

### Task 5: Sync the portfolio docs

Update `docs/test_portfolio.md` so KG-2dscan reads as fixed and RC6/RC8 are documented.

**Files:**
- Modify: `leanncd/docs/test_portfolio.md`

- [ ] **Step 1: Update RC6 row (§7) and add RC8**

Change the RC6 row (currently "CONFIRMED SILENT-WRONG … `[F]` (KG-2dscan)") to reflect the fix (`[N]`, correct output `[[0,0],[0,1]]`), and add an RC8 row for the 3-D scan.

- [ ] **Step 2: Update §14 KG-2dscan row → fixed**

Strike through the `KG-2dscan` row (mirror the `KG-scanagg` treatment): mark FIXED 2026-07-08, live tests RC6/RC8, note zero-default boundary semantics.

- [ ] **Step 3: Update §18 §A soundness table**

KG-2dscan was the sole remaining row in §18 §A. Remove it from the open-bugs table (mirror the KG-scanagg removal), leaving a parenthetical that both scan soundness bugs are now fixed; update §D tracking note.

- [ ] **Step 4: Update the §1 file table + §16 coverage matrix**

In the §1 table, update the `RecurrenceTest.lean` row to `RC2–RC8`. In §16, update the "2-D scan" / "Contraction inside a scan" rows to note RC6/RC8 now work.

- [ ] **Step 5: Commit**

```bash
cd leanncd
git add docs/test_portfolio.md
git commit -m "docs(scan): KG-2dscan fixed — multi-axis scans supported (RC6/RC8)"
```

---

## Self-Review

**Spec coverage:**
- n-D data model → Task 1 (types) + Task 2 (grouping/eval). ✓
- Positional base recovery → Task 2 Step 3. ✓
- Component grouping → Task 2 Step 4. ✓
- Nested eval + zero-default boundaries → Task 2 Steps 5–7. ✓
- Multi-axis causality + isAffine=false → Task 2 Step 4. ✓
- Full Br support → Task 3. ✓
- RC6 fix + RC8 generality → Task 2 / Task 4. ✓
- Docs → Task 5. ✓

**Type consistency:** `Stmt.iterInfo : List (UID × AxisSpec × Bool × Nat)` — the `isRecur` field is `t.2.2.1` and slot-position is `t.2.2.2` throughout (Task 1 Step 3, Task 2 Steps 3–4). `ScanStmt.scan` axis param is `List AxisSpec` everywhere (Tasks 1–2). `writeSliceAtMulti`/`iterSlotPositions`/`cartesianList` defined in Task 2 and used only there.

**Implementation-reality note:** this is a compiler-internals change; the exact Lean surface for a few library calls (`List.zipIdx` vs `enum`, `mergeSort` comparator, `HashMap` fold idioms, `BrProgram`/`BrBaseP` field names in Task 3) may differ slightly by toolchain. Each task ends in a `lake build`, which is the arbiter — adjust the call to the analogous available API and re-build. No behavior is left to guesswork; the failing/passing test at each task pins the intended result.

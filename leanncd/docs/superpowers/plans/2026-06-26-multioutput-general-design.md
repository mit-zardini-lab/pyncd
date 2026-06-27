# Design: fully general multi-output `BrBase` (coupled scans as first-class) — provable

**Supersedes §4 of `2026-06-26-scan-lowering-fix.md`** (the reject-vs-fuse decision). This is the
"fully general" path: multi-output steps, with coupled scans realized as a single multi-output
generator. Goal: `compile_wellFormed` provable sorry-free for the *whole* language including coupled
scans, with only `[propext, Classical.choice, Quot.sound]`.

## 0. The load-bearing principle (why this stays provable)

`realize` is a fold over a **monotonic live pool**: step `i` reads from `poolAt i`, applies its
generator, and prepends its outputs to get `poolAt (i+1)`. Every `WellFormed` conjunct and `stepPiece`
rely on `reads ⊆ poolAt i` and on the pool only growing. A *feedback edge* (a step reading a wire not
yet in the pool — its own or a later step's output) destroys both.

Therefore generality lives in **multi-output generators**, NOT in DAG feedback:

- A **scan is an atomic `Br` generator** (categorically a trace/fold). Its recurrence over the iteration
  axis is the generator's *internal semantics*, never a routing wire. Its interface is
  `dom = {initial states} ++ {per-step inputs}`, `cod = {output sequences}` (one per coupled tensor).
- `finalizeScans` already groups all coupled, shared-iteration recurrences into ONE `ScanStmt.scan`, so
  after scheduling there are **no inter-step back-edges** — the only self-reference is intra-scan, which
  the generator absorbs. Genuine inter-step cycles (e.g. `A:=B; B:=A`, which are NOT scans) are rejected
  by the acyclicity guard (§7).

Net: the routing DAG over steps stays acyclic and the pool stays monotonic. The single new axis of
generality is **a step may have `n ≥ 1` outputs**, addressed by wire slots. The data model already
anticipates this: `Wire.internal (step slot : Nat)` and `BrBaseP.outputWeaves : List WeaveShapeP` are
already slot-indexed lists — today the code just pins `slot = 0` and `length = 1`. The design below
removes those two pins and propagates `slot` everywhere. The proof *structure* is preserved; each lemma
generalizes `0 ↦ s` / `length 1 ↦ length = writes.length`.

## 1. Data model (no type changes needed)

- `Wire.internal (step slot : Nat)` — already present; `slot` becomes meaningful (`0..n-1`).
- `BrBaseP.outputWeaves : List WeaveShapeP` — already a list; length becomes `= writes.length`.
- `ThreadedComposed.{steps, routing, nExternal}` — unchanged.

## 2. PASS-1 maps: `buildNameToStep` returns `(step, slot)`

Currently `buildNameToStep : HashMap String Nat` (name → step). Generalize to
`HashMap String (Nat × Nat)` (name → (step, slot)). For step `i` writing `writes = [w₀,…,w_{n-1}]`
(in order), assign `wₛ ↦ (i, s)`. Implementation: fold over `stmts.zipIdx`, and for each, fold over
`sc.writes.zipIdx` inserting `wₛ ↦ (i, s)`.

**Lemmas to (re)prove (generalize existing):**
- `buildNameToStep_lt`: `nameToStep[nm] = (j, s) → j < stmts.length` (as today, on the step component).
- `buildNameToStep_slot_lt` (NEW): `nameToStep[nm] = (j, s) → s < (stmts.getD j default).writes.length`.
  By construction the inner `writes.zipIdx` fold only assigns `s < writes.length`. Same fold-invariant
  technique as `buildExtIndex`'s `goodExtState`.

## 3. `buildStep` for a multi-output scan

For `.scan name iterAx base recur isAff` (general — `base`/`recur` may cover several names):

- **writes** `= ScanStmt.writes sc` (e.g. `[G, H]`), ordered; this order fixes the output slots.
- **outputWeaves** `= writes.map (fun w => mkWeaveFor w)` — one output weave per written tensor, over the
  step `degree`. Length `= writes.length` (was `[stepMkWeave s]`, length 1).
- **input read set** (replaces `repStmt.readFactors`):
  `inputReads = (base.flatMap readFactors ++ recur.flatMap readFactors).filter (rf ∉self)` where
  `self = rf.1 ∈ writes`. I.e. ALL base+recur reads, **self-reads excluded** (the recurrence), so:
    - `base` reads give the **initial states** (e.g. `X` for `G[0]:=X`, `Y` for `H[0]:=Y`).
    - `recur` non-self reads give the **per-step inputs** (e.g. `W_G,U,W_H,V`).
  (Dedup as today via uid where needed.)
- **inputWeaves** `= inputReads.map weaveOf` (internal ⇒ producer's per-slot `tensorAxes`; external ⇒
  canonical weave). The init input `X:[j]` tiles the iteration axis in `degree`, so its `fixedAxesP = [j]`
  (rank 1) — matches `X`'s published rank. (Shape subtlety §6.)
- **routing** `= inputReads.map wireOf`, `wireOf rf = match nameToStep[rf.1] with | some (j,s) =>
  internal j s | none => external (extIndex[rf.1])`. No self-wires (excluded), so no `internal i _`.
- **degree** = dedup-by-uid of all axes across `base ++ recur` LHS+contracted (generalizes
  `stepDegAxes repStmt` from one stmt to the group).
- **reindexings** = one per input read, over `degree` (as today, per read).

`plain` and `scanPre` steps are single-output (`writes.length = 1`, `scanPre` has `readFactors = []`):
unchanged behavior, fall out as the `n = 1` case.

## 4. `poolAt`, `wireType`, `codObj` (generalize slot 0 → all slots)

- **`poolAt`**: prepend ALL of a step's output slots.
  ```
  outputSlots tc j := (List.range (tc.steps.getD j default).outputWeaves.length).map (Wire.internal j)
  poolAt i := (List.range i).foldl (fun p j => outputSlots tc j ++ p)
                ((List.range tc.nExternal).map Wire.external)
  poolAt_succ : poolAt (i+1) = outputSlots tc i ++ poolAt i      -- generalizes the `internal i 0 ::` form
  ```
- **`wireType (internal j s)`** `= weaveToArrayType (steps[j].outputWeaves.getD s [])` — already slot-aware;
  no change.
- **`codObj`** `= ` last step's outputWeaves mapped — already the full list; for `n` outputs it is the
  `n`-wire bundle. No change (was implicitly length 1).

## 5. `WellFormed` conjuncts — generalized statements, same proof structure

`WellFormed tc := wellFormedDom ∧ typeMatch ∧ outCount ∧ topo`.

- **Conjunct 3 `outCount` (was `wf_singleOutput`):** `∀ i < steps.length,
  (steps[i]).outputWeaves.length = (sp.stmts[i]).writes.length` (replaces `= 1`). Proof: `buildStep`
  sets `outputWeaves = writes.map _`; `List.length_map`. (If `realize` only needs *non-empty*, weaken to
  `≥ 1` since `writes ≠ []`; but the exact equality is cleaner for `codObj`.)
- **Conjunct 4 `topo` (`wf_topo`):** `∀ i < steps.length, ∀ w ∈ routing[i], w ∈ poolAt i`. Wire cases:
    - `external k`: `k < nExternal` via `buildExtIndex_lt_card`+`hne` ⇒ `mem_poolAt_external`. (unchanged)
    - `internal j s`: need `j < i` AND `s < n_j`.
      `j < i`: acyclicity guard §7 (no self-wires now, so this holds for scans too).
      `s < n_j`: `nameToStep[nm] = (j,s)` ⇒ `s < writes_j.length = outputWeaves_j.length`
      (`buildNameToStep_slot_lt` + conjunct 3). Then `mem_poolAt_internal` (generalized: `j<i ∧ s<n_j ⇒
      internal j s ∈ poolAt i`, same fold-membership proof).
- **Conjunct 2 `typeMatch` (`wf_typeMatch`):** `routing[i].map wireType = (steps[i]).inputWeaves.map
  weaveToArrayType`. Per input read `rf`:
    - internal `(j,s)`: `wireType (internal j s) = weaveToArrayType (outputWeaves[j][s])`. Need
      `fixedAxesP (outputWeaves[j][s]) = fixedAxesP (consumer input weave for rf)`. Generalize
      `buildStep_output_fixedAxes` to **per slot**: `fixedAxesP (outputWeaves[s]) = tensorAxes (writes[s]'s
      defining stmt)`, and the consumer's input weave for reading `writes[s]` is `tensorAxes` of that same
      stmt ⇒ equal ⇒ `weaveToArrayType_congr`. (Today's lemma is the `s = 0` case.)
    - external `k`: unchanged (`external_pointwise`).
- **Conjunct 1 `wellFormedDom` (`wf_dom`):** every `k < nExternal` referenced + rank agreement. Inits now
  wired (§3) ⇒ all externals referenced ⇒ surjectivity closes referencedness; rank agreement via
  `checkReadRanks` thread (scan-independent, as already scoped). Multi-output does not touch this.

## 6. Shape subtlety — initial-state inputs

Init `X:[j]` (the `l=0` slice) has one fewer axis than the output `G:[j,l]`. In `degree = {j, l, …}` the
init input weave is `fixed` on `j` and `tiled` on `l` ⇒ `fixedAxesP = [j]` (rank 1), matching `X`'s
published rank — so conjunct 2 and `wellFormedDom` rank-agreement hold. The init's `reindexing` maps its
`[j]` coords into `degree` (l-column = 0, the base index). Confirm `idxToRow`/`stepDegAxes` produce this
for a base stmt whose LHS uses `iterAt iterAx 0`.

## 7. Acyclicity guard (re-land, now valid)

Re-introduce the previously-reverted guard, generalized to `(step, slot)` wires:
```
routableInOrder stmts : Bool :=
  let ns := buildNameToStep stmts
  stmts.zipIdx.all (fun (sc, i) =>
    (allInputReads sc).all (fun rf =>           -- §3 input read set (self-reads already excluded)
      match ns[rf.1]? with | some (j,_) => decide (j < i) | none => true))
def routeCore sp := if routableInOrder sp.stmts then <build> else throw (.cyclicDataflow …)
```
- Valid now because scans no longer self-wire, so coupled scans are NOT rejected (the reason the first
  attempt was reverted). Only genuine inter-step cycles fail.
- `routeCore_routable : routeCore sp = .ok _ → routableInOrder sp.stmts = true` makes `j < i` trivial.
- Fix the 3 `routeCore_*` lemmas: positive `if`-branch (`if_pos`) recovers the existing proof bodies.
- Add `CompileError.cyclicDataflow` (Uid.lean).

## 8. `realize` fold — multi-output `stepPiece`/`finalPiece`

- **`stepPiece i`**: morphism `(poolAt i).map wireType → (poolAt (i+1)).map wireType`. Today it gathers
  reads, tensors the step beside the carried pool, lands on `internal i 0 :: pool`. Generalize the cod
  side via `poolAt_succ` (= `outputSlots i ++ poolAt i`): the step's `n` outputs become the first `n`
  pool entries. `BrMorph.tensor (stepMorph step) (idm pool)` already yields `cod(step) ++ pool`; with
  `n` outputs `cod(step) = outputWeaves.map weaveToArrayType` (length `n`), matching `outputSlots i`
  types by `wireType (internal i s) = weaveToArrayType (outputWeaves[s])`. The single-output `hcod`
  (`length_eq_one`) generalizes to `n` via `map`/`getElem` over `range n`.
- **`finalPiece`**: select the program output(s) from the full pool. Today selects `internal m 0`.
  Generalize to select the last step's slots that constitute `codObj` (all `n_m` of them, or the
  designated primary outputs — see §9). `wiringBy` over the selected wire list; membership via
  `poolAt_succ`.
- `stepMorph`/`realizeBrBaseP` already produce `dom = inputWeaves.map _`, `cod = outputWeaves.map _` —
  multi-output works unchanged (the generator is opaque, op-tagged).

## 9. Program-output selection — DECIDED: option (a)

The program's `cod` is **ALL of the last step's outputs** (`codObj` as-is — already = the full last-step
output list). For a coupled final scan that is the `n`-wire bundle `[G, H, …]`. `schedule.outputs`
(= the last stmt's `writes`) and `codObj` are already consistent with this; no change and no proof
impact. (A projection to a designated subset is explicitly NOT pursued.)

## 10. Cross-layer (`tsncd` executor) — OUT OF SCOPE

This effort is **Lean-only**: the formal model (`buildStep`/`route`/`realize`) and the
`compile_wellFormed` proof. The Python `tsncd` scan-executor contract (inits + `n` outputs, no
self-input) is acknowledged but **explicitly deferred** — not implemented, tested, or coordinated here.
No `#eval`/end-to-end executor assertions are in scope; correctness targets are the Lean build + the
`#guard` structural tests in `test/` + the sorry-free `compile_wellFormed`.

## 11. Re-proof sequencing (checklist)

1. `Wire`/`BrBaseP` unchanged; add `CompileError.cyclicDataflow` (Uid.lean) → rebuild.
2. `buildNameToStep : … (Nat × Nat)` + `buildNameToStep_lt` + `buildNameToStep_slot_lt`.
3. `buildStep` multi-output scan (§3) + `routableInOrder`/`routeCore` guard (§7).
4. Test suite: simple scans gain init wire/lose self-wire; coupled scans now COMPILE (example 5 flips from
   reject to a structural assertion: `steps=1`, `outputWeaves.length=2`, all 6 externals referenced).
5. `RouteSpec` re-proofs (per-slot/multi-stmt): `buildStep_inputWeaves`, `buildStep_wires_mapM`,
   `buildStep_output_fixedAxes` (per slot), `buildStep_outputWeaves_length` (= writes.length),
   `routeCore_*` (if_pos), `routeCore_routable`.
6. `poolAt`/`poolAt_succ`/`mem_poolAt_internal` (slot-aware) — Realize.lean + Agreement.lean helpers.
7. Conjuncts: `outCount` (3), `wf_typeMatch` (2), `wf_dom` (1, + surjectivity + arity thread),
   `wf_topo` (4, via `routeCore_routable` + `buildNameToStep_slot_lt`).
8. `realize` `stepPiece`/`finalPiece` multi-output (§8).
9. `compile_wellFormed` assembles; `#print axioms` clean.

## 12. Risk register
- **Multi-stmt `degree`/`reindexings` (§3)** — correctly unioning axes across `base ++ recur` and typing
  lower-rank inits (§6); most intricate compiler change. **Highest risk** (executor is out of scope, §10).
- **`realize` multi-output fold (§8)** — generalizing the `length_eq_one` casts to `range n`; mechanical
  but touches dependent-type casts.
- **`buildNameToStep` uniqueness** — multi-output means a step writes several names; ensure each live name
  has a unique (step,slot) (no name written by two steps post-DCE/SSA) so `nameToStep` is well-defined and
  `topo_bound` unambiguous.
- **No hidden inter-step feedback** — rely on `finalizeScans` grouping ALL shared-iteration recurrence
  into one scan; verify nothing produces a cross-step back-edge that isn't a true (rejected) cycle.

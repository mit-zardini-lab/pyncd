# Implementation plan: fully-general multi-output `BrBase` + provable `compile_wellFormed`

**Design:** `2026-06-26-multioutput-general-design.md` (the "why/what"). This doc is the "how/order/verify".
**Supersedes** the remaining-conjunct work in `2026-06-25-wellformed-forall-p.md` (`wf_dom`/`wf_topo`):
those close only after the model change here. `wf_typeMatch` (proven) WILL be re-opened by the
`inputWeaves` shape change and re-proved in Phase D.

## A. Files in the blast radius (dependency order, bottom → top)
1. `LeanNCD/Exec/Uid.lean` — `CompileError` (+`cyclicDataflow`).
2. `LeanNCD/DSL/Pipeline/Lowering.lean` — `buildNameToStep`, `buildStep`, `routableInOrder`, `routeCore`.
3. `LeanNCD/DSL/Pipeline/RouteSpec.lean` — ~8 characterization lemmas.
4. `LeanNCD/Bridge/Realize.lean` — `poolAt`, `wireType` (ok), `WellFormed` def, `codObj` (ok),
   `stepPiece`, `finalPiece`, `interpUpto`/`realize`.
5. `LeanNCD/Bridge/Agreement.lean` — conjunct lemmas + helpers + `compile_wellFormed`.
6. `test/DSL/CompileExamplesTest.lean`, `test/DSL/Pipeline/ScanAffineTest.lean`, others (`#guard`s).

RouteSpec→Realize→Agreement is a build chain: a broken lemma in (3) reddens (4),(5). Therefore use the
**sorry-scaffold rule** below to keep every commit green.

## B. Green-at-each-commit strategy (the discipline)
- A commit may contain `sorry`, but **must `lake build` clean** (sorries = warnings, not errors).
- When a compiler change (Phase A) breaks a downstream proof, **immediately replace that proof body with
  `sorry`** (keeping/updating the statement to the new shape) in the SAME commit, so the build stays green.
  Then fill the sorries one per commit (Phases B–E).
- Never leave a hard error across a commit boundary. Track the live sorry set in each commit message.
- After every Phase, run the **full verification harness** (§G) before committing.
- `#print axioms` clean (`[propext, Classical.choice, Quot.sound]`) is required ONLY at the final Phase F
  checkpoint; intermediate commits will transitively depend on `sorryAx`.

## C. Phase A — compiler change (one red→green-via-sorry landing)

Interdependent; land together, scaffolding broken proofs with `sorry`.

### Task A.0 — `CompileError.cyclicDataflow`
- `Uid.lean`: add `| cyclicDataflow : String → CompileError` (before `deriving`).
- Verify: `lake build` (full rebuild — Uid is low in the chain). Commit `feat(uid): cyclicDataflow error`.

### Task A.1 — `buildNameToStep : HashMap String (Nat × Nat)`
- `Lowering.lean:315-317`: fold over `stmts.zipIdx`; inner fold over `sc.writes.zipIdx` inserting
  `wₛ ↦ (i, s)`.
- This breaks `buildStep` (wire build) and RouteSpec lemmas — handled in A.2/A.3.

### Task A.2 — `buildStep` multi-output scan + slot wires
- `Lowering.lean:323-378`:
  - **wires:** `nameToStep[rf.1]? = some (j,s) ⇒ Wire.internal j s` (was `internal j 0`); `none ⇒ external`.
  - **inputWeaves (internal):** producer's per-slot `tensorAxes` — `tensorAxes` of the stmt defining the
    read name at slot `s` (see Task A.2a for "which stmt").
  - **scan input read set:** for `.scan _ _ base recur _`, `inputReads = (base.flatMap readFactors ++
    recur.flatMap readFactors)` filtered to drop `rf.1 ∈ writes` (self-reads); dedup as needed.
    `plain`/`scanPre` unchanged (fall out as the single-stmt case; `scanPre` readFactors `= []`).
  - **outputWeaves:** `writes.map mkWeaveFor` (length `= writes.length`); was `[stepMkWeave repStmt]`.
  - **degree:** dedup-by-uid of LHS+contracted axes across `base ++ recur` (generalize `stepDegAxes`).
  - **reindexings:** one per `inputReads` entry over `degree`.
- **Task A.2a (sub):** define the per-slot producer-axis source: a helper `slotDefiningStmt (sc) (s) :
  Stmt` = the stmt in `sc` that writes `writes[s]` (its base+recur defining stmt carrying the output axes),
  so `inputWeaves` for an internal read of `writes[s]` and `outputWeaves[s]` derive from the SAME axes
  (the conjunct-2 coincidence, preserved per slot).

### Task A.3 — `routableInOrder` + guarded `routeCore`
- `Lowering.lean` (after `buildNameToStep`): `routableInOrder stmts : Bool` checking, for each
  `(sc,i) ∈ stmts.zipIdx` and each `rf ∈ inputReads sc`, `match ns[rf.1]? with | some (j,_) => j < i |
  none => true`. (Factor `inputReads` out of `buildStep` so the guard and `buildStep` share it.)
- `routeCore`: `if routableInOrder sp.stmts then <existing do> else throw (.cyclicDataflow …)`.

### Task A.4 — scaffold broken proofs + fix tests, land Phase A green
- In `RouteSpec.lean`: any lemma whose statement mentions `internal _ 0` / single-output / `repStmt`
  readFactors — update the STATEMENT to the new shape and set body `:= by sorry` (or `:= sorry`).
  Likely: `buildStep_inputWeaves`, `buildStep_wires_mapM`, `buildStep_output_fixedAxes`,
  `buildStep_outputWeaves_length_one`, `routeCore_steps_length/routing_length/getD`, plus
  `buildNameToStep_lt`.
- In `Realize.lean`/`Agreement.lean`: if `WellFormed`'s conjunct 3 statement changes (Phase C), defer —
  for Phase A keep `WellFormed` as-is; `wf_singleOutput` may temporarily `sorry`.
- `test/`: update `#guard`s — coupled scan (CompileExamplesTest ex.5) now COMPILES with
  `steps==1 ∧ outputWeaves.length==2 ∧ nExternal==6` all referenced; simple scan routing loses the
  self-wire and gains the init wire. Add `#eval`-derived assertions.
- Verify: `lake build` green (warnings only). Commit
  `feat(route): multi-output buildStep + (step,slot) maps + acyclicity guard [proofs scaffolded]`,
  listing the sorry set.
- **STATUS (2026-06-26): Tasks A.0–A.4 DONE, committed `d5b6a61`, build GREEN, all tests pass.**
  Coupled scans compile multi-output (no self-wires, all externals referenced). Deviation from plan:
  A.0 was batched into the A.0–A.4 landing (one full rebuild instead of two); `#guard` tests needed NO
  changes (self-read exclusion means coupled scans no longer trip the guard). Remaining scaffold sorries:
  RouteSpec `routeCore_routable`, `buildStep_output_fixedAxes`, `buildNameToStep_lt`,
  `buildNameToStep_slot_lt`, `buildStep_inputWeaves`, `buildStep_wires_mapM`; Realize
  `stepPiece`/`finalPiece` hcod; Agreement `wf_singleOutput`(≥1), `port_external_weave`,
  `external_pointwise`, `internal_pointwise`, `wf_typeMatch`, `wf_dom`, `topo_bound`, `wf_topo`.

### Task A.5 — faithful scan outputs (drop nonlin-split `%nl` intermediates) — DO BEFORE Phase B

**Why first:** the output set is foundational to every B–D proof (`buildNameToStep` slots, `slotStmt`,
`outputWeaves`, `buildNameToStep_slot_lt`, `buildStep_output_fixedAxes`, `wf_typeMatch`, `wf_topo`).
Changing it after proving = re-prove everything; changing it now = ~4 def edits + a re-`#eval`. It also
serves the soundness goal: `splitNonlins` intermediates (`%nl`) currently surface as extra output slots,
so `codObj` (program `cod`) includes garbage — a milder instance of the unfaithfulness we are removing.

**The fix (clean — no temporal analysis needed).** True scan outputs are exactly the names written by
BOTH a base AND a recurrence stmt:
```
ScanStmt.outputs : List String
  | .plain s        => [s.lhsName]
  | .scan _ _ b r _ => (b.map Stmt.lhsName).filter (fun n => (r.map Stmt.lhsName).contains n)  -- base ∩ recur
  | .scanPre nm _ _ => [nm]
```
- simple: base `[S]` ∩ recur `[%nl,S]` = `[S]`; coupled: `[G,H]` ∩ `[%nlG,G,%nlH,H]` = `[G,H]`; even a
  nonlinear base `S[0]:=relu(X)` (base `[%nl0,S]`) ∩ recur `[%nlR,S]` = `[S]` — both intermediate kinds
  drop out. Robust: every true recurrence var has a base case (compiler errors `missingBaseCase`
  otherwise). `%nl` is only read self-internally (already excluded from `inputReadFactors`), so nothing
  cross-step references it ⇒ mapping only `outputs` in `buildNameToStep` is safe.

**Edits (Lowering.lean):** add `ScanStmt.outputs`; swap `writes → outputs` in
(i) `buildNameToStep` (fold `sc.outputs.zipIdx`), (ii) `ScanStmt.slotStmt` (`sc.outputs.getD s`),
(iii) `buildStep`'s `outputWeaves := (List.range sc.outputs.length).map sc.slotWeave`. Leave
`ScanStmt.writes` (used by scheduling/`topoSort`/`buildExtIndex`) untouched.

**Verify:** re-`#eval` → `outW = [1]` (simple), `[2]` (coupled); `lake build` green; tests pass.
**Proof-statement ripples:** the scaffold lemmas that say `…writes.length` become `…outputs.length`
(`buildStep_outputWeaves_length_one` — already proven, re-check the `simp`; `buildNameToStep_slot_lt`
stub — restate). Cheap because most are still `sorry`.
**Commit:** `fix(route): faithful scan outputs = base∩recur (drop %nl intermediates)`.

## D. Phase B — RouteSpec characterization lemmas (fill sorries)

One commit per lemma; `lean_diagnostic_messages` clean after each.

- **B.1 `buildNameToStep_lt`** (step bound) + **`buildNameToStep_slot_lt`** (NEW: `s < writes_j.length`,
  via the inner `writes.zipIdx` fold invariant — mirror `goodExtState`/`foldl_zipIdx_val_lt`).
- **B.2 `routeCore_steps_length` / `routeCore_routing_length` / `routeCore_getD`** — `if_pos` recovers the
  prior bodies (the guard branch); error branch via `simp [throw,…]`.
- **B.3 `routeCore_routable`** (NEW): `routeCore sp = .ok _ → routableInOrder sp.stmts = true`.
- **B.4 `buildStep_outputWeaves_length`** (NEW, replaces `_length_one`): `= writes.length`
  (`List.length_map`).
- **B.5 `buildStep_wires_mapM`** — RHS wire builder now yields `internal j s` / `external`.
- **B.6 `buildStep_inputWeaves`** — RHS per-read weave (internal per-slot `tensorAxes`; external canonical).
- **B.7 `buildStep_output_fixedAxes`** (per slot): `fixedAxesP (outputWeaves.getD s []) =
  tensorAxes (slotDefiningStmt sc s)`.
- Verify after each: `lean_diagnostic_messages RouteSpec.lean` clean; `lake build`.

## E. Phase C — Realize model generalization

- **C.1 `poolAt` + `poolAt_succ`** (`Realize.lean:163,196`): prepend `outputSlots tc j` (all slots) instead
  of `internal j 0`. Update `poolAt_succ` and `realizeDom_eq_poolAt_zero`.
- **C.2 `WellFormed` conjunct 3** (`Realize.lean:171-177`): replace single-output clause with
  `∀ i < steps.length, (steps.getD i default).outputWeaves.length = (??)`. NOTE: `WellFormed` is over `tc`
  alone (no `sp`), so phrase conjunct 3 as `outputWeaves.length ≥ 1` (sufficient for `codObj`/`finalPiece`)
  rather than `= writes.length` (which needs `sp`). The exact `= writes.length` lives in the Agreement
  proof, not the `WellFormed` contract. Pick `≥ 1` (or `≠ []`) for the conjunct.
- **C.3 `mem_poolAt_internal`** (Agreement helper): generalize to `j < i ∧ s < n_j ⇒ internal j s ∈
  poolAt i`. `mem_poolAt_external` unchanged.
- Verify: `lake build` (Realize + downstream may `sorry` on conjuncts until Phase D).

## F. Phase D — conjuncts (Agreement.lean)

- **D.1 conjunct 3** (`outCount`): `outputWeaves.length ≥ 1` from `buildStep_outputWeaves_length`
  (`writes ≠ []`). 
- **D.2 `wf_typeMatch`** (re-prove): per-slot internal case uses `buildStep_output_fixedAxes` (per slot) +
  `buildNameToStep_slot_lt` for `s`-bounds; external case unchanged (`external_pointwise`). The scan
  init/input reads are ordinary internal/external reads now.
- **D.3 `wf_dom`**: referencedness via `buildExtIndex` surjectivity (NEW lemma: every `k < extNames.card`
  is hit — strengthen `goodExtState`/`goodCard` with surjectivity, or a direct bijection argument) +
  `externalPort` existence from a wired read; rank agreement via `checkReadRanks` thread (NEW: extract
  arity-consistency from `hsp` — the deferred `compileToScheduled` plumbing).
- **D.4 `topo_bound` → `wf_topo`**: `topo_bound` now provable — `routeCore_routable hrc` +
  `buildNameToStep_slot_lt` give `j < i ∧ s < n_j` ⇒ `mem_poolAt_internal`. (Switch `topo_bound` to take
  `hrc`, drop the unprovable `hsp` form.)
- Verify each: `lean_verify` on the lemma; `lake build`.

## G. Phase E — realize fold (multi-output)

- **E.1 `stepPiece`** (`Realize.lean:204`): cod side via `poolAt_succ` = `outputSlots i ++ poolAt i`;
  generalize the `length_eq_one` `hcod` to `range n` (map/getElem over the `n` output slots).
- **E.2 `finalPiece`** (`Realize.lean:238`): select ALL last-step slots (§9 option a) — `wiringBy` over
  `(range n_m).map (internal m)`; `codObj` already = the full list.
- **E.3 `interpUpto`/`realize`**: should compose unchanged once `stepPiece`/`finalPiece` typecheck.
- Verify: `lake build`; `Bridge/AgreementTest` axiom line.

## H. Phase F — close-out

- Remove any residual scaffolding `sorry` (target: only the pre-existing out-of-scope ones —
  `fromThreadedComposed`, `realize_fromThreadedComposed_agree`).
- `#print axioms compile_wellFormed` ⇒ `[propext, Classical.choice, Quot.sound]` (run `lean_verify
  LeanNCD.compile_wellFormed`).
- Update `2026-06-25-wellformed-forall-p.md` RESUME POINT: Phase 4 DONE.
- Final `/lean4:checkpoint` (per-file + project build + axiom check). Commit.

## G(harness). Verification commands
- Build: `lake build` (full) or `lean_build` on the top file (`Bridge/Agreement.lean`).
- Per-file diagnostics: `lean_diagnostic_messages <file>`.
- Axioms: `lean_verify LeanNCD.<name>` (e.g. `LeanNCD.compile_wellFormed`, `LeanNCD.wf_typeMatch`).
- Structural tests: the `#guard`s build as part of `lake build` (test/ targets).
- Ground-truth probes: scratch `#eval cs.routing` / `.steps.map (op,inW,outW)` / `externalPort` /
  `wellFormedDom` on simple + coupled scans (as used during discovery; delete scratch after).

## I. Sequencing summary (commit cadence)
```
A.0-A.4 compiler + scaffold + tests         [DONE d5b6a61, green]
A.5 faithful scan outputs (base∩recur)      [green]  ← DO BEFORE Phase B (foundational)
B.1 … B.7 RouteSpec lemmas                  [green, sorries shrink]   (one commit each)
C.1-C.3 poolAt/WellFormed/mem_poolAt        [green]
D.1 … D.4 conjuncts                         [green, sorries shrink]   (one commit each)
E.1-E.3 realize fold                        [green]
F close-out + axiom check                   [green, only out-of-scope sorries remain]
```

## J. Risks / watch-items (from design §12, plus impl)
- **Multi-stmt `degree`/`reindexings` + lower-rank init typing (§3/§6)** — highest; verify with `#eval`
  on coupled + init-bearing scans before trusting the proofs.
- **Faithful outputs (A.5)** — RESOLVED via `outputs = base∩recur` (drops `%nl` intermediates); do it
  BEFORE Phase B to avoid re-proving against the loose `writes`-based output set.
- **`buildNameToStep` uniqueness** — each live name has a unique `(step,slot)` post-DCE/SSA; if a name can
  be written by two steps, `nameToStep`/`topo_bound` break. Add a guard/lemma if needed.
- **`WellFormed` conjunct-3 phrasing (C.2)** — keep it `tc`-only (`≥ 1`); don't leak `sp` into the
  contract.
- **`wf_dom` rank thread (D.3)** — the `checkReadRanks`-through-`compileToScheduled` extraction is the
  largest single proof; may warrant its own sub-plan if it balloons.
- **Scaffolding hygiene** — every `sorry` introduced in Phase A must be tracked and burned down by Phase F;
  none may masquerade as the pre-existing out-of-scope sorries.

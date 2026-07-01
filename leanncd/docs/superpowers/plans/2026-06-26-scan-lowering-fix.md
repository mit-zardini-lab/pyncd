# Scope: Option-B scan-lowering fix (sound scans → provable `wf_topo`/`wf_dom`)

**Status:** scoping (no code changes yet). **Owner decision required** on §4 (coupled scans).
**Origin:** Phase-4 `compile_wellFormed` work found `wf_topo` (and `wf_dom`) FALSE for scans —
a compiler/model soundness bug, not a proof gap. See
`2026-06-25-wellformed-forall-p.md` RESUME POINT and `Agreement.lean` `topo_bound` docstring.

## 1. The defect (ground truth, `#eval`'d through the real pipeline)

`buildStep` builds a scan step's wires/weaves from `repStmt.readFactors` only, where
`ScanStmt.repStmt (.scan _ _ base recur _) = recur.head?` ([Lowering.lean:281-284]). This drops the
`base` (initial-state) stmts and all of `recur.tail`, and turns the recurrence's self-read into a
wire onto the step's own output.

Simple scan `S[j,0]:=X[j]; S[j,l+1]:=relu(S[j,l]·W[j,k])`:
```
nExternal=2 (X=0, W=1)   steps=1   step=(scan, inputWeaves=2, outputWeaves=1)
routing = [[internal 0 0, external 1]]      -- internal 0 0 = S reading ITS OWN output; X (slot 0) never wired
```
Coupled scan (`G`,`H` mutually recurrent, §12.1 example 5):
```
nExternal=6 (X,Y,W_G,U,W_H,V)   steps=1   step=(scan, inputWeaves=4, outputWeaves=1)
routing = [[internal 0 0, external 2, internal 0 0, external 3]]
          -- G,H self-reads BOTH collapse to internal 0 0; X,Y,W_H,V (slots 0,1,4,5) never wired
```

Three independent defects, **all scans affected** (simple included):
- **D1 self-wire** `internal i 0` ∈ `routing[i]`, but `internal i 0 ∉ poolAt i` (pool gains it only at
  `poolAt (i+1)`). ⇒ `wf_topo` false; `realize.stepPiece`/`wiringBy` cannot gather it.
- **D2 dropped initial state** — `base` reads (e.g. `X`) are external slots but never appear in routing
  ⇒ `externalPort` = none ⇒ `wf_dom`/`wellFormedDom` false; realized morphism ignores the init.
- **D3 collapsed multi-output (coupled only)** — `G`,`H` (2 tensors) share one output slot, and the 2nd
  recurrence's inputs (`W_H`,`V`) are dropped. Conflicts with the single-output invariant (§4).

**Prevalence:** D1+D2 hit the simplest scan, so this is not niche — it is every self-recurrent scan.
Scans are the recurrent core of NCD; "reject all scans" is not acceptable. D3 is narrower (only
coupled/multi-output scans).

## 2. Target invariant (what makes the conjuncts provable)

After the fix, a successful compile must yield, for every step `i`:
- every `internal j 0 ∈ routing[i]` has `j < i` (no self-wires, no forward/cycle edges) — `wf_topo`;
- every external slot `k < nExternal` is referenced by some `routing` port — `wf_dom` referencedness;
- one output weave per step — `wf_singleOutput` (forces §4 decision).

## 3. The fix — simple (single-output) scans

Change scan lowering so the scan generator's inputs are **{initial state(s)} ∪ {per-step inputs}**, and
the self-recurrence is the scan op's own semantics (NOT a routing wire).

### 3a. `buildStep` (`Lowering.lean:323-378`)
For a `.scan name iterAx base recur isAff` step, replace the `repStmt.readFactors`-driven build with a
scan-specific read set:
- **self-reads excluded:** drop any read factor `rf` with `rf.1 ∈ sc.writes` (the recurrence over
  `iterAx`); these are internal to the scan generator, not pool wires.
- **initial state(s) included:** add the `base` stmts' read factors (e.g. `X` from `S[0]:=X`) as inputs.
- **per-step inputs:** the non-self reads from `recur` (e.g. `W`).
Resulting `routing` = wires for {init reads} ++ {non-self recur reads}; `inputWeaves` correspondingly.
**Shape note (D-shape):** the init input `X` has axes `[j]` (no `iterAx`), whereas the dropped self-read
`S[l]` had `[j, iterAx]`. So input-weave shapes change — the init weave omits the iteration slot. The
scan op lifts `init : A` over `iterAx` to `output : Seq_{iterAx} A`. `reindexings` (`StMatP`) for the init
input must express its coords over `degree` accordingly.

### 3b. degree / output weave
`stepDegAxes`/`stepMkWeave` already compute `degree`/`outputWeaves` from the rep stmt; verify they still
give the correct single output over `[j, iterAx]`. The init input being lower-rank must not corrupt
`degree` (degree is LHS ++ contracted of the rep stmt; init axes are a subset — OK, but confirm).

### 3c. realize / executor contract (cross-layer)
The Br `scan`/`scanAffine` generator is an opaque op-tagged `BrBase` in Lean (`realizeBrBaseP`), but the
**Python executor (`tsncd`) dispatches on `op=="scan"`** and consumes the input order/structure. Changing
inputs from {self,…} to {init,…} changes that contract. The executor's scan step must read the init from
input 0 (etc.). **This requires a coordinated change in `tsncd` scan execution** + its tests — track as a
sibling task; the Lean fix alone would desync the executor.

## 4. Coupled / multi-output scans — OWNER DECISION

A coupled scan writes >1 tensor (`ScanStmt.writes` length > 1) but `BrBaseP` has a single output, and
`wf_singleOutput` (conjunct 3, already proven) requires exactly one. So coupled scans cannot be faithfully
represented without breaking the single-output model. Two options:

- **(4-reject) Reject coupled/multi-output scans with a `CompileError`** (recommended near-term). Add a
  frontend guard (in `finalizeScans` or a new check) rejecting any `.scan` whose `writes` has length > 1
  (or whose `recur`/`base` cover multiple names). Cheap, makes the accepted language single-output, and all
  four conjuncts close. Cost: drops §12.1 example 5 from the supported set (document it).
- **(4-multi) Generalize `BrBaseP` to multi-output.** Re-opens `wf_singleOutput`, `codObj` (last step's
  single output), and the entire `realize` fold (`stepPiece`/`finalPiece` assume one output). Large; defer.

**Recommendation:** 4-reject now; revisit 4-multi only if coupled scans prove load-bearing for target
workloads (prevalence question for the owner).

## 5. Downstream proof impact (sequencing)

Changing `buildStep`'s `inputWeaves`/`routing`/`reindexings` re-opens lemmas that characterize them. Order:

1. **`buildStep` change** (§3) + **coupled-scan guard** (§4-reject). Build + run test suite
   (`CompileExamplesTest`, `ScanAffineTest`, `EvalExamplesTest`, `LoweringTest`, `RouteWeaveTest`).
   Expect example 5 to move to a "rejected" assertion; simple scans to gain the init wire + lose self-wire.
2. **`RouteSpec.lean` re-proofs:** `buildStep_inputWeaves`, `buildStep_wires_mapM`,
   `buildStep_output_fixedAxes`, `buildStep_outputWeaves_length_one` — their RHS shapes change for the scan
   case. The `plain`/`scanPre` cases are unaffected; only the `.scan` branch of each `cases sc` needs new
   handling.
3. **`wf_typeMatch` (conjunct 2) re-proof.** It currently matches the present `inputWeaves` shape
   (internal ⇒ producer `tensorAxes`; external ⇒ canonical weave). With the init input, the scan's init
   wire is a normal internal/external read (producer publishes `[j]`, consumer init expects `[j]`) — the
   existing internal/external pointwise lemmas should still apply, but the scan branch must be re-checked.
4. **`wf_dom` (conjunct 1).** Now provable: every external slot referenced (init + inputs all wired) +
   rank agreement (still needs the surjectivity-of-`buildExtIndex` lemma + `checkReadRanks` arity-consistency
   thread — independent of scans).
5. **`wf_topo` (conjunct 4) / `topo_bound`.** With no self-wires and a cycle guard
   (the `routableInOrder`-in-`routeCore` approach, now VALID because scans no longer self-wire), `topo_bound`
   becomes provable: `routeCore = .ok` ⇒ `routableInOrder sp.stmts = true` ⇒ `j < i`. Re-introduce the
   reverted `routableInOrder` guard (Lowering.lean) + `cyclicDataflow` `CompileError` (Uid.lean) + fix the 3
   `routeCore_*` lemmas (positive `if`-branch recovers existing proofs) + add `routeCore_routable`.
6. **`compile_wellFormed`** then assembles sorry-free; verify `#print axioms` = `[propext, Classical.choice,
   Quot.sound]`.

## 6. Risks / open questions
- **Executor desync (§3c):** the Lean and Python scan contracts must change together. Highest risk.
- **Init weave shape (§3a D-shape):** lower-rank init vs full-rank output — confirm `reindexings`/`degree`
  handle it; may need a dedicated init reindexing.
- **`scanPre` (recurMorphism escape hatch):** has `readFactors = []` already; verify the fix leaves it
  untouched (no base/recur to rewire).
- **`buildNameToStep` last-writer:** with coupled scans rejected, each live name has a unique writer, so
  `nameToStep[nm]` is unambiguous — simplifies `topo_bound`. Confirm SSA/uniqueness post-`finalizeScans`.
- **Multiple base stmts / non-`l` inits:** confirm `base` only contains genuine initial-state assignments.

## 7. Estimated shape of work
- §3 buildStep + §4 guard: focused compiler change (~1 file, Lowering.lean) + 1 CompileError + tests.
- §3c executor: coordinated `tsncd` change (separate, owner-driven).
- §5 re-proofs: moderate (RouteSpec scan branches + wf_typeMatch scan branch) + the deferred wf_dom rank
  thread + re-landing the cycle guard. wf_topo itself becomes easy once self-wires are gone.

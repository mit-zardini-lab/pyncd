# §8.2 acset agreement — `fromThreadedComposed` / `realizeSBr` / Prop 8 — Implementation Plan

> **✅ COMPLETE (2026-07-03).** All of Tasks A–E landed; the three target sorries are closed and the
> §8.2 agreement (`realize_fromThreadedComposed_agree`) is proved with axioms
> `[propext, Classical.choice, Quot.sound]` (no `sorryAx`). `Bridge/AcsetCodec.lean` (new),
> `Bridge/SBr.lean`, and the agreement declarations in `Agreement.lean` are all sorry-free; full
> `lake build` green (8587). The theorem carries an added `(hs : tc.WellShaped)` hypothesis (shape
> invariants `WellFormed` doesn't carry — `routing.length = steps.length` + per-step reindexing dims;
> satisfied by every compiled program). Remaining project sorries are only the pre-existing
> out-of-scope B+/G/H obligations (Br keystone, Algebra `Mat ℝ`), unrelated to this plan. Task F
> (regression `#guard`s) optional — the round trip was empirically checked on all 5 §12.1 examples.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`)
> syntax for tracking. Verification is Lean-native (`lake build` / `lean_diagnostic_messages` /
> `lean_verify`), not a test runner — there is no separate test-writing step per task; the `#print
> axioms`/`lake build` gate on each task IS the test.

**Goal:** Discharge the last 3 sorries in `LeanNCD/Bridge/Agreement.lean` /
`LeanNCD/Bridge/SBr.lean` — `fromThreadedComposed`, `realizeSBr`, `realize_fromThreadedComposed_agree`
— completing Prop 8 (DSL/CSV agreement): the routed-DAG realization of a `ThreadedComposed` and the
acset-table realization of its extraction are the **same** `Σ (dom cod : BrObj), BrMorph dom cod`.

**Architecture:** Encode each `ThreadedComposed` step as one acset "equation" (`equationIdx` = step
index), its output/input weaves as `ArrayRow`s, its `StMatP` reindexing as `SampleRow`s, and the wire
each input reads from as a synthetic string in the existing `wireLabel` field. Define a decode
`toThreadedComposed` that is an exact **left inverse** of `fromThreadedComposed`
(`toThreadedComposed (fromThreadedComposed tc) = tc`), then build `realizeSBr` by decoding + replaying
the already-proved `realize` fold, so the agreement theorem reduces to the round-trip lemma plus Lean's
built-in proof irrelevance (no new categorical content — mirrors the note in `SBr.lean`'s doc comment).

**Tech Stack:** Lean 4, Mathlib (pinned), the project's `Std.HashMap`/`List` idioms already used
throughout `LeanNCD/DSL/Pipeline/` and `LeanNCD/Bridge/`.

## Global Constraints

- `#print axioms` clean (`[propext, Classical.choice, Quot.sound]`) is required only at the final task;
  intermediate commits may transitively depend on `sorryAx` (same discipline as
  `2026-06-26-multioutput-impl-plan.md` §B).
- Never leave a hard error across a commit boundary — a broken proof becomes `sorry` in the SAME commit
  that broke it (sorry-scaffold rule).
- No `Acset.SBrInstance`/`ArrayRow`/`SampleRow`/etc. schema changes — the encoding reuses existing
  `Option String`/`Option OpTag` fields. (See Design Decision below for why this is sound.)
- `git add` only files touched; commit after each task.

---

## Design Decision (resolved 2026-07-01, do not re-litigate without new information)

`realize_fromThreadedComposed_agree` is a pure `BrMorph` **equality** — it does not require the acset
tables to be human-readable or to match the *unrelated* Python `acset/convert.py:from_tensor_program`
convention (that function builds `SBrInstance` from the higher-level `TensorProgram`/`TensorEquation`,
which still carries tensor names/operator predicates; `ThreadedComposed`/`BrBaseP` have already erased
those — wires are bare `Nat` indices, `BrOp` is a coarse 9-way enum vs. `OpTag`'s unrelated 10-way one).
Milestone F's CSV byte-fidelity claim is a *different* property, checked against real Python-built
instances, not `ThreadedComposed`-derived ones — it is untouched by this plan.

**Decision: `fromThreadedComposed` uses a systematic/synthetic encoding** (slot-index-based `AxisUID`s
via `Nat.pair`, a fixed internal `BrOp ↔ OpTag` bijection-on-image, wire routing packed into
`wireLabel` as `"ext:<k>"` / `"int:<j>:<s>"`). Operator/tensor/predicate semantics carry no meaning;
the ONLY correctness requirement is that `toThreadedComposed` (Task B) exactly inverts it (Task C).
Rejected alternative: threading real names/predicates from the DSL front-end through `BrBaseP` — much
larger scope (touches `Lowering.lean`/`Target.lean`, re-verifies the whole multi-output proof chain for
no benefit to Prop 8).

**CORRECTION (found during Task A implementation, 2026-07-02): `Axis.name` IS load-bearing and CANNOT
be dropped.** `Axis = {name : Option String, size : Numeric}` (`Base/St.lean:8`) is used directly
inside `BrObj = List ArrayType`, and `realize_fromThreadedComposed_agree`/`agree_dom`/`agree_cod` state
*exact* `BrObj` equality — so a decoded axis with `name := none` where the original had `some "i"` would
make the theorem false, not just harder. This does NOT reopen the "systematic encoding" decision above
(that was about operator/tensor/predicate semantics, which really are erased before `ThreadedComposed`)
— it's a narrower, orthogonal fact: axis names ALREADY exist verbatim inside `tc` and must simply survive
the round trip.

**Resolution — no schema change needed.** Exhaustive grep of every `AxisP` construction site in
`LeanNCD/` (`Lowering.lean`, `RouteSpec.lean`, `Agreement.lean` — 8 call sites, zero exceptions) shows
the codebase-wide invariant `AxisP.mk (some x) (SizeExpr.var x)` — the name is always redundantly
embedded inside its own `SizeExpr.var` payload — with the unnamed case always `AxisP.mk none
(SizeExpr.var "_")` (the reserved sentinel). So `axisSizes : List (AxisUID × SizeExpr)` (already in the
Task A plan below, unmodified) is sufficient: store the axis's real `SizeExpr` verbatim; decode derives
`AxisP.name` from it via `nameOfSizeExpr : SizeExpr → Option String := fun | .var "_" => none | .var s
=> some s | _ => none` and reuses the stored `SizeExpr` verbatim for `.size` (exact, not re-derived).
Task A/B below are written against this resolution. If a genuine counterexample surfaces during
implementation (a real `AxisP` with a non-`.var` size or a name/`.var` mismatch), STOP and re-raise —
the invariant is empirically verified, not structurally guaranteed by the type system.

---

## File Map

- Modify: `leanncd/LeanNCD/Bridge/Agreement.lean` — replace the `fromThreadedComposed`/
  `realize_fromThreadedComposed_agree` sorries (currently lines 404–412; line numbers will drift as
  earlier tasks land, re-locate by name each task).
- Modify: `leanncd/LeanNCD/Bridge/SBr.lean` — replace the `realizeSBr` sorry (currently line 12).
- Create: `leanncd/LeanNCD/Bridge/AcsetCodec.lean` — new file for `fromThreadedComposed`,
  `toThreadedComposed`, and the round-trip lemma (keeps `Agreement.lean`/`SBr.lean` from growing a
  large mechanical codec inline; import it from both).
- Create: `leanncd/test/Bridge/AcsetCodecTest.lean` — `#guard`/`#eval` round-trip checks on the
  §12.1 example programs (coupled scan + simple scan), mirroring how `CompileExamplesTest.lean`
  exercises `route`.

---

### Task A: `AxisUID` allocation + `fromThreadedComposed` (the encode direction)

**Files:**
- Create: `leanncd/LeanNCD/Bridge/AcsetCodec.lean`
- Modify: `leanncd/LeanNCD/Bridge/Agreement.lean` (remove the `fromThreadedComposed` sorry, re-export
  or call the `AcsetCodec` def)

**Interfaces (finalized 2026-07-02 against real Lean 4.30 APIs — verified compiling in isolation
before writing the real file; supersedes the sketch originally written here):**
- `Nat.pair`/`Nat.unpair`/`Nat.pair_eq_pair`/`Nat.unpair_pair` live in `Mathlib.Data.Nat.Pairing`
  (already reachable — every file in this project imports `Mathlib` wholesale). Produces:
  `axisUidFor : Nat → Nat → Acset.AxisUID := fun i k => ⟨.rawAxis, Nat.pair i k⟩`.
- **Numbers-in-strings use a UNARY encoding, not decimal.** `String.toNat?`/`toString` round-trip
  has no ready-made `∀ n, (toString n).toNat? = some n` lemma in core/Mathlib/Batteries (checked); a
  from-scratch decimal-digit proof is a real detour. Unary is trivially provable and the values
  involved (step/slot/external indices) are small enough that a string of that many characters is
  practically fine:
  ```lean
  def natToUnary (n : Nat) : String := String.ofList (List.replicate n '1')
  def unaryToNat (s : String) : Nat := s.length
  theorem unaryToNat_natToUnary (n : Nat) : unaryToNat (natToUnary n) = n := by
    simp [unaryToNat, natToUnary, String.length]
  ```
  (Verified: this exact lemma closes with plain `simp`, no induction needed — confirmed by isolated
  `lake env lean` check before adopting.)
- **`wireLabel` encodes the whole `Wire` as ONE `Nat` via `Nat.pair`, then unary-strings that** —
  NOT `String.splitOn`-with-delimiters as originally sketched. `splitOn` has no established
  `++`-interaction lemmas (`-- TODO: splitOn` literally in `Batteries/Data/String/Lemmas.lean`);
  proving `parseWireLabel (wireLabel w) = some w` through it would be its own sub-project. `Nat.pair`/
  `Nat.unpair` already round-trip cleanly (used for `axisUidFor`), so nesting pairs sidesteps the
  string-library gap entirely:
  ```lean
  def wireCode : Wire → Nat
    | .external k   => Nat.pair 0 k
    | .internal j s => Nat.pair 1 (Nat.pair j s)
  def wireOfCode (n : Nat) : Wire :=
    let (tag, rest) := Nat.unpair n
    if tag = 0 then .external rest else let (j, s) := Nat.unpair rest; .internal j s
  def wireLabel (w : Wire) : String := natToUnary (wireCode w)
  def parseWireLabel (s : String) : Option Wire := some (wireOfCode (unaryToNat s))
  ```
  (Verified: `wireOfCode_wireCode`/`parseWireLabel_wireLabel` both close with plain `cases <;> simp`
  — confirmed by compiling `LeanNCD/Bridge/AcsetCodec.lean` directly, exit code 0, no `sorry`.)
- **`BrOp` is NOT mapped into `Acset.OpTag` at all** (drops the earlier 9-vs-10 collision problem
  entirely) — since the design decision is systematic/no-semantic-fidelity, just store the `BrOp`
  constructor's own index directly: `brOpIdx : BrOp → Nat` (`.contract↦0, .maxreduce↦1, .scatter↦2,
  .relu↦3, .softmax↦4, .normalize↦5, .scan↦6, .scanAffine↦7, .scanPre↦8`), `brOpOfIdx : Nat → BrOp`
  (inverse on `0..8`, defaulting to `.contract` outside that range — never hit on real round-trip
  data). Encode via `elementwiseFn := some (natToUnary (brOpIdx b.op))` on the `slot = 0` output row
  ONLY (mirrors Python's `_add_equation` convention of putting operator info on the first/`slot=0`
  output row — a step's `op` is per-`BrBaseP`, not per-output-slot, so other output rows for a
  multi-output step leave `elementwiseFn := none`). `operatorTag`/`opPredicate`/`bias` are left `none`
  throughout — they carry Python-specific semantic distinctions (masked-softmax predicate strings,
  linear-layer bias flags) that `BrOp` doesn't make, and nothing here needs to reconstruct them.
- Produces: `nameOfSizeExpr : SizeExpr → Option String := fun | .var "_" => none | .var s => some s | _
  => none` — the decode-side inverse of the codebase-wide `AxisP.mk (some x) (SizeExpr.var x)` /
  `AxisP.mk none (SizeExpr.var "_")` invariant (see Design Decision CORRECTION). Used by Task B, not
  Task A itself (Task A stores the real `SizeExpr` verbatim; it never needs to derive a name).
- Produces: `fromThreadedComposed (tc : ThreadedComposed) : Acset.SBrInstance`.

- [ ] **Step 1: Add the codec file skeleton with `axisUidFor`, `natToUnary`/`unaryToNat` (+ its
  round-trip lemma), `wireLabel`/`parseWireLabel`, `brOpIdx`/`brOpOfIdx`, each as a standalone
  sorry-free total function/lemma (no `ThreadedComposed` dependency yet).** Verify: `lake env lean
  leanncd/LeanNCD/Bridge/AcsetCodec.lean` compiles with no errors.

**REVISED design (found during implementation, 2026-07-02) — `axisUidFor` needs a THIRD coordinate,
and the op index needs a slot-independent home:**

- `axisUidFor i k` (2-arg, degree-position only) is insufficient: `degree`, each output weave, and
  each input weave are ALL independent axis-position spaces that must round-trip independently (the
  E2a presentation carries no uid linking a weave's fixed axis back to a specific degree position —
  "the dependent typing is dropped", per `Target.lean:63`). Use a 3-arg version instead: `axisUidFor3
  (i slot pos : Nat) : Acset.AxisUID := ⟨.rawAxis, Nat.pair i (Nat.pair slot pos)⟩`, where `slot`
  ranges over a per-step-uniform numbering: `0..outLen-1` = output weaves, `outLen..outLen+inLen-1` =
  input weaves (`outLen + j` for read `j`), and `outLen+inLen` (one past the last real slot) = the
  reserved **degree slot**. No cross-referencing between spaces is needed for the round-trip — each
  weave/degree's fixed axes just need to survive independently.
- **Tiled slots need their own marker**, not just an absent row: `ArrayAxisRow` is emitted for EVERY
  weave position (not just `fixed` ones), so the decoder can recover the weave's total length
  (otherwise a weave ending in `.tiled` slots would silently lose length information). Fixed slots get
  `axisUid := axisUidFor3 i slot p`; tiled slots get the reserved sentinel `⟨.natAxis, 0⟩` (`.natAxis`
  is otherwise unused by this codec, so it unambiguously means "tiled, ignore `id`"). `isTarget` is
  unused by this codec (nothing in the round-trip needs it) — set `false` uniformly.
- **The op index goes on `EquationRow.lhsName`, NOT an output-slot `ArrayRow`.** A step's `BrOp` is
  per-`BrBaseP`, but `outputWeaves.length` could in principle be `0` for a non-well-formed `tc` (the
  codec must be total over ALL `ThreadedComposed`, not just compiler output) — there would be no
  "slot 0" row to carry it. `EquationRow` is unconditionally emitted once per step, so
  `lhsName := some (natToUnary (brOpIdx b.op))` has no such gap.
- **`degree` is just a weave with no `tiled` option** — reuse the SAME output/input weave encoding
  helper on `b.degree.map .fixed : WeaveShapeP` rather than writing separate degree-specific code.

**Implementation, using the 3-arg scheme above.** Two shared helpers, then per-step assembly:
```lean
def encodeAxisRows (i slot : Nat) (w : WeaveShapeP) : List ArrayAxisRow :=
  (List.range w.length).map fun p => match w.getD p .tiled with
    | .fixed _ => { equationIdx := i, arraySlot := slot, axisUid := axisUidFor3 i slot p,
                     isTarget := false, position := p }
    | .tiled   => { equationIdx := i, arraySlot := slot, axisUid := ⟨.natAxis, 0⟩,
                     isTarget := false, position := p }
def encodeAxisSizes (i slot : Nat) (w : WeaveShapeP) : List (AxisUID × SizeExpr) :=
  (List.range w.length).filterMap fun p => match w.getD p .tiled with
    | .fixed a => some (axisUidFor3 i slot p, a.size)
    | .tiled   => none
def fixedPositions (w : WeaveShapeP) : List Nat :=
  (List.range w.length).filter fun p => match w.getD p .tiled with
    | .fixed _ => true | .tiled => false
```
Per step `i` (`b := tc.steps.getD i default`, `reads := tc.routing.getD i []`, `outLen :=
b.outputWeaves.length`, `inLen := reads.length`, `degSlot := outLen + inLen`):
- `EquationRow ⟨i, some (natToUnary (brOpIdx b.op))⟩`.
- Output rows: `s ∈ [0,outLen)` → `ArrayRow ⟨i, s, none, false, none, none, .reals, none, none, none,
  none, none⟩` + `encodeAxisRows i s (b.outputWeaves.getD s [])` + `encodeAxisSizes i s (...)`.
- Input rows: `j ∈ [0,inLen)`, `w := reads.getD j (.external 0)` → `ArrayRow ⟨i, outLen+j, none, true,
  none, none, .reals, none, none, none, none, some (wireLabel w)⟩` + `encodeAxisRows i (outLen+j)
  (b.inputWeaves.getD j [])` + `encodeAxisSizes i (outLen+j) (...)`.
- Degree: `encodeAxisRows i degSlot (b.degree.map .fixed)` + `encodeAxisSizes i degSlot (...)` (no
  `ArrayRow` needed for the degree slot — nothing decodes an `ArrayRow` for it, only its
  `ArrayAxisRow`/`axisSizes` rows, which is fine: `ArrayAxisRow.arraySlot` need not reference a real
  `ArrayRow` here, this is an internal-only codec, not a literal acset with Python-style FK integrity).
- `SampleRow`s per read `j` (reindexing `m := b.reindexings.getD j default`, `inW := b.inputWeaves.getD
  j []`, `fps := fixedPositions inW`): for `c ∈ [0, m.codLen)`, `tgtUid := axisUidFor3 i (outLen+j)
  (fps.getD c 0)`, `row := m.coeffs.getD c []`, `nz := (List.range m.domLen).filter (fun d =>
  row.getD d 0 ≠ 0)`: if `nz = []`, one row `⟨i, outLen+j, tgtUid, tgtUid, 0, m.bias.getD c 0⟩`
  (self-referencing `srcUid = tgtUid` sentinel for "pure bias, no source axis"); else one row per
  `d ∈ nz`: `⟨i, outLen+j, axisUidFor3 i degSlot d, tgtUid, row.getD d 0, m.bias.getD c 0⟩` (bias
  redundantly repeated across all rows sharing `(outLen+j, tgtUid)` — decode reads it off any one).

Concatenate all steps' rows (`List.flatMap` over `tc.steps.zipIdx`, threading `tc.routing.getD i []`
in parallel per step).

Verify: `lake env lean leanncd/LeanNCD/Bridge/AcsetCodec.lean`; `#eval fromThreadedComposed <a
concrete tc from tl!{}>` on the §12.1 coupled-scan example — sanity-check table row counts by hand
(steps × (outputs+inputs+1) weave-encodings, etc.) before moving on.

**Consequence for Task C:** `toThreadedComposed_fromThreadedComposed` should take `(h : tc.WellFormed)`
as a hypothesis, not hold unconditionally — `nExternal` is NOT recoverable from `routing` alone for an
arbitrary (non-well-formed) `tc` (an unreferenced external slot leaves no trace in any wire), and
`WellFormed`'s `wellFormedDom` conjunct is exactly what guarantees every external slot `< nExternal` IS
referenced. This matches how the lemma is actually used in Task E (always under a `WellFormed`
witness), so it costs nothing to add.

- [x] **Step 3: Wire up `Agreement.lean`.** [DONE 2026-07-02] Replaced the `sorry` body with `:=
  AcsetCodec.fromThreadedComposed tc` (dropped the stale `noncomputable` — the codec is fully
  computable, no `MvPolynomial`/`Classical.choice`). `lean_verify LeanNCD.fromThreadedComposed` ⇒
  `[propext]`, no `sorryAx`. Full `lake build` green (8587 jobs — one more than the prior 8586, the
  new `AcsetCodec.lean` module). Remaining sorries in `Agreement.lean`: only
  `realize_fromThreadedComposed_agree` (Task E); `agree_dom`/`agree_cod` are real proofs, still
  transitively sorry through it.

**TASK A DONE (2026-07-02).** `fromThreadedComposed` implemented and verified sorry-free. Sanity-checked
against real compiled examples (`test/DSL/CompileExamplesTest.lean`'s matmul and coupled-scan cases via
a scratch eval file — NOT committed, structure only): step/array/axis/sample row counts match hand
calculation exactly for both; axis names round-trip losslessly including the compiler's synthetic
external-read names (`W_0`/`W_1` etc., per `route`'s canonical-external-weave convention). Two design
corrections surfaced during implementation and are folded into the sections above: (1) axis names are
load-bearing, fixed via the `SizeExpr.var` invariant, no schema change; (2) `wireLabel` uses nested
`Nat.pair` instead of `String.splitOn` (no established `++`-interaction lemmas for `splitOn` yet). Next:
Task B (`toThreadedComposed`, the decode direction) — NOT started.

- [ ] **Step 4: Commit.**
  ```bash
  git add leanncd/LeanNCD/Bridge/AcsetCodec.lean leanncd/LeanNCD/Bridge/Agreement.lean
  git commit -m "feat(acset): fromThreadedComposed — systematic encode of ThreadedComposed as SBrInstance"
  ```

### Task B: `toThreadedComposed` (the decode direction)

**Files:**
- Modify: `leanncd/LeanNCD/Bridge/AcsetCodec.lean`

**Interfaces (revised 2026-07-02 to match Task A's actual final design — supersedes the sketch
originally written here, which referenced the superseded `axisUidFor`/`brOpToOpTag`/`splitOn`
scheme):**
- Consumes: `axisUidFor3`, `brOpOfIdx`, `parseWireLabel`, `nameOfSizeExpr`, `fixedPositions` (all Task
  A).
- Produces: `toThreadedComposed (s : Acset.SBrInstance) : ThreadedComposed` — total, defaulting on
  malformed/partial tables (never throws), matching the "no hypothesis on `realizeSBr`'s input"
  requirement noted in `SBr.lean`'s doc comment.

- [ ] **Step 1: `decodeWeaveAt (s) (i slot) : WeaveShapeP`** — gather `s.arrayAxes` rows with matching
  `equationIdx`/`arraySlot`, length = row count (Task A emits one row per position, contiguous), each
  position `p` looked up by `position == p`: `axisUid.type = .natAxis ⇒ .tiled`; else `.fixed
  ⟨nameOfSizeExpr size, size⟩` where `size := s.axisSizes` looked up by `axisUid` (defaulting to
  `.lit 0` — unreachable for real encoder output, only relevant for garbage input).
- [ ] **Step 2: `decodeReindexing (s) (i degSlot arraySlot domLen) (inW : WeaveShapeP) : StMatP`** —
  `fps := fixedPositions inW` (same helper Task A uses to encode), `codLen := fps.length`; for `c ∈
  [0,codLen)`, `tgtUid := axisUidFor3 i arraySlot (fps.getD c 0)`; `coeffs[c][d] :=` the `SampleRow`
  with that `tgtUid` and `srcUid = axisUidFor3 i degSlot d`'s `coeff`, else `0`; `bias[c] :=` any
  matching-`tgtUid` row's `offset`, else `0`.
- [ ] **Step 3: `decodeStep (s) (i) : BrBaseP × List Wire`** — `outLen`/`inLen` from counting
  `s.arrays` rows with that `equationIdx` split by `isInput`; `degSlot := outLen+inLen`;
  `outputWeaves := (range outLen).map (decodeWeaveAt s i)`, `inputWeaves := (range
  inLen).map (fun j => decodeWeaveAt s i (outLen+j))`, `degree := (decodeWeaveAt s i
  degSlot).map (fun | .fixed a => a | .tiled => default)`; `reindexings := (range inLen).map (fun j
  => decodeReindexing s i degSlot (outLen+j) degree.length (inputWeaves.getD j []))`; `op :=
  brOpOfIdx (unaryToNat ((s.equations.find? (fun e => e.equationIdx = i)).bind (·.lhsName)
  |>.getD ""))`; reads: for `j ∈ [0,inLen)`, the `ArrayRow` at `slot = outLen+j`'s `wireLabel` through
  `parseWireLabel`, defaulting to `.external 0` if missing/unparseable.
- [ ] **Step 4: Assemble `ThreadedComposed`.** `n := s.equations.length` (equations are emitted
  exactly once per step, in step order, by Task A's `foldl` — so `s.equations[i].equationIdx = i`
  for a well-formed encoder output); `steps/routing := (range n).map (decodeStep s ·)` unzipped;
  `nExternal := 1 + max` over all decoded wires' external indices (`0` if none) — see the
  Design Decision note: this is only guaranteed to equal the ORIGINAL `tc.nExternal` when `tc` is
  `WellFormed` (hence Task C's theorem takes that as a hypothesis).
  Verify: `lake env lean leanncd/LeanNCD/Bridge/AcsetCodec.lean` compiles; avoid `==`/`BEq` on
  `AxisUID`/`ArrayType`-ish fields without checking a `BEq` instance actually exists (`SBrInstance.lean`
  derives `DecidableEq`, not necessarily `BEq`) — prefer `match`/`decide`/`Nat`'s native `BEq` (used
  for `Nat`-typed fields like `arraySlot`/`equationIdx`/`position`, which are fine).

**TASK B DONE (2026-07-02).** Compiles clean (no `sorry`, no errors). **Empirically verified via
`decide (toThreadedComposed (fromThreadedComposed tc) == tc)` on ALL FIVE §12.1 example programs**
(matmul, masked attention, strided conv — nontrivial reindexing coefficients — upsample/scatter,
coupled scan) — every one returns `true` (scratch eval file, not committed). This is strong empirical
evidence Task C's theorem is actually TRUE and the encoding is correct end-to-end, not just
individually-plausible per-field. Implementation matched the plan sketch above almost exactly — no
further design surprises. Next: Task C, the formal proof of what was just checked empirically.
- [ ] **Step 3: Commit.**
  ```bash
  git add leanncd/LeanNCD/Bridge/AcsetCodec.lean
  git commit -m "feat(acset): toThreadedComposed — decode SBrInstance back to a routed DAG"
  ```

### Task C: round-trip lemma

**▶▶ TASK C DONE (2026-07-03) ◀◀** `toThreadedComposed_fromThreadedComposed` is **PROVED sorry-free**,
axioms `[propext, Classical.choice, Quot.sound]` (verified via `lean_verify`); full `lake build` green
(8587). `decodeStep_eq` assembly closed with helpers `slotWeave_out/in/deg`, `map_fixed_inv`,
`inputWeaves_len` (from WellFormed conj-2), `mem_from_equations`/`from_equation_find` (op recovery),
`from_inputRow_find` (reads recovery), wiring `decodeWeaveAt_from`/`decodeReindexing_from`/counts via
`List.ext_getElem`. `AcsetCodec.lean` is now sorry-free. NEXT: Task D (`realizeSBr`) then Task E
(`realize_fromThreadedComposed_agree`), threading the `(hs : tc.WellShaped)` hypothesis (discharge it for
compiled programs via `routeCore`/route length + shape lemmas).

**(historical) PAUSE / RESUME POINT (end of 2026-07-02 session):**
Both hard structural lemmas are PROVED and committed green; `AcsetCodec.lean` compiles with **one
remaining `sorry`**: `decodeStep_eq` (line ~1306), the per-step `BrBaseP`+reads assembly. Two statement
gaps were found & fixed by ADDING hypotheses (not touching `WellFormed`): the theorem is now
`toThreadedComposed_fromThreadedComposed (tc) (h : tc.WellFormed) (hs : tc.WellShaped)`, where
`WellShaped` bundles `routing.length = steps.length` + per-step reindexing-matrix shape constraints
(both hold for all compiled programs; discharged in Task E via route lemmas).

PROVEN (all in `AcsetCodec.lean`, committed): the isolation infra (`filter_flatten_tagged(_aux)`,
`from_field_filter`), `stepInsts_*`, the four `encodeStep_*_eqIdx` tagging lemmas, two-slot name
encoding (`nameUidFor3`/`lookupName`), `from_outLen`/`from_inLen`/`equations_length`, `from_nExternal`
(the `WellFormed`-dependent `nExternal` reconstruction), the full `steps`/`routing`/`nExternal`
top-level assembly of `toThreadedComposed_fromThreadedComposed` (reduces to `decodeStep_eq`),
**`decodeWeaveAt_from`** (weave round trip — keystone) with `slotWeave`/`filterSlot_flatMap_off`/
`find?_unique`/`lookupSize_from`/`lookupName_from`/`mem_from_axisSizes`, and **`decodeReindexing_from`**
(reindexing/matrix round trip) with `mem_encodeReindexing_*`/`encodeReindexing_exists_tgt`/
`fixedPositions_getD_inj`/`find?_map_getD`.

REMAINING = just `decodeStep_eq` assembly (`simp only [decodeStep]` then field-by-field `Prod`/`BrBaseP`
equality). Building blocks all exist: rewrite decoded `outLen`/`inLen` via `from_outLen`/`from_inLen`;
`outputWeaves`/`inputWeaves`/`degree` fields via `decodeWeaveAt_from` + `slotWeave` case-eval at
output/input/deg slots (+ the `WellFormed` conjunct-2 fact `inputWeaves.length = routing.length` for the
inputWeaves length); `reindexings` field via `decodeReindexing_from` (its statement already matches
`decodeStep`'s call exactly); `degree` also needs `(l.map .fixed).map (fixed→a|tiled→default) = l`. Two
small NEW pieces to add: **op** recovery (`equations.find?` for eq `i` gives `some (natToUnary (brOpIdx
op))` → `brOpOfIdx (unaryToNat …) = op` via `unaryToNat_natToUnary`+`brOpOfIdx_brOpIdx`) and **reads**
recovery (input-row `find?` by `slot = outLen+j` → `wireLabel` → `parseWireLabel_wireLabel` → `routing[j]`).
Use `List.ext_getElem` for the `(range …).map` fields. NOTE: two subagent runs stalled here on
heavy tactics — work in small compile-checked steps, avoid `simp`/`decide` on large goals.

**Files:**
- Modify: `leanncd/LeanNCD/Bridge/AcsetCodec.lean`

**Interfaces:**
- Consumes: `fromThreadedComposed` (Task A), `toThreadedComposed` (Task B).
- Produces: `theorem toThreadedComposed_fromThreadedComposed (tc : ThreadedComposed)
  (h : tc.WellFormed) : toThreadedComposed (fromThreadedComposed tc) = tc` — the `WellFormed`
  hypothesis was added during Task A implementation (see the note at the end of Task A): `nExternal`
  isn't recoverable from `routing` alone without it, and Task E's only call site already has it.

This is the hardest task — expect it to be the B.7-equivalent of this plan (an order/grouping fight,
not a deep category-theory one). Likely needed sub-lemmas, proved bottom-up, one commit each:

**IN PROGRESS (2026-07-02) — resume state.** Restructured `fromThreadedComposed` to fieldwise-flatten
of a new `stepInsts` helper (cleaner isolation reasoning). PROVEN & committed in `AcsetCodec.lean`:
- `filter_flatten_tagged_aux` / `filter_flatten_tagged` — **the general isolation lemma** (offset
  induction): filtering `(L.map …).flatten` by `key = i ∧ P` recovers `P`-filtered sublist `i` when
  each sublist `k` is tagged `key = k`. This is the workhorse — it applies BOTH cross-step (key =
  `equationIdx`, sublists = steps) AND within-step per-slot (key = `arraySlot`, sublists = the
  `(range …).flatMap encodeAxisRows` groups, since `flatMap = (map …).flatten`).
- `stepInsts_length`, `stepInsts_getElem` (`(stepInsts tc)[k] = encodeStep k tc.steps[k] (routing k)`).
- `encodeAxisRows_eqIdx`, `encodeStep_{arrays,arrayAxes,samples,equations}_eqIdx` (the `htag` suppliers).
- `from_field_filter` (in scratch, ready to port): generic reduction
  `(proj (from tc)).filter (decide (eqIdx=i) && P) = (proj (encodeStep i …)).filter P` for
  `i < steps.length`, instantiable per field via `Bool.decide_and` + the tagging lemmas. Bridge
  `decide (A ∧ B) = decide A && decide B` is `Bool.decide_and`.

**REMAINING (the mechanical grind, each its own mini-proof):**
1. **`decodeWeaveAt (from tc) i slot = <original weave>`** for a valid slot — reduce arrayAxes filter
   via `from_field_filter` (P = `arraySlot = slot`) to inst-`i`, then within inst-`i` use
   `filter_flatten_tagged` AGAIN (key = `arraySlot`) to pick the one `encodeAxisRows i slot w` group
   (the 3-way `output ++ input ++ deg` append needs `filter_append` + showing the non-matching groups'
   slots are out of range), then invert `encodeAxisRows`↔`decodeWeaveAt` (find? by position over a
   range-map recovers each slot; `.natAxis`⇒`.tiled`, else `.fixed ⟨nameOfSizeExpr size, size⟩`).
   Needs `lookupSize (from tc) (axisUidFor3 i slot p) = <that axis's size>` — global uid uniqueness via
   `axisUidFor3` injectivity (`Nat.pair_eq_pair`) so `find?` over the flattened axisSizes hits inst-i's
   entry (first & only match). This is the ONE empirical name-invariant step (Step 3b below): the size
   stored is `SizeExpr.var name` (or `.var "_"`), so `nameOfSizeExpr` recovers `.name` and the stored
   `SizeExpr` IS `.size` verbatim.
2. **`decodeReindexing (from tc) … = <original StMatP>`** — hardest; same reduction to inst-i samples,
   then invert the `SampleRow` matrix encoding (coeffs/bias per (fixed-pos, dom-pos)). Depends on the
   scan/`route` producing matrices whose stored shape matches `codLen = fixedPositions.length`.
3. **op / reads / degree** round trips — op via `brOpOfIdx_brOpIdx` + `unaryToNat_natToUnary` on the
   `equations.find?`; reads via `parseWireLabel_wireLabel`; degree via the weave round trip on `degSlot`.
4. **`decodeStep (from tc) i = (tc.steps[i], tc.routing.getD i [])`** — assemble 1–3 (`outLen`/`inLen`
   recovered by array-count via `from_field_filter` + `List.length_filter`-ish + tagging).
5. **Assembly** — `ThreadedComposed` ext on steps/routing (both from step 4 via `List.ext_getElem`) and
   **`nExternal`** (the only part needing `h : WellFormed`: conjunct-4 `w ∈ poolAt i` bounds every
   routing external index `< nExternal`, and `wellFormedDom` referencedness gives every `k < nExternal`
   IS referenced ⇒ `max + 1 = nExternal`; handle `nExternal = 0` separately).

**CORRECTNESS FIX (2026-07-02): two-slot name encoding.** `realizeAxis` keeps `a.name`, which flows
into `realize`'s `dom`/`cod` `BrObj`s (via `targetAxes : StObj = List Axis`), so the agreement's exact
Σ-equality requires the round trip to recover `AxisP.name` EXACTLY. The original encoding recovered the
name from `a.size` (via a `size = .var name` "canonical axis" invariant) — true of all compiler output
but NOT implied by `WellFormed`, so the theorem would have been FALSE for a pathological `WellFormed`
`tc` (e.g. axis `⟨some "i", .lit 5⟩`). Fixed by storing the name in a dedicated `.normAxis`-tagged
`axisSizes` entry (`nameUidFor3`/`lookupName`), independent of size — round trip is now UNCONDITIONAL
(verified on a name/size-mismatch case + all 5 §12.1 examples; only `h : WellFormed` needed). This adds
a second `axisSizes` entry per named axis but no extra proof burden (same `find?`-uniqueness machinery
as `lookupSize`). `nameOfSizeExpr` removed.

Original bottom-up step list (partially subsumed above):
- [ ] **Step 1: `axisUidFor`/`Nat.pair` injectivity lemma** — `axisUidFor i k = axisUidFor i' k' → i =
  i' ∧ k = k'` (direct from `Nat.pair_eq_pair`/`Nat.unpair_pair` in Mathlib). Verify:
  `lean_diagnostic_messages` clean.
- [ ] **Step 2: `brOpToOpTag`/`opTagToBrOp` round-trip** — `opTagToBrOp (brOpToOpTag op) = op` (finite
  case-split, `decide` or `cases op <;> rfl`). Verify: clean.
- [ ] **Step 3: `parseWireLabel (wireLabel w) = some w`** — `cases w <;> simp [wireLabel,
  parseWireLabel]` (string round-trip via `s!"ext:{k}"`/`toString`/`String.toNat?` — confirm
  `String.toNat?.toNat!` round-trips for arbitrary `Nat`, or use a decimal-digit lemma from Mathlib/
  Std if the raw round-trip isn't `rfl`-level). Verify: clean.
- [ ] **Step 3b: `nameOfSizeExpr` recovers `AxisP.name` exactly, for every `AxisP` reachable from a
  compiled `tc`.** Since `nameOfSizeExpr` is defined by cases on `SizeExpr`, NOT on `AxisP`, state it
  as: `∀ a : AxisP, (∃ i k, a = (tc.steps.getD i default).degree.getD k default) → nameOfSizeExpr a.size
  = a.name` (or the simpler unconditional `∀ x, nameOfSizeExpr (SizeExpr.var x) = if x = "_" then none
  else some x` plus a side lemma that every real `AxisP` construction site satisfies `a.size =
  SizeExpr.var (a.name.getD "_")` — whichever shape falls out naturally once Task A/B's concrete field
  layout is fixed). This is the ONE genuinely empirical step (see Design Decision CORRECTION) — if it
  fails on some real `tc`, STOP, do not paper over it with a fallback default. Verify: clean.
- [ ] **Step 4: per-equation weave/degree/reindexing round-trip** — the grouping-by-`equationIdx` and
  sorting-by-`position`/`slot` in Task B's decode must recover exactly Task A's construction order.
  Likely needs a `List.mergeSort`/`List.range`-indexed rebuild lemma analogous to
  `map_eq_range_map_getD` (`Realize.lean:173`) or `foldl_preserves_inv` (used for B.1 in the
  multioutput plan) — reuse those helpers rather than re-deriving from scratch.
- [ ] **Step 5: assemble the full theorem** from Steps 1–4 via `List.ext_getElem`/`congr` on each
  `ThreadedComposed` field. Verify: `lean_verify LeanNCD.AcsetCodec.toThreadedComposed_fromThreadedComposed`
  shows `[propext, Classical.choice, Quot.sound]` (no `sorryAx`); full `lake build` green.
- [ ] **Step 6: Commit** (one commit per step above is fine if `lake build` stays green with the
  remaining steps scaffolded `sorry`; final commit message):
  ```bash
  git add leanncd/LeanNCD/Bridge/AcsetCodec.lean
  git commit -m "feat(acset): prove toThreadedComposed ∘ fromThreadedComposed = id"
  ```

### Task D: `realizeSBr`

**Files:**
- Modify: `leanncd/LeanNCD/Bridge/SBr.lean`

**Interfaces:**
- Consumes: `toThreadedComposed` (Task B), `ThreadedComposed.WellFormed`/`realize`
  (`Bridge/Realize.lean`, already sorry-free).
- Produces: `noncomputable def realizeSBr (s : Acset.SBrInstance) : Σ (dom cod : BrObj), BrMorph dom cod`.

- [ ] **Step 1: Implement via classical case-split** (no `Decidable` instance needed — `noncomputable`
  already, use `open Classical in` / `Classical.byCases`):
  ```lean
  noncomputable def realizeSBr (s : Acset.SBrInstance) : Σ (dom cod : BrObj), BrMorph dom cod :=
    let tc := toThreadedComposed s
    if h : tc.WellFormed then realize tc h else ⟨[], [], BrMorph.idm []⟩
  ```
  Verify: `lake env lean leanncd/LeanNCD/Bridge/SBr.lean` compiles (the `if h :` on a
  non-`Decidable` `Prop` typechecks via `Classical.propDecidable`, already the ambient pattern
  elsewhere in the bridge — confirm by checking how `finalPiece`'s `by_cases hz : tc.steps.length = 0`
  is set up, though that one IS decidable; if `dite` doesn't elaborate directly on `tc.WellFormed`,
  add `have : Decidable tc.WellFormed := Classical.propDecidable _` locally).
- [ ] **Step 2: Commit.**
  ```bash
  git add leanncd/LeanNCD/Bridge/SBr.lean
  git commit -m "feat(acset): realizeSBr — decode + replay the realize fold"
  ```

### Task E: `realize_fromThreadedComposed_agree` + close-out

**Files:**
- Modify: `leanncd/LeanNCD/Bridge/Agreement.lean`

**Interfaces:**
- Consumes: `toThreadedComposed_fromThreadedComposed` (Task C), `realizeSBr` (Task D).
- Produces: `theorem realize_fromThreadedComposed_agree (tc : ThreadedComposed) (h : tc.WellFormed) :
  realize tc h = realizeSBr (fromThreadedComposed tc)` — sorry-free. `agree_dom`/`agree_cod` need no
  code change (already real `congr_arg` proofs); they simply stop being transitively-sorry.

- [ ] **Step 1: Prove the theorem.**
  ```lean
  theorem realize_fromThreadedComposed_agree (tc : ThreadedComposed) (h : tc.WellFormed) :
      realize tc h = realizeSBr (fromThreadedComposed tc) := by
    unfold realizeSBr
    rw [toThreadedComposed_fromThreadedComposed]
    simp only [dif_pos h]
  ```
  (The `dif_pos h` branch yields `realize tc h'` for whatever proof term the `dite`/`Classical`
  machinery manufactures; Lean 4's kernel-level proof irrelevance for `Prop` makes this
  definitionally/propositionally equal to `realize tc h` with no extra lemma — if `simp` doesn't close
  it outright, `exact rfl` or `congrArg (realize tc) (Subsingleton.elim _ h)` will.)
- [ ] **Step 2: Full verification pass.**
  - `lake build` — full project, green.
  - `lean_verify LeanNCD.realize_fromThreadedComposed_agree` ⇒ `[propext, Classical.choice, Quot.sound]`.
  - `lean_verify LeanNCD.agree_dom` / `LeanNCD.agree_cod` ⇒ same, now genuinely sorry-free (not just
    elaborating).
  - `grep -rn sorry leanncd/LeanNCD/` ⇒ **empty**. This is the whole-project sorry-free milestone.
- [ ] **Step 3: Update docs.**
  - `SORRY_INVENTORY.md`: mark the Milestone E2b "Named obligations" section fully resolved.
  - `2026-06-25-wellformed-forall-p.md`/`2026-06-26-multioutput-impl-plan.md`: no change needed (they
    already correctly scope §8.2 as separate from `compile_wellFormed`).
- [ ] **Step 4: `/lean4:checkpoint`, then commit.**
  ```bash
  git add leanncd/LeanNCD/Bridge/Agreement.lean leanncd/SORRY_INVENTORY.md
  git commit -m "feat(acset): realize_fromThreadedComposed_agree — Prop 8 proved, project sorry-free"
  ```

### Task F: regression test

**Files:**
- Create: `leanncd/test/Bridge/AcsetCodecTest.lean`

**Interfaces:**
- Consumes: `fromThreadedComposed`/`toThreadedComposed`/`realizeSBr` (Tasks A/B/D), the §12.1 example
  `ThreadedComposed`s already built by `tl!{}` in `test/DSL/CompileExamplesTest.lean`.

- [ ] **Step 1: Write round-trip `#guard`s** for at least the coupled-scan and simple-scan §12.1
  examples (the two that exercise multi-output/multi-input routing):
  ```lean
  #guard toThreadedComposed (fromThreadedComposed coupledScanExample) == coupledScanExample
  #guard toThreadedComposed (fromThreadedComposed simpleScanExample) == coupledScanExample  -- (fix name)
  ```
  (Needs `DecidableEq ThreadedComposed`, already `deriving`d on `ThreadedComposed`/`BrBaseP`/`Wire` in
  `Target.lean` — confirm `==` resolves via that instance, not `Repr`-based comparison.)
- [ ] **Step 2: Run.** `lake build` — the `#guard`s fire at elaboration; a failing guard is a build
  error, not a runtime failure.
- [ ] **Step 3: Commit.**
  ```bash
  git add leanncd/test/Bridge/AcsetCodecTest.lean
  git commit -m "test(acset): round-trip guards for fromThreadedComposed/toThreadedComposed"
  ```

---

## Sequencing summary

```
A   fromThreadedComposed (encode)                 — new file + Agreement.lean wiring
B   toThreadedComposed (decode)                    — same file
C   toThreadedComposed_fromThreadedComposed        — the hard proof (expect B.7-style sub-fights)
D   realizeSBr                                     — SBr.lean, small once C exists
E   realize_fromThreadedComposed_agree + close-out — Agreement.lean, ~one line of proof given C+D
F   regression #guards                             — new test file
```

## Risks / watch-items

- **StMatP → SampleRow encoding generality (Task A Step 2).** `SampleRow` is one-coeff-per-row; a
  dense `StMatP` row needs multiple `SampleRow`s. Confirmed workable (no schema change), but Task C's
  round-trip proof needs a real per-row list-reconstruction lemma — budget real proof time here, not
  just plumbing.
- **`BrOp`/`OpTag` cardinality mismatch (9 vs. 10, Task A Step 2 note).** Do NOT rely on `operatorTag`
  alone if two `BrOp`s must map to the same `OpTag` — use the `elementwiseFn`-string escape hatch (or
  find a cleaner single-field scheme during implementation) so `opTagToBrOp` is a true inverse; a
  non-injective encoding silently breaks Task C, not loudly.
- **`realizeSBr`'s missing `WellFormed` hypothesis (Task D).** Confirmed intentional by re-reading the
  signature — `realizeSBr` must be TOTAL over arbitrary (possibly garbage) `SBrInstance` values, unlike
  `realize`. The classical `dite` degenerate-default branch is the correct shape; don't try to add a
  hypothesis to `realizeSBr`'s signature to "simplify" the proof — that would change the theorem `SBr.lean`
  already commits to.
- **Proof irrelevance assumption (Task E).** Double-check with `lean_goal`/`lean_diagnostic_messages`
  that Lean 4's defeq proof irrelevance actually discharges the `dif_pos h` mismatch with `rfl`/`simp`
  before assuming it — if it doesn't reduce automatically, fall back to `Subsingleton.elim`
  (`Prop`s are `Subsingleton` by `proofIrrel`, a one-line detour, not a design change).

# §8.2 acset agreement — `fromThreadedComposed` / `realizeSBr` / Prop 8 — Implementation Plan

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

**Interfaces:**
- Produces: `axisUidFor : Nat → Nat → Acset.AxisUID := fun i k => ⟨.rawAxis, Nat.pair i k⟩` (step `i`,
  within-step degree-axis position `k`; `Nat.pair` is Mathlib's injective pairing, `Nat.pair`/
  `Nat.unpair`, so distinct `(i,k)` never collide — no arbitrary bound assumption).
- Produces: `brOpToOpTag : BrOp → Acset.OpTag` (fixed injective map, documented as internal-only, e.g.
  `contract↦identity, maxreduce↦elementwise, scatter↦linear, relu↦elementwise·¹, softmax↦softmax,
  normalize↦normalize, scan↦addition, scanAffine↦embedding, scanPre↦weightedTriangularLower` — ¹NOTE:
  `maxreduce` and `relu` collide on `elementwise`; disambiguate by encoding the ORIGINAL `BrOp`
  constructor index as a decimal string in `elementwiseFn` instead, e.g. `some "brop:1"` for
  `maxreduce`, so `opTagToBrOp` is a true left-inverse — do NOT rely on `operatorTag` alone. Revise
  this sub-scheme during implementation if a cleaner single-field encoding presents itself; the only
  hard requirement is that Task C's round-trip lemma holds.)
- Produces: `wireLabel (w : Wire) : String := match w with | .external k => s!"ext:{k}" | .internal j s
  => s!"int:{j}:{s}"` and its parser `parseWireLabel : String → Option Wire`.
- Produces: `nameOfSizeExpr : SizeExpr → Option String := fun | .var "_" => none | .var s => some s | _
  => none` — the decode-side inverse of the codebase-wide `AxisP.mk (some x) (SizeExpr.var x)` /
  `AxisP.mk none (SizeExpr.var "_")` invariant (see Design Decision CORRECTION). Used by Task B, not
  Task A itself (Task A stores the real `SizeExpr` verbatim; it never needs to derive a name).
- Produces: `fromThreadedComposed (tc : ThreadedComposed) : Acset.SBrInstance`.

- [ ] **Step 1: Add the codec file skeleton with `axisUidFor`, `brOpToOpTag`, `wireLabel`/
  `parseWireLabel`, each as a standalone sorry-free total function (no `ThreadedComposed` dependency
  yet).** Verify: `lake env lean leanncd/LeanNCD/Bridge/AcsetCodec.lean` compiles with no errors.

- [ ] **Step 2: Implement `fromThreadedComposed` per-step.** For step `i` with `BrBaseP` `b` and reads
  `tc.routing.getD i []`:
  - One `EquationRow ⟨i, none⟩`.
  - For each output slot `s < b.outputWeaves.length`: one `ArrayRow` at `slot := s`, `isInput :=
    false`, `operatorTag := some (brOpToOpTag b.op)`, plus one `ArrayAxisRow` per `fixed` axis in
    `b.outputWeaves.getD s []` (position = index within that weave, `axisUid := axisUidFor i` of the
    matching **degree** position — recovered via `List.findIdx?` against `b.degree`, since B.7
    guarantees per-slot output axes are a sub-list of degree in order; `isTarget := false`).
  - For each read `j`, wire `w := (tc.routing.getD i []).getD j (.external 0)`: one `ArrayRow` at
    `slot := b.outputWeaves.length + j`, `isInput := true`, `wireLabel := some (wireLabel w)`, plus one
    `ArrayAxisRow` per `fixed` axis in `b.inputWeaves.getD j []` (`isTarget := true` for axes NOT in
    `b.degree`, i.e. contracted; else `false`).
  - `SampleRow`s: for reindexing `b.reindexings.getD j default : StMatP` (`codLen × domLen` matrix +
    bias), for each `(c, d)` with `coeffs.getD c [] |>.getD d 0 ≠ 0`: one `SampleRow ⟨i, j, axisUidFor
    i d, axisUidFor i c, coeffs[c][d], bias.getD c 0⟩` (bias repeated redundantly per nonzero-coeff row
    for slot `(i,j,c)`; if a `c` has NO nonzero coeffs, one `SampleRow` with `coeff := 0, offset :=
    bias.getD c 0` so pure-bias axes are not silently dropped).
  - `axisSizes`: one `(axisUidFor i k, (b.degree.getD k default).size)` per degree position `k`,
    unioned (`List.union` or plain `++` — duplicates across steps are impossible since `axisUidFor` is
    injective in `i`) across all steps.
  Concatenate all steps' rows (`List.flatMap` over `tc.steps.zipIdx`, threading the routing list in
  parallel via `tc.routing.getD i []`).
  Verify: `lake env lean leanncd/LeanNCD/Bridge/AcsetCodec.lean`; `#eval fromThreadedComposed <a
  concrete tc from tl!{}>` on the §12.1 coupled-scan example — sanity-check table row counts by hand
  (steps × (outputs+inputs) rows, etc.) before moving on.

- [ ] **Step 3: Wire up `Agreement.lean`.** Replace `noncomputable def fromThreadedComposed (tc :
  ThreadedComposed) : Acset.SBrInstance := sorry` with `:= AcsetCodec.fromThreadedComposed tc` (import
  `LeanNCD.Bridge.AcsetCodec`). Verify: `lake build` green; `grep -n sorry
  leanncd/LeanNCD/Bridge/Agreement.lean` shows only `realize_fromThreadedComposed_agree` (and
  `agree_dom`/`agree_cod`, which are not literal sorries but transitively depend on it).

- [ ] **Step 4: Commit.**
  ```bash
  git add leanncd/LeanNCD/Bridge/AcsetCodec.lean leanncd/LeanNCD/Bridge/Agreement.lean
  git commit -m "feat(acset): fromThreadedComposed — systematic encode of ThreadedComposed as SBrInstance"
  ```

### Task B: `toThreadedComposed` (the decode direction)

**Files:**
- Modify: `leanncd/LeanNCD/Bridge/AcsetCodec.lean`

**Interfaces:**
- Consumes: `axisUidFor`, `brOpToOpTag`/its intended inverse `opTagToBrOp`, `parseWireLabel` (Task A).
- Produces: `toThreadedComposed (s : Acset.SBrInstance) : ThreadedComposed` — total, defaulting on
  malformed/partial tables (never throws; unrecognized rows are skipped, matching the "no hypothesis on
  `realizeSBr`'s input" requirement noted in `SBr.lean`'s doc comment).

- [ ] **Step 1: Group `s.equations`/`s.arrays`/`s.arrayAxes`/`s.samples` by `equationIdx` (`Std.HashMap
  Nat (List ArrayRow)` etc., built via one fold each — mirror the `goodExtState`-style fold idiom
  already used in `RouteSpec.lean`).** For each `equationIdx` present, reconstruct one `BrBaseP`:
  - `outputWeaves`: rows with `isInput = false`, sorted by `slot`, each weave rebuilt from its
    `ArrayAxisRow`s sorted by `position` — per row, `size := s.axisSizes.lookup axisUid` (the verbatim
    stored `SizeExpr`) and `name := nameOfSizeExpr size` (see the CORRECTION in Design Decision above —
    `.fixed ⟨nameOfSizeExpr size, size⟩`, NOT `⟨none, size⟩`).
  - `inputWeaves`/wires: rows with `isInput = true`, sorted by `slot`; each row's `wireLabel` decodes
    via `parseWireLabel` to the `Wire` at that read position.
  - `degree`: reconstruct from the UNION of all `axisUid`s appearing in this equation's rows whose
    `AxisUID.id` decodes (via `Nat.unpair`) to `(equationIdx, _)`, ordered by the `k` component.
  - `reindexings`: per input slot `j`, rebuild the `codLen × domLen` matrix from that slot's
    `SampleRow`s (`codLen := (inputWeaves.getD j []).length`... actually `weaveRank`, since only
    `fixed` slots are cod positions — match Task A's convention exactly) plus bias from the pure-bias
    rows.
  - `op`: `opTagToBrOp` applied to the output row's `operatorTag`, falling back to parsing
    `elementwiseFn`'s `"brop:<n>"` tag when the `OpTag` collision case (Task A Step 2 note) applies.
- [ ] **Step 2: Assemble `ThreadedComposed`** — `steps` = the reconstructed `BrBaseP`s ordered by
  `equationIdx`; `routing` = each step's decoded wire list (already built above); `nExternal` =
  `1 + ` the max `k` appearing in any `"ext:<k>"` wireLabel (or `0` if none).
  Verify: `lake env lean leanncd/LeanNCD/Bridge/AcsetCodec.lean` compiles.
- [ ] **Step 3: Commit.**
  ```bash
  git add leanncd/LeanNCD/Bridge/AcsetCodec.lean
  git commit -m "feat(acset): toThreadedComposed — decode SBrInstance back to a routed DAG"
  ```

### Task C: round-trip lemma

**Files:**
- Modify: `leanncd/LeanNCD/Bridge/AcsetCodec.lean`

**Interfaces:**
- Consumes: `fromThreadedComposed` (Task A), `toThreadedComposed` (Task B).
- Produces: `theorem toThreadedComposed_fromThreadedComposed (tc : ThreadedComposed) :
  toThreadedComposed (fromThreadedComposed tc) = tc`.

This is the hardest task — expect it to be the B.7-equivalent of this plan (an order/grouping fight,
not a deep category-theory one). Likely needed sub-lemmas, proved bottom-up, one commit each:
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

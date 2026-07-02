-- LeanNCD/Bridge/AcsetCodec.lean
import Mathlib.Data.Nat.Pairing   -- Nat.pair/unpair
import LeanNCD.Acset.SBrInstance  -- Acset.SBrInstance and its row types
import LeanNCD.DSL.Target         -- ThreadedComposed, BrBaseP, Wire, BrOp
import LeanNCD.DSL.SizeExpr       -- SizeExpr
import LeanNCD.Bridge.Realize     -- ThreadedComposed.WellFormed (round-trip theorem, Task C)

/-! ### §8.2 acset codec — `ThreadedComposed ↔ Acset.SBrInstance`

A systematic/synthetic encoding (see `2026-07-01-acset-agreement-impl-plan.md`): no attempt at
human-readable names or Python `OpTag`/`from_tensor_program` fidelity, just enough to make
`toThreadedComposed (fromThreadedComposed tc) = tc` (Task C) provable. -/

namespace LeanNCD.AcsetCodec

open LeanNCD.Acset

/-- One `AxisUID` per `(step index, array-slot, within-slot position)` triple. `slot` ranges over a
    per-step-uniform numbering: `0 .. outLen-1` = output weaves, `outLen .. outLen+inLen-1` = input
    weaves (`outLen + j` for read `j`), and `outLen+inLen` (one past the last real slot) is the
    reserved **degree slot**. Each of these is an independent axis-position space — the E2a
    presentation carries no uid linking a weave's fixed axis back to a specific degree position
    ("the dependent typing is dropped", `Target.lean:63`), so they must round-trip independently.
    `Nat.pair` (nested) is injective, so distinct `(i, slot, pos)` never collide. -/
def axisUidFor3 (i slot pos : Nat) : Acset.AxisUID := ⟨.rawAxis, Nat.pair i (Nat.pair slot pos)⟩

/-- A `Nat` as a run of `n` `'1'` characters. Numbers here (step/slot/external indices) are small,
    so unary is fine in practice; the payoff is a trivial round-trip proof (`unaryToNat_natToUnary`
    below) versus a from-scratch decimal-digit induction (`String.toNat?`/`toString` have no
    ready-made round-trip lemma in core/Mathlib/Batteries — checked before choosing this). -/
def natToUnary (n : Nat) : String := String.ofList (List.replicate n '1')

/-- Decode: the length of a unary run IS the encoded number. -/
def unaryToNat (s : String) : Nat := s.length

@[simp] theorem unaryToNat_natToUnary (n : Nat) : unaryToNat (natToUnary n) = n := by
  simp [unaryToNat, natToUnary, String.length]

/-- Encode a `Wire` as a single `Nat` via `Nat.pair` (tag `0`/`1` selects external/internal;
    `.internal` pairs its two fields again). Avoids `String.splitOn` entirely — that has no
    established `++`-interaction lemmas in core/Batteries yet (`-- TODO: splitOn` in
    `Batteries/Data/String/Lemmas.lean`), whereas `Nat.pair`/`Nat.unpair` round-trip cleanly
    (`Nat.unpair_pair`, already used for `axisUidFor`). -/
def wireCode : Wire → Nat
  | .external k   => Nat.pair 0 k
  | .internal j s => Nat.pair 1 (Nat.pair j s)

def wireOfCode (n : Nat) : Wire :=
  let (tag, rest) := Nat.unpair n
  if tag = 0 then .external rest else
    let (j, s) := Nat.unpair rest
    .internal j s

@[simp] theorem wireOfCode_wireCode (w : Wire) : wireOfCode (wireCode w) = w := by
  cases w <;> simp [wireCode, wireOfCode]

/-- The wire, as a unary-encoded string (single `Nat`, no delimiters needed). -/
def wireLabel (w : Wire) : String := natToUnary (wireCode w)

def parseWireLabel (s : String) : Option Wire := some (wireOfCode (unaryToNat s))

@[simp] theorem parseWireLabel_wireLabel (w : Wire) : parseWireLabel (wireLabel w) = some w := by
  simp [parseWireLabel, wireLabel]

/-- The constructor index of a `BrOp` (0..8) — a systematic internal tag, NOT a mapping into
    Python's unrelated `Acset.OpTag` (which has different cardinality/semantics; see the Design
    Decision in the plan doc). -/
def brOpIdx : BrOp → Nat
  | .contract   => 0
  | .maxreduce  => 1
  | .scatter    => 2
  | .relu       => 3
  | .softmax    => 4
  | .normalize  => 5
  | .scan       => 6
  | .scanAffine => 7
  | .scanPre    => 8

/-- Inverse of `brOpIdx` on `0..8`; defaults to `.contract` outside that range (never hit on data
    produced by `brOpIdx`, only relevant for totality on arbitrary/garbage input). -/
def brOpOfIdx : Nat → BrOp
  | 0 => .contract
  | 1 => .maxreduce
  | 2 => .scatter
  | 3 => .relu
  | 4 => .softmax
  | 5 => .normalize
  | 6 => .scan
  | 7 => .scanAffine
  | _ => .scanPre

@[simp] theorem brOpOfIdx_brOpIdx (op : BrOp) : brOpOfIdx (brOpIdx op) = op := by
  cases op <;> rfl

/-- Decode-side inverse of the codebase-wide `AxisP.mk (some x) (SizeExpr.var x)` /
    `AxisP.mk none (SizeExpr.var "_")` construction invariant (verified by exhaustive grep of every
    `AxisP` construction site in `LeanNCD/` — see the Design Decision CORRECTION in the plan doc). -/
def nameOfSizeExpr : SizeExpr → Option String
  | .var "_" => none
  | .var s   => some s
  | _        => none

/-! ### `fromThreadedComposed` — the encode direction -/

/-- One `ArrayAxisRow` per weave POSITION (not just `fixed` ones) — a `.tiled` slot gets the
    reserved sentinel `⟨.natAxis, 0⟩` (`.natAxis` is otherwise unused by this codec, so it
    unambiguously flags "tiled, ignore `id`"). Emitting a row for every position (rather than
    skipping `.tiled` ones) lets the decoder recover the weave's total length — a weave ending in
    `.tiled` slots would otherwise lose that information. `isTarget` is unused by this codec
    (nothing in the round-trip needs it); set `false` uniformly. -/
def encodeAxisRows (i slot : Nat) (w : WeaveShapeP) : List ArrayAxisRow :=
  (List.range w.length).map fun p =>
    match w.getD p .tiled with
    | .fixed _ => { equationIdx := i, arraySlot := slot, axisUid := axisUidFor3 i slot p,
                     isTarget := false, position := p }
    | .tiled   => { equationIdx := i, arraySlot := slot, axisUid := ⟨.natAxis, 0⟩,
                     isTarget := false, position := p }

/-- The `axisSizes` contribution of one weave: one `(uid, size)` pair per `fixed` position (`tiled`
    slots have no axis, hence no size). -/
def encodeAxisSizes (i slot : Nat) (w : WeaveShapeP) : List (AxisUID × SizeExpr) :=
  (List.range w.length).filterMap fun p =>
    match w.getD p .tiled with
    | .fixed a => some (axisUidFor3 i slot p, a.size)
    | .tiled   => none

/-- The weave-positions (in order) that are `fixed` — the reindexing matrix's `codLen` cod-index `c`
    refers to the `c`-th fixed position, not literally weave-position `c` (tiled slots don't get a
    matrix row). -/
def fixedPositions (w : WeaveShapeP) : List Nat :=
  (List.range w.length).filter fun p =>
    match w.getD p .tiled with
    | .fixed _ => true
    | .tiled   => false

/-- `SampleRow`s for one input's reindexing `m : StMatP` (`codLen × domLen` matrix + bias),
    reindexingSlot = the input's array slot (`outLen + j`, matching its `ArrayRow.slot`), source
    axes in the DEGREE slot (`degSlot`), target axes in the input weave's slot (`arraySlot`) at its
    `c`-th fixed position. One row per nonzero matrix entry; a cod-position with no nonzero entries
    (pure bias, no source axis) gets one self-referencing (`srcUid = tgtUid`) `coeff := 0` row so the
    bias is not silently dropped. `offset` is redundantly repeated on every row sharing `(slot,
    tgtUid)` — decode reads it off any one of them. -/
def encodeReindexing (i degSlot arraySlot : Nat) (inW : WeaveShapeP) (m : StMatP) :
    List SampleRow :=
  let fps := fixedPositions inW
  (List.range m.codLen).flatMap fun c =>
    let tgtUid := axisUidFor3 i arraySlot (fps.getD c 0)
    let row := m.coeffs.getD c []
    let nz := (List.range m.domLen).filter fun d => row.getD d 0 ≠ 0
    let offset := m.bias.getD c 0
    match nz with
    | [] => [{ equationIdx := i, reindexingSlot := arraySlot, srcUid := tgtUid, tgtUid := tgtUid,
                coeff := 0, offset := offset }]
    | _  => nz.map fun d =>
        { equationIdx := i, reindexingSlot := arraySlot, srcUid := axisUidFor3 i degSlot d,
          tgtUid := tgtUid, coeff := row.getD d 0, offset := offset }

/-- The default (unused-field) `ArrayRow` shared by output/input rows — only `slot`/`isInput`/
    `elementwiseFn` (op index, output slot 0 only — see `fromThreadedComposed`)/`wireLabel` (input
    rows only) ever carry real content; everything else is `none`/`.reals` because it carries
    Python-specific semantic distinctions (masked-softmax predicates, linear-layer bias flags) that
    `BrOp` doesn't make and nothing here needs to reconstruct. -/
def blankArrayRow (i slot : Nat) (isInput : Bool) : ArrayRow :=
  { equationIdx := i, slot := slot, name := none, isInput := isInput, operatorTag := none,
    normAxis := none, datatypeTag := .reals, maxValue := none, bias := none,
    elementwiseFn := none, opPredicate := none, wireLabel := none }

/-- Encode one step (`i`, `BrBaseP` `b`, its routing reads) into its `SBrInstance` row
    contribution. `degSlot := outLen + inLen` (one past the last real array slot). -/
def encodeStep (i : Nat) (b : BrBaseP) (reads : List Wire) : SBrInstance :=
  let outLen := b.outputWeaves.length
  let inLen  := reads.length
  let degSlot := outLen + inLen
  let outputArrays := (List.range outLen).map fun s => blankArrayRow i s false
  let outputAxisRows := (List.range outLen).flatMap fun s =>
    encodeAxisRows i s (b.outputWeaves.getD s [])
  let outputAxisSizes := (List.range outLen).flatMap fun s =>
    encodeAxisSizes i s (b.outputWeaves.getD s [])
  let inputArrays := (List.range inLen).map fun j =>
    { blankArrayRow i (outLen + j) true with
        wireLabel := some (wireLabel (reads.getD j (.external 0))) }
  let inputAxisRows := (List.range inLen).flatMap fun j =>
    encodeAxisRows i (outLen + j) (b.inputWeaves.getD j [])
  let inputAxisSizes := (List.range inLen).flatMap fun j =>
    encodeAxisSizes i (outLen + j) (b.inputWeaves.getD j [])
  let degWeave : WeaveShapeP := b.degree.map .fixed
  let degAxisRows := encodeAxisRows i degSlot degWeave
  let degAxisSizes := encodeAxisSizes i degSlot degWeave
  let samples := (List.range inLen).flatMap fun j =>
    encodeReindexing i degSlot (outLen + j) (b.inputWeaves.getD j []) (b.reindexings.getD j default)
  { axisSizes := outputAxisSizes ++ inputAxisSizes ++ degAxisSizes
    equations := [{ equationIdx := i, lhsName := some (natToUnary (brOpIdx b.op)) }]
    arrays    := outputArrays ++ inputArrays
    arrayAxes := outputAxisRows ++ inputAxisRows ++ degAxisRows
    samples   := samples }

/-- The per-step `SBrInstance`s (one per step, in step order), each tagged with its step index.
    Factored out so the round-trip proof (Task C) can reason about the fieldwise flatten below and
    the per-step isolation separately. -/
def stepInsts (tc : ThreadedComposed) : List SBrInstance :=
  tc.steps.zipIdx.map fun (b, i) => encodeStep i b (tc.routing.getD i [])

/-- Encode a `ThreadedComposed` as its acset-table twin — a systematic/synthetic encoding (see
    `2026-07-01-acset-agreement-impl-plan.md`), one "equation" per step. Fieldwise flatten of
    `stepInsts` (rather than `foldl append`) so the decode-side filters reduce cleanly in the
    round-trip proof. Kept in the `AcsetCodec` namespace (not bare `LeanNCD`) so it doesn't collide
    with `Agreement.lean`'s `LeanNCD.fromThreadedComposed` declaration until that file is wired to
    call this one (Task A Step 3). -/
def fromThreadedComposed (tc : ThreadedComposed) : Acset.SBrInstance :=
  let insts := stepInsts tc
  { axisSizes := (insts.map (·.axisSizes)).flatten
    equations := (insts.map (·.equations)).flatten
    arrays    := (insts.map (·.arrays)).flatten
    arrayAxes := (insts.map (·.arrayAxes)).flatten
    samples   := (insts.map (·.samples)).flatten }

/-! ### `toThreadedComposed` — the decode direction

None of the `Acset` row types derive `BEq` (only `DecidableEq`), so equality tests below go through
`decide` rather than `==`. -/

/-- Look up an axis's stored `SizeExpr` by uid; `.lit 0` on a miss (unreachable for real encoder
    output — only relevant for garbage/malformed `s`, where `toThreadedComposed` must stay total). -/
def lookupSize (s : SBrInstance) (u : AxisUID) : SizeExpr :=
  ((s.axisSizes.find? fun p => decide (p.1 = u)).map Prod.snd).getD (.lit 0)

/-- Rebuild one weave (output/input/degree slot) from its `ArrayAxisRow`s. Length = row count for
    that `(equationIdx, arraySlot)` (Task A emits exactly one row per position, contiguous from `0`).
    A `.natAxis`-tagged row is the `.tiled` sentinel (see `encodeAxisRows`); otherwise `.fixed`, name
    recovered from the stored `SizeExpr` via `nameOfSizeExpr`. A missing position (garbage input)
    defaults to `.tiled`. -/
def decodeWeaveAt (s : SBrInstance) (i slot : Nat) : WeaveShapeP :=
  let rows := s.arrayAxes.filter fun r => decide (r.equationIdx = i ∧ r.arraySlot = slot)
  (List.range rows.length).map fun p =>
    match rows.find? fun r => decide (r.position = p) with
    | some r => match r.axisUid.type with
        | .natAxis => .tiled
        | _        => let size := lookupSize s r.axisUid; .fixed ⟨nameOfSizeExpr size, size⟩
    | none   => .tiled

/-- Rebuild one input's reindexing matrix from its `SampleRow`s — the exact inverse of
    `encodeReindexing`: `fps := fixedPositions inW` recovers the same cod-index ↔ weave-position
    correspondence Task A used to encode, so `coeffs[c][d]`/`bias[c]` are read straight off the
    `SampleRow` with matching `(tgtUid, srcUid)` / `tgtUid`, defaulting to `0` (matches `encodeReindexing`
    only emitting nonzero entries). -/
def decodeReindexing (s : SBrInstance) (i degSlot arraySlot domLen : Nat) (inW : WeaveShapeP) :
    StMatP :=
  let fps := fixedPositions inW
  let codLen := fps.length
  let mySamples := s.samples.filter fun r => decide (r.equationIdx = i ∧ r.reindexingSlot = arraySlot)
  let coeffs := (List.range codLen).map fun c =>
    let tgtUid := axisUidFor3 i arraySlot (fps.getD c 0)
    (List.range domLen).map fun d =>
      let srcUid := axisUidFor3 i degSlot d
      ((mySamples.find? fun r => decide (r.tgtUid = tgtUid ∧ r.srcUid = srcUid)).map (·.coeff)).getD 0
  let bias := (List.range codLen).map fun c =>
    let tgtUid := axisUidFor3 i arraySlot (fps.getD c 0)
    ((mySamples.find? fun r => decide (r.tgtUid = tgtUid)).map (·.offset)).getD 0
  { domLen := domLen, codLen := codLen, coeffs := coeffs, bias := bias }

/-- Rebuild one step's `BrBaseP` and its routing reads from `s`'s rows for `equationIdx = i`.
    `outLen`/`inLen` come straight from counting `ArrayRow`s (Task A emits exactly one per real
    output/input slot); `degSlot := outLen + inLen`, matching `encodeStep`. -/
def decodeStep (s : SBrInstance) (i : Nat) : BrBaseP × List Wire :=
  let outputRows := s.arrays.filter fun a => decide (a.equationIdx = i ∧ a.isInput = false)
  let inputRows  := s.arrays.filter fun a => decide (a.equationIdx = i ∧ a.isInput = true)
  let outLen := outputRows.length
  let inLen  := inputRows.length
  let degSlot := outLen + inLen
  let outputWeaves := (List.range outLen).map fun sN => decodeWeaveAt s i sN
  let inputWeaves  := (List.range inLen).map fun j => decodeWeaveAt s i (outLen + j)
  let degree : StObjP := (decodeWeaveAt s i degSlot).map fun
    | .fixed a => a
    | .tiled   => default
  let reindexings := (List.range inLen).map fun j =>
    decodeReindexing s i degSlot (outLen + j) degree.length (inputWeaves.getD j [])
  let opIdxStr := ((s.equations.find? fun e => decide (e.equationIdx = i)).bind (·.lhsName)).getD ""
  let op := brOpOfIdx (unaryToNat opIdxStr)
  let reads := (List.range inLen).map fun j =>
    match inputRows.find? fun a => decide (a.slot = outLen + j) with
    | some a => (a.wireLabel.bind parseWireLabel).getD (.external 0)
    | none   => .external 0
  ({ op := op, degree := degree, inputWeaves := inputWeaves, outputWeaves := outputWeaves,
     reindexings := reindexings }, reads)

/-- Decode an `SBrInstance` back into a `ThreadedComposed`. Total over ANY `s` (no `WellFormed`-style
    hypothesis) — `realizeSBr` (Task D) needs this, per `SBr.lean`'s doc comment. `nExternal` is
    reconstructed as `1 + ` the max referenced external index (`0` if none); this equals the
    ORIGINAL `tc.nExternal` only when `tc` was `WellFormed` (an unreferenced external slot leaves no
    trace in any wire) — hence Task C's round-trip theorem takes that as a hypothesis. -/
def toThreadedComposed (s : SBrInstance) : ThreadedComposed :=
  let n := s.equations.length
  let decoded := (List.range n).map fun i => decodeStep s i
  let allWires := decoded.flatMap Prod.snd
  let extKs := allWires.filterMap fun w => match w with
    | .external k   => some k
    | .internal _ _ => none
  { steps := decoded.map Prod.fst
    routing := decoded.map Prod.snd
    nExternal := match extKs.max? with | some m => m + 1 | none => 0 }

/-! ### Round-trip (Task C): `toThreadedComposed ∘ fromThreadedComposed = id`

The decode exactly inverts the encode. Empirically verified on all five §12.1 example programs
(scratch eval); this is the formal proof. `WellFormed` is needed only for `nExternal` (an
unreferenced external slot leaves no trace in the routing, so `nExternal` is otherwise unrecoverable);
`steps`/`routing` round-trip unconditionally. -/
theorem toThreadedComposed_fromThreadedComposed (tc : ThreadedComposed) (h : tc.WellFormed) :
    toThreadedComposed (fromThreadedComposed tc) = tc := by
  sorry

end LeanNCD.AcsetCodec

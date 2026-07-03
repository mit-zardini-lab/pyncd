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

/-- The name-slot uid for a fixed axis: same `id` as its `axisUidFor3`, retagged `.normAxis` so it
    never collides with the size entry (`.rawAxis`) or the tiled sentinel (`.natAxis`). Stores the
    axis NAME faithfully & independently of its size (two-slot encoding), so the round trip recovers
    `AxisP.name` unconditionally — not only under the `size = .var name` "canonical axis" invariant. -/
def nameUidFor3 (i slot pos : Nat) : Acset.AxisUID := ⟨.normAxis, Nat.pair i (Nat.pair slot pos)⟩

/-- The `axisSizes` contribution of one weave: per `fixed` position, one `(rawAxis-uid, size)` entry,
    plus (when the axis is named) one `(normAxis-uid, .var name)` name entry. `tiled` slots contribute
    nothing. The two entries per axis are the two-slot encoding that makes name recovery unconditional. -/
def encodeAxisSizes (i slot : Nat) (w : WeaveShapeP) : List (AxisUID × SizeExpr) :=
  (List.range w.length).flatMap fun p =>
    match w.getD p .tiled with
    | .fixed a => match a.name with
        | some nm => [(axisUidFor3 i slot p, a.size), (nameUidFor3 i slot p, .var nm)]
        | none    => [(axisUidFor3 i slot p, a.size)]
    | .tiled   => []

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

/-- Recover an axis's `name` from its dedicated `.normAxis`-tagged name entry (two-slot encoding, see
    `encodeAxisSizes`/`nameUidFor3`). Keyed on the same `id` as the size uid `u`, retagged `.normAxis`.
    Absence of the entry ⇒ `none` (faithfully encodes `AxisP.name = none`). Independent of the size,
    so name recovery does not rely on any `size = .var name` invariant. -/
def lookupName (s : SBrInstance) (u : AxisUID) : Option String :=
  match (s.axisSizes.find? fun p => decide (p.1 = ⟨.normAxis, u.id⟩)).map Prod.snd with
  | some (.var nm) => some nm
  | _              => none

/-- Rebuild one weave (output/input/degree slot) from its `ArrayAxisRow`s. Length = row count for
    that `(equationIdx, arraySlot)` (Task A emits exactly one row per position, contiguous from `0`).
    A `.natAxis`-tagged row is the `.tiled` sentinel (see `encodeAxisRows`); otherwise `.fixed`, with
    `size` from `lookupSize` and `name` from `lookupName` (the dedicated name slot). A missing position
    (garbage input) defaults to `.tiled`. -/
def decodeWeaveAt (s : SBrInstance) (i slot : Nat) : WeaveShapeP :=
  let rows := s.arrayAxes.filter fun r => decide (r.equationIdx = i ∧ r.arraySlot = slot)
  (List.range rows.length).map fun p =>
    match rows.find? fun r => decide (r.position = p) with
    | some r => match r.axisUid.type with
        | .natAxis => .tiled
        | _        => .fixed ⟨lookupName s r.axisUid, lookupSize s r.axisUid⟩
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

/-- Isolation, general form (base offset): if every row of sublist `k` is tagged `key = b + k`, then
    filtering the flattened list by `key = i ∧ P` recovers `P`-filtered sublist `i - b` (empty when
    `i < b`). The engine behind `decodeStep`'s per-`equationIdx` filters recovering exactly one step's
    rows from the fieldwise-flattened `fromThreadedComposed` tables. -/
theorem filter_flatten_tagged_aux {β : Type*} (key : β → Nat) (i : Nat) (P : β → Bool) :
    ∀ (L : List (List β)) (b : Nat),
      (∀ k (hk : k < L.length), ∀ r ∈ L[k], key r = b + k) →
      (L.map (fun l => l.filter (fun r => decide (key r = i) && P r))).flatten
        = if b ≤ i then (L.getD (i - b) []).filter P else [] := by
  intro L
  induction L with
  | nil => intro b _; simp
  | cons hd tl ih =>
    intro b htag
    have hhd : ∀ r ∈ hd, key r = b := by
      intro r hr
      have := htag 0 (by simp) r (by simpa using hr)
      simpa using this
    have htl : ∀ k (hk : k < tl.length), ∀ r ∈ tl[k], key r = (b+1) + k := by
      intro k hk r hr
      have := htag (k+1) (by simp; omega) r (by simpa using hr)
      omega
    have ihtl := ih (b+1) htl
    simp only [List.map_cons, List.flatten_cons]
    have hfhd : (hd.filter (fun r => decide (key r = i) && P r))
        = if b = i then hd.filter P else [] := by
      by_cases hbi : b = i
      · subst hbi; rw [if_pos rfl]
        apply List.filter_congr
        intro r hr; rw [hhd r hr]; simp
      · rw [if_neg hbi]
        apply List.filter_eq_nil_iff.mpr
        intro r hr hq
        rw [hhd r hr] at hq
        simp [hbi] at hq
    rw [hfhd, ihtl]
    by_cases hbi : b = i
    · subst hbi; simp
    · by_cases hlt : b < i
      · rw [if_neg (by omega), if_pos (by omega), if_pos (by omega)]
        have : i - b = (i - (b+1)) + 1 := by omega
        rw [this, List.getD_cons_succ, List.nil_append]
      · rw [if_neg hbi, if_neg (by omega), if_neg (by omega)]
        simp

/-- Isolation at base 0 (the shape `decodeStep` needs): each sublist `k` tagged `key = k`. -/
theorem filter_flatten_tagged {β : Type*} (L : List (List β)) (key : β → Nat) (i : Nat)
    (P : β → Bool) (htag : ∀ k (hk : k < L.length), ∀ r ∈ L[k], key r = k) :
    (L.flatten.filter (fun r => decide (key r = i) && P r)) = (L.getD i []).filter P := by
  rw [List.filter_flatten]
  have := filter_flatten_tagged_aux key i P L 0 (by simpa using htag)
  simpa using this

/-! #### `stepInsts` indexing and per-step `equationIdx` tagging (the `htag` for isolation) -/

@[simp] theorem stepInsts_length (tc : ThreadedComposed) :
    (stepInsts tc).length = tc.steps.length := by
  simp [stepInsts]

theorem stepInsts_getElem (tc : ThreadedComposed) (k : Nat) (hk : k < tc.steps.length) :
    (stepInsts tc)[k]'(by simp [stepInsts]; omega)
      = encodeStep k (tc.steps[k]) (tc.routing.getD k []) := by
  simp only [stepInsts]
  rw [List.getElem_map, List.getElem_zipIdx]
  simp

theorem encodeAxisRows_eqIdx (i slot : Nat) (w : WeaveShapeP) :
    ∀ r ∈ encodeAxisRows i slot w, r.equationIdx = i := by
  intro r hr
  simp only [encodeAxisRows, List.mem_map] at hr
  rcases hr with ⟨p, _, rfl⟩
  split <;> rfl

theorem encodeStep_arrays_eqIdx (i : Nat) (b : BrBaseP) (reads : List Wire) :
    ∀ r ∈ (encodeStep i b reads).arrays, r.equationIdx = i := by
  intro r hr
  simp only [encodeStep] at hr
  rcases List.mem_append.1 hr with h | h
  · rcases List.mem_map.1 h with ⟨s, _, rfl⟩; rfl
  · rcases List.mem_map.1 h with ⟨j, _, rfl⟩; rfl

theorem encodeStep_arrayAxes_eqIdx (i : Nat) (b : BrBaseP) (reads : List Wire) :
    ∀ r ∈ (encodeStep i b reads).arrayAxes, r.equationIdx = i := by
  intro r hr
  simp only [encodeStep] at hr
  rcases List.mem_append.1 hr with h | h
  · rcases List.mem_append.1 h with h | h
    · rcases List.mem_flatMap.1 h with ⟨s, _, hs⟩; exact encodeAxisRows_eqIdx _ _ _ _ hs
    · rcases List.mem_flatMap.1 h with ⟨j, _, hs⟩; exact encodeAxisRows_eqIdx _ _ _ _ hs
  · exact encodeAxisRows_eqIdx _ _ _ _ h

theorem encodeStep_samples_eqIdx (i : Nat) (b : BrBaseP) (reads : List Wire) :
    ∀ r ∈ (encodeStep i b reads).samples, r.equationIdx = i := by
  intro r hr
  simp only [encodeStep] at hr
  rcases List.mem_flatMap.1 hr with ⟨j, _, hj⟩
  simp only [encodeReindexing] at hj
  rcases List.mem_flatMap.1 hj with ⟨c, _, hc⟩
  split at hc
  · simp only [List.mem_singleton] at hc; rw [hc]
  · rcases List.mem_map.1 hc with ⟨d, _, rfl⟩; rfl

theorem encodeStep_equations_eqIdx (i : Nat) (b : BrBaseP) (reads : List Wire) :
    ∀ r ∈ (encodeStep i b reads).equations, r.equationIdx = i := by
  intro r hr
  simp only [encodeStep, List.mem_singleton] at hr
  rw [hr]

/-- Generic per-field isolation applied to `fromThreadedComposed`: filtering any flattened table
    field by `equationIdx = i ∧ P` recovers step `i`'s own rows filtered by `P`, for
    `i < tc.steps.length`. Instantiate with `proj := SBrInstance.arrays` (etc.), the matching
    `equationIdx` accessor, and the field's tagging lemma. -/
theorem from_field_filter {β : Type*} (tc : ThreadedComposed) (i : Nat)
    (hi : i < tc.steps.length) (proj : SBrInstance → List β) (key : β → Nat) (P : β → Bool)
    (hproj : proj (fromThreadedComposed tc) = ((stepInsts tc).map proj).flatten)
    (htag : ∀ (k : Nat) (b : BrBaseP) (reads : List Wire),
      ∀ r ∈ proj (encodeStep k b reads), key r = k) :
    (proj (fromThreadedComposed tc)).filter (fun r => decide (key r = i) && P r)
      = (proj (encodeStep i tc.steps[i] (tc.routing.getD i []))).filter P := by
  rw [hproj]
  rw [filter_flatten_tagged ((stepInsts tc).map proj) key i P ?htag]
  case htag =>
    intro k hk r hr
    rw [List.getElem_map] at hr
    have hk' : k < tc.steps.length := by simpa using hk
    rw [stepInsts_getElem tc k hk'] at hr
    exact htag k _ _ r hr
  have hlen : i < ((stepInsts tc).map proj).length := by simp; omega
  rw [List.getD_eq_getElem _ _ hlen, List.getElem_map, stepInsts_getElem tc i hi]

/-! #### Array-count round trips (step 1): output/input `ArrayRow`s recover exactly. -/

/-- The output `ArrayRow`s decode filters out of `fromThreadedComposed tc` are exactly the ones
    `encodeStep` emitted for step `i`: one blank row per output weave. -/
theorem from_outputRows (tc : ThreadedComposed) (i : Nat) (hi : i < tc.steps.length) :
    (fromThreadedComposed tc).arrays.filter (fun a => decide (a.equationIdx = i ∧ a.isInput = false))
      = (List.range (tc.steps[i]).outputWeaves.length).map (fun s => blankArrayRow i s false) := by
  have hpred : (fun a : ArrayRow => decide (a.equationIdx = i ∧ a.isInput = false))
      = (fun a => decide (a.equationIdx = i) && decide (a.isInput = false)) := by
    funext a; rw [Bool.decide_and]
  rw [hpred, from_field_filter tc i hi SBrInstance.arrays (·.equationIdx)
        (fun a => decide (a.isInput = false)) rfl encodeStep_arrays_eqIdx]
  simp only [encodeStep, List.filter_append]
  rw [List.filter_eq_self.2, List.filter_eq_nil_iff.2, List.append_nil]
  · intro a ha; simp only [List.mem_map] at ha; obtain ⟨k, _, rfl⟩ := ha; simp [blankArrayRow]
  · intro a ha; simp only [List.mem_map] at ha; obtain ⟨k, _, rfl⟩ := ha; simp [blankArrayRow]

/-- The input `ArrayRow`s decode filters out are exactly `encodeStep`'s input rows: one per read,
    at slot `outLen + j`, carrying that read's `wireLabel`. -/
theorem from_inputRows (tc : ThreadedComposed) (i : Nat) (hi : i < tc.steps.length) :
    (fromThreadedComposed tc).arrays.filter (fun a => decide (a.equationIdx = i ∧ a.isInput = true))
      = (List.range (tc.routing.getD i []).length).map (fun j =>
          { blankArrayRow i ((tc.steps[i]).outputWeaves.length + j) true with
              wireLabel := some (wireLabel ((tc.routing.getD i []).getD j (.external 0))) }) := by
  have hpred : (fun a : ArrayRow => decide (a.equationIdx = i ∧ a.isInput = true))
      = (fun a => decide (a.equationIdx = i) && decide (a.isInput = true)) := by
    funext a; rw [Bool.decide_and]
  rw [hpred, from_field_filter tc i hi SBrInstance.arrays (·.equationIdx)
        (fun a => decide (a.isInput = true)) rfl encodeStep_arrays_eqIdx]
  simp only [encodeStep, List.filter_append]
  rw [List.filter_eq_nil_iff.2, List.filter_eq_self.2, List.nil_append]
  · intro a ha; simp only [List.mem_map] at ha; obtain ⟨k, _, rfl⟩ := ha; simp [blankArrayRow]
  · intro a ha; simp only [List.mem_map] at ha; obtain ⟨k, _, rfl⟩ := ha; simp [blankArrayRow]

/-- Output row count = number of output weaves (unblocks concrete slot indices `outLen + j`). -/
theorem from_outLen (tc : ThreadedComposed) (i : Nat) (hi : i < tc.steps.length) :
    ((fromThreadedComposed tc).arrays.filter
        (fun a => decide (a.equationIdx = i ∧ a.isInput = false))).length
      = (tc.steps[i]).outputWeaves.length := by
  rw [from_outputRows tc i hi]; simp

/-- Input row count = number of routing reads. -/
theorem from_inLen (tc : ThreadedComposed) (i : Nat) (hi : i < tc.steps.length) :
    ((fromThreadedComposed tc).arrays.filter
        (fun a => decide (a.equationIdx = i ∧ a.isInput = true))).length
      = (tc.routing.getD i []).length := by
  rw [from_inputRows tc i hi]; simp

/-- One equation row per step, so the equation table's length is the step count. -/
theorem equations_length (tc : ThreadedComposed) :
    (fromThreadedComposed tc).equations.length = tc.steps.length := by
  simp only [fromThreadedComposed, List.length_flatten, List.map_map]
  rw [show ((stepInsts tc).map ((·.length) ∘ (·.equations)))
        = List.replicate (stepInsts tc).length 1 from ?_]
  · simp
  · apply List.ext_getElem <;> simp [Function.comp, encodeStep, stepInsts]

/-- The per-step shape invariant the encoding assumes but `WellFormed` does not carry (the dropped
    dependent `StMat degree (inputWeaves i).targetAxes` typing, `Target.lean:63`). `fromThreadedComposed`
    only iterates `steps`/`reads`, and `decodeReindexing` normalizes each matrix to `codLen ×
    domLen`, so the round trip needs: routing and steps agree in length; per step, one reindexing per
    read; and each reindexing matrix has the shape its input weave / degree dictate. -/
def _root_.LeanNCD.ThreadedComposed.WellShaped (tc : ThreadedComposed) : Prop :=
  tc.routing.length = tc.steps.length ∧
  ∀ i, i < tc.steps.length →
    let b := tc.steps.getD i default
    let reads := tc.routing.getD i []
    b.reindexings.length = reads.length ∧
    ∀ j, j < reads.length →
      let m := b.reindexings.getD j default
      let inW := b.inputWeaves.getD j []
      m.codLen = (fixedPositions inW).length ∧
      m.domLen = b.degree.length ∧
      m.coeffs.length = m.codLen ∧
      (∀ row ∈ m.coeffs, row.length = m.domLen) ∧
      m.bias.length = m.codLen

/-! #### Machinery for the per-step inversion `decodeStep_eq` (steps 2–6). -/

/-- `find?` returns the unique element satisfying a decidable predicate. -/
theorem find?_unique {α} (l : List α) (P : α → Bool) (x : α)
    (hx : x ∈ l) (hpx : P x) (huniq : ∀ y ∈ l, P y → y = x) : l.find? P = some x := by
  induction l with
  | nil => simp at hx
  | cons a t ih =>
    by_cases hpa : P a
    · rw [List.find?_cons_of_pos hpa]
      exact congrArg some (huniq a (List.mem_cons_self ..) hpa)
    · rw [List.find?_cons_of_neg hpa]
      have hxt : x ∈ t := by
        rcases List.mem_cons.1 hx with h | h
        · exact absurd (h ▸ hpx) (by simpa using hpa)
        · exact h
      exact ih hxt (fun y hy hpy => huniq y (List.mem_cons_of_mem _ hy) hpy)

/-- `axisUidFor3` is injective in all three coordinates (`Nat.pair`, nested). -/
theorem axisUidFor3_inj {i s p i' s' p' : Nat} (he : axisUidFor3 i s p = axisUidFor3 i' s' p') :
    i = i' ∧ s = s' ∧ p = p' := by
  simp only [axisUidFor3, AxisUID.mk.injEq, true_and] at he
  rw [Nat.pair_eq_pair, Nat.pair_eq_pair] at he
  exact ⟨he.1, he.2.1, he.2.2⟩

/-- Every `encodeAxisSizes` entry is either a `.rawAxis` size entry or a `.normAxis` name entry, with
    its uid/value pinned to a `fixed` position of the weave. -/
theorem mem_encodeAxisSizes (i slot : Nat) (w : WeaveShapeP) (u : AxisUID) (sz : SizeExpr)
    (hmem : (u, sz) ∈ encodeAxisSizes i slot w) :
    (∃ p a, p < w.length ∧ w.getD p .tiled = .fixed a ∧ u = axisUidFor3 i slot p ∧ sz = a.size)
    ∨ (∃ p a nm, p < w.length ∧ w.getD p .tiled = .fixed a ∧ a.name = some nm
          ∧ u = nameUidFor3 i slot p ∧ sz = .var nm) := by
  simp only [encodeAxisSizes, List.mem_flatMap, List.mem_range] at hmem
  obtain ⟨p, hp, hpm⟩ := hmem
  rcases hw : w.getD p .tiled with a | _
  · simp only [hw] at hpm
    rcases hn : a.name with _ | nm
    · simp only [hn, List.mem_singleton, Prod.mk.injEq] at hpm
      exact Or.inl ⟨p, a, hp, hw, hpm.1, hpm.2⟩
    · simp only [hn, List.mem_cons, List.not_mem_nil,
        or_false, Prod.mk.injEq] at hpm
      rcases hpm with h | h
      · exact Or.inl ⟨p, a, hp, hw, h.1, h.2⟩
      · exact Or.inr ⟨p, a, nm, hp, hw, hn, h.1, h.2⟩
  · simp only [hw] at hpm; simp at hpm

/-- The weave `encodeStep` places at array slot `slot`: output slots `0..outLen-1`, input slots
    `outLen..outLen+inLen-1`, degree slot `outLen+inLen`, else empty. -/
def slotWeave (b : BrBaseP) (reads : List Wire) (slot : Nat) : WeaveShapeP :=
  let outLen := b.outputWeaves.length
  let inLen := reads.length
  if slot < outLen then b.outputWeaves.getD slot []
  else if slot < outLen + inLen then b.inputWeaves.getD (slot - outLen) []
  else if slot = outLen + inLen then b.degree.map .fixed else []

theorem encodeAxisRows_arraySlot (i slot : Nat) (w : WeaveShapeP) :
    ∀ r ∈ encodeAxisRows i slot w, r.arraySlot = slot := by
  intro r hr; simp only [encodeAxisRows, List.mem_map] at hr
  obtain ⟨p, _, rfl⟩ := hr; split <;> rfl

/-- Filtering a `flatMap` of `encodeAxisRows` groups (indexed at `base + j`) by `arraySlot = slot`
    isolates the single matching group. -/
theorem filterSlot_flatMap_off (i base n slot : Nat) (g : Nat → WeaveShapeP) :
    ((List.range n).flatMap (fun j => encodeAxisRows i (base + j) (g j))).filter
        (fun r => decide (r.arraySlot = slot))
      = if base ≤ slot ∧ slot - base < n then encodeAxisRows i slot (g (slot - base)) else [] := by
  have hmap : (List.range n).flatMap (fun j => encodeAxisRows i (base + j) (g j))
      = ((List.range n).map (fun j => encodeAxisRows i (base + j) (g j))).flatten := by
    rw [List.flatMap_def]
  rw [hmap, List.filter_flatten]
  have htag : ∀ k (hk : k < ((List.range n).map (fun j => encodeAxisRows i (base + j) (g j))).length),
      ∀ r ∈ ((List.range n).map (fun j => encodeAxisRows i (base + j) (g j)))[k],
        r.arraySlot = base + k := by
    intro k hk r hr
    rw [List.getElem_map, List.getElem_range] at hr
    exact encodeAxisRows_arraySlot i (base + k) (g k) r hr
  have := filter_flatten_tagged_aux (β := ArrayAxisRow) (·.arraySlot) slot (fun _ => true)
    ((List.range n).map (fun j => encodeAxisRows i (base + j) (g j))) base (by simpa using htag)
  simp only [Bool.and_true, List.filter_true] at this
  rw [this]
  by_cases hb : base ≤ slot
  · rw [if_pos hb]
    by_cases hlt : slot - base < n
    · rw [List.getD_eq_getElem _ _ (by simpa using hlt), List.getElem_map, List.getElem_range,
        if_pos ⟨hb, hlt⟩]
      congr 1; omega
    · rw [List.getD_eq_default _ _ (by simpa using hlt), if_neg (by tauto)]
  · rw [if_neg hb, if_neg (by tauto)]

/-- Base-0 specialisation of `filterSlot_flatMap_off` (groups indexed directly by `s`). -/
theorem filterSlot_flatMap0 (i n slot : Nat) (g : Nat → WeaveShapeP) :
    ((List.range n).flatMap (fun s => encodeAxisRows i s (g s))).filter
        (fun r => decide (r.arraySlot = slot))
      = if slot < n then encodeAxisRows i slot (g slot) else [] := by
  have := filterSlot_flatMap_off i 0 n slot g
  simpa using this

/-- One step's `arrayAxes`, filtered by `arraySlot = slot`, is exactly `slot`'s `encodeAxisRows`. -/
theorem encodeStep_arrayAxes_slot (i : Nat) (b : BrBaseP) (reads : List Wire) (slot : Nat) :
    (encodeStep i b reads).arrayAxes.filter (fun r => decide (r.arraySlot = slot))
      = encodeAxisRows i slot (slotWeave b reads slot) := by
  simp only [encodeStep, List.filter_append]
  rw [filterSlot_flatMap0 i _ slot (fun s => b.outputWeaves.getD s []),
      filterSlot_flatMap_off i b.outputWeaves.length _ slot (fun j => b.inputWeaves.getD j [])]
  have hdeg : (encodeAxisRows i (b.outputWeaves.length + reads.length) (b.degree.map .fixed)).filter
      (fun r => decide (r.arraySlot = slot))
      = if b.outputWeaves.length + reads.length = slot then
          encodeAxisRows i (b.outputWeaves.length + reads.length) (b.degree.map .fixed) else [] := by
    by_cases hd : b.outputWeaves.length + reads.length = slot
    · rw [if_pos hd]; apply List.filter_eq_self.2
      intro r hr; rw [encodeAxisRows_arraySlot _ _ _ r hr, hd]; simp
    · rw [if_neg hd]; apply List.filter_eq_nil_iff.2
      intro r hr hq; rw [encodeAxisRows_arraySlot _ _ _ r hr] at hq; simp at hq; omega
  rw [hdeg]
  unfold slotWeave
  by_cases h1 : slot < b.outputWeaves.length
  · rw [if_pos h1, if_neg (by omega), if_neg (by omega), List.append_nil, List.append_nil, if_pos h1]
  · by_cases h2 : slot < b.outputWeaves.length + reads.length
    · rw [if_neg h1, if_pos ⟨by omega, by omega⟩, if_neg (by omega), List.nil_append,
        List.append_nil, if_neg h1, if_pos h2]
    · by_cases h3 : b.outputWeaves.length + reads.length = slot
      · rw [if_neg h1, if_neg (by omega), if_pos h3, List.nil_append, List.nil_append,
          if_neg h1, if_neg (by omega), if_pos (by omega)]
        rw [h3]
      · rw [if_neg h1, if_neg (by omega), if_neg h3, List.append_nil, List.append_nil,
          if_neg h1, if_neg (by omega), if_neg (by omega)]
        simp [encodeAxisRows]

/-- Membership in the flattened `axisSizes` is membership in some step's `axisSizes`. -/
theorem mem_from_axisSizes (tc : ThreadedComposed) (x : AxisUID × SizeExpr) :
    x ∈ (fromThreadedComposed tc).axisSizes ↔
      ∃ i', ∃ (hi' : i' < tc.steps.length),
        x ∈ (encodeStep i' (tc.steps[i']) (tc.routing.getD i' [])).axisSizes := by
  simp only [fromThreadedComposed, List.mem_flatten, List.mem_map]
  constructor
  · rintro ⟨l, ⟨inst, hinst, rfl⟩, hx⟩
    rw [stepInsts] at hinst
    simp only [List.mem_map, List.mem_zipIdx_iff_getElem?] at hinst
    obtain ⟨⟨bb, kk⟩, hget, rfl⟩ := hinst
    rw [List.getElem?_eq_some_iff] at hget
    obtain ⟨hk, he⟩ := hget
    have hk' : kk < tc.steps.length := by simpa using hk
    refine ⟨kk, hk', ?_⟩
    rw [show bb = tc.steps[kk] from he.symm] at hx; exact hx
  · rintro ⟨i', hi', hx⟩
    refine ⟨_, ⟨(stepInsts tc)[i']'(by simp [stepInsts]; omega), List.getElem_mem _, rfl⟩, ?_⟩
    rw [stepInsts_getElem tc i' hi']; exact hx

/-- Every step-`axisSizes` entry comes from some slot's `encodeAxisSizes` (with the `slotWeave`). -/
theorem mem_step_axisSizes_form (i : Nat) (b : BrBaseP) (reads : List Wire)
    (x : AxisUID × SizeExpr) (hx : x ∈ (encodeStep i b reads).axisSizes) :
    ∃ slot', x ∈ encodeAxisSizes i slot' (slotWeave b reads slot') := by
  simp only [encodeStep] at hx
  rcases List.mem_append.1 hx with h | h
  · rcases List.mem_append.1 h with h | h
    · rw [List.mem_flatMap] at h; obtain ⟨s, hs, hxs⟩ := h
      rw [List.mem_range] at hs
      refine ⟨s, ?_⟩; unfold slotWeave; rw [if_pos hs]; exact hxs
    · rw [List.mem_flatMap] at h; obtain ⟨j, hj, hxj⟩ := h
      rw [List.mem_range] at hj
      refine ⟨b.outputWeaves.length + j, ?_⟩
      unfold slotWeave; rw [if_neg (by omega), if_pos (by omega)]
      simpa using hxj
  · refine ⟨b.outputWeaves.length + reads.length, ?_⟩
    unfold slotWeave; rw [if_neg (by omega), if_neg (by omega), if_pos rfl]; exact h

/-- A slot's `encodeAxisSizes` (with `slotWeave`) sits inside the step's `axisSizes`. -/
theorem encodeAxisSizes_slot_sub (i : Nat) (b : BrBaseP) (reads : List Wire) (slot : Nat)
    (x : AxisUID × SizeExpr) (hx : x ∈ encodeAxisSizes i slot (slotWeave b reads slot)) :
    x ∈ (encodeStep i b reads).axisSizes := by
  simp only [encodeStep]
  unfold slotWeave at hx
  by_cases h1 : slot < b.outputWeaves.length
  · rw [if_pos h1] at hx
    apply List.mem_append.2; left; apply List.mem_append.2; left
    rw [List.mem_flatMap]; exact ⟨slot, List.mem_range.2 h1, hx⟩
  · by_cases h2 : slot < b.outputWeaves.length + reads.length
    · rw [if_neg h1, if_pos h2] at hx
      apply List.mem_append.2; left; apply List.mem_append.2; right
      rw [List.mem_flatMap]
      refine ⟨slot - b.outputWeaves.length, List.mem_range.2 (by omega), ?_⟩
      rw [show b.outputWeaves.length + (slot - b.outputWeaves.length) = slot from by omega]
      exact hx
    · by_cases h3 : slot = b.outputWeaves.length + reads.length
      · rw [if_neg h1, if_neg h2, if_pos h3] at hx
        apply List.mem_append.2; right
        rw [h3] at hx; exact hx
      · rw [if_neg h1, if_neg h2, if_neg h3] at hx
        simp [encodeAxisSizes] at hx

theorem nameUidFor3_inj {i s p i' s' p' : Nat} (he : nameUidFor3 i s p = nameUidFor3 i' s' p') :
    i = i' ∧ s = s' ∧ p = p' := by
  simp only [nameUidFor3, AxisUID.mk.injEq, true_and] at he
  rw [Nat.pair_eq_pair, Nat.pair_eq_pair] at he
  exact ⟨he.1, he.2.1, he.2.2⟩

theorem mem_encodeAxisSizes_size (i slot p : Nat) (w : WeaveShapeP) (a : AxisP)
    (hp : p < w.length) (hw : w.getD p .tiled = .fixed a) :
    (axisUidFor3 i slot p, a.size) ∈ encodeAxisSizes i slot w := by
  simp only [encodeAxisSizes, List.mem_flatMap, List.mem_range]
  refine ⟨p, hp, ?_⟩
  simp only [hw]
  split <;> simp

theorem mem_encodeAxisSizes_name (i slot p : Nat) (w : WeaveShapeP) (a : AxisP) (nm : String)
    (hp : p < w.length) (hw : w.getD p .tiled = .fixed a) (hnm : a.name = some nm) :
    (nameUidFor3 i slot p, SizeExpr.var nm) ∈ encodeAxisSizes i slot w := by
  simp only [encodeAxisSizes, List.mem_flatMap, List.mem_range]
  refine ⟨p, hp, ?_⟩
  simp only [hw, hnm]; simp

/-- `lookupSize` recovers the stored `SizeExpr` of a fixed axis (step 2, size half). Uses global
    uid uniqueness (`axisUidFor3` injective) + membership. -/
theorem lookupSize_from (tc : ThreadedComposed) (i : Nat) (hi : i < tc.steps.length)
    (slot p : Nat) (a : AxisP)
    (hp : p < (slotWeave (tc.steps[i]) (tc.routing.getD i []) slot).length)
    (hw : (slotWeave (tc.steps[i]) (tc.routing.getD i []) slot).getD p .tiled = .fixed a) :
    lookupSize (fromThreadedComposed tc) (axisUidFor3 i slot p) = a.size := by
  have hmem : (axisUidFor3 i slot p, a.size) ∈ (fromThreadedComposed tc).axisSizes := by
    rw [mem_from_axisSizes]
    exact ⟨i, hi, encodeAxisSizes_slot_sub i _ _ slot _ (mem_encodeAxisSizes_size i slot p _ a hp hw)⟩
  have huniq : ∀ y ∈ (fromThreadedComposed tc).axisSizes,
      decide (y.1 = axisUidFor3 i slot p) → y = (axisUidFor3 i slot p, a.size) := by
    rintro ⟨u', sz'⟩ hy hyd
    simp only [decide_eq_true_eq] at hyd
    rw [mem_from_axisSizes] at hy
    obtain ⟨i'', hi'', hy⟩ := hy
    obtain ⟨slot'', hy⟩ := mem_step_axisSizes_form _ _ _ _ hy
    rcases mem_encodeAxisSizes _ _ _ _ _ hy with
      ⟨p'', a'', hp'', hw'', hu, hsz⟩ | ⟨p'', a'', nm, hp'', hw'', hnm, hu, hsz⟩
    · rw [hu] at hyd
      obtain ⟨rfl, rfl, rfl⟩ := axisUidFor3_inj hyd
      rw [hw] at hw''
      injection hw'' with ha
      simp only [Prod.mk.injEq]
      exact ⟨hu, by rw [hsz, ← ha]⟩
    · rw [hu] at hyd
      simp [nameUidFor3, axisUidFor3] at hyd
  have := find?_unique (fromThreadedComposed tc).axisSizes
    (fun y => decide (y.1 = axisUidFor3 i slot p)) (axisUidFor3 i slot p, a.size)
    hmem (by simp) huniq
  simp only [lookupSize, this, Option.map_some, Option.getD_some]

/-- `lookupName` recovers a fixed axis's name (step 2, name half) — `some nm` from the dedicated
    `.normAxis` entry, or `none` when unnamed (no such entry exists). -/
theorem lookupName_from (tc : ThreadedComposed) (i : Nat) (hi : i < tc.steps.length)
    (slot p : Nat) (a : AxisP)
    (hp : p < (slotWeave (tc.steps[i]) (tc.routing.getD i []) slot).length)
    (hw : (slotWeave (tc.steps[i]) (tc.routing.getD i []) slot).getD p .tiled = .fixed a) :
    lookupName (fromThreadedComposed tc) (axisUidFor3 i slot p) = a.name := by
  have hid : (⟨.normAxis, (axisUidFor3 i slot p).id⟩ : AxisUID) = nameUidFor3 i slot p := rfl
  simp only [lookupName, hid]
  rcases hn : a.name with _ | nm
  · have hnone : (fromThreadedComposed tc).axisSizes.find?
        (fun q => decide (q.1 = nameUidFor3 i slot p)) = none := by
      rw [List.find?_eq_none]
      rintro ⟨u', sz'⟩ hy hyd
      simp only [decide_eq_true_eq] at hyd
      rw [mem_from_axisSizes] at hy
      obtain ⟨i'', hi'', hy⟩ := hy
      obtain ⟨slot'', hy⟩ := mem_step_axisSizes_form _ _ _ _ hy
      rcases mem_encodeAxisSizes _ _ _ _ _ hy with
        ⟨p'', a'', hp'', hw'', hu, hsz⟩ | ⟨p'', a'', nm', hp'', hw'', hnm', hu, hsz⟩
      · rw [hu] at hyd; simp [nameUidFor3, axisUidFor3] at hyd
      · rw [hu] at hyd
        obtain ⟨rfl, rfl, rfl⟩ := nameUidFor3_inj hyd
        rw [hw] at hw''; injection hw'' with ha
        rw [← ha] at hnm'; rw [hn] at hnm'; exact absurd hnm' (by simp)
    rw [hnone]; rfl
  · have hmem : (nameUidFor3 i slot p, SizeExpr.var nm) ∈ (fromThreadedComposed tc).axisSizes := by
      rw [mem_from_axisSizes]
      exact ⟨i, hi, encodeAxisSizes_slot_sub i _ _ slot _
        (mem_encodeAxisSizes_name i slot p _ a nm hp hw hn)⟩
    have huniq : ∀ y ∈ (fromThreadedComposed tc).axisSizes,
        decide (y.1 = nameUidFor3 i slot p) → y = (nameUidFor3 i slot p, SizeExpr.var nm) := by
      rintro ⟨u', sz'⟩ hy hyd
      simp only [decide_eq_true_eq] at hyd
      rw [mem_from_axisSizes] at hy
      obtain ⟨i'', hi'', hy⟩ := hy
      obtain ⟨slot'', hy⟩ := mem_step_axisSizes_form _ _ _ _ hy
      rcases mem_encodeAxisSizes _ _ _ _ _ hy with
        ⟨p'', a'', hp'', hw'', hu, hsz⟩ | ⟨p'', a'', nm', hp'', hw'', hnm', hu, hsz⟩
      · rw [hu] at hyd; simp [nameUidFor3, axisUidFor3] at hyd
      · rw [hu] at hyd
        obtain ⟨rfl, rfl, rfl⟩ := nameUidFor3_inj hyd
        rw [hw] at hw''; injection hw'' with ha
        rw [← ha] at hnm'; rw [hn] at hnm'
        simp only [Option.some.injEq] at hnm'
        simp only [Prod.mk.injEq]
        exact ⟨hu, by rw [hsz, hnm']⟩
    have := find?_unique (fromThreadedComposed tc).axisSizes
      (fun y => decide (y.1 = nameUidFor3 i slot p)) (nameUidFor3 i slot p, SizeExpr.var nm)
      hmem (by simp) huniq
    rw [this]; rfl

theorem encodeAxisRows_length (i slot : Nat) (w : WeaveShapeP) :
    (encodeAxisRows i slot w).length = w.length := by simp [encodeAxisRows]

/-- `find?` on position `p` returns the `p`-th row (positions are the indices, all distinct). -/
theorem encodeAxisRows_find (i slot : Nat) (w : WeaveShapeP) (p : Nat) (hp : p < w.length) :
    (encodeAxisRows i slot w).find? (fun r => decide (r.position = p))
      = some ((encodeAxisRows i slot w)[p]'(by rw [encodeAxisRows_length]; exact hp)) := by
  apply find?_unique
  · exact List.getElem_mem _
  · have hb : p < (encodeAxisRows i slot w).length := by rw [encodeAxisRows_length]; exact hp
    have hpos : ((encodeAxisRows i slot w)[p]'hb).position = p := by
      simp only [encodeAxisRows, List.getElem_map, List.getElem_range]; split <;> rfl
    simp [hpos]
  · intro y hy hyd
    simp only [decide_eq_true_eq] at hyd
    simp only [encodeAxisRows, List.mem_map, List.mem_range] at hy
    obtain ⟨q, hq, rfl⟩ := hy
    have hqpos : (match w.getD q .tiled with
        | .fixed _ => (⟨i, slot, axisUidFor3 i slot q, false, q⟩ : ArrayAxisRow)
        | .tiled => ⟨i, slot, ⟨.natAxis, 0⟩, false, q⟩).position = q := by split <;> rfl
    rw [hqpos] at hyd; subst hyd
    simp only [encodeAxisRows, List.getElem_map, List.getElem_range]

/-- Weave round trip (step 3): `decodeWeaveAt` recovers exactly the weave `encodeStep` put at `slot`
    — the keystone, inverting `encodeAxisRows` position-by-position via the `lookup*` lemmas. -/
theorem decodeWeaveAt_from (tc : ThreadedComposed) (i : Nat) (hi : i < tc.steps.length) (slot : Nat) :
    decodeWeaveAt (fromThreadedComposed tc) i slot
      = slotWeave (tc.steps[i]) (tc.routing.getD i []) slot := by
  set w := slotWeave (tc.steps[i]) (tc.routing.getD i []) slot with hwdef
  have hrows : (fromThreadedComposed tc).arrayAxes.filter
      (fun r => decide (r.equationIdx = i ∧ r.arraySlot = slot)) = encodeAxisRows i slot w := by
    rw [show (fun r : ArrayAxisRow => decide (r.equationIdx = i ∧ r.arraySlot = slot))
          = (fun r => decide (r.equationIdx = i) && decide (r.arraySlot = slot)) from
        by funext r; rw [Bool.decide_and]]
    rw [from_field_filter tc i hi SBrInstance.arrayAxes (·.equationIdx)
          (fun r => decide (r.arraySlot = slot)) rfl encodeStep_arrayAxes_eqIdx,
        encodeStep_arrayAxes_slot]
  simp only [decodeWeaveAt, hrows, encodeAxisRows_length]
  apply List.ext_getElem
  · simp [encodeAxisRows_length]
  · intro k hk1 _
    have hkw : k < w.length := by simpa using hk1
    simp only [List.getElem_map, List.getElem_range]
    rw [encodeAxisRows_find i slot w k hkw]
    have hget : (encodeAxisRows i slot w)[k]'(by rw [encodeAxisRows_length]; exact hkw)
        = (match w.getD k .tiled with
            | .fixed _ => (⟨i, slot, axisUidFor3 i slot k, false, k⟩ : ArrayAxisRow)
            | .tiled => ⟨i, slot, ⟨.natAxis, 0⟩, false, k⟩) := by
      simp only [encodeAxisRows, List.getElem_map, List.getElem_range]
    rw [hget, ← List.getD_eq_getElem w .tiled hkw]
    rcases hwk : w.getD k .tiled with a | _
    · simp only [hwk]
      rw [lookupName_from tc i hi slot k a (by rw [← hwdef]; exact hkw) (by rw [← hwdef]; exact hwk),
          lookupSize_from tc i hi slot k a (by rw [← hwdef]; exact hkw) (by rw [← hwdef]; exact hwk)]
      simp only [axisUidFor3]
    · simp only [hwk]

/-- The per-step inversion (steps 1–6): decoding step `i` recovers exactly the encoded `BrBaseP`
    and its routing reads. Needs `WellFormed` (for `inputWeaves.length = reads.length`, conjunct 2)
    and `WellShaped` (for the reindexing dimensions). -/
theorem decodeStep_eq (tc : ThreadedComposed) (h : tc.WellFormed) (hs : tc.WellShaped)
    (i : Nat) (hi : i < tc.steps.length) :
    decodeStep (fromThreadedComposed tc) i = (tc.steps[i], tc.routing.getD i []) := by
  simp only [decodeStep]
  sorry

/-- Only external wires `< nExternal` live in any pool: the fold prepends internal output slots
    onto the external base `(range nExternal).map .external`. -/
theorem mem_poolAt_external (tc : ThreadedComposed) (i k : Nat) :
    Wire.external k ∈ tc.poolAt i → k < tc.nExternal := by
  induction i with
  | zero =>
    intro hmem
    simp only [ThreadedComposed.poolAt, List.range_zero, List.foldl_nil, List.mem_map] at hmem
    obtain ⟨x, hx, hxe⟩ := hmem; rw [List.mem_range] at hx; cases hxe; exact hx
  | succ n ih =>
    intro hmem
    rw [tc.poolAt_succ] at hmem
    rcases List.mem_append.1 hmem with h | h
    · simp only [ThreadedComposed.outputSlots, List.mem_map] at h
      obtain ⟨x, _, hxe⟩ := h; cases hxe
    · exact ih h

/-- If `externalPort k` finds a port, external slot `k` really is referenced in the routing. -/
theorem externalPort_mem (tc : ThreadedComposed) (k : Nat) (i j : Nat)
    (hp : tc.externalPort k = some (i, j)) :
    ∃ wires ∈ tc.routing, Wire.external k ∈ wires := by
  simp only [ThreadedComposed.externalPort] at hp
  obtain ⟨i', hi', hinner⟩ := List.exists_of_findSome?_eq_some hp
  obtain ⟨j', hj', hmatch⟩ := List.exists_of_findSome?_eq_some hinner
  rw [List.mem_range] at hi' hj'
  set wires := tc.routing.getD i' [] with hw
  rcases hwj : wires.getD j' (Wire.external 0) with k' | ⟨s1, s2⟩
  · rw [hwj] at hmatch
    by_cases hkk : k' == k
    · have : k' = k := by simpa using hkk
      subst this
      refine ⟨wires, ?_, ?_⟩
      · rw [hw, List.getD_eq_getElem _ _ hi']; exact List.getElem_mem hi'
      · rw [← hwj, List.getD_eq_getElem _ _ hj']; exact List.getElem_mem hj'
    · simp [hkk] at hmatch
  · rw [hwj] at hmatch; simp at hmatch

/-- The reconstructed external count matches (step 7 — the only part needing `WellFormed`). -/
theorem from_nExternal (tc : ThreadedComposed) (h : tc.WellFormed) (hsh : tc.WellShaped) :
    (toThreadedComposed (fromThreadedComposed tc)).nExternal = tc.nExternal := by
  set s := fromThreadedComposed tc with hsdef
  have hn : s.equations.length = tc.steps.length := equations_length tc
  set extKs := ((List.range s.equations.length).map (decodeStep s)).flatMap Prod.snd |>.filterMap
      (fun w => match w with | .external k => some k | .internal _ _ => none) with hextKs
  have hmemK : ∀ k, k ∈ extKs ↔
      ∃ i, i < tc.steps.length ∧ Wire.external k ∈ tc.routing.getD i [] := by
    intro k
    rw [hextKs, List.mem_filterMap]
    constructor
    · rintro ⟨w, hw, hwk⟩
      rw [List.mem_flatMap] at hw
      obtain ⟨pr, hpr, hwpr⟩ := hw
      rw [List.mem_map] at hpr
      obtain ⟨i, hi, rfl⟩ := hpr
      rw [List.mem_range, hn] at hi
      rw [decodeStep_eq tc h hsh i hi] at hwpr
      cases w with
      | external k' => simp at hwk; subst hwk; exact ⟨i, hi, hwpr⟩
      | internal => simp at hwk
    · rintro ⟨i, hi, hmem⟩
      refine ⟨Wire.external k, ?_, rfl⟩
      rw [List.mem_flatMap]
      refine ⟨decodeStep s i, ?_, ?_⟩
      · rw [List.mem_map]; exact ⟨i, by rw [List.mem_range, hn]; exact hi, rfl⟩
      · rw [decodeStep_eq tc h hsh i hi]; exact hmem
  have hall : ∀ k ∈ extKs, k < tc.nExternal := by
    intro k hk
    rw [hmemK] at hk
    obtain ⟨i, hi, hmem⟩ := hk
    exact mem_poolAt_external tc i k (h.2.2.2 i hi _ hmem)
  have href : ∀ k, k < tc.nExternal → k ∈ extKs := by
    intro k hkN
    rw [hmemK]
    have hpred := (List.all_eq_true.mp h.1) k (List.mem_range.mpr hkN)
    rcases hep : tc.externalPort k with _ | ⟨i, j⟩
    · rw [hep] at hpred; simp at hpred
    · obtain ⟨wires, hwires, hmem⟩ := externalPort_mem tc k i j hep
      obtain ⟨idx, hidx, rfl⟩ := List.mem_iff_getElem.mp hwires
      refine ⟨idx, by rw [← hsh.1]; exact hidx, ?_⟩
      rw [List.getD_eq_getElem _ _ hidx]; exact hmem
  show (match extKs.max? with | some m => m + 1 | none => 0) = tc.nExternal
  rcases hN : tc.nExternal with _ | m
  · have : extKs = [] := by
      rcases hL : extKs with _ | ⟨x, xs⟩
      · rfl
      · exact absurd (hall x (by rw [hL]; exact List.mem_cons_self ..)) (by rw [hN]; omega)
    rw [this]; rfl
  · have hmax : extKs.max? = some m := by
      rw [List.max?_eq_some_iff]
      refine ⟨href m (by rw [hN]; omega), fun b hb => ?_⟩
      have := hall b hb; rw [hN] at this; omega
    rw [hmax]

theorem toThreadedComposed_fromThreadedComposed (tc : ThreadedComposed) (h : tc.WellFormed)
    (hs : tc.WellShaped) :
    toThreadedComposed (fromThreadedComposed tc) = tc := by
  have hlen := hs.1
  set s := fromThreadedComposed tc with hsdef
  have hn : s.equations.length = tc.steps.length := equations_length tc
  have hsteps : (toThreadedComposed s).steps = tc.steps := by
    simp only [toThreadedComposed, List.map_map]
    rw [hn]; apply List.ext_getElem
    · simp
    · intro k h1 _; simp only [List.getElem_map, List.getElem_range, Function.comp]
      rw [decodeStep_eq tc h hs k (by simpa using h1)]
  have hrouting : (toThreadedComposed s).routing = tc.routing := by
    simp only [toThreadedComposed, List.map_map]
    rw [hn]; apply List.ext_getElem
    · simp [hlen]
    · intro k h1 _; simp only [List.getElem_map, List.getElem_range, Function.comp]
      rw [decodeStep_eq tc h hs k (by simpa using h1),
        List.getD_eq_getElem _ _ (by simpa [hlen] using h1)]
  calc toThreadedComposed s
      = ⟨(toThreadedComposed s).steps, (toThreadedComposed s).routing,
          (toThreadedComposed s).nExternal⟩ := rfl
    _ = tc := by rw [hsteps, hrouting, from_nExternal tc h hs]

end LeanNCD.AcsetCodec

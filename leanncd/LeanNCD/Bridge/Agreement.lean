-- LeanNCD/Bridge/Agreement.lean
import LeanNCD.Bridge.Realize
import LeanNCD.Bridge.SBr
import LeanNCD.DSL.Compile
import LeanNCD.DSL.Pipeline.RouteSpec

namespace LeanNCD

open Std

/-! ## Phase 4 — `compile_wellFormed`: every compiled program is `WellFormed`

`compile = compileToScheduled >>= route`; `compile_eq_route` isolates `tc` as `route`'s output of a
scheduled program `sp`, after which each `WellFormed` conjunct is a fact about `routeCore sp`. -/

/-- Plumbing: a successful `compile` factors as a successful `compileToScheduled` (giving `sp`) whose
    `routeCore` produced `tc`'s steps/routing, with `tc.nExternal = sp.extNames.card`. -/
theorem compile_eq_route {p : TLProgram} {s : Nat} {tc : ThreadedComposed} {s' : Nat}
    (h : (TLProgram.compile p).run s = .ok tc s') :
    ∃ (sp : ScheduledProgram) (s₁ : Nat), (TLProgram.compileToScheduled p).run s = .ok sp s₁ ∧
      routeCore sp = .ok (tc.steps, tc.routing) ∧ tc.nExternal = sp.extNames.card := by
  have hcr : TLProgram.compile p = TLProgram.compileToScheduled p >>= route := by
    simp only [TLProgram.compile, TLProgram.compileToScheduled, Bind.kleisliRight, bind_assoc]
  rw [hcr, EStateM.run_bind] at h
  cases hcs : (TLProgram.compileToScheduled p).run s with
  | error e s₁ => rw [hcs] at h; simp at h
  | ok sp s₁ =>
      rw [hcs] at h
      dsimp only at h
      unfold route at h
      rcases hrcc : routeCore sp with _ | ⟨steps, routing⟩
      · rw [hrcc] at h; simp [EStateM.run, throw, throwThe, MonadExceptOf.throw, EStateM.throw] at h
      · rw [hrcc] at h
        simp only [EStateM.run, pure, EStateM.pure, EStateM.Result.ok.injEq] at h
        obtain ⟨htc, -⟩ := h
        subst htc
        exact ⟨sp, s₁, rfl, hrcc, rfl⟩

-- NOTE(phaseB): restated to `≥ 1` to match the weakened `WellFormed` conjunct 3 (multi-output scans).
/-- Conjunct 3 (at least one output): every routed step has at least one output weave. -/
theorem wf_singleOutput {sp : ScheduledProgram} {steps : List BrBaseP} {routing : List (List Wire)}
    (hrc : routeCore sp = .ok (steps, routing)) :
    ∀ i, i < steps.length → (steps.getD i default).outputWeaves.length ≥ 1 := by
  intro i hi
  have hik : i < sp.stmts.length := (routeCore_steps_length hrc) ▸ hi
  have hbuild := routeCore_getD hrc i hik
  rw [buildStep_outputWeaves_length_one hbuild]
  have hne := buildStep_ok_outputs_ne hbuild
  have hnil : (sp.stmts.getD i default).outputs ≠ [] := fun hc => by rw [hc] at hne; simp at hne
  exact List.length_pos_of_ne_nil hnil

/-- The realized weave's target (retained) axes are the realized presentation fixed axes. -/
theorem realizeWeaveShape_targetAxes (w : WeaveShapeP) :
    (realizeWeaveShape w).targetAxes = (fixedAxesP w).map realizeAxis := by
  induction w with
  | nil => rfl
  | cons s t ih =>
      cases s <;>
        simp_all [realizeWeaveShape, WeaveShape.targetAxes, fixedAxesP, realizeWeaveSlot]

/-- Bridge: `weaveToArrayType` depends only on a weave's fixed axes (`fixedAxesP`). -/
theorem weaveToArrayType_congr {w₁ w₂ : WeaveShapeP} (h : fixedAxesP w₁ = fixedAxesP w₂) :
    weaveToArrayType w₁ = weaveToArrayType w₂ := by
  unfold weaveToArrayType
  rw [realizeWeaveShape_targetAxes, realizeWeaveShape_targetAxes, h]

/-- `fixedAxesP` of an all-`fixed` weave returns its axes verbatim. -/
private theorem fixedAxesP_map_fixed (axes : List AxisP) :
    fixedAxesP (axes.map (fun a => WeaveSlotP.fixed a)) = axes := by
  induction axes with
  | nil => rfl
  | cons a t ih => simp [fixedAxesP, List.map_cons, ih]

/-- The rank (count of `fixed` slots) of an external read's all-`fixed` weave is its position count. -/
private theorem weaveRank_range_fixed (nm : String) (n : Nat) :
    weaveRank ((List.range n).map (fun pos =>
      WeaveSlotP.fixed (AxisP.mk (some (nm ++ "_" ++ toString pos))
        (SizeExpr.var (nm ++ "_" ++ toString pos))))) = n := by
  rw [weaveRank, List.countP_map, List.countP_eq_length.mpr (by intro x _; rfl), List.length_range]

/-- From `wellFormedDom`: each external slot `k < nExternal` has a first consuming port `(i₀, j₀)`,
    and every port consuming `k` has the same input-weave rank as `(i₀, j₀)`. -/
private theorem wellFormedDom_rank {tc : ThreadedComposed} (hwfd : tc.wellFormedDom = true)
    {k : Nat} (hk : k < tc.nExternal) :
    ∃ i₀ j₀, tc.externalPort k = some (i₀, j₀) ∧
      ∀ i j, i < tc.routing.length → j < (tc.routing.getD i []).length →
        (tc.routing.getD i []).getD j (.external 0) = .external k →
        weaveRank ((tc.steps.getD i default).inputWeaves.getD j [])
          = weaveRank ((tc.steps.getD i₀ default).inputWeaves.getD j₀ []) := by
  unfold ThreadedComposed.wellFormedDom at hwfd
  rw [List.all_eq_true] at hwfd
  have hk' := hwfd k (List.mem_range.mpr hk)
  revert hk'
  cases hep : tc.externalPort k with
  | none => intro hk'; simp [hep] at hk'
  | some pair =>
      obtain ⟨i₀, j₀⟩ := pair
      intro hk'; simp only [hep] at hk'; rw [List.all_eq_true] at hk'
      refine ⟨i₀, j₀, rfl, ?_⟩
      intro i j hi hj hext
      have hi' := hk' i (List.mem_range.mpr hi); rw [List.all_eq_true] at hi'
      have hj' := hi' j (List.mem_range.mpr hj); rw [hext] at hj'
      simp only [beq_self_eq_true, Bool.not_true, Bool.false_or, beq_iff_eq] at hj'
      exact hj'

/-- `externalPort k = some (i₀, j₀)` reflects an actual in-bounds `Wire.external k` port. -/
private theorem externalPort_decode {tc : ThreadedComposed} {k i₀ j₀ : Nat}
    (hep : tc.externalPort k = some (i₀, j₀)) :
    i₀ < tc.routing.length ∧ j₀ < (tc.routing.getD i₀ []).length ∧
    (tc.routing.getD i₀ []).getD j₀ (.external 0) = .external k := by
  unfold ThreadedComposed.externalPort at hep
  obtain ⟨i, hi_mem, hi_eq⟩ := List.exists_of_findSome?_eq_some hep
  obtain ⟨j, hj_mem, hj_eq⟩ := List.exists_of_findSome?_eq_some hi_eq
  simp only [List.mem_range] at hi_mem hj_mem
  revert hj_eq
  cases hw : (tc.routing.getD i []).getD j (Wire.external 0) with
  | internal s sl => intro hj_eq; simp at hj_eq
  | external k' =>
      simp only []
      by_cases hk : k' == k
      · intro hj_eq; simp only [hk, if_true, Option.some.injEq, Prod.mk.injEq] at hj_eq
        obtain ⟨hi0, hj0⟩ := hj_eq; subst hi0; subst hj0
        rw [beq_iff_eq] at hk; subst hk; exact ⟨hi_mem, hj_mem, hw⟩
      · intro hj_eq; rw [if_neg (by simpa using hk)] at hj_eq; exact absurd hj_eq (by simp)

/-- At an in-bounds `Wire.external k` port of a routed `tc`, the input weave is exactly the
    read factor's external (all-`fixed`, bound-named) weave, and its name resolves to slot `k`. -/
private theorem port_external_weave {sp : ScheduledProgram} {tc : ThreadedComposed}
    (hrc : routeCore sp = .ok (tc.steps, tc.routing))
    {i j k : Nat} (hi : i < tc.steps.length)
    (hj : j < (tc.routing.getD i []).length)
    (hw : (tc.routing.getD i []).getD j (Wire.external 0) = .external k) :
    ∃ rf : String × List IdxExpr,
      (buildExtIndex sp.extNames sp.stmts)[rf.1]? = some k ∧
      (buildNameToStep sp.stmts)[rf.1]? = none ∧
      (tc.steps.getD i default).inputWeaves.getD j [] =
        (List.range rf.2.length).map (fun pos =>
          WeaveSlotP.fixed (AxisP.mk (some (rf.1 ++ "_" ++ toString pos))
            (SizeExpr.var (rf.1 ++ "_" ++ toString pos)))) := by
  have hik : i < sp.stmts.length := (routeCore_steps_length hrc) ▸ hi
  have hbuild := routeCore_getD hrc i hik
  have hwires := buildStep_wires_mapM hbuild
  have hiw := buildStep_inputWeaves hbuild
  set sc := sp.stmts.getD i default with hsc
  have hjlen : j < sc.inputReadFactors.length := (mapM_ok_length' hwires) ▸ hj
  set rf := sc.inputReadFactors.getD j default with hrfdef
  have hrfget : sc.inputReadFactors[j]'hjlen = rf := (List.getD_eq_getElem _ _ hjlen).symm
  -- the wire builder at position j evaluates to `.ok (external k)`
  have hwb := mapM_ok_getD' hwires j default (Wire.external 0) hjlen
  rw [hw] at hwb
  -- extract: name is unrouted (none) and external-indexed to `k`
  have hnames : (buildNameToStep sp.stmts)[rf.1]? = none ∧
      (buildExtIndex sp.extNames sp.stmts)[rf.1]? = some k := by
    revert hwb
    cases hns' : (buildNameToStep sp.stmts)[rf.1]? with
    | some p => intro hwb; simp at hwb
    | none =>
        cases hext' : (buildExtIndex sp.extNames sp.stmts)[rf.1]? with
        | some k' => intro hwb; simp only [Except.ok.injEq, Wire.external.injEq] at hwb; subst hwb; exact ⟨rfl, rfl⟩
        | none => intro hwb; simp at hwb
  obtain ⟨hns_none, hext_k⟩ := hnames
  refine ⟨rf, hext_k, hns_none, ?_⟩
  rw [hiw, List.getD_eq_getElem _ _ (by rw [List.length_map]; exact hjlen), List.getElem_map, hrfget,
    hns_none]

/-- External pointwise type match: an external read's wire carries exactly its input weave's type.
    Uses `buildExtIndex_lt_card` (slot `< nExternal`), `wellFormedDom` (rank agreement across ports),
    and `buildExtIndex_injective` (the first port for slot `k` reads the SAME name) to equate the two
    external weaves (same bound name, same rank ⇒ identical weave). -/
private theorem external_pointwise {sp : ScheduledProgram} {tc : ThreadedComposed}
    (hrc : routeCore sp = .ok (tc.steps, tc.routing))
    (hwfd : tc.wellFormedDom = true) (hne : tc.nExternal = sp.extNames.card)
    {i pos k : Nat} (rf : String × List IdxExpr)
    (hi : i < tc.steps.length) (hpos : pos < (tc.routing.getD i []).length)
    (hwport : (tc.routing.getD i []).getD pos (Wire.external 0) = .external k)
    (hextrf : (buildExtIndex sp.extNames sp.stmts)[rf.1]? = some k)
    (hiwrf : (tc.steps.getD i default).inputWeaves.getD pos [] =
        (List.range rf.2.length).map (fun p =>
          WeaveSlotP.fixed (AxisP.mk (some (rf.1 ++ "_" ++ toString p))
            (SizeExpr.var (rf.1 ++ "_" ++ toString p))))) :
    tc.wireType (Wire.external k) =
      weaveToArrayType ((List.range rf.2.length).map (fun p =>
        WeaveSlotP.fixed (AxisP.mk (some (rf.1 ++ "_" ++ toString p))
          (SizeExpr.var (rf.1 ++ "_" ++ toString p))))) := by
  have hk : k < tc.nExternal := hne ▸ buildExtIndex_lt_card hextrf
  obtain ⟨i₀, j₀, hep, hrank⟩ := wellFormedDom_rank hwfd hk
  obtain ⟨hi0, hj0, hw0⟩ := externalPort_decode hep
  have hsteps_len : tc.routing.length = tc.steps.length := by
    rw [routeCore_routing_length hrc, routeCore_steps_length hrc]
  have hi0' : i₀ < tc.steps.length := hsteps_len ▸ hi0
  obtain ⟨rf₀, hext0, hns0, hiw0⟩ := port_external_weave hrc hi0' hj0 hw0
  have hnames : rf₀.1 = rf.1 := buildExtIndex_injective hext0 hextrf
  have hi_routing : i < tc.routing.length := hsteps_len ▸ hi
  have hragree := hrank i pos hi_routing hpos hwport
  rw [hiwrf, weaveRank_range_fixed] at hragree
  rw [hiw0, hnames, weaveRank_range_fixed] at hragree
  simp only [ThreadedComposed.wireType, hep]
  apply weaveToArrayType_congr
  rw [hiw0, hnames, ← hragree]

/-- Internal pointwise type match: a producer wire `internal j 0` carries the consumer's published
    weave type, because `buildStep_output_fixedAxes` makes both weaves share the same fixed axes. -/
-- NOTE(phaseB): restated to the new `(j, slot)` / `slotStmt` shape; body to be proved in Phase B.
private theorem internal_pointwise {sp : ScheduledProgram} {tc : ThreadedComposed}
    (hrc : routeCore sp = .ok (tc.steps, tc.routing))
    {rf : String × List IdxExpr} {j slot : Nat}
    (hns : (buildNameToStep sp.stmts)[rf.1]? = some (j, slot)) :
    tc.wireType (Wire.internal j slot)
      = weaveToArrayType ((tensorAxes ((sp.stmts.getD j default).slotStmt slot)).map
          (fun a => WeaveSlotP.fixed a)) := by
  have hj : j < sp.stmts.length := buildNameToStep_lt hns
  have hslot : slot < (sp.stmts.getD j default).outputs.length := buildNameToStep_slot_lt hns
  have hbuild := routeCore_getD hrc j hj
  simp only [ThreadedComposed.wireType]
  apply weaveToArrayType_congr
  rw [fixedAxesP_map_fixed]
  exact buildStep_output_fixedAxes hbuild hslot

/-- Conjunct 2 (producer ⊳ consumer type match). The de-risked one: producer output and consumer
    input weaves share fixed axes by construction (`buildStep_output_fixedAxes`); external reads match
    via `wellFormedDom`'s rank agreement plus `buildExtIndex` injectivity (`external_pointwise`). -/
theorem wf_typeMatch {sp : ScheduledProgram} {tc : ThreadedComposed}
    (hrc : routeCore sp = .ok (tc.steps, tc.routing))
    (hwfd : tc.wellFormedDom = true) (hne : tc.nExternal = sp.extNames.card) :
    ∀ i, i < tc.steps.length →
      (tc.routing.getD i []).map tc.wireType
        = (tc.steps.getD i default).inputWeaves.map weaveToArrayType := by
  intro i hi
  have hik : i < sp.stmts.length := (routeCore_steps_length hrc) ▸ hi
  have hbuild := routeCore_getD hrc i hik
  have hwires := buildStep_wires_mapM hbuild
  have hiw := buildStep_inputWeaves hbuild
  rw [hiw, List.map_map]
  apply List.ext_getElem
  · rw [List.length_map, List.length_map, mapM_ok_length' hwires]
  · intro p h1 h2
    have hplen : p < (sp.stmts.getD i default).inputReadFactors.length := by
      simpa [List.length_map] using h2
    have h1' : p < (tc.routing.getD i []).length := by simpa using h1
    rw [List.getElem_map, List.getElem_map]
    set rf := (sp.stmts.getD i default).inputReadFactors[p]'hplen with hrfdef
    have hwb := mapM_ok_getD' hwires p default (Wire.external 0) hplen
    rw [List.getD_eq_getElem _ _ hplen, List.getD_eq_getElem _ _ h1'] at hwb
    cases hns : (buildNameToStep sp.stmts)[rf.1]? with
    | some pjs =>
        obtain ⟨jj, ss⟩ := pjs
        rw [hns, Except.ok.injEq] at hwb
        rw [← hwb]; simp only [Function.comp_apply, hns]
        exact internal_pointwise hrc hns
    | none =>
        cases hext : (buildExtIndex sp.extNames sp.stmts)[rf.1]? with
        | some k =>
            rw [hns, hext, Except.ok.injEq] at hwb
            rw [← hwb]; simp only [Function.comp_apply, hns]
            refine external_pointwise hrc hwfd hne rf hi h1' ?_ hext ?_
            · rw [List.getD_eq_getElem _ _ h1', ← hwb]
            · rw [hiw, List.getD_eq_getElem _ _ (by rw [List.length_map]; exact hplen),
                List.getElem_map, ← hrfdef, hns]
        | none => rw [hns, hext] at hwb; simp at hwb

/-- Conjunct 1 (`wellFormedDom`). Needs: every external slot referenced + rank agreement across
    consuming ports. (Pipeline-property dependent — `extNames ⊆ reads`, `extIndex` bound.) -/
theorem wf_dom {sp : ScheduledProgram} {tc : ThreadedComposed} {s s₁ : Nat} {p : TLProgram}
    (hsp : (TLProgram.compileToScheduled p).run s = .ok sp s₁)
    (hrc : routeCore sp = .ok (tc.steps, tc.routing)) (hne : tc.nExternal = sp.extNames.card) :
    tc.wellFormedDom = true := by
  sorry

/-! ### Conjunct 4 (`wf_topo`) infrastructure -/

/-- Membership in the fold base survives any `++`-prepending fold (`poolAt` prepends `outputSlots`). -/
private theorem mem_foldl_prepend {x : Wire} {f : Nat → List Wire} :
    ∀ (l : List Nat) (b : List Wire), x ∈ b →
      x ∈ l.foldl (fun p j => f j ++ p) b := by
  intro l
  induction l with
  | nil => intro b hb; simpa using hb
  | cons c u ihu =>
      intro b hb; simp only [List.foldl_cons]; exact ihu _ (List.mem_append_right _ hb)

/-- Part 1 (external): `poolAt i` contains every external wire with slot `< nExternal`. -/
theorem mem_poolAt_external {tc : ThreadedComposed} {i k : Nat} (h : k < tc.nExternal) :
    Wire.external k ∈ tc.poolAt i := by
  unfold ThreadedComposed.poolAt
  exact mem_foldl_prepend _ _ (List.mem_map.mpr ⟨k, List.mem_range.mpr h, rfl⟩)

/-- Part 1 (internal): `poolAt i` contains every producer wire `internal j s` with `j < i` and slot
    `s < n_j` (the producer step's output count) — generalized for multi-output steps. -/
theorem mem_poolAt_internal {tc : ThreadedComposed} {i j s : Nat} (h : j < i)
    (hs : s < (tc.steps.getD j default).outputWeaves.length) :
    Wire.internal j s ∈ tc.poolAt i := by
  unfold ThreadedComposed.poolAt
  have hj : j ∈ List.range i := List.mem_range.mpr h
  have hmemOut : Wire.internal j s ∈ tc.outputSlots j := by
    unfold ThreadedComposed.outputSlots
    exact List.mem_map.mpr ⟨s, List.mem_range.mpr hs, rfl⟩
  suffices H : ∀ (l : List Nat) (base : List Wire), j ∈ l →
      Wire.internal j s ∈ l.foldl (fun p x => tc.outputSlots x ++ p) base by
    exact H _ _ hj
  intro l
  induction l with
  | nil => intro base hb; simp at hb
  | cons a t ih =>
      intro base hb
      simp only [List.foldl_cons]
      rcases List.mem_cons.mp hb with h' | h'
      · subst h'; exact mem_foldl_prepend t _ (List.mem_append_left _ hmemOut)
      · exact ih _ h'

/-- Part 3 (the topological bound). Claims an internal producer wire `internal j 0` in `routing[i]`
    has `j < i`.

    ⚠️ FALSE AS STATED — TWO distinct counterexamples found (2026-06-26), both verified through the
    real pipeline. This sorry is the precise locus of a soundness/modeling gap, NOT a proof
    difficulty; closing it requires a design change (see plan-doc RESUME POINT):

    1. **True cycles.** The pipeline never rejects cyclic dataflow; `topoSortFuel`'s cycle branch
       falls back to source order. Program `A[a] := B[a]; B[a] := A[a]` compiles to
       `routing = [[internal 1 0], [internal 0 0]]` — step 0 has a FORWARD edge (`j = 1 > i = 0`).

    2. **Scan self-recurrence (deeper).** A coupled/self scan (e.g. `G[l+1] := f(G[l], H[l])`,
       `H[l+1] := f(H[l], G[l])`) lowers to ONE scan step `i` that reads `G`/`H` — both written by
       step `i` itself — so `buildStep` emits `internal i 0` self-wires (`j = i`). These are VALID
       programs, but `internal i 0 ∉ poolAt i` (the pool gains it only at `poolAt (i+1)`), so
       `realize`'s `stepPiece`/`wiringBy` cannot gather them. The threaded monotonic-pool model
       does not represent intra-step scan recurrence; self-reads should not be routing wires.

    The fix is the design owner's call (reject cycles AND absorb scan self-reads into the scan
    generator, or extend `poolAt`/the `WellFormed.topo` conjunct to admit `internal i 0`). -/
-- NOTE(phaseB): restated to take `hrc` (routeCore success ⇒ routableInOrder), which — with the
-- Phase-A acyclicity guard + self-read exclusion — makes BOTH former counterexamples inapplicable:
-- cycles are rejected up front, and scan self-reads are not in `inputReadFactors`.
theorem topo_bound {sp : ScheduledProgram} {steps : List BrBaseP} {routing : List (List Wire)}
    (hrc : routeCore sp = .ok (steps, routing))
    {i : Nat} (hi : i < sp.stmts.length) {rf : String × List IdxExpr}
    (hrf : rf ∈ (sp.stmts.getD i default).inputReadFactors)
    {j slot : Nat} (hns : (buildNameToStep sp.stmts)[rf.1]? = some (j, slot)) : j < i := by
  have hro := routeCore_routable hrc
  unfold routableInOrder at hro
  rw [List.all_eq_true] at hro
  have hmem : (sp.stmts.getD i default, i) ∈ sp.stmts.zipIdx := by
    rw [List.mk_mem_zipIdx_iff_getElem?, List.getElem?_eq_getElem hi, List.getD_eq_getElem _ _ hi]
  have hp := hro _ hmem
  dsimp only at hp
  rw [List.all_eq_true] at hp
  have hp2 := hp rf hrf
  rw [hns] at hp2
  exact of_decide_eq_true hp2

/-- Conjunct 4 (topological — reads ⊆ pool). Each routing wire is a producer `internal j slot`
    (`j < i` by `topo_bound`, `slot < n_j` by `buildNameToStep_slot_lt`) or an external `k`
    (`k < nExternal`) — all in `poolAt i` by `mem_poolAt_internal`/`mem_poolAt_external`. -/
theorem wf_topo {sp : ScheduledProgram} {tc : ThreadedComposed}
    (hrc : routeCore sp = .ok (tc.steps, tc.routing)) (hne : tc.nExternal = sp.extNames.card) :
    ∀ i, i < tc.steps.length → ∀ w ∈ tc.routing.getD i [], w ∈ tc.poolAt i := by
  intro i hi w hw
  have hik : i < sp.stmts.length := (routeCore_steps_length hrc) ▸ hi
  have hbuild := routeCore_getD hrc i hik
  have hwires := buildStep_wires_mapM hbuild
  obtain ⟨p, hp, hpw⟩ := List.getElem_of_mem hw
  have hplen : p < (sp.stmts.getD i default).inputReadFactors.length := (mapM_ok_length' hwires) ▸ hp
  have hwb := mapM_ok_getD' hwires p default (Wire.external 0) hplen
  rw [List.getD_eq_getElem _ _ hp, hpw] at hwb
  set rf := (sp.stmts.getD i default).inputReadFactors.getD p default with hrfdef
  have hrfmem : rf ∈ (sp.stmts.getD i default).inputReadFactors := by
    rw [hrfdef, List.getD_eq_getElem _ _ hplen]; exact List.getElem_mem _
  cases hns : (buildNameToStep sp.stmts)[rf.1]? with
  | some pjs =>
      obtain ⟨jj, ss⟩ := pjs
      rw [hns, Except.ok.injEq] at hwb
      subst hwb
      have hji : jj < i := topo_bound hrc hik hrfmem hns
      have hbuildj := routeCore_getD hrc jj (buildNameToStep_lt hns)
      have hslot : ss < (tc.steps.getD jj default).outputWeaves.length := by
        rw [buildStep_outputWeaves_length_one hbuildj]; exact buildNameToStep_slot_lt hns
      exact mem_poolAt_internal hji hslot
  | none =>
      cases hext : (buildExtIndex sp.extNames sp.stmts)[rf.1]? with
      | some k =>
          rw [hns, hext, Except.ok.injEq] at hwb
          subst hwb
          exact mem_poolAt_external (hne ▸ buildExtIndex_lt_card hext)
      | none => rw [hns, hext] at hwb; simp at hwb

/-- **The compiler theorem: every compiled program is `WellFormed`** (discharges `realize`'s
    precondition on real input). -/
theorem compile_wellFormed (p : TLProgram) (s : Nat) (tc : ThreadedComposed) (s' : Nat)
    (h : (TLProgram.compile p).run s = .ok tc s') : tc.WellFormed := by
  obtain ⟨sp, s₁, hsp, hrc, hne⟩ := compile_eq_route h
  have hdom := wf_dom hsp hrc hne
  exact ⟨hdom, wf_typeMatch hrc hdom hne, wf_singleOutput hrc, wf_topo hrc hne⟩

/-- Every compiled program crosses the bridge: the formal morphism exists. -/
noncomputable def realizeCompiled (p : TLProgram) (s : Nat) (tc : ThreadedComposed) (s' : Nat)
    (h : (TLProgram.compile p).run s = .ok tc s') : Σ (dom cod : BrObj), BrMorph dom cod :=
  realize tc (compile_wellFormed p s tc s' h)



/-- §8.2 acset extraction (`from_tensor_program`): a ThreadedComposed's tabular twin.
    OBLIGATION (`sorry`): the extraction algorithm (acset.md). -/
noncomputable def fromThreadedComposed (tc : ThreadedComposed) : Acset.SBrInstance := sorry

/-- **Prop 8 (DSL/CSV agreement).** The DSL-path realization of `tc` and the CSV-path
    realization of its extracted `SBrInstance` are the SAME `Br` morphism (equal as
    `Σ (dom cod : BrObj), BrMorph dom cod` values — same objects AND same morphism). -/
theorem realize_fromThreadedComposed_agree (tc : ThreadedComposed) (h : tc.WellFormed) :
    realize tc h = realizeSBr (fromThreadedComposed tc) := sorry

/-- **Prop 8′ (axis identity on the nose, §7.4).** Both paths share the §7.4 UID coequalizer,
    so the realized domain objects coincide. -/
theorem agree_dom (tc : ThreadedComposed) (h : tc.WellFormed) :
    (realize tc h).1 = (realizeSBr (fromThreadedComposed tc)).1 :=
  congr_arg (·.1) (realize_fromThreadedComposed_agree tc h)

/-- Prop 8′ (cod). -/
theorem agree_cod (tc : ThreadedComposed) (h : tc.WellFormed) :
    (realize tc h).2.1 = (realizeSBr (fromThreadedComposed tc)).2.1 :=
  congr_arg (·.2.1) (realize_fromThreadedComposed_agree tc h)

end LeanNCD

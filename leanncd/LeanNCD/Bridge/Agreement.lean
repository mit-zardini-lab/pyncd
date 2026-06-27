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

/-- Conjunct 3 (single output): every routed step has exactly one output weave. -/
theorem wf_singleOutput {sp : ScheduledProgram} {steps : List BrBaseP} {routing : List (List Wire)}
    (hrc : routeCore sp = .ok (steps, routing)) :
    ∀ i, i < steps.length → (steps.getD i default).outputWeaves.length = 1 := by
  intro i hi
  have hi' : i < sp.stmts.length := routeCore_steps_length hrc ▸ hi
  exact buildStep_outputWeaves_length_one (routeCore_getD hrc i hi')

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
  have hi' : i < sp.stmts.length := routeCore_steps_length hrc ▸ hi
  have hbs := routeCore_getD hrc i hi'
  have hiw := buildStep_inputWeaves hbs
  have hwm := buildStep_wires_mapM hbs
  set s := (sp.stmts.getD i default).repStmt.getD emptyStmt with hs
  set ns := buildNameToStep sp.stmts with hns
  set ext := buildExtIndex sp.extNames sp.stmts with hext
  have hlen : (tc.routing.getD i []).length = s.readFactors.length := mapM_ok_length' hwm
  have hj' : j < s.readFactors.length := hlen ▸ hj
  set rf := s.readFactors.getD j (("", []) : String × List IdxExpr) with hrf
  have hwb := mapM_ok_getD' hwm j (("", []) : String × List IdxExpr) (Wire.external 0) hj'
  rw [hw, ← hrf] at hwb
  have hnone : ns[rf.1]? = none := by
    revert hwb
    cases hnsr : ns[rf.1]? with
    | some jj => intro hwb; simp only [hnsr] at hwb; exact absurd hwb (by simp)
    | none => intro _; rfl
  refine ⟨rf, ?_, hnone, ?_⟩
  · revert hwb
    cases hnsr : ns[rf.1]? with
    | some jj => intro hwb; simp only [hnsr] at hwb; exact absurd hwb (by simp)
    | none =>
        cases hextr : ext[rf.1]? with
        | some kk => intro hwb; simp only [hnsr, hextr] at hwb
                     simp only [Except.ok.injEq, Wire.external.injEq] at hwb
                     exact congrArg some hwb
        | none => intro hwb; simp only [hnsr] at hwb; exact absurd hwb (by simp)
  · rw [hiw, List.getD_eq_getElem _ _ (by rw [List.length_map]; exact hj'),
        List.getElem_map,
        ← List.getD_eq_getElem s.readFactors (("", []) : String × List IdxExpr) hj', ← hrf]
    simp only [hnone]

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
private theorem internal_pointwise {sp : ScheduledProgram} {tc : ThreadedComposed}
    (hrc : routeCore sp = .ok (tc.steps, tc.routing))
    {rf : String × List IdxExpr} {j : Nat}
    (hns : (buildNameToStep sp.stmts)[rf.1]? = some j) :
    tc.wireType (Wire.internal j 0)
      = weaveToArrayType ((tensorAxes ((sp.stmts.getD j default).repStmt.getD emptyStmt)).map
          (fun a => WeaveSlotP.fixed a)) := by
  simp only [ThreadedComposed.wireType]
  apply weaveToArrayType_congr
  rw [fixedAxesP_map_fixed]
  exact buildStep_output_fixedAxes (routeCore_getD hrc j (buildNameToStep_lt hns))

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
  have hi' : i < sp.stmts.length := routeCore_steps_length hrc ▸ hi
  have hbs := routeCore_getD hrc i hi'
  have hiw := buildStep_inputWeaves hbs
  have hwm := buildStep_wires_mapM hbs
  set s := (sp.stmts.getD i default).repStmt.getD emptyStmt with hs
  set ns := buildNameToStep sp.stmts with hns
  have hlenW : (tc.routing.getD i []).length = s.readFactors.length := mapM_ok_length' hwm
  apply List.ext_getElem
  · rw [List.length_map, List.length_map, hiw, List.length_map, hlenW]
  · intro pos h1 h2
    rw [List.getElem_map, List.getElem_map]
    have hpos_rf : pos < s.readFactors.length := by
      rw [List.length_map] at h1; rw [hlenW] at h1; exact h1
    have hpos_route : pos < (tc.routing.getD i []).length := by
      rw [List.length_map] at h1; exact h1
    set rf := s.readFactors.getD pos (("", []) : String × List IdxExpr) with hrf
    have hweave : (tc.steps.getD i default).inputWeaves.getD pos [] =
        (match ns[rf.1]? with
        | some j => (tensorAxes ((sp.stmts.getD j default).repStmt.getD emptyStmt)).map
                      (fun a => WeaveSlotP.fixed a)
        | none => (List.range rf.2.length).map (fun pos =>
                    WeaveSlotP.fixed (AxisP.mk (some (rf.1 ++ "_" ++ toString pos))
                      (SizeExpr.var (rf.1 ++ "_" ++ toString pos))))) := by
      rw [hiw, List.getD_eq_getElem _ _ (by rw [List.length_map]; exact hpos_rf),
          List.getElem_map,
          ← List.getD_eq_getElem s.readFactors (("", []) : String × List IdxExpr) hpos_rf, ← hrf]
      rfl
    have hwire := mapM_ok_getD' hwm pos (("", []) : String × List IdxExpr) (Wire.external 0) hpos_rf
    rw [← hrf] at hwire
    have hpos_iw : pos < (tc.steps.getD i default).inputWeaves.length := by
      rw [hiw, List.length_map]; exact hpos_rf
    rw [show (tc.routing.getD i [])[pos] = (tc.routing.getD i []).getD pos (Wire.external 0) from
        (List.getD_eq_getElem _ _ hpos_route).symm]
    rw [show ((tc.steps.getD i default).inputWeaves)[pos]
          = (tc.steps.getD i default).inputWeaves.getD pos [] from
        (List.getD_eq_getElem _ _ hpos_iw).symm]
    rw [hweave]
    cases hnsr : ns[rf.1]? with
    | some jj =>
        simp only [hnsr] at hwire ⊢
        simp only [Except.ok.injEq] at hwire
        rw [← hwire]
        exact internal_pointwise hrc hnsr
    | none =>
        simp only [hnsr] at hwire ⊢
        cases hextr : (buildExtIndex sp.extNames sp.stmts)[rf.1]? with
        | none => simp only [hextr] at hwire; exact absurd hwire (by simp)
        | some k =>
            simp only [hextr, Except.ok.injEq] at hwire
            rw [← hwire]
            apply external_pointwise hrc hwfd hne rf hi hpos_route
            · rw [← hwire]
            · exact hextr
            · rw [hweave, hnsr]

/-- Conjunct 1 (`wellFormedDom`). Needs: every external slot referenced + rank agreement across
    consuming ports. (Pipeline-property dependent — `extNames ⊆ reads`, `extIndex` bound.) -/
theorem wf_dom {sp : ScheduledProgram} {tc : ThreadedComposed} {s s₁ : Nat} {p : TLProgram}
    (hsp : (TLProgram.compileToScheduled p).run s = .ok sp s₁)
    (hrc : routeCore sp = .ok (tc.steps, tc.routing)) (hne : tc.nExternal = sp.extNames.card) :
    tc.wellFormedDom = true := by
  sorry

/-! ### Conjunct 4 (`wf_topo`) infrastructure -/

/-- Membership in the fold base survives the `poolAt` prepending fold. -/
private theorem mem_foldl_prepend {x : Wire} :
    ∀ (l : List Nat) (b : List Wire), x ∈ b →
      x ∈ l.foldl (fun p j => Wire.internal j 0 :: p) b := by
  intro l
  induction l with
  | nil => intro b hb; simpa using hb
  | cons c u ihu =>
      intro b hb; simp only [List.foldl_cons]; exact ihu _ (List.mem_cons_of_mem _ hb)

/-- Part 1 (external): `poolAt i` contains every external wire with slot `< nExternal`. -/
theorem mem_poolAt_external {tc : ThreadedComposed} {i k : Nat} (h : k < tc.nExternal) :
    Wire.external k ∈ tc.poolAt i := by
  unfold ThreadedComposed.poolAt
  exact mem_foldl_prepend _ _ (List.mem_map.mpr ⟨k, List.mem_range.mpr h, rfl⟩)

/-- Part 1 (internal): `poolAt i` contains every producer wire `internal j 0` with `j < i`. -/
theorem mem_poolAt_internal {tc : ThreadedComposed} {i j : Nat} (h : j < i) :
    Wire.internal j 0 ∈ tc.poolAt i := by
  unfold ThreadedComposed.poolAt
  have hj : j ∈ List.range i := List.mem_range.mpr h
  suffices H : ∀ (l : List Nat) (base : List Wire), j ∈ l →
      Wire.internal j 0 ∈ l.foldl (fun p x => Wire.internal x 0 :: p) base by
    exact H _ _ hj
  intro l
  induction l with
  | nil => intro base hb; simp at hb
  | cons a t ih =>
      intro base hb
      simp only [List.foldl_cons]
      rcases List.mem_cons.mp hb with h' | h'
      · subst h'; exact mem_foldl_prepend t _ List.mem_cons_self
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
theorem topo_bound {sp : ScheduledProgram} {s s₁ : Nat} {p : TLProgram}
    (hsp : (TLProgram.compileToScheduled p).run s = .ok sp s₁)
    {i : Nat} (hi : i < sp.stmts.length) {rf : String × List IdxExpr}
    (hrf : rf ∈ ((sp.stmts.getD i default).repStmt.getD emptyStmt).readFactors)
    {j : Nat} (hns : (buildNameToStep sp.stmts)[rf.1]? = some j) : j < i := by
  sorry

/-- Conjunct 4 (topological — reads ⊆ pool). Needs `topoSort` correctness + `extIndex` bound. -/
theorem wf_topo {sp : ScheduledProgram} {tc : ThreadedComposed} {s s₁ : Nat} {p : TLProgram}
    (hsp : (TLProgram.compileToScheduled p).run s = .ok sp s₁)
    (hrc : routeCore sp = .ok (tc.steps, tc.routing)) (hne : tc.nExternal = sp.extNames.card) :
    ∀ i, i < tc.steps.length → ∀ w ∈ tc.routing.getD i [], w ∈ tc.poolAt i := by
  intro i hi w hw
  have hi' : i < sp.stmts.length := routeCore_steps_length hrc ▸ hi
  have hbs := routeCore_getD hrc i hi'
  have hwm := buildStep_wires_mapM hbs
  set s' := (sp.stmts.getD i default).repStmt.getD emptyStmt with hs'
  set ns := buildNameToStep sp.stmts with hns
  set ext := buildExtIndex sp.extNames sp.stmts with hext
  -- locate `w` at some position in the routing list
  obtain ⟨pos, hpos_lt, hpos_eq⟩ := List.getElem_of_mem hw
  have hlen : (tc.routing.getD i []).length = s'.readFactors.length := mapM_ok_length' hwm
  have hpos_rf : pos < s'.readFactors.length := hlen ▸ hpos_lt
  set rf := s'.readFactors.getD pos (("", []) : String × List IdxExpr) with hrf
  have hrf_mem : rf ∈ s'.readFactors := by
    rw [hrf, List.getD_eq_getElem _ _ hpos_rf]; exact List.getElem_mem hpos_rf
  have hwire := mapM_ok_getD' hwm pos (("", []) : String × List IdxExpr) (Wire.external 0) hpos_rf
  rw [← hrf] at hwire
  -- `routing[i].getD pos default = w`
  have hwpos : (tc.routing.getD i []).getD pos (Wire.external 0) = w := by
    rw [List.getD_eq_getElem _ _ hpos_lt, hpos_eq]
  rw [hwpos] at hwire
  cases hnsr : ns[rf.1]? with
  | some jj =>
      simp only [hnsr] at hwire
      simp only [Except.ok.injEq] at hwire
      -- w = internal jj 0, jj < i
      have hjj : jj < i := topo_bound hsp hi' hrf_mem hnsr
      rw [← hwire]
      exact mem_poolAt_internal hjj
  | none =>
      simp only [hnsr] at hwire
      cases hextr : ext[rf.1]? with
      | none => simp only [hextr] at hwire; exact absurd hwire (by simp)
      | some k =>
          simp only [hextr, Except.ok.injEq] at hwire
          have hk : k < tc.nExternal := hne ▸ buildExtIndex_lt_card hextr
          rw [← hwire]
          exact mem_poolAt_external hk

/-- **The compiler theorem: every compiled program is `WellFormed`** (discharges `realize`'s
    precondition on real input). -/
theorem compile_wellFormed (p : TLProgram) (s : Nat) (tc : ThreadedComposed) (s' : Nat)
    (h : (TLProgram.compile p).run s = .ok tc s') : tc.WellFormed := by
  obtain ⟨sp, s₁, hsp, hrc, hne⟩ := compile_eq_route h
  have hdom := wf_dom hsp hrc hne
  exact ⟨hdom, wf_typeMatch hrc hdom hne, wf_singleOutput hrc, wf_topo hsp hrc hne⟩

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

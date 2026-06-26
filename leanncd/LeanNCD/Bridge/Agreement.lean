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

/-- Conjunct 2 (producer ⊳ consumer type match). The de-risked one: producer output and consumer
    input weaves share fixed axes by construction (`buildStep_output_fixedAxes`). -/
theorem wf_typeMatch {sp : ScheduledProgram} {tc : ThreadedComposed}
    (hrc : routeCore sp = .ok (tc.steps, tc.routing)) :
    ∀ i, i < tc.steps.length →
      (tc.routing.getD i []).map tc.wireType
        = (tc.steps.getD i default).inputWeaves.map weaveToArrayType := by
  sorry

/-- Conjunct 1 (`wellFormedDom`). Needs: every external slot referenced + rank agreement across
    consuming ports. (Pipeline-property dependent — `extNames ⊆ reads`, `extIndex` bound.) -/
theorem wf_dom {sp : ScheduledProgram} {tc : ThreadedComposed} {s s₁ : Nat} {p : TLProgram}
    (hsp : (TLProgram.compileToScheduled p).run s = .ok sp s₁)
    (hrc : routeCore sp = .ok (tc.steps, tc.routing)) (hne : tc.nExternal = sp.extNames.card) :
    tc.wellFormedDom = true := by
  sorry

/-- Conjunct 4 (topological — reads ⊆ pool). Needs `topoSort` correctness + `extIndex` bound. -/
theorem wf_topo {sp : ScheduledProgram} {tc : ThreadedComposed} {s s₁ : Nat} {p : TLProgram}
    (hsp : (TLProgram.compileToScheduled p).run s = .ok sp s₁)
    (hrc : routeCore sp = .ok (tc.steps, tc.routing)) (hne : tc.nExternal = sp.extNames.card) :
    ∀ i, i < tc.steps.length → ∀ w ∈ tc.routing.getD i [], w ∈ tc.poolAt i := by
  sorry

/-- **The compiler theorem: every compiled program is `WellFormed`** (discharges `realize`'s
    precondition on real input). -/
theorem compile_wellFormed (p : TLProgram) (s : Nat) (tc : ThreadedComposed) (s' : Nat)
    (h : (TLProgram.compile p).run s = .ok tc s') : tc.WellFormed := by
  obtain ⟨sp, s₁, hsp, hrc, hne⟩ := compile_eq_route h
  exact ⟨wf_dom hsp hrc hne, wf_typeMatch hrc, wf_singleOutput hrc, wf_topo hsp hrc hne⟩

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

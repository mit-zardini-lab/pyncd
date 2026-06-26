import LeanNCD.DSL.Pipeline.Lowering

/-!
# `routeCore` structural specification (Phase 3 of the `WellFormed` plan)

Pure `List`/`Except` lemmas that turn `routeCore sp = .ok (steps, routing)` into per-index facts —
the bridge the `WellFormed` conjunct proofs (Phase 4) stand on. No `EStateM`; `routeCore` is a bare
`mapM` over `buildStep` with the named PASS-1 map builders (`buildNameToStep`/`buildExtIndex`/
`buildNameToType`, all in `Lowering.lean`).
-/

namespace LeanNCD

open Std

/-! ## Helpers: `List.mapM` over `Except` -/

/-- If `l.mapM f = .ok r` then `r.length = l.length`. -/
private theorem mapM_ok_length {ε α β : Type} {f : α → Except ε β} :
    ∀ {l : List α} {r : List β}, l.mapM f = .ok r → r.length = l.length := by
  intro l
  induction l with
  | nil => intro r h; simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h; subst h; rfl
  | cons a t ih =>
      intro r h
      rw [List.mapM_cons] at h
      cases hfa : f a with
      | error e => simp [hfa, bind, Except.bind] at h
      | ok b =>
          cases hft : t.mapM f with
          | error e => simp [hfa, hft, bind, Except.bind] at h
          | ok bs =>
              simp only [hfa, hft, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
              subst h
              simp [List.length_cons, ih hft]

/-- If `l.mapM f = .ok r` and `i < l.length`, then `f (l.getD i d₁) = .ok (r.getD i d₂)`
    (the i-th input maps successfully to the i-th output). -/
private theorem mapM_ok_getD {ε α β : Type} [Inhabited β] {f : α → Except ε β} :
    ∀ {l : List α} {r : List β}, l.mapM f = .ok r →
    ∀ (i : Nat) (d₁ : α) (d₂ : β), i < l.length →
    f (l.getD i d₁) = .ok (r.getD i d₂) := by
  intro l
  induction l with
  | nil => intro r _ i _ _ hi; simp at hi
  | cons a t ih =>
      intro r h i d₁ d₂ hi
      rw [List.mapM_cons] at h
      cases hfa : f a with
      | error e => simp [hfa, bind, Except.bind] at h
      | ok b =>
          cases hft : t.mapM f with
          | error e => simp [hfa, hft, bind, Except.bind] at h
          | ok bs =>
              simp only [hfa, hft, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
              subst h
              cases i with
              | zero => rw [List.getD_cons_zero, List.getD_cons_zero]; exact hfa
              | succ j =>
                  rw [List.getD_cons_succ, List.getD_cons_succ]
                  exact ih hft j d₁ d₂ (Nat.lt_of_succ_lt_succ hi)

/-! ## Lemma 1: `routeCore` output lengths equal `sp.stmts.length` -/

theorem routeCore_steps_length {sp : ScheduledProgram}
    {steps : List BrBaseP} {routing : List (List Wire)}
    (h : routeCore sp = .ok (steps, routing)) :
    steps.length = sp.stmts.length := by
  unfold routeCore at h
  cases hm : sp.stmts.mapM (buildStep (buildNameToStep sp.stmts)
      (buildExtIndex sp.extNames sp.stmts) (buildNameToType sp.stmts)) with
  | error e => rw [hm] at h; simp [bind, Except.bind] at h
  | ok pairs =>
      rw [hm] at h
      simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      rw [← h.1, List.length_map]
      exact mapM_ok_length hm

theorem routeCore_routing_length {sp : ScheduledProgram}
    {steps : List BrBaseP} {routing : List (List Wire)}
    (h : routeCore sp = .ok (steps, routing)) :
    routing.length = sp.stmts.length := by
  unfold routeCore at h
  cases hm : sp.stmts.mapM (buildStep (buildNameToStep sp.stmts)
      (buildExtIndex sp.extNames sp.stmts) (buildNameToType sp.stmts)) with
  | error e => rw [hm] at h; simp [bind, Except.bind] at h
  | ok pairs =>
      rw [hm] at h
      simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      rw [← h.2, List.length_map]
      exact mapM_ok_length hm

/-! ## Lemma 2: per-index characterization -/

/-- The i-th output pair of `routeCore` equals `buildStep` of the i-th stmt, under the PASS-1 maps. -/
theorem routeCore_getD {sp : ScheduledProgram}
    {steps : List BrBaseP} {routing : List (List Wire)}
    (h : routeCore sp = .ok (steps, routing)) (i : Nat) (hi : i < sp.stmts.length) :
    buildStep (buildNameToStep sp.stmts) (buildExtIndex sp.extNames sp.stmts)
        (buildNameToType sp.stmts) (sp.stmts.getD i default)
      = .ok (steps.getD i default, routing.getD i []) := by
  unfold routeCore at h
  cases hm : sp.stmts.mapM (buildStep (buildNameToStep sp.stmts)
      (buildExtIndex sp.extNames sp.stmts) (buildNameToType sp.stmts)) with
  | error e => rw [hm] at h; simp [bind, Except.bind] at h
  | ok pairs =>
      rw [hm] at h
      simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨hs, hr⟩ := h
      have hlen : pairs.length = sp.stmts.length := mapM_ok_length hm
      have hip : i < pairs.length := hlen ▸ hi
      rw [mapM_ok_getD hm i default default hi, ← hs, ← hr]
      congr 1
      rw [List.getD_eq_getElem _ _ hip,
          List.getD_eq_getElem _ _ (by simpa using hip),
          List.getD_eq_getElem _ _ (by simpa using hip),
          List.getElem_map, List.getElem_map]

/-! ## Lemma 3: `buildStep` always produces exactly one output weave -/

/-- The tail of `buildStep` is `(wires-mapM) >>= fun wires => pure (step, wires)`; if it succeeds
    the produced first component is exactly `step`. -/
private theorem bind_pure_pair_ok {ε γ δ : Type} {B : Except ε γ} {s b : δ} {w : γ}
    (h : (B >>= fun x => pure (s, x)) = Except.ok (b, w)) : b = s := by
  cases hB : B with
  | error e => rw [hB] at h; simp [bind, Except.bind] at h
  | ok v =>
      rw [hB] at h
      simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      exact h.1.symm

theorem buildStep_outputWeaves_length_one
    {ns ext : Std.HashMap String Nat} {nt : Std.HashMap String (List AxisP)}
    {sc : ScanStmt} {b : BrBaseP} {w : List Wire}
    (h : buildStep ns ext nt sc = .ok (b, w)) :
    b.outputWeaves.length = 1 := by
  unfold buildStep at h
  cases sc with
  | plain s => simp only [pure_bind] at h; rw [bind_pure_pair_ok h]; rfl
  | scan nm ax bs rs aff => simp only [pure_bind] at h; rw [bind_pure_pair_ok h]; rfl
  | scanPre nm ax tc =>
      by_cases he : tc.steps.isEmpty = true
      · simp [he, bind, Except.bind] at h
      · simp only [he, pure_bind] at h
        rw [bind_pure_pair_ok h]; rfl

end LeanNCD

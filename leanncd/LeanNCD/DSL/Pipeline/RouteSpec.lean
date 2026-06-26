import LeanNCD.DSL.Pipeline.Lowering

/-!
# `routeCore` structural specification (Phase 3 of the `WellFormed` plan)

Pure `List`/`Except` lemmas that turn `routeCore sp = .ok (steps, routing)` into per-index facts —
the bridge the `WellFormed` conjunct proofs (Phase 4) stand on. No `EStateM`; `routeCore` is a bare
`mapM` over `buildStep` with the named PASS-1 map builders (`buildNameToStep`/`buildExtIndex`/
and `sp.stmts` for internal-read producer axes, all in `Lowering.lean`).
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
      (buildExtIndex sp.extNames sp.stmts) sp.stmts) with
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
      (buildExtIndex sp.extNames sp.stmts) sp.stmts) with
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
        sp.stmts (sp.stmts.getD i default)
      = .ok (steps.getD i default, routing.getD i []) := by
  unfold routeCore at h
  cases hm : sp.stmts.mapM (buildStep (buildNameToStep sp.stmts)
      (buildExtIndex sp.extNames sp.stmts) sp.stmts) with
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
    {ns ext : Std.HashMap String Nat} {stmts : List ScanStmt}
    {sc : ScanStmt} {b : BrBaseP} {w : List Wire}
    (h : buildStep ns ext stmts sc = .ok (b, w)) :
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

/-! ## Lemma 4 (conjunct-2 engine): a built step publishes `tensorAxes` of its rep stmt -/

/-- Presentation-level fixed (retained) axes of a weave — the `.fixed` slots' `AxisP`s, in order. -/
def fixedAxesP (w : WeaveShapeP) : List AxisP :=
  w.filterMap (fun s => match s with | .fixed a => some a | .tiled => none)

/-- The dedup step function (matches `dedupByUid`'s `foldl`). -/
private def dstep (acc : List AxisSpec) (a : AxisSpec) : List AxisSpec :=
  if acc.any (fun b => b.uid == a.uid) then acc else acc ++ [a]

private theorem dedupByUid_eq_foldl (xs : List AxisSpec) :
    dedupByUid xs = xs.foldl dstep [] := rfl

/-- Every element produced by folding `dstep` came from `acc` or the folded list. -/
private theorem mem_foldl_dstep {a : AxisSpec} :
    ∀ (xs acc : List AxisSpec), a ∈ xs.foldl dstep acc → a ∈ acc ∨ a ∈ xs := by
  intro xs
  induction xs with
  | nil => intro acc h; exact Or.inl h
  | cons b t ih =>
      intro acc h
      simp only [List.foldl_cons] at h
      rcases ih _ h with h' | h'
      · unfold dstep at h'
        by_cases hb : acc.any (fun c => c.uid == b.uid)
        · simp [hb] at h'; exact Or.inl h'
        · simp [hb] at h'
          rcases h' with h' | h'
          · exact Or.inl h'
          · exact Or.inr (by simp [h'])
      · exact Or.inr (List.mem_cons_of_mem _ h')

/-- Folding `dstep` over a list all of whose elements fail `P`, then filtering by `P`, equals just
    filtering the accumulator: the appended elements all fail `P` and drop out. -/
private theorem foldl_dstep_filter (P : AxisSpec → Bool) :
    ∀ (ys acc : List AxisSpec), (∀ b ∈ ys, P b = false) →
    (ys.foldl dstep acc).filter P = acc.filter P := by
  intro ys
  induction ys with
  | nil => intro acc _; rfl
  | cons b t ih =>
      intro acc hbad
      simp only [List.foldl_cons]
      rw [ih (dstep acc b) (fun x hx => hbad x (List.mem_cons_of_mem _ hx))]
      unfold dstep
      by_cases hb : acc.any (fun c => c.uid == b.uid)
      · simp [hb]
      · simp only [hb, Bool.false_eq_true, if_false, List.filter_append]
        have : [b].filter P = [] := by simp [List.filter, hbad b (List.mem_cons_self ..)]
        rw [this, List.append_nil]

/-- Deduping `xs ++ ys` then dropping the `ys`-uid elements recovers `dedupByUid xs`, when every
    `ys` element's uid is absent from `xs` (the disjointness `buildStep` gives between `lhsAxes` and
    the contracted axes). -/
private theorem dedup_append_filter {xs ys : List AxisSpec}
    (hdisj : ∀ a ∈ ys, a.uid ∉ xs.map (·.uid)) :
    (dedupByUid (xs ++ ys)).filter (fun a => !(ys.map (·.uid)).contains a.uid) = dedupByUid xs := by
  rw [dedupByUid_eq_foldl, dedupByUid_eq_foldl, List.foldl_append]
  set P : AxisSpec → Bool := fun a => !(ys.map (·.uid)).contains a.uid with hP
  have hbad : ∀ b ∈ ys, P b = false := by
    intro b hb
    simp only [hP, List.contains_eq_mem, Bool.not_eq_false', decide_eq_true_eq, List.mem_map]
    exact ⟨b, hb, rfl⟩
  rw [foldl_dstep_filter P ys (xs.foldl dstep []) hbad]
  apply List.filter_eq_self.mpr
  intro a ha
  rcases (mem_foldl_dstep xs [] ha) with h | h
  · simp at h
  · have haxs : a.uid ∈ xs.map (·.uid) := List.mem_map.mpr ⟨a, h, rfl⟩
    have hnot : ¬ (a.uid ∈ ys.map (·.uid)) := by
      intro hcon
      obtain ⟨b, hb, hbe⟩ := List.mem_map.mp hcon
      exact hdisj b hb (hbe ▸ haxs)
    simpa [hP, List.contains_iff_mem] using hnot

/-- `fixedAxesP` of a `mkWeave`-shaped map: the `.fixed` slots are exactly the non-contracted axes,
    mapped to their `AxisP`. -/
private theorem fixedAxesP_mapWeave (l : List AxisSpec) (cu : List UID) :
    fixedAxesP (l.map (fun a => if cu.contains a.uid then WeaveSlotP.tiled
                                else WeaveSlotP.fixed (AxisP.mk (some a.name) (SizeExpr.var a.name))))
      = (l.filter (fun a => !cu.contains a.uid)).map
          (fun a => AxisP.mk (some a.name) (SizeExpr.var a.name)) := by
  induction l with
  | nil => rfl
  | cons a t ih =>
      by_cases hc : a.uid ∈ cu <;>
        simp_all [fixedAxesP, List.map_cons, List.contains_eq_mem]

/-- Fact A — the conjunct-2 engine: a step built by `buildStep` publishes exactly `tensorAxes` of its
    rep stmt as the fixed axes of its (single) output weave. Since `buildStep` derives an internal
    read's input weave from the same `tensorAxes` of the producer (post-refactor), producer output
    and consumer input weaves share these fixed axes — making conjunct 2 hold by construction. -/
theorem buildStep_output_fixedAxes
    {ns ext : Std.HashMap String Nat} {stmts : List ScanStmt}
    {sc : ScanStmt} {b : BrBaseP} {w : List Wire}
    (h : buildStep ns ext stmts sc = .ok (b, w)) :
    fixedAxesP (b.outputWeaves.getD 0 []) = tensorAxes (sc.repStmt.getD emptyStmt) := by
  -- 1. b is the constructed step; its single output weave is `stepMkWeave s`.
  have hb : b.outputWeaves.getD 0 [] = stepMkWeave (sc.repStmt.getD emptyStmt) := by
    unfold buildStep at h
    cases sc with
    | plain s => simp only [pure_bind] at h; rw [bind_pure_pair_ok h]; rfl
    | scan nm ax bs rs aff => simp only [pure_bind] at h; rw [bind_pure_pair_ok h]; rfl
    | scanPre nm ax tc =>
        by_cases he : tc.steps.isEmpty = true
        · simp [he, bind, Except.bind] at h
        · simp only [he, pure_bind] at h; rw [bind_pure_pair_ok h]; rfl
  -- 2. `fixedAxesP (stepMkWeave s)` keeps the non-contracted degree axes, which dedup to `lhsAxes`.
  rw [hb]
  unfold stepMkWeave tensorAxes
  rw [fixedAxesP_mapWeave]
  congr 1
  unfold stepDegAxes
  apply dedup_append_filter
  intro a ha
  simpa [List.contains_eq_mem] using (List.mem_filter.mp ha).2

end LeanNCD

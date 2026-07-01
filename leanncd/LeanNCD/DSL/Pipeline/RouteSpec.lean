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
  by_cases hro : routableInOrder sp.stmts
  · rw [if_pos hro] at h
    cases hm : sp.stmts.mapM (buildStep (buildNameToStep sp.stmts)
        (buildExtIndex sp.extNames sp.stmts) sp.stmts) with
    | error e => rw [hm] at h; simp [bind, Except.bind] at h
    | ok pairs =>
        rw [hm] at h
        simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
        rw [← h.1, List.length_map]
        exact mapM_ok_length hm
  · rw [if_neg hro] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h

theorem routeCore_routing_length {sp : ScheduledProgram}
    {steps : List BrBaseP} {routing : List (List Wire)}
    (h : routeCore sp = .ok (steps, routing)) :
    routing.length = sp.stmts.length := by
  unfold routeCore at h
  by_cases hro : routableInOrder sp.stmts
  · rw [if_pos hro] at h
    cases hm : sp.stmts.mapM (buildStep (buildNameToStep sp.stmts)
        (buildExtIndex sp.extNames sp.stmts) sp.stmts) with
    | error e => rw [hm] at h; simp [bind, Except.bind] at h
    | ok pairs =>
        rw [hm] at h
        simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
        rw [← h.2, List.length_map]
        exact mapM_ok_length hm
  · rw [if_neg hro] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h

/-! ## Lemma 2: per-index characterization -/

/-- The i-th output pair of `routeCore` equals `buildStep` of the i-th stmt, under the PASS-1 maps. -/
theorem routeCore_getD {sp : ScheduledProgram}
    {steps : List BrBaseP} {routing : List (List Wire)}
    (h : routeCore sp = .ok (steps, routing)) (i : Nat) (hi : i < sp.stmts.length) :
    buildStep (buildNameToStep sp.stmts) (buildExtIndex sp.extNames sp.stmts)
        sp.stmts (sp.stmts.getD i default)
      = .ok (steps.getD i default, routing.getD i []) := by
  unfold routeCore at h
  by_cases hro : routableInOrder sp.stmts
  · rw [if_pos hro] at h
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
  · rw [if_neg hro] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h

/-- A successful `routeCore` implies the program routes in topological order. -/
theorem routeCore_routable {sp : ScheduledProgram}
    {steps : List BrBaseP} {routing : List (List Wire)}
    (h : routeCore sp = .ok (steps, routing)) :
    routableInOrder sp.stmts = true := by
  unfold routeCore at h
  by_cases hro : routableInOrder sp.stmts = true
  · exact hro
  · rw [if_neg hro] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h

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

/-- A successful `buildStep` implies its step guard passed (the guard throws otherwise, so `.ok`
    forces `stepGuardOk = true`). -/
theorem buildStep_ok_guard
    {ns : Std.HashMap String (Nat × Nat)} {ext : Std.HashMap String Nat} {stmts : List ScanStmt}
    {sc : ScanStmt} {b : BrBaseP} {w : List Wire}
    (h : buildStep ns ext stmts sc = .ok (b, w)) :
    sc.stepGuardOk = true := by
  by_cases hg : sc.stepGuardOk = true
  · exact hg
  · exfalso
    unfold buildStep at h
    simp only [Bool.not_eq_true] at hg
    cases sc with
    | plain s => simp [hg, bind, Except.bind, pure, Except.pure] at h
    | scan nm ax bs rs aff => simp [hg, bind, Except.bind, pure, Except.pure] at h
    | scanPre nm ax tc =>
        simp only [bind, Except.bind, pure, Except.pure] at h
        split at h <;> simp [hg] at h

/-- A successful `buildStep` implies its outputs are consistent (projected from the guard). -/
theorem buildStep_ok_consistent
    {ns : Std.HashMap String (Nat × Nat)} {ext : Std.HashMap String Nat} {stmts : List ScanStmt}
    {sc : ScanStmt} {b : BrBaseP} {w : List Wire}
    (h : buildStep ns ext stmts sc = .ok (b, w)) :
    sc.outputAxesConsistent = true := by
  have hg := buildStep_ok_guard h
  unfold ScanStmt.stepGuardOk at hg
  simp only [Bool.and_eq_true] at hg
  exact hg.2

/-- A successful `buildStep` implies the step has at least one true output. -/
theorem buildStep_ok_outputs_ne
    {ns : Std.HashMap String (Nat × Nat)} {ext : Std.HashMap String Nat} {stmts : List ScanStmt}
    {sc : ScanStmt} {b : BrBaseP} {w : List Wire}
    (h : buildStep ns ext stmts sc = .ok (b, w)) :
    sc.outputs.isEmpty = false := by
  have hg := buildStep_ok_guard h
  unfold ScanStmt.stepGuardOk at hg
  simp only [Bool.and_eq_true] at hg
  simpa using hg.1

-- NOTE(phaseB): renamed-in-place; true length is `sc.outputs.length` (base∩recur for scans).
theorem buildStep_outputWeaves_length_one
    {ns : Std.HashMap String (Nat × Nat)} {ext : Std.HashMap String Nat} {stmts : List ScanStmt}
    {sc : ScanStmt} {b : BrBaseP} {w : List Wire}
    (h : buildStep ns ext stmts sc = .ok (b, w)) :
    b.outputWeaves.length = sc.outputs.length := by
  have hg := buildStep_ok_guard h
  unfold buildStep at h
  cases sc with
  | plain s =>
      simp only [hg, Bool.not_true, pure_bind] at h; rw [bind_pure_pair_ok h]
      simp [List.length_map, List.length_range]
  | scan nm ax bs rs aff =>
      simp only [hg, Bool.not_true, pure_bind] at h; rw [bind_pure_pair_ok h]
      simp [List.length_map, List.length_range]
  | scanPre nm ax tc =>
      by_cases he : tc.steps.isEmpty = true
      · simp [he, bind, Except.bind] at h
      · simp only [hg, Bool.not_true, he, pure_bind] at h
        rw [bind_pure_pair_ok h]
        simp [List.length_map, List.length_range]

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

/-- `fixedAxesP` of a `slotWeave`-shaped map (fixed WHEN the uid is retained): the `.fixed` slots
    are exactly the retained axes, mapped to their `AxisP`. Positive-polarity analogue of
    `fixedAxesP_mapWeave`. -/
private theorem fixedAxesP_mapWeave_pos (l : List AxisSpec) (ru : List UID) :
    fixedAxesP (l.map (fun a => if ru.contains a.uid
        then WeaveSlotP.fixed (AxisP.mk (some a.name) (SizeExpr.var a.name))
        else WeaveSlotP.tiled))
      = (l.filter (fun a => ru.contains a.uid)).map
          (fun a => AxisP.mk (some a.name) (SizeExpr.var a.name)) := by
  induction l with
  | nil => rfl
  | cons a t ih =>
      by_cases hc : a.uid ∈ ru <;>
        simp_all [fixedAxesP, List.map_cons, List.contains_eq_mem]

/-- The `outputWeaves` of a built step equal the per-slot `slotWeave` map. -/
theorem buildStep_outputWeaves
    {ns : Std.HashMap String (Nat × Nat)} {ext : Std.HashMap String Nat} {stmts : List ScanStmt}
    {sc : ScanStmt} {b : BrBaseP} {w : List Wire}
    (h : buildStep ns ext stmts sc = .ok (b, w)) :
    b.outputWeaves = (List.range sc.outputs.length).map sc.slotWeave := by
  have hg := buildStep_ok_guard h
  unfold buildStep at h
  cases sc with
  | plain s => simp only [hg, Bool.not_true, pure_bind] at h; exact congrArg BrBaseP.outputWeaves (bind_pure_pair_ok h)
  | scan nm ax bs rs aff =>
      simp only [hg, Bool.not_true, pure_bind] at h; exact congrArg BrBaseP.outputWeaves (bind_pure_pair_ok h)
  | scanPre nm ax tc =>
      by_cases he : tc.steps.isEmpty = true
      · simp [he, bind, Except.bind] at h
      · simp only [hg, Bool.not_true, he, pure_bind] at h
        exact congrArg BrBaseP.outputWeaves (bind_pure_pair_ok h)

/-- Fact A — the conjunct-2 engine: a step built by `buildStep` publishes exactly `tensorAxes` of its
    rep stmt as the fixed axes of its (single) output weave. Since `buildStep` derives an internal
    read's input weave from the same `tensorAxes` of the producer (post-refactor), producer output
    and consumer input weaves share these fixed axes — making conjunct 2 hold by construction. -/
-- NOTE(phaseB): per-slot statement for the new multi-output shape. Body to be proved in Phase B.
theorem buildStep_output_fixedAxes
    {ns : Std.HashMap String (Nat × Nat)} {ext : Std.HashMap String Nat} {stmts : List ScanStmt}
    {sc : ScanStmt} {b : BrBaseP} {w : List Wire} {s : Nat}
    (h : buildStep ns ext stmts sc = .ok (b, w)) (hs : s < sc.outputs.length) :
    fixedAxesP (b.outputWeaves.getD s []) = tensorAxes (sc.slotStmt s) := by
  have hg := buildStep_ok_consistent h
  have hgetD : b.outputWeaves.getD s [] = sc.slotWeave s := by
    rw [buildStep_outputWeaves h, List.getD_eq_getElem _ _ (by simpa using hs)]
    simp [List.getElem_map, List.getElem_range]
  rw [hgetD]
  unfold ScanStmt.slotWeave
  rw [fixedAxesP_mapWeave_pos]
  have hcons := of_decide_eq_true (List.all_eq_true.mp hg s (List.mem_range.mpr hs))
  rw [hcons]
  rfl

/-! ## Lemma 5: `buildNameToStep` value bound -/

-- NOTE(phaseB): the old helpers `foldl_insert_val_lt`/`foldl_zipIdx_val_lt` were written for the
-- old `String ↦ Nat` fold (inserting `i`). The new `buildNameToStep` folds `nm ↦ (i, s)` over
-- `sc.outputs.zipIdx`; the bounds are re-derived below via a single fold-invariant lemma.

/-- A `foldl` of a `step` that preserves an invariant `I` (for arguments drawn from the folded
    list) carries `I` from the initial map through to the result. The fold's value-bounds for
    `buildNameToStep` instantiate this with `I m := every recorded value is in range`. -/
private theorem foldl_preserves_inv {α β γ : Type} [BEq α] [Hashable α]
    {I : Std.HashMap α β → Prop}
    (step : Std.HashMap α β → γ → Std.HashMap α β) :
    ∀ (l : List γ) (init : Std.HashMap α β),
      I init → (∀ m c, c ∈ l → I m → I (step m c)) → I (l.foldl step init) := by
  intro l
  induction l with
  | nil => intro init hinit _; exact hinit
  | cons c t ih =>
      intro init hinit hstep
      simp only [List.foldl_cons]
      refine ih (step init c) (hstep init c (List.mem_cons_self ..) hinit) ?_
      intro m c' hc' hm
      exact hstep m c' (List.mem_cons_of_mem _ hc') hm

/-- The structural bound on `buildNameToStep`: every recorded value `(j, s)` has `j` a valid step
    index and `s` a valid output slot of step `j`. Both public bounds project out of this. -/
private theorem buildNameToStep_value {stmts : List ScanStmt} {nm : String} {j s : Nat}
    (h : (buildNameToStep stmts)[nm]? = some (j, s)) :
    j < stmts.length ∧ s < (stmts.getD j default).outputs.length := by
  -- Invariant: every recorded value `(j, s)` is in range. A concrete `let` (not a metavariable)
  -- keeps `foldl_preserves_inv`'s `I (foldl …)` unification first-order.
  let I : Std.HashMap String (Nat × Nat) → Prop :=
    fun m => ∀ (k : String) (v : Nat × Nat), m[k]? = some v →
      v.1 < stmts.length ∧ v.2 < (stmts.getD v.1 default).outputs.length
  suffices hmain : I (buildNameToStep stmts) from hmain nm (j, s) h
  unfold buildNameToStep
  refine foldl_preserves_inv (I := I) _ _ _ ?_ ?_
  · intro k v hk; simp at hk
  · rintro m ⟨sc, i⟩ hmem hIm
    obtain ⟨hi, hsc⟩ := List.mem_zipIdx' hmem
    dsimp only
    refine foldl_preserves_inv (I := I) _ _ _ hIm ?_
    rintro m' ⟨n, t⟩ hmem' hIm'
    obtain ⟨ht, -⟩ := List.mem_zipIdx' hmem'
    intro k v hk
    rw [Std.HashMap.getElem?_insert] at hk
    split at hk
    · simp only [Option.some.injEq] at hk
      subst hk
      refine ⟨hi, ?_⟩
      have hgetD : stmts.getD i default = sc := by
        rw [List.getD_eq_getElem _ _ hi]; exact hsc.symm
      simp only [hgetD]; exact ht
    · exact hIm' k v hk

/-- Every step-index value in `buildNameToStep stmts` is a valid step index `< stmts.length`. -/
theorem buildNameToStep_lt {stmts : List ScanStmt} {nm : String} {j s : Nat}
    (h : (buildNameToStep stmts)[nm]? = some (j, s)) : j < stmts.length :=
  (buildNameToStep_value h).1

/-- Every slot value in `buildNameToStep stmts` is a valid output slot for its producer step. -/
theorem buildNameToStep_slot_lt {stmts : List ScanStmt} {nm : String} {j s : Nat}
    (h : (buildNameToStep stmts)[nm]? = some (j, s)) :
    s < ((stmts.getD j default).outputs.length) :=
  (buildNameToStep_value h).2

/-! ## Lemma 6: `buildStep` field-extraction -/

/-- The second component of a successful `(B >>= fun x => pure (s, x)) = .ok (b, w)` is `B = .ok w`. -/
private theorem bind_pure_pair_ok_snd {ε γ δ : Type} {B : Except ε γ} {s b : δ} {w : γ}
    (h : (B >>= fun x => pure (s, x)) = Except.ok (b, w)) : B = .ok w := by
  cases hB : B with
  | error e => rw [hB] at h; simp [bind, Except.bind] at h
  | ok v =>
      simp only [hB, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      exact congrArg Except.ok h.2

/-- The `inputWeaves` of a built step equal the per-read-factor map from `buildStep`. -/
-- NOTE(phaseB): restated to the new `inputReadFactors` / `(j, slot)` shape. Body to be proved later.
theorem buildStep_inputWeaves
    {ns : Std.HashMap String (Nat × Nat)} {ext : Std.HashMap String Nat} {stmts : List ScanStmt}
    {sc : ScanStmt} {b : BrBaseP} {w : List Wire}
    (h : buildStep ns ext stmts sc = .ok (b, w)) :
    b.inputWeaves = sc.inputReadFactors.map (fun rf =>
      match ns[rf.1]? with
      | some (j, slot) => (tensorAxes ((stmts.getD j default).slotStmt slot)).map
                            (fun a => WeaveSlotP.fixed a)
      | none   => (List.range rf.2.length).map (fun pos =>
                    WeaveSlotP.fixed (AxisP.mk (some (rf.1 ++ "_" ++ toString pos))
                      (SizeExpr.var (rf.1 ++ "_" ++ toString pos))))) := by
  have hg := buildStep_ok_guard h
  unfold buildStep at h
  cases sc with
  | plain s => simp only [hg, Bool.not_true, pure_bind] at h; exact congrArg BrBaseP.inputWeaves (bind_pure_pair_ok h)
  | scan nm ax bs rs aff =>
      simp only [hg, Bool.not_true, pure_bind] at h; exact congrArg BrBaseP.inputWeaves (bind_pure_pair_ok h)
  | scanPre nm ax tc =>
      by_cases he : tc.steps.isEmpty = true
      · simp [he, bind, Except.bind] at h
      · simp only [hg, Bool.not_true, he, pure_bind] at h
        exact congrArg BrBaseP.inputWeaves (bind_pure_pair_ok h)

/-- The wires from a successful `buildStep` equal the `mapM` of the per-read-factor wire builder. -/
-- NOTE(phaseB): restated to the new `inputReadFactors` / `Wire.internal j slot` shape.
theorem buildStep_wires_mapM
    {ns : Std.HashMap String (Nat × Nat)} {ext : Std.HashMap String Nat} {stmts : List ScanStmt}
    {sc : ScanStmt} {b : BrBaseP} {w : List Wire}
    (h : buildStep ns ext stmts sc = .ok (b, w)) :
    sc.inputReadFactors.mapM (fun rf =>
      match ns[rf.1]? with
      | some (j, slot) => Except.ok (Wire.internal j slot)
      | none   => match ext[rf.1]? with
        | some k => Except.ok (Wire.external k)
        | none   => Except.error (CompileError.undeclaredName rf.1)) = .ok w := by
  have hg := buildStep_ok_guard h
  unfold buildStep at h
  cases sc with
  | plain s => simp only [hg, Bool.not_true, pure_bind] at h; exact bind_pure_pair_ok_snd h
  | scan nm ax bs rs aff => simp only [hg, Bool.not_true, pure_bind] at h; exact bind_pure_pair_ok_snd h
  | scanPre nm ax tc =>
      by_cases he : tc.steps.isEmpty = true
      · simp [he, bind, Except.bind] at h
      · simp only [hg, Bool.not_true, he, pure_bind] at h; exact bind_pure_pair_ok_snd h

/-! ## Lemma 7: buildExtIndex injectivity -/

private def goodExtState (m : Std.HashMap String Nat) (cnt : Nat) : Prop :=
  (∀ nm : String, ∀ v : Nat, m[nm]? = some v → v < cnt) ∧
  (∀ nm₁ nm₂ : String, ∀ v : Nat, m[nm₁]? = some v → m[nm₂]? = some v → nm₁ = nm₂)

private lemma goodExtState_step (extNames : Finset String) (nm : String)
    {state : Std.HashMap String Nat × Nat}
    (hgood : goodExtState state.1 state.2) :
    goodExtState (if decide (nm ∈ extNames) && !state.1.contains nm
                  then (state.1.insert nm state.2, state.2 + 1)
                  else state).1
                 (if decide (nm ∈ extNames) && !state.1.contains nm
                  then (state.1.insert nm state.2, state.2 + 1)
                  else state).2 := by
  by_cases h : decide (nm ∈ extNames) && !state.1.contains nm
  · simp only [h, ite_true]
    refine ⟨fun (nm' : String) (v : Nat) hv => ?_, fun (nm₁ nm₂ : String) (v : Nat) hv₁ hv₂ => ?_⟩
    · simp only [HashMap.getElem?_insert] at hv; split at hv
      · cases hv; omega
      · exact Nat.lt_succ_of_lt (hgood.1 nm' v hv)
    · simp only [HashMap.getElem?_insert] at hv₁ hv₂
      split at hv₁ <;> split at hv₂
      · simp_all
      · cases hv₁; exact absurd (hgood.1 nm₂ _ hv₂) (by omega)
      · cases hv₂; exact absurd (hgood.1 nm₁ _ hv₁) (by omega)
      · exact hgood.2 nm₁ nm₂ v hv₁ hv₂
  · simp only [h]; exact hgood

private lemma goodExtState_foldl_reads (extNames : Finset String) (reads : List String)
    {state : Std.HashMap String Nat × Nat}
    (hgood : goodExtState state.1 state.2) :
    goodExtState (reads.foldl (fun (m, cnt) nm =>
      if decide (nm ∈ extNames) && !m.contains nm then
        (m.insert nm cnt, cnt + 1)
      else (m, cnt)) state).1
      (reads.foldl (fun (m, cnt) nm =>
        if decide (nm ∈ extNames) && !m.contains nm then
          (m.insert nm cnt, cnt + 1)
        else (m, cnt)) state).2 := by
  induction reads generalizing state with
  | nil => exact hgood
  | cons nm t ih =>
      simp only [List.foldl_cons]
      exact ih (goodExtState_step extNames nm hgood)

private lemma goodExtState_foldl_stmts (extNames : Finset String) (stmts : List ScanStmt)
    {state : Std.HashMap String Nat × Nat}
    (hgood : goodExtState state.1 state.2) :
    goodExtState (stmts.foldl (fun (acc : Std.HashMap String Nat × Nat) sc =>
      sc.reads.foldl (fun (m, cnt) nm =>
        if decide (nm ∈ extNames) && !m.contains nm then
          (m.insert nm cnt, cnt + 1)
        else (m, cnt)) acc) state).1
      (stmts.foldl (fun (acc : Std.HashMap String Nat × Nat) sc =>
        sc.reads.foldl (fun (m, cnt) nm =>
          if decide (nm ∈ extNames) && !m.contains nm then
            (m.insert nm cnt, cnt + 1)
          else (m, cnt)) acc) state).2 := by
  induction stmts generalizing state with
  | nil => exact hgood
  | cons sc t ih =>
      simp only [List.foldl_cons]
      exact ih (goodExtState_foldl_reads extNames sc.reads hgood)

theorem buildExtIndex_injective {extNames : Finset String} {stmts : List ScanStmt}
    {nm₁ nm₂ : String} {k : Nat}
    (h₁ : (buildExtIndex extNames stmts)[nm₁]? = some k)
    (h₂ : (buildExtIndex extNames stmts)[nm₂]? = some k) : nm₁ = nm₂ := by
  unfold buildExtIndex at h₁ h₂
  have hgood := goodExtState_foldl_stmts extNames stmts (state := ({}, 0))
    ⟨fun nm v h => by simp at h, fun nm₁ nm₂ v h => by simp at h⟩
  exact hgood.2 nm₁ nm₂ k h₁ h₂

/-! ## Lemma 7b: buildExtIndex value bound (`< extNames.card`) -/

/-- Invariant: the running counter equals the cardinality of a sub-Finset of `extNames` whose
    members are exactly the map's keys. Gives `cnt ≤ extNames.card` throughout the fold. -/
private def goodCard (extNames : Finset String) (m : Std.HashMap String Nat) (cnt : Nat) : Prop :=
  ∃ S : Finset String, S ⊆ extNames ∧ cnt = S.card ∧ ∀ nm, m.contains nm = true ↔ nm ∈ S

private lemma goodCard_step (extNames : Finset String) (nm : String)
    {state : Std.HashMap String Nat × Nat}
    (hgood : goodCard extNames state.1 state.2) :
    goodCard extNames
      (if decide (nm ∈ extNames) && !state.1.contains nm
       then (state.1.insert nm state.2, state.2 + 1) else state).1
      (if decide (nm ∈ extNames) && !state.1.contains nm
       then (state.1.insert nm state.2, state.2 + 1) else state).2 := by
  obtain ⟨S, hSsub, hcnt, hmem⟩ := hgood
  by_cases h : decide (nm ∈ extNames) && !state.1.contains nm
  · simp only [h, if_true]
    simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at h
    obtain ⟨hne, hnc⟩ := h
    have hnotin : nm ∉ S := fun hc => by
      have := (hmem nm).mpr hc; rw [hnc] at this; exact absurd this (by simp)
    refine ⟨insert nm S, Finset.insert_subset hne hSsub, ?_, ?_⟩
    · rw [hcnt, Finset.card_insert_of_notMem hnotin]
    · intro nm'
      rw [HashMap.contains_insert]
      simp only [Finset.mem_insert]
      constructor
      · intro hc
        rcases (Bool.or_eq_true _ _).mp hc with h1 | h1
        · exact Or.inl ((beq_iff_eq).mp h1).symm
        · exact Or.inr ((hmem nm').mp h1)
      · intro hc
        rcases hc with h1 | h1
        · subst h1; exact (Bool.or_eq_true _ _).mpr (Or.inl (beq_self_eq_true nm'))
        · exact (Bool.or_eq_true _ _).mpr (Or.inr ((hmem nm').mpr h1))
  · simp only [h]
    exact ⟨S, hSsub, hcnt, hmem⟩

private lemma goodCard_foldl_reads (extNames : Finset String) (reads : List String)
    {state : Std.HashMap String Nat × Nat}
    (hgood : goodCard extNames state.1 state.2) :
    goodCard extNames
      (reads.foldl (fun (m, cnt) nm =>
        if decide (nm ∈ extNames) && !m.contains nm then (m.insert nm cnt, cnt + 1)
        else (m, cnt)) state).1
      (reads.foldl (fun (m, cnt) nm =>
        if decide (nm ∈ extNames) && !m.contains nm then (m.insert nm cnt, cnt + 1)
        else (m, cnt)) state).2 := by
  induction reads generalizing state with
  | nil => exact hgood
  | cons nm t ih => simp only [List.foldl_cons]; exact ih (goodCard_step extNames nm hgood)

private lemma goodCard_foldl_stmts (extNames : Finset String) (stmts : List ScanStmt)
    {state : Std.HashMap String Nat × Nat}
    (hgood : goodCard extNames state.1 state.2) :
    goodCard extNames
      (stmts.foldl (fun (acc : Std.HashMap String Nat × Nat) sc =>
        sc.reads.foldl (fun (m, cnt) nm =>
          if decide (nm ∈ extNames) && !m.contains nm then (m.insert nm cnt, cnt + 1)
          else (m, cnt)) acc) state).1
      (stmts.foldl (fun (acc : Std.HashMap String Nat × Nat) sc =>
        sc.reads.foldl (fun (m, cnt) nm =>
          if decide (nm ∈ extNames) && !m.contains nm then (m.insert nm cnt, cnt + 1)
          else (m, cnt)) acc) state).2 := by
  induction stmts generalizing state with
  | nil => exact hgood
  | cons sc t ih =>
      simp only [List.foldl_cons]
      exact ih (goodCard_foldl_reads extNames sc.reads hgood)

/-- Every value assigned by `buildExtIndex` is strictly less than `extNames.card`. -/
theorem buildExtIndex_lt_card {extNames : Finset String} {stmts : List ScanStmt}
    {nm : String} {k : Nat}
    (h : (buildExtIndex extNames stmts)[nm]? = some k) : k < extNames.card := by
  unfold buildExtIndex at h
  -- `k < cnt_final` (goodExtState) and `cnt_final ≤ extNames.card` (goodCard).
  have hgood := goodExtState_foldl_stmts extNames stmts (state := ({}, 0))
    ⟨fun nm v h => by simp at h, fun nm₁ nm₂ v h => by simp at h⟩
  have hlt : k < (stmts.foldl (fun (acc : Std.HashMap String Nat × Nat) sc =>
      sc.reads.foldl (fun (m, cnt) nm =>
        if decide (nm ∈ extNames) && !m.contains nm then (m.insert nm cnt, cnt + 1)
        else (m, cnt)) acc) ({}, 0)).2 := hgood.1 nm k h
  obtain ⟨S, hSsub, hcnt, -⟩ := goodCard_foldl_stmts extNames stmts (state := ({}, 0))
    ⟨∅, Finset.empty_subset _, by simp, fun nm => by simp⟩
  exact _root_.lt_of_lt_of_le hlt (hcnt ▸ Finset.card_le_card hSsub)

/-! ## Public re-exports of the `mapM` per-index facts (used by the agreement layer) -/

/-- Public form: `l.mapM f = .ok r → r.length = l.length`. -/
theorem mapM_ok_length' {ε α β : Type} {f : α → Except ε β} {l : List α} {r : List β}
    (h : l.mapM f = .ok r) : r.length = l.length := mapM_ok_length h

/-- Public form: `l.mapM f = .ok r → f (l.getD i d₁) = .ok (r.getD i d₂)` for `i < l.length`. -/
theorem mapM_ok_getD' {ε α β : Type} [Inhabited β] {f : α → Except ε β} {l : List α} {r : List β}
    (h : l.mapM f = .ok r) (i : Nat) (d₁ : α) (d₂ : β) (hi : i < l.length) :
    f (l.getD i d₁) = .ok (r.getD i d₂) := mapM_ok_getD h i d₁ d₂ hi

end LeanNCD

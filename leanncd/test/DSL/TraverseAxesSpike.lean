-- test/DSL/TraverseAxesSpike.lean
--
-- E1 prototype (IdxExpr slice only): does one `traverseAxes` subsume `IdxExpr.mapUID`
-- (remap), `specsIdx` (collect AxisSpecs), and `idxAxisUIDs` (collect UIDs)?
-- See docs/superpowers/specs/2026-07-15-e1-traverseaxes-prototype-design.md.
import LeanNCD.DSL.Traverse
import LeanNCD.Eval.Contract
import Mathlib.Control.Traversable.Instances

namespace LeanNCD

open LeanNCD.Eval (idxAxisUIDs predAxisUIDs)

/-- Local copy of `Structural.lean`'s private `specsIdx`, for comparison only — NOT the
    source of truth. Keep byte-identical to `Structural.lean:26-27` by inspection. -/
private def specsIdx' : IdxExpr → List AxisSpec
  | .axis a => [a] | .const _ => [] | .scale _ a => [a]
  | .shift a _ => [a] | .affine _ xs => xs.map (·.2)

/-- Minimal local constant-functor: `ConstL α β` is always `α`, ignoring `β`. Used to
    instantiate `traverseAxes` at a "collecting" applicative — `pure` is the empty list,
    `seq` combines via list append (List's monoid), no Mathlib `Monoid`/`Multiplicative`
    wrapper ceremony needed. -/
structure ConstL (α : Type) (β : Type) where run : α

instance {γ : Type} : Functor (ConstL (List γ)) where
  map _ x := ⟨x.run⟩

instance {γ : Type} : Applicative (ConstL (List γ)) where
  pure _ := ⟨[]⟩
  seq f x := ⟨f.run ++ (x ()).run⟩

/-- The E1 prototype: one traversal over `IdxExpr`'s single `AxisSpec` occurrences,
    generic over any `Applicative f`. Instantiated at `Id` it should behave as a remap
    (subsuming `IdxExpr.mapUID`); at a collecting applicative (`ConstL (List α)`) it
    should behave as a collector (subsuming `specsIdx`/`idxAxisUIDs`). -/
def IdxExpr.traverseAxes [Applicative f] (g : AxisSpec → f AxisSpec) : IdxExpr → f IdxExpr
  | .axis a      => IdxExpr.axis <$> g a
  | .const n     => pure (IdxExpr.const n)
  | .scale c a   => IdxExpr.scale c <$> g a
  | .shift a n   => (fun a' => IdxExpr.shift a' n) <$> g a
  | .affine n xs =>
      IdxExpr.affine n <$> Traversable.traverse (fun (ca : Int × AxisSpec) => Prod.mk ca.1 <$> g ca.2) xs

/-- Remap: instantiating `traverseAxes` at `Id` with `g := AxisSpec.mapUID f` should
    reproduce the existing `IdxExpr.mapUID`. -/
theorem traverseAxes_id_eq_mapUID (f : UData → UData) (e : IdxExpr) :
    IdxExpr.traverseAxes (f := Id) (AxisSpec.mapUID f) e = IdxExpr.mapUID f e := by
  cases e with
  | axis a => rfl
  | const n => rfl
  | scale c a => rfl
  | shift a n => rfl
  | affine n xs =>
      have hEq : (fun (ca : Int × AxisSpec) => Prod.mk ca.1 <$> AxisSpec.mapUID f ca.2 :
            Int × AxisSpec → Id (Int × AxisSpec))
          = pure ∘ (fun ca => (ca.1, AxisSpec.mapUID f ca.2)) := rfl
      simp only [IdxExpr.traverseAxes, IdxExpr.mapUID, Traversable.traverse, hEq,
        List.traverse_eq_map_id]
      rfl

/-- Collect `AxisSpec`s: instantiating at `ConstL (List AxisSpec)` with `g := fun a => ⟨[a]⟩`
    should reproduce `specsIdx'` (the local copy of `Structural.lean`'s private `specsIdx`). -/
theorem traverseAxes_const_eq_specsIdx (e : IdxExpr) :
    (IdxExpr.traverseAxes (f := ConstL (List AxisSpec)) (fun a => ⟨[a]⟩) e).run = specsIdx' e := by
  cases e with
  | axis a => rfl
  | const n => rfl
  | scale c a => rfl
  | shift a n => rfl
  | affine n xs =>
      have hmap : (fun (ca : Int × AxisSpec) => Prod.mk ca.1 <$> (⟨[ca.2]⟩ : ConstL (List AxisSpec) AxisSpec))
          = (fun ca => (⟨[ca.2]⟩ : ConstL (List AxisSpec) (Int × AxisSpec))) := rfl
      have core : ∀ ys : List (Int × AxisSpec),
          (Traversable.traverse (fun ca => (⟨[ca.2]⟩ : ConstL (List AxisSpec) (Int × AxisSpec))) ys).run
            = ys.map (·.2) := by
        intro ys
        induction ys with
        | nil => rfl
        | cons hd tl ih =>
            show [hd.2] ++
                (Traversable.traverse (fun ca => (⟨[ca.2]⟩ : ConstL (List AxisSpec) (Int × AxisSpec))) tl).run
              = hd.2 :: List.map (·.2) tl
            rw [ih]
            rfl
      show (IdxExpr.affine n <$>
          Traversable.traverse (fun ca => Prod.mk ca.1 <$> (⟨[ca.2]⟩ : ConstL (List AxisSpec) AxisSpec)) xs :
          ConstL (List AxisSpec) IdxExpr).run = xs.map (·.2)
      rw [hmap]
      exact core xs

/-- Collect UIDs: instantiating at `ConstL (List UID)` with `g := fun a => ⟨[a.uid]⟩` should
    reproduce `idxAxisUIDs` (`Eval/Contract.lean`). -/
theorem traverseAxes_const_eq_idxAxisUIDs (e : IdxExpr) :
    (IdxExpr.traverseAxes (f := ConstL (List UID)) (fun a => ⟨[a.uid]⟩) e).run = idxAxisUIDs e := by
  cases e with
  | axis a => rfl
  | const n => rfl
  | scale c a => rfl
  | shift a n => rfl
  | affine n xs =>
      have hmap : (fun (ca : Int × AxisSpec) => Prod.mk ca.1 <$> (⟨[ca.2.uid]⟩ : ConstL (List UID) AxisSpec))
          = (fun ca => (⟨[ca.2.uid]⟩ : ConstL (List UID) (Int × AxisSpec))) := rfl
      have core : ∀ ys : List (Int × AxisSpec),
          (Traversable.traverse (fun ca => (⟨[ca.2.uid]⟩ : ConstL (List UID) (Int × AxisSpec))) ys).run
            = ys.map (·.2.uid) := by
        intro ys
        induction ys with
        | nil => rfl
        | cons hd tl ih =>
            show [hd.2.uid] ++
                (Traversable.traverse (fun ca => (⟨[ca.2.uid]⟩ : ConstL (List UID) (Int × AxisSpec))) tl).run
              = hd.2.uid :: List.map (·.2.uid) tl
            rw [ih]
            rfl
      show (IdxExpr.affine n <$>
          Traversable.traverse (fun ca => Prod.mk ca.1 <$> (⟨[ca.2.uid]⟩ : ConstL (List UID) AxisSpec)) xs :
          ConstL (List UID) IdxExpr).run = xs.map (·.2.uid)
      rw [hmap]
      exact core xs

/-- Local copy of `Structural.lean`'s private `specsPred`, for comparison only — NOT the
    source of truth. Keep byte-identical to `Structural.lean:30-31` by inspection. -/
private def specsPred' : PredArith → List AxisSpec
  | .embed e => specsIdx' e | .mul a b => specsPred' a ++ specsPred' b | .iabs a => specsPred' a

/-- Extends the E1 prototype to `PredArith`: genuine self-recursion (`.mul`/`.iabs`) and
    composition with the already-proven `IdxExpr.traverseAxes` (via `.embed`). -/
def PredArith.traverseAxes [Applicative f] (g : AxisSpec → f AxisSpec) : PredArith → f PredArith
  | .embed e => PredArith.embed <$> IdxExpr.traverseAxes g e
  | .mul a b => PredArith.mul <$> PredArith.traverseAxes g a <*> PredArith.traverseAxes g b
  | .iabs a  => PredArith.iabs <$> PredArith.traverseAxes g a

/- Remap: instantiating `traverseAxes` at `Id` with `g := AxisSpec.mapUID f` should
    reproduce the existing `PredArith.mapUID`.

    NOT PROVEN — commented out rather than left with a `sorry`. `PredArith.mapUID` is declared
    `partial def` in `Traverse.lean:22` (it genuinely self-recurses via `.mul`/`.iabs`, unlike
    `IdxExpr.mapUID`, which is also `partial` but never calls itself). Lean generates no
    equation lemmas for a `partial def`, and `unfold`/`delta`/`rfl` all fail to make any
    progress on `PredArith.mapUID f (PredArith.embed e) = PredArith.embed (IdxExpr.mapUID f e)`
    — confirmed even for this non-recursive `.embed` case, so the blocker is not proof search
    difficulty but the outright absence of any unfolding principle for `PredArith.mapUID` in
    the proof layer. Closing this would require changing `PredArith.mapUID`'s definition
    (e.g. dropping `partial` in favor of ordinary structural recursion, which the function's
    shape appears to support) — out of scope for this spike.
theorem traverseAxes_id_eq_predMapUID (f : UData → UData) (e : PredArith) :
    PredArith.traverseAxes (f := Id) (AxisSpec.mapUID f) e = PredArith.mapUID f e := by
  induction e with
  | embed e => exact traverseAxes_id_eq_mapUID f e
  | mul a b iha ihb => simp [PredArith.traverseAxes, PredArith.mapUID, iha, ihb]
  | iabs a iha => simp [PredArith.traverseAxes, PredArith.mapUID, iha]
-/

/-- Collect `AxisSpec`s: instantiating at `ConstL (List AxisSpec)` with `g := fun a => ⟨[a]⟩`
    should reproduce `specsPred'` (the local copy of `Structural.lean`'s private `specsPred`). -/
theorem traverseAxes_const_eq_specsPred (e : PredArith) :
    (PredArith.traverseAxes (f := ConstL (List AxisSpec)) (fun a => ⟨[a]⟩) e).run = specsPred' e := by
  induction e with
  | embed e => exact traverseAxes_const_eq_specsIdx e
  | mul a b iha ihb =>
      show ((PredArith.traverseAxes (f := ConstL (List AxisSpec)) (fun a => ⟨[a]⟩) a).run ++
            (PredArith.traverseAxes (f := ConstL (List AxisSpec)) (fun a => ⟨[a]⟩) b).run
              : List AxisSpec) = specsPred' a ++ specsPred' b
      rw [iha, ihb]
  | iabs a iha =>
      show (PredArith.traverseAxes (f := ConstL (List AxisSpec)) (fun a => ⟨[a]⟩) a).run = specsPred' a
      exact iha

/-- Collect UIDs: instantiating at `ConstL (List UID)` with `g := fun a => ⟨[a.uid]⟩` should
    reproduce `predAxisUIDs` (`Eval/Contract.lean`). -/
theorem traverseAxes_const_eq_predAxisUIDs (e : PredArith) :
    (PredArith.traverseAxes (f := ConstL (List UID)) (fun a => ⟨[a.uid]⟩) e).run = predAxisUIDs e := by
  induction e with
  | embed e => exact traverseAxes_const_eq_idxAxisUIDs e
  | mul a b iha ihb =>
      show ((PredArith.traverseAxes (f := ConstL (List UID)) (fun a => ⟨[a.uid]⟩) a).run ++
            (PredArith.traverseAxes (f := ConstL (List UID)) (fun a => ⟨[a.uid]⟩) b).run
              : List UID) = predAxisUIDs a ++ predAxisUIDs b
      rw [iha, ihb]
  | iabs a iha =>
      show (PredArith.traverseAxes (f := ConstL (List UID)) (fun a => ⟨[a.uid]⟩) a).run = predAxisUIDs a
      exact iha

end LeanNCD

import Mathlib
import LeanNCD.Core.Graded
import LeanNCD.Core.Weave
import LeanNCD.Mixins.Temporal
import LeanNCD.Grothendieck.Split

namespace LeanNCD
open CategoryTheory

section Core
variable {D C : Type} [ColoredPROP D] [ColoredPROP C] [DGradedColoredPROP D C]

/-- Prop 8.1 (functoriality half): the lift respects composition (`act.map_comp`). The
    product-category composite `(f, 𝟙 P) ≫ (g, 𝟙 P)` equals `(f ≫ g, 𝟙 P)`, so the lift of a
    composite is the composite of the lifts. Proved sorry-free. -/
theorem lift_functorial {X Y Z : C} (f : SmallCategory.hom X Y) (g : SmallCategory.hom Y Z)
    (P : Dᵒᵖ) :
    DGradedColoredPROP.act.map (X := (X, P)) (Y := (Z, P)) (SmallCategory.comp f g, 𝟙 P)
      = SmallCategory.comp
          (DGradedColoredPROP.act.map (X := (X, P)) (Y := (Y, P)) (f, 𝟙 P))
          (DGradedColoredPROP.act.map (X := (Y, P)) (Y := (Z, P)) (g, 𝟙 P)) := by
  show DGradedColoredPROP.act.map (X := (X, P)) (Y := (Z, P)) (SmallCategory.comp f g, 𝟙 P)
      = DGradedColoredPROP.act.map (X := (X, P)) (Y := (Y, P)) (f, 𝟙 P)
          ≫ DGradedColoredPROP.act.map (X := (Y, P)) (Y := (Z, P)) (g, 𝟙 P)
  rw [← Functor.map_comp]
  congr 1
  ext
  · rfl
  · simp

/-- Prop 8.2: weave uniqueness (re-export of §5). Needs `[Elemental C]` — elementality is the opt-in
    mixin (not a base `ColoredPROP` field); for `C = Br` it reduces to `brCancelPoint`. -/
theorem weave_subsingleton [Elemental C] {X Y : C} (g : SmallCategory.hom X Y) :
    Subsingleton (Weave (D := D) g) :=
  weave_unique (D := D) g

end Core

section Temporal
variable {D C : Type} [ColoredPROP D] [ColoredPROP C] [TemporalGraded D C]

/-- Prop 8.8: Scan batches along `P` orthogonal to `L` — this IS `lift_fold_dist`: an orthogonal
    degree `P` distributes through the fold over the temporal object `L`. Genuine re-export. -/
def scan_batches (N : ℕ) (X : C) (P : Dᵒᵖ) :
    DGradedColoredPROP.act.obj
        (DGradedColoredPROP.act.obj (X, Opposite.op (TemporalGraded.L (D := D) (C := C))), P)
      ≅ DGradedColoredPROP.act.obj (X, Opposite.op (TemporalGraded.L (D := D) (C := C))) :=
  TemporalGraded.lift_fold_dist N X P

/-- Prop 8.7 (Scan-as-catamorphism): the `N`-fold iterate is stable under restriction along the
    reflexive prefix map `[0..N] ↪ [0..N]` (`iotaTo (le_refl N) = 𝟙`); i.e. restricting the iterate
    to its own top prefix is the identity post-composition. A genuine equation over `iterate`
    and `restrict`. Proved sorry-free. -/
theorem scan_catamorphism (N : ℕ) (X : C)
    (step : SmallCategory.hom X X) :
    SmallCategory.comp (TemporalGraded.iterate (D := D) N X step)
        (TemporalGraded.restrict (D := D) (le_refl N) X)
      = TemporalGraded.iterate (D := D) N X step := by
  -- `restrict` along the reflexive prefix map is the identity (`restrict_id`), so the
  -- post-composition collapses by `comp_id`.
  rw [TemporalGraded.restrict_id]
  exact SmallCategory.comp_id _

end Temporal

-- §8.3: grothendieck_split — already stated in `LeanNCD/Grothendieck/Split.lean`; not restated here.
-- §8.4: equivariance — `SymmetryGraded` is a stub class extending `DGradedColoredPROP` with no
--        genuine equivariance field/equation yet (EM-machinery gated). OMITTED — nothing faithful
--        to state over the class fields without faking a `True`.
-- §8.5: deferred — needs Inst(C♯)/pushout (Milestone D/G).
-- §8.6: obstruction species — no class field carries the obstruction datum yet. OMITTED.

end LeanNCD

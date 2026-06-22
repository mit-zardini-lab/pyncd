import Mathlib
import LeanNCD.Core.Graded

namespace LeanNCD

open CategoryTheory

/-- A weave: a witness of the (Broadcast-gen) factorization `g = lam ; [f, P] ; ρ` for a morphism
    `g`, with `f` the degree-trivial base op, `P` the degree, `lam`/`ρ` the boundary reindexings.
    This is the cartesian-lift datum of the grading fibration `C → D` (graded_prop.md §3.3). -/
structure Weave {D C : Type} [ColoredPROP D] [ColoredPROP C] [DGradedColoredPROP D C]
    {X Y : C} (g : SmallCategory.hom X Y) where
  X' : C
  Y' : C
  f  : SmallCategory.hom X' Y'
  P  : Dᵒᵖ
  lam : SmallCategory.hom X (DGradedColoredPROP.act.obj (X', P))
  ρ  : SmallCategory.hom (DGradedColoredPROP.act.obj (Y', P)) Y
  factors :
    g = SmallCategory.comp
          (SmallCategory.comp lam
            (DGradedColoredPROP.act.map (f, 𝟙 P)))
          ρ

/-- Prop 8.2: the weave is a datum, not a choice — at most one, up to the canonical coherence isos.
    Proof draws on `Elemental.elemental` (points separate morphisms) + `broadcast_gen` (existence).
    Takes `[Elemental C]` as a hypothesis: elementality is an opt-in mixin (not a `ColoredPROP`
    field) — see `Base/ColoredPROP.lean`. For `C = Br` this instance reduces to `brCancelPoint`. -/
theorem weave_unique {D C : Type} [ColoredPROP D] [ColoredPROP C] [Elemental C]
    [DGradedColoredPROP D C]
    {X Y : C} (g : SmallCategory.hom X Y) : Subsingleton (Weave (D := D) g) := by
  sorry  -- SIGNATURE (proof milestone): from Elemental.elemental + broadcast_gen.

end LeanNCD

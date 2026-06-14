import Mathlib
import LeanNCD.Algebra.Target
import LeanNCD.Core.Graded

namespace LeanNCD
open CategoryTheory

/-- The algebra functor `F : C → V` — a strong symmetric monoidal, `D`-equivariant functor into a
    target actegory (graded_prop.md Def 7.2); the categorical content of `construct()`. A `class`
    (not `structure`) so `ParaAlgebra` can `extend` it. -/
class Algebra (D C : Type) (V : Type*) [ColoredPROP D] [ColoredPROP C] [Category V]
    (R : Type) [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R] where
  F       : C ⥤ V
  equivar : ∀ (X : C) (P : Dᵒᵖ),
              F.obj (DGradedColoredPROP.act.obj (X, P))
                ≅ (TargetActegory.actV (D := D) (V := V) (R := R)).obj (F.obj X, P)

/-- The `Para` refinement: `Para(C) → Para(V)` 2-functor, weight tying as passes-as-2-cells. STUB. -/
class ParaAlgebra (D C : Type) (V : Type*) [ColoredPROP D] [ColoredPROP C] [Category V]
    (R : Type) [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R]
    extends Algebra D C V R

end LeanNCD

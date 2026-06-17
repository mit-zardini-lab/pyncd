import Mathlib
import LeanNCD.Algebra.Target
import LeanNCD.Core.Graded

namespace LeanNCD
open CategoryTheory

/-- The algebra functor `F : C → V` — a strong symmetric monoidal, `D`-equivariant functor into a
    target actegory (graded_prop.md Def 7.2); the categorical content of `construct()`. A `class`
    (not `structure`) so `ParaAlgebra` can `extend` it. -/
class Algebra (D C : Type) (V : Type*) [ColoredPROP D] [ColoredPROP C] [Category V] [MonoidalCategory V]
    [SymmetricCategory V]
    (R : Type) [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R] where
  F        : C ⥤ V
  /-- `F` is **strong symmetric monoidal** (graded_prop.md Def 7.2). `Functor.Braided F` bundles the
      strong-monoidal structure (`F.Monoidal`: μ/ε with invertible structure maps + pentagon/unitor
      coherences) together with `F.LaxBraided`'s braiding-compatibility law
      `μ X Y ≫ F.map (β_ X Y) = β_ (F X) (F Y) ≫ μ Y X`. `C` is symmetric monoidal via the Seam
      adapter (`instSymmetricOfColoredPROP`); `V` is symmetric monoidal by the class binder. -/
  Fbraided : F.Braided
  equivar  : ∀ (X : C) (P : Dᵒᵖ),
              F.obj (DGradedColoredPROP.act.obj (X, P))
                ≅ (TargetActegory.actV (D := D) (V := V) (R := R)).obj (F.obj X, P)

/-- The `Para` refinement: `Para(C) → Para(V)` 2-functor, weight tying as passes-as-2-cells. STUB. -/
class ParaAlgebra (D C : Type) (V : Type*) [ColoredPROP D] [ColoredPROP C] [Category V] [MonoidalCategory V]
    [SymmetricCategory V]
    (R : Type) [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R]
    extends Algebra D C V R

end LeanNCD

import Mathlib
import LeanNCD.Seam.Adapter

namespace LeanNCD

open CategoryTheory

/-- Extend the shape map `sh : gen_C → D` to a monoid homomorphism `sh* : C → D` on objects,
    by folding `ColoredPROP.tensor` over the color list of `X`. Satisfies (Sh-⊗):
    `sh*(X ⊗ Y) = sh*(X) ⊗ sh*(Y)` (not needed yet). -/
def sh_star {C D : Type} [ColoredPROP D] [ColoredPROP C]
    (sh : ColoredPROP.gen (ob := C) → D) (X : C) : D :=
  ((ColoredPROP.toList X).map sh).foldr ColoredPROP.tensor ColoredPROP.unit

end LeanNCD

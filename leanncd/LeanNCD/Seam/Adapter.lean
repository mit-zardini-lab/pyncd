import Mathlib
import LeanNCD.Base.ColoredPROP

namespace LeanNCD

open CategoryTheory

/-- The seam: every `ColoredPROP` (hence every `SmallCategory`) is a Mathlib `Category`.
    The sorry-free half of the §3 adapter — forwards the `SmallCategory` data and laws; no
    strictification needed for the bare category. -/
instance instCategoryOfColoredPROP {O : Type} [ColoredPROP O] : Category O where
  Hom X Y := SmallCategory.hom X Y
  id X := SmallCategory.id X
  comp f g := SmallCategory.comp f g
  id_comp := SmallCategory.id_comp
  comp_id := SmallCategory.comp_id
  assoc := SmallCategory.assoc

end LeanNCD

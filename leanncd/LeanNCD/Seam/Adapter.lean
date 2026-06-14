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

/-- The §11 strictification seam: the strict monoidal structure on a `ColoredPROP`.
    Data are `ColoredPROP.tensor`/`tensorHom`/`swap`; the structural isos are `eqToIso` of
    the (strict) `tensor_*` laws proved in Milestone A, so the category is genuinely strict.
    Built directly on `instCategoryOfColoredPROP` (no `Category` diamond). The coherence/
    naturality fields require functoriality laws of `tensorHom` (e.g. `tensorHom (𝟙) (𝟙) = 𝟙`,
    interchange) that the `ColoredPROP` class does not (yet) carry, so they are deferred as
    `-- SIGNATURE` `sorry`s — this is the stated-with-sorry milestone outcome. -/
noncomputable instance instMonoidalOfColoredPROP {O : Type} [ColoredPROP O] :
    MonoidalCategory O where
  tensorObj X Y := ColoredPROP.tensor X Y
  tensorUnit := (ColoredPROP.unit : O)
  whiskerLeft X _ _ f := ColoredPROP.tensorHom (𝟙 X) f
  whiskerRight f Y := ColoredPROP.tensorHom f (𝟙 Y)
  tensorHom f g := ColoredPROP.tensorHom f g
  associator X Y Z := eqToIso (ColoredPROP.tensor_assoc X Y Z)
  leftUnitor X := eqToIso (ColoredPROP.tensor_unit_l X)
  rightUnitor X := eqToIso (ColoredPROP.tensor_unit_r X)
  tensorHom_def := by sorry -- SIGNATURE (§11 strictification)
  id_tensorHom_id := by sorry -- SIGNATURE (§11 strictification)
  tensorHom_comp_tensorHom := by sorry -- SIGNATURE (§11 strictification)
  whiskerLeft_id := by sorry -- SIGNATURE (§11 strictification)
  id_whiskerRight := by sorry -- SIGNATURE (§11 strictification)
  associator_naturality := by sorry -- SIGNATURE (§11 strictification)
  leftUnitor_naturality := by sorry -- SIGNATURE (§11 strictification)
  rightUnitor_naturality := by sorry -- SIGNATURE (§11 strictification)
  pentagon := by sorry -- SIGNATURE (§11 strictification)
  triangle := by sorry -- SIGNATURE (§11 strictification)

/-- The §11 strictification seam: the symmetric structure, with braiding `ColoredPROP.swap`.
    The braiding self-inverse (`hom_inv_id`/`inv_hom_id`) needs `swap X Y ≫ swap Y X = 𝟙`,
    which the `ColoredPROP` class does not carry, so those two are deferred as `sorry`;
    the braided naturality/hexagon/symmetry fields are discharged by `aesop_cat`. -/
noncomputable instance instSymmetricOfColoredPROP {O : Type} [ColoredPROP O] :
    SymmetricCategory O where
  braiding X Y :=
    { hom := ColoredPROP.swap X Y
      inv := ColoredPROP.swap Y X
      hom_inv_id := by sorry -- SIGNATURE (§11 strictification)
      inv_hom_id := by sorry -- SIGNATURE (§11 strictification)
    }
  braiding_naturality_right := by aesop_cat
  braiding_naturality_left := by aesop_cat
  hexagon_forward := by aesop_cat
  hexagon_reverse := by aesop_cat
  symmetry := by aesop_cat

end LeanNCD

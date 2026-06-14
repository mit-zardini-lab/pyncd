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

/-- Whiskering an `eqToHom` on the right is itself an `eqToHom` — `tensorHom` is functorial on
    objects (uses `ColoredPROP.tensorHom_id`). -/
private theorem tensorHom_eqToHom_id {O : Type} [ColoredPROP O] {a b : O} (c : O) (h : a = b) :
    ColoredPROP.tensorHom (eqToHom h) (𝟙 c)
      = eqToHom (show ColoredPROP.tensor a c = ColoredPROP.tensor b c by rw [h]) := by
  subst h; simp only [eqToHom_refl]; exact ColoredPROP.tensorHom_id a c

/-- Whiskering an `eqToHom` on the left is itself an `eqToHom`. -/
private theorem tensorHom_id_eqToHom {O : Type} [ColoredPROP O] (a : O) {c d : O} (h : c = d) :
    ColoredPROP.tensorHom (𝟙 a) (eqToHom h)
      = eqToHom (show ColoredPROP.tensor a c = ColoredPROP.tensor a d by rw [h]) := by
  subst h; simp only [eqToHom_refl]; exact ColoredPROP.tensorHom_id a c

/-- The §11 strictification seam: the strict monoidal structure on a `ColoredPROP`.
    Data are `ColoredPROP.tensor`/`tensorHom`/`swap`; the structural isos are `eqToIso` of
    the (strict) `tensor_*` laws proved in Milestone A, so the category is genuinely strict.
    Built directly on `instCategoryOfColoredPROP` (no `Category` diamond). The bifunctor
    coherences (`tensorHom_def`, `id_tensorHom_id`, `tensorHom_comp_tensorHom`, `whiskerLeft_id`,
    `id_whiskerRight`) follow from `ColoredPROP.tensorHom_id`/`tensorHom_comp`; the structural-iso
    coherences `pentagon`/`triangle` follow from functoriality-on-objects of `tensorHom`. The three
    naturality squares (`associator_naturality`, `leftUnitor_naturality`, `rightUnitor_naturality`)
    relate `tensorHom` of genuine morphisms across the object (re)association/unit equalities — an
    associativity/unit coherence of `tensorHom` itself, which is NOT implied by the bifunctor +
    symmetry laws — so they remain deferred as `-- SIGNATURE` `sorry`s. -/
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
  tensorHom_def := by
    intro X₁ Y₁ X₂ Y₂ f g
    show ColoredPROP.tensorHom f g
        = SmallCategory.comp (ColoredPROP.tensorHom f (𝟙 X₂)) (ColoredPROP.tensorHom (𝟙 Y₁) g)
    rw [← ColoredPROP.tensorHom_comp]
    show ColoredPROP.tensorHom f g
        = ColoredPROP.tensorHom (SmallCategory.comp f (SmallCategory.id Y₁))
                                (SmallCategory.comp (SmallCategory.id X₂) g)
    rw [SmallCategory.comp_id, SmallCategory.id_comp]
  id_tensorHom_id := by intro X Y; exact ColoredPROP.tensorHom_id X Y
  tensorHom_comp_tensorHom := by
    intro X₁ Y₁ Z₁ X₂ Y₂ Z₂ f₁ f₂ g₁ g₂
    exact (ColoredPROP.tensorHom_comp f₁ g₁ f₂ g₂).symm
  whiskerLeft_id := by intro X Y; exact ColoredPROP.tensorHom_id X Y
  id_whiskerRight := by intro X Y; exact ColoredPROP.tensorHom_id X Y
  associator_naturality := by
    sorry -- SIGNATURE (§11): tensorHom-associativity coherence, not implied by the bifunctor/symmetry laws
  leftUnitor_naturality := by
    sorry -- SIGNATURE (§11): tensorHom left-unit coherence, not implied by the bifunctor/symmetry laws
  rightUnitor_naturality := by
    sorry -- SIGNATURE (§11): tensorHom right-unit coherence, not implied by the bifunctor/symmetry laws
  pentagon := by
    intro W X Y Z
    simp only [eqToIso.hom, tensorHom_eqToHom_id, tensorHom_id_eqToHom, eqToHom_trans]
  triangle := by
    intro X Y
    simp only [eqToIso.hom, tensorHom_eqToHom_id, tensorHom_id_eqToHom, eqToHom_trans]

/-- The §11 strictification seam: the symmetric structure, with braiding `ColoredPROP.swap`.
    The braiding self-inverse (`hom_inv_id`/`inv_hom_id`) and `symmetry` are exactly
    `ColoredPROP.swap_swap`. The remaining fields (`braiding_naturality_left`/`_right`,
    `hexagon_forward`/`_reverse`) are naturality of `swap` and the hexagon identities — independent
    symmetric-monoidal coherences of `swap` that the `ColoredPROP` class does not carry (and that
    `swap_swap` alone does not imply), so they remain `-- SIGNATURE` `sorry`s.
    (Previously these "closed by `aesop_cat`" only because the then-`sorry` braiding inverse let the
    discharger ride on `sorry`; with the inverse now proved real they are honestly deferred.) -/
noncomputable instance instSymmetricOfColoredPROP {O : Type} [ColoredPROP O] :
    SymmetricCategory O where
  braiding X Y :=
    { hom := ColoredPROP.swap X Y
      inv := ColoredPROP.swap Y X
      hom_inv_id := ColoredPROP.swap_swap X Y
      inv_hom_id := ColoredPROP.swap_swap Y X
    }
  braiding_naturality_right := by
    sorry -- SIGNATURE (§11): naturality of swap, not implied by swap_swap
  braiding_naturality_left := by
    sorry -- SIGNATURE (§11): naturality of swap, not implied by swap_swap
  hexagon_forward := by
    sorry -- SIGNATURE (§11): hexagon identity for swap, not implied by swap_swap
  hexagon_reverse := by
    sorry -- SIGNATURE (§11): hexagon identity for swap, not implied by swap_swap
  symmetry := by intro X Y; exact ColoredPROP.swap_swap X Y

end LeanNCD

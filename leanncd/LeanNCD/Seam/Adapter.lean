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

/-- Naturality across `eqToHom` reduces to a heterogeneous equality: if `A ≍ B` and their
    domains/codomains agree by the object equalities, the `eqToHom`-conjugated square commutes.
    Used to discharge the strict `MonoidalCategory` associator/unitor naturalities from the
    `ColoredPROP` `tensorHom_assoc`/`tensorHom_unit_*` `HEq` fields. -/
private theorem natOfHEq {C : Type*} [Category C] {X Y X' Y' : C}
    (A : X ⟶ Y) (B : X' ⟶ Y') (hX : X = X') (hY : Y = Y') (hAB : HEq A B) :
    A ≫ eqToHom hY = eqToHom hX ≫ B := by
  subst hX hY
  rw [eqToHom_refl, eqToHom_refl, Category.comp_id, Category.id_comp, eq_of_heq hAB]

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
    intro X₁ X₂ X₃ Y₁ Y₂ Y₃ f₁ f₂ f₃
    exact natOfHEq _ _ (ColoredPROP.tensor_assoc X₁ X₂ X₃) (ColoredPROP.tensor_assoc Y₁ Y₂ Y₃)
      (ColoredPROP.tensorHom_assoc f₁ f₂ f₃)
  leftUnitor_naturality := by
    intro X Y f
    exact natOfHEq _ _ (ColoredPROP.tensor_unit_l X) (ColoredPROP.tensor_unit_l Y)
      (ColoredPROP.tensorHom_unit_l f)
  rightUnitor_naturality := by
    intro X Y f
    exact natOfHEq _ _ (ColoredPROP.tensor_unit_r X) (ColoredPROP.tensor_unit_r Y)
      (ColoredPROP.tensorHom_unit_r f)
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
    intro X Y Z f
    -- X ◁ f ≫ (β X Z).hom = (β X Y).hom ≫ f ▷ X  reduces to swap_natural (𝟙 X) f
    exact ColoredPROP.swap_natural (𝟙 X) f
  braiding_naturality_left := by
    intro X Y f Z
    -- f ▷ Z ≫ (β Y Z).hom = (β X Z).hom ≫ Z ◁ f  reduces to swap_natural f (𝟙 Z)
    exact ColoredPROP.swap_natural f (𝟙 Z)
  hexagon_forward := by
    intro X Y Z
    have ha : ∀ A B C : O, (MonoidalCategoryStruct.associator A B C) =
        eqToIso (ColoredPROP.tensor_assoc A B C) := fun A B C => rfl
    have hb : ∀ A B C : O, (MonoidalCategoryStruct.associator A B C).hom =
        eqToHom (ColoredPROP.tensor_assoc A B C) := fun A B C => (by rw [ha, eqToIso.hom])
    have hc : ∀ {A B : O} (h : A = B), eqToHom h = SmCat.coh h := fun h => (by cases h; rfl)
    have he : ∀ {A B : O} (f : SmallCategory.hom A B) (C : O),
        MonoidalCategoryStruct.whiskerRight f C = ColoredPROP.tensorHom f (SmallCategory.id C) :=
      by intros; rfl
    have hf : ∀ (A : O) {B C : O} (f : SmallCategory.hom B C),
        MonoidalCategoryStruct.whiskerLeft A f = ColoredPROP.tensorHom (SmallCategory.id A) f :=
      by intros; rfl
    have hg : ∀ {A B C : O} (f : A ⟶ B) (g : B ⟶ C),
        f ≫ g = SmallCategory.comp f g := by intros; rfl
    simp only [hb, hc, he, hf, hg]
    rw [← SmallCategory.assoc, ← SmallCategory.assoc]
    exact ColoredPROP.swap_hexagon_fwd X Y Z
  hexagon_reverse := by
    intro X Y Z
    have ha : ∀ A B C : O, (MonoidalCategoryStruct.associator A B C) =
        eqToIso (ColoredPROP.tensor_assoc A B C) := fun A B C => rfl
    have ha' : ∀ A B C : O, (MonoidalCategoryStruct.associator A B C).inv =
        eqToHom (ColoredPROP.tensor_assoc A B C).symm := fun A B C => (by rw [ha, eqToIso.inv])
    have hc : ∀ {A B : O} (h : A = B), eqToHom h = SmCat.coh h := fun h => (by cases h; rfl)
    have he : ∀ {A B : O} (f : SmallCategory.hom A B) (C : O),
        MonoidalCategoryStruct.whiskerRight f C = ColoredPROP.tensorHom f (SmallCategory.id C) :=
      by intros; rfl
    have hf : ∀ (A : O) {B C : O} (f : SmallCategory.hom B C),
        MonoidalCategoryStruct.whiskerLeft A f = ColoredPROP.tensorHom (SmallCategory.id A) f :=
      by intros; rfl
    have hg : ∀ {A B C : O} (f : A ⟶ B) (g : B ⟶ C),
        f ≫ g = SmallCategory.comp f g := by intros; rfl
    simp only [ha', hc, he, hf, hg]
    rw [← SmallCategory.assoc, ← SmallCategory.assoc]
    exact ColoredPROP.swap_hexagon_rev X Y Z
  symmetry := by intro X Y; exact ColoredPROP.swap_swap X Y

end LeanNCD

import Mathlib
import LeanNCD.Base.Category

namespace LeanNCD

/-- Transport `id a` along an object equality `h : a = b`, yielding a morphism `a ⟶ b`.
    The `SmallCategory`-level analog of Mathlib's `eqToHom`; used to state the hexagon
    coherence of `swap` with the strict associator. -/
def SmCat.coh {ob : Type} [SmallCategory ob] {a b : ob} (h : a = b) :
    SmallCategory.hom a b :=
  cast (congrArg (SmallCategory.hom a) h) (SmallCategory.id a)

/-- A colored PROP: a symmetric strict monoidal category whose object monoid is the free monoid
    `O*` over a color set `gen`, with `⊗` = list concatenation and `I` = the empty list.
    `elemental` (the (Elem) axiom) states that points separate parallel morphisms. -/
class ColoredPROP (ob : Type) extends SmallCategory ob where
  gen       : Type
  toList    : ob → List gen
  ofList    : List gen → ob
  tensor    : ob → ob → ob := fun a b => ofList (toList a ++ toList b)
  unit      : ob := ofList []
  tensor_assoc  : ∀ a b c, tensor (tensor a b) c = tensor a (tensor b c)
  tensor_unit_l : ∀ a, tensor unit a = a
  tensor_unit_r : ∀ a, tensor a unit = a
  swap      : ∀ a b, hom (tensor a b) (tensor b a)
  tensorHom : ∀ {a b c d}, hom a b → hom c d → hom (tensor a c) (tensor b d)
  -- tensorHom is a bifunctor; swap is a symmetry (the morphism-level symmetric-monoidal laws).
  tensorHom_id   : ∀ (a c : ob), tensorHom (id a) (id c) = id (tensor a c)
  tensorHom_comp : ∀ {a b c d e g : ob}
                     (f₁ : hom a b) (f₂ : hom b c) (g₁ : hom d e) (g₂ : hom e g),
                     tensorHom (comp f₁ f₂) (comp g₁ g₂) = comp (tensorHom f₁ g₁) (tensorHom f₂ g₂)
  swap_swap      : ∀ (a b : ob), comp (swap a b) (swap b a) = id (tensor a b)
  -- swap is natural: sliding a tensor of morphisms past the symmetry.
  swap_natural : ∀ {a b c d : ob} (f : hom a b) (g : hom c d),
                   comp (tensorHom f g) (swap b d) = comp (swap a c) (tensorHom g f)
  -- hexagon identities: coherence of swap with the strict associator (SMC coherence axioms).
  swap_hexagon_fwd : ∀ (X Y Z : ob),
      comp (comp (SmCat.coh (tensor_assoc X Y Z)) (swap X (tensor Y Z)))
           (SmCat.coh (tensor_assoc Y Z X)) =
      comp (comp (tensorHom (swap X Y) (id Z)) (SmCat.coh (tensor_assoc Y X Z)))
           (tensorHom (id Y) (swap X Z))
  swap_hexagon_rev : ∀ (X Y Z : ob),
      comp (comp (SmCat.coh (tensor_assoc X Y Z).symm) (swap (tensor X Y) Z))
           (SmCat.coh (tensor_assoc Z X Y).symm) =
      comp (comp (tensorHom (id X) (swap Y Z)) (SmCat.coh (tensor_assoc X Z Y).symm))
           (tensorHom (swap X Z) (id Y))
  -- tensorHom respects the strict object-monoid laws up to `HEq` across the object equalities
  -- `tensor_assoc`/`tensor_unit_*`. These supply the `MonoidalCategory` naturality coherences in the
  -- seam (`Seam/Adapter.lean`), via `conj_eqToHom_iff_heq` (naturality across `eqToHom` ⟺ `HEq`).
  tensorHom_assoc : ∀ {a₁ b₁ a₂ b₂ a₃ b₃ : ob}
                      (f : hom a₁ b₁) (g : hom a₂ b₂) (h : hom a₃ b₃),
                      HEq (tensorHom (tensorHom f g) h) (tensorHom f (tensorHom g h))
  tensorHom_unit_l : ∀ {a b : ob} (f : hom a b), HEq (tensorHom (id unit) f) f
  tensorHom_unit_r : ∀ {a b : ob} (f : hom a b), HEq (tensorHom f (id unit)) f
  elemental : ∀ {X Y} (f g : hom X Y),
                (∀ x : hom unit X, comp x f = comp x g) → f = g

-- TEST: the class and its key field resolve at the expected types.
#check (ColoredPROP : Type → Type 1)
#check @ColoredPROP.elemental

end LeanNCD

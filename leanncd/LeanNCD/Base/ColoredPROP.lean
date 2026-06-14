import Mathlib
import LeanNCD.Base.Category

namespace LeanNCD

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
  elemental : ∀ {X Y} (f g : hom X Y),
                (∀ x : hom unit X, comp x f = comp x g) → f = g

-- TEST: the class and its key field resolve at the expected types.
#check (ColoredPROP : Type → Type 1)
#check @ColoredPROP.elemental

end LeanNCD

namespace LeanNCD

/-- The categorical skeleton: a Lean 4 typeclass over an object type `ob`.
    Following Holtzen (2025); carried by `ColoredPROP` and everything above. -/
class SmallCategory (ob : Type) : Type 1 where
  hom     : ob → ob → Type
  id      : ∀ x, hom x x
  comp    : ∀ {X Y Z}, hom X Y → hom Y Z → hom X Z
  id_comp : ∀ {X Y} (f : hom X Y), comp (id X) f = f
  comp_id : ∀ {X Y} (f : hom X Y), comp f (id Y) = f
  assoc   : ∀ {W X Y Z} (f : hom W X) (g : hom X Y) (h : hom Y Z),
              comp (comp f g) h = comp f (comp g h)

-- Scoped to avoid conflict with CategoryTheory's ⟶ and Function.comp (Milestone B imports Mathlib).
-- `scoped` inside the `ColoredPROPNotations` namespace; activate with `open scoped ColoredPROPNotations`.
namespace ColoredPROPNotations
scoped infixl:65 " ⟶ " => SmallCategory.hom
scoped notation:65 a " ∘ " b => SmallCategory.comp b a
end ColoredPROPNotations

-- TEST (written first): the one-object category on Unit must satisfy SmallCategory.
-- Definitional eta for the Unit structure makes every law `rfl`.
example : SmallCategory Unit where
  hom _ _ := Unit
  id _ := ()
  comp _ _ := ()
  id_comp := by intros; rfl
  comp_id := by intros; rfl
  assoc := by intros; rfl

end LeanNCD

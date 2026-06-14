import Mathlib
import LeanNCD.Base.St

namespace LeanNCD

inductive DType
  | reals
  | nat : Numeric → DType

structure ArrayType where
  dtype : DType
  shape : StObj                     -- shape lives in Ob(St)

abbrev BrObj := List ArrayType      -- a product of arrays

inductive WeaveSlot
  | fixed : Axis → WeaveSlot        -- retained axis
  | tiled : WeaveSlot               -- contracted axis

abbrev WeaveShape := List WeaveSlot

def WeaveShape.targetAxes (w : WeaveShape) : StObj :=
  w.filterMap fun s => match s with | .fixed a => some a | .tiled => none

/-- A single base morphism of Br: one base operation with its reindexings (from St),
    input weaves, and output weaves. The `reindexings` field is where St lives inside Br. -/
structure BrBase (dom cod : BrObj) where
  op           : String
  degree       : StObj
  inputWeaves  : Fin dom.length → WeaveShape
  outputWeaves : Fin cod.length → WeaveShape
  reindexings  : ∀ i : Fin dom.length, StMat degree (inputWeaves i).targetAxes

/-- The free category on BrBase: morphisms are lists of base operations threaded
    sequentially; `nil` is the identity, composition is list concatenation. -/
inductive BrMorph : BrObj → BrObj → Type
  | nil  : (a : BrObj) → BrMorph a a
  | cons : BrBase a b → BrMorph b c → BrMorph a c

def BrMorph.comp {a b c : BrObj} : BrMorph a b → BrMorph b c → BrMorph a c
  | .nil _,     g => g
  | .cons f fs, g => .cons f (BrMorph.comp fs g)

@[simp] theorem BrMorph.nil_comp {a b : BrObj} (g : BrMorph a b) :
    BrMorph.comp (.nil a) g = g := rfl

theorem BrMorph.comp_nil {a b : BrObj} (f : BrMorph a b) :
    BrMorph.comp f (.nil b) = f := by
  induction f with
  | nil _ => rfl
  | cons _ _ ih => simp [BrMorph.comp, ih]

theorem BrMorph.comp_assoc {a b c d : BrObj}
    (f : BrMorph a b) (g : BrMorph b c) (h : BrMorph c d) :
    BrMorph.comp (BrMorph.comp f g) h = BrMorph.comp f (BrMorph.comp g h) := by
  induction f with
  | nil _ => rfl
  | cons _ _ ih => simp [BrMorph.comp, ih]

instance Br : ColoredPROP BrObj where
  gen    := ArrayType
  toList := id
  ofList := id
  hom    := BrMorph
  id     := .nil
  comp   := BrMorph.comp
  id_comp := BrMorph.nil_comp
  comp_id := BrMorph.comp_nil
  assoc   := BrMorph.comp_assoc
  tensor_assoc  := by intro a b c; simp [List.append_assoc]
  tensor_unit_l := by intro a; simp
  tensor_unit_r := by intro a; simp [List.append_nil]
  swap := sorry       -- SIGNATURE (Milestone B+): interleaving rearrangement morphism (§2.3)
  tensorHom := sorry  -- SIGNATURE (Milestone B+): ProductOfMorphisms / parallel product (§2.3)
  tensorHom_id   := by sorry -- SIGNATURE (Milestone G): tensorHom_id for Br
  tensorHom_comp := by sorry -- SIGNATURE (Milestone G): tensorHom_comp for Br
  swap_swap      := by sorry -- SIGNATURE (Milestone G): swap_swap for Br
  elemental := sorry  -- SIGNATURE (Milestone B+): Br is elemental, theory.md argument (§2.3)

end LeanNCD

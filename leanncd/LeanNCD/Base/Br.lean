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

end LeanNCD

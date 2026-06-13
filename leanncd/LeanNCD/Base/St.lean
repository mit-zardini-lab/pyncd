import Mathlib
import LeanNCD.Base.ColoredPROP
import LeanNCD.Base.Numeric

namespace LeanNCD

/-- An axis: an optional name and a symbolic size. -/
structure Axis where
  name : Option String
  size : Numeric

/-- A shape = an ordered list of axes. The objects of St. -/
abbrev StObj := List Axis

/-- A stride morphism `dom → cod`: an affine coordinate transform stored as a coefficient
    matrix over `Numeric` plus a bias vector. Row `j` is the linear combination of input
    coordinates producing output coordinate `j`. -/
structure StMat (dom cod : StObj) where
  coeffs : Matrix (Fin cod.length) (Fin dom.length) Numeric
  bias   : Fin cod.length → Numeric

noncomputable def StMat.id (a : StObj) : StMat a a where
  coeffs := 1
  bias _ := 0

noncomputable def StMat.comp {a b c : StObj} (f : StMat a b) (g : StMat b c) : StMat a c where
  coeffs := g.coeffs * f.coeffs
  bias i := dotProduct (g.coeffs i) f.bias + g.bias i

end LeanNCD

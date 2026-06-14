import Mathlib
import LeanNCD.Seam.Adapter
import LeanNCD.Base.St

namespace LeanNCD
open CategoryTheory

/-- A right `D`-actegory `V` parameterised by a value semiring `R` — the target of `construct()`.
    `actV` is the value-level lift (`P` acts by appending `R`-valued dimensions); the υ/α/δ
    coherences mirror §4, now in `V`. Coherence fields deferred (only `actV` stated). -/
class TargetActegory (D : Type) (V : Type*) [ColoredPROP D] [Category V] (R : Type) [CommSemiring R] where
  actV : (V × Dᵒᵖ) ⥤ V

/-- The default target actegory: finitely-generated (finite-dimensional) `R`-modules (Mathlib). -/
abbrev Mat (R : Type) [CommRing R] := FGModuleCat R

noncomputable instance : TargetActegory StObj (Mat ℝ) ℝ where
  actV := sorry  -- SIGNATURE: appends ℝ-typed dimensions; composition = matrix multiply over ℝ

end LeanNCD

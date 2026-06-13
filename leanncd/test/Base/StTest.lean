import Mathlib
import LeanNCD.Base.St

namespace LeanNCD

-- A one-axis shape and its identity stride morphism.
noncomputable def a0 : Axis := ⟨some "i", MvPolynomial.X "n"⟩

-- TEST: the identity stride matrix has the unit coefficient matrix and zero bias (definitional).
example : (StMat.id [a0]).coeffs = (1 : Matrix (Fin 1) (Fin 1) Numeric) := rfl
example : (StMat.id [a0]).bias = (fun _ => 0) := rfl

end LeanNCD

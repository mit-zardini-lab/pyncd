import Mathlib
import LeanNCD.Base.St

namespace LeanNCD

-- A one-axis shape and its identity stride morphism.
noncomputable def a0 : Axis := ⟨some "i", MvPolynomial.X "n"⟩

-- TEST: the identity stride matrix has the unit coefficient matrix and zero bias (definitional).
-- Coefficients are `Coeff = MvPolynomial String ℤ` (signed), not the ℕ size type `Numeric`.
example : (StMat.id [a0]).coeffs = (1 : Matrix (Fin 1) (Fin 1) Coeff) := rfl
example : (StMat.id [a0]).bias = (fun _ => 0) := rfl

-- TEST: the St category laws are proved with NO sorry.
-- `#print axioms` must list only [propext, Classical.choice, Quot.sound] — never sorryAx.
#print axioms StMat.id_comp
#print axioms StMat.comp_id
#print axioms StMat.comp_assoc

-- TEST: St resolves as a ColoredPROP instance.
noncomputable example : ColoredPROP StObj := inferInstance

end LeanNCD

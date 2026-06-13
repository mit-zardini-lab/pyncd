import Mathlib
import LeanNCD.Base.Numeric

namespace LeanNCD

-- Numeric is a CommSemiring (free from Mathlib).
noncomputable example : CommSemiring Numeric := inferInstance

-- Symbolic axis-size arithmetic discharges by `ring` — the property that makes StMat laws provable.
example (a b : String) :
    (MvPolynomial.X a + MvPolynomial.X b : Numeric) ^ 2
      = MvPolynomial.X a ^ 2 + 2 * (MvPolynomial.X a * MvPolynomial.X b) + MvPolynomial.X b ^ 2 := by
  ring

end LeanNCD

import Mathlib
import LeanNCD.Base.Numeric

namespace LeanNCD

-- Coeff is a CommRing (free from Mathlib) — signed reindex coefficients.
noncomputable example : CommRing Coeff := inferInstance

-- Symbolic coefficient arithmetic discharges by `ring` — the property that makes StMat laws provable.
example (a b : String) :
    (MvPolynomial.X a + MvPolynomial.X b : Coeff) ^ 2
      = MvPolynomial.X a ^ 2 + 2 * (MvPolynomial.X a * MvPolynomial.X b) + MvPolynomial.X b ^ 2 := by
  ring

end LeanNCD

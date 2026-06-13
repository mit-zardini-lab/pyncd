import Mathlib

namespace LeanNCD

/-- Symbolic dimension expressions: the free commutative semiring on `String`-named generators.
    `MvPolynomial.X s` is a FreeNumeric (a symbolic axis size); `MvPolynomial.C ↑n` a literal.
    `CommSemiring` and `DecidableEq` come free from Mathlib; `ring` discharges size identities. -/
abbrev Numeric := MvPolynomial String ℕ

end LeanNCD

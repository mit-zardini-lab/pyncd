import Mathlib

namespace LeanNCD

/-- Symbolic dimension expressions: the free commutative semiring on `String`-named generators.
    `MvPolynomial.X s` is a FreeNumeric (a symbolic axis size); `MvPolynomial.C ↑n` a literal.
    `CommSemiring` and `DecidableEq` come free from Mathlib; `ring` discharges size identities.
    `Numeric` is the *size* type, so its coefficients are `ℕ` (sizes are non-negative). -/
abbrev Numeric := MvPolynomial String ℕ

/-- Affine-transform coefficients — the entries of an `StMat` stride matrix and its bias.
    The free commutative *ring* on `String`-named generators, with `ℤ` coefficients. Distinct
    from `Numeric` because reindexing coefficients/offsets are *signed* — a look-back read
    `X[i-1]` has offset `-1` — which the ℕ-semiring `Numeric` cannot represent. `CommRing`,
    so `ring` discharges the `StMat` laws; symbolic strides remain available via `X`. -/
abbrev Coeff := MvPolynomial String ℤ

end LeanNCD

import Mathlib

namespace LeanNCD

/-- Affine-transform coefficients — the entries of an `StMat` stride matrix and its bias.
    The free commutative *ring* on `String`-named generators, with `ℤ` coefficients. Signed
    because reindexing coefficients/offsets are *signed* — a look-back read `X[i-1]` has offset
    `-1` — which an ℕ-semiring cannot represent. `CommRing`,
    so `ring` discharges the `StMat` laws; symbolic strides remain available via `X`. -/
abbrev Coeff := MvPolynomial String ℤ

end LeanNCD

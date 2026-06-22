import Mathlib   -- for Numeric / MvPolynomial in the toNumeric bridge
import LeanNCD.Base.Numeric

namespace LeanNCD

/-- A computable symbolic axis-size expression — the DSL/executable mirror of `Numeric` (§7.2).
    Variables, ℕ literals, `+`, `-`, `*`, and floor-`/`-by-literal. Computable + `DecidableEq` +
    `ToExpr`, unlike `MvPolynomial`, so the elaborator can construct/compare/embed sizes at
    elaboration time. -/
inductive SizeExpr
  | var : String → SizeExpr
  | lit : Nat → SizeExpr
  | add : SizeExpr → SizeExpr → SizeExpr
  | sub : SizeExpr → SizeExpr → SizeExpr   -- Nat.sub (saturating at 0)
  | mul : SizeExpr → SizeExpr → SizeExpr
  | div : SizeExpr → Nat → SizeExpr        -- floor-division by a positive literal
  deriving DecidableEq, Repr, Inhabited, Lean.ToExpr

/-- Concrete evaluation under a variable assignment. Uses `Nat.sub` (saturating) and `Nat.div`
    (floor). This is the computable path for any front-end size reasoning. -/
def SizeExpr.eval (env : String → Nat) : SizeExpr → Nat
  | .var s   => env s
  | .lit n   => n
  | .add a b => a.eval env + b.eval env
  | .sub a b => a.eval env - b.eval env
  | .mul a b => a.eval env * b.eval env
  | .div e n => e.eval env / n

/-- Bridge to the proof-side `Numeric = MvPolynomial String ℕ` (§2.1). Noncomputable (it builds
    `MvPolynomial`); used only when crossing to the algebra/proof side — NOT in the DSL front-end.

    APPROXIMATIONS: `sub` and `div` are not representable in `MvPolynomial String ℕ`:
    * `.sub a b` maps to `a.toNumeric` (b dropped) — ℕ-polynomial has no negation.
    * `.div e _` maps to `e.toNumeric` (divisor dropped) — floor-div is not polynomial.
    Both are acceptable because `realize` (the sole consumer of `toNumeric`) is sorry-dependent
    on Milestone-B+ sorries; `eval` is the correct computable path for any DSL-side reasoning. -/
noncomputable def SizeExpr.toNumeric : SizeExpr → Numeric
  | .var s   => MvPolynomial.X s
  | .lit n   => MvPolynomial.C (n : ℕ)
  | .add a b => a.toNumeric + b.toNumeric
  | .sub a _ => a.toNumeric             -- APPROXIMATE: b dropped; ℕ-polynomial has no negation
  | .mul a b => a.toNumeric * b.toNumeric
  | .div e _ => e.toNumeric             -- APPROXIMATE: divisor dropped; floor-div is not polynomial

end LeanNCD

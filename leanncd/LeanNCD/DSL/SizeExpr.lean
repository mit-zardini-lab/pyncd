import Mathlib   -- for Numeric / MvPolynomial in the toNumeric bridge
import LeanNCD.Base.Numeric

namespace LeanNCD

/-- A computable symbolic axis-size expression — the DSL/executable mirror of `Numeric` (§7.2).
    Variables, ℕ literals, `+`, `*`. Computable + `DecidableEq` + `ToExpr`, unlike `MvPolynomial`,
    so the elaborator can construct/compare/embed sizes at elaboration time. -/
inductive SizeExpr
  | var : String → SizeExpr
  | lit : Nat → SizeExpr
  | add : SizeExpr → SizeExpr → SizeExpr
  | mul : SizeExpr → SizeExpr → SizeExpr
  deriving DecidableEq, Repr, Inhabited, Lean.ToExpr

/-- Bridge to the proof-side `Numeric = MvPolynomial String ℕ` (§2.1). Noncomputable (it builds
    `MvPolynomial`); used only when crossing to the algebra/proof side — NOT in the DSL front-end. -/
noncomputable def SizeExpr.toNumeric : SizeExpr → Numeric
  | .var s   => MvPolynomial.X s
  | .lit n   => MvPolynomial.C (n : ℕ)
  | .add a b => a.toNumeric + b.toNumeric
  | .mul a b => a.toNumeric * b.toNumeric

end LeanNCD

import Mathlib   -- backs the `deriving DecidableEq, Repr, Inhabited, Lean.ToExpr` clause

namespace LeanNCD

/-- A computable symbolic axis-size expression (§7.2): the sole axis-size type, used by both the
    math tower (`Base/St`) and the DSL presentation layer. Variables, ℕ literals, `+`, `-`, `*`,
    and floor-`/`-by-literal. Computable + `DecidableEq` +
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

end LeanNCD

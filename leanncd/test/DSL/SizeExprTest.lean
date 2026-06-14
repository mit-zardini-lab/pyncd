import LeanNCD.DSL.SizeExpr

namespace LeanNCD

open SizeExpr

-- DecidableEq: structurally distinct size exprs compare unequal; equal ones equal.
#guard decide (SizeExpr.mul (.var "m") (.lit 2) = SizeExpr.mul (.var "m") (.lit 2))
#guard ! decide (SizeExpr.var "m" = SizeExpr.var "n")

-- ToExpr exists (used by the tlprog! macro later).
#check (inferInstance : Lean.ToExpr SizeExpr)

end LeanNCD

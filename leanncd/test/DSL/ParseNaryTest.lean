import LeanNCD.DSL.Elab

namespace LeanNCD
open Lean Elab

-- 3-factor product flattens to 3 factors.
run_cmd do
  let p ← Command.liftTermElabM <| LeanNCD.elabTLProdTerm (← `(tl_prod_term| A[i] · B[i] · C[i]))
  unless p.factors.length == 3 do throwError s!"expected 3 factors, got {p.factors.length}"

-- 3-term sum flattens to 3 terms.
run_cmd do
  let s ← Command.liftTermElabM <| LeanNCD.elabTLSumExpr (← `(tl_sum_expr| A[i] + B[i] + C[i]))
  unless s.terms.length == 3 do throwError s!"expected 3 terms, got {s.terms.length}"

-- mixed: sum of products (each multi-factor) — A·B + C·D + E
run_cmd do
  let s ← Command.liftTermElabM <| LeanNCD.elabTLSumExpr (← `(tl_sum_expr| A[i] · B[i] + C[i] · D[i] + E[i]))
  unless s.terms.length == 3 do throwError s!"expected 3 terms, got {s.terms.length}"
  match s.terms with
  | t0 :: _ => unless t0.factors.length == 2 do throwError "term 0 should have 2 factors"
  | _       => throwError "expected at least one term"

end LeanNCD

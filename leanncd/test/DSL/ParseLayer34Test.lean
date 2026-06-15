import LeanNCD.DSL.Elab

namespace LeanNCD
open Lean Elab

run_cmd do
  let b ← Command.liftTermElabM <| LeanNCD.elabTLBoolExpr (← `(tl_bool_expr| s ≤ q))
  match b with | .rel .le _ _ => pure () | _ => throwError s!"expected ≤ rel, got {repr b}"

run_cmd do
  let r ← Command.liftTermElabM <| LeanNCD.elabTLRHS (← `(tl_rhs| softmax(where s ≤ q)(Q[q, d] · K[s, d])))
  match r.nonlin with | .softmax (some _) => pure () | _ => throwError "expected masked softmax"
  match r.body.terms with
  | [t] => unless t.factors.length == 2 do throwError "expected two factors"
  | _   => throwError "expected one product term"

run_cmd do
  -- Iverson factor + plain read in a product
  let p ← Command.liftTermElabM <| LeanNCD.elabTLProdTerm (← `(tl_prod_term| W[i, k] · X[k, j]))
  unless p.factors.length == 2 do throwError "expected two factors"

end LeanNCD

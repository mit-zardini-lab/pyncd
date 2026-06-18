import LeanNCD.DSL.Elab

namespace LeanNCD
open Lean Elab

run_cmd do
  let ds ← Command.liftTermElabM <| LeanNCD.elabTLDecl (← `(tl_decl| tensor A(i)))
  match ds with
  | [.tensor name axes] =>
      unless name == "A" do throwError "decl name"
      unless axes.length == 1 do throwError "axis count"
  | _ => throwError "expected single tensor decl"

-- general affine read: i + p  (axis + axis) → affine with two unit-coefficient terms
run_cmd do
  let e ← Command.liftTermElabM <| LeanNCD.elabTLIdxExpr (← `(tl_idx_expr| i + p))
  match e with
  | .affine c terms => unless c == 0 && terms.length == 2 do throwError s!"expected affine 0 [_,_], got {repr e}"
  | _ => throwError s!"expected affine, got {repr e}"

-- general affine read: 2 * j + r  (literal stride + axis) → affine
run_cmd do
  let e ← Command.liftTermElabM <| LeanNCD.elabTLIdxExpr (← `(tl_idx_expr| 2 * j + r))
  match e with
  | .affine _ terms => unless terms.length == 2 do throwError s!"expected 2 terms, got {repr e}"
  | _ => throwError s!"expected affine, got {repr e}"

end LeanNCD

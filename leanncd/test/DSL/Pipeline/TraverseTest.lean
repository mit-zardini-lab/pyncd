-- test/DSL/Pipeline/TraverseTest.lean
import LeanNCD.DSL.Traverse
namespace LeanNCD
open Lean Elab
-- Remap UID 1 → 99 everywhere; names untouched.
def bump : UData → UData := fun d => if d.uid == 1 then { d with uid := 99 } else d
run_cmd do
  let a : AxisSpec := { name := "i", uid := 1, kind := .real none }
  let r := TermTraversable.traverseUID bump (IdxExpr.axis a)
  match r with
  | .axis a' => unless a'.uid == 99 && a'.name == "i" do throwError "axis uid not remapped"
  | _ => throwError "shape changed"
-- identity law (spot check)
run_cmd do
  let p : TLProgram := { decls := [], stmts := [
    .assign "Y" [.free { name := "i", uid := 1, kind := .real none }]
      { body := { terms := [] }, nonlin := .identity } ] }
  unless (TermTraversable.traverseUID id p == p) do throwError "traverseUID id ≠ id"
end LeanNCD

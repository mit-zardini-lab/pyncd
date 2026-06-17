import LeanNCD.Eval.Scatter
namespace LeanNCD.Eval
open Std
private def ax (nm : String) (u : Nat) : AxisSpec := { name := nm, uid := u, kind := .real none }
private def tensorOf (shape : List Nat) (xs : List Float) : DenseTensor := ⟨shape, xs.toArray⟩

-- upsample: Out[2*i, 2*j] := X[i,j], X = [[1,2],[3,4]] (2×2) ⇒ 4×4 with X at even coords, 0 elsewhere.
run_cmd do
  let i := ax "i" 1; let j := ax "j" 2
  let X := tensorOf [2,2] [1,2, 3,4]
  let env : HashMap String DenseTensor := ({} : HashMap String DenseTensor).insert "X" X
  let slots : List LHSSlot := [.affine (.scale 2 i), .affine (.scale 2 j)]
  let rhs : RHSExpr := { body := { terms := [{ factors := [.read "X" [.axis i, .axis j]] }] }, nonlin := .identity }
  let sizes := (({} : HashMap UID Nat).insert 1 2).insert 2 2   -- i↦2, j↦2
  match evalScatter env sizes "Out" slots rhs { fill := 0, reduce := none } [4,4] with
  | .error e => throwError e
  | .ok (_, Out) =>
      -- X values at even coords:
      unless Out.get! [0,0] == 1.0 && Out.get! [0,2] == 2.0 && Out.get! [2,0] == 3.0 && Out.get! [2,2] == 4.0 do
        throwError s!"upsample even coords wrong: {repr Out.data}"
      -- odd coords are fill (0):
      unless Out.get! [0,1] == 0.0 && Out.get! [1,1] == 0.0 && Out.get! [3,3] == 0.0 do throwError "upsample fill wrong"

-- reduce sum: two source axes mapping onto the SAME output coord should accumulate.
-- Out[0,0] := X[i,j] for all i,j with reduce sum ⇒ sum of all X = 1+2+3+4 = 10.
run_cmd do
  let i := ax "i" 1; let j := ax "j" 2
  let X := tensorOf [2,2] [1,2, 3,4]
  let env : HashMap String DenseTensor := ({} : HashMap String DenseTensor).insert "X" X
  let slots : List LHSSlot := [.affine (.const 0), .affine (.const 0)]
  let rhs : RHSExpr := { body := { terms := [{ factors := [.read "X" [.axis i, .axis j]] }] }, nonlin := .identity }
  let sizes := (({} : HashMap UID Nat).insert 1 2).insert 2 2
  match evalScatter env sizes "Out" slots rhs { fill := 0, reduce := some "sum" } [4,4] with
  | .error e => throwError e
  | .ok (_, Out) =>
      unless Out.get! [0,0] == 10.0 do throwError s!"reduce sum wrong: {repr Out.data}"

end LeanNCD.Eval

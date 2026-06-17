import LeanNCD.Eval.Eval
namespace LeanNCD.Eval
open Std
private def tensorOf (shape : List Nat) (xs : List Float) : DenseTensor := ⟨shape, xs.toArray⟩
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "W" (tensorOf [2,3] [1,2,3, 4,5,6])).insert "X" (tensorOf [3,2] [1,0, 0,1, 1,1])
  match TLProgram.eval (tlprog!{ Y[i,j] := W[i,k] · X[k,j] }) env with
  | .error e => throwError e
  | .ok out => match out["Y"]? with
    | some Y => unless DenseTensor.approxEq Y (tensorOf [2,2] [4,5, 10,11]) do throwError s!"matmul e2e wrong: {repr Y.data}"
    | none => throwError "no Y"
end LeanNCD.Eval

import LeanNCD.Eval.Contract
namespace LeanNCD.Eval
open Std
private def ax (nm : String) (u : Nat) : AxisSpec := { name := nm, uid := u, kind := .real none }
private def tensorOf (shape : List Nat) (xs : List Float) : DenseTensor := ⟨shape, xs.toArray⟩

-- W (2×3) · X (3×2) = known 2×2 product.
-- W = [[1,2,3],[4,5,6]], X = [[1,0],[0,1],[1,1]]  ⇒  W·X = [[4,5],[10,11]]
run_cmd do
  let i := ax "i" 1; let k := ax "k" 2; let j := ax "j" 3
  let W := tensorOf [2,3] [1,2,3, 4,5,6]
  let X := tensorOf [3,2] [1,0, 0,1, 1,1]
  let env : HashMap String DenseTensor := (({} : HashMap String DenseTensor).insert "W" W).insert "X" X
  let mm : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors := [.read "W" [.axis i, .axis k], .read "X" [.axis k, .axis j]] }] }, nonlin := .identity }
  match inferAxisSizes env [mm] with
  | .error e => throwError e
  | .ok sizes =>
    match evalAssign env sizes "Y" [.free i, .free j] { body := { terms := [{ factors := [.read "W" [.axis i, .axis k], .read "X" [.axis k, .axis j]] }] }, nonlin := .identity } with
    | .error e => throwError e
    | .ok (_, Y) =>
        let expected := tensorOf [2,2] [4,5, 10,11]
        unless DenseTensor.approxEq Y expected do throwError s!"matmul wrong: {repr Y.data}"

-- elementwise sum of two terms: Z[i] := A[i] + B[i]  (A=[1,2,3], B=[10,20,30]) ⇒ [11,22,33]
run_cmd do
  let i := ax "i" 1
  let A := tensorOf [3] [1,2,3]; let B := tensorOf [3] [10,20,30]
  let env : HashMap String DenseTensor := (({} : HashMap String DenseTensor).insert "A" A).insert "B" B
  let rhs : RHSExpr := { body := { terms := [{ factors := [.read "A" [.axis i]] }, { factors := [.read "B" [.axis i]] }] }, nonlin := .identity }
  match inferAxisSizes env [.assign "Z" [.free i] rhs] with
  | .error e => throwError e
  | .ok sizes => match evalAssign env sizes "Z" [.free i] rhs with
    | .error e => throwError e
    | .ok (_, Z) => unless DenseTensor.approxEq Z (tensorOf [3] [11,22,33]) do throwError "sum wrong"

-- missing input tensor ⇒ .error (not silent 0)
run_cmd do
  let i := ax "i" 1
  let A := tensorOf [3] [1,2,3]
  let env : HashMap String DenseTensor := ({} : HashMap String DenseTensor).insert "A" A
  let sizes : HashMap UID Nat := ({} : HashMap UID Nat).insert 1 3
  let rhs : RHSExpr := { body := { terms := [{ factors := [.read "MISSING" [.axis i]] }] }, nonlin := .identity }
  match evalAssign env sizes "Q" [.free i] rhs with
  | .error _ => pure ()
  | .ok _ => throwError "expected error for missing input tensor"
end LeanNCD.Eval

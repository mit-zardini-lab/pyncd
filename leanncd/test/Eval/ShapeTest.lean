import LeanNCD.Eval.Shape
namespace LeanNCD.Eval
open Std

private def ax (nm : String) (u : Nat) : AxisSpec := { name := nm, uid := u, kind := .real none }

-- matmul reads W[i,k], X[k,j] with W:[2,3], X:[3,4] ⇒ i↦2, k↦3, j↦4; Y[i,j] shape [2,4].
run_cmd do
  let i := ax "i" 1; let k := ax "k" 2; let j := ax "j" 3
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "W" (DenseTensor.zeros [2,3])).insert "X" (DenseTensor.zeros [3,4])
  let mm : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors := [.read "W" [.axis i, .axis k], .read "X" [.axis k, .axis j]] }] }, nonlin := .identity }
  match inferAxisSizes {} env [mm] with
  | .error e => throwError e
  | .ok (sizes, _) =>
      unless sizes[1]? == some 2 && sizes[2]? == some 3 && sizes[3]? == some 4 do
        throwError s!"wrong sizes: {sizes[1]?},{sizes[2]?},{sizes[3]?}"
      unless outputShape sizes (Stmt.lhsSlots mm) == [2,4] do throwError "wrong output shape"

-- conflict: same uid bound to two different sizes ⇒ error.
run_cmd do
  let i := ax "i" 1
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "A" (DenseTensor.zeros [2])).insert "B" (DenseTensor.zeros [5])
  let s : Stmt := .assign "Y" [.free i]
    { body := { terms := [{ factors := [.read "A" [.axis i], .read "B" [.axis i]] }] }, nonlin := .identity }
  match inferAxisSizes {} env [s] with
  | .error _ => pure ()
  | .ok _    => throwError "expected axis size conflict"

-- intermediate name not in env is skipped without error; outputShape of unpinned axis ⇒ 0.
run_cmd do
  let i := ax "i" 1
  let env : HashMap String DenseTensor := {}
  let s : Stmt := .assign "Y" [.free i]
    { body := { terms := [{ factors := [.read "Tmp" [.axis i]] }] }, nonlin := .identity }
  match inferAxisSizes {} env [s] with
  | .error e => throwError e
  | .ok (sizes, _) =>
      unless outputShape sizes (Stmt.lhsSlots s) == [0] do throwError "expected [0] for unpinned axis"

end LeanNCD.Eval

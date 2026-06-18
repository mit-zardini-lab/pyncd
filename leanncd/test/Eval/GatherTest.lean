import LeanNCD.Eval.Gather
namespace LeanNCD.Eval
open Std
private def ax (u : Nat) : AxisSpec := { name := "x", uid := u, kind := .real none }
private def coordOf (ps : List (Nat × Int)) : HashMap UID Int := ps.foldl (fun m (u,v) => m.insert u v) {}
-- evalIdx:
#guard evalIdx (coordOf [(1,3),(2,1)]) (.affine 0 [(2, ax 1), (1, ax 2)]) == 7   -- 2*3 + 1
#guard evalIdx (coordOf [(1,0)]) (.shift (ax 1) (-1)) == -1                      -- look-back
#guard evalIdx (coordOf [(1,4)]) (.scale 2 (ax 1)) == 8
-- evalBool: |i - j| ≤ 1 at i=0,j=2 is false; at i=1,j=2 true
#guard ! evalBool (coordOf [(1,0),(2,2)]) (.rel .le (.iabs (.embed (.affine 0 [(1, ax 1),(-1, ax 2)]))) (.embed (.const 1)))
#guard   evalBool (coordOf [(1,1),(2,2)]) (.rel .le (.iabs (.embed (.affine 0 [(1, ax 1),(-1, ax 2)]))) (.embed (.const 1)))
-- gather: in-range read, out-of-range pad, iverson
run_cmd do
  let X := (DenseTensor.zeros [3]).set! [1] 5.0    -- X = [0,5,0]
  let env : HashMap String DenseTensor := ({} : HashMap String DenseTensor).insert "X" X
  -- read X[i] at i=1 ⇒ 5.0
  match gather env (coordOf [(1,1)]) (.read "X" [.axis (ax 1)]) with | .ok v => unless v == 5.0 do throwError "read" | .error e => throwError e
  -- read X[i-1] at i=0 ⇒ out of range ⇒ 0.0
  match gather env (coordOf [(1,0)]) (.read "X" [.shift (ax 1) (-1)]) with | .ok v => unless v == 0.0 do throwError "pad" | .error e => throwError e
  -- iverson [i ≤ 2] at i=1 ⇒ 1.0
  match gather env (coordOf [(1,1)]) (.iverson (.rel .le (.embed (.axis (ax 1))) (.embed (.const 2)))) with | .ok v => unless v == 1.0 do throwError "iverson" | .error e => throwError e
end LeanNCD.Eval

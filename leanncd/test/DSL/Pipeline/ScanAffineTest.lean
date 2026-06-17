-- test/DSL/Pipeline/ScanAffineTest.lean
import LeanNCD.DSL.Compile
namespace LeanNCD
open Lean

-- helper: compile + check whether any routed step has the given op
private def hasOp (p : TLProgram) (op : String) : Bool :=
  match TLProgram.compile p |>.run 0 with
  | .ok tc _ => tc.steps.any (·.op == op)
  | .error _ _ => false

-- AFFINE scan: identity-nonlin recurrence ⇒ op = "scan_affine".
private def affineScan : TLProgram := tlprog!{
  S[j, 0]    := X[j]
  S[j, l +1] := S[j, l] · A[j, k]
}
#guard hasOp affineScan "scan_affine"
#guard ! hasOp affineScan "scan"          -- not the plain tag

-- NONLINEAR scan: relu recurrence ⇒ op = "scan" (NOT scan_affine).
private def reluScan : TLProgram := tlprog!{
  S[j, 0]    := X[j]
  S[j, l +1] := relu(S[j, l] · A[j, k])
}
#guard hasOp reluScan "scan"
#guard ! hasOp reluScan "scan_affine"
end LeanNCD

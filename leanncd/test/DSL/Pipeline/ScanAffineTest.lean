-- test/DSL/Pipeline/ScanAffineTest.lean
import LeanNCD.DSL.Compile
namespace LeanNCD
open Lean

-- helper: compile + check whether any routed step has the given op
private def hasOp (p : TLProgram) (op : BrOp) : Bool :=
  match TLProgram.compile p |>.run 0 with
  | .ok tc _ => tc.steps.any (·.op == op)
  | .error _ _ => false

-- AFFINE scan: identity-nonlin recurrence ⇒ op = .scanAffine.
private def affineScan : TLProgram := tlprog!{
  S[j, 0]    := X[j]
  S[j, l +1] := S[j, l] · A[j, k]
}
#guard hasOp affineScan BrOp.scanAffine
#guard ! hasOp affineScan BrOp.scan          -- not the plain tag

-- NONLINEAR scan: relu recurrence ⇒ op = .scan (NOT .scanAffine).
private def reluScan : TLProgram := tlprog!{
  S[j, 0]    := X[j]
  S[j, l +1] := relu(S[j, l] · A[j, k])
}
#guard hasOp reluScan .scan
#guard ! hasOp reluScan .scanAffine
end LeanNCD

-- test/DSL/Pipeline/RecurMorphismTest.lean
import LeanNCD.DSL.Compile
namespace LeanNCD
open Lean

private def hasOp (p : TLProgram) (op : BrOp) : Bool :=
  match TLProgram.compile p |>.run 0 with
  | .ok tc _ => tc.steps.any (·.op == op)
  | .error _ _ => false

private def isErr (p : TLProgram) : Bool :=
  match TLProgram.compile p |>.run 0 with
  | .ok _ _ => false
  | .error _ _ => true

private def iterAxis : AxisSpec := { name := "l", uid := 0, kind := .nat none }

-- a well-formed pre-built step morphism (one BrBaseP, no inputs):
private def stepTC : ThreadedComposed :=
  { steps := [{ op := BrOp.contract, degree := [], inputWeaves := [], outputWeaves := [[.tiled]],
                reindexings := [] }],
    routing := [[]], nExternal := 0 }

-- a recurMorphism program compiles to a routed step tagged op="scan_pre":
private def recurProg : TLProgram :=
  { decls := [], stmts := [ .recurMorphism "S" iterAxis stepTC ] }
#guard hasOp recurProg BrOp.scanPre

-- an EMPTY pre-built morphism is rejected:
private def emptyTC : ThreadedComposed := { steps := [], routing := [], nExternal := 0 }
private def badProg : TLProgram :=
  { decls := [], stmts := [ .recurMorphism "S" iterAxis emptyTC ] }
#guard isErr badProg
end LeanNCD

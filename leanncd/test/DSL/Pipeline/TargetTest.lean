-- test/DSL/Pipeline/TargetTest.lean
import LeanNCD.DSL.Target
namespace LeanNCD
open Lean
-- ToExpr round-trip + DecidableEq are the point: these must be first-order.
run_cmd do
  let tc : ThreadedComposed :=
    { steps := [{ op := BrOp.contract, degree := [],
                  inputWeaves := [[.tiled]], outputWeaves := [[.tiled]],
                  reindexings := [{ domLen := 1, codLen := 1, coeffs := [[1]], bias := [0] }] }],
      routing := [[⟨0, 0⟩]], nExternal := 1 }
  -- DecidableEq:
  unless (tc == tc) do throwError "ThreadedComposed DecidableEq failed"
  -- ToExpr is derivable (compile-time embedding relies on it):
  let _ : Expr := Lean.toExpr tc
  pure ()
#guard (Wire.mk 0 1) ≠ (Wire.mk 1 0)
end LeanNCD

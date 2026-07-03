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
      routing := [[.external 0]], nExternal := 1 }
  -- DecidableEq:
  unless (tc == tc) do throwError "ThreadedComposed DecidableEq failed"
  -- ToExpr is derivable (compile-time embedding relies on it):
  let _ : Expr := Lean.toExpr tc
  pure ()
#guard (Wire.internal 0 1) ≠ (Wire.internal 1 0)
#guard (Wire.external 0) ≠ (Wire.internal 0 0)   -- the disambiguation the inductive guarantees

/-! ## Track A (Option 1): `StMatP` well-formedness — the reindexing record's shape invariants.
    `coeffs` must be `codLen × domLen` and `bias` length `codLen`; these are unenforced by the
    record type, so `StMatP.wellFormed`/`StMatP.validate` make them checkable + fail-loud. -/

-- a well-formed 2×3 reindexing (codLen = 2 rows, each domLen = 3, bias length 2):
#guard (StMatP.mk 3 2 [[1,0,0],[0,1,0]] [0,0]).wellFormed
-- wrong number of rows (coeffs.length ≠ codLen):
#guard ! (StMatP.mk 3 2 [[1,0,0]] [0,0]).wellFormed
-- a row of the wrong width (≠ domLen):
#guard ! (StMatP.mk 3 2 [[1,0,0],[0,1]] [0,0]).wellFormed
-- bias of the wrong length (≠ codLen):
#guard ! (StMatP.mk 3 2 [[1,0,0],[0,1,0]] [0]).wellFormed
-- `validate` returns the matrix on success, `shapeMismatch` on failure:
#guard ((StMatP.mk 3 2 [[1,0,0],[0,1,0]] [0,0]).validate).toOption.isSome
#guard ((StMatP.mk 3 2 [[1,0,0]] [0,0]).validate).toOption.isNone
end LeanNCD

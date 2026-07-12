import Eval.PropertyOracle.Oracle

namespace LeanNCD.PropertyOracle
open LeanNCD LeanNCD.Eval

-- THE ORACLE: every generated program obeys both laws, else fail the build with a counterexample.
run_cmd do
  match runAll with
  | none => pure ()
  | some msg => throwError s!"E6 property oracle FAILED:\n{msg}"

-- TEST-THE-TESTER (a): a known-good tiny program passes both laws.
private def i0 : AxisSpec := ⟨"i", 1, .real (some (.lit 2))⟩
private def goodProg : TLProgram :=
  { decls := [.axis i0 (some 2), .tensor "A" [i0], .tensor "B" [i0]],
    stmts := [.assign "Y" [.free i0]
      ⟨⟨[⟨[.read "A" [.axis i0]]⟩, ⟨[.read "B" [.axis i0]]⟩]⟩, .identity, .sum⟩] }
private def goodEnv : Std.HashMap String DenseTensor :=
  (({} : Std.HashMap String DenseTensor).insert "A" ⟨[2], #[1.0,2.0]⟩).insert "B" ⟨[2], #[3.0,4.0]⟩
#guard (checkLaws goodProg goodEnv).isNone

-- TEST-THE-TESTER (b): the oracle HAS TEETH — a deliberately-wrong "materialization" that drops a
-- term must be caught by `evalAgreesOn` on `goodProg` (Y = A + B vs a bogus Y = A).
private def bogusSplit : TLProgram :=
  { goodProg with stmts := [.assign "Y" [.free i0] ⟨⟨[⟨[.read "A" [.axis i0]]⟩]⟩, .identity, .sum⟩] }
#guard ! evalAgreesOn (producedNames goodProg)
          (TLProgram.eval goodProg goodEnv) (TLProgram.eval bogusSplit goodEnv)

-- TEST-THE-TESTER (c) (Task 5 — reordering teeth, non-vacuous positive): a Y-DEPENDENT
-- 2-statement program `Y := A + B; Z := Y` passes both laws — in particular, the reordering
-- law's permutation `[Z; Y]` forces `schedule`/`topoSort` to reorder producer-before-consumer
-- to reproduce the baseline, which `checkLaws` verifies via `programPermutations`.
private def yStmt : Stmt :=
  .assign "Y" [.free i0] ⟨⟨[⟨[.read "A" [.axis i0]]⟩, ⟨[.read "B" [.axis i0]]⟩]⟩, .identity, .sum⟩
private def yDepGoodProg : TLProgram :=
  { decls := [.axis i0 (some 2), .tensor "A" [i0], .tensor "B" [i0]],
    stmts := [yStmt, .assign "Z" [.free i0] ⟨⟨[⟨[.read "Y" [.axis i0]]⟩]⟩, .identity, .sum⟩] }
#guard (checkLaws yDepGoodProg goodEnv).isNone

-- TEST-THE-TESTER (d): a genuinely-DIFFERENT program (not a permutation of `yDepGoodProg`'s
-- statements — `Z` here reads `A` directly instead of the produced `Y`) must NOT agree with it;
-- `evalAgreesOn` catches the semantic difference on the produced names.
private def yDepDifferentProg : TLProgram :=
  { yDepGoodProg with
    stmts := [yStmt, .assign "Z" [.free i0] ⟨⟨[⟨[.read "A" [.axis i0]]⟩]⟩, .identity, .sum⟩] }
#guard ! evalAgreesOn (producedNames yDepGoodProg)
          (TLProgram.eval yDepGoodProg goodEnv) (TLProgram.eval yDepDifferentProg goodEnv)

end LeanNCD.PropertyOracle

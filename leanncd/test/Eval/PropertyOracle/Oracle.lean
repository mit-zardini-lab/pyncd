import Eval.PropertyOracle.Compare
import Eval.PropertyOracle.Transforms
import Eval.PropertyOracle.Gen
import LeanNCD.Eval.Eval

namespace LeanNCD.PropertyOracle
open LeanNCD LeanNCD.Eval

/-- Check both laws on one program. Returns `none` if OK, or `some msg` describing the first
    violation (baseline must be `.ok`; every permutation and the split must agree on the
    program's produced names). -/
def checkLaws (p : TLProgram) (env : Std.HashMap String DenseTensor) : Option String :=
  let names := producedNames p
  let base := TLProgram.eval p env
  match base with
  | .error e => some s!"baseline did not evaluate (generator well-formedness gap): {e}\n{repr p}"
  | .ok _ =>
      -- reordering: every permutation must agree with the baseline
      let reorderBad := (programPermutations p).find? (fun p' =>
        ! evalAgreesOn names base (TLProgram.eval p' env))
      match reorderBad with
      | some p' => some s!"REORDERING law violated.\nbase: {repr p}\nperm: {repr p'}"
      | none =>
        -- materialization: the split must agree with the baseline
        if evalAgreesOn names base (TLProgram.eval (materializeSplit p) env) then none
        else some s!"MATERIALIZATION law violated.\nbase: {repr p}\nsplit: {repr (materializeSplit p)}"

/-- Run both laws over the whole generator; `none` if all pass, else the first failure message. -/
def runAll : Option String :=
  enumPrograms.findSome? (fun (p, env) => checkLaws p env)

end LeanNCD.PropertyOracle

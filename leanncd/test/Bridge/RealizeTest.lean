-- test/Bridge/RealizeTest.lean
import LeanNCD.Bridge.Realize
namespace LeanNCD
-- typecheck (noncomputable ⇒ no #eval):
noncomputable example (a : AxisP) : Axis := realizeAxis a
noncomputable example (s : StObjP) : StObj := realizeStObj s
noncomputable example (m : StMatP) (d c : StObj) : StMat d c := realizeStMat m d c
-- realizeAxis / realizeStObj must be sorry-FREE (no `sorryAx` in their axiom list):
#print axioms realizeAxis
#print axioms realizeStObj
-- intToNumeric legitimately uses sorryAx (the negative-coeff obligation):
#print axioms intToNumeric
end LeanNCD

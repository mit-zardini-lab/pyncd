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
-- intToCoeff is now SORRY-FREE (Coeff = MvPolynomial String ℤ is signed):
#print axioms intToCoeff
-- WEAVE realizations + dependent realizeBrBaseP (Task 2):
noncomputable example (w : WeaveShapeP) : WeaveShape := realizeWeaveShape w
noncomputable example (b : BrBaseP) : Σ (dom cod : BrObj), BrBase dom cod := realizeBrBaseP b
#print axioms realizeWeaveShape   -- must be sorry-FREE
#print axioms realizeBrBaseP      -- now SORRY-FREE (realizeStMat + intToCoeff are sorry-free)
-- COMPOSITE realize (Task 3): the threaded DAG → one Br morphism.
noncomputable example (tc : ThreadedComposed) : Σ (dom cod : BrObj), BrMorph dom cod := realize tc
#print axioms realize    -- uses sorryAx (expected: the composite obligation)
end LeanNCD

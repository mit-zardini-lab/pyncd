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
-- WEAVE realizations + dependent realizeBrBaseP (Task 2):
noncomputable example (w : WeaveShapeP) : WeaveShape := realizeWeaveShape w
noncomputable example (b : BrBaseP) : Σ (dom cod : BrObj), BrBase dom cod := realizeBrBaseP b
#print axioms realizeWeaveShape   -- must be sorry-FREE
#print axioms realizeBrBaseP      -- sorry-free IF you gave the real reindexings term; else uses sorryAx
-- COMPOSITE realize (Task 3): the threaded DAG → one Br morphism.
noncomputable example (tc : ThreadedComposed) : Σ (dom cod : BrObj), BrMorph dom cod := realize tc
#print axioms realize    -- uses sorryAx (expected: the composite obligation)
end LeanNCD

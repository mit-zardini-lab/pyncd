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

-- BEHAVIOR: realize's dom/cod derivation (Realize.lean) — dom = FIRST step's input weaves,
-- cod = LAST step's output weaves (the documented-partial heuristic; the full routing-walk
-- derivation is a later obligation). These pin the current shape behavior so that change is a
-- conscious, test-breaking one. (Only dom/cod are projected — the `sorry` morphism body, in
-- `.snd.snd`, is never forced.)
private def mkStep (ins outs : Nat) : BrBaseP :=
  { op := .contract, degree := [],
    inputWeaves := List.replicate ins [], outputWeaves := List.replicate outs [],
    reindexings := [] }
private def tc2 : ThreadedComposed :=
  { steps := [mkStep 2 1, mkStep 1 3], routing := [], nExternal := 0 }
-- dom = first step's 2 inputs; cod = last step's 3 outputs:
example : (realize tc2).fst.length = 2 := by rfl
example : (realize tc2).snd.fst.length = 3 := by rfl
-- empty program ⇒ empty dom and cod:
example : (realize { steps := [], routing := [], nExternal := 0 }).fst = [] := by rfl
example : (realize { steps := [], routing := [], nExternal := 0 }).snd.fst = [] := by rfl
end LeanNCD

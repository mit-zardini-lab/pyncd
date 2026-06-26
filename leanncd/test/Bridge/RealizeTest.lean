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
-- COMPOSITE realize: the threaded DAG → ONE Br morphism. Now SORRY-FREE under `WellFormed`
-- (the body is the `interpUpto`/`finalPiece` fold). `#print axioms` must NOT list `sorryAx`.
noncomputable example (tc : ThreadedComposed) (h : tc.WellFormed) :
    Σ (dom cod : BrObj), BrMorph dom cod := realize tc h
#print axioms realize    -- [propext, Classical.choice, Quot.sound] — sorryAx GONE

-- BEHAVIOR: `realize`'s `dom`/`cod` are `realizeDom`/`codObj` (independent of the proof `h`), so they
-- are tested directly. Both are noncomputable (realized `ArrayType`); only their LENGTH is checked.
private def mkStep (ins outs : Nat) : BrBaseP :=
  { op := .contract, degree := [],
    inputWeaves := List.replicate ins [], outputWeaves := List.replicate outs [],
    reindexings := [] }

-- matmul-shaped: 1 step, 2 external inputs (slots 0,1), 1 output. nExternal = 2.
private def tcMatmul : ThreadedComposed :=
  { steps := [mkStep 2 1], routing := [[.external 0, .external 1]], nExternal := 2 }
#guard tcMatmul.externalPort 0 == some (0, 0)
#guard tcMatmul.externalPort 1 == some (0, 1)
#guard tcMatmul.wellFormedDom
example : (realizeDom tcMatmul).length = 2 := by rfl     -- dom: one entry per external slot
example : tcMatmul.codObj.length = 1 := by rfl           -- cod: last step's outputs

-- two-layer net: steps [H ⟵ W1,X ; Y ⟵ W2,H]. Externals W1,X,W2 = slots 0,1,2; H internal.
private def tc2Layer : ThreadedComposed :=
  { steps := [mkStep 2 1, mkStep 2 1],
    routing := [[.external 0, .external 1], [.external 2, .internal 0 0]], nExternal := 3 }
#guard tc2Layer.externalPort 0 == some (0, 0)
#guard tc2Layer.externalPort 1 == some (0, 1)
#guard tc2Layer.externalPort 2 == some (1, 0)   -- W2 consumed at step 1, input 0
#guard tc2Layer.wellFormedDom
example : (realizeDom tc2Layer).length = 3 := by rfl

-- empty program ⇒ empty dom and cod:
example : realizeDom { steps := [], routing := [], nExternal := 0 } = [] := by rfl
example : ThreadedComposed.codObj { steps := [], routing := [], nExternal := 0 } = [] := by rfl

-- NEGATIVE: one external slot read at two different ranks ⇒ not well-formed. One step, two inputs
-- both reading external slot 0: input 0 at rank 1 (`[.fixed _]`), input 1 at rank 0 (`[]`).
private def tcBadRank : ThreadedComposed :=
  { steps := [{ op := .contract, degree := [],
                inputWeaves := [[.fixed default], []], outputWeaves := [[]], reindexings := [] }],
    routing := [[.external 0, .external 0]], nExternal := 1 }
#guard ! tcBadRank.wellFormedDom

-- COMBINATORIAL PLAN (`wirePlan`): per-step selections + final selection, by the §4 liveness fold.
-- matmul: one step reading both externals, no carry ⇒ sel [0,1]; final picks the sole output.
#guard (tcMatmul.wirePlan.1.map (·.sel)) == [[0, 1]]
#guard tcMatmul.wirePlan.2.2 == [0]
-- two-layer: step0 reads W1,X and carries W2 ⇒ [0,1,2]; step1 reads W2,H (swapped in pool) ⇒ [1,0].
#guard (tc2Layer.wirePlan.1.map (·.sel)) == [[0, 1, 2], [1, 0]]
#guard tc2Layer.wirePlan.2.2 == [0]
-- fan-out: ext0 read by BOTH steps. step0 ⇒ copy (reads pos 0, carries pos 0) = [0,0]; step1 ⇒ [1]
-- (reads ext0 at pool pos 1, DROPPING step0's now-dead output) — exercising copy and discard.
private def tcFan : ThreadedComposed :=
  { steps := [mkStep 1 1, mkStep 1 1], routing := [[.external 0], [.external 0]], nExternal := 1 }
#guard (tcFan.wirePlan.1.map (·.sel)) == [[0, 0], [1]]
#guard tcFan.wirePlan.2.2 == [0]
end LeanNCD

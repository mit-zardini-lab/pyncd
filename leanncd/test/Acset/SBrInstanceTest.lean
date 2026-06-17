import LeanNCD.Acset.SBrInstance
namespace LeanNCD.Acset
-- a small instance is constructible and DecidableEq works:
def sampleInst : SBrInstance :=
  { axisSizes := [(⟨.rawAxis, 0⟩, .lit 5)],
    equations := [{ equationIdx := 0, lhsName := some "Y" }],
    arrays    := [{ equationIdx := 0, slot := 0, name := some "Y", isInput := false,
                    operatorTag := some .identity, normAxis := none, datatypeTag := .reals,
                    maxValue := none, bias := none, elementwiseFn := none,
                    opPredicate := none, wireLabel := none }],
    arrayAxes := [{ equationIdx := 0, arraySlot := 0, axisUid := ⟨.rawAxis, 0⟩,
                    isTarget := true, position := 0 }],
    samples   := [{ equationIdx := 0, reindexingSlot := 1, srcUid := ⟨.rawAxis, 0⟩,
                    tgtUid := ⟨.normAxis, 1⟩, coeff := 1, offset := -1 }] }
#guard sampleInst == sampleInst
#guard decide (sampleInst ≠ { sampleInst with samples := [] })
#guard (sampleInst.samples.head!.offset == -1)   -- negative offset round-trips as Int
end LeanNCD.Acset

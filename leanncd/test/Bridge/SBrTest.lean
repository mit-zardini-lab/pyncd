import LeanNCD.Bridge.SBr
namespace LeanNCD
-- realizeSBr now takes the full-fidelity Acset.SBrInstance:
noncomputable example (s : Acset.SBrInstance) : Σ (dom cod : BrObj), BrMorph dom cod := realizeSBr s
#print axioms realizeSBr   -- uses sorryAx (the CSV-path obligation)
end LeanNCD

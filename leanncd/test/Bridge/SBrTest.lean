import LeanNCD.Bridge.SBr
namespace LeanNCD
-- the structure + a row type are usable:
example : EquationRow := { op := "contract", outputArray := 0 }
noncomputable example (s : SBrInstance) : Σ (dom cod : BrObj), BrMorph dom cod := realizeSBr s
#print axioms realizeSBr   -- uses sorryAx (the CSV-path obligation)
end LeanNCD

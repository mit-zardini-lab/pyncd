import LeanNCD.Bridge.Agreement
namespace LeanNCD
noncomputable example (tc : ThreadedComposed) : Acset.SBrInstance := fromThreadedComposed tc
-- the agreement theorem has the intended Σ-equality TYPE (not weakened):
example (tc : ThreadedComposed) :
    (realize tc = realizeSBr (fromThreadedComposed tc)) = (realize tc = realizeSBr (fromThreadedComposed tc)) := rfl
-- the theorems exist with the right statements:
#check @realize_fromThreadedComposed_agree
#check @agree_dom
#check @agree_cod
#print axioms realize_fromThreadedComposed_agree   -- uses sorryAx
end LeanNCD

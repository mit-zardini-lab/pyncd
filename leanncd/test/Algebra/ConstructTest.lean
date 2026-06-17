import LeanNCD.Algebra.Construct

namespace LeanNCD
open CategoryTheory

-- The flagship algebra resolves by instance search.
noncomputable example : Algebra StObj BrObj (Mat ℝ) ℝ := inferInstance

#check @construct_correspondence
#check @semiring_choice_split

#print axioms instAlgebraBrMatR          -- uses sorryAx (expected)
#print axioms construct_correspondence    -- uses sorryAx (expected)
#print axioms semiring_choice_split       -- uses sorryAx (expected)

end LeanNCD

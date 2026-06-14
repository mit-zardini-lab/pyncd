import Mathlib
import LeanNCD.Algebra.Target
import LeanNCD.Base.St

namespace LeanNCD
open CategoryTheory

#check @TargetActegory
noncomputable example : TargetActegory StObj (Mat ℝ) ℝ := inferInstance

end LeanNCD

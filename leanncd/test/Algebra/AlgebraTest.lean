import Mathlib
import LeanNCD.Algebra.Target
import LeanNCD.Algebra.Algebra
import LeanNCD.Base.St

namespace LeanNCD
open CategoryTheory

#check @TargetActegory
noncomputable example : TargetActegory StObj (Mat ℝ) ℝ := inferInstance

#check @Algebra
#check @ParaAlgebra
-- ParaAlgebra extends Algebra:
example {D C : Type} {V : Type*} [ColoredPROP D] [ColoredPROP C] [Category V]
    {R : Type} [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R]
    [ParaAlgebra D C V R] : Algebra D C V R := inferInstance

end LeanNCD

import Mathlib
import LeanNCD.Algebra.Target
import LeanNCD.Algebra.Algebra
import LeanNCD.Base.St

namespace LeanNCD
open CategoryTheory

#check @TargetActegory
noncomputable example : TargetActegory StObj (Mat ℝ) ℝ := inferInstance

/-- Named handle on the default `Mat ℝ` target actegory, so `#print axioms` can inspect it
    (now carrying the full υ/α/δ/triangle/pentagon/naturality coherences). -/
@[reducible] noncomputable def matTargetActegory : TargetActegory StObj (Mat ℝ) ℝ := inferInstance
#print axioms matTargetActegory   -- expect: sorryAx (all coherence fields are `sorry`)

#check @Algebra
#check @ParaAlgebra
-- ParaAlgebra extends Algebra:
example {D C : Type} {V : Type*} [ColoredPROP D] [ColoredPROP C] [Category V] [MonoidalCategory V]
    {R : Type} [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R]
    [ParaAlgebra D C V R] : Algebra D C V R := inferInstance

end LeanNCD

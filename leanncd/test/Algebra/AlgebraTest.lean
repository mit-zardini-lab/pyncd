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
    [SymmetricCategory V]
    {R : Type} [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R]
    [ParaAlgebra D C V R] : Algebra D C V R := inferInstance

-- The Algebra class elaborates with the new strong-symmetric-monoidal field; `F` and `Fbraided`
-- are accessible from a variable instance.
section
variable (D C : Type) (V : Type*) [ColoredPROP D] [ColoredPROP C] [Category V] [MonoidalCategory V]
  [SymmetricCategory V] (R : Type) [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R]
  [inst : Algebra D C V R]
example : C ⥤ V := inst.F
example : inst.F.Braided := inst.Fbraided
-- The new equivar coherence (coh) fields are accessible and typecheck as fields:
#check @inst.equivar_nat   -- equivar natural in the `C`-variable
#check @inst.equivar_υ     -- equivar carries `υ` to `υ_V`
#check @inst.equivar_α     -- equivar carries `α` to `α_V`
#check @inst.equivar_δ     -- equivar carries `δ` to `δ_V` (mediated by `F.μ`)
#check @inst.F_ev_p        -- `F` preserves the §4.1 evaluation `ev_p`
end

end LeanNCD

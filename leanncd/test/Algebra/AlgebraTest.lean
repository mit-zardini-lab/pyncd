import Mathlib
import LeanNCD.Algebra.Target
import LeanNCD.Algebra.Algebra
import LeanNCD.Base.St

namespace LeanNCD
open CategoryTheory

#check @TargetActegory

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

-- The Para fields of `ParaAlgebra` elaborate as intended: the `Para(C)→Para(V)` action on
-- parametric morphisms, its `μ`-mediated defining law, and the weight-tying reparameterization 2-cell.
section
variable (D C : Type) (V : Type*) [ColoredPROP D] [ColoredPROP C] [Category V] [MonoidalCategory V]
  [SymmetricCategory V] (R : Type) [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R]
  [inst : ParaAlgebra D C V R]
#check @inst.paraMap       -- Para(C)→Para(V) action on 1-cells
#check @inst.paraMap_eq    -- paraMap factors as μ ≫ F.map f
#check @inst.weightTie     -- weight tying as a reparameterization 2-cell
end

end LeanNCD

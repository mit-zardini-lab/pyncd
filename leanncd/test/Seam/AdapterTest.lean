import Mathlib
import LeanNCD.Seam.Adapter
import LeanNCD.Base.St
import LeanNCD.Base.Br

namespace LeanNCD

open CategoryTheory

-- TEST: the seam gives St and Br objects a Mathlib Category instance.
-- `noncomputable` because the underlying St/Br ColoredPROP data is noncomputable (Numeric);
-- the seam instance itself is computable — synthesis is all this test checks.
noncomputable example : Category StObj := inferInstance
noncomputable example : Category BrObj := inferInstance

-- TEST: Mathlib morphism notation (𝟙, ≫) works on these objects via the seam.
example (a : StObj) : (𝟙 a) ≫ (𝟙 a) = 𝟙 a := by simp

-- TEST: the Category instance carries no sorry.
#print axioms instCategoryOfColoredPROP

end LeanNCD

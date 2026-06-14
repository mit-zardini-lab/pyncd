import Mathlib
import LeanNCD.Mixins.Temporal

namespace LeanNCD
open CategoryTheory

example {D C : Type} [ColoredPROP D] [ColoredPROP C] [TemporalGraded D C] : True := by trivial
#check @TemporalGraded.L
#check @TemporalGraded.iterate
example {D C : Type} [ColoredPROP D] [ColoredPROP C] [TemporalGraded D C] :
    DGradedColoredPROP D C := inferInstance

end LeanNCD

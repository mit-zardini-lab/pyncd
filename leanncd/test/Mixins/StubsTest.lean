import Mathlib
import LeanNCD.Mixins.Stubs

namespace LeanNCD
open CategoryTheory

example {D C : Type} [ColoredPROP D] [ColoredPROP C] [RouteStructure D C] :
    DGradedColoredPROP D C := inferInstance
example {D C : Type} [ColoredPROP D] [ColoredPROP C] (T : Monad D) [SymmetryGraded D C T] :
    DGradedColoredPROP D C := inferInstance

end LeanNCD

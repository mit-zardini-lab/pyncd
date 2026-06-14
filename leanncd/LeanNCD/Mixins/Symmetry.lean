import Mathlib
import LeanNCD.Core.Graded

namespace LeanNCD
open CategoryTheory

/-- `SymmetryGraded` — Prop 8.4 equivariance via the Eilenberg–Moore category of a symmetry monad
    `T` on `D`. STUB: the EM-machinery is gated (equivariance_unification.md). -/
class SymmetryGraded (D C : Type) [ColoredPROP D] [ColoredPROP C]
    (T : CategoryTheory.Monad D)
    extends DGradedColoredPROP D C

end LeanNCD

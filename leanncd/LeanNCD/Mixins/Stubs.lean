import Mathlib
import LeanNCD.Core.Graded

/-! # Mixin stubs — Route and Symmetry
    Merged one-class stubs (formerly `Mixins/Route.lean` + `Mixins/Symmetry.lean`). -/

namespace LeanNCD
open CategoryTheory

/-- `RouteStructure` — Prop 8.6(ii): data-dependent reindexing (no fixed `D`-morphism, hence no
    weave). STUB: the data-dependent coproduct injection + gate-as-Para-parameter are deferred. -/
class RouteStructure (D C : Type) [ColoredPROP D] [ColoredPROP C]
    extends DGradedColoredPROP D C

/-- `SymmetryGraded` — Prop 8.4 equivariance via the Eilenberg–Moore category of a symmetry monad
    `T` on `D`. STUB: the EM-machinery is gated (equivariance_unification.md). -/
class SymmetryGraded (D C : Type) [ColoredPROP D] [ColoredPROP C]
    (T : CategoryTheory.Monad D)
    extends DGradedColoredPROP D C

end LeanNCD

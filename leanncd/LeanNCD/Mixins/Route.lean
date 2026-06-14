import Mathlib
import LeanNCD.Core.Graded

namespace LeanNCD

/-- `RouteStructure` — Prop 8.6(ii): data-dependent reindexing (no fixed `D`-morphism, hence no
    weave). STUB: the data-dependent coproduct injection + gate-as-Para-parameter are deferred. -/
class RouteStructure (D C : Type) [ColoredPROP D] [ColoredPROP C]
    extends DGradedColoredPROP D C

end LeanNCD

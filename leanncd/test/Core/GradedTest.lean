import Mathlib
import LeanNCD.Core.Graded
import LeanNCD.Base.St
import LeanNCD.Base.Br

namespace LeanNCD

-- TEST: sh_star folds a per-color shape map over an object's color list.
-- For a trivial shape map into St-objects, sh_star of the empty object is the unit ([]).
example : sh_star (C := BrObj) (D := StObj) (fun _ => ([] : StObj)) ([] : BrObj) = ([] : StObj) :=
  rfl

end LeanNCD

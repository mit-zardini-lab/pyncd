import Mathlib
import LeanNCD.Props.Generic

namespace LeanNCD
open CategoryTheory

#check @lift_functorial          -- 8.1
#check @weave_subsingleton       -- 8.2
#check @scan_batches             -- 8.8
#check @scan_catamorphism        -- 8.7

-- 8.1 functoriality half proved sorry-free: confirm no `sorryAx` in its axioms.
#print axioms lift_functorial

end LeanNCD

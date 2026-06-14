import Mathlib
import LeanNCD.Core.Graded
import LeanNCD.Base.St
import LeanNCD.Base.Br

namespace LeanNCD

-- TEST: sh_star folds a per-color shape map over an object's color list.
-- For a trivial shape map into St-objects, sh_star of the empty object is the unit ([]).
example : sh_star (C := BrObj) (D := StObj) (fun _ => ([] : StObj)) ([] : BrObj) = ([] : StObj) :=
  rfl

open CategoryTheory in
-- TEST: the class elaborates and is usable as a hypothesis; data fields have expected types.
example {C D : Type} [ColoredPROP D] [ColoredPROP C] [DGradedColoredPROP D C] : True := by
  trivial

#check @DGradedColoredPROP.act
#check @DGradedColoredPROP.sh
#check @DGradedColoredPROP.broadcast_gen

open CategoryTheory in
-- TEST: ev_p is defined as a morphism family derived from act + υ.
#check @ev_p

-- TEST: Eq. 3 (ev_p naturality) is recorded as a lemma.
#check @ev_p_naturality

-- TEST: ev_p_naturality is proved sorry-free (no `sorryAx` among its axioms — only the
-- standard classical/quotient axioms remain).
#print axioms ev_p_naturality

end LeanNCD

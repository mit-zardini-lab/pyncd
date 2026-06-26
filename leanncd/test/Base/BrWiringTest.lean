import LeanNCD.Base.BrWiring

namespace LeanNCD

-- Each `example`'s type ascription forces the wiring's codomain `sel.map s.get` to reduce to the
-- expected bundle — so these check BOTH construction and the target-object computation.

-- permutation (swap two wires): sel = [1, 0] ⇒ [A,B] ↦ [B,A]
noncomputable example (A B : ArrayType) : BrMorph [A, B] [B, A] := BrMorph.wiring [A, B] [1, 0]

-- identity selection: sel = [0, 1] ⇒ [A,B] ↦ [A,B]
noncomputable example (A B : ArrayType) : BrMorph [A, B] [A, B] := BrMorph.wiring [A, B] [0, 1]

-- copy (fan-out): sel = [0, 0] ⇒ [A] ↦ [A,A]
noncomputable example (A : ArrayType) : BrMorph [A] [A, A] := BrMorph.wiring [A] [0, 0]

-- discard one wire: sel = [1] ⇒ [A,B] ↦ [B]  (A dropped)
noncomputable example (A B : ArrayType) : BrMorph [A, B] [B] := BrMorph.wiring [A, B] [1]

-- discard the whole bundle: sel = [] ⇒ [A,B] ↦ []
noncomputable example (A B : ArrayType) : BrMorph [A, B] [] := BrMorph.wiring [A, B] []

-- copy-and-permute together: sel = [1, 0, 1] ⇒ [A,B] ↦ [B,A,B]
noncomputable example (A B : ArrayType) : BrMorph [A, B] [B, A, B] := BrMorph.wiring [A, B] [1, 0, 1]

-- the gather primitive and its inverse type as expected:
noncomputable example (A B : ArrayType) : BrMorph [A, B] [B, A] := BrMorph.rotateToFront [A, B] 1

-- TEST: the wiring morphisms are sorry-free (`[propext, Quot.sound]`; `Classical.choice` from
-- `noncomputable`/Mathlib is expected, `sorryAx` must NOT appear).
#print axioms BrMorph.rotateToFront
#print axioms BrMorph.pick
#print axioms BrMorph.delAll
#print axioms BrMorph.wiring

-- wiring by identity (`wiringBy`): select Nat-tagged wires from a pool by identity (here a
-- permutation [0,1] ↦ [1,0]). Codomain is `tgt.map f` exactly, by construction (no cast).
noncomputable example (f : Nat → ArrayType) :
    BrMorph ([0, 1].map f) ([1, 0].map f) :=
  BrMorph.wiringBy f [0, 1] [1, 0] (by decide)
#print axioms BrMorph.pickBy
#print axioms BrMorph.wiringBy

end LeanNCD

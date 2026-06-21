import Mathlib
import LeanNCD.Base.Br

namespace LeanNCD

noncomputable def ax_i : Axis := ⟨some "i", MvPolynomial.X "n"⟩

-- TEST: targetAxes keeps the .fixed axes in order and drops the .tiled slots.
example :
    WeaveShape.targetAxes [WeaveSlot.tiled, WeaveSlot.fixed ax_i, WeaveSlot.tiled] = [ax_i] :=
  rfl

-- TEST: `idm` is a left identity for composition (now via the quotient, not definitional).
example {a b : BrObj} (g : BrMorph a b) : BrMorph.comp (BrMorph.idm a) g = g :=
  BrMorph.id_comp g

-- TEST: the Br category + symmetric-monoidal laws are proved with NO sorry
-- (only `[propext, Quot.sound]`; in particular `swap_swap` and `tensorHom_comp` — impossible in
-- the old free-list presentation — now hold).
#print axioms BrMorph.id_comp
#print axioms BrMorph.comp_id
#print axioms BrMorph.assoc
#print axioms BrMorph.tensor_id
#print axioms BrMorph.tensor_comp
#print axioms BrMorph.braid_braid

-- TEST: Br resolves as a ColoredPROP instance.
example : ColoredPROP BrObj := inferInstance

end LeanNCD

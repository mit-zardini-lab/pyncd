import Mathlib
import LeanNCD.Base.Br

namespace LeanNCD

def ax_i : Axis := ⟨some "i", SizeExpr.var "n"⟩

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

-- TEST: the per-wire CD comonoid (copy `Δ`, discard `ε`) and its laws.
-- Generators carry the expected wire types:
noncomputable example (A : ArrayType) : BrMorph [A] [A, A] := BrMorph.copyW A
noncomputable example (A : ArrayType) : BrMorph [A] []     := BrMorph.delW A
-- Comonoid laws are sorry-free (`[propext, Quot.sound]`), like the SMC laws above — they are `Rel`
-- constructors of the presentation, so they hold by `Quot.sound`; only the right counit crosses a cast.
#print axioms BrMorph.copyW_coassoc
#print axioms BrMorph.copyW_counitl
#print axioms BrMorph.copyW_cocomm
#print axioms BrMorph.copyW_counitr_heq

end LeanNCD

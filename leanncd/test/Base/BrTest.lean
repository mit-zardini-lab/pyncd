import Mathlib
import LeanNCD.Base.Br

namespace LeanNCD

noncomputable def ax_i : Axis := ⟨some "i", MvPolynomial.X "n"⟩

-- TEST: targetAxes keeps the .fixed axes in order and drops the .tiled slots.
example :
    WeaveShape.targetAxes [WeaveSlot.tiled, WeaveSlot.fixed ax_i, WeaveSlot.tiled] = [ax_i] :=
  rfl

-- TEST: nil is a definitional left identity for composition.
example {a b : BrObj} (g : BrMorph a b) : BrMorph.comp (.nil a) g = g := rfl

end LeanNCD

import Mathlib
import LeanNCD.Base.ColoredPROP
import LeanNCD.Base.Numeric

namespace LeanNCD

/-- An axis: an optional name and a symbolic size. -/
structure Axis where
  name : Option String
  size : Numeric

/-- A shape = an ordered list of axes. The objects of St. -/
abbrev StObj := List Axis

/-- A stride morphism `dom → cod`: an affine coordinate transform stored as a coefficient
    matrix over `Numeric` plus a bias vector. Row `j` is the linear combination of input
    coordinates producing output coordinate `j`. -/
@[ext]
structure StMat (dom cod : StObj) where
  coeffs : Matrix (Fin cod.length) (Fin dom.length) Numeric
  bias   : Fin cod.length → Numeric

noncomputable def StMat.id (a : StObj) : StMat a a where
  coeffs := 1
  bias _ := 0

noncomputable def StMat.comp {a b c : StObj} (f : StMat a b) (g : StMat b c) : StMat a c where
  coeffs := g.coeffs * f.coeffs
  bias i := dotProduct (g.coeffs i) f.bias + g.bias i

theorem StMat.id_comp {a b : StObj} (f : StMat a b) : StMat.comp (StMat.id a) f = f := by
  apply StMat.ext
  · simp [StMat.comp, StMat.id, Matrix.mul_one]
  · funext i
    simp [StMat.comp, StMat.id]

theorem StMat.comp_id {a b : StObj} (f : StMat a b) : StMat.comp f (StMat.id b) = f := by
  apply StMat.ext
  · simp [StMat.comp, StMat.id, Matrix.one_mul]
  · funext i
    simp [StMat.comp, StMat.id, Matrix.one_apply, dotProduct]

theorem StMat.comp_assoc {a b c d : StObj} (f : StMat a b) (g : StMat b c) (h : StMat c d) :
    StMat.comp (StMat.comp f g) h = StMat.comp f (StMat.comp g h) := by
  apply StMat.ext
  · simp [StMat.comp, Matrix.mul_assoc]
  · funext i
    simp only [StMat.comp, Matrix.mul_apply, dotProduct, Finset.mul_sum, mul_add,
      Finset.sum_add_distrib, Finset.sum_mul, mul_assoc]
    rw [add_assoc, Finset.sum_comm]

noncomputable instance St : ColoredPROP StObj where
  gen    := Axis
  toList := id
  ofList := id
  hom    := StMat
  id     := StMat.id
  comp   := StMat.comp
  id_comp := StMat.id_comp
  comp_id := StMat.comp_id
  assoc   := StMat.comp_assoc
  tensor_assoc  := by intro a b c; simp [List.append_assoc]
  tensor_unit_l := by intro a; simp
  tensor_unit_r := by intro a; simp [List.append_nil]
  tensorHom {a b c d} f g :=                            -- block-diagonal product
    let eC : Fin (a ++ c).length ≃ Fin a.length ⊕ Fin c.length :=
      (finCongr (List.length_append (as := a) (bs := c))).trans finSumFinEquiv.symm
    let eB : Fin (b ++ d).length ≃ Fin b.length ⊕ Fin d.length :=
      (finCongr (List.length_append (as := b) (bs := d))).trans finSumFinEquiv.symm
    { coeffs := Matrix.reindex eB.symm eC.symm (Matrix.fromBlocks f.coeffs 0 0 g.coeffs)
      bias   := fun i => Sum.elim f.bias g.bias (eB i) }
  swap := sorry        -- SIGNATURE (Milestone B+): permutation matrix, zero bias (§2.2)
  tensorHom_id   := by sorry -- SIGNATURE (Milestone G): tensorHom_id for St
  tensorHom_comp := by sorry -- SIGNATURE (Milestone G): tensorHom_comp for St
  swap_swap      := by sorry -- SIGNATURE (Milestone G): swap_swap for St
  elemental := sorry   -- SIGNATURE (Milestone B+): stride matrices separated by their points (§2.2)

end LeanNCD

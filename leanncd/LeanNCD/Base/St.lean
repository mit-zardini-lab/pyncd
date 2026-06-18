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
    matrix over `Coeff` plus a bias vector. Row `j` is the linear combination of input
    coordinates producing output coordinate `j`. The entries are `Coeff = MvPolynomial String ℤ`
    (signed), NOT `Numeric` (the ℕ size type): reindexing offsets can be negative (look-back). -/
@[ext]       -- generate the extensionality lemma StMat.ext for St
structure StMat (dom cod : StObj) where
  coeffs : Matrix (Fin cod.length) (Fin dom.length) Coeff
  bias   : Fin cod.length → Coeff

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
  swap a b :=                                            -- block-anti-diagonal permutation
    let eAB : Fin (a ++ b).length ≃ Fin a.length ⊕ Fin b.length :=
      (finCongr (List.length_append (as := a) (bs := b))).trans finSumFinEquiv.symm
    let eBA : Fin (b ++ a).length ≃ Fin b.length ⊕ Fin a.length :=
      (finCongr (List.length_append (as := b) (bs := a))).trans finSumFinEquiv.symm
    { coeffs := Matrix.reindex eBA.symm eAB.symm
        (Matrix.fromBlocks 0 1 1 0 :
          Matrix (Fin b.length ⊕ Fin a.length) (Fin a.length ⊕ Fin b.length) Coeff)
      bias   := fun _ => 0 }
  tensorHom_id   := by
    intro a c  -- a c : StObj (the two shapes being tensored)
    apply StMat.ext
    · -- fromBlocks 1 0 0 1, reindexed by eB/eC, equals the identity matrix.
      -- EmbeddingLike.apply_eq_iff_eq handles e i = e j ↔ i = j.
      ext i j
      simp [StMat.id, Matrix.one_apply, EmbeddingLike.apply_eq_iff_eq]
    · -- Sum.elim (fun _ => 0) (fun _ => 0) collapses to the zero bias.
      funext i
      simp [StMat.id]
  tensorHom_comp := by
    intro a b c d e g f₁ f₂ g₁ g₂  -- shapes a b c, d e g : StObj; f₁ : a→b, f₂ : b→c, g₁ : d→e, g₂ : e→g
    apply StMat.ext
    · -- Coefficients: reindex/fromBlocks block-diagonal multiply, middle equiv cancels.
      simp only [StMat.comp, Matrix.reindex_apply, Equiv.symm_symm,
        Matrix.submatrix_mul_equiv, Matrix.fromBlocks_multiply, Matrix.mul_zero,
        Matrix.zero_mul, add_zero, zero_add]
    · -- Bias: reindex the inner sum, then case-split feeds both arms one `simp`.
      funext i
      simp only [StMat.comp, Matrix.reindex_apply, Equiv.symm_symm,
        dotProduct, Matrix.submatrix_apply]
      rw [Equiv.sum_comp ((finCongr (List.length_append (as := b) (bs := e))).trans
            finSumFinEquiv.symm)
          (fun y => Matrix.fromBlocks f₂.coeffs 0 0 g₂.coeffs
            (((finCongr (List.length_append (as := c) (bs := g))).trans finSumFinEquiv.symm) i) y *
            Sum.elim f₁.bias g₁.bias y)]
      rcases h : ((finCongr (List.length_append (as := c) (bs := g))).trans finSumFinEquiv.symm) i
        with k | k <;>
        simp [Matrix.fromBlocks, Fintype.sum_sum_type, mul_comm]
  swap_swap a b := by
    apply StMat.ext
    · -- (swap b a).coeffs * (swap a b).coeffs = 1.
      -- Rewrite both reindexes as submatrices, merge via submatrix_mul_equiv, and use
      -- fromBlocks 0 1 1 0 * fromBlocks 0 1 1 0 = 1.
      show Matrix.reindex _ _ _ * Matrix.reindex _ _ _ = (1 : Matrix _ _ Coeff)
      rw [Matrix.reindex_apply, Matrix.reindex_apply, Equiv.symm_symm, Equiv.symm_symm,
        Matrix.submatrix_mul_equiv, Matrix.fromBlocks_multiply]
      simp [Matrix.fromBlocks_one, -Equiv.coe_trans, Matrix.submatrix_one_equiv]
    · -- Both swap biases are zero, so dotProduct (coeffs i) 0 + 0 = 0.
      funext i
      simp [StMat.comp, StMat.id]
  elemental := by                       -- points (§2.2) separate parallel stride matrices
    intro X Y f g h  -- X Y : StObj; f g : StMat X Y; h : ∀ point x : StMat [] X, x ≫ f = x ≫ g
    -- Bias agrees: feed the zero point `x = 0`, whose dot products vanish.
    have hbias : f.bias = g.bias := by
      funext i
      simpa [StMat.comp, dotProduct] using
        congrArg (fun m => StMat.bias m i) (h { coeffs := 0, bias := 0 })
    -- Coefficients agree column-by-column: feed `x.bias = Pi.single k 1`.
    refine StMat.ext ?_ hbias
    funext i k
    have hk := congrArg (fun m => StMat.bias m i) (h { coeffs := 0, bias := Pi.single k 1 })
    simp only [StMat.comp, dotProduct_single, mul_one, congrFun hbias i] at hk
    exact add_right_cancel (b := g.bias i) hk

end LeanNCD

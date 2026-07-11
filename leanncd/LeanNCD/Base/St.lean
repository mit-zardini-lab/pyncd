import Mathlib
import LeanNCD.Base.ColoredPROP
import LeanNCD.Base.Numeric
import LeanNCD.Base.SizeExpr

namespace LeanNCD

/-- An axis: an optional name and a symbolic size. -/
structure Axis where
  name : Option String
  size : SizeExpr

/-- A shape = an ordered list of axes. The objects of St. -/
abbrev StObj := List Axis

/-- A stride morphism `dom → cod`: an affine coordinate transform stored as a coefficient
    matrix over `Coeff` plus a bias vector. Row `j` is the linear combination of input
    coordinates producing output coordinate `j`. The entries are `Coeff = MvPolynomial String ℤ`
    (signed), NOT `Numeric` (the ℕ size type): reindexing offsets can be negative (look-back).

    The indices `Fin cod.length` / `Fin dom.length` count **axes**, not elements within an axis.
    `Axis.size` is symbolic metadata (a computable `SizeExpr`, was `Numeric`); the type of `StMat` does not enforce that
    coordinate values stay within `[0, n)` for an axis of size `n`. That bound only becomes
    operative at evaluation time (Milestone I / `DenseTensor`), where symbolic sizes are
    instantiated to concrete `Nat`s. -/
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

/-- The column/row index identity for the left strict unitor on `St`: the block-decomposition
    equiv `(Fin unit.length ⊕ Fin l.length) ≃ Fin (unit ++ l).length` sends the right injection of
    `x` back to `x` (the empty left block `Fin unit.length = Fin 0` carries no coordinates). -/
private theorem StMatAux.unit_col_idx (l : StObj) (x : Fin (List.length l)) :
    x = (((finCongr (List.length_append (as := (id [] : StObj)) (bs := l))).trans
        finSumFinEquiv.symm).symm.symm.symm (Sum.inr x)) := by
  apply Fin.ext
  simp only [Equiv.symm_symm, Equiv.symm_trans_apply, finSumFinEquiv_apply_right,
    finCongr_symm_apply, Fin.val_cast, Fin.val_natAdd,
    show List.length (id [] : StObj) = 0 from rfl, Nat.zero_add]

/-- Heterogeneous `StMat` extensionality: two stride matrices over equal domains/codomains are
    `HEq` when their coefficient matrices and bias vectors are `HEq`. The bridge from the concrete
    cross-type tensor laws (`tensorHom_unit_r`/`tensorHom_assoc`) to genuine `HEq`. -/
private theorem StMat.hext {a b a' b' : StObj} (m : StMat a b) (n : StMat a' b')
    (ha : a = a') (hb : b = b')
    (hc : HEq m.coeffs n.coeffs) (hbi : HEq m.bias n.bias) : HEq m n := by
  subst ha hb; apply heq_of_eq; apply StMat.ext
  · exact eq_of_heq hc
  · exact eq_of_heq hbi

/-- Heterogeneous matrix extensionality over `Fin`-index size equalities. -/
private theorem StMatAux.matrix_hext {m m' n n' : ℕ}
    (M : Matrix (Fin m) (Fin n) Coeff) (N : Matrix (Fin m') (Fin n') Coeff)
    (hm : m = m') (hn : n = n')
    (he : ∀ i j, M i j = N (Fin.cast hm i) (Fin.cast hn j)) : HEq M N := by
  subst hm hn; apply heq_of_eq; funext i j; simpa using he i j

/-- Heterogeneous function extensionality over a `Fin`-size equality. -/
private theorem StMatAux.fun_hext {m m' : ℕ} (F : Fin m → Coeff) (G : Fin m' → Coeff)
    (hm : m = m') (he : ∀ i, F i = G (Fin.cast hm i)) : HEq F G := by
  subst hm; apply heq_of_eq; funext i; simpa using he i

/-- The column/row index identity for the right strict unitor on `St`: with the empty *right* block
    `Fin (id []).length = Fin 0`, the block-decomposition equiv sends every index to the left
    injection `Sum.inl` of the corresponding coordinate. -/
private theorem StMatAux.unitr_idx {N p q : ℕ} (hq : q = 0) (h : N = p + q) (i : Fin N) :
    (((finCongr h).trans (finSumFinEquiv (m := p) (n := q)).symm).symm.symm) i
      = Sum.inl (Fin.cast (by omega) i) := by
  simp only [Equiv.symm_symm, Equiv.trans_apply]
  rw [Equiv.symm_apply_eq]
  apply Fin.ext
  rw [finSumFinEquiv_apply_left, finCongr_apply]
  simp

/-- `Fin.cast` across `(a+b)+c = a+(b+c)` sends the doubly-left-injected index to the
    left-injection: the bridge for the first block under list reassociation. -/
private theorem StMatAux.cast_castAdd_castAdd (a b c : ℕ) (x : Fin a) :
    Fin.cast (Nat.add_assoc a b c) (Fin.castAdd c (Fin.castAdd b x)) = Fin.castAdd (b + c) x := by
  apply Fin.ext; simp

/-- `Fin.cast` across `(a+b)+c = a+(b+c)` sends the middle index to the right-then-left injection. -/
private theorem StMatAux.cast_castAdd_natAdd (a b c : ℕ) (x : Fin b) :
    Fin.cast (Nat.add_assoc a b c) (Fin.castAdd c (Fin.natAdd a x))
      = Fin.natAdd a (Fin.castAdd c x) := by
  apply Fin.ext; simp

/-- `Fin.cast` across `(a+b)+c = a+(b+c)` sends the right index to the doubly-right injection. -/
private theorem StMatAux.cast_natAdd (a b c : ℕ) (x : Fin c) :
    Fin.cast (Nat.add_assoc a b c) (Fin.natAdd (a + b) x) = Fin.natAdd a (Fin.natAdd b x) := by
  apply Fin.ext; simp; omega

attribute [local simp] StMatAux.cast_castAdd_castAdd StMatAux.cast_castAdd_natAdd
  StMatAux.cast_natAdd finSumFinEquiv_symm_apply_castAdd finSumFinEquiv_symm_apply_natAdd

/-- Block-diagonal associativity entrywise: a nested `fromBlocks (fromBlocks A 0 0 B) 0 0 C`
    reindexed through the `((·++·)++·)` length splits equals `fromBlocks A 0 0 (fromBlocks B 0 0 C)`
    reindexed through the `(·++(·++·))` splits. The index-equiv bridge under `List.append_assoc` is
    handled by the three `Fin.cast`/`addCases` lemmas above; the 9 block cases close by `simp`. -/
private theorem StMatAux.block_reassoc
    {p₁ q₁ r₁ p₂ q₂ r₂ L₁ L₂ R₁ R₂ Lc Ld : ℕ}
    (A : Matrix (Fin p₁) (Fin p₂) Coeff) (B : Matrix (Fin q₁) (Fin q₂) Coeff)
    (C : Matrix (Fin r₁) (Fin r₂) Coeff)
    (hL₁ : L₁ = p₁ + q₁) (hL₂ : L₂ = p₂ + q₂) (hR₁ : R₁ = q₁ + r₁) (hR₂ : R₂ = q₂ + r₂)
    (hLc : Lc = L₁ + r₁) (hLd : Ld = L₂ + r₂) (hRc : Lc = p₁ + R₁) (hRd : Ld = p₂ + R₂)
    (i : Fin Lc) (j : Fin Ld) :
    Matrix.fromBlocks
        ((Matrix.fromBlocks A 0 0 B).submatrix ((finCongr hL₁).trans finSumFinEquiv.symm)
          ((finCongr hL₂).trans finSumFinEquiv.symm)) 0 0 C
        (((finCongr hLc).trans finSumFinEquiv.symm) i) (((finCongr hLd).trans finSumFinEquiv.symm) j)
      = Matrix.fromBlocks A 0 0
        ((Matrix.fromBlocks B 0 0 C).submatrix ((finCongr hR₁).trans finSumFinEquiv.symm)
          ((finCongr hR₂).trans finSumFinEquiv.symm))
        (((finCongr hRc).trans finSumFinEquiv.symm) i)
        (((finCongr hRd).trans finSumFinEquiv.symm) j) := by
  subst hL₁ hL₂ hR₁ hR₂ hLc hLd
  simp only [Equiv.trans_apply, finCongr_apply, Fin.cast_eq_self]
  refine Fin.addCases (fun i12 => ?_) (fun i3 => ?_) i <;>
    refine Fin.addCases (fun j12 => ?_) (fun j3 => ?_) j
  · refine Fin.addCases (fun i1 => ?_) (fun i2 => ?_) i12 <;>
      refine Fin.addCases (fun j1 => ?_) (fun j2 => ?_) j12 <;> simp [Matrix.submatrix_apply]
  · refine Fin.addCases (fun i1 => ?_) (fun i2 => ?_) i12 <;> simp [Matrix.submatrix_apply]
  · refine Fin.addCases (fun j1 => ?_) (fun j2 => ?_) j12 <;> simp [Matrix.submatrix_apply]
  · simp [Matrix.submatrix_apply]

/-- Bias-side analogue of `block_reassoc`: the nested `Sum.elim` of biases under the `((·++·)++·)`
    splits equals the reassociated `Sum.elim`-of-`Sum.elim` under the `(·++(·++·))` splits. -/
private theorem StMatAux.bias_reassoc
    {p q r L R Lo : ℕ} (F : Fin p → Coeff) (G : Fin q → Coeff) (H : Fin r → Coeff)
    (hL : L = p + q) (hR : R = q + r) (hLo : Lo = L + r) (hRo : Lo = p + R) (i : Fin Lo) :
    Sum.elim (fun k => Sum.elim F G (((finCongr hL).trans finSumFinEquiv.symm) k)) H
        (((finCongr hLo).trans finSumFinEquiv.symm) i)
      = Sum.elim F (fun k => Sum.elim G H (((finCongr hR).trans finSumFinEquiv.symm) k))
          (((finCongr hRo).trans finSumFinEquiv.symm) i) := by
  subst hL hR hLo
  simp only [Equiv.trans_apply, finCongr_apply, Fin.cast_eq_self]
  refine Fin.addCases (fun i12 => ?_) (fun i3 => ?_) i
  · refine Fin.addCases (fun i1 => ?_) (fun i2 => ?_) i12 <;> simp
  · simp

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
  tensor_assoc  := by intro a b c  -- a b c : StObj
                      simp [List.append_assoc]
  tensor_unit_l := by intro a  -- a : StObj
                      simp
  tensor_unit_r := by intro a  -- a : StObj
                      simp [List.append_nil]
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
  swap_natural := by
    intro a b c d f g  -- a b c d : StObj; f : StMat a b; g : StMat c d
    apply StMat.ext
    · -- perm_{b,d} * blockdiag(f,g) = blockdiag(g,f) * perm_{a,c}
      -- Both sides: merge reindex into submatrix, fromBlocks multiply, then the
      -- anti-diagonal permutation swaps the blocks: 0·f=0, 1·g=g (LHS top), etc.
      show Matrix.reindex _ _ _ * Matrix.reindex _ _ _
          = Matrix.reindex _ _ _ * Matrix.reindex _ _ _
      rw [Matrix.reindex_apply, Matrix.reindex_apply, Matrix.reindex_apply, Matrix.reindex_apply,
        Equiv.symm_symm, Equiv.symm_symm, Equiv.symm_symm, Equiv.symm_symm,
        Matrix.submatrix_mul_equiv, Matrix.submatrix_mul_equiv,
        Matrix.fromBlocks_multiply, Matrix.fromBlocks_multiply]
      simp [Matrix.mul_zero, Matrix.zero_mul, Matrix.one_mul, Matrix.mul_one]
    · -- Bias: LHS = dotProduct (perm_{b,d} row) (blockdiag(f,g).bias); RHS = blockdiag(g,f).bias.
      funext i
      simp only [StMat.comp, dotProduct, Matrix.reindex_apply, Equiv.symm_symm,
        Matrix.submatrix_apply, add_zero]
      rw [Equiv.sum_comp ((finCongr (List.length_append (as := b) (bs := d))).trans
            finSumFinEquiv.symm)
          (fun y => Matrix.fromBlocks (0 : Matrix _ _ Coeff) 1 1 0
              (((finCongr (List.length_append (as := d) (bs := b))).trans finSumFinEquiv.symm) i) y
            * Sum.elim f.bias g.bias y)]
      rcases h : ((finCongr (List.length_append (as := d) (bs := b))).trans finSumFinEquiv.symm) i
        with k | k <;>
        simp [Matrix.fromBlocks, Fintype.sum_sum_type, Matrix.one_apply, Finset.sum_ite_eq]
  swap_hexagon_fwd := by sorry   -- SIGNATURE (§11): hexagon for StMat swap, deferred
  swap_hexagon_rev := by sorry   -- SIGNATURE (§11): hexagon for StMat swap, deferred
  -- Cross-type `HEq` of nested block-diagonal stride matrices across `List.append_assoc`. Split by
  -- `StMat.hext` into coeffs/bias `HEq`, then discharged entrywise via `StMatAux.block_reassoc`
  -- (the 9-case `fromBlocks`/`Fin.addCases` block-reassociation) and `StMatAux.bias_reassoc`, with
  -- the `finCongr`/`finSumFinEquiv` index casts bridged by the three `StMatAux.cast_*` lemmas.
  tensorHom_assoc := by
    intro a₁ b₁ a₂ b₂ a₃ b₃ f g h  -- a₁ b₁ a₂ b₂ a₃ b₃ : StObj; f : StMat a₁ b₁; g : StMat a₂ b₂; h : StMat a₃ b₃
    apply StMat.hext _ _ (by simp [List.append_assoc]) (by simp [List.append_assoc])
    · -- Coefficients: the nested block-diagonal reindex reassociates (`StMatAux.block_reassoc`).
      apply StMatAux.matrix_hext _ _ (by simp) (by simp)
      intro i j
      simp only [Matrix.reindex_apply, Equiv.symm_symm]
      exact StMatAux.block_reassoc f.coeffs g.coeffs h.coeffs
        (List.length_append ..) (List.length_append ..) (List.length_append ..)
        (List.length_append ..) (List.length_append ..) (List.length_append ..)
        (by simp) (by simp) i j
    · -- Bias: the nested `Sum.elim` of biases reassociates (`StMatAux.bias_reassoc`).
      apply StMatAux.fun_hext _ _ (by simp)
      intro i
      exact StMatAux.bias_reassoc f.bias g.bias h.bias
        (List.length_append ..) (List.length_append ..) (List.length_append ..) (by simp) i
  tensorHom_unit_l := by
    -- `tensor unit a = [] ++ a = a` definitionally, so this HEq is same-type; prove the
    -- underlying `StMat` equality `tensorHom (id []) f = f` componentwise.
    intro a b f  -- a b : StObj; f : StMat a b
    apply heq_of_eq
    apply StMat.ext
    · -- Coefficients: `reindex (fromBlocks (id[]) 0 0 f.coeffs) = f.coeffs`. Flip via the reindex
      -- equiv to a `submatrix`, then the empty left block kills the `inl` rows/cols.
      show (Matrix.reindex _ _ (Matrix.fromBlocks (StMat.id []).coeffs 0 0 f.coeffs)) = f.coeffs
      rw [← Equiv.eq_symm_apply, Matrix.reindex_symm, Matrix.reindex_apply]
      funext x y
      rcases x with x | x
      · exact x.elim0
      rcases y with y | y
      · exact y.elim0
      rw [Matrix.fromBlocks_apply₂₂, Matrix.submatrix_apply]
      congr 1
      · exact StMatAux.unit_col_idx b x
      · exact StMatAux.unit_col_idx a y
    · -- Bias: `Sum.elim (id[]).bias f.bias` at the right-injected index reduces to `f.bias`.
      funext i
      show Sum.elim (StMat.id []).bias f.bias _ = f.bias i
      rcases h : (((finCongr (List.length_append (as := (id [] : StObj)) (bs := b))).trans
          finSumFinEquiv.symm)) i with k | k
      · exact k.elim0
      · show f.bias k = f.bias i
        congr 1
        have hk : finSumFinEquiv (Sum.inr k)
            = (finCongr (List.length_append (as := (id [] : StObj)) (bs := b))) i := by
          rw [← h, Equiv.trans_apply, Equiv.apply_symm_apply]
        apply Fin.ext
        have hv := Fin.val_eq_of_eq hk
        simp only [finSumFinEquiv_apply_right, finCongr_apply, Fin.val_natAdd, Fin.val_cast,
          show List.length (id [] : StObj) = 0 from rfl, Nat.zero_add] at hv
        exact hv
  tensorHom_unit_r := by
    -- Cross-type HEq across `a ++ [] = a` (`List.append_nil`): the right tensor factor `id []` has
    -- the empty `Fin 0` block, so `tensorHom f (id [])` is `f` reindexed back through the
    -- left-injection. Bridge concrete `StMat` cross-type equality to `HEq` via `StMat.hext`.
    intro a b f  -- a b : StObj; f : StMat a b
    apply StMat.hext
    · show a ++ (id [] : StObj) = a; simp
    · show b ++ (id [] : StObj) = b; simp
    · -- Coefficients: empty right block selects `f.coeffs` via `fromBlocks_apply₁₁`.
      show HEq (Matrix.reindex _ _ (Matrix.fromBlocks f.coeffs 0 0 (StMat.id []).coeffs)) f.coeffs
      apply StMatAux.matrix_hext _ _ (by simp) (by simp)
      intro i j
      rw [Matrix.reindex_apply, Matrix.submatrix_apply]
      calc Matrix.fromBlocks f.coeffs 0 0 (StMat.id []).coeffs _ _
          = Matrix.fromBlocks f.coeffs 0 0 (StMat.id []).coeffs
              (Sum.inl (Fin.cast (by simp) i)) (Sum.inl (Fin.cast (by simp) j)) := by
            congr 1 <;> exact (StMatAux.unitr_idx rfl (List.length_append ..) _)
        _ = f.coeffs (Fin.cast (by simp) i) (Fin.cast (by simp) j) := Matrix.fromBlocks_apply₁₁ ..
    · -- Bias: `Sum.elim f.bias (id[]).bias` at the left-injected index reduces to `f.bias`.
      show HEq (fun i => Sum.elim f.bias (StMat.id []).bias
          (((finCongr (List.length_append (as := b) (bs := (id [] : StObj)))).trans
            finSumFinEquiv.symm) i)) f.bias
      apply StMatAux.fun_hext _ _ (by simp)
      intro i
      rcases h : (((finCongr (List.length_append (as := b) (bs := (id [] : StObj)))).trans
          finSumFinEquiv.symm)) i with k | k
      · show f.bias k = _
        congr 1
        have hk : finSumFinEquiv (Sum.inl k)
            = (finCongr (List.length_append (as := b) (bs := (id [] : StObj)))) i := by
          rw [← h, Equiv.trans_apply, Equiv.apply_symm_apply]
        apply Fin.ext
        have hv := Fin.val_eq_of_eq hk
        simp only [finSumFinEquiv_apply_left, finCongr_apply, Fin.val_castAdd, Fin.val_cast] at hv
        exact hv
      · exact k.elim0

/-- St elementality — the **(Elem)** mixin, proved sorry-free: a stride matrix is determined by its
    action on points (global elements). Moved out of the `ColoredPROP` instance in the 2026-06-22
    `elemental`→mixin demotion (see `Base/ColoredPROP.lean`); the proof is unchanged. -/
instance : Elemental StObj where
  elemental := by                       -- points (§2.2) separate parallel stride matrices
    intro X Y f g h  -- X Y : StObj; f g : StMat X Y; h : ∀ point x, x ≫ f = x ≫ g
    -- Re-type `h` from `SmallCategory.comp` to `StMat.comp` (defeq for the `St` instance), so the
    -- `StMat.comp` simp lemmas fire (they no longer match the bare `SmallCategory.comp` projection
    -- now that `elemental` is a separate `Elemental` instance rather than a `ColoredPROP` field).
    have h : ∀ x : StMat [] X, StMat.comp x f = StMat.comp x g := h
    -- Bias agrees: feed the zero point `x = 0`, whose dot products vanish.
    have hbias : f.bias = g.bias := by
      funext i
      simpa [StMat.comp, dotProduct] using
        congrArg (fun m => StMat.bias m i) (h { coeffs := 0, bias := 0 })
    -- Coefficients agree column-by-column: feed `x.bias = Pi.single k 1` (one-hot vector at k)
    refine StMat.ext ?_ hbias
    funext i k
    have hk := congrArg (fun m => StMat.bias m i) (h { coeffs := 0, bias := Pi.single k 1 })
    simp only [StMat.comp, dotProduct_single, mul_one, congrFun hbias i] at hk
    exact add_right_cancel (b := g.bias i) hk

end LeanNCD

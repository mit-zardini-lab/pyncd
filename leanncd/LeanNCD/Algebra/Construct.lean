import LeanNCD.Algebra.Algebra
import LeanNCD.Instances.StBr        -- instDGradedStBr

namespace LeanNCD
open CategoryTheory

/-- The flagship algebra: `Br` evaluates into finite-dimensional ℝ-modules (`Mat ℝ`) — the
    categorical content of `construct()` for the standard (×, +) semiring `R = ℝ`. All
    structure fields SIGNATURE (`sorry`): `F` is the `Br → Mat ℝ` evaluation functor (each `BrBase`
    ↦ its ℝ-linear map); the strong-symmetric-monoidal structure (`Fbraided`) and the
    equivariance laws (`equivar`, `equivar_nat`, `equivar_υ`, `equivar_α`, `equivar_δ`, `F_ev_p`)
    are the genuine §7.5 obligations. `noncomputable` (the `Mat ℝ` target is noncomputable). -/
noncomputable instance instAlgebraBrMatR : Algebra StObj BrObj (Mat ℝ) ℝ where
  F           := sorry   -- the Br → Mat ℝ evaluation functor
  Fbraided    := sorry   -- the strong-symmetric-monoidal structure on F
  equivar     := sorry
  equivar_nat := sorry
  equivar_υ   := sorry
  equivar_α   := sorry
  equivar_δ   := sorry
  F_ev_p      := sorry

/-- §7.5 `construct()` correspondence (statement; proof SIGNATURE).

    There is no Python `construct` in Lean to equate against, so we state the faithful
    categorical content of `construct()`: evaluating a `Br` morphism in `Mat ℝ` realizes the
    §4.1 *evaluation* `ev_p` as the `actV`-mediated value-evaluation downstairs. Concretely, for
    a point `p : P ⟶ I_St` and a `Br` object `X`, the algebra functor `F` carries the upstairs
    slice `ev_p p X : X ⊛ P ⟶ X` to the `Mat ℝ`-evaluation `actV.map (𝟙, p) ≫ υ_V`, mediated by
    the equivariance iso. This is exactly `construct`'s ℝ-valued read of a lifted `Br` morphism
    (plug the point `p` into the appended dimension, then contract via the unitor). It is the
    `Algebra.F_ev_p` law specialized to the flagship instance — a NON-vacuous equation between two
    genuinely different `Mat ℝ`-morphisms. -/
theorem construct_correspondence
    {P : StObjᵒᵖ} (p : P ⟶ (Opposite.op (ColoredPROP.unit : StObj))) (X : BrObj) :
    instAlgebraBrMatR.F.map (ev_p p X)
      = (instAlgebraBrMatR.equivar X P).hom
        ≫ (TargetActegory.actV (D := StObj) (V := Mat ℝ) (R := ℝ)).map
            (X := (instAlgebraBrMatR.F.obj X, P))
            (Y := (instAlgebraBrMatR.F.obj X, Opposite.op (ColoredPROP.unit : StObj)))
            (𝟙 (instAlgebraBrMatR.F.obj X), p)
        ≫ (TargetActegory.υ_V (D := StObj) (R := ℝ) (instAlgebraBrMatR.F.obj X)).hom :=
  sorry

/-- §7.5 value-semiring choice (statement; proof SIGNATURE).

    The "∃/∧-vs-Σ/× split is exactly the choice of `R`." Contraction in a `TargetActegory`
    target is "multiply, then combine" using the value semiring's `(×, +)`. The two §7.5 readings
    differ precisely in the *additive* combine:

      • `R = ℝ`   ⇒ combine is `Σ` (genuine summation/counting); `ℝ`'s `+` is NOT idempotent
                    (`(1 : ℝ) + 1 = 2 ≠ 1`).
      • `R = Bool` ⇒ combine is `∃` (`∨`); `Bool`'s `+` (= `∨`) IS idempotent
                    (`(true : Bool) + true = true`).

    So the additive structure of `R` — idempotent or not — is exactly what switches
    exists-of-conjunctions (relational/predicate) contraction for sum-of-products (tensor/linear)
    contraction. We state this faithful, NON-vacuous fact about the two value semirings directly
    (the `Bool` target category itself is a deferred Task-2 obligation — see `Target.lean` §7.5
    note — so we do not reference a `TargetActegory _ _ Bool` instance here). -/
theorem semiring_choice_split :
    ((1 : ℝ) + 1 ≠ 1) ∧ ((true : Bool) + true = true) :=
  sorry

end LeanNCD

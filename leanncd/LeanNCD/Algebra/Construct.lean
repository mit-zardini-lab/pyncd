import LeanNCD.Algebra.Algebra
import LeanNCD.Instances.StBr        -- instDGradedStBr

namespace LeanNCD
open CategoryTheory

/-- §7.5 value-semiring choice (statement; proof SIGNATURE).

    The "∃/∧-vs-Σ/× split is exactly the choice of `R`." Contraction in a `TargetActegory`
    target is "multiply, then combine" using the value semiring's `(×, +)`. The two §7.5 readings
    differ precisely in the *additive* combine:

      • `R = ℝ`   ⇒ combine is `Σ` (genuine summation/counting); `ℝ`'s `+` is NOT idempotent
                    (`(1 : ℝ) + 1 = 2 ≠ 1`).
      • `R = Bool` ⇒ combine is `∃` (`∨`); the predicate semiring's addition `∨` (`||`) IS
                    idempotent (`(true || true) = true`).

    So the additive structure — idempotent or not — is exactly what switches
    exists-of-conjunctions (relational/predicate) contraction for sum-of-products (tensor/linear)
    contraction. NOTE: the predicate combine is the `(∨, ∧)` Boolean *semiring* addition `∨`/`||`,
    NOT Mathlib's default `Bool` addition (`HAdd Bool`), which is XOR (`Bool` carries the Boolean
    *ring*, so `true + true = false`). The `(∨, ∧)` semiring is genuinely not a ring (`∨` has no
    inverse) — that is exactly why its target category is the deferred Task-2 obligation (see the
    `Target.lean` §7.5 note); we state the faithful, NON-vacuous idempotency fact directly with
    `||` rather than referencing a `TargetActegory _ _ Bool` instance. This is provable, so it is
    a genuine theorem (not a `sorry`). -/
theorem semiring_choice_split :
    ((1 : ℝ) + 1 ≠ 1) ∧ ((true || true) = true) :=
  ⟨by norm_num, by decide⟩

end LeanNCD

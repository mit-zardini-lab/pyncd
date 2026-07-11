import Mathlib
import LeanNCD.Seam.Adapter
import LeanNCD.Base.St

namespace LeanNCD
open CategoryTheory MonoidalCategory

/-- A right `D`-actegory `V` parameterised by a value semiring `R` — the target of `construct()`.
    `actV` is the value-level lift (`P` acts by appending `R`-valued dimensions); the υ/α/δ
    coherences mirror §4 (`DGradedColoredPROP`), now transposed from the `ColoredPROP` `C` to the
    symmetric monoidal value category `V`: `ColoredPROP.tensor`/`unit`/`tensorHom` on `C` become
    `MonoidalCategory`'s `⊗`/`𝟙_ V`/`⊗` on `V`, and `act` becomes `actV`. The grading variable
    `Dᵒᵖ` is unchanged (it is still a `ColoredPROP`). -/
class TargetActegory (D : Type) (V : Type*) [ColoredPROP D] [Category V] [MonoidalCategory V]
    (R : Type) [CommSemiring R] where
  actV : (V × Dᵒᵖ) ⥤ V
  -- (Dist-⊗): the lift distributes over `⊗_V` (mirrors `DGradedColoredPROP.δ`).
  δ_V  : ∀ (X Y : V) (P : Dᵒᵖ),
           actV.obj (X ⊗ Y, P) ≅ (actV.obj (X, P) ⊗ actV.obj (Y, P))
  -- (Dist-I): the lift distributes over `𝟙_ V` (mirrors `DGradedColoredPROP.δ0`).
  δ0_V : ∀ (P : Dᵒᵖ), actV.obj ((𝟙_ V), P) ≅ (𝟙_ V)
  -- (Act-unit): the unit grading `I_D` acts trivially (mirrors `DGradedColoredPROP.υ`).
  υ_V  : ∀ (X : V), actV.obj (X, Opposite.op (ColoredPROP.unit : D)) ≅ X
  -- (Act-assoc): two consecutive lifts compose by `⊗_D` on the grading (mirrors `…α`).
  α_V  : ∀ (X : V) (P Q : Dᵒᵖ),
           actV.obj (actV.obj (X, P), Q) ≅
             actV.obj (X, Opposite.op (ColoredPROP.tensor Q.unop P.unop))
  -- (Act-unit / Act-assoc): `V` is a right `D`-actegory. Mirrors `DGradedColoredPROP.act_unit_assoc`.
  --  • Triangle: reassociating a lift by the unit `P ⊗ I` and applying the unitor at the inner
  --    `I` agrees with the outer unitor (`eqToHom` bridges `I ⊗ P = P` on the `D`-grading).
  --  • Pentagon: the two reassociations of a triple lift `((X ⊛ P) ⊛ Q) ⊛ R` agree (`eqToHom`
  --    bridges `(R ⊗ Q) ⊗ P = R ⊗ (Q ⊗ P)` on the `D`-grading).
  act_unit_assoc_V :
    (∀ (X : V) (P : Dᵒᵖ),
        (α_V X P (Opposite.op (ColoredPROP.unit : D))).hom ≫
            eqToHom (by rw [ColoredPROP.tensor_unit_l, Opposite.op_unop])
          = (υ_V (actV.obj (X, P))).hom)
    ∧ (∀ (X : V) (P Q R : Dᵒᵖ),
        (α_V (actV.obj (X, P)) Q R).hom ≫ (α_V X P (Opposite.op (ColoredPROP.tensor R.unop Q.unop))).hom
          = actV.map (X := (actV.obj (actV.obj (X, P), Q), R))
                     (Y := (actV.obj (X, Opposite.op (ColoredPROP.tensor Q.unop P.unop)), R))
              ((α_V X P Q).hom, 𝟙 R)
            ≫ (α_V X (Opposite.op (ColoredPROP.tensor Q.unop P.unop)) R).hom
            ≫ eqToHom (by rw [ColoredPROP.tensor_assoc]))
  -- (υ-naturality): the unitor υ_V : (− ⊛ I_D) ≅ Id is natural in `V`. Mirrors `…υ_nat`.
  υ_nat_V : ∀ {X Y : V} (f : X ⟶ Y),
    actV.map (X := (X, Opposite.op (ColoredPROP.unit : D)))
             (Y := (Y, Opposite.op (ColoredPROP.unit : D))) (f, 𝟙 (Opposite.op (ColoredPROP.unit : D)))
        ≫ (υ_V Y).hom
      = (υ_V X).hom ≫ f
  -- (Dist-nat / Dist-coh): the distributors are natural. Mirrors `DGradedColoredPROP.dist_coh`.
  --  • `δ_V`-naturality in the `V`-variable (in both tensor factors `f`, `g`).
  --  • `δ0_V`-naturality in the `D`-variable (for `g : P ⟶ Q` in `Dᵒᵖ`).
  dist_coh_V :
    (∀ {X X' Y Y' : V} (f : X ⟶ X') (g : Y ⟶ Y') (P : Dᵒᵖ),
        actV.map (X := (X ⊗ Y, P)) (Y := (X' ⊗ Y', P)) (MonoidalCategory.tensorHom f g, 𝟙 P)
            ≫ (δ_V X' Y' P).hom
          = (δ_V X Y P).hom
            ≫ MonoidalCategory.tensorHom
                (actV.map (X := (X, P)) (Y := (X', P)) (f, 𝟙 P))
                (actV.map (X := (Y, P)) (Y := (Y', P)) (g, 𝟙 P)))
    ∧ (∀ {P Q : Dᵒᵖ} (g : P ⟶ Q),
        actV.map (X := ((𝟙_ V), P)) (Y := ((𝟙_ V), Q)) (𝟙 (𝟙_ V), g)
            ≫ (δ0_V Q).hom
          = (δ0_V P).hom ≫ 𝟙 (𝟙_ V))

/-- The default target actegory: finitely-generated (finite-dimensional) `R`-modules (Mathlib). -/
abbrev Mat (R : Type) [CommRing R] := FGModuleCat R

-- RECORDED OBSTRUCTION (SORRY_INVENTORY Milestone H): NO `TargetActegory StObj (Mat ℝ) ℝ` instance
-- is provided (was: 8 `sorry` fields) —
-- a faithful dimension-adding `actV` is mathematically impossible over `FGModuleCat ℝ`. Two walls:
-- (1) axis sizes are symbolic (`Numeric = MvPolynomial String ℕ`), so there is no finite-dim module
--     of dimension `|P|`; (2) `δ_V` forces `dim∘actV` multiplicative in the V-variable, i.e.
--     `f(P) = f(P)² ⟹ f(P) = 1`, so any `δ_V`-respecting lift PRESERVES dimension. And `actV = id`
--     is inconsistent with the intended evaluator `F` (equivariance `F(X⊛P) ≅ actV(F X, P)` would
--     need `dim F(X⊛P) = dim F(X)`, false). A faithful ℝ-valued actegory needs CONCRETE (Nat) sizes
--     (the Milestone I `DenseTensor` regime), not the symbolic math tower. Deferred by design.

/-! ## §7.5 — The `R = Bool` predicate target (deferred formalization obligation)

The `TargetActegory` class above is parameterised by the value semiring `R : CommSemiring`
(line 15), and §7.5 turns precisely on that parameter:

  • `R = ℝ`   ⇒ contraction is (`×`, then `Σ`) — the *tensor* / linear-algebra reading,
                 realised by the `matTargetActegory` instance over `Mat ℝ = FGModuleCat ℝ`.
  • `R = Bool` ⇒ contraction is (`∧`, then `∃`) — the *predicate* / relational reading.

The "∃/∧-vs-Σ/× split is exactly the choice of `R`" (the proposition Task 6 will state as
`semiring_choice_split`): the same actegory skeleton instantiated at a different value
semiring switches sum-of-products contraction for exists-of-conjunctions contraction.

**The wrinkle (semantics, not typechecking).** The predicate reading needs the `(∨, ∧)` Boolean
*semiring* — addition `∨` (= `∃`), multiplication `∧`. The subtlety is that Mathlib's `Bool`
*type* does NOT carry that semiring as its default arithmetic: `Bool` is the Boolean *ring*
(`+` = XOR, `*` = `∧`; `instance : CommRing Bool := BooleanRing.toCommRing`), so `true + true =
false` and `Mat Bool = FGModuleCat Bool` *does* elaborate — but it computes over the WRONG
(XOR) addition, not `∨`/`∃`. The `(∨, ∧)` semiring we actually want is genuinely *not a ring*
(`∨` has no additive inverse: no `b` with `true ∨ b = false`), so it is a `CommSemiring` but
not a `CommRing` — and it is NOT the algebra `FGModuleCat` would put on `Bool`. So the obstacle
is not "`Mat Bool` fails to typecheck" (it doesn't fail); it is that reusing `Mat`/`Bool`'s
default ring gives XOR-modules, the wrong semantics for `(∧, ∃)` contraction.

**The obligation.** The `R = Bool` case requires a value category `V` carrying the `(∨, ∧)`
semiring structure — a `Bool`-*semimodule* / relations target realizing (`∧`, `∃`): e.g. the
category of finite sets and *relations* (`Rel`), or finitely-generated join-semilattice
semimodules (= finite powersets), whose hom-sets are Boolean matrices and whose composition is
(`∧`-then-`∃`) Boolean matrix multiply. Such a `V` carries `Category V`, a symmetric
`MonoidalCategory V`, and a `TargetActegory StObj V Bool` (with `R = Bool` AS THE `(∨,∧)`
semiring, e.g. via a `Tropical`/lattice wrapper so the `CommSemiring Bool` in scope is `(∨, ∧)`
rather than XOR) whose `actV` appends `Bool`-typed dimensions. We do not construct `V` here: a
faithful relations / `Bool`-semimodule category with the full υ/α/δ/triangle/pentagon/naturality
coherences — over the `(∨, ∧)` (not XOR) semiring — is a substantial development, and reusing
`FGModuleCat Bool` would misrepresent the structure (XOR, not `∃`). This is therefore recorded
as a deferred formalization obligation; the §7.5 split is witnessed for now by the idempotency
proxy `semiring_choice_split` (Task 6), which uses `∨`/`||` directly rather than `Bool`'s XOR `+`. -/

end LeanNCD

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

noncomputable instance : TargetActegory StObj (Mat ℝ) ℝ where
  actV := sorry  -- SIGNATURE: appends ℝ-typed dimensions; composition = matrix multiply over ℝ
  δ_V := sorry
  δ0_V := sorry
  υ_V := sorry
  α_V := sorry
  act_unit_assoc_V := sorry
  υ_nat_V := sorry
  dist_coh_V := sorry

/-! ## §7.5 — The `R = Bool` predicate target (deferred formalization obligation)

The `TargetActegory` class above is parameterised by the value semiring `R : CommSemiring`
(line 15), and §7.5 turns precisely on that parameter:

  • `R = ℝ`   ⇒ contraction is (`×`, then `Σ`) — the *tensor* / linear-algebra reading,
                 realised by the `matTargetActegory` instance over `Mat ℝ = FGModuleCat ℝ`.
  • `R = Bool` ⇒ contraction is (`∧`, then `∃`) — the *predicate* / relational reading.

The "∃/∧-vs-Σ/× split is exactly the choice of `R`" (the proposition Task 6 will state as
`semiring_choice_split`): the same actegory skeleton instantiated at a different value
semiring switches sum-of-products contraction for exists-of-conjunctions contraction.

**The wrinkle.** Although the *class* `TargetActegory _ _ Bool` typechecks (`Bool` is a
`CommSemiring`: `(⊕, ⊗) = (∨, ∧)`, with `false`/`true` as `0`/`1`), the *default realization*
does NOT. `Mat R := FGModuleCat R` (line 68) requires `[CommRing R]`, because finitely-generated
modules need additive inverses on the scalars. `Bool` has none: `true` has no additive
inverse under `∨` (there is no `b` with `true ∨ b = false`), so `Bool` is a `CommSemiring`
but *not* a `CommRing`. Hence `Mat Bool = FGModuleCat Bool` does not elaborate, and the
predicate target cannot reuse the `R = ℝ` value category.

**The obligation.** The `R = Bool` case requires a *different* value category `V` — a
`Bool`-semimodule / relations target realizing (`∧`, `∃`): e.g. the category of finite
sets and *relations* (`Rel`), or finitely-generated `Bool`-semimodules (free join-semilattices
= finite powersets), whose hom-sets are Boolean matrices and whose composition is
(`∧`-then-`∃`) Boolean matrix multiply. Such a `V` carries `Category V`, a symmetric
`MonoidalCategory V`, and a `TargetActegory StObj V Bool` whose `actV` appends `Bool`-typed
dimensions. We do not construct `V` here: a faithful relations / `Bool`-semimodule category
with the full υ/α/δ/triangle/pentagon/naturality coherences is a substantial development,
and forcing `FGModuleCat Bool` (which fails to typecheck) or a degenerate stub would
misrepresent the structure. This is therefore recorded as a deferred formalization
obligation; `semiring_choice_split` (Task 6) is stated abstractly over the class, so it
applies to any such future `Bool`-target instance without depending on `Mat`. -/

end LeanNCD

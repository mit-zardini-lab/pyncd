import Mathlib
import LeanNCD.Seam.Adapter

namespace LeanNCD

open CategoryTheory

/-- Extend the shape map `sh : gen_C → D` to a monoid homomorphism `sh* : C → D` on objects,
    by folding `ColoredPROP.tensor` over the color list of `X`. Satisfies (Sh-⊗):
    `sh*(X ⊗ Y) = sh*(X) ⊗ sh*(Y)` (not needed yet). -/
def sh_star {C D : Type} [ColoredPROP D] [ColoredPROP C]
    (sh : ColoredPROP.gen (ob := C) → D) (X : C) : D :=
  ((ColoredPROP.toList X).map sh).foldr ColoredPROP.tensor ColoredPROP.unit

open CategoryTheory in
/-- The core: a `D`-graded colored PROP (graded_prop.md §3.1).

    Data: a shape map `sh` on generators; a right `D`-action `act : (C × Dᵒᵖ) ⥤ C` (the "lift");
    distributivity isos `δ`/`δ0` (lift preserves `⊗`/`I` in the `C`-variable); and right-actegory
    coherence isos `υ` (unit) and `α` (associativity).

    Laws (all genuine `Prop`s): `sh_act` (the lift shifts the shape by `P`); `act_unit_assoc`
    (the actegory triangle + pentagon); `dist_coh` (`δ`/`δ0` naturality); and `broadcast_gen`
    (every generator factors through a lift). Proofs are supplied by instances (Milestone C). -/
class DGradedColoredPROP (D C : Type) [ColoredPROP D] [ColoredPROP C] where
  sh    : ColoredPROP.gen (ob := C) → D
  act   : (C × Dᵒᵖ) ⥤ C
  δ     : ∀ (X Y : C) (P : Dᵒᵖ),
            act.obj (ColoredPROP.tensor X Y, P) ≅
              ColoredPROP.tensor (act.obj (X, P)) (act.obj (Y, P))
  δ0    : ∀ (P : Dᵒᵖ), act.obj ((ColoredPROP.unit : C), P) ≅ (ColoredPROP.unit : C)
  υ     : ∀ (X : C), act.obj (X, Opposite.op (ColoredPROP.unit : D)) ≅ X
  α     : ∀ (X : C) (P Q : Dᵒᵖ),
            act.obj (act.obj (X, P), Q) ≅
              act.obj (X, Opposite.op (ColoredPROP.tensor Q.unop P.unop))
  sh_act : ∀ (X : C) (P : Dᵒᵖ),
             sh_star sh (act.obj (X, P)) = ColoredPROP.tensor (sh_star sh X) P.unop
  -- (Act-unit / Act-assoc): `C` is a right `D`-actegory.
  --  • Triangle: reassociating a lift by the unit `P ⊗ I` and applying the unitor at the inner
  --    `I` agrees with the outer unitor (the `eqToHom` bridges `I ⊗ P = P`).
  --  • Pentagon: the two reassociations of a triple lift `((X ⊛ P) ⊛ Q) ⊛ R` agree (the `eqToHom`
  --    bridges `(R ⊗ Q) ⊗ P = R ⊗ (Q ⊗ P)`).
  act_unit_assoc :
    (∀ (X : C) (P : Dᵒᵖ),
        (α X P (Opposite.op (ColoredPROP.unit : D))).hom ≫
            eqToHom (by rw [ColoredPROP.tensor_unit_l, Opposite.op_unop])
          = (υ (act.obj (X, P))).hom)
    ∧ (∀ (X : C) (P Q R : Dᵒᵖ),
        (α (act.obj (X, P)) Q R).hom ≫ (α X P (Opposite.op (ColoredPROP.tensor R.unop Q.unop))).hom
          = act.map (X := (act.obj (act.obj (X, P), Q), R))
                    (Y := (act.obj (X, Opposite.op (ColoredPROP.tensor Q.unop P.unop)), R))
              ((α X P Q).hom, 𝟙 R)
            ≫ (α X (Opposite.op (ColoredPROP.tensor Q.unop P.unop)) R).hom
            ≫ eqToHom (by rw [ColoredPROP.tensor_assoc]))
  -- (Dist-nat / Dist-coh): the distributors are natural.
  --  • `δ`-naturality in the `C`-variable (in both tensor factors `f`, `g`).
  --  • `δ0`-naturality in the `D`-variable (for `g : P ⟶ Q` in `Dᵒᵖ`).
  dist_coh :
    (∀ {X X' Y Y' : C} (f : SmallCategory.hom X X') (g : SmallCategory.hom Y Y') (P : Dᵒᵖ),
        act.map (X := (ColoredPROP.tensor X Y, P)) (Y := (ColoredPROP.tensor X' Y', P))
              (ColoredPROP.tensorHom f g, 𝟙 P)
            ≫ (δ X' Y' P).hom
          = (δ X Y P).hom
            ≫ ColoredPROP.tensorHom (act.map (X := (X, P)) (Y := (X', P)) (f, 𝟙 P))
                                    (act.map (X := (Y, P)) (Y := (Y', P)) (g, 𝟙 P)))
    ∧ (∀ {P Q : Dᵒᵖ} (g : P ⟶ Q),
        act.map (X := ((ColoredPROP.unit : C), P)) (Y := ((ColoredPROP.unit : C), Q))
              (SmallCategory.id (ColoredPROP.unit : C), g)
            ≫ (δ0 Q).hom
          = (δ0 P).hom ≫ SmallCategory.id (ColoredPROP.unit : C))
  broadcast_gen :
    ∀ {X Y : C} (g : SmallCategory.hom X Y),
      ∃ (X' Y' : C) (f : SmallCategory.hom X' Y') (P : Dᵒᵖ)
        (lam : SmallCategory.hom X (act.obj (X', P)))
        (ρ : SmallCategory.hom (act.obj (Y', P)) Y),
        g = SmallCategory.comp (SmallCategory.comp lam (act.map (f, 𝟙 P))) ρ

end LeanNCD

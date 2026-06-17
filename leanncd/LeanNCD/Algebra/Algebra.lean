import Mathlib
import LeanNCD.Algebra.Target
import LeanNCD.Core.Graded

namespace LeanNCD
open CategoryTheory

/-- The algebra functor `F : C → V` — a strong symmetric monoidal, `D`-equivariant functor into a
    target actegory (graded_prop.md Def 7.2); the categorical content of `construct()`. A `class`
    (not `structure`) so `ParaAlgebra` can `extend` it. -/
class Algebra (D C : Type) (V : Type*) [ColoredPROP D] [ColoredPROP C] [Category V] [MonoidalCategory V]
    [SymmetricCategory V]
    (R : Type) [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R] where
  F        : C ⥤ V
  /-- `F` is **strong symmetric monoidal** (graded_prop.md Def 7.2). `Functor.Braided F` bundles the
      strong-monoidal structure (`F.Monoidal`: μ/ε with invertible structure maps + pentagon/unitor
      coherences) together with `F.LaxBraided`'s braiding-compatibility law
      `μ X Y ≫ F.map (β_ X Y) = β_ (F X) (F Y) ≫ μ Y X`. `C` is symmetric monoidal via the Seam
      adapter (`instSymmetricOfColoredPROP`); `V` is symmetric monoidal by the class binder. -/
  Fbraided : F.Braided
  equivar  : ∀ (X : C) (P : Dᵒᵖ),
              F.obj (DGradedColoredPROP.act.obj (X, P))
                ≅ (TargetActegory.actV (D := D) (V := V) (R := R)).obj (F.obj X, P)
  -- (equivar-nat): `equivar` is natural in the `C`-variable `X` (at fixed grading `P`). The
  -- naturality square for the `D`-equivariance iso: `F` carries the lift-functoriality of `act`
  -- (in the `C`-factor `f`, identity on `P`) to that of `actV`. Mirrors that both `act` and `actV`
  -- are functors and `equivar` is a natural iso between `(F ∘ act⟨−,P⟩)` and `(actV⟨F −,P⟩)`.
  equivar_nat : ∀ {X Y : C} (f : X ⟶ Y) (P : Dᵒᵖ),
    F.map (DGradedColoredPROP.act.map (X := (X, P)) (Y := (Y, P)) (f, 𝟙 P))
        ≫ (equivar Y P).hom
      = (equivar X P).hom
        ≫ (TargetActegory.actV (D := D) (V := V) (R := R)).map
            (X := (F.obj X, P)) (Y := (F.obj Y, P)) (F.map f, 𝟙 P)
  -- (equivar-υ): `equivar` carries the `C`-unitor `υ` to the `V`-unitor `υ_V`. Acting by the unit
  -- grading `I_D` and applying the unitor downstairs in `V` agrees with applying it upstairs in `C`
  -- (via `F`) and transporting along `equivar`. Mirrors `DGradedColoredPROP.υ` ↦ `TargetActegory.υ_V`.
  equivar_υ : ∀ (X : C),
    F.map (DGradedColoredPROP.υ X).hom
      = (equivar X (Opposite.op (ColoredPROP.unit : D))).hom
        ≫ (TargetActegory.υ_V (D := D) (R := R) (F.obj X)).hom
  -- (equivar-α): `equivar` carries the `C`-associator `α` to the `V`-associator `α_V`. Reassociating
  -- two consecutive lifts (by `P` then `Q`) into one lift (by `Q ⊗ P`) commutes with `F` modulo the
  -- `equivar` isos at each nesting level. The left edge is `F (α X P Q)`; the right edge is `α_V`
  -- after stripping the two `equivar`s (outer at `Q`, inner at `P`). Mirrors `…α` ↦ `…α_V`.
  equivar_α : ∀ (X : C) (P Q : Dᵒᵖ),
    F.map (DGradedColoredPROP.α X P Q).hom
        ≫ (equivar X (Opposite.op (ColoredPROP.tensor Q.unop P.unop))).hom
      = (equivar (DGradedColoredPROP.act.obj (X, P)) Q).hom
        ≫ (TargetActegory.actV (D := D) (V := V) (R := R)).map
            (X := (F.obj (DGradedColoredPROP.act.obj (X, P)), Q))
            (Y := ((TargetActegory.actV (D := D) (V := V) (R := R)).obj (F.obj X, P), Q))
            ((equivar X P).hom, 𝟙 Q)
        ≫ (TargetActegory.α_V (D := D) (R := R) (F.obj X) P Q).hom
  -- (equivar-δ): `equivar` carries the `C`-distributor `δ` to the `V`-distributor `δ_V`, mediated by
  -- `F`'s strong-monoidal comparison `μ` (from `Fbraided.toMonoidal`). The C-side splits a lift of a
  -- tensor `(X ⊗ Y) ⊛ P` into `(X ⊛ P) ⊗ (Y ⊛ P)`; `F` carries this, `μIso⁻¹` (`= δ F`) breaks the
  -- `F`-image of each `⊗_C` into `⊗_V`, and the two `equivar`s land on `δ_V`. Mirrors
  -- `DGradedColoredPROP.δ` ↦ `TargetActegory.δ_V` (the (Dist-⊗) law). Both edges go
  -- `F.obj (act (X ⊗ Y, P)) ⟶ actV (F X, P) ⊗ actV (F Y, P)`.
  equivar_δ : ∀ (X Y : C) (P : Dᵒᵖ),
    haveI : F.Monoidal := Fbraided.toMonoidal
    -- left edge: split in `C`, push through `F`, break `F (− ⊗ −)` into `⊗_V`, then `equivar` each factor
    F.map (DGradedColoredPROP.δ X Y P).hom
        ≫ (Functor.Monoidal.μIso F (DGradedColoredPROP.act.obj (X, P)) (DGradedColoredPROP.act.obj (Y, P))).inv
        ≫ MonoidalCategory.tensorHom (equivar X P).hom (equivar Y P).hom
      = -- right edge: transport the whole lift to `V` via `equivar`, re-express the grading object
        -- via `actV.map (μIso⁻¹, 𝟙)`, then split with `δ_V`
        (equivar (ColoredPROP.tensor X Y) P).hom
        ≫ (TargetActegory.actV (D := D) (V := V) (R := R)).map
            (X := (F.obj (ColoredPROP.tensor X Y), P))
            (Y := (MonoidalCategory.tensorObj (F.obj X) (F.obj Y), P))
            ((Functor.Monoidal.μIso F X Y).inv, 𝟙 P)
        ≫ (TargetActegory.δ_V (D := D) (R := R) (F.obj X) (F.obj Y) P).hom
  -- (F-ev_p): `F` preserves the §4.1 evaluation `ev_p` (the slice of the lift at a point
  -- `p : P ⟶ I_D`). Upstairs `ev_p p X : act.obj (X, P) ⟶ X`; downstairs the corresponding
  -- `V`-evaluation is `actV.map (𝟙, p)` post-composed with `υ_V`. `equivar` mediates the `act`-image.
  -- Mirrors `ev_p` (Graded.lean) under `F`.
  F_ev_p : ∀ {P : Dᵒᵖ} (p : P ⟶ (Opposite.op (ColoredPROP.unit : D))) (X : C),
    F.map (ev_p p X)
      = (equivar X P).hom
        ≫ (TargetActegory.actV (D := D) (V := V) (R := R)).map
            (X := (F.obj X, P)) (Y := (F.obj X, Opposite.op (ColoredPROP.unit : D))) (𝟙 (F.obj X), p)
        ≫ (TargetActegory.υ_V (D := D) (R := R) (F.obj X)).hom

/-- The `Para` refinement: the `Para(C) → Para(V)` 2-functor + weight tying as a reparameterization
    2-cell (§7.5 / graded_prop.md §7 / the §11 "lightweight-Para" note).

    `Para(C)` is the parametric category over the monoidal `C`: a morphism `X ⟶ Y` is a PAIR
    `(P : C, f : P ⊗ X ⟶ Y)` — a parameter object `P` plus a map. The algebra functor `F : C ⥤ V`
    induces a 2-functor `Para(C) → Para(V)` sending the parametric morphism `(P, f)` to
    `(F P, paraMap P f)`, where the `V`-side map `F P ⊗ F X ⟶ F Y` is the `F`-image of `f` mediated
    by `F`'s lax-monoidal comparison `μ`. **Weight tying** is a reparameterization 2-cell: a map of
    parameter objects `Δ : P' ⟶ P` (collapsing tied weights) induces the corresponding 2-cell.

    Kept LIGHTWEIGHT: we state the action-on-1-cells map + its `μ`-mediated defining law + the
    reparameterization-2-cell (weight-tying) law as obligations; no double-category / 2-category
    typeclass machinery. -/
class ParaAlgebra (D C : Type) (V : Type*) [ColoredPROP D] [ColoredPROP C] [Category V] [MonoidalCategory V]
    [SymmetricCategory V]
    (R : Type) [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R]
    extends Algebra D C V R where
  /-- The `Para(C) → Para(V)` action on parametric morphisms: a parameter `P : C` and a map
      `f : P ⊗ X ⟶ Y` lift to parameter `F.obj P` and a `V`-map `F.obj P ⊗ F.obj X ⟶ F.obj Y`
      (built from `F.map f` mediated by the lax-monoidal `μ`). This is the 2-functor's
      action on 1-cells. -/
  paraMap : ∀ {X Y : C} (P : C) (_f : MonoidalCategory.tensorObj P X ⟶ Y),
              MonoidalCategory.tensorObj (toAlgebra.F.obj P) (toAlgebra.F.obj X) ⟶ toAlgebra.F.obj Y
  /-- `paraMap` is the `F`-image mediated by `μ`: it factors as the lax comparison
      `μ : F P ⊗ F X ⟶ F (P ⊗ X)` followed by `F.map f`. (The 2-functor's action-on-1-cells
      coherence.) `F.LaxMonoidal` comes from `Fbraided.toMonoidal`. -/
  paraMap_eq : ∀ {X Y : C} (P : C) (f : MonoidalCategory.tensorObj P X ⟶ Y),
                 haveI : toAlgebra.F.Monoidal := toAlgebra.Fbraided.toMonoidal
                 paraMap P f
                   = Functor.LaxMonoidal.μ toAlgebra.F P X ≫ toAlgebra.F.map f
  /-- Weight tying as a reparameterization 2-cell: a map of parameter objects `Δ : P' ⟶ P`
      (collapsing tied weights) induces the corresponding 2-cell on parametric morphisms.
      Precomposing the parameter by `Δ` upstairs (`(Δ ⊗ 𝟙 X) ≫ f`) corresponds downstairs to
      precomposing the `V`-side parameter by `F.map Δ`. Genuine reparameterization-2-cell law. -/
  weightTie : ∀ {X Y : C} {P' P : C} (Δ : P' ⟶ P) (f : MonoidalCategory.tensorObj P X ⟶ Y),
                paraMap P' (MonoidalCategory.tensorHom Δ (𝟙 X) ≫ f)
                  = MonoidalCategory.tensorHom (toAlgebra.F.map Δ) (𝟙 (toAlgebra.F.obj X))
                      ≫ paraMap P f

end LeanNCD

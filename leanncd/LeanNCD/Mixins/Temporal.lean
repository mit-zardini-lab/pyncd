import Mathlib
import LeanNCD.Core.Graded

namespace LeanNCD
open CategoryTheory

/-- `TemporalGraded` — the Scan mixin (graded_prop.md §3.4): a temporal object `L` carrying the
    augmented-simplex / `(ℕ,+,0)`-length structure, a directed family of prefix objects `[0..m]`
    with prefix inclusions `ιₘ : [0..m] ↪ L`, a directed restriction action, finite iteration
    (cata), a state-history trace, and lift–fold distributivity.

    The prefix inclusions are made first-class as a *directed* family: `prefix : ℕ → D` are the
    prefix objects `[0..m]`, `iotaTo m n` are the prefix-order maps `[0..m] ↪ [0..n]` (for `m ≤ n`),
    and `iota m : [0..m] ↪ L` includes each prefix into the temporal object. The compatibility laws
    `iotaTo_id`/`iotaTo_comp` make `prefix` a functor `(ℕ,≤) ⥤ D`, and `iota_factor` makes `L` a
    cocone over it (the genuine prefix-order law that replaces the doc's ill-typed `ι₀ = 𝟙`). -/
class TemporalGraded (D C : Type) [ColoredPROP D] [ColoredPROP C]
    extends DGradedColoredPROP D C where
  /-- The temporal object `L = [0..∞]` (augmented simplex / `length` grading). -/
  L          : D
  /-- The prefix objects `[0..m] ∈ Ob D`. -/
  «prefix»   : ℕ → D
  /-- The prefix-order maps `[0..m] ↪ [0..n]` for `m ≤ n`. -/
  iotaTo     : ∀ {m n : ℕ}, m ≤ n → SmallCategory.hom («prefix» m) («prefix» n)
  /-- The prefix inclusions `ιₘ : [0..m] ↪ L`. -/
  iota       : ∀ m : ℕ, SmallCategory.hom («prefix» m) L
  /-- Directed restriction `(− ⊛ [0..n]) ⟶ (− ⊛ [0..m])` along `ιₘ` (for `m ≤ n`). -/
  restrict   : ∀ {m n : ℕ}, m ≤ n → ∀ (X : C),
                 SmallCategory.hom (DGradedColoredPROP.act.obj (X, Opposite.op («prefix» n)))
                                   (DGradedColoredPROP.act.obj (X, Opposite.op («prefix» m)))
  /-- Finite iteration: the `N`-fold iterate of a step, landing in the state lifted over `[0..N]`. -/
  iterate    : ∀ (N : ℕ) (X : C) (_step : SmallCategory.hom X X),
                 SmallCategory.hom X (DGradedColoredPROP.act.obj (X, Opposite.op («prefix» N)))
  /-- The state-history trace (scanl) of the `N`-fold iterate. -/
  trace      : ∀ (N : ℕ) (X : C) (_step : SmallCategory.hom X X),
                 SmallCategory.hom X (DGradedColoredPROP.act.obj (X, Opposite.op («prefix» N)))
  /-- Lift–fold distributivity: an orthogonal degree `P` distributes through the fold. -/
  lift_fold_dist : ∀ (_N : ℕ) (X : C) (P : Dᵒᵖ),
                     DGradedColoredPROP.act.obj
                         (DGradedColoredPROP.act.obj (X, Opposite.op L), P)
                       ≅ DGradedColoredPROP.act.obj (X, Opposite.op L)
  -- Prefix-order compatibility (genuine, well-typed). `prefix`/`iotaTo` form a functor `(ℕ,≤) ⥤ D`
  -- and `iota` exhibits `L` as a cocone over it:
  /-- Reflexivity: the prefix map at `m ≤ m` is the identity. -/
  iotaTo_id   : ∀ (m : ℕ), iotaTo (le_refl m) = SmallCategory.id («prefix» m)
  /-- Transitivity: prefix maps compose (`[0..m] → [0..n] → [0..k]`). -/
  iotaTo_comp : ∀ {m n k : ℕ} (hmn : m ≤ n) (hnk : n ≤ k),
                  SmallCategory.comp (iotaTo hmn) (iotaTo hnk) = iotaTo (le_trans hmn hnk)
  /-- Cocone law: each inclusion `ιₘ` factors through the larger prefix, `ιₘ = (m↪n) ; ι_n`. -/
  iota_factor : ∀ {m n : ℕ} (hmn : m ≤ n),
                  SmallCategory.comp (iotaTo hmn) (iota n) = iota m

end LeanNCD

import Mathlib
import LeanNCD.Base.St

namespace LeanNCD

inductive DType
  | reals
  | nat : Numeric → DType

structure ArrayType where
  dtype : DType
  shape : StObj                     -- shape lives in Ob(St)

abbrev BrObj := List ArrayType      -- a product of arrays

inductive WeaveSlot
  | fixed : Axis → WeaveSlot        -- retained axis
  | tiled : WeaveSlot               -- contracted axis

abbrev WeaveShape := List WeaveSlot

def WeaveShape.targetAxes (w : WeaveShape) : StObj :=
  w.filterMap fun s => match s with | .fixed a => some a | .tiled => none

/-- A single base morphism of Br: one base operation with its reindexings (from St),
    input weaves, and output weaves. The `reindexings` field is where St lives inside Br. -/
structure BrBase (dom cod : BrObj) where
  op           : String
  degree       : StObj
  inputWeaves  : Fin dom.length → WeaveShape
  outputWeaves : Fin cod.length → WeaveShape
  reindexings  : ∀ i : Fin dom.length, StMat degree (inputWeaves i).targetAxes

/-! ## Br as a free *strict* symmetric monoidal category on `BrBase` generators.

A free list of `BrBase` operations — the natural candidate for representing a Br morphism as a
sequential pipeline of generators — fails on two fronts. First, it is 1-dimensional: parallel
composition `f ⊗ g` has no first-class constructor, so `tensorHom` must be faked and
`tensorHom_comp` (interchange) becomes unprovable. Second, even with `swap` as a list element,
`swap a b ; swap b a` is a two-element list, never definitionally equal to `nil`/`id` —
`swap_swap` is unprovable by `noConfusion`. A single concrete record also fails: there is no
canonical form for a composite broadcast op (§2.3), unlike `St` whose morphisms are `StMat` records.

The fix: work with **raw term syntax quotiented by equations**. `Hom` is an inductive tree with
`comp`, `tensor`, and `braid` as first-class constructors. `Rel` is an inductive proposition on
pairs of `Hom` terms — the smallest congruence containing exactly the SMC laws the `ColoredPROP`
fields need (category laws, bifunctor laws, braid involution, naturality, hexagons). `BrMorph` is
the `Quotient` of `Hom` by `Rel`. Then `.comp (.braid a b) (.braid b a)` and `.id (a ++ b)` are
distinct `Hom` trees, but `Rel.braid_involution` makes them equal in the quotient, so `swap_swap`
holds by `Quot.sound`. `tensorHom` and `swap` descend through `Rel`, and
`swap_swap`/`tensorHom_comp`/`tensorHom_id` all hold by `Quot.sound`. -/

inductive Hom : BrObj → BrObj → Type
  | id     : (a : BrObj) → Hom a a
  | gen    : {a b : BrObj} → BrBase a b → Hom a b
  | comp   : {a b c : BrObj} → Hom a b → Hom b c → Hom a c
  | tensor : {a b c d : BrObj} → Hom a b → Hom c d → Hom (a ++ c) (b ++ d)
  | braid  : (a b : BrObj) → Hom (a ++ b) (b ++ a)

inductive Rel : {a b : BrObj} → Hom a b → Hom a b → Prop
  -- equivalence
  | refl  {a b} (f : Hom a b) : Rel f f
  | symm  {a b} {f g : Hom a b} : Rel f g → Rel g f
  | trans {a b} {f g h : Hom a b} : Rel f g → Rel g h → Rel f h
  -- congruence
  | comp_congr {a b c} {f f' : Hom a b} {g g' : Hom b c} :
      Rel f f' → Rel g g' → Rel (.comp f g) (.comp f' g')
  | tensor_congr {a b c d} {f f' : Hom a b} {g g' : Hom c d} :
      Rel f f' → Rel g g' → Rel (.tensor f g) (.tensor f' g')
  -- category laws
  | id_comp {a b} (f : Hom a b) : Rel (.comp (.id a) f) f
  | comp_id {a b} (f : Hom a b) : Rel (.comp f (.id b)) f
  | assoc {a b c d} (f : Hom a b) (g : Hom b c) (h : Hom c d) :
      Rel (.comp (.comp f g) h) (.comp f (.comp g h))
  -- bifunctor laws
  | tensor_id (a c : BrObj) : Rel (.tensor (.id a) (.id c)) (.id (a ++ c))
  | interchange {a b c d e k} (f : Hom a b) (f' : Hom b c) (g : Hom d e) (g' : Hom e k) :
      Rel (.tensor (.comp f f') (.comp g g')) (.comp (.tensor f g) (.tensor f' g'))
  -- symmetry
  | braid_involution (a b : BrObj) : Rel (.comp (.braid a b) (.braid b a)) (.id (a ++ b))
  | braid_natural {a b c d} (f : Hom a b) (g : Hom c d) :
      Rel (.comp (.tensor f g) (.braid b d)) (.comp (.braid a c) (.tensor g f))
  -- strict left unit for `tensor` (same-type: `[] ++ a = a` definitionally)
  | tensor_unitl {a b} (f : Hom a b) : Rel (.tensor (.id []) f) f
  -- strict right unit for `tensor` (cross-type: `a ++ [] = a` is `List.append_nil`, not
  -- definitional, so we relate `tensor f (id [])` to the `cast` of `f` across the object equality).
  -- This is a true law of the free strict monoidal category.
  | tensor_unitr {a b} (f : Hom a b) :
      Rel (.tensor f (.id []))
        (cast (show Hom a b = Hom (a ++ []) (b ++ []) by rw [List.append_nil, List.append_nil]) f)
  -- strict associativity for `tensor` (cross-type: `(a₁ ++ a₂) ++ a₃ = a₁ ++ (a₂ ++ a₃)` is
  -- `List.append_assoc`). Relate the two bracketings of the tensor via the `cast` across the object
  -- equality. A true law of the free strict monoidal category.
  | tensor_assoc_coh {a₁ b₁ a₂ b₂ a₃ b₃}
      (f : Hom a₁ b₁) (g : Hom a₂ b₂) (h : Hom a₃ b₃) :
      Rel (.tensor (.tensor f g) h)
        (cast (show Hom (a₁ ++ (a₂ ++ a₃)) (b₁ ++ (b₂ ++ b₃))
                   = Hom ((a₁ ++ a₂) ++ a₃) ((b₁ ++ b₂) ++ b₃)
               by rw [List.append_assoc, List.append_assoc]) (.tensor f (.tensor g h)))
  -- hexagon identities for braid: the two SMC coherence axioms relating braid and the
  -- strict list-associativity. The casts transport `id` across `List.append_assoc` (and its
  -- symmetric) to produce the associator morphisms in the free strict SMC.
  | braid_hexagon_fwd (X Y Z : BrObj) :
      Rel (.comp (.comp (cast (congrArg (Hom ((X ++ Y) ++ Z)) (List.append_assoc X Y Z)) (.id _))
                       (.braid X (Y ++ Z)))
                 (cast (congrArg (Hom ((Y ++ Z) ++ X)) (List.append_assoc Y Z X)) (.id _)))
          (.comp (.comp (.tensor (.braid X Y) (.id Z))
                       (cast (congrArg (Hom ((Y ++ X) ++ Z)) (List.append_assoc Y X Z)) (.id _)))
                 (.tensor (.id Y) (.braid X Z)))
  | braid_hexagon_rev (X Y Z : BrObj) :
      Rel (.comp (.comp (cast (congrArg (Hom (X ++ (Y ++ Z))) (List.append_assoc X Y Z).symm) (.id _))
                       (.braid (X ++ Y) Z))
                 (cast (congrArg (Hom (Z ++ (X ++ Y))) (List.append_assoc Z X Y).symm) (.id _)))
          (.comp (.comp (.tensor (.id X) (.braid Y Z))
                       (cast (congrArg (Hom (X ++ (Z ++ Y))) (List.append_assoc X Z Y).symm) (.id _)))
                 (.tensor (.braid X Z) (.id Y)))

instance setoidHom (a b : BrObj) : Setoid (Hom a b) where
  r := Rel
  iseqv := ⟨Rel.refl, Rel.symm, Rel.trans⟩

/-- The free strict symmetric monoidal category on `BrBase`, as a quotient of the raw syntax. -/
abbrev BrMorph (a b : BrObj) : Type := Quotient (setoidHom a b)

namespace BrMorph

def mk {a b : BrObj} (f : Hom a b) : BrMorph a b := Quotient.mk _ f

def comp {a b c : BrObj} : BrMorph a b → BrMorph b c → BrMorph a c :=
  Quotient.lift₂ (fun f g => mk (.comp f g))
    (fun _ _ _ _ hf hg => Quotient.sound (Rel.comp_congr hf hg))

def tensor {a b c d : BrObj} : BrMorph a b → BrMorph c d → BrMorph (a ++ c) (b ++ d) :=
  Quotient.lift₂ (fun f g => mk (.tensor f g))
    (fun _ _ _ _ hf hg => Quotient.sound (Rel.tensor_congr hf hg))

def idm (a : BrObj) : BrMorph a a := mk (.id a)

def braid (a b : BrObj) : BrMorph (a ++ b) (b ++ a) := mk (.braid a b)

@[simp] theorem id_comp {a b} (f : BrMorph a b) : comp (idm a) f = f := by
  refine Quotient.inductionOn f (fun f => ?_); exact Quotient.sound (Rel.id_comp f)

@[simp] theorem comp_id {a b} (f : BrMorph a b) : comp f (idm b) = f := by
  refine Quotient.inductionOn f (fun f => ?_); exact Quotient.sound (Rel.comp_id f)

theorem assoc {a b c d} (f : BrMorph a b) (g : BrMorph b c) (h : BrMorph c d) :
    comp (comp f g) h = comp f (comp g h) := by
  refine Quotient.inductionOn₃ f g h (fun f g h => ?_); exact Quotient.sound (Rel.assoc f g h)

theorem tensor_id (a c : BrObj) : tensor (idm a) (idm c) = idm (a ++ c) :=
  Quotient.sound (Rel.tensor_id a c)

theorem tensor_comp {a b c d e k} (f : BrMorph a b) (f' : BrMorph b c)
    (g : BrMorph d e) (g' : BrMorph e k) :
    tensor (comp f f') (comp g g') = comp (tensor f g) (tensor f' g') := by
  refine Quotient.inductionOn₂ f f' (fun f f' => ?_)
  refine Quotient.inductionOn₂ g g' (fun g g' => ?_)
  exact Quotient.sound (Rel.interchange f f' g g')

theorem braid_braid (a b : BrObj) : comp (braid a b) (braid b a) = idm (a ++ b) :=
  Quotient.sound (Rel.braid_involution a b)

theorem braid_natural {a b c d} (f : BrMorph a b) (g : BrMorph c d) :
    comp (tensor f g) (braid b d) = comp (braid a c) (tensor g f) := by
  refine Quotient.inductionOn₂ f g (fun f g => ?_)
  exact Quotient.sound (Rel.braid_natural f g)

theorem tensor_unitl {a b} (f : BrMorph a b) : tensor (idm []) f = f := by
  refine Quotient.inductionOn f (fun f => ?_)
  exact Quotient.sound (Rel.tensor_unitl f)

end BrMorph

/-- Forward hexagon for `BrMorph.braid`, with GENERIC equality proofs as implicit arguments.
    Taking arbitrary `h₁ h₂ h₃` (vs. fixing `List.append_assoc`) means the instance field can
    supply `tensor_assoc` proofs directly — proof irrelevance bridges to `List.append_assoc`
    internally, so no `proof_irrel` boilerplate is needed at the call site. -/
theorem BrMorph.swap_hexagon_fwd {X Y Z : BrObj}
    {h₁ : (X ++ Y) ++ Z = X ++ (Y ++ Z)}
    {h₂ : (Y ++ Z) ++ X = Y ++ (Z ++ X)}
    {h₃ : (Y ++ X) ++ Z = Y ++ (X ++ Z)} :
    BrMorph.comp (BrMorph.comp
        (Quotient.mk _ (cast (congrArg (Hom ((X ++ Y) ++ Z)) h₁) (.id _)))
        (BrMorph.braid X (Y ++ Z)))
      (Quotient.mk _ (cast (congrArg (Hom ((Y ++ Z) ++ X)) h₂) (.id _))) =
    BrMorph.comp (BrMorph.comp
        (BrMorph.tensor (BrMorph.braid X Y) (BrMorph.idm Z))
        (Quotient.mk _ (cast (congrArg (Hom ((Y ++ X) ++ Z)) h₃) (.id _))))
      (BrMorph.tensor (BrMorph.idm Y) (BrMorph.braid X Z)) := by
  rw [show h₁ = List.append_assoc X Y Z from proof_irrel _ _,
      show h₂ = List.append_assoc Y Z X from proof_irrel _ _,
      show h₃ = List.append_assoc Y X Z from proof_irrel _ _]
  exact Quotient.sound (Rel.braid_hexagon_fwd X Y Z)

theorem BrMorph.swap_hexagon_rev {X Y Z : BrObj}
    {h₁ : X ++ (Y ++ Z) = (X ++ Y) ++ Z}
    {h₂ : Z ++ (X ++ Y) = (Z ++ X) ++ Y}
    {h₃ : X ++ (Z ++ Y) = (X ++ Z) ++ Y} :
    BrMorph.comp (BrMorph.comp
        (Quotient.mk _ (cast (congrArg (Hom (X ++ (Y ++ Z))) h₁) (.id _)))
        (BrMorph.braid (X ++ Y) Z))
      (Quotient.mk _ (cast (congrArg (Hom (Z ++ (X ++ Y))) h₂) (.id _))) =
    BrMorph.comp (BrMorph.comp
        (BrMorph.tensor (BrMorph.idm X) (BrMorph.braid Y Z))
        (Quotient.mk _ (cast (congrArg (Hom (X ++ (Z ++ Y))) h₃) (.id _))))
      (BrMorph.tensor (BrMorph.braid X Z) (BrMorph.idm Y)) := by
  rw [show h₁ = (List.append_assoc X Y Z).symm from proof_irrel _ _,
      show h₂ = (List.append_assoc Z X Y).symm from proof_irrel _ _,
      show h₃ = (List.append_assoc X Z Y).symm from proof_irrel _ _]
  exact Quotient.sound (Rel.braid_hexagon_rev X Y Z)

/-- A manufactured point `[] → X` (a `BrBase` with empty domain; always constructible). Used to
    separate parallel morphisms in `elemental`. -/
def brPoint (X : BrObj) : BrBase [] X where
  op := "pt"
  degree := []
  inputWeaves := fun i => nomatch i
  outputWeaves := fun _ => []
  reindexings := fun i => nomatch i

/-- The hard content of `elemental`, isolated on raw syntax: **left-cancellation of a point**.

    OBLIGATION (`sorry`) — the strict-SMC normal-form milestone. With the chosen `Rel`,
    `(Hom, comp, tensor, id, braid)` is the free strict *symmetric* monoidal category on the
    `BrBase` generators. This lemma is TRUE — `gen (brPoint X)` participates in no `Rel`
    constructor, so a leading point cannot be rewritten away — but proving it needs a
    `Rel`-respecting normal form on `Hom`.

    WHAT DOESN'T WORK — direct induction on the `Rel` derivation. The wall is the `trans`
    (congruence-closure) case, NOT `interchange`: `trans` introduces an unconstrained intermediate
    term, and chaining the induction hypotheses requires that term to ALSO be point-prefixed —
    which is the cancellation statement itself (circular). The empty domain of `brPoint X` only
    tames the *base* cases (e.g. `interchange`, whose `tensor`-headed LHS cannot be a point-prefixed
    `comp`); it does not shortcut the milestone.

    THE PLANNED ROUTE — NbE / initiality (skeleton validated 2026-06-21, scratch file). Interpret
    `Hom` into a concrete canonical model `N a b` of free-strict-SMC morphisms (typed string
    diagrams: a generator-node set + a color-preserving wiring bijection, equality up to graph iso),
    with `eval : Hom → N` and `quote : N → Hom`, then discharge `brCancelPoint` from three lemmas:
      * `sound`    : `Rel f g → eval f = eval g`   — dissolves the congruence closure (`trans`/
                     `comp_congr` become `Eq.trans`/`congrArg`).
      * `section_` : `Rel f (quote (eval f))`       — the gating bulk: any two sequentializations of
                     a node-set are `Rel`-equal (the `interchange` + braid-naturality content).
      * `eval_point_injective` — the empty-domain point is `N`'s unique input-less node, so deleting
                     it is well-defined and injective (this is where the empty domain earns its keep).
    In `N`, `interchange`/`∘`-assoc/unit are structural and ALL braid laws (involution, naturality,
    hexagon) become `Equiv` facts on the wiring. A several-hundred-line development (defining `N` is
    the gating bulk), deferred as its own milestone. -/
theorem brCancelPoint {X Y : BrObj} {f g : Hom X Y}
    (h : Rel (.comp (.gen (brPoint X)) f) (.comp (.gen (brPoint X)) g)) : Rel f g := by
  sorry

/-- `Quotient.mk` of a `cast` across object equalities is heterogeneously equal to the `Quotient.mk`
    of the original morphism: the quotient types `BrMorph a' b'` and `BrMorph a b` coincide once the
    object equalities are substituted, and `cast` is then the identity. Bridges the `cast`-`Rel`
    constructors (`tensor_unitr`/`tensor_assoc_coh`) to the `HEq` `ColoredPROP` fields. -/
private theorem brHEq_mk_cast {a b a' b' : BrObj} (f : Hom a b) (ha : a' = a) (hb : b' = b) :
    HEq (Quotient.mk (setoidHom a' b')
          (cast (show Hom a b = Hom a' b' by rw [ha, hb]) f))
        (Quotient.mk (setoidHom a b) f) := by
  subst ha hb; simp [cast_eq]

/-- Strict right unit for `tensor` on `BrMorph`, as an `HEq` across `a ++ [] = a`. Discharged from
    `Rel.tensor_unitr` (a true strict-monoidal law) via `brHEq_mk_cast`. -/
theorem BrMorph.tensor_unitr_heq {a b : BrObj} (f : BrMorph a b) :
    HEq (BrMorph.tensor f (BrMorph.idm [])) f := by
  refine Quotient.inductionOn f (fun f => ?_)
  have hrel : BrMorph.tensor (Quotient.mk _ f) (BrMorph.idm [])
      = Quotient.mk _ (cast (show Hom a b = Hom (a ++ []) (b ++ []) by
          rw [List.append_nil, List.append_nil]) f) := Quotient.sound (Rel.tensor_unitr f)
  rw [hrel]
  exact brHEq_mk_cast f (List.append_nil a) (List.append_nil b)

/-- Strict associativity for `tensor` on `BrMorph`, as an `HEq` across `List.append_assoc`.
    Discharged from `Rel.tensor_assoc_coh` (a true strict-monoidal law) via `brHEq_mk_cast`. -/
theorem BrMorph.tensor_assoc_heq {a₁ b₁ a₂ b₂ a₃ b₃ : BrObj}
    (f : BrMorph a₁ b₁) (g : BrMorph a₂ b₂) (h : BrMorph a₃ b₃) :
    HEq (BrMorph.tensor (BrMorph.tensor f g) h) (BrMorph.tensor f (BrMorph.tensor g h)) := by
  refine Quotient.inductionOn₃ f g h (fun f g h => ?_)
  have hrel : BrMorph.tensor (BrMorph.tensor (Quotient.mk _ f) (Quotient.mk _ g)) (Quotient.mk _ h)
      = Quotient.mk _ (cast (show Hom (a₁ ++ (a₂ ++ a₃)) (b₁ ++ (b₂ ++ b₃))
                   = Hom ((a₁ ++ a₂) ++ a₃) ((b₁ ++ b₂) ++ b₃)
               by rw [List.append_assoc, List.append_assoc])
              (.tensor f (.tensor g h))) :=
    Quotient.sound (Rel.tensor_assoc_coh f g h)
  rw [hrel]
  exact brHEq_mk_cast (.tensor f (.tensor g h))
    (List.append_assoc a₁ a₂ a₃) (List.append_assoc b₁ b₂ b₃)

/-- Casting `BrMorph.idm a` (= `Quotient.mk _ (Hom.id a)`) along an object equality `h : a = b`
    via the outer `Quotient`-type cast equals a `Quotient.mk` of the inner `Hom`-level cast.
    Bridge between the outer-cast form that `SmCat.coh h` unfolds to in `instance Br` and the
    `Quotient.mk` form used in `Rel.braid_hexagon_fwd/rev` constructors. No `SmallCategory`
    instance required; proved by `cases h; rfl` (both sides reduce to `BrMorph.idm a`). -/
private theorem cast_quot_id {a b : BrObj} (h : a = b) :
    cast (congrArg (BrMorph a) h) (BrMorph.idm a)
    = Quotient.mk (setoidHom a b) (cast (congrArg (Hom a) h) (Hom.id a)) := by
  cases h; rfl

private theorem Br_swap_hexagon_fwd (X Y Z : BrObj)
    (h₁ : (X ++ Y) ++ Z = X ++ (Y ++ Z))
    (h₂ : (Y ++ Z) ++ X = Y ++ (Z ++ X))
    (h₃ : (Y ++ X) ++ Z = Y ++ (X ++ Z)) :
    BrMorph.comp (BrMorph.comp
        (cast (congrArg (BrMorph ((X ++ Y) ++ Z)) h₁) (BrMorph.idm ((X ++ Y) ++ Z)))
        (BrMorph.braid X (Y ++ Z)))
      (cast (congrArg (BrMorph ((Y ++ Z) ++ X)) h₂) (BrMorph.idm ((Y ++ Z) ++ X)))
    = BrMorph.comp (BrMorph.comp
        (BrMorph.tensor (BrMorph.braid X Y) (BrMorph.idm Z))
        (cast (congrArg (BrMorph ((Y ++ X) ++ Z)) h₃) (BrMorph.idm ((Y ++ X) ++ Z))))
      (BrMorph.tensor (BrMorph.idm Y) (BrMorph.braid X Z)) := by
  rw [cast_quot_id h₁, cast_quot_id h₂, cast_quot_id h₃]
  exact @BrMorph.swap_hexagon_fwd X Y Z h₁ h₂ h₃

private theorem Br_swap_hexagon_rev (X Y Z : BrObj)
    (h₁ : X ++ (Y ++ Z) = (X ++ Y) ++ Z)
    (h₂ : Z ++ (X ++ Y) = (Z ++ X) ++ Y)
    (h₃ : X ++ (Z ++ Y) = (X ++ Z) ++ Y) :
    BrMorph.comp (BrMorph.comp
        (cast (congrArg (BrMorph (X ++ (Y ++ Z))) h₁) (BrMorph.idm (X ++ (Y ++ Z))))
        (BrMorph.braid (X ++ Y) Z))
      (cast (congrArg (BrMorph (Z ++ (X ++ Y))) h₂) (BrMorph.idm (Z ++ (X ++ Y))))
    = BrMorph.comp (BrMorph.comp
        (BrMorph.tensor (BrMorph.idm X) (BrMorph.braid Y Z))
        (cast (congrArg (BrMorph (X ++ (Z ++ Y))) h₃) (BrMorph.idm (X ++ (Z ++ Y)))))
      (BrMorph.tensor (BrMorph.braid X Z) (BrMorph.idm Y)) := by
  rw [cast_quot_id h₁, cast_quot_id h₂, cast_quot_id h₃]
  exact @BrMorph.swap_hexagon_rev X Y Z h₁ h₂ h₃

instance Br : ColoredPROP BrObj where
  gen    := ArrayType
  toList := id
  ofList := id
  hom    := BrMorph
  id     := BrMorph.idm
  comp   := BrMorph.comp
  id_comp := BrMorph.id_comp
  comp_id := BrMorph.comp_id
  assoc   := BrMorph.assoc
  tensor_assoc  := List.append_assoc
  tensor_unit_l := by intro a; simp
  tensor_unit_r := by intro a; simp [List.append_nil]
  swap a b := BrMorph.braid a b                  -- symmetry is now a first-class generator (§2.3)
  tensorHom {a b c d} f g := BrMorph.tensor f g  -- tensor is now a first-class generator
  tensorHom_id   := BrMorph.tensor_id            -- by Quot.sound (Rel.tensor_id)
  tensorHom_comp := fun f₁ f₂ g₁ g₂ => BrMorph.tensor_comp f₁ f₂ g₁ g₂  -- interchange, by Quot.sound
  swap_swap      := BrMorph.braid_braid          -- braid involution, by Quot.sound
  swap_natural   := fun f g => BrMorph.braid_natural f g  -- braid naturality, by Quot.sound
  swap_hexagon_fwd := fun X Y Z =>
    Br_swap_hexagon_fwd X Y Z
      (List.append_assoc X Y Z) (List.append_assoc Y Z X) (List.append_assoc Y X Z)
  swap_hexagon_rev := fun X Y Z =>
    Br_swap_hexagon_rev X Y Z
      (List.append_assoc X Y Z).symm (List.append_assoc Z X Y).symm (List.append_assoc X Z Y).symm
  tensorHom_assoc := fun f g h => BrMorph.tensor_assoc_heq f g h  -- cast-Rel, by Quot.sound + HEq
  tensorHom_unit_l := fun f => heq_of_eq (BrMorph.tensor_unitl f)
  tensorHom_unit_r := fun f => BrMorph.tensor_unitr_heq f         -- cast-Rel, by Quot.sound + HEq
  elemental := by
    -- Points separate quotient morphisms. The reduction to raw-syntax point-cancellation is
    -- sorry-free; ALL the hard content is isolated in `brCancelPoint` (the normal-form milestone).
    intro X Y f g h
    refine Quotient.inductionOn₂ f g (fun f g h => ?_) h
    have hpt := h (BrMorph.mk (.gen (brPoint X)))
    simp only [BrMorph.comp, BrMorph.mk, Quotient.lift₂_mk] at hpt
    exact Quotient.sound (brCancelPoint (Quotient.exact hpt))
end LeanNCD

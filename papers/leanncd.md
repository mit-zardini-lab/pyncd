# Lean 4 Encoding of the `D`-Graded Colored PROP Framework

This document describes a Lean 4 formalisation of the NCD categorical framework as a **single structure** — the `D`-graded colored PROP of [graded_prop.md](graded_prop.md) — rather than as two independent categories. `St` and `Br` are one instantiation (`D = St`, `C = Br`); the model level (`D = Br`) and the swapped-`D` rows are others. The encoding is a layered tower of typeclasses parameterised by other typeclasses, so the propositions of [graded_prop.md §8](graded_prop.md#8-propositions-the-synthesis-organizes) are stated once, generically, and inherited at every instantiation.

The intent is **formalizability, not formalization**: definitions are given as Lean `class`/`structure` data plus named `Prop`-field laws, in the shape a Lean development transcribes directly. No proofs are written — signatures and proof obligations only.

## Contents

1. [Orientation: one structure, one seam](#1-orientation-one-structure-one-seam)
2. [The base: `ColoredPROP`](#2-the-base-coloredprop)
3. [The seam adapter into Mathlib](#3-the-seam-adapter-into-mathlib)
4. [The core: `DGradedColoredPROP D C`](#4-the-core-dgradedcoloredprop-d-c)
5. [Weaves as cartesian-lift data](#5-weaves-as-cartesian-lift-data)
6. [Mixins: Scan, Route, Symmetry, Para](#6-mixins-scan-route-symmetry-para)
7. [The Grothendieck construction and executable seam](#7-the-grothendieck-construction-and-executable-seam)
8. [Algebras and `construct()`](#8-algebras-and-construct)
9. [Acsets and the executable layer](#9-acsets-and-the-executable-layer)
10. [The propositions as generic theorems](#10-the-propositions-as-generic-theorems)
11. [Instantiation and future extensions](#11-instantiation-and-future-extensions)
12. [Lean formalization notes](#12-lean-formalization-notes)
13. [The tensor-logic DSL](#13-the-tensor-logic-dsl)
14. [Appendix: out of scope](#14-appendix-out-of-scope)

## 1. Orientation: one structure, one seam

The framework is a single parametric development organized around **one structure** — a `D`-graded colored PROP, parameterised by an index PROP `D` and an operation PROP `C`. What might appear to be bookkeeping (UIDs, `Context`, names) is ordinary categorical data: symbolic sizes are the **fiber of the Grothendieck construction** `∫Dat`, and axis identity/alignment is the **pushout/coequalizer** of composition. Both live inside this single development. "Prove the propositions once, inherit them everywhere" becomes, in Lean, parametricity over a typeclass.

The encoding is therefore a layered tower of typeclasses, each parameterised by the classes below it. The following diagram makes that structure explicit. Read top-to-bottom as dependency order; `[X]` is a Lean typeclass constraint; `├`/`└` are sibling mixins on the same parent class; `⇣ … →` is a lateral bridge to Mathlib (the [§3](#3-the-seam-adapter-into-mathlib) seam adapter, not a new layer); `(mixin, full)` means fully specified here, `(mixin, STUB)` declared with body deferred. The three layers and their mixin branches:

```text
ColoredPROP O                                    -- lightweight base; St, Br instances
   ⇣ adapter (the seam)  →  Mathlib MonoidalCategory / SymmetricCategory
DGradedColoredPROP D C   [ColoredPROP D] [ColoredPROP C]   -- core: sh, act, δ, υ, α, axioms
   ├ TemporalGraded   D C   (mixin, full)   -- Scan, Def 3.3–3.5
   ├ RouteStructure   D C   (mixin, STUB)     -- Route, Prop 8.6(ii)      [future_ideas]
   └ SymmetryGraded   D C T (mixin, STUB)     -- equivariance monad, Prop 8.4 [gated; equiv_unif A3]
Algebra D C V   [DGradedColoredPROP D C] [TargetActegory D V]   -- construct()
   └ ParaAlgebra ...        (mixin, STUB)      -- weight tying, pass-as-2-cell  [prop_ideas: Para Refinement]
```

The base `ColoredPROP` carries the lightweight definitions and the `St`/`Br` instances; the core `DGradedColoredPROP D C` adds the grading data and laws; capabilities (`Scan`, `Route`, symmetry, `Para`) are composable mixins, and `Algebra` is the `construct()` functor into a target actegory. An instantiation pays only for the layers it declares, and the proven core is never edited — new domains are new instances, new capabilities are new mixins.

The organizing seam is the **proposition/computation** seam: **does Lean *prove* this or *compute* this?** This is the seam [graded_prop.md](graded_prop.md) itself draws. [§6](graded_prop.md#6-composition-as-pushout) is "a correctness/specification lens, not a composition algorithm": the pushout *explains and certifies* what `Context` does, it does not replace it — so the coequalizer is the *specification* and union-find plus a fresh-name counter is the *implementation*. [§7](graded_prop.md#7-algebras-construct-and-the-para-refinement) keeps the lightweight `Para` encoding and tells us to **note the gap explicitly rather than pay for a bicategory only the specification uses**. Read as the organizing principle of this document: the propositional core (PROP/actegory laws, `∫Dat`, pushout-as-colimit, equivariance, weave uniqueness) is stated over UID-free types and proved; the executable realization (fresh-UID counter, union-find, acset tables / CSV) sits on the other side of the seam, realizing the specification without being proved against it line by line. The seam adapter of [§3](#3-the-seam-adapter-into-mathlib) is exactly this boundary turned into a definition.

## 2. The base: `ColoredPROP`

Categories are encoded, following [Holtzen (2025)](https://sholtzen.dev/articles/leancat-1.html), as a Lean 4 typeclass parameterised by an object type `ob : Type`. This is the categorical skeleton on which everything else rests.

```lean
class SmallCategory (ob : Type) : Type 1 where
  hom     : ob → ob → Type
  id      : ∀ x, hom x x
  comp    : ∀ {X Y Z}, hom X Y → hom Y Z → hom X Z
  id_comp : ∀ {X Y} (f : hom X Y), comp (id X) f = f
  comp_id : ∀ {X Y} (f : hom X Y), comp f (id Y) = f
  assoc   : ∀ {W X Y Z} (f : hom W X) (g : hom X Y) (h : hom Y Z),
              comp (comp f g) h = comp f (comp g h)

-- Scoped to avoid conflict with CategoryTheory's ⟶ and Function.comp.
-- Open `ColoredPROPNotations` in files that use these; files importing Mathlib
-- use CategoryTheory's ⟶ directly via the §3 seam adapter.
-- NOTE: the `scoped[NS] notation …` attribute one-liner is a Mathlib extension; it does not
-- parse in a file with no Mathlib import. The base category file is deliberately Mathlib-free,
-- so it uses the equivalent core-Lean form (a `namespace`-delimited `scoped`):
namespace ColoredPROPNotations
scoped infixl:65 " ⟶ " => SmallCategory.hom
scoped notation:65 a " ∘ " b => SmallCategory.comp b a
end ColoredPROPNotations
```

Objects are themselves Lean types, so the monoidal product on objects can be definitional list concatenation; morphisms carry enough structure that the category laws fall to ring axioms (`St`) or list induction (`Br`) with no quotient; and the laws `id_comp`/`comp_id`/`assoc` are propositional equalities discharged by tactics.

Both **St** and **Br** are *colored PROPs* ([graded_prop.md §2](graded_prop.md), Definition 2.1): symmetric strict monoidal categories whose object monoid is the free monoid `O*` over a set of colors, with `⊗` = list concatenation and `I` = the empty list. The base class carries exactly this structure, including the elemental separation axiom `elemental` (the `(Elem)` axiom of [graded_prop.md §2](graded_prop.md)).

```lean
class ColoredPROP (ob : Type) extends SmallCategory ob where
  gen       : Type          -- O, the color set; ob = List gen = O* (the free monoid on O)
  toList    : ob → List gen
  ofList    : List gen → ob
  tensor    : ob → ob → ob := fun a b => ofList (toList a ++ toList b)
  unit      : ob := ofList []
  tensor_assoc  : ∀ a b c, tensor (tensor a b) c = tensor a (tensor b c)
  tensor_unit_l : ∀ a, tensor unit a = a
  tensor_unit_r : ∀ a, tensor a unit = a
  swap      : ∀ a b, hom (tensor a b) (tensor b a)
  tensorHom : ∀ {a b c d}, hom a b → hom c d → hom (tensor a c) (tensor b d)
  -- morphism-level symmetric-monoidal laws (tensorHom is a bifunctor; swap is an involution):
  tensorHom_id   : ∀ a c, tensorHom (id a) (id c) = id (tensor a c)
  tensorHom_comp : ∀ {a b c d e g} (f₁ : hom a b) (f₂ : hom b c) (g₁ : hom d e) (g₂ : hom e g),
                     tensorHom (comp f₁ f₂) (comp g₁ g₂) = comp (tensorHom f₁ g₁) (tensorHom f₂ g₂)
  swap_swap      : ∀ a b, comp (swap a b) (swap b a) = id (tensor a b)
  elemental : ∀ {X Y} (f g : hom X Y),
                (∀ x : hom unit X, comp x f = comp x g) → f = g   -- (Elem)
```

For both **St** and **Br**, `ob = List gen`, so `toList = id` and `ofList = id`, and the three strictness laws `tensor_assoc`/`tensor_unit_l`/`tensor_unit_r` reduce to `List.append_assoc`, `List.nil_append`, and `List.append_nil` respectively — all dischargeable by `simp [List.append_assoc/nil_append/append_nil]`. The `swap` morphism is a rearrangement that interleaves or separates the two sub-lists, and `tensorHom` is the parallel product. The morphism-level laws `tensorHom_id`/`tensorHom_comp` (bifunctoriality of the parallel product) and `swap_swap` (the symmetry is an involution) complete the symmetric-monoidal structure; they are what let the §3 seam exhibit the bifunctorial part of a Mathlib `MonoidalCategory` sorry-free (see [§12](#12-lean-formalization-notes)).

The `elemental` field: writing `El(X) := hom unit X` for the *points* (global elements) of `X`, it states that points separate parallel morphisms — `(∀ x : hom unit X, x ; f = x ; g) → f = g`. Equivalently the family `{(x ; −)}_{x ∈ El(X)}` is jointly injective. This is a single `Prop`-valued field, and it is precisely what makes the cartesian-lift datum of a morphism **unique** — see weave uniqueness (forward-ref [§5](#5-weaves-as-cartesian-lift-data), Prop 8.2). Without it, Eq. 3 (point-naturality) holds by functoriality but a weave's `(f, P, ρ)` factorisation would not be forced.

The `ColoredPROP` typeclass earns its keep in three ways: generic rearrangements (any list permutation induces a morphism in any colored PROP, proved once), the `St → Br` relationship (the `reindexings` of a `BrBase` are a family of **St** morphisms inside a **Br** morphism — the data of a monoidal functor `St → Br`), and the interchange law `(f ; g) ⊗ (h ; k) = (f ⊗ h) ; (g ⊗ k)`, derivable once from `tensorHom` and `assoc`.

### 2.1 `Numeric`

Both **St** and **Br** rely on symbolic dimension expressions — terms in a free commutative ring on `String`-named generators. Two roles need DIFFERENT coefficient rings: axis **sizes** are non-negative, so they live in `Numeric = MvPolynomial String ℕ` (a `CommSemiring`); affine **stride coefficients and offsets** are *signed* (a look-back read `X[i-1]` has offset `-1`), so they live in `Coeff = MvPolynomial String ℤ` (a `CommRing`). Both are Mathlib types with `DecidableEq` and a `ring`-ready algebra for free, covering the Python `Numeric` hierarchy (`Integer`, `FreeNumeric`, `Addition`, `Multiplication`, `Power`) uniformly.

```lean
abbrev Numeric := MvPolynomial String ℕ   -- axis SIZES (non-negative); a CommSemiring
abbrev Coeff   := MvPolynomial String ℤ   -- StMat COEFFICIENTS/offsets (signed); a CommRing
-- free variable s  ↦  MvPolynomial.X s     (a degree-1 monomial)
-- literal n        ↦  MvPolynomial.C ↑n    (a constant polynomial)
-- addition, multiplication ↦ ring operations
-- instances : CommSemiring Numeric / CommRing Coeff, DecidableEq — free from Mathlib (NONCOMPUTABLE)
```

These are the minimal types making `StMat`'s laws provable: free commutative algebras on a `String`-indexed generator set. The `ring` tactic works immediately over any `CommSemiring`/`CommRing`, so all three `StMat` laws discharge without setup over `Coeff`. `MvPolynomial.X s` plays the role of a `FreeNumeric` — a name carrying no interpretation until indeterminates are substituted. **(Why two types:** an earlier design used `Numeric` for both roles, but its ℕ-semiring has no additive inverses, so negative look-back offsets — supported by the DSL and the [§13.3](#133-abstract-syntax) `IdxExpr` — were unrepresentable; separating out the signed `Coeff` fixes that while keeping sizes non-negative.)

Mathlib's `CommSemiring (MvPolynomial …)` instance is **noncomputable** (it factors through `AddMonoidAlgebra`); so are `MvPolynomial.X`/`.C` and even `DecidableEq` (which resolves through `Classical.propDecidable`). This has no effect on the proof tower — `ring`/`simp` and the [§10](#10-the-propositions-as-generic-theorems) proofs work fine over a noncomputable semiring — but it has two consequences (`Coeff = MvPolynomial String ℤ` is noncomputable for the same reason). (i) Any `def` that *builds* a value over `Numeric`/`Coeff` (so `StMat.id`, `StMat.comp`, and the `St` instance of [§2.2](#22-st--stride-matrices)) must be marked `noncomputable`. (ii) **Compiled code cannot construct or `decide`-compare `Numeric`/`Coeff` values at all** — neither `#eval`, `decide`, nor `native_decide` reduces them. The executable / DSL layer must therefore not construct `Numeric` directly: the tensor-logic DSL ([§13](#13-the-tensor-logic-dsl)) carries sizes in a *computable* `SizeExpr` mirror, interpreted into `Numeric` only on the proof side (see the note in [§13.3](#133-abstract-syntax)).

### 2.2 `St` — stride matrices

**St** instantiates `ColoredPROP` with `gen = Axis`. Its objects are shapes (lists of axes, whose sizes are `Numeric`); its morphisms are affine coordinate transforms stored as stride matrices over `Coeff`.

```lean
structure Axis where
  name : Option String
  size : Numeric      -- symbolic SIZE (non-negative, ℕ); filled in at configuration time

abbrev StObj := List Axis  -- a shape = an ordered list of axes
```

A morphism `dom → cod` is a matrix `Λ ∈ Coeff^{|cod|×|dom|}` of signed coefficients plus a bias vector, with row `j` giving the linear combination of input coordinates producing output coordinate `j`. Coefficients live in `Coeff = MvPolynomial String ℤ` ([§2.1](#21-numeric)), not the size type `Numeric`, so look-back offsets (negative) are representable. Using Mathlib's `Matrix` gives the composition law for free:

```lean
@[ext] structure StMat (dom cod : StObj) where   -- @[ext]: equality of stride matrices is entrywise on coeffs + bias
  coeffs : Matrix (Fin cod.length) (Fin dom.length) Coeff   -- signed (ℤ), NOT Numeric (ℕ)
  bias   : Fin cod.length → Coeff

noncomputable def StMat.id (a : StObj) : StMat a a where   -- noncomputable: Numeric semiring (§2.1)
  coeffs := 1        -- Matrix.one : Matrix (Fin n) (Fin n) Numeric
  bias _ := 0

noncomputable def StMat.comp (f : StMat a b) (g : StMat b c) : StMat a c where
  coeffs := g.coeffs * f.coeffs                         -- Matrix.mul
  bias i := dotProduct (g.coeffs i) f.bias + g.bias i  -- ∑_k g[i,k] * f.bias[k] + g.bias[i]
```

`Matrix.mul` is `(A * B) i j = ∑_k A i k * B k j`; `dotProduct v w = ∑_k v k * w k` handles the bias update. (`dotProduct` lives in the **root** namespace in current Mathlib — notation `⬝ᵥ` — not under `Matrix.`.)

```lean
-- The three category laws are proved as named theorems (each `by apply StMat.ext`, splitting into a
-- coeffs goal and a `funext`'d bias goal), then referenced in the instance:
--   StMat.id_comp   : coeffs by Matrix.mul_one;  bias by dotProduct_zero
--   StMat.comp_id   : coeffs by Matrix.one_mul;  bias by simp [Matrix.one_apply, dotProduct]
--   StMat.comp_assoc: coeffs by Matrix.mul_assoc; bias by a simp-set normalizing both double-sums
--                     (Matrix.mul_apply, dotProduct, Finset.mul_sum, …) then `rw [add_assoc, Finset.sum_comm]`
-- (The bias of comp_assoc needs the sum-reordering rewrite — bare `ring` does not reach under the ∑.)
noncomputable instance St : ColoredPROP StObj where   -- noncomputable: StMat.id/comp build over Numeric (§2.1)
  gen    := Axis
  toList := id
  ofList := id
  hom    := StMat
  id     := StMat.id
  comp   := StMat.comp
  id_comp       := StMat.id_comp
  comp_id       := StMat.comp_id
  assoc         := StMat.comp_assoc
  tensor_assoc  := by intro a b c; simp [List.append_assoc]
  tensor_unit_l := by intro a; simp
  tensor_unit_r := by intro a; simp [List.append_nil]
  tensorHom f g :=                                      -- block-diagonal, reindexed to Fin (·.length)
    -- fromBlocks is indexed by `Fin _ ⊕ Fin _`; bridge to `Fin (a ++ c).length` via
    -- `(finCongr (List.length_append ..)).trans finSumFinEquiv.symm` and `Matrix.reindex`.
    { coeffs := Matrix.reindex eB eC (Matrix.fromBlocks f.coeffs 0 0 g.coeffs)
      bias   := fun i => Sum.elim f.bias g.bias (eB i) }
  swap          := …   -- SIGNATURE: permutation matrix (Matrix.reindex of 1), zero bias
  elemental     := …   -- SIGNATURE: stride matrices are separated by their points (global elements ⊢ each row)
```

The category laws discharge using Mathlib's `Matrix` API (`Matrix.mul_one`/`one_mul`/`mul_assoc` for coefficients; root-namespace `dotProduct` lemmas + a sum-reordering rewrite for the bias), all over the noncomputable `CommRing Coeff` ([§2.1](#21-numeric); the proofs are `CommRing`-generic, so they were unchanged when coefficients moved from `Numeric` to the signed `Coeff`). `Matrix.fromBlocks` builds the block-diagonal `tensorHom` (reindexed through `finSumFinEquiv` to land in `Fin (·.length)`). `swap` (a reindexed identity matrix) and `elemental` are the two `SIGNATURE` fields left as obligations: `elemental` holds because a stride matrix is determined by its action on global elements (points) — evaluating against the basis points recovers each coefficient row, so two stride matrices agreeing on all points are equal — but its proof is deferred.

### 2.3 `Br` — free category over broadcasted base morphisms

**Br** instantiates `ColoredPROP` with `gen = ArrayType`. Because there is no single canonical way to compose two arbitrary broadcasted operations, **Br** morphisms are a free list — the analog of `Composed[Array[B,A], Broadcasted[B,A]]` — making all category laws trivial list lemmas.

```lean
inductive DType
  | reals
  | nat : Numeric → DType           -- Natural(max_value)

structure ArrayType where
  dtype : DType
  shape : StObj                     -- shape lives in Ob(St)

abbrev BrObj := List ArrayType      -- a product of arrays
```

A single `BrBase` is the root morphism of **Br**, bundling one base operation with its reindexings (from **St**), input weaves, and output weaves.

```lean
inductive WeaveSlot
  | fixed : Axis → WeaveSlot   -- retained axis: the reindexing selects a value for this axis at each degree step
  | tiled : WeaveSlot           -- contracted axis: the base op processes the full extent of this axis

abbrev WeaveShape := List WeaveSlot
-- per-array slot list; distinct from §5's `structure Weave (g)` (the cartesian-lift factorization datum)

def WeaveShape.targetAxes (w : WeaveShape) : StObj :=
  w.filterMap fun | .fixed a => some a | _ => none

structure BrBase (dom cod : BrObj) where
  op           : String
  degree       : StObj                          -- shared loop shape P
  inputWeaves  : Fin dom.length → WeaveShape
  outputWeaves : Fin cod.length → WeaveShape
  -- Each reindexing is a St morphism P → (target axes of that input's weave).
  -- This is the locus where St lives inside Br.
  reindexings  : ∀ i : Fin dom.length,
                   StMat degree (inputWeaves i).targetAxes
```

The `reindexings` field precisely captures the four cases from the paper — identity, deletion (broadcast), duplication (diagonal), affine scaling (strided convolution) — each a different `StMat`.

```lean
inductive BrMorph : BrObj → BrObj → Type
  | nil  : (a : BrObj) → BrMorph a a
  | cons : BrBase a b → BrMorph b c → BrMorph a c

def BrMorph.comp : BrMorph a b → BrMorph b c → BrMorph a c
  | .nil _,     g => g
  | .cons f fs, g => .cons f (BrMorph.comp fs g)
```

This is the free category on `BrBase`: morphisms are lists of base operations threaded sequentially (`nil` the empty identity), composition is list concatenation.

```lean
instance Br : ColoredPROP BrObj where
  gen    := ArrayType
  toList := id
  ofList := id
  hom    := BrMorph
  id     := .nil
  comp   := BrMorph.comp
  -- nil ++ g = g definitionally:
  id_comp := by intros; rfl
  -- f ++ nil = f, by induction on f:
  comp_id := by
    intro _ _ f; induction f with
    | nil _      => rfl
    | cons _ _ ih => simp [BrMorph.comp, ih]
  -- list concatenation is associative, by induction on f:
  assoc := by
    intro _ _ _ _ f _ _; induction f with
    | nil _      => rfl
    | cons _ _ ih => simp [BrMorph.comp, ih]
  tensor_assoc  := by simp [List.append_assoc]
  tensor_unit_l := by simp
  tensor_unit_r := by simp [List.append_nil]
  swap a b      := .cons ⟨"swap",
      [],                                                            -- degree: no loop shape
      fun i => ((a ++ b).get i).shape.map (fun _ => .tiled),        -- all input axes tiled
      fun j => ((b ++ a).get j).shape.map (fun _ => .tiled),        -- all output axes tiled
      fun _ => ⟨Matrix.empty, Fin.elim0⟩⟩                          -- StMat [] []: trivial reindexing
    (.nil _)
  -- Routing (input i → output π(i)) is determined by dom/cod types and realized by
  -- the Algebra F : C → V; BrBase records the weave shape, not the routing map.
  tensorHom f g :=
    -- Sequential encoding of parallel composition: extend f to act on (a ++ c) → (b ++ c)
    -- by passing c-arrays unchanged, then extend g on (b ++ c) → (b ++ d) similarly.
    BrMorph.comp (BrMorph.extendRight f c) (BrMorph.extendLeft g b)
  -- def BrMorph.extendRight (f : BrMorph a b) (c : BrObj) : BrMorph (a ++ c) (b ++ c)
  --   := augment each BrBase step of f with identity pass-throughs for the c-arrays
  -- def BrMorph.extendLeft (g : BrMorph c d) (b : BrObj) : BrMorph (b ++ c) (b ++ d)
  --   := symmetric; b-arrays unchanged
  elemental     := …   -- Br is elemental: see theory.md §Elemental Categories (graded_prop.md (Elem-C))
```

The category and tensor-strictness laws discharge sorry-free by `rfl` or one-step structural induction, because list concatenation is associative and `nil` is a two-sided unit definitionally. Because `BrMorph` is a free list carrying no `Numeric` arithmetic (the `StMat` reindexings ride inside `BrBase` but are never evaluated in composition), **`instance Br` is computable** — no `noncomputable` is needed, in contrast to the `St` instance of [§2.2](#22-st--stride-matrices). The `swap`, `tensorHom` (via the sketched `extendRight`/`extendLeft`), and `elemental` fields are the deferred `SIGNATURE` obligations — the bodies above show the intended constructions, but the present implementation stubs them. The `elemental` field is the `(Elem-C)` instance of [graded_prop.md §2](graded_prop.md) — **Br is elemental**, witnessed by the argument in [theory.md §Elemental Categories](theory.md).

The two instances embody a complementary split. **St is semantic**: a stride morphism is the *denotation* of a coordinate transform, not a syntax tree, so composition collapses to a single `Matrix.mul` and the laws come from Mathlib's `Matrix` API plus `ring`. **Br is syntactic (free)**: a composed sequence of broadcasted operations is stored as a list with no canonical simplified form, so the laws are free gifts from list algebra; the price is that symbolic reasoning over **Br** pattern-matches the list rather than inspecting one record.

## 3. The seam adapter into Mathlib

The base class above is deliberately lightweight, so its `St`/`Br` instances keep the "tensor = list concat, strictness = `rfl`/`simp`" elegance. But the propositions of [§10](#10-the-propositions-as-generic-theorems) want Mathlib — `MonoidalCategory`, `SymmetricCategory`, `Grothendieck`, `Limits.pushout`, monoidal functors. A single stated **adapter** bridges the two, turning any `ColoredPROP O` into a Mathlib strict symmetric monoidal category.

```lean
-- The bare category is direct and SORRY-FREE: forward the SmallCategory data and laws.
instance instCategoryOfColoredPROP [ColoredPROP O] : CategoryTheory.Category O where
  Hom X Y := SmallCategory.hom X Y
  id X    := SmallCategory.id X
  comp f g := SmallCategory.comp f g
  id_comp := SmallCategory.id_comp ; comp_id := SmallCategory.comp_id ; assoc := SmallCategory.assoc

-- The monoidal/symmetric structure is the *strict* one read off the ColoredPROP data, built
-- DIRECTLY on the category above (NOT via FreeMonoidalCategory.ofEquivalence — that would supply
-- a second Category instance and create a diamond). The structural isos are eqToIso of the
-- already-proved strictness laws, so the category is strict by construction:
noncomputable instance [ColoredPROP O] : CategoryTheory.MonoidalCategory O where
  tensorObj X Y := ColoredPROP.tensor X Y
  tensorUnit    := ColoredPROP.unit
  whiskerLeft X _ _ g := ColoredPROP.tensorHom (𝟙 X) g
  whiskerRight f Z    := ColoredPROP.tensorHom f (𝟙 Z)
  tensorHom f g := ColoredPROP.tensorHom f g
  associator  X Y Z := eqToIso (ColoredPROP.tensor_assoc  X Y Z)   -- strict: associator = eqToIso
  leftUnitor  X     := eqToIso (ColoredPROP.tensor_unit_l X)       -- strict: unitor    = eqToIso
  rightUnitor X     := eqToIso (ColoredPROP.tensor_unit_r X)
  -- tensorHom_def / whiskerLeft_id / id_whiskerRight / pentagon / triangle: discharged from the
  -- ColoredPROP morphism laws (tensorHom_id, tensorHom_comp) over the strict eqToIso structure.
  -- associator/leftUnitor/rightUnitor NATURALITY: §12 obligations (…) — need a tensorHom-vs-
  -- structural-equality transport coherence the lightweight class does not carry; see §12.
  … 
noncomputable instance [ColoredPROP O] : CategoryTheory.SymmetricCategory O where
  braiding X Y := { hom := ColoredPROP.swap X Y, inv := ColoredPROP.swap Y X, .. }  -- inv laws: swap_swap
  -- symmetry, braiding hom_inv_id/inv_hom_id: from swap_swap.
  -- braiding naturality + hexagon: §12 obligations (…) — not implied by swap_swap.
  …
```

This is the hybrid foundation of the proposition/computation separation: everything **above** the seam — the instances and executable defs (`St`, `Br`, `StMat.comp`, the union-find of [§7](#7-the-grothendieck-construction-and-executable-seam)) — speaks `ColoredPROP`; everything **below** it — the [§10](#10-the-propositions-as-generic-theorems) theorems — speaks Mathlib. The adapter *is* the proposition/computation boundary of [§1](#1-orientation-one-structure-one-seam) made into a definition: it is the one place where "what Lean computes" is handed to "what Lean proves."

The adapter is produced **once** and reused by every [§10](#10-the-propositions-as-generic-theorems) theorem; per-instance proof obligations recur, but the bridge does not. The bare `Category` half is direct and sorry-free — it just forwards the `SmallCategory` data and laws. The strictification is the crux of the *monoidal* half: rather than carrying the development over `FreeMonoidalCategory (Discrete O)` and transferring along an equivalence (which would re-supply a `Category` instance and create a diamond), the monoidal structure is read off the `ColoredPROP` data directly, with the structural isos given by **`eqToIso` of the already-proved strictness laws** (`tensor_assoc`/`tensor_unit_l`/`tensor_unit_r`). This makes the category strict by construction and lets a `ColoredPROP` law stated with `=` line up with a Mathlib `MonoidalCategory` whose coherences are isos. Given the morphism-level `ColoredPROP` laws (`tensorHom_id`/`tensorHom_comp`/`swap_swap`, [§2](#2-the-base-coloredprop)), the *bifunctorial* seam coherences — `tensorHom_def`, the whisker identities, pentagon, triangle, and the braiding inverse/symmetry — discharge sorry-free; the *deeper* coherences (associator/unitor naturality, braiding naturality, hexagon) are the genuine §11 obligations left as `…` (see the [§12](#12-lean-formalization-notes) note).

## 4. The core: `DGradedColoredPROP D C`

Everything above is the base on which the actual subject of this document sits. A `D`-graded colored PROP ([graded_prop.md §3.1](graded_prop.md#31-data)) is a colored PROP `C` (the *operations*) together with the data that exhibits it over a second colored PROP `D` (the *index*): a shape map, a lift action, and the distributivity and action-coherence isomorphisms making `C` a right `D`-actegory. The core class collects exactly that data plus the four named laws.

```lean
/-- Extension of the shape map `sh : gen_C → D` to a monoid homomorphism on objects `C → D`.
    This is the `sh*` used in the (Sh-⊛) law; it satisfies (Sh-⊗): `sh*(X ⊗ Y) = sh*(X) ⊗ sh*(Y)`. -/
def sh_star {C D : Type} [ColoredPROP D] [ColoredPROP C]
    (sh : ColoredPROP.gen (ob := C) → D) (X : C) : D :=
  (X.toList.map sh).foldr ColoredPROP.tensor ColoredPROP.unit
-- instance : Fintype sh_star (follows from Fintype on the color set)

class DGradedColoredPROP (D C : Type) [ColoredPROP D] [ColoredPROP C] where
  sh    : ColoredPROP.gen (ob := C) → D        -- shape map: each C-color's underlying D-shape; extends to sh* (monoid hom)
  act   : (C ×ᶜ Dᵒᵖ) ⥤ C                        -- lift action (Mathlib functor, via seam)
  δ     : ∀ X Y P, act.obj (tensor X Y, P) ≅ tensor (act.obj (X,P)) (act.obj (Y,P))
  δ0    : ∀ P, act.obj (unit, P) ≅ unit
  υ     : ∀ X, act.obj (X, unit_D) ≅ X
  α     : ∀ X P Q, act.obj (act.obj (X,P), Q) ≅ act.obj (X, tensor Q P)
  sh_act : ∀ X P, sh_star sh (act.obj (X,P)) = tensor (sh_star sh X) P   -- (Sh-⊛)
  act_unit_assoc :                                                          -- right D-actegory
    (∀ X, …) ∧                                                            --   triangle (υ/α coherence at the unit)
    (∀ X P Q, …)                                                          --   pentagon (α coherence)
  υ_nat : ∀ {X Y} (f : hom X Y), act.map ⟨f, 𝟙⟩ ≫ (υ Y).hom = (υ X).hom ≫ f
  -- (υ-naturality): the unitor is natural in C. Stated as a law here (like δ/δ0 naturality below)
  -- because `act` functoriality alone does NOT give it — and Eq. 3 (§4.1) needs it.
  dist_coh :                                                                -- δ, δ0 naturality + interchange
    (∀ {X Y} (f : hom X Y) P, (δ X Y P).hom ≫ tensor (act.map ⟨f,𝟙⟩) (act.map ⟨f,𝟙⟩) = act.map ⟨tensorHom f f, 𝟙⟩ ≫ (δ X Y P).hom) ∧
    (∀ P, (δ0 P).hom ≫ 𝟙 unit = act.map ⟨𝟙, 𝟙⟩ ≫ (δ0 P).hom)           --   δ0 naturality
  broadcast_gen :                                                            -- (Broadcast-gen)
    ∀ {X Y : C} (g : hom X Y),
      ∃ (X' Y' : C) (f : hom X' Y') (P : D)
        (lam : hom X (act.obj (X', P)))                          -- leading reindex: X → X' ⊛ P
        (ρ : hom (act.obj (Y', P)) Y),                            -- trailing reindex: Y' ⊛ P → Y
        g = comp (comp lam (act.map ⟨f, 𝟙⟩)) ρ ∧
        ∀ Q (h : hom unit_D Q), act.map ⟨f, h⟩ = act.map ⟨f, 𝟙⟩  -- f is degree-trivial
  -- NB: the leading `lam` is required — without it `act.map ⟨f,𝟙⟩ : X' ⊛ P → Y' ⊛ P` has domain
  -- `X' ⊛ P`, not `X`, so it cannot equal `g : X → Y`. `Weave` (§5) bundles exactly this datum.
```

Each field is a direct transcription of the [graded_prop.md §3.1](graded_prop.md#31-data) data. `sh` is the **shape map** `sh : O_C → Ob D` sending each `C`-color to its underlying `D`-shape (its list of sub-wires); it extends to the monoid homomorphism `sh*` on objects used by the (Sh-⊛) law. `act` is the **lift action** `act : C × Dᵒᵖ ⥤ C` — a Mathlib functor, available because the seam adapter of [§3](#3-the-seam-adapter-into-mathlib) makes `C` and `D` Mathlib categories. Its two specializations are theory.md's lift notation: `[f, P] := act(f, 𝟙_P)` is the **batch lift** of `f` (covariant in `C`), and `[X, η] := act(𝟙_X, η)` is the **reindexing** along `η : P → Q` (contravariant in `D`, since `D` enters opposite).

The four isomorphism fields are the **distributivity** and **action-coherence** isos. `δ` and `δ0` make the lift distribute over juxtaposition — `(X ⊗ Y) ⊛ P ≅ (X ⊛ P) ⊗ (Y ⊛ P)` and `I_C ⊛ P ≅ I_C`. `υ` and `α` are the actegory coherence isos — `υ : X ⊛ I_D ≅ X` (lifting by the unit shape is trivial) and `α : (X ⊛ P) ⊛ Q ≅ X ⊛ (Q ⊗ P)` (composing two lifts; the order `Q ⊗ P` is what makes `⊛` a **right** action of `(D, ⊗, I_D)`). Together `υ`/`α` are precisely the unitor and multiplicator exhibiting `C` as a right `D`-actegory.

The `Prop`-valued fields are the named laws of [graded_prop.md §3.2](graded_prop.md#32-axioms):

- `sh_act` is **(Sh-⊛)**: `sh*(X ⊛ P) = sh*(X) ⊗ P` — lifting by `P` appends `P` to the shape.
- `act_unit_assoc` bundles **(Act-unit / Act-assoc)**: `υ` and `α` satisfy the triangle and pentagon coherences, i.e. `C` is a right `D`-actegory.
- `υ_nat` is **unitor naturality**: `[f, I_D] ; υ_Y = υ_X ; f`. It is a *separate* law because `act` functoriality alone does not force it; the derived Eq. 3 of [§4.1](#41-derived-ev_p-and-eq-3) depends on it.
- `dist_coh` bundles **(Dist-nat / Dist-coh)**: `δ`, `δ0` are natural and satisfy the interchange coherence with `υ`, `α`, and the symmetry `σ` — the lift is a strong symmetric monoidal action in the `C`-variable.
- `broadcast_gen` is **(Broadcast-gen)**: every `C`-morphism factors as `lam ; [f, P] ; ρ` with `f` degree-trivial (built without `act` over a non-unit `P`) and `lam`/`ρ` reindexings assembled from `act(𝟙, −)` and the coherence isos. (The leading `lam : X → X' ⊛ P` is needed for the factorization to typecheck — `[f, P]` starts at `X' ⊛ P`, not `X`.) This is the generation principle that makes weaves ([§5](#5-weaves-as-cartesian-lift-data)) exist.

Note that **(Act-functor)** of [graded_prop.md §3.2](graded_prop.md#32-axioms) (`act` respects identities and composition in both variables, so `[f ; g, P] = [f, P] ; [g, P]`) and **(Sh-⊗)** (`sh*` is a monoid homomorphism) are not separate fields: the first is the `Functor` laws already carried by `act`, the second is built into the definition of `sh*` from `sh`. And **(Elem-C)** is not here either — it is the `elemental` field of `ColoredPROP` ([§2](#2-the-base-coloredprop)), inherited from the instance `[ColoredPROP C]`.

`D` and `C` are **explicit class parameters**, not `outParam`s. This is deliberate: with both free, Lean's instance search needs them pinned for `[DGradedColoredPROP D C]` to resolve predictably (and so that `Br`-as-graded and `Br`-as-index occupy distinct instance positions without collision). The instance-resolution discipline this implies is taken up in [§12](#12-lean-formalization-notes).

### 4.1 Derived: `ev_p` and Eq. 3

Theory.md's batch-lift defining property (Eq. 3) is **not** an axiom of the core class — it is *derived*. For a point `p : I_D → P` in `D`, the *slice at `p`* is a derived morphism family, and its naturality square is Eq. 3.

```lean
def ev_p [DGradedColoredPROP D C] (p : hom unit_D P) (X : C) : hom (act.obj (X,P)) X :=
  act.map ⟨𝟙 X, p⟩ ≫ (υ X).hom                          -- act(𝟙, p) post-composed with the unitor
-- Eq. 3:  [f,P] ≫ ev_p p Y = ev_p p X ≫ f             -- naturality of ev_p (theorem ev_p_naturality)
```

`ev_p` is a `def`, not a field: it is `act.map ⟨𝟙, p⟩` post-composed with the unitor `υ`. Its naturality square at `f : X → Y` is exactly **(Eq. 3)** `[f, P] ; ev_p p Y = ev_p p X ; f`. The proof combines the two `act.map` factors via `Functor.map_comp` (the product-category interchange `(f,𝟙) ; (𝟙,p) = (𝟙,p) ; (f,𝟙)`) and then slides the unitor past `f` — which is **`υ_nat`**. So Eq. 3 is derived (not posited as its own axiom), but it is *not* free from `act` functoriality alone: it rests on the `υ_nat` law of [§4](#4-the-core-dgradedcoloredprop-d-c). The remaining genuine content lives in `elemental` (the points `ev_p` jointly separate morphisms), which pins down the weave of [§5](#5-weaves-as-cartesian-lift-data) — Eq. 3 holds once `υ_nat` is supplied, and does not by itself force the factorization.

## 5. Weaves as cartesian-lift data

The (Broadcast-gen) law says every `C`-morphism factors as `lam ; [f, P] ; ρ`. A **weave** is a witness of that factorization for a particular morphism — and, as [graded_prop.md §3.3](graded_prop.md#33-weaves-as-cartesian-lift-data) shows, it is precisely the cartesian-lift datum of the grading fibration `C → D`.

> **Naming.** `WeaveShape` ([§2](#2-the-base-coloredprop)) is the *per-array slot list* (`List WeaveSlot`, the shape of one wire). The `Weave g` below is a *different* concept: the cartesian-lift factorization witness for a whole morphism `g`. The Python type named `Weave` maps to the former; this `structure Weave` is the latter.

```lean
structure Weave [DGradedColoredPROP D C] {X Y : C} (g : hom X Y) where
  X' : C ; Y' : C
  f       : hom X' Y'    -- base op (degree-trivial)
  P       : D
  lam     : hom X (act.obj (X', P))   -- leading reindex (the broadcast_gen `lam`)
  ρ       : hom (act.obj (Y', P)) Y   -- trailing reindex: act(𝟙_{Y'}, η) composed with coherence isos,
                                       -- where η : P → (degree fitting X→Y over Y'); assembled
                                       -- from [Y', −] and the α/υ isos of DGradedColoredPROP
  factors : g = lam ≫ [f, P] ≫ ρ

theorem weave_unique [DGradedColoredPROP D C] {X Y} (g : hom X Y) :
    Subsingleton (Weave g)         -- Prop 8.2, from elemental + broadcast_gen
```

A `Weave g` records the (Broadcast-gen) factorization of `g`: a degree-trivial base op `f`, a degree `P ∈ D`, the boundary reindexings `lam`/`ρ` (assembled from `act(𝟙, −)` and the coherence isos), and a proof that `g = lam ; [f, P] ; ρ`. It bundles exactly the witnesses of the `broadcast_gen` field of [§4](#4-the-core-dgradedcoloredprop-d-c), so uniqueness can be stated as a `Subsingleton`. Per wire, the shape `sh(color) ∈ Ob D` is a list of sub-colors that the factorization partitions into **target** sub-colors (acted on directly by `f`) and **tiling** sub-colors (supplied by the degree `P` through `ρ`); the permutation relating the canonical "targets-first" order to the wire's actual sub-color order is theory.md's `Ω_w`, recovered from the symmetry `σ`. This is **precisely the cartesian-lift datum** of the grading fibration `C → D` ([graded_prop.md §3.3](graded_prop.md#33-weaves-as-cartesian-lift-data)): a weave is the choice of how a morphism's wires sit over their `D`-shapes, with the tiling part pulled back along the degree.

`weave_unique` (Proposition 8.2) makes `Weave g` a `Subsingleton` — at most one weave, up to the canonical coherence isos. This is what turns `Weave` into a **datum, not a choice**: the factorization is forced, not selected. The proof draws on both the `elemental` field of `[ColoredPROP C]` (points separate morphisms, so the degree `P` and the target/tiling partition are determined) and `broadcast_gen` (a factorization exists at all). Without `elemental`, Eq. 3 would still hold (given `υ_nat`) but the `(lam, f, P, ρ)` factorization would not be unique.

## 6. Mixins: Scan, Route, Symmetry, Para

The core `DGradedColoredPROP D C` of [§4](#4-the-core-dgradedcoloredprop-d-c) carries the lift, the actegory coherences, and the weave factorization — and nothing more. Capabilities beyond that bare grading are added as **composable mixins**: a `Scan`, a `Route`, an equivariance constraint, a `Para` refinement. Each is its own typeclass; an instantiation declares only the mixins it actually uses and pays only for their fields and obligations. This is exactly what keeps the proven core un-edited as the framework grows: a new capability is a new class layered on `DGradedColoredPROP`, never a field added to it, so the [§10](#10-the-propositions-as-generic-theorems) theorems stated over the core continue to hold unchanged at every instantiation. The four mixins of this family are `TemporalGraded` (Scan, given in full below), the `RouteStructure` and `SymmetryGraded` stubs, and `ParaAlgebra` (forward-referenced to [§7](#7-the-grothendieck-construction-and-executable-seam), since it layers on the `Algebra` rather than on the graded PROP).

### 6.1 `TemporalGraded` — Scan

```lean
class TemporalGraded (D C : Type) [ColoredPROP D] [ColoredPROP C]
    extends DGradedColoredPROP D C where
  L          : D                    -- temporal object; the colimit/top of the prefix family
  prefix     : ℕ → D                -- the prefix objects [0..m]
  iotaTo     : ∀ {m n}, m ≤ n → hom (prefix m) (prefix n)   -- (ℕ,≤)-functor of prefix inclusions
  iota       : ∀ m : ℕ, hom (prefix m) L                    -- ιₘ : [0..m] ↪ L (cocone into L)
  -- the prefix family is a functor (ℕ,≤) ⥤ D, and L is a cocone over it:
  iotaTo_id   : ∀ m, iotaTo (le_refl m) = 𝟙 (prefix m)
  iotaTo_comp : ∀ {m n k} (h₁ : m ≤ n) (h₂ : n ≤ k), comp (iotaTo h₁) (iotaTo h₂) = iotaTo (le_trans h₁ h₂)
  iota_factor : ∀ {m n} (h : m ≤ n), comp (iotaTo h) (iota n) = iota m
  -- directed restriction action along the prefix maps: act(X, [m↪n]) : X ⊛ [0..n] → X ⊛ [0..m]
  restrict   : ∀ {m n} (h : m ≤ n) (X : C),
                 hom (act.obj (X, prefix n)) (act.obj (X, prefix m))   -- contravariant in D-slot
  -- finite N-fold iteration of a step endomorphism (cata):
  iterate    : ∀ (N : ℕ) (X : C) (step : hom X X),
                 hom X (act.obj (X, prefix N))               -- cata(step), output indexed by [0..N]
  -- state history (scanl): records the full trajectory; truncating along [m↪N] agrees with iterate m
  trace      : ∀ (N : ℕ) (X : C) (step : hom X X),
                 hom X (act.obj (X, prefix N))
  lift_fold_dist : ∀ (N : ℕ) (X : C) (P : D),
                     act.obj (act.obj (X, prefix N), P) ≅ act.obj (X, prefix N)
                     -- act(Scan, P) ≅ Scan(act(step, P)) for P orthogonal to L
-- NB: the design's `iota 0 = 𝟙 unit_D` is ill-typed (codomains differ); the prefix-family form above
-- — `prefix`/`iotaTo`/`iota` with the functor laws `iotaTo_id`/`iotaTo_comp` and the cocone law
-- `iota_factor` — is the genuine well-typed encoding of Definition 3.3's prefix inclusions.
-- (D-objects sit in the `Dᵒᵖ` slot of `act` via `Opposite.op`; elided here for readability.)
```

`TemporalGraded` internalizes the four additions of [graded_prop.md §3.4](graded_prop.md#34-the-temporal-grading-and-making-scan-first-class) that turn `Scan` from a bare generator into a definition. `L` is the **temporal object** of Definition 3.3 — an `Ob D` carrying the augmented-simplex / `(ℕ,+,0)` length grading, with prefix inclusions `ιₘ : [0..m] ↪ [0..N]`. `restrict` is the **directed action**: restriction natural transformations `act(−, ιₘ) : (− ⊛ [0..N]) ⇒ (− ⊛ [0..m])` along the `ιₘ`, satisfying the unit and composition laws of an action. `iterate` is the **finite iteration** of Definition 3.4 — for a parametric step endofunctor (the per-step inputs ride as parameters) and a length `N`, the `N`-fold iterate and its catamorphism `cata(step)` exist (for fixed `N` this is plain `N`-fold composition; no fixpoint, no natural-numbers object, until unbounded length is wanted). `trace` is the **state history** of Definition 3.5 — codomain `H ⊗ L_{N+1}` (scanl), with the coherence that truncating the trace along `ιₘ` agrees with running the `m`-fold. `lift_fold_dist` is the **lift–fold distributivity** law: for an ordinary degree `P` orthogonal to `L`, `act(Scan, P) ≅ Scan(act(step, P))`.

With these fields in hand, **`Scan := cata(step)` is a definition** over the temporal grading, not a generator posited by hand. Two consequences follow rather than being assumed. First, the **prefix-restriction law is a corollary** (Proposition 8.7): theory.md's law that `Scan_N` restricted to the first `m` steps equals `Scan_m` falls out of the catamorphism universal property `cata(step) ∘ in = step ∘ F(cata(step))` and its uniqueness — it need not be a separate axiom. Second, **`Scan` batches** along any axis `P` orthogonal to `L` (Proposition 8.8): `lift_fold_dist` is exactly what makes `act(Scan, P) ≅ Scan(act(step, P))`, so a batched recurrence is one fold run independently per batch coordinate, and `Scan` participates in the `vmap`/batch strategies like any other morphism.

`TemporalGraded` uses `extends DGradedColoredPROP` — genuine inheritance, not a signed-empty stub — because `Scan` needs the **whole core**: the lift `act`, the actegory coherences, and the weave factorization are all load-bearing. `cata(step)` is built from `iterate` over `act`; `trace` is typed by the lift; and `lift_fold_dist` is a statement *about* `act`, so it cannot even be phrased without the actegory. The stubs of [§6.2](#62-route-and-symmetry-stubs) also `extends DGradedColoredPROP`, but their bodies are deferred — they declare their parameters and leave the fields signed-empty (`…`), because the machinery they need is not yet formalized.

The key obstruction that forces `Scan` out of the weave story lives in the `restrict` field: the directed action carries restrictions along the prefix inclusions `ιₘ`, but **no point-evaluation `ev_q`** for a single point `q : I → L`. That absence is precisely the Proposition 8.6(i) obstruction — `Scan`'s lift couples positions (the output at `ℓ` depends on positions `< ℓ`), so `ev_q` fails to be natural (Eq. 3 fails along `L`), and `Scan` lies only in the image of the directed sub-action indexed by the `ιₘ`, never by points. It is *not* a weave along `L`; Proposition 8.7 is its positive home, and Proposition 8.8 confirms the obstruction is only along `L` and never along orthogonal axes.

### 6.2 Route and Symmetry stubs

```lean
class RouteStructure (D C) [ColoredPROP D] [ColoredPROP C]
    extends DGradedColoredPROP D C where … -- STUB: data-dependent coproduct injection, gate as Para param
class SymmetryGraded (D C : Type) (T : Monad D) [ColoredPROP D] [ColoredPROP C]
    extends DGradedColoredPROP D C where … -- STUB: equivariance via EM-category of T; gated (equiv_unif A3)
```

`RouteStructure` is the second species of the Proposition 8.6 obstruction — Proposition 8.6(ii). Here the reindexing **depends on input values**, so it is not a fixed `D`-morphism at all: there is no single `η` to lift, hence no weave. The generator is a data-dependent coproduct injection whose routing map is carried as a `Para` parameter (the gate). The motivating example is sparse / top-`k` mixture-of-experts ([future_ideas.md §6.4](future_ideas.md#64-stacking-levels-weaves-over-models-mixture-of-experts-ensembles)), where the expert each item reads is `argmax`-selected at runtime.

`SymmetryGraded` is Proposition 8.4's equivariance, encoded via the **Eilenberg–Moore category** of a symmetry monad `T` on `D` (hence the extra parameter `T : Monad D`). It is **gated** on that EM-machinery ([equivariance_unification.md](equivariance_unification.md)): equivariance is reachable for finite groups, but the graded-PROP-dependent parts of the encoding wait on this formalization being in place.

Both are **signed stubs**: the class is declared with its parameters and its `extends DGradedColoredPROP`, but the fields are deferred (`…`) — the data is named, the bodies are future work.

`ParaAlgebra` (the `Para` refinement mixin on `construct()`) is *not* introduced here. It is presented in [§8](#8-algebras-and-construct) alongside the `Algebra` class, because it layers on the algebra — refining `Para(C) → Para(V)` as a 2-functor, with weight tying as passes-as-2-cells — rather than on the graded PROP. It is listed in this section's title only because it belongs to the same mixin family.

## 7. The Grothendieck construction and executable seam

Symbolic axis *sizes* and axis *identity/alignment* are ordinary categorical data on the graded-PROP reading. Sizes are the **fiber of a Grothendieck construction**; alignment is a **pushout/coequalizer**. This section states both, and makes the proposition/computation seam of [§1](#1-orientation-one-structure-one-seam) concrete: the categorical objects are the *specification*, and the executable structures (a fresh-UID counter, a union-find `Context`) are the *implementation* that realizes it.

### 7.1 The structure/data split as `∫Dat`

Strip the numeric content from the `D`-colors and one is left with the **structural index PROP** `D♯` (colors are formal size-symbols, morphisms are symbolic reindexings); `C♯` is then `D♯`-graded ([graded_prop.md §5](graded_prop.md#5-the-structuredata-split-as-a-grothendieck-construction)). A **data functor** `Dat : C♯ → Set` sends each structural object to its set of admissible size-assignments, acting *trivially on morphisms* — data is unconstrained by connectivity ([acset.md §The Grothendieck Construction](acset.md#the-grothendieck-construction)). The **Grothendieck construction** `∫Dat` has objects `(c, d)` with `c ∈ C♯` and `d ∈ Dat(c)`, and morphisms the `C♯`-morphisms with no compatibility condition on data, and

> **`C ≅ ∫Dat`**

recovers the fully-sized graded PROP (graded_prop.md Prop 8.3).

```lean
-- C♯ is the structural index PROP: same connectivity as C, but all Numeric size fields erased.
-- Concretely, `C♯` is built by replacing every `Axis.size : Numeric` with `Unit` throughout
-- `C`'s color type, leaving the shape (list-of-wires) structure intact.
-- The quotient construction: two C-morphisms are C♯-equivalent iff they agree on connectivity
-- (weave shapes, op names, routing) and differ only on Numeric attributes.
-- structuralCongruence C : HomRel C — the connectivity-agreement relation (a Congruence).
abbrev CSharp (C : Type) [ColoredPROP C] : Type :=
  CategoryTheory.Quotient (structuralCongruence C)
notation "Cˢʰᵃʳᵖ" => CSharp
-- `C♯`/`Cˢʰᵃʳᵖ` is notation for `CSharp` — the superscript chars are not legal Lean identifiers,
-- so the real identifier is `CSharp`.

-- The data functor: each C♯-object ↦ its set of valid size-assignments; trivial on morphisms.
def Dat (C : Type) [ColoredPROP C] : CSharp C ⥤ Type :=
  …   -- obj := fun c => { d : C // Quotient.mk _ d = c }; map := fun _ => id  (graded_prop.md Def 5.1)
-- Valued in `Cat` via `typeToCat` (the discrete-category functor `Type ⥤ Cat`) for `Grothendieck`:
noncomputable def Dat' (C : Type) [ColoredPROP C] : CSharp C ⥤ CategoryTheory.Cat :=
  Dat C ⋙ typeToCat
-- Prop 8.3: the structure/data split. Stated as the existence of an equivalence `C ≌ ∫Dat`
-- (definitional `Iso.refl` if `C` is *built* as `∫Dat`; a constructed equivalence otherwise):
theorem grothendieck_split (C : Type) [ColoredPROP C] :
    Nonempty (C ≌ CategoryTheory.Grothendieck (Dat' C)) := …
```

Mathlib supplies `CategoryTheory.Grothendieck` for the Grothendieck construction of a functor into `Cat`; with `Dat` valued in discrete categories of size-assignments, `∫Dat` is a direct instance. The iso `C ≅ ∫Dat` is a per-instantiation theorem in general, but it is **definitional — `Iso.refl` — if `C` is *built* as `∫Dat`**, which is the recommended posture: define the sized PROP as the integral, and the splitting is true by construction rather than by proof.

### 7.2 `FreeNumeric` is the fiber, not a layer

The `Numeric := MvPolynomial String ℕ` of [§2.1](#21-numeric) is precisely the data the fiber `Dat(c)` ranges over. A symbolic axis size is a term in the free commutative semiring on `String`-named generators; a `FreeNumeric` — the Python `UTerm` that names an as-yet-unknown size — is a single generator `MvPolynomial.X s`, carrying no interpretation until indeterminates are substituted.

```lean
abbrev Numeric := MvPolynomial String ℕ
-- MvPolynomial.X s  -- a FreeNumeric: a symbolic axis size, the fiber datum Dat(c)
-- a size-assignment d ∈ Dat(c) picks a Numeric for each structural axis-symbol of c
```

Symbolic sizes — `FreeNumeric` and `Numeric` — are the **`Dat(c)` fiber data** of the Grothendieck construction of [§7.1](#71-the-structuredata-split-as-dat), as categorical as the structural skeleton `C♯` itself. They live in the fiber over `C♯`.

`Numeric = MvPolynomial String ℕ` is the **proof-side** fiber representation: it is chosen so the `ring` tactic discharges the `StMat` laws for free, at the price of being noncomputable ([§2.1](#21-numeric)). The **executable** presentations of the fiber — the DSL ([§13](#13-the-tensor-logic-dsl)) and the acset tables ([§9](#9-acsets-and-the-executable-layer)) — cannot use it directly, because compiled code can neither construct nor compare `MvPolynomial` values. They carry a *computable* mirror of the fiber (a `SizeExpr` inductive: variables, literals, `+`, `*`, with `DecidableEq`/`ToExpr`), with an interpretation `SizeExpr → Numeric` crossing to the proof side. This is the proposition/computation seam applied to the fiber datum itself; the mechanism is detailed in [§13](#13-the-tensor-logic-dsl).

### 7.3 Composition as pushout

Autoalignment (`@`, `Context`) builds a composite from separately-constructed pieces by gluing them along a shared boundary. This is **not** the primitive composition of morphisms in `C`; it is the composition of *open systems* — structured cospans of acset presentations, each carrying explicit input/output interfaces ([graded_prop.md §6](graded_prop.md#6-composition-as-pushout)). It has two stages, and **only the second is a colimit**.

**Stage 1 — interface discovery (heuristic, *not* categorical).** Decide *which* boundary colors of `cod(f)` and `dom(g)` are identified — construct the span `B → inst f`, `B → inst g`. pyncd does this by positional pairing plus shape-based `(name, size)` matching, inserting identities and prepending rearrangements to reconcile arity and order. This step is a **choice**: it is correct exactly when the `(name, size)` signature determines the axis, and a wrong choice silently over- or under-glues. **Nothing here is a pushout** — it is the construction of the span the pushout will act on. The interesting, failure-prone part of composition — where composition actually *fails*, where the design decision lives — is here, *outside* the colimit.

**Stage 2 — gluing (the pushout).** Given the span, the composite is the **pushout** that identifies the matched boundary colors `B` (the cup), computed componentwise over the schema. On the `Axis` component it is a **coequalizer of UIDs** — exactly what `Context` union-find computes; the **canonical class representative is the universal cocone vertex**. Because pyncd chooses canonical representatives, the pushout is taken on the nose, which strictifies cospan composition so that `;` is *strictly* associative.

```lean
-- Inst(C♯) is a finitely-cocomplete copresheaf category; Stage 2 is a Mathlib colimit:
-- CategoryTheory.Limits.pushout (span legs B → inst f, B → inst g)
-- Stage 1 (span construction) is NOT part of it and is implemented separately.
-- Associativity of `;` is the pasting lemma for pushouts, strictified by the
-- canonical-representative choice — no bespoke proof over `Context` is needed (Prop 8.5).
```

The failure mode lives in the *attributes*, not the structure: the pushout requires the size/datatype data on glued axes to **unify**, and "no consistent attribute assignment" is precisely "the pushout does not exist." `Context` handles the `Axis`-component coequalizer; the size-consistency check is the remainder of the pushout. So [§6 of graded_prop.md](graded_prop.md#6-composition-as-pushout) is a *correctness/specification lens, not a composition algorithm*: the pushout explains and certifies what `Context` does; it does not replace it.

### 7.4 The seam, concrete: union-find realizes the coequalizer

This is where the proposition/computation seam of [§1](#1-orientation-one-structure-one-seam) becomes tangible. The coequalizer of [§7.3](#73-composition-as-pushout) is the **specification**; the executable structures (a fresh-UID counter, a union-find `Context`) are the **implementation**. They meet at the seam, and neither replaces the other.

**Fresh-UID counter — `FreshM`.** Constructing a new symbolic axis mints a fresh identity. Python does this with `random.randint` as a construction side-effect; Lean 4 is pure, so the fresh-name counter is threaded explicitly. Compilation can also fail (shape mismatches, missing base cases, illegal contractions), so `FreshM` is `EStateM CompileError ℕ` — Lean core's combined error+state monad (`Init.Control.EStateM`) rather than a bare `StateM ℕ`. This is the *executable* side — it computes identities and validates constraints; it proves nothing.

```lean
-- The executable-seam modules (LeanNCD/Exec/Uid.lean, Traversable.lean) are Lean-core-only — no
-- Mathlib — so they write `Nat`/`Type u` where this pseudocode shows `ℕ`/`Type*`. `DynamicName` is
-- realized as a `String` stub (display-only, §14).
abbrev UID := ℕ

structure UData where
  uid  : UID
  name : Option DynamicName := none
  deriving Repr, DecidableEq, Inhabited

/-- Combined error + UID-counter monad for DSL compilation.
    `EStateM ε σ α` (Lean core, Init.Control.EStateM) = `σ → Result ε σ α`.
    Compilation validates structure (shape mismatches, missing base cases, etc.)
    and mints fresh UIDs, so both capabilities are needed together. -/
abbrev FreshM := EStateM CompileError ℕ

def freshUData : FreshM UData := do
  let n ← get; set (n + 1); return ⟨n, none⟩
-- Constructing a new axis / FreeNumeric runs in FreshM; pure code (composition, proofs) does not.
-- A counter, not random ints: term construction becomes reproducible and testable.
-- UIDs carry no semantic content — only equality/inequality of two UIDs matters.
```

**Union-find — `Context` / `EqClass`.** The `Axis`-component coequalizer is *computed* as a pure-functional union-find. An `EqClass` is one equivalence class — a `Finset` of UIDs together with its canonical representative; a `Context` is a disjoint list of them. `Context.merge` unions a new class with any overlapping existing ones (Python's `Context.append_bucket`); `Context.apply` substitutes every UID by its class representative throughout a term. The **canonical representative is the member with the largest UID** — and *this is the universal cocone vertex* of the Stage-2 pushout: choosing it on the nose is what strictifies composition.

```lean
/-- Decoration pairing a value with its UID — the canonical representative of an EqClass. -/
structure WithUID (α : Type*) where
  data : α
  uid  : UID    -- the UID of this representative; same as data.uid if α carries a uid field
-- For α = UData, data.uid = uid definitionally; for other α, uid is stored separately.

/-- Errors that FreshM compilation can throw. -/
inductive CompileError
  | shapeMismatch    : String → String → CompileError   -- expected shape, actual shape
  | missingBaseCase  : String → CompileError             -- tensor name missing iterAt stmt
  | causalityViolation : String → CompileError           -- l+1 appears on RHS for iteration axis l
  | overlappingScatter : String → CompileError           -- non-injective scatter without reduce sum
  | linearWeightAmbiguous : String → CompileError        -- linear weight in ≠1 product factors
  | undeclaredName   : String → CompileError             -- name used but not declared
  deriving Repr, DecidableEq

/-- Typeclass for types whose UID references can be traversed and substituted.
    One explicit instance per decorated type; instances derived mechanically by structural recursion.
    Laws: traverseUID id = id, traverseUID (f ∘ g) = traverseUID f ∘ traverseUID g. -/
class TermTraversable (α : Type*) where
  traverseUID : (UData → UData) → α → α
-- Required instances (at minimum): AxisSpec, IdxExpr, PredArith, BoolExpr, Stmt,
--   BrBase, ThreadedComposed. Derive mechanically by structural recursion on each type.

/-- One equivalence class: a set of UIDs with one canonical representative. -/
structure EqClass (α : Type*) where
  bucket    : Finset UID
  canonical : WithUID α            -- representative chosen by largest UID = cocone vertex

/-- A context is a disjoint list of equality classes. -/
structure Context (α : Type*) where
  classes : List (EqClass α)

/-- Merge a new class in, unioning with any overlapping classes (= Context.append_bucket). -/
def Context.merge (ctx : Context α) (cls : EqClass α) : Context α :=
  -- `List.filter` needs a `Bool`, so the overlap test is `decide`d (Finset disjointness is Decidable):
  let overlaps : EqClass α → Bool := fun c => ! decide (Disjoint c.bucket cls.bucket)
  let overlapping := ctx.classes.filter overlaps
  let merged : EqClass α := overlapping.foldl
    (fun acc c => ⟨acc.bucket ∪ c.bucket,
                   if acc.canonical.uid ≥ c.canonical.uid          -- WithUID.uid (the rep's UID), the
                   then acc.canonical else c.canonical⟩)           -- general key — not `.data.uid`
    cls
  ⟨merged :: ctx.classes.filter (fun c => ! overlaps c)⟩
-- instance : DecidableEq UID := inferInstance  -- free from abbrev UID := ℕ

/-- Substitute every UID in each class by its canonical representative throughout a term.
    The target type `β` is independent of the context's element type `α`: substitution is purely
    UID-level (driven by `bucket` + `canonical.uid`), so a `Context` of axes applies to a whole
    program term. -/
def Context.apply [TermTraversable β] (ctx : Context α) (target : β) : β :=
  ctx.classes.foldl (fun t cls =>
    TermTraversable.traverseUID
      (fun d => if d.uid ∈ cls.bucket then { d with uid := cls.canonical.uid } else d) t)
    target
```

The framing is the whole point. **The pushout/coequalizer is the spec; union-find plus the fresh-name counter is the implementation; they meet at the seam.** The Stage-2 colimit *certifies* the gluing — associativity (the pasting lemma), the precise error semantics ("no cocone" = alignment failure, "inconsistent attributes" = size mismatch), the canonical representative as cocone vertex — while `Context` *computes* it in near-linear time. A Lean development proves the coequalizer is what it claims; it does not re-derive `Context` line-by-line, and pyncd would never invoke a generic colimit solver. The substitution machinery `Context.apply` rides on a `TermTraversable` typeclass — one explicit traversal instance per decorated type — but that, the `WithUID` decoration, and `DynamicName` are display/identity bookkeeping on the executable side, never propositional content.

Unlike the propositional tower (§2–§11, transcribed as signatures with `sorry`), this executable seam is **fully implemented and `sorry`-free** (`LeanNCD/Exec/`): `FreshM`/`freshUData` mint an increasing counter and propagate `CompileError`; `Context.merge` unions overlapping buckets keeping the largest-UID representative; `Context.apply` substitutes UIDs by canonical representatives. It is verified by evaluated tests (`#guard` plus an LSpec suite) — `#eval`/`decide` work here precisely because nothing crosses into the noncomputable `Numeric` algebra. The per-type `TermTraversable` instances for the DSL AST (`AxisSpec`, `IdxExpr`, …, `BrBase`, `ThreadedComposed`) are written alongside the compiler in [§13](#13-the-tensor-logic-dsl).

## 8. Algebras and `construct()`

The algebra `F` (graded_prop.md Def 7.2 / [§7](graded_prop.md#7-algebras-construct-and-the-para-refinement)) is the strong symmetric monoidal, `D`-equivariant functor `C → V` into a target actegory — the categorical content of `ConstructedModule.construct()`. It is the last layer of the tower, and the clearest instance of the doc's recurring shape: a typeclass parameterised by the classes below it. The target `V` is itself a right `D`-actegory parameterised by a value semiring `R` (graded_prop.md Def 7.1); the algebra is parametric on both the source graded PROP `C`, the target `V`, and `R`. The coherence/monoidal/equivariance/`Para` laws are all **stated** (signatures + `sorry`, math-tower style); only the `Mat ℝ` instance proofs are deferred.

```lean
-- `V` is `Type*` (a module category lives in `Type 1`); `D`/`C`/`R` are `Type`. `R` is a value
-- semiring carried as `(R : Type) [CommSemiring R]`. `V` is symmetric monoidal.
class TargetActegory (D : Type) (V : Type*) [ColoredPROP D] [Category V] [MonoidalCategory V]
    (R : Type) [CommSemiring R] where
  actV : (V × Dᵒᵖ) ⥤ V               -- P acts by appending R-valued dimensions
  -- The full υ/α/δ coherences of §4, now TRANSPOSED into `V` (⊗_V/𝟙_ V for tensor/unit on `C`):
  δ_V  : …   -- (Dist-⊗) the lift distributes over ⊗_V          (mirrors DGradedColoredPROP.δ)
  δ0_V : …   -- (Dist-I)  the lift distributes over 𝟙_ V         (mirrors …δ0)
  υ_V  : …   -- (Act-unit) the unit grading I_D acts trivially   (mirrors …υ)
  α_V  : …   -- (Act-assoc) consecutive lifts compose by ⊗_D     (mirrors …α)
  act_unit_assoc_V : …  -- triangle + pentagon (right D-actegory; eqToHom bridges the D-grading)
  υ_nat_V : …           -- υ_V naturality in V
  dist_coh_V : …        -- δ_V/δ0_V naturality (in V and in Dᵒᵖ)

-- Default: R = ℝ, the standard (×, +) semiring (multiply, then sum over contracted indices).
/-- The default target actegory: finite-dimensional/finitely-generated `R`-modules, encoded as
    Mathlib's `FGModuleCat R`. (The design's `FdMod` is not a Mathlib v4.30 name; `FGModuleCat` is
    its realization. `ModuleCat R` is the all-modules fallback.) -/
abbrev Mat (R : Type) [CommRing R] := FGModuleCat R

noncomputable instance : TargetActegory StObj (Mat ℝ) ℝ where
  actV := sorry   -- appends ℝ-typed dimensions; composition = matrix multiply over ℝ
  …               -- all coherence fields `sorry` (the Mat ℝ instance is the deferred obligation)

/-- The algebra functor F : C → V, a strong symmetric monoidal D-equivariant functor.
    Declared as a `class` (not `structure`) so that `ParaAlgebra` can extend it via
    typeclass inheritance, and so that `construct()` can be invoked via instance search. -/
class Algebra (D C : Type) (V : Type*) [ColoredPROP D] [ColoredPROP C] [Category V] [MonoidalCategory V]
    [SymmetricCategory V]
    (R : Type) [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R] where
  F        : C ⥤ V
  Fbraided : F.Braided          -- F is STRONG SYMMETRIC MONOIDAL (Mathlib `Functor.Braided`:
                                -- μ/ε invertible + pentagon/unitor + the braiding law);
                                -- `C` is symmetric via the Seam adapter, `V` by the class binder
  equivar  : ∀ (X : C) (P : Dᵒᵖ),
               F.obj (act.obj (X, P)) ≅ (TargetActegory.actV (D := D)).obj (F.obj X, P)  -- D-equivariance
  -- The equivariance/preservation coherence laws (real, non-vacuous equations):
  equivar_nat : …   -- equivar natural in the C-variable
  equivar_υ   : …   -- equivar carries C's υ to V's υ_V
  equivar_α   : …   -- equivar carries C's α to V's α_V
  equivar_δ   : …   -- equivar carries C's δ to V's δ_V, mediated by F's μ (from Fbraided.toMonoidal)
  F_ev_p      : …   -- F preserves the §4.1 evaluation ev_p (actV.map (𝟙, p) ≫ υ_V downstairs)
-- A morphism of algebras is a MonoidalNatTrans; weight tying collapses parameters via Δ.

class ParaAlgebra (D C : Type) (V : Type*) [ColoredPROP D] [ColoredPROP C] [Category V] [MonoidalCategory V]
    [SymmetricCategory V]
    (R : Type) [CommSemiring R] [DGradedColoredPROP D C] [TargetActegory D V R]
    extends Algebra D C V R where
  paraMap    : …   -- the Para(C) → Para(V) action on 1-cells: (P, f) ↦ (F P, paraMap P f)
  paraMap_eq : …   -- paraMap factors as the lax μ followed by F.map f
  weightTie  : …   -- weight tying as a reparameterization 2-cell (precompose parameter by Δ / F.map Δ)
-- Kept LIGHTWEIGHT: the action-on-1-cells + its μ-mediated law + the 2-cell law as obligations;
-- no double-/2-category machinery.
```

These laws are now stated rather than deferred: the `υ_V`/`α_V`/`δ_V` coherences (with triangle/pentagon and naturality), `F` strong-symmetric-monoidal via `Functor.Braided` under `[SymmetricCategory V]`, the `equivar_*`/`F_ev_p` equivariance/preservation laws, and the `paraMap`/`weightTie` `Para` refinement. Their further interpretation lives in the propositions and instantiation sections ([§10](#10-the-propositions-as-generic-theorems), [§11](#11-instantiation-and-future-extensions)) and the lightweight-`Para` note of [§12](#12-lean-formalization-notes); the trained model is a *section of the Para fibration over `∫Dat`*, tying the algebra back to the Grothendieck split of [§7.1](#71-the-structuredata-split-as-dat).

**The `R = Bool` target wrinkle.** §8 turns on the value-semiring parameter `R`: `R = ℝ` gives (`×`, then `Σ`) — the tensor/linear reading, realised over `Mat ℝ = FGModuleCat ℝ`; `R = Bool` gives (`∧`, then `∃`) — the predicate/relational reading. The wrinkle is *semantic*, not typechecking: the predicate reading needs the `(∨, ∧)` Boolean *semiring* (addition `∨` = `∃`), which is genuinely not a ring (`∨` has no additive inverse). But Mathlib's `Bool` *type* carries the Boolean *ring* (`+` = XOR, `*` = `∧`; via `BooleanRing.toCommRing`), so `true + true = false` and `Mat Bool = FGModuleCat Bool` *does* elaborate — over the WRONG (XOR) addition, not `∨`/`∃`. The predicate target therefore needs a separate `TargetActegory _ V Bool` over a relations / `(∨,∧)`-semimodule value category `V` (`∧`-then-`∃` Boolean matrix multiply, with `Bool` carrying the `(∨,∧)` semiring rather than XOR) — recorded as a deferred formalization obligation. The split is witnessed for now by the idempotency proxy `semiring_choice_split` (`(1:ℝ)+1 ≠ 1 ∧ true || true = true`, using `∨`/`||`, not XOR `+`).

**Flagship + propositions (now in Lean, signatures + `sorry`).** `instAlgebraBrMatR : Algebra StObj BrObj (Mat ℝ) ℝ` is the flagship instance (`Br` evaluates into ℝ-modules; all 8 fields `sorry`). Two propositions are stated and verified by `#print axioms` (each lists `sorryAx`): `construct_correspondence` — the `F_ev_p` law specialized, stating that `F` realizes `construct()`'s ℝ-valued read (plug the point `p`, then contract via the unitor); and `semiring_choice_split` — the Σ/×-vs-∃/∧ split *as the choice of `R`*, stated as a PROXY via additive idempotency (`(1:ℝ)+1 ≠ 1 ∧ true+true = true`) since the `Bool` target category is the deferred obligation above.

## 9. Acsets and the executable layer

### 9.1 `SBrInstance` as a finite presentation of an `∫Dat`-morphism

An acset instance is a **finite presentation of a single `∫Dat`-morphism**: its `C♯`-part is the connectivity, its `Dat`-part the sizes, coefficients, and datatypes ([graded_prop.md §5](graded_prop.md#5-the-structuredata-split-as-a-grothendieck-construction)). For `Br` the schema is `S_Br` and the instance is `SBrInstance`; the schema, the five entity types (`Axis`, `Equation`, `Array`, `ArrayAxis`, `Sample`), the C-set/attribute split, and the worked encoding are developed in [acset.md](acset.md) — referenced here, not re-derived. What matters for the Lean encoding is the relationship to [§7](#7-the-grothendieck-construction-and-executable-seam): an `SBrInstance` exported via `write_sbr`/`read_sbr` *is* a functor `G : S_Br → Set` — one point of the `S_Br`-instance category, i.e. one `∫Dat`-morphism — and its CSV tables are the C-set/attribute halves of that morphism written separately.

The `SBrInstance`'s four tables become a Lean `structure` whose fields are finite lists, mirroring acset.md's [Lean encoding section](acset.md#from-sbrinstance-to-a-diagram-in-br-a-lean-encoding):

```lean
structure SBrInstance where
  equations  : List EquationRow                 -- one row per Equation
  arrays     : List ArrayRow                    -- Array rows: slot, is_input, datatype_tag, op_predicate, …
  array_axes : List ArrayAxisRow                -- ArrayAxis rows: is_target, position (the Weave._shape interleaving)
  samples    : List SampleRow                   -- Sample rows: (src, tgt, coeff, offset) — the affine reindexing
  axis_sizes : List (UID × Numeric)             -- the Dat-part: each axis-UID's symbolic size
-- EquationRow, ArrayRow, ArrayAxisRow, SampleRow are defined in acset.md
-- (§"From SBrInstance to a Diagram in Br — a Lean Encoding"); see that document
-- for the full field lists. SBrInstance is partially specified here; the row types
-- complete the definition.
```

The full-fidelity Lean `SBrInstance` is implemented in `leanncd/LeanNCD/Acset/SBrInstance.lean` (`namespace LeanNCD.Acset`), mirroring the Python `acset/instances.py` exactly: `AxisType`/`AxisUID`, `OpTag` (10 variants), `DataTag` (3), `EquationRow`, the 12-field `ArrayRow`, `ArrayAxisRow`, `SampleRow`, and the five-table `SBrInstance`. To be **computable** (so CSV I/O can run — `Numeric` is noncomputable, [§2.1](#21-numeric)), axis sizes use the [§13.3](#133-abstract-syntax) `SizeExpr` and coefficients use `ℤ` (so the sketch's `axis_sizes : List (UID × Numeric)` is realized as `List (AxisUID × SizeExpr)`). This is the structure the CSV path of [§9.2](#92-the-seam-made-tangible) reads and writes.

acset.md interprets this `G` as a strict monoidal functor `D : J → Br` from a finite index category `J` (objects = the program's arrays, morphisms = its equations) into the `Br` of [§2.3](#23-br--free-category-over-broadcasted-base-morphisms), using two Mathlib shortcuts that are exactly this document's choices: **`MvPolynomial String ℕ` for `Numeric`** (so `ring` discharges the `StMat` laws — the `Dat`-fiber type of [§7.2](#72-freenumeric-is-the-fiber-not-a-layer)) and **`Matrix` for `StMat.coeffs`** (for the matrix lemmas). `J` is the *free strict monoidal category* on the equation quiver — Mathlib's `FreeMonoidalCategory` strictified — so specifying `D` on generators determines it uniquely; the functor laws are a consequence. The `axis_sizes` table populates the `Dat(c)` fiber; the `equations`/`arrays`/`array_axes`/`samples` connectivity is the `C♯`-morphism. This `D : J → Br` is the finite, written-down witness of one object/morphism of the `Grothendieck Dat'` instance of [§7.1](#71-the-structuredata-split-as-dat).

### 9.2 The seam made tangible

The acset tables and their CSV serialization are the **executable realization** of the `∫Dat` *specification* — the same proposition/computation seam as [§7](#7-the-grothendieck-construction-and-executable-seam), now in fully concrete form. `write_sbr`/`read_sbr` write the two halves of an `∫Dat`-morphism to separate tables — connectivity (the C-set part: `equations`, `arrays`, `array_axes`, `samples`) and data (the attribute part: `axis_sizes`, coefficients, datatypes) — which is precisely the Grothendieck split serialized. The same `Axis` UIDs appear in the term world's `Weave` objects and in the acset's `ArrayAxis` rows, so any `Context`-mediated unification (the [§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer) coequalizer computation) is reflected in both views without a round-trip. The categorical object `∫Dat` is what a Lean development *proves about*; the acset tables and CSV are what pyncd *computes and stores*. They meet at the seam.

There are thus **two ways to populate one `∫Dat`-morphism**, and they are complementary, not rival. The tensor-logic DSL of [§13](#13-the-tensor-logic-dsl) builds one statically, as a `ThreadedComposed` term ([§13.5](#135-semantic-compilation)); `read_sbr` builds one dynamically, as an `SBrInstance` read from CSV tables exported by the Python acset machinery. The acset extraction (`from_tensor_program`) turns a `ThreadedComposed` into an `SBrInstance`, and `write_sbr`/`read_sbr` round-trip that `SBrInstance` to and from CSV — so a morphism authored in the DSL and one read from CSV are the *same* object, and either may be checked against the other. Both routes share the [§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer) UID coequalizer and so agree on axis identity on the nose.

This agreement is realized in Lean in `leanncd/LeanNCD/Bridge/`. `Realize.lean` realizes the [§13.5](#135-semantic-compilation) computable presentation into the math tower — `realizeAxis`/`realizeStObj`/`realizeWeaveShape`/`realizeBrBaseP` are `sorry`-free, and `realize : ThreadedComposed → Σ dom cod, BrMorph dom cod` threads the routed DAG (the multi-input/permutation glue rests on `Br.tensorHom`/`swap`, so it is a `sorry`). `SBr.lean`'s `realizeSBr` realizes an `SBrInstance` ([§9.1](#91-sbrinstance-as-a-finite-presentation-of-an-dat-morphism)) as a `Br` morphism. `Agreement.lean` states `fromThreadedComposed` (the `from_tensor_program` extraction) and the **agreement Props** — `realize_fromThreadedComposed_agree : realize tc = realizeSBr (fromThreadedComposed tc)` (full equality of the two realized `Br` morphisms) plus `agree_dom`/`agree_cod` — as faithful, `sorry`-proved statements. So "DSL path = CSV path = same morphism" is now a stated Lean theorem, pending the `Br.tensorHom`/`swap` glue and the extraction algorithm. (Signatures + `sorry`, math-tower style.)

The CSV serialization is implemented in Lean in `leanncd/LeanNCD/Acset/` and is **fully executable and `sorry`-free**: `Csv.lean` is the field/row codec and `Io.lean` provides `writeSBr : SBrInstance → List (String × String)` and `readSBr : List (String × String) → Except CsvError SBrInstance` over the exact five-table format Python's `acset/csv_io.py` emits (same column orders, encodings, and `\r\n` terminators). Validated two ways: the Lean round-trip `readSBr (writeSBr inst) = .ok inst`, and a **byte-for-byte cross-check against a Python `write_sbr` fixture** (`test/Acset/fixtures/`, from `acset.csv_io.write_sbr` on a Python `SBrInstance`) — Lean's `readSBr` parses the exact Python CSVs and `writeSBr` reproduces them identically. The CSV path of "two ways to populate one morphism" is concretely realized on the Lean side and interop-tested against the Python `read_sbr`/`write_sbr`; only the noncomputable `realizeSBr`/`fromThreadedComposed`/agreement *proofs* above remain `sorry` (the data round-trip is done).

### 9.3 Lean tower reference

The table below collects the Lean type names from each layer of the tower, with implementation notes. Inline pointers in earlier sections give local context; this is the consolidated view.

| Lean | Python | Notes |
| --- | --- | --- |
| `SmallCategory` / `ColoredPROP` | implicit / `ProductCategory` | category and monoidal laws are unstated in Python; paper-level only |
| `ColoredPROP.elemental` | — | new `(Elem)` field; no Python witness |
| `List gen` (objects) | `ProdObject[L]` | Python wraps `tuple[L,…]` in a Term; Lean uses `List` directly |
| `StMat` | `StrideMorphism` | stride *matrix* (`Matrix … Numeric` + bias) vs bundled stride record |
| `BrBase` | `Broadcasted` | base op + reindexings; `Fin`-indexed weaves vs runtime tuples |
| `BrMorph` | `Composed` | free list of `BrBase` vs `content: tuple[M,…]` |
| `ThreadedComposed` | `ThreadedComposed` | routed presentation of a `BrMorph` ([§13.5](#135-semantic-compilation)); the DSL's output. `from_tensor_program` extracts its `SBrInstance` |
| `ProductOfMorphisms` ↔ `tensorHom` | `ProductOfMorphisms[L, M]` | `ColoredPROP.tensorHom` (a morphism) vs a data wrapper |
| `DGradedColoredPROP.act` | batch lift `[f,P]` | the lift action; `[f,P] = act(f, 𝟙_P)`, `[X,η] = act(𝟙_X, η)` |
| `WeaveShape` / `structure Weave` | `Weave._shape` | per-array shape (`WeaveShape`); [§5](#5-weaves-as-cartesian-lift-data)'s `structure Weave` = the cartesian-lift factorization witness |
| `∫Dat` instance (`Grothendieck Dat'`) | `SBrInstance` | finite presentation of one `∫Dat`-morphism |
| `Numeric` = `MvPolynomial String ℕ` | `Numeric` / `FreeNumeric` | the `Dat(c)` fiber; `MvPolynomial.X s` ↔ a `FreeNumeric` generator |
| `Context` / `EqClass` | `Context` / `EqualityClass` | union-find = the *implementation* of the coequalizer ([§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer)) |
| `FreshM` = `EStateM CompileError ℕ` | random-int UID side-effect | fresh-name counter + validation errors; Lean threads state explicitly, Python mutates a global source and raises exceptions |
| `TermTraversable` | `deep_reconstruct` | per-type traversal instance vs `__dataclass_fields__` reflection |
| `Algebra.F` | `ConstructedModule.construct()` | the algebra functor `C → V` (full class in the [§8](#8-algebras-and-construct) / propositions development) |
| `DynamicName` | `DynamicName` | display only — see [§14](#14-appendix-out-of-scope), out of scope |

The table's organizing principle is the proposition/computation seam of [§1](#1-orientation-one-structure-one-seam): each type is placed by its categorical role — fiber datum, coequalizer implementation, fresh-name counter, traversal, or display — on one side or the other. There is one tower with one seam.

## 10. The propositions as generic theorems

Every proposition of [graded_prop.md §8](graded_prop.md#8-propositions-the-synthesis-organizes) is a statement about the **core class fields and axioms only**. Each begins with the same generic preamble:

```lean
variable {D C : Type} [ColoredPROP D] [ColoredPROP C] [DGradedColoredPROP D C]
```

and mentions nothing beyond `act`, `δ`/`δ0`/`υ`/`α`, `sh`, and the named `Prop`-fields (`sh_act`, `act_unit_assoc`, `υ_nat`, `dist_coh`, `broadcast_gen`, plus the base `elemental`). Because the only hypotheses are class members, each proposition is **proved once, at the graded-PROP level, and inherited at every instance** — the `DGradedColoredPROP StObj BrObj` of [§11](#11-instantiation-and-future-extensions), the `DGradedColoredPROP BrObj CMod` MoE level, a future `Graph→C`, and the swapped-`D` rows all receive it with no per-domain proof. This is the Lean form of [graded_prop.md](graded_prop.md)'s central promise, and it *is* parametricity over a typeclass: a `theorem` whose only free assumption is `[DGradedColoredPROP D C]` applies verbatim wherever that instance resolves.

| Proposition | Lean statement (sketch) | Mathlib machinery | Per-instance cost |
| --- | --- | --- | --- |
| **8.1** Lift functoriality / distribution | `[f ; g, P] = [f, P] ; [g, P]` and `[f ⊗ g, P] = [f, P] ⊗ [g, P]` | `act` is a `Functor` (so `Functor.map_comp`) + the `δ` distributivity iso | free |
| **8.2** Weave uniqueness | `Subsingleton (Weave g)` | `elemental` (base) + `broadcast_gen` — elements separate, so `P` and the target/tiling partition are determined | free |
| **8.3** Grothendieck splitting | `C ≅ ∫Dat` | `CategoryTheory.Grothendieck` on `Dat` | `Iso.refl` if `C` is built as `∫Dat`; otherwise one constructed equivalence |
| **8.4** Equivariance | `F` is `T`-equivariant `↔` `F` lifts to the EM-category of the symmetry monad `T` (morphism of `T`-algebras) | EM-category / `Monad.Algebra`; gated on `SymmetryGraded` (the `T : Monad D` parameter of [§6.2](#62-route-and-symmetry-stubs)) | body deferred — finite-`G` is reachable now, the graded-PROP-dependent parts wait; see [equivariance_unification.md](equivariance_unification.md) |
| **8.5** Composition associativity | composition in `Inst(C♯)` is associative and unital | `CategoryTheory.Limits.pushout` + the pasting lemma | free (strictified by canonical representatives, so the pasting is definitional) |
| **8.6** Two obstruction species | a morphism failing to admit a weave does so as `Scan` (species i, coupled but data-independent) or `Route` (species ii, data-dependent); the litmus is whether the reindexing is a fixed, point-natural `D`-morphism | the litmus is `D`-uniform — phrased entirely over `act` and Eq. 3, no `D`-specific input | abstract litmus free; positive identification of species (i) as `Scan` needs `TemporalGraded` (its `L`), as 8.7–8.8 |
| **8.7** `Scan` as a catamorphism | `Scan := cata(step)`; the prefix-restriction law is a corollary of the catamorphism universal property | `TemporalGraded` (the `iterate`/`trace` fields of [§6.1](#61-temporalgraded--scan)) | free given `TemporalGraded`; explains the affine fast path (step algebra factors through a **monoid** → parallel prefix in `O(log N)`) |
| **8.8** `Scan` batches | `act(Scan, P) ≅ Scan(act(step, P))` for `P` orthogonal to `L` | `lift_fold_dist` ([§6.1](#61-temporalgraded--scan)) | free given `TemporalGraded` |

The inheritance is about the **theorems**, not the instance obligations. When a new `instance : DGradedColoredPROP D C` is declared, the propositions above transfer to it for free — but the instance must still **discharge the coherence `Prop`-fields** of the core: the actegory triangle and pentagon (`act_unit_assoc`), unitor naturality (`υ_nat`), the distributivity coherences (`dist_coh`), and `sh_act`/`broadcast_gen`/`elemental`. "Inherit everywhere" names the proven propositions riding on those fields; it does not waive the obligation to *supply* the fields. Each new domain pays that fixed, finite coherence cost once; everything built on top of the core is then free.

In the current Lean transcription, **8.1** (lift functoriality) is *proved sorry-free* (`act.map_comp`) and **8.8** (Scan batches) is a sorry-free re-export of `lift_fold_dist`; **8.2** (weave uniqueness) and **8.7** (Scan-as-catamorphism) are stated genuinely with `sorry` proofs; **8.3** (the Grothendieck split) is in [§7.1](#71-the-structuredata-split-as-dat). **8.4/8.5/8.6 are stated only in prose here, not yet as Lean theorems**: 8.4 needs a genuine equivariance field on `SymmetryGraded` (currently a stub), 8.5 needs the `Inst(C♯)`/pushout formalization ([§7.3](#73-composition-as-pushout)), and 8.6 needs a classification datum — they are deferred rather than written as vacuous placeholders.

## 11. Instantiation and future extensions

### 11.1 `D = St`, `C = Br` — the flagship instance

Today's instantiation is `D = St`, `C = Br`: an index PROP of axis lengths (colors = `Numeric` sizes, morphisms = stride matrices) grading an operation PROP of broadcasted arrays. In Lean the *types* are `StObj`/`BrObj` (`St`/`Br` name the `ColoredPROP` *instances*), so the instance is `DGradedColoredPROP StObj BrObj`; it is `noncomputable` (it builds data over `Numeric`). The instance header supplies the core fields; the named `Prop`-fields are discharged by the laws established in [§2](#2-the-base-coloredprop)–[§4](#4-the-core-dgradedcoloredprop-d-c). `sh` is concrete (`fun a => a.shape`); the lift `act`, the coherence isos, and the laws are the genuine but deferred content.

```lean
noncomputable instance : DGradedColoredPROP StObj BrObj where
  sh    := fun a => a.shape        -- the array's shape: sh([a, A]) = A
  act   := …                       -- batch lift + reindexing (theory.md Lift Operations)
  δ     := …                       -- [X ⊗ Y, P] ≅ [X,P] ⊗ [Y,P]   (batch lift distributes)
  δ0    := …                       -- [I, P] ≅ I
  υ     := …                       -- [X, I_St] ≅ X   (grading by the unit shape is trivial)
  α     := …                       -- [[X,P],Q] ≅ [X, Q ⊗ P]
  sh_act         := …              -- (Sh-⊛): sh*([X,P]) = sh*(X) ⊗ P
  act_unit_assoc := …              -- actegory triangle + pentagon, by St affine-stride algebra
  υ_nat          := …              -- unitor naturality, by the batch-lift defn
  dist_coh       := …              -- δ/δ0 naturality + interchange, from the batch-lift defn
  broadcast_gen  := …              -- every Br morphism factors as lam ; [f,P] ; ρ (Def 13)
  elemental      := …              -- Br is elemental (base ColoredPROP field)
```

| Core field | pyncd realization |
| --- | --- |
| `sh` | `sh([a, A]) = A` — the array's shape ([theory.md §Objects in Br](theory.md#objects-in-br)) |
| `act` | the batch lift `[f, P]` + object/morphism reindexing + broadcasted-stride lift ([theory.md §Lift Operations](theory.md#lift-operations)) |
| `δ` | `[X ⊗ Y, P] = [X,P] ⊗ [Y,P]` ([theory.md §Batch Lift](theory.md#batch-lift-f-p-def-11)) |
| `δ0` | `[I, P] ≅ I` — lifting the monoidal unit is trivial |
| `υ` | `[X, I_St] ≅ X` — grading by the unit (empty) shape is the identity reindexing |
| `α` | `[[X,P],Q] ≅ [X, Q ⊗ P]` — nested lifts compose the batch shapes |
| `sh_act` | `(Sh-⊛)`: a lifted array's shape is the base shape tensored with the batch shape `P` |
| `act_unit_assoc` | actegory triangle/pentagon, discharged by `St`'s affine-stride matrix algebra (`Matrix.mul_assoc` + `ring`) |
| `υ_nat` | unitor naturality `[f, I_St] ; υ_Y = υ_X ; f`, from the batch-lift definition |
| `dist_coh` | `δ`/`δ0` naturality and interchange with `υ`/`α`/swap, from the batch-lift definition |
| `broadcast_gen` | `(Broadcast-gen)`: every `Br` morphism is a broadcasted operation ([theory.md §Broadcasting](theory.md#broadcasting)) |
| `elemental` | `Br` is elemental ([theory.md §Elemental Categories](theory.md#elemental-categories)) — the base `ColoredPROP` field |

Every row is a definitional unfolding of the core with `D := St`, `C := Br`. With the instance in place, all of [§10](#10-the-propositions-as-generic-theorems) holds for `Br` with no `Br`-specific proof.

### 11.2 The additive-extension menu

The framework grows by **adding instances and mixins**, never by editing the proven core. Each direction below is one or the other.

| Extension | Lean addition | Kind |
| --- | --- | --- |
| `D = Br` mixture-of-experts | `instance : DGradedColoredPROP BrObj CMod` ([graded_prop.md §9.2](graded_prop.md#92-the-speculative-third-level-d--br)) — models as wires, `⊛` tiles a base computation over a family of models | **new instance** |
| swap-`D`: graph / incidence cat. | `instance : DGradedColoredPROP Graph C` — gather-along-edge reindexing; GNNs, meshes (fixed graph = weave; per-sample graph = `Route`) ([graded_prop.md §9.3](graded_prop.md#93-the-horizontal-axis-swapping-d)) | **new instance** |
| swap-`D`: group `BG` / `Rep(G)` | `instance : DGradedColoredPROP (Rep G) C` — group-translation reindexing; equivariant & steerable nets | **new instance** |
| swap-`D`: Markov cat. `Stoch` | `instance : DGradedColoredPROP Stoch C` — Markov-kernel reindexing; sampling, VAE, SMC | **new instance** |
| swap-`D`: metric / enriched cat. | `instance : DGradedColoredPROP Metric C` — distance-kernel reindexing; continuous conv, neural fields | **new instance** |
| swap-`D`: partition lattice | `instance : DGradedColoredPROP Partition C` — assignment-map reindexing; pooling, clustering, slots (fixed = weave; learned = `Route`) | **new instance** |
| swap-`D`: resource monoid | `instance : DGradedColoredPROP Resource C` — store-vs-recompute reindexing; checkpointing / scheduling ([future_ideas.md §8](future_ideas.md#8-prioritized-implementation-roadmap) item 4.5) | **new instance** |
| data-dependent routing | `class RouteStructure` ([§6.2](#62-route-and-symmetry-stubs)) — Prop 8.6(ii), the gate as a `Para` parameter | **new mixin** |
| equivariance | `class SymmetryGraded` ([§6.2](#62-route-and-symmetry-stubs)) — Prop 8.4 via the EM-category of `T : Monad D`, gated | **new mixin** |
| weight tying / passes | `class ParaAlgebra` ([§8](#8-algebras-and-construct)) — `Para(C) → Para(V)` 2-functor, passes-as-2-cells | **new mixin** |
| unbounded recurrence | corecursion / coalgebra refinement of `TemporalGraded` — the `cata`/`ana` companion | **new mixin** |

No row is a core edit: a new domain is a new `instance` (it supplies the core fields and discharges the coherence obligations of [§10](#10-the-propositions-as-generic-theorems)), and a new capability is a new mixin layered on top (`extends DGradedColoredPROP`, or `extends Algebra` for `ParaAlgebra`). The classification of [graded_prop.md §8.6](graded_prop.md#8-propositions-the-synthesis-organizes) is `D`-uniform, so every swap-`D` row inherits the same weave-vs-`Scan`-vs-`Route` decision procedure.

The MoE level deserves a specific word, because it is the one place the same category `Br` (type `BrObj`) appears twice. The vertical stack `D = Br` reuses **all** of [§10](#10-the-propositions-as-generic-theorems) with zero new proof, and this is not a coincidence — it is forced by the instance-resolution discipline. `Br`-as-graded — the `DGradedColoredPROP StObj BrObj` instance of [§11.1](#111-d--st-c--br--the-flagship-instance) — occupies the **`C` position** of `DGradedColoredPROP`. `Br`-as-index — the `[ColoredPROP BrObj]` argument of `instance : DGradedColoredPROP BrObj CMod` — occupies the **`D` position**. The two are different parameter slots of the class, so the two roles of `Br` never collide in instance search: declaring `DGradedColoredPROP BrObj CMod` does not overlap or shadow `DGradedColoredPROP StObj BrObj`. Because the [§10](#10-the-propositions-as-generic-theorems) theorems are generic in both `D` and `C`, they fire for `DGradedColoredPROP BrObj CMod` the instant it resolves, exactly as they fire for `DGradedColoredPROP StObj BrObj` — the MoE level is new data, not new mathematics.

## 12. Lean formalization notes

The strategy of [graded_prop.md §10](graded_prop.md#10-lean-formalization-notes) reads as a Mathlib shopping list, and most of the tower lands on existing `CategoryTheory.*` machinery. The base colored PROP and the seam adapter ([§2](#2-the-base-coloredprop)–[§3](#3-the-seam-adapter-into-mathlib)) are `CategoryTheory.MonoidalCategory` / `SymmetricCategory` over `FreeMonoidalCategory (Discrete O)`. The Grothendieck split of [§7.1](#71-the-structuredata-split-as-dat) is `CategoryTheory.Grothendieck` applied to the data functor `Dat'`. The composition-as-pushout of [§6](#6-mixins-scan-route-symmetry-para)/[§7.3](#73-composition-as-pushout) and the associativity of [Prop 8.5](#10-the-propositions-as-generic-theorems) are `CategoryTheory.Limits.pushout` plus the pasting lemma. The `Algebra` functor `F : C ⥤ V` and its morphisms are `MonoidalFunctor` / `MonoidalNatTrans`. The gated equivariance of [§6.2](#62-route-and-symmetry-stubs)/[§8.4](#10-the-propositions-as-generic-theorems) is `CategoryTheory.Action` / `Rep` / `Monad.Algebra` (the Eilenberg–Moore category of the symmetry monad). And the architecture relations `R` quotienting `C♯` ([§7](#7-the-grothendieck-construction-and-executable-seam)) are `CategoryTheory.Quotient`. The `D`-actegory coherence bundle of the core — the triangle/pentagon and the distributivity isos — is the one piece Mathlib carries only partially (`Action` covers a monoid acting, not a full monoidal-category actegory), so it is hand-rolled as functor-plus-natural-iso algebra; it is routine, not deep.

Beyond the coverage map, several honest notes shape any transcription:

- **Strictification strategy.** Build the Mathlib `MonoidalCategory` structure directly from the `ColoredPROP` data, taking the structural isos to be **`eqToIso` of the strictness laws** (`tensor_assoc`, `tensor_unit_l`, `tensor_unit_r`) already proved in [§2](#2-the-base-coloredprop). This makes the category strict by construction and lets a `ColoredPROP` law stated with `=` line up with a Mathlib `MonoidalCategory` whose coherences are isos, applied at the seam adapter of [§3](#3-the-seam-adapter-into-mathlib). It builds **on** the sorry-free `Category` instance (which just forwards `SmallCategory`), so there is no second `Category` instance and no diamond — preferred over carrying the tower over `FreeMonoidalCategory (Discrete O)` and transferring along an equivalence.

- **Strictness is the real friction.** The monoidal-seam coherences are the *single* genuine difficulty, and they split in two. `ColoredPROP` now carries the morphism-level **bifunctoriality + involution** laws — `tensorHom_id` (`tensorHom 𝟙 𝟙 = 𝟙`), `tensorHom_comp` (interchange), and `swap_swap` (`swap ; swap = 𝟙`) — proved nowhere generically but supplied per instance. Over the `eqToIso` strict structure these discharge the *bifunctorial* coherences of the seam (`tensorHom_def`, the `whiskerLeft`/`whiskerRight` identities, `pentagon`, `triangle`, and the braiding `hom_inv_id`/`inv_hom_id`/`symmetry`) sorry-free. What they do **not** imply — and what currently stays stated-with-`sorry` — are the *deeper* coherences: the associator/unitor **naturalities** (which need a `tensorHom`-vs-structural-equality transport coherence, or definitional strictness of `tensor`, neither available for `List.append`'s propositional associativity) and the **braiding naturality + hexagon** identities. Encoding those as further `ColoredPROP` fields would amount to re-stating Mathlib's full symmetric-monoidal axiom set on the lightweight base, so they are left as honest seam obligations (they are not consumed until the [§10](#10-the-propositions-as-generic-theorems) theorems). Nothing else in the development fights the type theory.

- **Inheritance is for theorems, not obligations.** The [§10](#10-the-propositions-as-generic-theorems) theorems transfer to every instance for free, because their only hypothesis is `[DGradedColoredPROP D C]`. But each new `instance` must still **discharge the coherence `Prop`-fields** of the core — the actegory triangle and pentagon (`act_unit_assoc`), the distributivity coherences (`dist_coh`), and `sh_act`/`broadcast_gen`. "Prove once, inherit everywhere" names the proven propositions; it does not waive the per-instance obligation to *supply* the coherence fields.

- **Instance-resolution discipline.** `D` and `C` are **explicit** class parameters, not `outParam`s. With both free, Lean's instance search needs them pinned, so `[DGradedColoredPROP D C]` resolves predictably and `Br`-as-graded (the `C` slot) never collides with `Br`-as-index (the `D` slot) in search — the non-collision relied on by the MoE level of [§11.2](#112-the-additive-extension-menu).

- **Mixins, not a tall tower.** Scan, Route, Symmetry, and Para are kept as composable mixin classes layered on the core ([§6](#6-mixins-scan-route-symmetry-para)), rather than as one deep `extends` chain. A tall tower would invite typeclass *diamonds* (a capability reachable by two `extends` paths) and degrade instance-search performance; independent mixins let each instantiation pay only for the capabilities it declares.

- **The Para 2-category gap.** The `Para(V)` encoding is kept **lightweight and 1-categorical**: a parameterized 1-cell is `Σ (P : V), (P ⊗ A ⟶ B)`, and the "2-cells" (reparameterizations, weight tying) are carried as an explicit reparameterization *relation* rather than as genuine 2-morphisms of a bicategory. The gap to a full `Para` bicategory is **noted, not paid for** — a full bicategorical implementation would be machinery only the specification uses ([graded_prop.md §7](graded_prop.md#7-algebras-construct-and-the-para-refinement)).

- **Equivariance is gated.** Proposition 8.4's body depends on the `SymmetryGraded` mixin and the Eilenberg–Moore-category machinery for the symmetry monad `T`. The finite-group case is reachable with present Mathlib (`Action`/`Rep`), but the graded-PROP-dependent parts of the encoding wait on this very formalization being in place; the proposition is *stated* now and its proof *gated* ([equivariance_unification.md](equivariance_unification.md)).

## 13. The tensor-logic DSL

A Lean 4 DSL embedding for tensor logic, following the syntax-category + elaboration pattern of the Lean 4 metaprogramming book (ch. 8): a BNF grammar defines the surface language; Lean inductive types give the abstract syntax; `declare_syntax_cat`/`syntax` rules connect them to Lean's parser; value-returning `elabXxx : Syntax → MetaM <value>` functions walk the syntax tree (building concrete AST *values*, not `Expr`s — see [§13.4](#134-concrete-syntax-and-elaboration)); and `TLProgram.compile : TLProgram → FreshM ThreadedComposed` lowers programs to morphisms in `Br`.

The **front-end** — [§13.2](#132-bnf-grammar) (grammar), [§13.3](#133-abstract-syntax) (AST), [§13.4](#134-concrete-syntax-and-elaboration) (concrete syntax + elaborators) — is implemented in `leanncd/LeanNCD/DSL/` and is **fully executable and `sorry`-free**: the entry point `tlprog!{ … } : TLProgram` (Stage 1) parses surface syntax into a concrete `TLProgram` value, and all seven [§13.2](#132-bnf-grammar) examples parse. The **back-end** — the `TLProgram.compile` pipeline of [§13.5](#135-semantic-compilation) and the `tl!{}` compile macro (Stage 2) — is implemented in `leanncd/LeanNCD/DSL/Pipeline/` (the noncomputable `ThreadedComposed → BrMorph` bridge is in `leanncd/LeanNCD/Bridge/`; see [§13.5](#135-semantic-compilation)). The text below marks where the implementation generalized or deferred relative to this design (axis sizes carried in a computable `SizeExpr`, value-returning elaborators, `Stmt = assign | scatter`, general-affine reads, n-ary products/sums).

Compilation is a two-stage process. **Stage 1** (`MetaM`): `elabTLProgram` parses concrete syntax into a typed `TLProgram` value. **Stage 2** (`FreshM`): `TLProgram.compile` lowers the program to a `ThreadedComposed` morphism, minting fresh UIDs and validating semantic constraints via the `FreshM` monad of [§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer). The entry point runs Stage 2 at elaboration time, embedding the resulting `ThreadedComposed` as a compile-time constant:

```lean
elab "tl!{" p:tl_program "}" : term => do
  let prog ← elabTLProgram p                            -- Stage 1: MetaM TLProgram value
  match TLProgram.compile prog |>.run 0 with            -- Stage 2: run FreshM at elaboration time
  | .ok tc _ => return toExpr tc                        -- embed ThreadedComposed as term constant
  | .error e _ => throwError s!"tl!{{...}}: {repr e}"  -- surface CompileError to Lean's elaborator
-- toExpr requires: deriving Lean.ToExpr on ThreadedComposed and all nested types (see §13.3)
```

The Stage-1 entry point is a standalone macro — `tlprog!{ … } : TLProgram` (`elab "tlprog!{" p:tl_program "}" : term => do return Lean.toExpr (← elabTLProgram p.raw)`) — which parses surface syntax into a `TLProgram` value and embeds it via `ToExpr`. The full `tl!{}` above additionally runs Stage 2 (`compile`), as described in [§13.5](#135-semantic-compilation).

### 13.1 Quick start: entry points and examples

The DSL is implemented in `leanncd/LeanNCD/DSL/`. Two import paths expose the two entry points:

```lean
import LeanNCD.DSL.Elab    -- Stage 1 only: `tlprog!{…} : TLProgram`
import LeanNCD.DSL.Compile  -- Stage 1 + 2: `tl!{…} : ThreadedComposed`
```

**Stage 1 — parse to AST.** `tlprog!{…}` elaborates surface syntax into a `TLProgram` value at compile time with no lowering to `Br`. Useful for inspecting the parsed AST or writing parse-level tests:

```lean
import LeanNCD.DSL.Elab

private def mm : TLProgram := tlprog!{ Y[i,j] := W[i,k] · X[k,j] }
#check mm       -- TLProgram
#eval repr mm   -- inspect the parsed AST
```

**Stage 1 + 2 — parse and compile.** `tl!{…}` additionally runs `TLProgram.compile` (the [§13.5](#135-semantic-compilation) pipeline) at elaboration time, mints UIDs, validates semantic constraints via `FreshM`, and embeds the resulting `ThreadedComposed` as a compile-time constant. A `CompileError` (shape mismatch, causality violation, missing base case, etc.) surfaces as a `throwError` during Lean elaboration:

```lean
import LeanNCD.DSL.Compile

private def mm := tl!{ Y[i,j] := W[i,k] · X[k,j] }
#check mm             -- ThreadedComposed
#eval mm.nExternal    -- 2   (W and X are external inputs)
#eval mm.steps.length -- 1
```

**Evaluating on concrete `Float` tensors.** `TLProgram.eval` ([§13.6](#136-evaluation)) is a reference evaluator that runs a `TLProgram` on named input tensors and returns named output tensors:

```lean
import LeanNCD.Eval.Eval
open Std

def main : IO Unit := do
  let env := (({} : HashMap String DenseTensor)
    |>.insert "W" ⟨[2,3], #[1,2,3, 4,5,6]⟩
    |>.insert "X" ⟨[3,2], #[1,0, 0,1, 1,1]⟩)
  match TLProgram.eval (tlprog!{ Y[i,j] := W[i,k] · X[k,j] }) env with
  | .ok out  => IO.println s!"Y = {repr out[\"Y\"]?.get!.data}"
  | .error e => IO.println s!"Error: {repr e}"
```

**Test files.** Three files exercise the complete pipeline from surface syntax to numeric output:

| File | Entry point | What it checks |
| --- | --- | --- |
| `leanncd/test/DSL/ParseExamplesTest.lean` | `tlprog!{}` | Stage 1: the five [§13.2](#132-bnf-grammar) example programs parse |
| `leanncd/test/DSL/CompileExamplesTest.lean` | `tl!{}` | Stage 1+2: the five examples compile; structural `#guard` assertions |
| `leanncd/test/Eval/EvalExamplesTest.lean` | `TLProgram.eval` | End-to-end: thirteen programs evaluated with numeric assertions |

**Running the tests.** From `leanncd/`:

```bash
lake build   # build all modules (DSL, Eval, Bridge, Acset)
lake test    # run the full test suite
```

### 13.2 BNF grammar

Extends Domingos' tensor-logic notation (implicit Σ over contracted axes, Einstein product) with: axis typing (ℝ/ℕ/norm), tensor declarations, Iverson predicates, nonlinearities with optional masks, affine index arithmetic (Slice/Reindex/Scatter), and temporal recursion (Scan).

```text
-- Layer 1: Axis specifications and declarations
decl        ::= 'tensor'    name ':' shape
              | 'predicate' name ':' shape
              | 'linear'    name ':' in_shape '→' out_shape ['bias']
              | 'axis'       name ':' axis_kind ['=' n]   -- declares an axis's dtype + optional pinned size

-- A tensor's shape lists only its axis NAMES (and order). An axis's dtype/size lives in an
-- `axis` declaration; the softmax/normalize reduction axis is marked on the output slot (`m.`).
shape       ::= '(' ')'
              | '(' axis_spec (',' axis_spec)* ')'

axis_spec   ::= name

axis_kind   ::= 'ℝ'    ['[' size ']']   -- real axis;          bracket = size (else symbolic)
              | 'ℕ'    ['[' size ']']   -- discrete axis
-- An `axis l : ℕ = 3` declaration pins a concrete loop/iteration extent that no input tensor's
-- shape would otherwise fix (it seeds size inference); omitting `= n` declares dtype only.

-- A size is a symbolic dimension term. In the implemented front-end it elaborates to a
-- computable `SizeExpr` (§13.3), the DSL mirror of `Numeric` (§2.1 / §7.2).
-- Omitting the bracket leaves the size a fresh generator, minted in Stage 2.
size        ::= n                       -- literal (SizeExpr.lit)
              | name                    -- symbolic generator (SizeExpr.var)
              | size '*' size
              | size '+' size
              | '(' size ')'

-- Layer 2: Index expressions (general integer-affine; n ∈ ℤ)
-- The implementation generalized the single-term forms below to a left-associative
-- '+'/'-' sum of terms (each term a bare axis_name, a literal n, or n '*' axis_name),
-- so reads like `i+p` and `2*j+r` parse. Symbolic-coefficient strides (`s*j`, an
-- ident coefficient) remain out of scope — IdxExpr carries integer coefficients only.
idx_expr    ::= term (('+' | '-') term)*
term        ::= axis_name
              | n
              | n '*' axis_name
              | '(' idx_expr ')'

-- Layer 2.5: Predicate arithmetic (extends idx_expr with non-affine products and absolute value)
-- Only valid inside bool_expr; forbidden in tensor index slots.
pred_term   ::= idx_expr
              | 'imul(' pred_term ',' pred_term ')'
              | '|' pred_term '|'                   -- integer absolute value (e.g. |i−j| ≤ n)
              | '(' pred_term ')'

-- Layer 3: Iverson predicates
bool_expr   ::= pred_term rel_op pred_term
              | bool_expr '∧' bool_expr
              | bool_expr '∨' bool_expr
              | '¬' bool_expr
              | 'ieq(' pred_term ',' pred_term ')'
              | '(' bool_expr ')'

rel_op      ::= '<' | '≤' | '=' | '≠' | '>' | '≥'

-- Layer 4: RHS expressions
rhs         ::= nonlin '(' sum_expr ')'
              | sum_expr

sum_expr    ::= prod_term ('+' prod_term)*

prod_term   ::= factor ('·' factor)*

factor      ::= name '[' idx_expr (',' idx_expr)* ']'
              | '[' bool_expr ']'

nonlin      ::= 'relu'
              | 'softmax'
              | 'softmax'   '(' 'where' bool_expr ')'
              | 'normalize'
              | 'normalize' '(' 'where' bool_expr ')'

-- Layer 5: Statements
stmt        ::= assign | base_case | recur_step | scatter_write

assign      ::= name '[' out_slot (',' out_slot)* ']' ':=' rhs

-- An output slot is an axis name, optionally suffixed `.` to mark it the softmax/normalize
-- reduction axis (at most one per stmt; required when the RHS applies softmax/normalize).
out_slot    ::= axis_name | axis_name '.'

-- l+1 and 0 may appear in any slot position; the iteration axis l is identified by the l+1 slot
base_case   ::= name '[' base_slot_list ']'  ':=' rhs
recur_step  ::= name '[' recur_slot_list ']' ':=' rhs

base_slot_list  ::= (out_slot ',')* n (',' out_slot)*
recur_slot_list ::= (out_slot ',')* axis_name '+' '1' (',' out_slot)*

-- Affine LHS: every slot is a (possibly affine) output coordinate
scatter_write ::= name '[' affine_slot (',' affine_slot)* ']' ':=' rhs
                    ['fill' n] ['reduce' 'sum']

affine_slot ::= axis_name
              | n '*' axis_name
              | axis_name '+' n
              | n '*' axis_name '+' n

-- Layer 6: Programs
program     ::= decl* stmt+
```

**Contracted axes** are implicit: any `axis_name` appearing in a `prod_term` but absent from the LHS is summed over — Domingos' convention, unchanged.

**Coupled scans** require no special syntax: two `recur_step` stmts for different tensor names whose iteration axis (the `axis_name` in `axis_name '+' '1'`) carries the same UID are automatically grouped into a coupled `Scan` (`n_states > 1`) by the semantic compiler.

**Semantic constraints** enforced by the compiler, not the grammar:

- `l+1` on the RHS of a `recur_step` is a causality violation and is rejected, where `l` is that step's iteration axis; look-ahead reads on non-iteration axes are permitted
- Scatter with overlapping writes requires `reduce sum`
- A `recur_step` without a matching `base_case` for the same name is an error
- A `linear`-declared weight must multiply exactly one activation factor

**Nine representative examples** (two exercise predicates; the last two are the transformer, flat and scanned):

```text
-- Matmul (Domingos base: k is contracted)
Y[i,j] := W[i,k] · X[k,j]

-- Causal masked attention (norm axis marked `s.` + Iverson mask)
tensor A : (q, s)
A[q,s.] := softmax(where s ≤ q)(Q[q,d] · K[s,d])

-- Strided convolution (affine Reindex reads). NOTE: a symbolic stride `s*j` needs an ident
-- coefficient, which integer-coefficient IdxExpr cannot carry (§13.3); the parsed form uses a
-- concrete stride, e.g. `X[i+p, 2*j+r]`.
Y[i,j] := W[p,r] · X[i+p, s*j+r]

-- Upsample 2× (affine Scatter write)
tensor Out : (i, j)
Out[2*i, 2*j] := X[i,j]

-- Coupled scan: G and H share iteration axis l (coupled Scan, n_states=2)
G[j, 0]   := X[j]
G[j, l+1] := relu(G[j,l] · W_G[j,k] + H[j,l] · U[j,k])
H[j, 0]   := Y[j]
H[j, l+1] := relu(H[j,l] · W_H[j,k] + G[j,l] · V[j,k])

-- Predicate declaration + masked aggregation: edge(i,j) is a Bool-typed adjacency
-- predicate gating a doubly-contracted feature product (all indices contracted → scalar)
predicate edge : (i, j)
Result[] := F[t,i] · F[t,j] · edge[i,j]

-- Iverson-bracket predicate: a tridiagonal band mask via |·| (integer absolute value)
Band[i,j] := A[i,j] · [|i − j| ≤ 1]

-- Single-layer transformer (L=1 unrolled, Option A — papers/transformer_example.md).
-- Nine equations flat (no scan): Q/K/V projections (contract m), causal masked softmax
-- (norm s), SV aggregation (contract s), output projection (contract h,k), attention
-- residual + normalize (norm m), FFN relu in (contract m), FFN out (contract d), FFN
-- residual + normalize (norm m). normalize stands in for rmsnorm.
Q[q, h, k]       := W_Q[h, k, m] · X[q, m]
K[s, h, k]       := W_K[h, k, m] · X[s, m]
V[s, h, k]       := W_V[h, k, m] · X[s, m]
tensor S : (h, q, s)
S[h, q, s.]      := softmax(where s ≤ q)(Q[q, h, k] · K[s, h, k])
AttnOut[q, h, k] := S[h, q, s] · V[s, h, k]
Attn[q, m]       := W_O[m, h, k] · AttnOut[q, h, k]
tensor A : (q, m)
A[q, m.]         := normalize(Attn[q, m] + X[q, m])
F[q, d]          := relu(W_in[d, m] · A[q, m])
Y[q, m]          := W_out[m, d] · F[q, d]
tensor H : (q, m)
H[q, m.]         := normalize(Y[q, m] + A[q, m])

-- Multi-layer transformer as a SCAN over the layer axis l. The layer hidden state H[q,m,l] is
-- the only scan state (base = embeddings); the attention+FFN block becomes per-step
-- INTERMEDIATES, recomputed from H[·,·,l] each step. `finalizeScans` routes any non-iteration
-- stmt that transitively reads a scan state into the recurrence body (in source order). The loop
-- extent L and the key-position extent s are pinned by `axis` decls (since H is now produced, not
-- an input, no tensor shape fixes them); norm axes are marked on the output slot (`s.`, `m.`).
axis l : ℕ = 3
axis s : ℕ = 2
tensor S : (h, q, s)
tensor A : (q, m)
tensor H : (q, m, l)
H[q, m, 0]       := X[q, m]                              -- base case: layer 0 = embeddings
Q[q, h, k]       := W_Q[h, k, m] · H[q, m, l]           -- per-step intermediate (reads state at l)
K[s, h, k]       := W_K[h, k, m] · H[s, m, l]
V[s, h, k]       := W_V[h, k, m] · H[s, m, l]
S[h, q, s.]      := softmax(where s ≤ q)(Q[q, h, k] · K[s, h, k])
AttnOut[q, h, k] := S[h, q, s] · V[s, h, k]
Attn[q, m]       := W_O[m, h, k] · AttnOut[q, h, k]
A[q, m.]         := normalize(Attn[q, m] + H[q, m, l])
F[q, d]          := relu(W_in[d, m] · A[q, m])
Y[q, m]          := W_out[m, d] · F[q, d]
H[q, m., l+1]    := normalize(Y[q, m] + A[q, m])         -- state recurrence: writes layer l+1
```

### 13.3 Abstract syntax

Direct formalization of the BNF layers as Lean inductive types. `UID` from [§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer); `Numeric` (= `MvPolynomial String ℕ`, [§2.1](#21-numeric)) from [§2](#2-the-base-coloredprop); `FreshM` from [§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer). Axis sizes are carried in a computable `SizeExpr` (below), not `Numeric` directly.

```lean
-- Layer 0: computable axis-size arithmetic (the DSL mirror of Numeric)
-- NOTE (computability). `Numeric = MvPolynomial String ℕ` is NONCOMPUTABLE — its `X`/`C`,
-- the semiring, and even `DecidableEq` all resolve through `Classical` (see §2.1). But the
-- elaborator builds a concrete `TLProgram` *value* (`elabTLProgram` returns a value, not an
-- `Expr`), and `compile` runs at elaboration time, and the `tlprog!`/`tl!` macros need `ToExpr`
-- plus size-equality dedup — none of which compiled metacode can do over `MvPolynomial`. So the
-- DSL carries axis sizes in this COMPUTABLE `SizeExpr`, with `SizeExpr.toNumeric` used only when
-- crossing to the proof side. (The integer stride coefficients of `IdxExpr` are plain `ℤ`, which
-- is already computable, so they need no mirror.)
inductive SizeExpr
  | var : String → SizeExpr               -- symbolic generator (was a FreeNumeric, §2.1)
  | lit : Nat → SizeExpr                   -- literal dimension
  | add : SizeExpr → SizeExpr → SizeExpr
  | mul : SizeExpr → SizeExpr → SizeExpr
  deriving DecidableEq, Repr, Inhabited, Lean.ToExpr

noncomputable def SizeExpr.toNumeric : SizeExpr → Numeric   -- proof-side bridge only
  | .var s => MvPolynomial.X s | .lit n => MvPolynomial.C (n : ℕ)
  | .add a b => a.toNumeric + b.toNumeric | .mul a b => a.toNumeric * b.toNumeric

-- Layer 1
inductive AxisKind
  | real   : Option SizeExpr → AxisKind  -- ℝ axis; coordinate DType.reals (§2.3)
  | nat    : Option SizeExpr → AxisKind  -- ℕ axis; coordinate DType.nat   (§2.3)
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited
-- The `Option SizeExpr` is the axis SIZE (Axis.size, §2.2): `some s` concrete, `none` a fresh
-- generator minted in Stage 2 (§7.2). The real/nat tag is the §2.3 DType of coordinates along
-- the axis (fixing the assembled array's `ArrayType.dtype`). The softmax/normalize reduction
-- axis is NO LONGER an AxisKind: it is marked on the output slot (`LHSSlot.freeNorm`, below) and
-- consumed by the evaluator / `splitNonlins` (§13.5) directly — a per-statement property, not an
-- intrinsic axis kind (the same axis can be a softmax axis in one stmt and a contraction in another).

structure AxisSpec where
  name : String
  uid  : UID       -- identity key for the Context coequalizer (§7.4); assigned in Stage 2
  kind : AxisKind

inductive Decl
  | tensor    : String → List AxisSpec → Decl   -- a tensor's axes are NAMES only; dtype/size live on `axis`
  | predicate : String → List AxisSpec → Decl   -- Boolean-valued: R = Bool target semiring (§8)
  | linear    : String → (inAxes outAxes : List AxisSpec) → (bias : Bool) → Decl
  | axis      : AxisSpec → Option Nat → Decl    -- `axis l : ℕ = 3`: an axis's dtype + optional pinned size
```

```lean
-- Layer 2
inductive IdxExpr
  | axis   : AxisSpec → IdxExpr                         -- free or contracted axis
  | const  : ℤ → IdxExpr                               -- constant coordinate (Slice)
  | scale  : ℤ → AxisSpec → IdxExpr                    -- n * a
  | shift  : AxisSpec → ℤ → IdxExpr                    -- a + n  (n < 0 = look-back)
  | affine : ℤ → List (ℤ × AxisSpec) → IdxExpr         -- n + Σ cᵢ·aᵢ (general Reindex)
  -- Note: '(' idx_expr ')' is surface grouping; elabTLIdxExpr recurses into the inner expression
```

```lean
-- Layer 2.5: Predicate arithmetic (extends IdxExpr with non-affine products and absolute value)
inductive PredArith
  | embed : IdxExpr → PredArith                         -- lift any affine expression
  | mul   : PredArith → PredArith → PredArith           -- imul; non-affine product
  | iabs  : PredArith → PredArith                       -- |e|: integer absolute value
  -- |e| is an integer VALUE, not a boolean; it appears as an operand in comparisons
  -- such as `|i − j| ≤ n` = rel(le, iabs(embed(affine i−j)), embed(const n)).
  -- Note: '(' pred_term ')' is surface grouping
```

```lean
-- Layer 3
inductive RelOp | lt | le | eq | ne | ge | gt

inductive BoolExpr
  | rel  : RelOp → PredArith → PredArith → BoolExpr    -- both operands are PredArith values
  | and  : BoolExpr → BoolExpr → BoolExpr
  | or   : BoolExpr → BoolExpr → BoolExpr
  | not  : BoolExpr → BoolExpr
  | ieq  : PredArith → PredArith → BoolExpr            -- modular equality (wrapping int comparison)
  -- Note: '(' bool_expr ')' is surface grouping; |e| lives in PredArith, not here
```

```lean
-- Layer 4
inductive Nonlin
  | identity  : Nonlin
  | relu      : Nonlin
  | softmax   : Option BoolExpr → Nonlin
  | normalize : Option BoolExpr → Nonlin

inductive Factor
  | read    : String → List IdxExpr → Factor            -- name[e₁,...,eₙ]
  | iverson : BoolExpr → Factor                         -- [P]

structure ProdTerm where factors : List Factor
structure SumExpr  where terms   : List ProdTerm
structure RHSExpr  where body : SumExpr; nonlin : Nonlin
```

```lean
-- Layer 5
inductive LHSSlot
  | free     : AxisSpec → LHSSlot                       -- ordinary free axis
  | freeNorm : AxisSpec → LHSSlot                       -- free axis marked `m.` = softmax/normalize axis
  | iterAt   : AxisSpec → ℤ → LHSSlot                  -- l = n  (base case)
  | iterNext : AxisSpec → LHSSlot                       -- l + 1  (recurrence step)
  | affine   : IdxExpr → LHSSlot                        -- affine output slot (Scatter)

structure ScatterOpts where
  fill   : Int := 0                 -- `Int`, not `Float`: Lean's `Float` has no `DecidableEq`,
  reduce : Option String := none    -- which the AST's `deriving DecidableEq` requires
                                    -- none = injective required; 'sum' = accumulate

inductive Stmt
  | assign        : String → List LHSSlot → RHSExpr → Stmt
  | scatter       : String → List LHSSlot → RHSExpr → ScatterOpts → Stmt
  -- The `recurMorphism` escape hatch carries the computable
  -- `ThreadedComposed` (§13.5 back-end). Programmatic-only (no `tlprog!` surface syntax):
  --   | recurMorphism : String → AxisSpec → ThreadedComposed → Stmt
  -- escape hatch: String = tensor name, AxisSpec = iteration axis,
  -- ThreadedComposed = a pre-built step morphism (§13.5). A value, not a metaprogramming
  -- Expr: it is the same morphism type `compile` produces.

structure TLProgram where
  decls : List Decl
  stmts : List Stmt
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited
-- Deriving strategy for the tlprog!/tl! macros (requires Lean.ToExpr on all DSL types).
-- Every front-end type above — SizeExpr, AxisKind, AxisSpec, Decl, IdxExpr,
--   PredArith, RelOp, BoolExpr, Nonlin, Factor, ProdTerm, SumExpr, RHSExpr, LHSSlot,
--   ScatterOpts, Stmt, TLProgram — derives `DecidableEq, Repr, Lean.ToExpr` (with `Inhabited`
--   where a default is needed). Because sizes are `SizeExpr` (not `Numeric`), `ToExpr` derives
--   automatically — no `MvPolynomial.toExpr` is needed for the front-end.
-- The §13.5 ThreadedComposed/BrBase/StMat are realized as COMPUTABLE
--   first-order presentations (BrBaseP/StMatP/AxisP/WeaveSlotP/Wire, sizes in SizeExpr,
--   coeffs in ℤ), so `ToExpr` derives directly and the anticipated noncomputable-`Numeric`
--   ToExpr question (coeff enumeration vs ring normal form) is SIDESTEPPED, not answered —
--   it would only resurface in the bridge that maps a presentation to a real BrMorph.
```

### 13.4 Concrete syntax and elaboration

Following the IMP language pattern of the Lean 4 metaprogramming book (ch. 8): one `declare_syntax_cat` per BNF layer, `syntax` rules transcribing each production, and a `partial def elabXxx` function per category. Where the book builds `Expr` terms, the implementation instead has each elaborator **return the AST value directly** — `elabXxx : Syntax → MetaM <value>` (e.g. `elabTLDecl : Syntax → MetaM Decl`) — interpreting syntax by structural recursion rather than via `mkAppM ``Constructor`. `MetaM` (not `TermElabM`) suffices, since no term-level elaboration is needed; `partial` is required because Lean's termination checker cannot verify that syntax consumption decreases; each function matches with `` `(tl_cat| …) `` quotations and falls through to `throwUnsupportedSyntax` on mismatch. **Surface conventions:** identifiers are read with `x.getId.eraseMacroScopes.getString!` (a bare `getString!` panics on quotation-introduced macro scopes); the scan-step LHS token `+1` is a single atom, so it is written spaced as `l +1`.

**Syntax categories:**

```lean
declare_syntax_cat tl_size
declare_syntax_cat tl_axis_kind
declare_syntax_cat tl_axis_spec
declare_syntax_cat tl_named_shape
declare_syntax_cat tl_axis_decl_item
declare_syntax_cat tl_linear_item
declare_syntax_cat tl_shape
declare_syntax_cat tl_decl
declare_syntax_cat tl_idx_expr
declare_syntax_cat tl_pred_term
declare_syntax_cat tl_rel_op
declare_syntax_cat tl_bool_expr
declare_syntax_cat tl_nonlin
declare_syntax_cat tl_factor
declare_syntax_cat tl_prod_term
declare_syntax_cat tl_sum_expr
declare_syntax_cat tl_rhs
declare_syntax_cat tl_lhs_slot
declare_syntax_cat tl_stmt
declare_syntax_cat tl_program
```

**Representative syntax rules (one per BNF layer):**

```lean
-- Layer 1: axis kinds (bracket holds a tl_size term elaborating to SizeExpr, §13.3)
syntax num                   : tl_size
syntax ident                 : tl_size
syntax tl_size "*" tl_size   : tl_size
syntax tl_size "+" tl_size   : tl_size
syntax "(" tl_size ")"       : tl_size

syntax "ℝ"                   : tl_axis_kind
syntax "ℝ[" tl_size "]"      : tl_axis_kind
syntax "ℕ"                   : tl_axis_kind
syntax "ℕ[" tl_size "]"      : tl_axis_kind

syntax ident : tl_axis_spec               -- an axis name (bare ident)
syntax ident "(" tl_axis_spec,* ")" : tl_named_shape   -- `T(a, b, c)`
syntax ident ":" tl_axis_kind         : tl_axis_decl_item  -- dtype only
syntax ident ":" tl_axis_kind "=" num : tl_axis_decl_item  -- dtype + pinned size
syntax ident "(" tl_axis_spec,* ")" "→" "(" tl_axis_spec,* ")"        : tl_linear_item  -- `W(d) → (dff)`
syntax ident "(" tl_axis_spec,* ")" "→" "(" tl_axis_spec,* ")" "bias" : tl_linear_item  -- with bias
syntax "(" tl_axis_spec,* ")" : tl_shape   -- kept; no longer used by any tl_decl rule

syntax "tensor"    tl_named_shape,+                        : tl_decl   -- `tensor A(q,m), B(x,y)`
syntax "predicate" tl_named_shape,+                        : tl_decl   -- `predicate P(i,j)`
syntax "linear"    tl_linear_item,+                        : tl_decl   -- `linear W(d) → (dff), V(dff) → (d)`
syntax "axis"      tl_axis_decl_item,+                     : tl_decl   -- `axis l : ℕ = 3, s : ℕ = 2`

-- Layer 2: index expressions — GENERALIZED to general integer-affine sums (the AST's
-- IdxExpr.affine carries a full `List (ℤ × AxisSpec)`, so the grammar must reach it).
-- A left-associative `+`/`-` sum of terms; `*` (prec 70) binds tighter than `+`/`-` (prec 65).
syntax:70 num "*" ident                       : tl_idx_expr  -- literal-coefficient term
syntax:max ident                              : tl_idx_expr
syntax:max num                                : tl_idx_expr
syntax:65 tl_idx_expr:65 " + " tl_idx_expr:66 : tl_idx_expr
syntax:65 tl_idx_expr:65 " - " tl_idx_expr:66 : tl_idx_expr  -- look-back (n < 0)
syntax:max "(" tl_idx_expr ")"                : tl_idx_expr
-- elabTLIdxExpr collapses a single bare term to IdxExpr.axis/const/scale and a `±n` to
-- IdxExpr.shift; anything longer becomes IdxExpr.affine. Symbolic-coefficient strides
-- (`s*j`, an ident coefficient) are NOT representable in integer-coefficient IdxExpr.

-- Layer 2.5: predicate arithmetic
syntax tl_idx_expr                               : tl_pred_term
syntax "imul(" tl_pred_term "," tl_pred_term ")" : tl_pred_term
syntax "|" tl_pred_term "|"                      : tl_pred_term  -- iabs; value, not bool
syntax "(" tl_pred_term ")"                      : tl_pred_term

-- Layer 3: predicates
syntax tl_pred_term "<"  tl_pred_term             : tl_bool_expr
syntax tl_pred_term "≤"  tl_pred_term             : tl_bool_expr
syntax tl_pred_term "="  tl_pred_term             : tl_bool_expr
syntax tl_pred_term "≠"  tl_pred_term             : tl_bool_expr
syntax tl_pred_term ">"  tl_pred_term             : tl_bool_expr
syntax tl_pred_term "≥"  tl_pred_term             : tl_bool_expr
syntax tl_bool_expr "∧" tl_bool_expr              : tl_bool_expr
syntax tl_bool_expr "∨" tl_bool_expr              : tl_bool_expr
syntax "¬" tl_bool_expr                           : tl_bool_expr
syntax "ieq(" tl_pred_term "," tl_pred_term ")"   : tl_bool_expr
syntax "(" tl_bool_expr ")"                        : tl_bool_expr

-- Layer 4: RHS
syntax ident "[" tl_idx_expr,* "]"     : tl_factor
syntax "[" tl_bool_expr "]"            : tl_factor

-- N-ARY (left-recursive): `·` binds tighter than `+`, both left-associative, so that
-- `A·B·C` flattens to ProdTerm with 3 factors and `A+B+C` to SumExpr with 3 terms.
syntax:70 tl_prod_term:70 " · " tl_factor:71   : tl_prod_term
syntax:71 tl_factor:71                          : tl_prod_term

syntax:65 tl_sum_expr:65 " + " tl_prod_term:66 : tl_sum_expr
syntax:66 tl_prod_term:66                        : tl_sum_expr

-- `atomic("(" "where")` LEFT-FACTORS the masked variants against the bare token. Without it,
-- the shared `softmax (` prefix made the parser commit to the where-variant on seeing `(`, so an
-- UNMASKED `softmax(A[i]+B[i])` failed ("expected 'where'"). `atomic` makes the `( where`
-- lookahead all-or-nothing: when there is no `where` it rewinds the `(`, the bare `softmax` rule
-- wins, and the `(sum)` is consumed at the tl_rhs level below.
syntax "relu"                                           : tl_nonlin
syntax "softmax"                                        : tl_nonlin
syntax "softmax"   atomic("(" "where") tl_bool_expr ")" : tl_nonlin
syntax "normalize"                                      : tl_nonlin
syntax "normalize" atomic("(" "where") tl_bool_expr ")" : tl_nonlin

syntax tl_nonlin "(" tl_sum_expr ")"   : tl_rhs
syntax tl_sum_expr                      : tl_rhs

-- Layer 5: statements
syntax ident "[" tl_lhs_slot,* "]" ":=" tl_rhs : tl_stmt
syntax ident                 : tl_lhs_slot
syntax ident "."             : tl_lhs_slot   -- norm marker: the softmax/normalize reduction axis
syntax num                   : tl_lhs_slot
syntax ident "+1"            : tl_lhs_slot
syntax num "*" ident         : tl_lhs_slot
syntax ident "+" num         : tl_lhs_slot
syntax num "*" ident "+" num : tl_lhs_slot

-- Layer 6
syntax (tl_decl <|> tl_stmt)* : tl_program  -- accepts interleaved decls/stmts; compiler validates decl* stmt+ ordering

-- Note: Stmt.recurMorphism has no syntax rule yet (surface form TBD).
-- The escape hatch is accessible programmatically but not via tl!{} surface syntax.
-- A provisional form: syntax ident "." "recur" "(" ident "," term ")" : tl_stmt
-- would allow: `Name.recur(l, morphism)` where morphism : ThreadedComposed.
```

**Elaboration function signatures (one per syntax category):**

```lean
-- Every elaborator returns its AST value directly in MetaM (no Expr-building):
partial def elabTLSize      : Syntax → MetaM SizeExpr
partial def elabTLAxisKind  : Syntax → MetaM AxisKind
partial def elabTLAxisSpec  : Syntax → MetaM AxisSpec        -- uid := 0; assigned in Stage 2
partial def elabTLShape     : Syntax → MetaM (List AxisSpec)
partial def elabTLDecl      : Syntax → MetaM Decl
partial def elabTLIdxExpr   : Syntax → MetaM IdxExpr
partial def elabTLPredTerm  : Syntax → MetaM PredArith
partial def elabTLBoolExpr  : Syntax → MetaM BoolExpr
partial def elabTLNonlin    : Syntax → MetaM Nonlin
partial def elabTLFactor    : Syntax → MetaM Factor
partial def elabTLProdTerm  : Syntax → MetaM ProdTerm        -- flattens the n-ary `·` chain
partial def elabTLSumExpr   : Syntax → MetaM SumExpr         -- flattens the n-ary `+` chain
partial def elabTLRHS       : Syntax → MetaM RHSExpr
partial def elabTLLHSSlot   : Syntax → MetaM LHSSlot
partial def elabTLStmt      : Syntax → MetaM Stmt            -- always builds Stmt.assign
partial def elabTLProgram   : Syntax → MetaM TLProgram      -- routes children by syntax-kind prefix
-- The elaborators interpret Syntax nodes by structural recursion, returning concrete AST values
-- rather than building Lean Expr terms — so no kernel reduction is needed to extract a TLProgram
-- at Stage 2. RelOp is read inline by elabTLBoolExpr (no separate elabTLRelOp). Every
-- `name[…] := rhs` parses to Stmt.assign; the assign/scatter split is a back-end concern
-- (lowerArith reclassifies affine-LHS writes as scatter).
```

The elaborator is pure syntax-walking with no side effects: UID minting and axis unification happen in Stage 2, not Stage 1.

### 13.5 Semantic compilation

The 8-phase `TLProgram.compile` pipeline, the typed intermediates (`LabeledProgram` … `ScheduledProgram`), `ScanStmt`, `Wire`, `ThreadedComposed`, and the `tl!{ … } : ThreadedComposed` compile macro are **implemented in `leanncd/LeanNCD/DSL/Pipeline/` + `Target.lean`/`Compile.lean`, fully executable and `sorry`-free**: the first five [§13.2](#132-bnf-grammar) examples compile end-to-end (the two predicate examples are parse-tested only; predicate *evaluation* — the [§8](#8-algebras-and-construct) Bool-semiring semantics — is deferred). Building on the executable `FreshM`/`Context` seam of [§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer), the pipeline consumes `TLProgram` values and produces a **computable, first-order *presentation* of the Br morphism** — `ThreadedComposed`/`BrBaseP`/`StMatP`/`AxisP`/`WeaveSlotP` are `List`-based, `SizeExpr`/`Int`-valued, `deriving Lean.ToExpr` mirrors of the noncomputable math-tower `Br`/`St` types (the same `Numeric`→`SizeExpr` move made for the AST). The bridge (`leanncd/LeanNCD/Bridge/`, signatures + `sorry`) maps the presentation into the math tower — leaf realizations are `sorry`-free; the routed-DAG composite (`realize : ThreadedComposed → Σ dom cod, BrMorph dom cod`) rests on `Br.tensorHom`/`swap` (a `sorry`). The CSV-side `SBrInstance`/`realizeSBr` and the [§9](#9-acsets-and-the-executable-layer) agreement Props are stated with `sorry` — see the [§9.2](#92-the-seam-made-tangible) note. Reindexing coefficients are signed (`Coeff = MvPolynomial String ℤ`, [§2.1](#21-numeric)/[§2.2](#22-st--stride-matrices)), so the presentation's `Int` coeffs realize faithfully (look-back offsets included). The `Stmt.recurMorphism`/`ScanStmt.scanPre` escape hatch is implemented (consistent-collapse: the pre-built `ThreadedComposed` is validated, then routed as a single `op="scan_pre"` step), as is the `ScanAffine` fast path (`route` emits `op="scan_affine"` when the recurrence is nonlinearity-free, so routed scan steps carry `op ∈ {scan, scan_affine, scan_pre}`). **Still deferred:** predicate *evaluation* (the [§8](#8-algebras-and-construct) Bool-semiring Algebra — and with it threading a `DType`/op-semiring tag onto `BrBaseP`) and the `Br` coherence `sorry`s. The signatures below are shown in their implemented (presentation) form; the per-phase **implementation notes** after the phase table record the simplifications.

```lean
/-- Named alias for the declaration environment built by resolveDecls. -/
abbrev DeclEnv := Std.HashMap String Decl   -- requires BEq String, Hashable String (both in core)

/-- A statement after finalizeScans has grouped iterAt/iterNext pairs into Scan nodes.
    Replaces bare Stmt.assign/scatter with explicit Scan steps. -/
inductive ScanStmt
  | plain  : Stmt → ScanStmt                                  -- non-recursive statement
  | scan   : String → AxisSpec → List Stmt → List Stmt → Bool → ScanStmt
             -- (tensor name, iteration axis, base stmts, recurrence stmts, isAffine flag)
  -- The `scanPre` case carries the computable `ThreadedComposed`
  -- supplied via Stmt.recurMorphism (programmatic-only):
  | scanPre : String → AxisSpec → ThreadedComposed → ScanStmt
              -- (Stmt.recurMorphism case: step morphism provided directly)

/-- Typed intermediate representations for the 8-phase pipeline.
    Each type carries the invariant guaranteed by its producing phase. -/
structure LabeledProgram where
  decls : List Decl
  stmts : List Stmt    -- every AxisSpec.uid is a fresh non-zero UID (assignUIDs invariant)

structure ResolvedProgram where
  decls    : List Decl
  stmts    : List Stmt
  env      : DeclEnv                    -- resolved declaration map
  extNames : Finset String              -- externally declared (input) tensor names
  extraStmts : Array Stmt               -- bias-add stmts emitted for linear...bias:=true

structure CanonicalProgram where
  decls    : List Decl
  stmts    : List Stmt
  env      : DeclEnv
  extNames : Finset String
  ctx      : Context AxisSpec           -- canonical axis equivalence classes (unifyAxes result)

structure LoweredProgram where
  decls    : List Decl
  stmts    : List Stmt                  -- affine-LHS assigns reclassified to Stmt.scatter
  env      : DeclEnv
  extNames : Finset String
  ctx      : Context AxisSpec
  auxStmts : Array Stmt                 -- `#[]` — affine READS are NOT split into separate
                                        -- Slice/Reindex steps but folded into the consuming step's
                                        -- `reindexings` at `route` (see the implementation notes below)

structure ScanProgram where
  decls    : List Decl
  stmts    : List ScanStmt             -- iterAt/iterNext grouped into ScanStmt.scan nodes
  env      : DeclEnv
  extNames : Finset String
  ctx      : Context AxisSpec

structure LinearProgram where
  decls    : List Decl
  stmts    : List ScanStmt             -- no nonlinearity in RHSExpr.nonlin (split into BrBase ops)
  env      : DeclEnv
  extNames : Finset String
  ctx      : Context AxisSpec

structure ScheduledProgram where
  decls    : List Decl
  stmts    : List ScanStmt             -- live stmts in reverse-topological order (BFS from output)
  env      : DeclEnv
  extNames : Finset String
  ctx      : Context AxisSpec

-- COMPUTABLE PRESENTATION TYPES (Target.lean). The math-tower Br/St types (§2.2/§2.3) are
-- noncomputable (over `Numeric`) and use function-valued fields (`Fin _ → …`), so they cannot
-- `deriving ToExpr`. The pipeline therefore lowers them to first-order, `List`-based, `SizeExpr`/`Int`-
-- valued mirrors — the term-world PRESENTATION the tl!{} macro embeds. Each `P`-type drops the
-- dependent length/shape indices of its math-tower twin (an invariant the bridge re-establishes).
structure StMatP where                          -- presentation of StMat (§2.2): integer-affine map
  domLen : Nat; codLen : Nat
  coeffs : List (List Int)                       -- codLen × domLen; `Int` mirrors StMat's signed
  bias   : List Int                              -- `Coeff = ℤ`-poly (§2.2) — realizeStMat: Int→Coeff
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited
structure AxisP where name : Option String; size : SizeExpr   -- presentation of Axis: size is SizeExpr
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited
abbrev StObjP := List AxisP
inductive WeaveSlotP | fixed : AxisP → WeaveSlotP | tiled : WeaveSlotP
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited
abbrev WeaveShapeP := List WeaveSlotP
structure BrBaseP where                          -- presentation of BrBase (§2.3): Fin-functions → Lists
  op : String; degree : StObjP
  inputWeaves outputWeaves : List WeaveShapeP
  reindexings : List StMatP
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

/-- A wire in the DAG: identifies a specific output slot of a specific step (or the external inputs). -/
structure Wire where
  step : Nat       -- index into steps; step = nExternal is the "external input" sentinel
  slot : Nat       -- output slot index within that step
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

/-- A routed DAG of (presented) Br base morphisms: the term-world presentation of one Br morphism. -/
structure ThreadedComposed where
  steps     : List BrBaseP            -- the lowered operations (presentation of §2.3 BrBase)
  routing   : List (List Wire)        -- routing[i] = the input wires of steps[i]
                                      -- (was `Fin steps.length → ℕ → Wire`; a List for ToExpr)
  nExternal : Nat                     -- number of external inputs (declared tensors / weights)
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited
-- Because every field is first-order with `SizeExpr` sizes, `Lean.ToExpr` derives directly — no
-- `MvPolynomial.toExpr` is needed, and the tl!{} macro embeds the value as a compile-time constant.
-- A ThreadedComposed PRESENTS one `BrMorph` (§2.3): composing and tensoring `steps` along
-- `routing` collapses to a single morphism of `Br` (realized by the bridge in `leanncd/LeanNCD/Bridge/`). It is
-- the term-world twin of the acset `SBrInstance` (§9.1) — the acset extraction (Python
-- `from_tensor_program`) turns one into the other, and `write_sbr`/`read_sbr` round-trip that
-- SBrInstance to and from CSV. So the DSL path (this section) and the CSV path (§9) land on the
-- very same `∫Dat`-morphism.

/-- Lower a TLProgram to a ThreadedComposed morphism.
    Runs in FreshM (= EStateM CompileError ℕ, Lean core Init.Control.EStateM):
    mints fresh UIDs for synthetic intermediates and throws CompileError on
    validation failures. Kleisli composition (>=> from Init.Core) sequences
    the typed phases; each phase narrows the type invariant. -/
def TLProgram.compile : TLProgram → FreshM ThreadedComposed :=
  assignUIDs >=> resolveDecls >=> unifyAxes >=> lowerArith
             >=> finalizeScans >=> splitNonlins >=> schedule >=> route
```

The pipeline is a typed chain; each phase boundary carries a more constrained intermediate type so that Python-comment invariants become enforced by construction:

```text
TLProgram
  →[assignUIDs]    LabeledProgram        -- every AxisSpec has a fresh UID (name-keyed by axis name)
  →[resolveDecls]  ResolvedProgram       -- DeclEnv built; external names = read-not-produced
  →[unifyAxes]     CanonicalProgram      -- axis UIDs are canonical (pure)
  →[lowerArith]    LoweredProgram        -- affine-LHS assigns reclassified to Stmt.scatter
  →[finalizeScans] ScanProgram           -- iterAt/iterNext grouped into ScanStmt.scan nodes
  →[splitNonlins]  LinearProgram         -- nonlinearity isolated into its own step
  →[schedule]      ScheduledProgram      -- live stmts (DCE); root = last stmt's output
  →[route]         ThreadedComposed      -- one BrBaseP per stmt; routing wires + nExternal
```

| Phase | What it does | Key Lean idiom |
| --- | --- | --- |
| **assignUIDs** | Traverses `decls` and `stmts`; mints a fresh UID for each `AxisSpec` via `freshUData` ([§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer)). | `FreshM`; `List.mapM freshUData` traverses declarations |
| **resolveDecls** | Builds `DeclEnv : Std.HashMap String Decl` (`Std.Data.HashMap`; `String` has `BEq` and `Hashable`). Validates: `linear` weight appears in exactly one product factor; every declared name has a consistent shape across stmts; throws `CompileError` on violation. `linear ... bias:=true` appends a bias-add stmt to the returned `ResolvedProgram`. Marks each name as external (declared) or internal (produced by a stmt) — drives routing. Predicate-typed names are tagged here; the tag tells the Algebra ([§8](#8-algebras-and-construct)) to evaluate that output in the Boolean value semiring `R = Bool` rather than `R = ℝ`. | `FreshM`; validation errors via `throw`; bias stmts accumulated in `ResolvedProgram.extraStmts : Array Stmt` |
| **unifyAxes** | The [§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer) UID coequalizer, computed in batch. Collects the `(uid_a, uid_b)` identifications from axis occurrences sharing a name within program scope (Domingos' name-binding, [§13.2](#132-bnf-grammar)), feeds them to `Context.merge`, and applies the result with `Context.apply`. The canonical representative is the **largest UID** — the universal cocone vertex of [§7.3](#73-composition-as-pushout) — so a DSL-built morphism and a CSV-built one agree on axis identity on the nose. The whole program is known statically, so this runs once rather than incrementally (Python's `Context.append_iter`), but it is the *same* coequalizer with the *same* representative rule. | Pure (`ResolvedProgram → CanonicalProgram`); lifted to `FreshM` by `pure`; `Context` / `EqClass` ([§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer)) |
| **lowerArith** | `IdxExpr.const` reads → fresh `Slice` intermediate; `IdxExpr.affine` reads → fresh `Reindex` intermediate; affine `LHSSlot`s → `Scatter` (injectivity checked; `reduce = some "sum"` required for non-injective maps). Each is a `BrBase` ([§2.3](#23-br--free-category-over-broadcasted-base-morphisms)) whose `reindexings` field carries the affine map as an `St` stride matrix `StMat` — the locus where `St` lives inside `Br`. Non-zero fill prepends a fill-initialization stmt. Auxiliary stmts are stored in `LoweredProgram.auxStmts : Array Stmt`, not a global. | `FreshM`; `freshUData` mints UIDs for synthetic intermediates; auxiliary stmts in output type, not a writer monad |
| **finalizeScans** | Groups stmts by name + iteration axis UID; pairs `iterAt`/`iterNext` slots into `Scan` nodes; stmts sharing the same iteration-axis UID across names form a coupled `Scan` (`n_states > 1`). Each `Scan` is the `cata(step)` of the `TemporalGraded` mixin ([§6.1](#61-temporalgraded--scan)) over the iteration axis as temporal object `L`; the prefix-restriction and batching laws it obeys are Props 8.7–8.8. `Stmt.recurMorphism` supplies the step morphism directly, bypassing equation lowering for that scan state. Validates: every `recur_step` has a matching `base_case`; `l+1` absent from RHS for the iteration axis. | `FreshM`; pure grouping; `throw` on missing base case |
| **splitNonlins** | Lifts `relu`/`softmax`/`normalize` out of `RHSExpr.nonlin` into a separate composed step. These are genuinely nonlinear, so they are not reindexings (`StMat` is affine); each becomes a `BrBase` op ([§2.3](#23-br--free-category-over-broadcasted-base-morphisms)) whose numeric semantics are supplied by the Algebra functor `F : C → V` into the target actegory ([§8](#8-algebras-and-construct)). For `softmax`/`normalize` the reduction dimension is the output slot marked `m.` (`LHSSlot.freeNorm`, [§13.3](#133-abstract-syntax)); masked variants emit an alignment-permutation step computed from the `where` mask. | `FreshM`; `freshUData` mints UIDs for nonlin step intermediates |
| **schedule** | Backward reachability BFS from the output name simultaneously determines liveness (DCE) and produces a valid reverse-topological order. Two passes in Python; one here because the BFS visit order is already a reverse topo order. | Pure (`String → List ScanStmt → List ScanStmt`); lifted to `FreshM` by `pure` |
| **route** | Detects contracted axes (present in a `ProdTerm` but absent from the LHS) and builds one `BrBase` ([§2.3](#23-br--free-category-over-broadcasted-base-morphisms)) per stmt, carrying the `tensor`/`predicate` tag from `DeclEnv`. The contraction *arithmetic* is not fixed here but at evaluation, by the Algebra's value semiring `R` ([§8](#8-algebras-and-construct)): `R = ℝ` (×, then Σ) for `tensor` outputs, `R = Bool` (∧, then ∃) for `predicate` outputs — the ∃/∧-vs-Σ split is exactly that choice of `R`. Assigns index slots; builds `ThreadedComposed.routing` and `n_external`. Automatic associative-scan detection (nonlinearity-free recurrence, flagged in `finalizeScans`) tags the routed step `op="scan_affine"` — the `ScanAffine` case where the step algebra factors through a monoid, i.e. Prop 8.7's `O(log N)` parallel prefix; a `recurMorphism` step is tagged `op="scan_pre"`. | Pure (`List ScanStmt → DeclEnv → Context → ThreadedComposed`); lifted to `FreshM` by `pure` |

**Implementation notes.** The phase-table rows above describe the full design; the implemented pipeline simplifies several rows (all `sorry`-free):

- **assignUIDs** binds by axis *name* (the parser emits `uid := 0` for every axis), reusing the [§13.4](#134-concrete-syntax-and-elaboration) `traverseUID`; a `freshNonZero` guard skips the sentinel `0`.
- **resolveDecls** is purely constructive (it never throws): an undeclared read name is an *external input* (the [§13.2](#132-bnf-grammar) examples read `W`/`X`/`Q`/`K` with no `tensor` decl), so `extNames` = read-not-produced. `linear`-weight arity, shape consistency, and bias materialization are deferred (no example declares a `linear` weight); `extraStmts := #[]`. The `tensor`/`predicate` value-semiring tagging is an [§8](#8-algebras-and-construct) *semantic* concern deferred to the bridge, not carried in the presentation.
- **lowerArith** reclassifies affine-LHS assigns to `Stmt.scatter` with a conservative injectivity guard (`overlappingScatter` on a dimension-collapsing constant coordinate); affine *reads* are left in place and folded into the consuming step's `reindexings` at `route` (which is exactly where `St` lives inside `BrBase`, [§2.3](#23-br--free-category-over-broadcasted-base-morphisms)) rather than emitted as separate Slice/Reindex steps; `auxStmts := #[]`.
- **finalizeScans** groups by iteration-axis UID into (coupled) `Scan` nodes; a pre-pass makes each base case adopt a same-named recurrence's iteration axis (the parser emits scan base cases with a placeholder iteration-axis name); `missingBaseCase`/`causalityViolation` guards fire; a `recurMorphism` stmt becomes `ScanStmt.scanPre`, and the `isAffine` flag on `ScanStmt.scan` is set here (before `splitNonlins`) when the recurrence is nonlinearity-free.
- **schedule** does backward-reachability DCE; the output root is the last stmt's written name(s) (single-result-at-tail — a genuine multi-output-not-at-tail program would need an explicit outputs field).
- **route** builds one `BrBaseP` per stmt with contracted axes (read axes absent from the LHS) as `tiled` weave slots; each read's affine `IdxExpr` becomes an integer-coefficient `StMatP` via an `idxToRow` translation; inputs are wired to their producer step or to the external sentinel (`step = nExternal`). The `ScanAffine` fast path is implemented as an `op` tag — `op="scan_affine"` when the recurrence is nonlinearity-free, `op="scan_pre"` for a `recurMorphism` step, else `op="scan"` (so routed scan steps carry `op ∈ {scan, scan_affine, scan_pre}`); the value-semiring contraction arithmetic is deferred to the bridge.

The result is a `ThreadedComposed` (a presentation of a `BrMorph`, [§2.3](#23-br--free-category-over-broadcasted-base-morphisms)) — a finite presentation of an `∫Dat`-morphism, the very thing an `SBrInstance` ([§9.1](#91-sbrinstance-as-a-finite-presentation-of-an-dat-morphism)) presents in tabular form. The DSL path of this section and the CSV path of [§9](#9-acsets-and-the-executable-layer) therefore produce the same categorical object: the acset extraction relates the `ThreadedComposed` and `SBrInstance` presentations, and `write_sbr`/`read_sbr` serialize the latter to and from CSV — none of these steps changes the morphism.

**Python correspondence:**

| Lean DSL | Python DSL | Notes |
| --- | --- | --- |
| `tensor A(q, m), B(x, y)` | `tl.Name.tensor(*axes)` | grouped; no colon |
| `predicate P(i, j)` | `tl.Name.predicate(*axes)` | Bool-typed; same form |
| `linear Name : in → out [bias]` | `tl.Name.linear(out_axes=…, in_axes=…, bias=…)` | weight declaration |
| `Name[i,j] := rhs` | `tl.Name[i,j] = rhs` | normal assignment |
| `Name[0, j] := rhs` | `tl.Name[j, 0] = rhs` | scan base case |
| `Name[l+1, j] := rhs` | `tl.Name[j, l+1] = rhs` | scan recurrence step |
| `Name[2*i] := rhs` | `tl.Name[2*i] = rhs` | affine Scatter write |
| `A[i,k] · B[k,j]` | `tl.A[i,k] * tl.B[k,j]` | Einstein product; k contracted |
| `A[i] + B[i]` | `tl.A[i] + tl.B[i]` | elementwise sum |
| `[i < j]` | `i < j` (Iverson via monkey-patch) | Iverson bracket |
| `relu(…)` | `relu(…)` | ReLU nonlinearity |
| `softmax(where P)(…)` | `softmax(…, where=P)` | masked softmax |
| `normalize(where P)(…)` | `normalize(…, where=P)` | masked normalize |
| `X[n]` | `tl.X[n]` (int index) | Slice — constant read |
| `X[i + n]` | `tl.X[i + n]` (affine expr) | Reindex — affine read |
| `Y[n*i] := …` | `tl.Y[n*i] = …` (affine LHS) | Scatter — affine write |
| `Stmt.recurMorphism name axis morphism` | `tl.name.recur(l, morphism)` | escape hatch; programmatic-only (no surface syntax); routes as `op="scan_pre"` |
| `elabTLProgram` (Stage 1) | — | `Syntax → MetaM TLProgram` (value, not Expr); no Python analogue |
| `TLProgram.compile` (Stage 2) | `tl.to_morphism()` | `TLProgram → FreshM ThreadedComposed`; run at elaboration time via `.run 0` |

### 13.6 Evaluation

The DSL has a reference **`Float` evaluator**,
`TLProgram.eval : TLProgram → Std.HashMap String DenseTensor → Except EvalError (HashMap String DenseTensor)`
(in `leanncd/LeanNCD/Eval/`), which runs a program on concrete input tensors and produces concrete
output tensors (`DenseTensor` = a row-major `{ shape, data : Array Float }`). It is fully executable
and `sorry`-free.

The evaluator interprets the **pre-route `ScheduledProgram`** rather than the routed
`ThreadedComposed` of [§13.5](#135-semantic-compilation): the routed presentation keeps only a scan's
representative recurrence step (it is lossy for scans), whereas the `ScheduledProgram` retains the full
`base`/`recur` stmt lists, so coupled and base cases evaluate. The output **dtype** — tensor vs
predicate, i.e. the `(Σ, ×)` vs `(∃, ∧)` contraction — is read from the decls (a `predicate` output
contracts in `(max, min)` on 0/1 Floats), which sidesteps the deferred `BrBaseP` dtype gap. All eleven
example programs (the seven §13.2/predicate examples plus look-back, outer product, contraction+relu,
and normalize) evaluate with hand-checked numeric assertions in `test/Eval/EvalExamplesTest.lean`.
Symbolic-size evaluation and the `scanPre`/`recurMorphism` escape hatches are out of scope (they raise
an `EvalError`).

## 14. Appendix: out of scope

Two families of structure are deliberately **not encoded**, because they carry no propositional or computational content the framework reasons about. The first is **`DynamicName` and its LaTeX rendering** — the human-readable, mathematically-typeset names attached to axes and arrays. The second is the **`Block` display metadata** — the layout and presentation bookkeeping the visualizer consumes. Both are *semantically transparent*: erasing them changes no morphism, no shape, no composite, and no proof. They ride on the executable side of the seam as identity/display decoration (the `WithUID` decoration of [§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer) carries the optional `DynamicName`), and they are left exactly where the current document already leaves `Block` — outside the encoding, mentioned but never formalized.

This matches the document's overall stance, restated once here: it writes **no proved Lean**. Every `class`, `structure`, `def`, and `theorem` above is a *signature* with named `Prop`-field laws and named proof obligations; the elisions (`…`) in bodies are intentional, marking exactly the obligations a future Lean development would discharge. The deliverable is **formalizability, not formalization** — the shape into which the propositions of [graded_prop.md §8](graded_prop.md#8-propositions-the-synthesis-organizes) transcribe directly, not the discharged proofs themselves.

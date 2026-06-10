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
7. [Grothendieck split and composition as pushout](#7-grothendieck-split-and-composition-as-pushout)
8. [Acsets and Python interop](#8-acsets-and-python-interop)
9. [The propositions as generic theorems](#9-the-propositions-as-generic-theorems)
10. [Instantiation and future extensions](#10-instantiation-and-future-extensions)
11. [Lean formalization notes](#11-lean-formalization-notes)
12. [Appendix: out of scope](#12-appendix-out-of-scope)

## 1. Orientation: one structure, one seam

The current state of the design predates [graded_prop.md](graded_prop.md): it formalised `St` and `Br` as two independent PROP instances and divided everything into a "Layer 1 — Mathematical Encoding" and a "Layer 2 — Representation" (UIDs, `Context`, names). That split was an artifact of not yet having the vocabulary `graded_prop.md` now supplies. The reframing dissolves it. The scattered constructions are recognised as **one structure** — a `D`-graded colored PROP — and the two things the old "representation layer" was carrying turn out to be ordinary categorical data: symbolic sizes are the **fiber of the Grothendieck construction** `∫Dat`, and axis identity/alignment is the **pushout/coequalizer** of composition. Neither is a separate representation layer; both live inside the single development. There is **one** development, parametric on an index PROP `D` and an operation PROP `C`, and "prove the propositions once, inherit them everywhere" becomes, in Lean, plain parametricity over a typeclass.

The encoding is therefore a layered tower of typeclasses, each parameterised by the classes below it:

```
ColoredPROP O                                    -- lightweight base; St, Br instances
   ⇣ adapter (the seam)  →  Mathlib MonoidalCategory / SymmetricCategory
DGradedColoredPROP D C   [ColoredPROP D] [ColoredPROP C]   -- core: sh, act, δ, υ, α, axioms
   ├ TemporalGraded   D C   (extends)   -- Scan, Def 3.3–3.5
   ├ RouteStructure   D C   (mixin, STUB)     -- Route, Prop 8.6(ii)      [future_ideas]
   └ SymmetryGraded   D C T (mixin, STUB)     -- equivariance monad, Prop 8.4 [gated; equiv_unif A3]
Algebra D C V   [DGradedColoredPROP D C] [TargetActegory D V]   -- construct()
   └ ParaAlgebra ...        (mixin, STUB)      -- weight tying, pass-as-2-cell  [prop_ideas §7]
```

The base `ColoredPROP` carries the lightweight definitions and the `St`/`Br` instances; the core `DGradedColoredPROP D C` adds the grading data and laws; capabilities (`Scan`, `Route`, symmetry, `Para`) are composable mixins, and `Algebra` is the `construct()` functor into a target actegory. An instantiation pays only for the layers it declares, and the proven core is never edited — new domains are new instances, new capabilities are new mixins.

The single seam that remains is **not** the old "is this mathematics?" boundary but a thinner, differently-drawn one — the **proposition/computation** seam: **does Lean *prove* this or *compute* this?** This is the seam [graded_prop.md](graded_prop.md) itself draws. [§6](graded_prop.md#6-composition-as-pushout) is "a correctness/specification lens, not a composition algorithm": the pushout *explains and certifies* what `Context` does, it does not replace it — so the coequalizer is the *specification* and union-find plus a fresh-name counter is the *implementation*. [§7](graded_prop.md#7-algebras-construct-and-the-para-refinement) keeps the lightweight `Para` encoding and tells us to **note the gap explicitly rather than pay for a bicategory only the specification uses**. Read as the organizing principle of this document: the propositional core (PROP/actegory laws, `∫Dat`, pushout-as-colimit, equivariance, weave uniqueness) is stated over UID-free types and proved; the executable realization (fresh-UID counter, union-find, acset tables / CSV) sits on the other side of the seam, realizing the specification without being proved against it line by line. The seam adapter of §3 is exactly this boundary turned into a definition.

## 2. The base: `ColoredPROP`

Categories are encoded, following [Holtzen (2025)](https://sholtzen.dev/articles/leancat-1.html), as a Lean 4 typeclass parameterised by an object type `ob : Type`. This is the categorical skeleton on which everything else rests; it is carried over from the current design unchanged.

```lean
class SmallCategory (ob : Type) : Type 1 where
  hom     : ob → ob → Type
  id      : ∀ x, hom x x
  comp    : ∀ {X Y Z}, hom X Y → hom Y Z → hom X Z
  id_comp : ∀ {X Y} (f : hom X Y), comp (id X) f = f
  comp_id : ∀ {X Y} (f : hom X Y), comp f (id Y) = f
  assoc   : ∀ {W X Y Z} (f : hom W X) (g : hom X Y) (h : hom Y Z),
              comp (comp f g) h = comp f (comp g h)

infixl:65 " ⟶ " => SmallCategory.hom
notation:65 a " ∘ " b => SmallCategory.comp b a
```

Objects are themselves Lean types, so the monoidal product on objects can be definitional list concatenation; morphisms carry enough structure that the category laws fall to ring axioms (`St`) or list induction (`Br`) with no quotient; and the laws `id_comp`/`comp_id`/`assoc` are propositional equalities discharged by tactics. `-- Python: no typeclass — categories emerge implicitly from the @ operator / Composed wrapper.`

Both **St** and **Br** are *colored PROPs* ([graded_prop.md §2](graded_prop.md), Definition 2.1): symmetric strict monoidal categories whose object monoid is the free monoid `O*` over a set of colors, with `⊗` = list concatenation and `I` = the empty list. The base class carries exactly this structure, **with one addition over the current doc**: the elemental separation axiom `elemental` (the `(Elem)` axiom of [graded_prop.md §2](graded_prop.md)).

```lean
class ColoredPROP (ob : Type) extends SmallCategory ob where
  gen       : Type
  toList    : ob → List gen
  ofList    : List gen → ob
  tensor    : ob → ob → ob := fun a b => ofList (toList a ++ toList b)
  unit      : ob := ofList []
  tensor_assoc  : ∀ a b c, tensor (tensor a b) c = tensor a (tensor b c)
  tensor_unit_l : ∀ a, tensor unit a = a
  tensor_unit_r : ∀ a, tensor a unit = a
  swap      : ∀ a b, hom (tensor a b) (tensor b a)
  tensorHom : ∀ {a b c d}, hom a b → hom c d → hom (tensor a c) (tensor b d)
  elemental : ∀ {X Y} (f g : hom X Y),
                (∀ x : hom unit X, comp x f = comp x g) → f = g   -- (Elem)
```

For both **St** and **Br**, `ob = List gen`, so `toList = id` and `ofList = id`, and the three strictness laws `tensor_assoc`/`tensor_unit_l`/`tensor_unit_r` reduce to `List.append_assoc`, `List.nil_append`, and `List.append_nil` respectively — all dischargeable by `simp [List.append_assoc/nil_append/append_nil]`. The `swap` morphism is a rearrangement that interleaves or separates the two sub-lists, and `tensorHom` is the parallel product.

The **new** field is `elemental`: writing `El(X) := hom unit X` for the *points* (global elements) of `X`, it states that points separate parallel morphisms — `(∀ x : hom unit X, x ; f = x ; g) → f = g`. Equivalently the family `{(x ; −)}_{x ∈ El(X)}` is jointly injective. This is a single `Prop`-valued field, and it is the genuine extra content the current doc lacks: it is precisely what makes the cartesian-lift datum of a morphism **unique** — see weave uniqueness (forward-ref [§5](#5-weaves-as-cartesian-lift-data), Prop 8.2). Without it, Eq. 3 (point-naturality) holds by functoriality but a weave's `(f, P, ρ)` factorisation would not be forced.

The `ColoredPROP` typeclass earns its keep in three ways, as before: generic rearrangements (any list permutation induces a morphism in any colored PROP, proved once), the `St → Br` relationship (the `reindexings` of a `BrBase` are a family of **St** morphisms inside a **Br** morphism — the data of a monoidal functor `St → Br`), and the interchange law `(f ; g) ⊗ (h ; k) = (f ⊗ h) ; (g ⊗ k)`, derivable once from `tensorHom` and `assoc`. `-- Python: no class corresponds to ColoredPROP — the monoidal structure is a paper-level concept (ProdObject[L] wrapping tuple[L,...]).`

### 2.1 `Numeric`

Both **St** and **Br** rely on symbolic dimension expressions — axis sizes and stride coefficients are terms in a free commutative semiring. The right type is Mathlib's `MvPolynomial String ℕ`, already a `CommSemiring` and `DecidableEq` with no extra proof work, covering the Python `Numeric` hierarchy (`Integer`, `FreeNumeric`, `Addition`, `Multiplication`, `Power`) uniformly.

```lean
abbrev Numeric := MvPolynomial String ℕ
-- free variable s  ↦  MvPolynomial.X s     (a degree-1 monomial)
-- literal n        ↦  MvPolynomial.C ↑n    (a constant polynomial)
-- addition, multiplication ↦ ring operations
-- instance : CommSemiring Numeric           -- free from Mathlib
-- instance : DecidableEq Numeric            -- free from Mathlib
-- Python: Numeric (Integer / FreeNumeric / Addition / Multiplication / Power)
```

This is the minimal type making `StMat`'s laws provable: the free commutative semiring on a `String`-indexed generator set, exactly the algebra symbolic axis sizes inhabit. The `ring` tactic works immediately over any `CommSemiring`, so all three `StMat` laws discharge without setup. `MvPolynomial.X s` plays the role of a `FreeNumeric` — a name carrying no interpretation until indeterminates are substituted.

### 2.2 `St` — stride matrices

**St** instantiates `ColoredPROP` with `gen = Axis`. Its objects are shapes (lists of axes); its morphisms are affine coordinate transforms stored as stride matrices over `Numeric`.

```lean
structure Axis where
  name : Option String
  size : Numeric      -- symbolic; filled in at configuration time
-- Python: Axis (abstract UTerm, backed by RawAxis; UID dropped — that is Layer-2)

abbrev StObj := List Axis  -- a shape = an ordered list of axes
-- Python: ProdObject[Axis]
```

A morphism `dom → cod` is a matrix `Λ ∈ ℕ^{|cod|×|dom|}` of `Numeric` coefficients plus a bias vector, with row `j` giving the linear combination of input coordinates producing output coordinate `j`. Using Mathlib's `Matrix` gives the composition law for free:

```lean
structure StMat (dom cod : StObj) where
  coeffs : Matrix (Fin cod.length) (Fin dom.length) Numeric
  bias   : Fin cod.length → Numeric
-- Python: StrideMorphism

def StMat.id (a : StObj) : StMat a a where
  coeffs := 1        -- Matrix.one : Matrix (Fin n) (Fin n) Numeric
  bias _ := 0

def StMat.comp (f : StMat a b) (g : StMat b c) : StMat a c where
  coeffs := g.coeffs * f.coeffs                               -- Matrix.mul
  bias i := Matrix.dotProduct (g.coeffs i) f.bias + g.bias i -- ∑_k g[i,k] * f.bias[k] + g.bias[i]
```

`Matrix.mul` is `(A * B) i j = ∑_k A i k * B k j`; `Matrix.dotProduct v w = ∑_k v k * w k` handles the bias update.

```lean
instance St : ColoredPROP StObj where
  gen    := Axis
  toList := id
  ofList := id
  hom    := StMat
  id     := StMat.id
  comp   := StMat.comp
  -- Coefficient laws: Matrix.one_mul, Matrix.mul_one, Matrix.mul_assoc (Mathlib).
  -- Bias laws: dotProduct linearity, discharged by ring over CommSemiring Numeric.
  id_comp       := by intro _ _ f; simp [StMat.comp, StMat.id, Matrix.one_mul,
                                          Matrix.dotProduct_zero]
  comp_id       := by intro _ _ f; simp [StMat.comp, StMat.id, Matrix.mul_one,
                                          Matrix.dotProduct, Finset.sum_ite_eq']
  assoc         := by intro _ _ _ _ f g h; simp [StMat.comp, Matrix.mul_assoc,
                                                   Matrix.dotProduct_mulVec]; ring
  tensor_assoc  := by simp [List.append_assoc]
  tensor_unit_l := by simp
  tensor_unit_r := by simp [List.append_nil]
  swap a b      := ⟨Matrix.reindex ..., fun i => 0⟩   -- permutation matrix, zero bias
  tensorHom f g :=                                      -- block-diagonal
    { coeffs := Matrix.fromBlocks f.coeffs 0 0 g.coeffs
      bias   := Fin.append f.bias g.bias }
  elemental     := …   -- stride matrices are separated by their points (global elements ⊢ each row);
                       -- a row of Λ is recovered by evaluating at the basis points, so f = g
```

All category laws discharge using Mathlib's `Matrix` API (`Matrix.one_mul`/`mul_one`/`mul_assoc` for coefficients, `ring` over `CommSemiring Numeric` for bias); `Matrix.fromBlocks` builds the block-diagonal `tensorHom`; `Matrix.reindex` the permutation `swap`. `elemental` holds because a stride matrix is determined by its action on global elements (points): evaluating against the basis points recovers each coefficient row, so two stride matrices agreeing on all points are equal.

### 2.3 `Br` — free category over broadcasted base morphisms

**Br** instantiates `ColoredPROP` with `gen = ArrayType`. Because there is no single canonical way to compose two arbitrary broadcasted operations, **Br** morphisms are a free list — the analog of `Composed[Array[B,A], Broadcasted[B,A]]` — making all category laws trivial list lemmas.

```lean
inductive DType
  | reals
  | nat : Numeric → DType           -- Natural(max_value)
-- Python: Datatype / Reals / Natural

structure ArrayType where
  dtype : DType
  shape : StObj                     -- shape lives in Ob(St)
-- Python: Array[B, A]

abbrev BrObj := List ArrayType      -- a product of arrays
-- Python: ProdObject[Array[B,A]]
```

A single `BrBase` is the root morphism of **Br**, bundling one base operation with its reindexings (from **St**), input weaves, and output weaves.

```lean
inductive WeaveSlot
  | fixed : Axis → WeaveSlot   -- retained axis: the reindexing selects a value for this axis at each degree step
  | tiled : WeaveSlot           -- contracted axis: the base op processes the full extent of this axis
-- Python: WeaveMode.TILED ↔ .fixed (retained); concrete Axis slot ↔ .tiled (contracted) — convention inverted

abbrev WeaveShape := List WeaveSlot
-- per-array slot list; distinct from §5's `structure Weave (g)` (the cartesian-lift factorization datum)
-- Python: Weave[B, A] (the Python type is named Weave; it maps to WeaveShape here)

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
-- Python: Broadcasted[B, A, O]
```

The `reindexings` field precisely captures the four cases from the paper — identity, deletion (broadcast), duplication (diagonal), affine scaling (strided convolution) — each a different `StMat`.

```lean
inductive BrMorph : BrObj → BrObj → Type
  | nil  : (a : BrObj) → BrMorph a a
  | cons : BrBase a b → BrMorph b c → BrMorph a c
-- Python: nil ↔ identity Rearrangement; cons ↔ Composed[L, M]

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
  swap a b      := .cons ⟨"swap", [], ..., ...⟩ (.nil _)
  tensorHom f g := ...   -- run f and g in parallel via ProductOfMorphisms
  elemental     := …   -- Br is elemental: see theory.md §Elemental Categories (graded_prop.md (Elem-C))
```

No `sorry` appears: the category laws discharge by `rfl` or one-step structural induction, because list concatenation is associative and `nil` is a two-sided unit definitionally. The `elemental` field is the `(Elem-C)` instance of [graded_prop.md §2](graded_prop.md) — **Br is elemental**, witnessed by the argument in [theory.md §Elemental Categories](theory.md).

The two instances embody a complementary split. **St is semantic**: a stride morphism is the *denotation* of a coordinate transform, not a syntax tree, so composition collapses to a single `Matrix.mul` and the laws come from Mathlib's `Matrix` API plus `ring`. **Br is syntactic (free)**: a composed sequence of broadcasted operations is stored as a list with no canonical simplified form, so the laws are free gifts from list algebra; the price is that symbolic reasoning over **Br** pattern-matches the list rather than inspecting one record. This split mirrors the Python implementation: `StrideMorphism`s compose by multiplying coefficient matrices, `Broadcasted`s by wrapping in `Composed([...])`.

## 3. The seam adapter into Mathlib

The base class above is deliberately lightweight, so its `St`/`Br` instances keep the "tensor = list concat, strictness = `rfl`/`simp`" elegance. But the propositions of [§9](#9-the-propositions-as-generic-theorems) want Mathlib — `MonoidalCategory`, `SymmetricCategory`, `Grothendieck`, `Limits.pushout`, monoidal functors. A single stated **adapter** bridges the two, turning any `ColoredPROP O` into a Mathlib strict symmetric monoidal category.

```lean
-- Strictify FreeMonoidalCategory (Discrete O); produced once, used by all §9 theorems.
instance [ColoredPROP O] : CategoryTheory.MonoidalCategory O := …
instance [ColoredPROP O] : CategoryTheory.SymmetricCategory O := …
```

This is the hybrid foundation of the proposition/computation separation: everything **above** the seam — the instances and executable defs (`St`, `Br`, `StMat.comp`, the union-find of [§7](#7-grothendieck-split-and-composition-as-pushout)) — speaks `ColoredPROP`; everything **below** it — the [§9](#9-the-propositions-as-generic-theorems) theorems — speaks Mathlib. The adapter *is* the proposition/computation boundary of [§1](#1-orientation-one-structure-one-seam) made into a definition: it is the one place where "what Lean computes" is handed to "what Lean proves."

The adapter is produced **once** and reused by every §9 theorem; per-instance proof obligations recur, but the bridge does not. The strictification strategy is the crux: Mathlib categories are not strict, so the development is carried out over `FreeMonoidalCategory (Discrete O)` and **strictified once**, so that downstream all associators and unitors become `Iso.refl` and the PROP equations (`tensor_assoc`, `tensor_unit_l`, `tensor_unit_r`) hold definitionally rather than up to coherent isomorphism. This is what lets a `ColoredPROP` law stated with `=` line up with a Mathlib `MonoidalCategory` whose coherences are isos.

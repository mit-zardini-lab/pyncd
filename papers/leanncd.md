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

The prior design predated [graded_prop.md](graded_prop.md): it formalised `St` and `Br` as two independent PROP instances and divided everything into a "Layer 1 — Mathematical Encoding" and a "Layer 2 — Representation" (UIDs, `Context`, names). That split was an artifact of not yet having the vocabulary `graded_prop.md` now supplies. This reframing dissolves it. The scattered constructions are recognised as **one structure** — a `D`-graded colored PROP — and the two things the old "representation layer" was carrying turn out to be ordinary categorical data: symbolic sizes are the **fiber of the Grothendieck construction** `∫Dat`, and axis identity/alignment is the **pushout/coequalizer** of composition. Neither is a separate representation layer; both live inside the single development. There is **one** development, parametric on an index PROP `D` and an operation PROP `C`, and "prove the propositions once, inherit them everywhere" becomes, in Lean, parametricity over a typeclass.

The encoding is therefore a layered tower of typeclasses, each parameterised by the classes below it:

```
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

The single seam that remains is **not** the old "is this mathematics?" boundary but a thinner, differently-drawn one — the **proposition/computation** seam: **does Lean *prove* this or *compute* this?** This is the seam [graded_prop.md](graded_prop.md) itself draws. [§6](graded_prop.md#6-composition-as-pushout) is "a correctness/specification lens, not a composition algorithm": the pushout *explains and certifies* what `Context` does, it does not replace it — so the coequalizer is the *specification* and union-find plus a fresh-name counter is the *implementation*. [§7](graded_prop.md#7-algebras-construct-and-the-para-refinement) keeps the lightweight `Para` encoding and tells us to **note the gap explicitly rather than pay for a bicategory only the specification uses**. Read as the organizing principle of this document: the propositional core (PROP/actegory laws, `∫Dat`, pushout-as-colimit, equivariance, weave uniqueness) is stated over UID-free types and proved; the executable realization (fresh-UID counter, union-find, acset tables / CSV) sits on the other side of the seam, realizing the specification without being proved against it line by line. The seam adapter of §3 is exactly this boundary turned into a definition.

## 2. The base: `ColoredPROP`

Categories are encoded, following [Holtzen (2025)](https://sholtzen.dev/articles/leancat-1.html), as a Lean 4 typeclass parameterised by an object type `ob : Type`. This is the categorical skeleton on which everything else rests; it is carried over from the prior design unchanged.

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

## 4. The core: `DGradedColoredPROP D C`

Everything above is the base on which the actual subject of this document sits. A `D`-graded colored PROP ([graded_prop.md §3.1](graded_prop.md#31-data)) is a colored PROP `C` (the *operations*) together with the data that exhibits it over a second colored PROP `D` (the *index*): a shape map, a lift action, and the distributivity and action-coherence isomorphisms making `C` a right `D`-actegory. The core class collects exactly that data plus the four named laws.

```lean
class DGradedColoredPROP (D C : Type) [ColoredPROP D] [ColoredPROP C] where
  sh    : ColoredPROP.gen (ob := C) → D        -- shape map: each C-color's underlying D-shape; extends to sh* (monoid hom)
  act   : (C ×ᶜ Dᵒᵖ) ⥤ C                        -- lift action (Mathlib functor, via seam)
  δ     : ∀ X Y P, act.obj (tensor X Y, P) ≅ tensor (act.obj (X,P)) (act.obj (Y,P))
  δ0    : ∀ P, act.obj (unit, P) ≅ unit
  υ     : ∀ X, act.obj (X, unit_D) ≅ X
  α     : ∀ X P Q, act.obj (act.obj (X,P), Q) ≅ act.obj (X, tensor Q P)
  sh_act         : ∀ X P, sh* (act.obj (X,P)) = tensor (sh* X) P     -- (Sh-⊛)
  act_unit_assoc : …    -- actegory triangle + pentagon (υ, α coherences) → right D-actegory
  dist_coh       : …    -- δ, δ0 naturality + interchange with υ/α/σ
  broadcast_gen  : …    -- (Broadcast-gen): every C-morphism = [f,P] ; ρ, f degree-trivial
```

Each field is a direct transcription of the [graded_prop.md §3.1](graded_prop.md#31-data) data. `sh` is the **shape map** `sh : O_C → Ob D` sending each `C`-color to its underlying `D`-shape (its list of sub-wires); it extends to the monoid homomorphism `sh*` on objects used by the (Sh-⊛) law. `act` is the **lift action** `act : C × Dᵒᵖ ⥤ C` — a Mathlib functor, available because the seam adapter of [§3](#3-the-seam-adapter-into-mathlib) makes `C` and `D` Mathlib categories. Its two specializations are theory.md's lift notation: `[f, P] := act(f, 𝟙_P)` is the **batch lift** of `f` (covariant in `C`), and `[X, η] := act(𝟙_X, η)` is the **reindexing** along `η : P → Q` (contravariant in `D`, since `D` enters opposite). `-- Python: batch lift [f,P]`

The four isomorphism fields are the **distributivity** and **action-coherence** isos. `δ` and `δ0` make the lift distribute over juxtaposition — `(X ⊗ Y) ⊛ P ≅ (X ⊛ P) ⊗ (Y ⊛ P)` and `I_C ⊛ P ≅ I_C`. `υ` and `α` are the actegory coherence isos — `υ : X ⊛ I_D ≅ X` (lifting by the unit shape is trivial) and `α : (X ⊛ P) ⊛ Q ≅ X ⊛ (Q ⊗ P)` (composing two lifts; the order `Q ⊗ P` is what makes `⊛` a **right** action of `(D, ⊗, I_D)`). Together `υ`/`α` are precisely the unitor and multiplicator exhibiting `C` as a right `D`-actegory.

The four `Prop`-valued fields are the named laws of [graded_prop.md §3.2](graded_prop.md#32-axioms), one field each:

- `sh_act` is **(Sh-⊛)**: `sh*(X ⊛ P) = sh*(X) ⊗ P` — lifting by `P` appends `P` to the shape.
- `act_unit_assoc` bundles **(Act-unit / Act-assoc)**: `υ` and `α` satisfy the triangle and pentagon coherences, i.e. `C` is a right `D`-actegory.
- `dist_coh` bundles **(Dist-nat / Dist-coh)**: `δ`, `δ0` are natural and satisfy the interchange coherence with `υ`, `α`, and the symmetry `σ` — the lift is a strong symmetric monoidal action in the `C`-variable.
- `broadcast_gen` is **(Broadcast-gen)**: every `C`-morphism factors as `[f, P] ; ρ` with `f` degree-trivial (built without `act` over a non-unit `P`) and `ρ` a reindexing assembled from `act(𝟙, −)` and the coherence isos. This is the generation principle that makes weaves ([§5](#5-weaves-as-cartesian-lift-data)) exist.

Note that **(Act-functor)** of [graded_prop.md §3.2](graded_prop.md#32-axioms) (`act` respects identities and composition in both variables, so `[f ; g, P] = [f, P] ; [g, P]`) and **(Sh-⊗)** (`sh*` is a monoid homomorphism) are not separate fields: the first is the `Functor` laws already carried by `act`, the second is built into the definition of `sh*` from `sh`. And **(Elem-C)** is not here either — it is the `elemental` field of `ColoredPROP` ([§2](#2-the-base-coloredprop)), inherited from the instance `[ColoredPROP C]`.

`D` and `C` are **explicit class parameters**, not `outParam`s. This is deliberate: with both free, Lean's instance search needs them pinned for `[DGradedColoredPROP D C]` to resolve predictably (and so that `Br`-as-graded and `Br`-as-index occupy distinct instance positions without collision). The instance-resolution discipline this implies is taken up in [§11](#11-lean-formalization-notes).

### 4.1 Derived: `ev_p` and Eq. 3

Theory.md's batch-lift defining property (Eq. 3) is **not** an axiom of the core class — it falls out of `act` being a functor. For a point `p : I_D → P` in `D`, the *slice at `p`* is a derived natural transformation, and its naturality square is Eq. 3.

```lean
def ev_p [DGradedColoredPROP D C] (p : hom unit_D P) : (act ⊛ P) ⟶ Id := act.map ⟨𝟙, p⟩ ≫ υ
-- Eq. 3:  [f,P] ≫ (Y ⊛ p) = (X ⊛ p) ≫ f   -- naturality of ev_p, from Functor.map_comp
```

`ev_p` is a `def`, not a field: it is `act.map ⟨𝟙, p⟩` post-composed with the unitor `υ`, a natural transformation `(− ⊛ P) ⇒ (− ⊛ I_D) ≅ Id_C`. Because `act` is a functor, each `ev_p` is *automatically* natural — its naturality square at `f : X → Y` is exactly **(Eq. 3)** `[f, P] ; (Y ⊛ p) = (X ⊛ p) ; f`, discharged by `Functor.map_comp`. So Eq. 3 is **built in** by functoriality of `act`, not posited ([graded_prop.md §3.2](graded_prop.md#32-axioms)). The genuine content lives elsewhere: it is `elemental` (the points `ev_p` jointly separate morphisms) that pins down the weave of [§5](#5-weaves-as-cartesian-lift-data) — Eq. 3 alone holds for free and does not force the factorization.

## 5. Weaves as cartesian-lift data

The (Broadcast-gen) law says every `C`-morphism factors as `[f, P] ; ρ`. A **weave** is a witness of that factorization for a particular morphism — and, as [graded_prop.md §3.3](graded_prop.md#33-weaves-as-cartesian-lift-data) shows, it is precisely the cartesian-lift datum of the grading fibration `C → D`.

> **Naming.** `WeaveShape` ([§2](#2-the-base-coloredprop)) is the *per-array slot list* (`List WeaveSlot`, the shape of one wire). The `Weave g` below is a *different* concept: the cartesian-lift factorization witness for a whole morphism `g`. The Python type named `Weave` maps to the former; this `structure Weave` is the latter.

```lean
structure Weave [DGradedColoredPROP D C] {X Y : C} (g : hom X Y) where
  f       : hom X' Y'    -- base op (degree-trivial)
  P       : D
  ρ       : …            -- reindexing assembled from act(id,−) + coherence isos
  factors : g = [f, P] ≫ ρ

theorem weave_unique [DGradedColoredPROP D C] {X Y} (g : hom X Y) :
    Subsingleton (Weave g)         -- Prop 8.2, from elemental + broadcast_gen
```

A `Weave g` records the (Broadcast-gen) factorization of `g`: a degree-trivial base op `f`, a degree `P ∈ D`, a reindexing `ρ` (assembled from `act(𝟙, −)` and the coherence isos), and a proof that `g = [f, P] ; ρ`. Per wire, the shape `sh(color) ∈ Ob D` is a list of sub-colors that the factorization partitions into **target** sub-colors (acted on directly by `f`) and **tiling** sub-colors (supplied by the degree `P` through `ρ`); the permutation relating the canonical "targets-first" order to the wire's actual sub-color order is theory.md's `Ω_w`, recovered from the symmetry `σ`. This is **precisely the cartesian-lift datum** of the grading fibration `C → D` ([graded_prop.md §3.3](graded_prop.md#33-weaves-as-cartesian-lift-data)): a weave is the choice of how a morphism's wires sit over their `D`-shapes, with the tiling part pulled back along the degree. `-- Python: Weave._shape records the per-wire TILED/target partition`

`weave_unique` (Proposition 8.2) makes `Weave g` a `Subsingleton` — at most one weave, up to the canonical coherence isos. This is what turns `Weave` into a **datum, not a choice**: the factorization is forced, not selected. The proof draws on both the `elemental` field of `[ColoredPROP C]` (points separate morphisms, so the degree `P` and the target/tiling partition are determined) and `broadcast_gen` (a factorization exists at all). Without `elemental`, Eq. 3 would still hold by functoriality but the `(f, P, ρ)` factorization would not be unique.

## 6. Mixins: Scan, Route, Symmetry, Para

The core `DGradedColoredPROP D C` of [§4](#4-the-core-dgradedcoloredprop-d-c) carries the lift, the actegory coherences, and the weave factorization — and nothing more. Capabilities beyond that bare grading are added as **composable mixins**: a `Scan`, a `Route`, an equivariance constraint, a `Para` refinement. Each is its own typeclass; an instantiation declares only the mixins it actually uses and pays only for their fields and obligations. This is exactly what keeps the proven core un-edited as the framework grows: a new capability is a new class layered on `DGradedColoredPROP`, never a field added to it, so the [§9](#9-the-propositions-as-generic-theorems) theorems stated over the core continue to hold unchanged at every instantiation. The four mixins of this family are `TemporalGraded` (Scan, given in full below), the `RouteStructure` and `SymmetryGraded` stubs, and `ParaAlgebra` (forward-referenced to [§7](#7-grothendieck-split-and-composition-as-pushout), since it layers on the `Algebra` rather than on the graded PROP).

### 6.1 `TemporalGraded` — Scan

```lean
class TemporalGraded (D C : Type) [ColoredPROP D] [ColoredPROP C]
    extends DGradedColoredPROP D C where
  L          : D                    -- temporal object (Δ₊ / ℕ-graded); prefix inclusions ιₘ
  restrict   : ∀ m, …               -- directed action act(−, ιₘ); NO point-evaluation ev_q
  iterate    : …                    -- finite iteration of a parametric step endofunctor (Def 3.4)
  trace      : …                    -- scanl state history H ⊗ L_{N+1} (Def 3.5)
  lift_fold_dist : …                -- act(Scan,P) ≅ Scan(act(step,P)) for P orthogonal to L
```

`TemporalGraded` internalizes the four additions of [graded_prop.md §3.4](graded_prop.md#34-the-temporal-grading-and-making-scan-first-class) that turn `Scan` from a bare generator into a definition. `L` is the **temporal object** of Definition 3.3 — an `Ob D` carrying the augmented-simplex / `(ℕ,+,0)` length grading, with prefix inclusions `ιₘ : [0..m] ↪ [0..N]`. `restrict` is the **directed action**: restriction natural transformations `act(−, ιₘ) : (− ⊛ [0..N]) ⇒ (− ⊛ [0..m])` along the `ιₘ`, satisfying the unit and composition laws of an action. `iterate` is the **finite iteration** of Definition 3.4 — for a parametric step endofunctor (the per-step inputs ride as parameters) and a length `N`, the `N`-fold iterate and its catamorphism `cata(step)` exist (for fixed `N` this is plain `N`-fold composition; no fixpoint, no natural-numbers object, until unbounded length is wanted). `trace` is the **state history** of Definition 3.5 — codomain `H ⊗ L_{N+1}` (scanl), with the coherence that truncating the trace along `ιₘ` agrees with running the `m`-fold. `lift_fold_dist` is the **lift–fold distributivity** law: for an ordinary degree `P` orthogonal to `L`, `act(Scan, P) ≅ Scan(act(step, P))`.

With these fields in hand, **`Scan := cata(step)` is a definition** over the temporal grading, not a generator posited by hand. Two consequences follow rather than being assumed. First, the **prefix-restriction law is a corollary** (Proposition 8.7): theory.md's law that `Scan_N` restricted to the first `m` steps equals `Scan_m` falls out of the catamorphism universal property `cata(step) ∘ in = step ∘ F(cata(step))` and its uniqueness — it need not be a separate axiom. Second, **`Scan` batches** along any axis `P` orthogonal to `L` (Proposition 8.8): `lift_fold_dist` is exactly what makes `act(Scan, P) ≅ Scan(act(step, P))`, so a batched recurrence is one fold run independently per batch coordinate, and `Scan` participates in the `vmap`/batch strategies like any other morphism. `-- Python: Scan in TensorDSL.py`

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

`ParaAlgebra` (the `Para` refinement mixin on `construct()`) is *not* introduced here. It is presented in [§7](#7-grothendieck-split-and-composition-as-pushout) alongside the `Algebra` class, because it layers on the algebra — refining `Para(C) → Para(V)` as a 2-functor, with weight tying as passes-as-2-cells — rather than on the graded PROP. It is listed in this section's title only because it belongs to the same mixin family.

## 7. Grothendieck split and composition as pushout

Two of the constructions the current design parked in its "representation layer" — symbolic axis *sizes* and axis *identity/alignment* — are, on the graded-PROP reading, ordinary categorical data. Sizes are the **fiber of a Grothendieck construction**; alignment is a **pushout/coequalizer**. This section states both, and then makes the proposition/computation seam of [§1](#1-orientation-one-structure-one-seam) concrete: the categorical objects are the *specification*, and the executable structures carried forward from the old "Layer 2" (a fresh-UID counter, a union-find `Context`) are the *implementation* that realizes it. There is no separate representation layer; the old Layer 2 is dissolved into this single development.

### 7.1 The structure/data split as `∫Dat`

Strip the numeric content from the `D`-colors and one is left with the **structural index PROP** `D♯` (colors are formal size-symbols, morphisms are symbolic reindexings); `C♯` is then `D♯`-graded ([graded_prop.md §5](graded_prop.md#5-the-structuredata-split-as-a-grothendieck-construction)). A **data functor** `Dat : C♯ → Set` sends each structural object to its set of admissible size-assignments, acting *trivially on morphisms* — data is unconstrained by connectivity ([acset.md §The Grothendieck Construction](acset.md#the-grothendieck-construction)). The **Grothendieck construction** `∫Dat` has objects `(c, d)` with `c ∈ C♯` and `d ∈ Dat(c)`, and morphisms the `C♯`-morphisms with no compatibility condition on data, and

> **`C ≅ ∫Dat`**

recovers the fully-sized graded PROP (graded_prop.md Prop 8.3).

```lean
-- Dat valued in (discrete categories of) size-assignments over the semiring `Numeric`.
def Dat [ColoredPROP C] : Cˢʰᵃʳᵖ ⥤ Type := …   -- the data functor; trivial on morphisms (graded_prop.md Def 5.1)
def Dat' [ColoredPROP C] : Cˢʰᵃʳᵖ ⥤ CategoryTheory.Cat := …   -- as a functor into Cat
-- C ≅ Grothendieck Dat'      -- CategoryTheory.Grothendieck of the data functor
example : C ≅ CategoryTheory.Grothendieck Dat' := … -- `Iso.refl`-level when C is *built* as ∫Dat
```

Mathlib supplies `CategoryTheory.Grothendieck` for the Grothendieck construction of a functor into `Cat`; with `Dat` valued in discrete categories of size-assignments, `∫Dat` is a direct instance. The iso `C ≅ ∫Dat` is a per-instantiation theorem in general, but it is **definitional — `Iso.refl` — if `C` is *built* as `∫Dat`**, which is the recommended posture: define the sized PROP as the integral, and the splitting is true by construction rather than by proof.

### 7.2 `FreeNumeric` is the fiber, not a layer

The `Numeric := MvPolynomial String ℕ` of [§2.1](#21-numeric) is precisely the data the fiber `Dat(c)` ranges over. A symbolic axis size is a term in the free commutative semiring on `String`-named generators; a `FreeNumeric` — the Python `UTerm` that names an as-yet-unknown size — is a single generator `MvPolynomial.X s`, carrying no interpretation until indeterminates are substituted.

```lean
abbrev Numeric := MvPolynomial String ℕ
-- MvPolynomial.X s  -- a FreeNumeric: a symbolic axis size, the fiber datum Dat(c)
-- a size-assignment d ∈ Dat(c) picks a Numeric for each structural axis-symbol of c
```

The point of stating this here is the **dissolution**: in the old design `FreeNumeric`, symbolic sizes, and `Numeric` lived in "Layer 2 — Representation," as if they were non-mathematical bookkeeping. They are not. They are the **`Dat(c)` fiber data** of the Grothendieck construction of [§7.1](#71-the-structuredata-split-as-dat) — as categorical as the structural skeleton `C♯` itself. There is no representation layer to carry them; they are part of the single graded-PROP development, living in the fiber over `C♯`.

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

This is where the proposition/computation seam of [§1](#1-orientation-one-structure-one-seam) becomes tangible. The coequalizer of [§7.3](#73-composition-as-pushout) is the **specification**; the executable structures carried forward from the old "Layer 2" are the **implementation**. They meet at the seam, and neither replaces the other.

**Fresh-UID counter — `TermM`.** Constructing a new symbolic axis mints a fresh identity. Python does this with `random.randint` as a construction side-effect; Lean 4 is pure, so the fresh-name counter is threaded explicitly as a state monad. This is the *executable* side — it computes identities; it proves nothing.

```lean
abbrev UID := ℕ

structure UData where
  uid  : UID
  name : Option DynamicName

abbrev TermM := StateM ℕ            -- the fresh-UID counter monad

def freshUData : TermM UData := do
  let n ← get; set (n + 1); return ⟨n, none⟩
-- Constructing a new axis / FreeNumeric runs in TermM; pure code (composition, proofs) does not.
-- A counter, not random ints: term construction becomes reproducible and testable.
-- UIDs carry no semantic content — only equality/inequality of two UIDs matters.
```

**Union-find — `Context` / `EqClass`.** The `Axis`-component coequalizer is *computed* as a pure-functional union-find. An `EqClass` is one equivalence class — a `Finset` of UIDs together with its canonical representative; a `Context` is a disjoint list of them. `Context.merge` unions a new class with any overlapping existing ones (Python's `Context.append_bucket`); `Context.apply` substitutes every UID by its class representative throughout a term. The **canonical representative is the member with the largest UID** — and *this is the universal cocone vertex* of the Stage-2 pushout: choosing it on the nose is what strictifies composition.

```lean
/-- One equivalence class: a set of UIDs with one canonical representative. -/
structure EqClass (α : Type*) where
  bucket    : Finset UID
  canonical : WithUID α            -- representative chosen by largest UID = cocone vertex

/-- A context is a disjoint list of equality classes. -/
structure Context (α : Type*) where
  classes : List (EqClass α)

/-- Merge a new class in, unioning with any overlapping classes (= Context.append_bucket). -/
def Context.merge (ctx : Context α) (cls : EqClass α) : Context α :=
  let overlapping := ctx.classes.filter (fun c => ¬ c.bucket.Disjoint cls.bucket)
  let merged : EqClass α := overlapping.foldl
    (fun acc c => ⟨acc.bucket ∪ c.bucket,
                   if acc.canonical.data.uid ≥ c.canonical.data.uid
                   then acc.canonical else c.canonical⟩)
    cls
  ⟨merged :: ctx.classes.filter (fun c => c.bucket.Disjoint cls.bucket)⟩

/-- Substitute every UID in each class by its canonical representative throughout a term. -/
def Context.apply [TermTraversable α] (ctx : Context α) (target : α) : α :=
  ctx.classes.foldl (fun t cls =>
    TermTraversable.traverseUID
      (fun d => if d.uid ∈ cls.bucket then cls.canonical.data else d) t)
    target
```

The framing is the whole point. **The pushout/coequalizer is the spec; union-find plus the fresh-name counter is the implementation; they meet at the seam.** The Stage-2 colimit *certifies* the gluing — associativity (the pasting lemma), the precise error semantics ("no cocone" = alignment failure, "inconsistent attributes" = size mismatch), the canonical representative as cocone vertex — while `Context` *computes* it in near-linear time. A Lean development proves the coequalizer is what it claims; it does not re-derive `Context` line-by-line, and pyncd would never invoke a generic colimit solver. The substitution machinery `Context.apply` rides on a `TermTraversable` typeclass — the Lean stand-in for Python's reflective `deep_reconstruct`, one explicit traversal instance per decorated type — but that, the `WithUID` decoration, and `DynamicName` are display/identity bookkeeping on the executable side, never propositional content.

### 7.5 Algebras and `construct()`

The algebra `F` (graded_prop.md Def 7.2 / [§7](graded_prop.md#7-algebras-construct-and-the-para-refinement)) is the strong symmetric monoidal, `D`-equivariant functor `C → V` into a target actegory — the categorical content of `ConstructedModule.construct()`. It is the last layer of the tower, and the clearest instance of the doc's recurring shape: a typeclass parameterised by the classes below it. The target `V` is itself a right `D`-actegory; the algebra is parametric on both the source graded PROP `C` and that target `V`:

```lean
class TargetActegory (D V : Type) [ColoredPROP D] where
  actV : (V ×ᶜ Dᵒᵖ) ⥤ V                              -- P acts by appending dimensions (PyTorch tensors)
  …                                                   -- same υ/α/δ coherences as §4, now in V

structure Algebra (D C V : Type) [DGradedColoredPROP D C] [TargetActegory D V] where
  F        : C ⥤ V                                    -- strong symmetric monoidal (Mathlib MonoidalFunctor)
  equivar  : ∀ X P, F.obj (act (X,P)) ≅ actV (F.obj X, P)   -- D-equivariance
  coh      : …                                        -- commutes with υ, α, δ; preserves ev_p
-- a morphism of algebras is a MonoidalNatTrans; weight tying collapses parameters via Δ.

class ParaAlgebra (D C V : Type) [DGradedColoredPROP D C] [TargetActegory D V]
    extends Algebra D C V where … -- STUB: Para(C) → Para(V) 2-functor; passes-as-2-cells, weight tying
```

The full development of these — the `equivar`/`coh` obligations, the `Para` refinement, weight tying as a reparameterization 2-cell — lives in the propositions and instantiation sections ([§9](#9-the-propositions-as-generic-theorems), [§10](#10-instantiation-and-future-extensions)) and the lightweight-`Para` note of [§11](#11-lean-formalization-notes); the trained model is a *section of the Para fibration over `∫Dat`*, tying the algebra back to the Grothendieck split of [§7.1](#71-the-structuredata-split-as-dat). `-- Python: ConstructedModule.construct()`

## 8. Acsets and Python interop

### 8.1 `SBrInstance` as a finite presentation of an `∫Dat`-morphism

An acset instance is a **finite presentation of a single `∫Dat`-morphism**: its `C♯`-part is the connectivity, its `Dat`-part the sizes, coefficients, and datatypes ([graded_prop.md §5](graded_prop.md#5-the-structuredata-split-as-a-grothendieck-construction)). For `Br` the schema is `S_Br` and the instance is `SBrInstance`; the schema, the five entity types (`Axis`, `Equation`, `Array`, `ArrayAxis`, `Sample`), the C-set/attribute split, and the worked encoding are developed in [acset.md](acset.md) — referenced here, not re-derived. What matters for the Lean encoding is the relationship to [§7](#7-grothendieck-split-and-composition-as-pushout): an `SBrInstance` exported via `write_sbr`/`read_sbr` *is* a functor `G : S_Br → Set` — one point of the `S_Br`-instance category, i.e. one `∫Dat`-morphism — and its CSV tables are the C-set/attribute halves of that morphism written separately.

The `SBrInstance`'s four tables become a Lean `structure` whose fields are finite lists, mirroring acset.md's [Lean encoding section](acset.md#from-sbrinstance-to-a-diagram-in-br-a-lean-encoding):

```lean
structure SBrInstance where
  equations  : List EquationRow                 -- one row per Equation
  arrays     : List ArrayRow                    -- Array rows: slot, is_input, datatype_tag, op_predicate, …
  array_axes : List ArrayAxisRow                -- ArrayAxis rows: is_target, position (the Weave._shape interleaving)
  samples    : List SampleRow                   -- Sample rows: (src, tgt, coeff, offset) — the affine reindexing
  axis_sizes : List (UID × Numeric)             -- the Dat-part: each axis-UID's symbolic size
```

acset.md interprets this `G` as a strict monoidal functor `D : J → Br` from a finite index category `J` (objects = the program's arrays, morphisms = its equations) into the `Br` of [§2.3](#23-br--free-category-over-broadcasted-base-morphisms), using two Mathlib shortcuts that are exactly this document's choices: **`MvPolynomial String ℕ` for `Numeric`** (so `ring` discharges the `StMat` laws — the `Dat`-fiber type of [§7.2](#72-freenumeric-is-the-fiber-not-a-layer)) and **`Matrix` for `StMat.coeffs`** (for the matrix lemmas). `J` is the *free strict monoidal category* on the equation quiver — Mathlib's `FreeMonoidalCategory` strictified — so specifying `D` on generators determines it uniquely; the functor laws are a consequence. The `axis_sizes` table populates the `Dat(c)` fiber; the `equations`/`arrays`/`array_axes`/`samples` connectivity is the `C♯`-morphism. This `D : J → Br` is the finite, written-down witness of one object/morphism of the `Grothendieck Dat'` instance of [§7.1](#71-the-structuredata-split-as-dat).

### 8.2 The seam made tangible

The acset tables and their CSV serialization are the **executable realization** of the `∫Dat` *specification* — the same proposition/computation seam as [§7](#7-grothendieck-split-and-composition-as-pushout), now in fully concrete form. `write_sbr`/`read_sbr` write the two halves of an `∫Dat`-morphism to separate tables — connectivity (the C-set part: `equations`, `arrays`, `array_axes`, `samples`) and data (the attribute part: `axis_sizes`, coefficients, datatypes) — which is precisely the Grothendieck split serialized. The same `Axis` UIDs appear in the term world's `Weave` objects and in the acset's `ArrayAxis` rows, so any `Context`-mediated unification (the [§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer) coequalizer computation) is reflected in both views without a round-trip. The categorical object `∫Dat` is what a Lean development *proves about*; the acset tables and CSV are what pyncd *computes and stores*. They meet at the seam.

### 8.3 Consolidated Python correspondence

The tables below collect the systematic Lean ↔ Python correspondence, **organized by the new tower** — base, core, mixins, the Grothendieck/pushout seam, the acset realization — rather than by the dissolved Layer 1 / Layer 2 split. Inline one-line pointers in earlier sections (e.g. `act` ↔ batch lift, `BrBase` ↔ `Broadcasted`) are the local view; this is the systematic one.

| Lean (new tower) | Python | Notes |
| --- | --- | --- |
| `SmallCategory` / `ColoredPROP` | implicit / `ProductCategory` | category and monoidal laws are unstated in Python; paper-level only |
| `ColoredPROP.elemental` | — | new `(Elem)` field; no Python witness |
| `List gen` (objects) | `ProdObject[L]` | Python wraps `tuple[L,…]` in a Term; Lean uses `List` directly |
| `StMat` | `StrideMorphism` | stride *matrix* (`Matrix … Numeric` + bias) vs bundled stride record |
| `BrBase` | `Broadcasted` | base op + reindexings; `Fin`-indexed weaves vs runtime tuples |
| `BrMorph` | `Composed` | free list of `BrBase` vs `content: tuple[M,…]` |
| `ProductOfMorphisms` ↔ `tensorHom` | `ProductOfMorphisms[L, M]` | `ColoredPROP.tensorHom` (a morphism) vs a data wrapper |
| `DGradedColoredPROP.act` | batch lift `[f,P]` | the lift action; `[f,P] = act(f, 𝟙_P)`, `[X,η] = act(𝟙_X, η)` |
| `WeaveShape` / `structure Weave` | `Weave._shape` | per-array shape (`WeaveShape`); §5's `structure Weave` = the cartesian-lift factorization witness |
| `∫Dat` instance (`Grothendieck Dat'`) | `SBrInstance` | finite presentation of one `∫Dat`-morphism |
| `Numeric` = `MvPolynomial String ℕ` | `Numeric` / `FreeNumeric` | the `Dat(c)` fiber; `MvPolynomial.X s` ↔ a `FreeNumeric` generator |
| `Context` / `EqClass` | `Context` / `EqualityClass` | union-find = the *implementation* of the coequalizer ([§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer)) |
| `TermM` = `StateM ℕ` | random-int UID side-effect | the fresh-name counter; Lean threads state, Python mutates a global source |
| `TermTraversable` | `deep_reconstruct` | per-type traversal instance vs `__dataclass_fields__` reflection |
| `Algebra.F` | `ConstructedModule.construct()` | the algebra functor `C → V` (full class in the §7.5 / propositions development) |
| `DynamicName` | `DynamicName` | display only — see [§12](#12-appendix-out-of-scope), out of scope |

The single coherent message of the table: every row that the old design would have filed under "Layer 2 — Representation" (`Numeric`/`FreeNumeric`, `Context`/`EqClass`, `TermM`, `TermTraversable`, `DynamicName`) is now placed by its categorical role — fiber datum, coequalizer implementation, fresh-name counter, traversal, or display — on one side or the other of the proposition/computation seam. There is no representation layer; there is one tower with one seam.

## 9. The propositions as generic theorems

Every proposition of [graded_prop.md §8](graded_prop.md#8-propositions-the-synthesis-organizes) is a statement about the **core class fields and axioms only**. Each begins with the same generic preamble:

```lean
variable {D C : Type} [ColoredPROP D] [ColoredPROP C] [DGradedColoredPROP D C]
```

and mentions nothing beyond `act`, `δ`/`δ0`/`υ`/`α`, `sh`, and the named `Prop`-fields (`sh_act`, `act_unit_assoc`, `dist_coh`, `broadcast_gen`, plus the base `elemental`). Because the only hypotheses are class members, each proposition is **proved once, at the graded-PROP level, and inherited at every instance** — the `DGradedColoredPROP St Br` of [§10](#10-instantiation-and-future-extensions), the `DGradedColoredPROP Br CMod` MoE level, a future `Graph→C`, and the swapped-`D` rows all receive it with no per-domain proof. This is the Lean form of [graded_prop.md](graded_prop.md)'s central promise, and it *is* parametricity over a typeclass: a `theorem` whose only free assumption is `[DGradedColoredPROP D C]` applies verbatim wherever that instance resolves.

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

The inheritance is about the **theorems**, not the instance obligations. When a new `instance : DGradedColoredPROP D C` is declared, the propositions above transfer to it for free — but the instance must still **discharge the coherence `Prop`-fields** of the core: the actegory triangle and pentagon (`act_unit_assoc`), the distributivity coherences (`dist_coh`), and `sh_act`/`broadcast_gen`/`elemental`. "Inherit everywhere" names the proven propositions riding on those fields; it does not waive the obligation to *supply* the fields. Each new domain pays that fixed, finite coherence cost once; everything built on top of the core is then free.

## 10. Instantiation and future extensions

### 10.1 `D = St`, `C = Br` — the flagship instance

Today's instantiation is `D = St`, `C = Br`: an index PROP of axis lengths (colors = `Numeric` sizes, morphisms = stride matrices) grading an operation PROP of broadcasted arrays. The instance header supplies the core fields; the named `Prop`-fields are discharged by the laws established in [§2](#2-the-base-coloredprop)–[§4](#4-the-core-dgradedcoloredprop-d-c).

```lean
instance : DGradedColoredPROP St Br where
  sh    := fun a => a.shape        -- the array's shape: sh([a, A]) = A
  act   := …                       -- batch lift + reindexing (theory.md Lift Operations)
  δ     := …                       -- [X ⊗ Y, P] ≅ [X,P] ⊗ [Y,P]   (batch lift distributes)
  δ0    := …                       -- [I, P] ≅ I
  υ     := …                       -- [X, I_St] ≅ X   (grading by the unit shape is trivial)
  α     := …                       -- [[X,P],Q] ≅ [X, Q ⊗ P]
  sh_act         := …              -- (Sh-⊛): sh*([X,P]) = sh*(X) ⊗ P
  act_unit_assoc := …              -- actegory triangle + pentagon, by St affine-stride algebra
  dist_coh       := …              -- δ/δ0 naturality + interchange, from the batch-lift defn
  broadcast_gen  := …              -- every Br morphism is a broadcasted operation (Def 13)
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
| `dist_coh` | `δ`/`δ0` naturality and interchange with `υ`/`α`/swap, from the batch-lift definition |
| `broadcast_gen` | `(Broadcast-gen)`: every `Br` morphism is a broadcasted operation ([theory.md §Broadcasting](theory.md#broadcasting)) |
| `elemental` | `Br` is elemental ([theory.md §Elemental Categories](theory.md#elemental-categories)) — the base `ColoredPROP` field |

Every row is a definitional unfolding of the core with `D := St`, `C := Br`. With the instance in place, all of [§9](#9-the-propositions-as-generic-theorems) holds for `Br` with no `Br`-specific proof.

### 10.2 The additive-extension menu

The framework grows by **adding instances and mixins**, never by editing the proven core. Each direction below is one or the other.

| Extension | Lean addition | Kind |
| --- | --- | --- |
| `D = Br` mixture-of-experts | `instance : DGradedColoredPROP Br CMod` ([graded_prop.md §9.2](graded_prop.md#92-the-speculative-third-level-d--br)) — models as wires, `⊛` tiles a base computation over a family of models | **new instance** |
| swap-`D`: graph / incidence cat. | `instance : DGradedColoredPROP Graph C` — gather-along-edge reindexing; GNNs, meshes (fixed graph = weave; per-sample graph = `Route`) ([graded_prop.md §9.3](graded_prop.md#93-the-horizontal-axis-swapping-d)) | **new instance** |
| swap-`D`: group `BG` / `Rep(G)` | `instance : DGradedColoredPROP (Rep G) C` — group-translation reindexing; equivariant & steerable nets | **new instance** |
| swap-`D`: Markov cat. `Stoch` | `instance : DGradedColoredPROP Stoch C` — Markov-kernel reindexing; sampling, VAE, SMC | **new instance** |
| swap-`D`: metric / enriched cat. | `instance : DGradedColoredPROP Metric C` — distance-kernel reindexing; continuous conv, neural fields | **new instance** |
| swap-`D`: partition lattice | `instance : DGradedColoredPROP Partition C` — assignment-map reindexing; pooling, clustering, slots (fixed = weave; learned = `Route`) | **new instance** |
| swap-`D`: resource monoid | `instance : DGradedColoredPROP Resource C` — store-vs-recompute reindexing; checkpointing / scheduling ([future_ideas.md §8](future_ideas.md#8-prioritized-implementation-roadmap) item 4.5) | **new instance** |
| data-dependent routing | `class RouteStructure` ([§6.2](#62-route-and-symmetry-stubs)) — Prop 8.6(ii), the gate as a `Para` parameter | **new mixin** |
| equivariance | `class SymmetryGraded` ([§6.2](#62-route-and-symmetry-stubs)) — Prop 8.4 via the EM-category of `T : Monad D`, gated | **new mixin** |
| weight tying / passes | `class ParaAlgebra` ([§7.5](#75-algebras-and-construct)) — `Para(C) → Para(V)` 2-functor, passes-as-2-cells | **new mixin** |
| unbounded recurrence | corecursion / coalgebra refinement of `TemporalGraded` — the `cata`/`ana` companion | **new mixin** |

No row is a core edit: a new domain is a new `instance` (it supplies the core fields and discharges the coherence obligations of [§9](#9-the-propositions-as-generic-theorems)), and a new capability is a new mixin layered on top (`extends DGradedColoredPROP`, or `extends Algebra` for `ParaAlgebra`). The classification of [graded_prop.md §8.6](graded_prop.md#8-propositions-the-synthesis-organizes) is `D`-uniform, so every swap-`D` row inherits the same weave-vs-`Scan`-vs-`Route` decision procedure.

The MoE level deserves a specific word, because it is the one place the same category `Br` appears twice. The vertical stack `D = Br` reuses **all** of [§9](#9-the-propositions-as-generic-theorems) with zero new proof, and this is not a coincidence — it is forced by the instance-resolution discipline. `Br`-as-graded — the `DGradedColoredPROP St Br` instance of [§10.1](#101-d--st-c--br--the-flagship-instance) — occupies the **`C` position** of `DGradedColoredPROP`. `Br`-as-index — the `[ColoredPROP Br]` argument of `instance : DGradedColoredPROP Br CMod` — occupies the **`D` position**. The two are different parameter slots of the class, so the two roles of `Br` never collide in instance search: declaring `DGradedColoredPROP Br CMod` does not overlap or shadow `DGradedColoredPROP St Br`. Because the §9 theorems are generic in both `D` and `C`, they fire for `DGradedColoredPROP Br CMod` the instant it resolves, exactly as they fire for `DGradedColoredPROP St Br` — the MoE level is new data, not new mathematics.

## 11. Lean formalization notes

The strategy of [graded_prop.md §10](graded_prop.md#10-lean-formalization-notes) reads as a Mathlib shopping list, and most of the tower lands on existing `CategoryTheory.*` machinery. The base colored PROP and the seam adapter ([§2](#2-the-base-coloredprop)–[§3](#3-the-seam-adapter-into-mathlib)) are `CategoryTheory.MonoidalCategory` / `SymmetricCategory` over `FreeMonoidalCategory (Discrete O)`. The Grothendieck split of [§7.1](#71-the-structuredata-split-as-dat) is `CategoryTheory.Grothendieck` applied to the data functor `Dat'`. The composition-as-pushout of [§6](#6-mixins-scan-route-symmetry-para)/[§7.3](#73-composition-as-pushout) and the associativity of [§8.5](#9-the-propositions-as-generic-theorems) are `CategoryTheory.Limits.pushout` plus the pasting lemma. The `Algebra` functor `F : C ⥤ V` and its morphisms are `MonoidalFunctor` / `MonoidalNatTrans`. The gated equivariance of [§6.2](#62-route-and-symmetry-stubs)/[§8.4](#9-the-propositions-as-generic-theorems) is `CategoryTheory.Action` / `Rep` / `Monad.Algebra` (the Eilenberg–Moore category of the symmetry monad). And the architecture relations `R` quotienting `C♯` ([§7](#7-grothendieck-split-and-composition-as-pushout)) are `CategoryTheory.Quotient`. The `D`-actegory coherence bundle of the core — the triangle/pentagon and the distributivity isos — is the one piece Mathlib carries only partially (`Action` covers a monoid acting, not a full monoidal-category actegory), so it is hand-rolled as functor-plus-natural-iso algebra; it is routine, not deep.

Beyond the coverage map, several honest notes shape any transcription:

- **Strictification strategy.** Develop the whole tower over `FreeMonoidalCategory (Discrete O)` and **strictify once**, so that downstream all associators and unitors are `Iso.refl` and the PROP equations (`tensor_assoc`, `tensor_unit_l`, `tensor_unit_r`) hold *definitionally* rather than up to coherent isomorphism. This is what lets a `ColoredPROP` law stated with `=` line up with a Mathlib `MonoidalCategory` whose coherences are isos, and it is applied at the seam adapter of [§3](#3-the-seam-adapter-into-mathlib).

- **Strictness is the real friction.** This strictification is the *single* genuine difficulty in the formalization. Mathlib's categories are not strict, so without the one-time strictification every PROP equation would have to be threaded through associator/unitor coherence by hand. The mitigation above contains the cost to one place; nothing else in the development fights the type theory.

- **Inheritance is for theorems, not obligations.** The [§9](#9-the-propositions-as-generic-theorems) theorems transfer to every instance for free, because their only hypothesis is `[DGradedColoredPROP D C]`. But each new `instance` must still **discharge the coherence `Prop`-fields** of the core — the actegory triangle and pentagon (`act_unit_assoc`), the distributivity coherences (`dist_coh`), and `sh_act`/`broadcast_gen`. "Prove once, inherit everywhere" names the proven propositions; it does not waive the per-instance obligation to *supply* the coherence fields.

- **Instance-resolution discipline.** `D` and `C` are **explicit** class parameters, not `outParam`s. With both free, Lean's instance search needs them pinned, so `[DGradedColoredPROP D C]` resolves predictably and `Br`-as-graded (the `C` slot) never collides with `Br`-as-index (the `D` slot) in search — the non-collision relied on by the MoE level of [§10.2](#102-the-additive-extension-menu).

- **Mixins, not a tall tower.** Scan, Route, Symmetry, and Para are kept as composable mixin classes layered on the core ([§6](#6-mixins-scan-route-symmetry-para)), rather than as one deep `extends` chain. A tall tower would invite typeclass *diamonds* (a capability reachable by two `extends` paths) and degrade instance-search performance; independent mixins let each instantiation pay only for the capabilities it declares.

- **The Para 2-category gap.** The `Para(V)` encoding is kept **lightweight and 1-categorical**: a parameterized 1-cell is `Σ (P : V), (P ⊗ A ⟶ B)`, and the "2-cells" (reparameterizations, weight tying) are carried as an explicit reparameterization *relation* rather than as genuine 2-morphisms of a bicategory. The gap to a full `Para` bicategory is **noted, not paid for** — a bicimplementation would be machinery only the specification uses ([graded_prop.md §7](graded_prop.md#7-algebras-construct-and-the-para-refinement)).

- **Equivariance is gated.** Proposition 8.4's body depends on the `SymmetryGraded` mixin and the Eilenberg–Moore-category machinery for the symmetry monad `T`. The finite-group case is reachable with present Mathlib (`Action`/`Rep`), but the graded-PROP-dependent parts of the encoding wait on this very formalization being in place; the proposition is *stated* now and its proof *gated* ([equivariance_unification.md](equivariance_unification.md)).

## 12. Appendix: out of scope

Two families of structure are deliberately **not encoded**, because they carry no propositional or computational content the framework reasons about. The first is **`DynamicName` and its LaTeX rendering** — the human-readable, mathematically-typeset names attached to axes and arrays. The second is the **`Block` display metadata** — the layout and presentation bookkeeping the visualizer consumes. Both are *semantically transparent*: erasing them changes no morphism, no shape, no composite, and no proof. They ride on the executable side of the seam as identity/display decoration (the `WithUID` decoration of [§7.4](#74-the-seam-concrete-union-find-realizes-the-coequalizer) carries the optional `DynamicName`), and they are left exactly where the current document already leaves `Block` — outside the encoding, mentioned but never formalized.

This matches the document's overall stance, restated once here: it writes **no proved Lean**. Every `class`, `structure`, `def`, and `theorem` above is a *signature* with named `Prop`-field laws and named proof obligations; the elisions (`…`) in bodies are intentional, marking exactly the obligations a future Lean development would discharge. The deliverable is **formalizability, not formalization** — the shape into which the propositions of [graded_prop.md §8](graded_prop.md#8-propositions-the-synthesis-organizes) transcribe directly, not the discharged proofs themselves.

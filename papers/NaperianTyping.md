# Naperian Typing for Tensor Logic DSL in a `D`-Graded Colored PROP

This note explains how to integrate Naperian/applicative array semantics into the existing LeanNCD tensor-logic pipeline and `D`-graded colored PROP formalism, with minimal disruption to current code paths.

It is meant as a practical bridge between:

- [`papers/graded_prop.md`](../papers/graded_prop.md) (categorical core),
- [`docs/tensor_logic_dsl.md`](../docs/tensor_logic_dsl.md) (user-facing DSL),
- [`leanncd/LeanNCD/DSL/Pipeline/*.lean`](../leanncd/LeanNCD/DSL/Pipeline/*.lean) (actual passes).

## Contents

1. [Motivation and alignment](#1-motivation-and-alignment)
2. [Semantic foundation: Naperian axes in a D-graded PROP](#2-semantic-foundation-naperian-axes-in-a-d-graded-prop)
   - [Gibbons' APLicative/Naperian results](#gibbons-aplictativenaperian-results)
   - [D-graded colored PROP substrate (what we already have)](#d-graded-colored-prop-substrate-what-we-already-have)
   - [Proposed mixin semantics (incremental, backward-compatible)](#proposed-mixin-semantics-incremental-backward-compatible)
   - [Intuition behind why this works](#intuition-behind-why-this-works)
   - [Categorical Integration: Naperian Structure in D-Graded PROPs](#categorical-integration-naperian-structure-in-d-graded-props)
3. [Compiler pipeline as law enforcement](#3-compiler-pipeline-as-law-enforcement)
4. [Optimization and rewrite opportunities](#4-optimization-and-rewrite-opportunities)
5. [Lean implementation strategy with dependent-type optimization](#5-lean-implementation-strategy-with-dependent-type-optimization)
   - [§5.5 Integration implementation plan](./NaperianTypingIntegrationPlan.md) *(separate document)*
6. [Payoff: benefits for users, maintainers, and formalization](#6-payoff-benefits-for-users-maintainers-and-formalization)
7. [References](#7-references)

---

## 1. Motivation and alignment

### Why this integration exists

The current pipeline already has strong structure:

- name/UID axis unification (`assignUIDs`, `unifyAxes`),
- affine index reindexing (`lowerArith`, `route` with `StMatP`),
- implicit contraction/reduction (contracted RHS-only axes),
- nonlinearity isolation (`splitNonlins`),
- temporal scan extraction (`finalizeScans`, `evalScan`),
- routed executable graph (`ThreadedComposed`).

What is missing is a single typed semantic layer that states why these transformations are correct and compositional. The proposal here is to add that layer by enriching the existing `DGradedColoredPROP` view with Naperian/applicative mixins.

### Tensor Logic semantic alignment

This integration is especially natural for Tensor Logic because Tensor Logic already treats equations as indexed contractions over named axes:

- **Axis identity by name/UID** (Tensor Logic) corresponds to **object identity in `D`**.
- **RHS-only axes contract** (Tensor Logic) corresponds to **`AxisReduce`**.
- **Affine index arithmetic** (`i+p`, `2*j+r`) corresponds to **`ReindexAction`** via affine `St` morphisms.
- **Softmax/normalize over a marked axis** corresponds to **typed reduction + pointwise lift discipline**.
- **Recurrence over `l`** corresponds to **`TemporalGraded` scan structure**.

So this is not an imposed abstraction; it formalizes the semantics Tensor Logic already uses operationally.

### Scope and rollout

This note explains how to integrate Naperian/applicative array semantics into the existing LeanNCD tensor-logic pipeline and `D`-graded colored PROP formalism, with minimal disruption to current code paths. It is meant as a practical bridge between:

- [`papers/graded_prop.md`](../papers/graded_prop.md) (categorical core),
- [`docs/tensor_logic_dsl.md`](../docs/tensor_logic_dsl.md) (user-facing DSL),
- [`leanncd/LeanNCD/DSL/Pipeline/*.lean`](../leanncd/LeanNCD/DSL/Pipeline/*.lean) (actual passes).

---

## 2. Semantic foundation: Naperian axes in a D-graded PROP

### Background: Applicative Functors and Lifting

Before diving into Naperian semantics, we briefly recap **applicative functors** and **applicative lifting**, as these are the foundational operations that Naperian functors formalize for shape-indexed types.

#### What is an Applicative Functor?

An **applicative functor** is an algebraic structure with two key operations:

1. **`pure : a → F a`** — wraps a value into the functor context, without any structure.
2. **`<*> : F (a → b) → F a → F b`** — applies a wrapped function to wrapped values.

These must satisfy laws (identity, composition, homomorphism, interchange) that ensure well-behaved composition.

**Intuition for arrays and tensors:**
- `pure x` broadcasts `x` to match any shape (replicate it everywhere).
- `f <*> xs` applies the function `f` pointwise to each element of `xs`.
- Combining: `liftA2 op xs ys` applies `op` pointwise: each element of the result is `op(xs[i], ys[i])` for aligned indices.

#### Applicative Lifting

**Applicative lifting** is the systematic way to lift an ordinary function into the applicative context:

- **`liftA : (a → b) → (F a → F b)`** lifts a unary function.
- **`liftA2 : (a → b → c) → (F a → F b → F c)`** lifts a binary function.
- And so on for arity.

**Concrete example:** Binary addition in the applicative context of lists:
```
liftA2 (+) [1, 2, 3] [10, 20, 30]  →  [11, 22, 33]
```
Each element of the output is the sum of corresponding elements in the inputs.

**Connection to broadcasting:** This is exactly what tensor broadcasting does:
- `pure 5` broadcasts the scalar `5` to every element of a shape.
- `liftA2 (+) x y` applies `+` element-wise, automatically aligning shapes.

#### Representable Functors (Naperian Functors)

A representable (or **Naperian**) functor is an applicative functor with an additional structure: **total and invertible indexing**.

For a representable functor `f`:
$$f \, a \cong (\text{Log} f \to a)$$

This isomorphism says: "A value in `f a` is completely determined by its values at each index."

**Concrete meaning:**
- `lookup : f a → Log f → a` retrieves a value at a specific index.
- `tabulate : (Log f → a) → f a` reconstructs `f a` from a function on indices.
- These are inverses: `tabulate (lookup fa) = fa` and `lookup (tabulate f) = f`.

*Note on terminology:* `Log f` (the "logarithm" of the functor) is the index type (same as the functor's **shape** or **dimensions**). The name comes from the exponential analogy: if a container `f a` is isomorphic to a function `Log f → a`, then `Log f` plays the role of the "exponent." Concretely: `Log(Maybe) = Bool`, `Log(Pair) = Bool`, `Log(Array n) = Fin n` (integers 0..n-1), and for a 2D array of shape `(m, n)`, we have `Log = Fin m × Fin n`.

**For arrays:** A 2D array of shape `(m, n)` is isomorphic to a function from pairs `(i, j) ∈ [0..m) × [0..n)` to values. Indexing is retrieval; tabulation is construction.

This representability is the key difference from general applicatives: it makes **indexing a type-level property**, not a runtime convention. Shape mismatches become compile-time errors.

### Gibbons' APLicative/Naperian results

Jeremy Gibbons' paper (see [References](#7-references)) gives a typed account of APL-style rank polymorphism via fixed-shape representable functors:

1. **Arrays as nested dimensions.**  
   An `n`-dimensional array is modeled as nested functors (shape at the type level).

2. **Applicative lifting = pointwise semantics.**  
   `pure` gives replication/broadcasting; `<*>`/`zipWith` gives pointwise lifted operators.

3. **Naperian (representable) dimensions = indexability.**  
   `f a ≅ (Log f → a)` with `lookup/tabulate`; indexing is total and invertible.
   
   *Understanding the isomorphism:* The phrase "with `lookup/tabulate`" means these two functions **witness** the isomorphism:
   - **`lookup : f a → Log f → a`** — extract a value at an index
   - **`tabulate : (Log f → a) → f a`** — build a container from a function on indices
   
   These are **inverses** of each other (bijection): `lookup (tabulate f) = f` and `tabulate (λi. lookup fa i) = fa`.
   
   **"Indexing is total and invertible"** means:
   - **Total**: You can look up any element of `Log f` without fear of out-of-bounds; the index type is exhaustive.
   - **Invertible**: If you know the value at every index, you can uniquely reconstruct the original container. There is no hidden state.


4. **Transpose is structural reindexing.**  
   `transpose :: f (g a) -> g (f a)` comes from representability, not ad hoc tensor code.

5. **Reductions/scans are typed over traversable structure.**  
   Folds/scans become first-class and shape-aware, not runtime conventions.

The import for us: these are exactly the semantic ingredients behind broadcasting, reindexing, contraction, and scan in tensor logic.

### D-graded colored PROP substrate (what we already have)

From [`papers/graded_prop.md`](../papers/graded_prop.md), we already have:

- index PROP `D` (for pyncd: typically `St`),
- operation PROP `C` (for pyncd: `Br`-like operational layer),
- shape map `sh`,
- right action/lift `act : C × Dᵒᵖ ⥤ C`,
- coherence (`δ`, `δ0`, `υ`, `α`): the natural isomorphisms ensuring the action is well-behaved — `δ` is distributivity `(X ⊗ Y) ⊛ P ≅ (X ⊛ P) ⊗ (Y ⊛ P)`, `δ0` is unit action, `υ` is left/right unit laws, and `α` is associativity of nested actions,
- evaluation slices `ev_p`,
- broadcast-generation/weave factorization story.

### Proposed mixin semantics (incremental, backward-compatible)

Keep `DGradedColoredPROP` as the base. Add semantic mixins as capabilities that reinterpret existing passes as enforcing/applying these laws, with Naperian semantics for axis points and applicative pointwise lift.

**What are these mixins?** Each mixin below is a **semantic law** or **capability** that describes how one or more existing compiler passes should behave. These are not new operations; rather, they are abstractions that capture the principles behind existing transformations. A compiler morphism (like `assignUIDs`, `lowerArith`, `splitNonlins`) may satisfy one or more of these mixins. By making these laws explicit at the type level, we gain:
- **Type safety**: shape and reindexing errors caught at compile time.
- **Compositional understanding**: each pass enforces one or more mixin laws, making their roles clearer.
- **Formal verification**: mixins are propositions that can be proved or tested.

The mixins are:

- **`NaperianAxis`**: finite point/index semantics on `D`-axes (enumerable coordinates).
- **`BroadcastJoin`**: canonical common degree for pointwise binary composition.
- **`ReindexAction`**: contravariant action of `D` morphisms on lifted objects.
- **`AxisReduce`**: typed elimination/fold along contracted axes.
- **`PointwiseLift`**: applicative lifting for unary/binary pointwise operations.
- **`TemporalGraded`** (already present): prefix-indexed scan semantics and lift-fold law.

Interpretation: these are laws on existing compiler behavior, not a mandatory user syntax change.

### Intuition behind why this works

Before diving into categorical formalism, it is worth stepping back and understanding the core intuition: **representability moves proof obligations from propositions into type constraints**, making properties that would otherwise require separate proofs into automatic consequences of the type structure itself.

#### The problem: scattered proof obligations

In the current pipeline, each morphism and composition is built, then its correctness is *asserted and proved separately*. For example, consider a matrix morphism `m : StMatP dom cod` mapping a domain `dom` to a codomain `cod`:

```lean
-- Current approach: build then prove correctness
def matMul (A : StMatP dom mid) (B : StMatP mid cod) : StMatP dom cod := 
  { data := compute_product A.data B.data
    h_correct_dims := sorry  -- ← NEW PROOF OBLIGATION
    -- Many more sorries for shape compatibility, axis ordering, etc.
  }
```

The type signature tells you the *intent* (domain dom, codomain cod), but the *guarantees* (that the output has the right shape, that intermediate indices match, etc.) are proved separately. This creates a maintenance burden: every new composition introduces new proof obligations.

#### The solution: encode constraints in types

Representability says: **make the shape part of the type signature itself**, not a property you prove about it. Instead of storing raw data and proving properties hold, use an indexed function view:

```lean
-- Naperian approach: shape is in the type
structure StMatP (dom cod : StObj) where
  lookup : dom.point → cod.point → Float

def matMul {dom mid cod : StObj}
  (A : StMatP dom mid) (B : StMatP mid cod) : StMatP dom cod :=
  { lookup := fun i k => ∑ j, A.lookup i j * B.lookup j k }
  -- Type checks automatically
  -- No separate proof of shape correctness needed
```

In the `matMul` example above, `mid` is an intermediate shape (a type parameter of the function). The type signature `A : StMatP dom mid` already **encodes** that matrix `A` maps domain indices (`dom.point`) to intermediate indices (`mid.point`). Similarly, `B : StMatP mid cod` encodes that `B` maps from intermediate to codomain indices. When we compose them in the function body, the sum `∑ j, A.lookup i j * B.lookup j k` *must* iterate over `j : mid.point` (the intermediate indices), and this constraint is enforced by the type system—you cannot use any other index type.

**The proof obligation vanishes because the type constructor itself is the constraint.** We cannot construct a matrix with the wrong dimensionality or compose matrices with mismatched intermediate shapes because the type forbids it.

#### Why representability enables this

Representability (in Gibbons' formulation) states that certain functors are isomorphic to indexed-function spaces:

$$\text{Representable } F \iff F(X) \cong (\text{Log } F \to X)$$

Here:
- `Log F` is the **index type** (the "logarithm" or point type of the functor)
- The right-hand side is **functions from indices to values**
- `lookup` and `tabulate` witness the isomorphism

In your context:
- `Log(StMatP dom cod)` is the product `dom.point × cod.point` (the set of all coordinates)
- `lookup` extracts a single element at coordinates `(i, j)`
- `tabulate` rebuilds from a function `(i, j) ↦ value(i, j)`

Because we work with the **indexed-function view** (not raw arrays), composition becomes function composition:

```lean
def ap {A B : Type} (f : A → (B → C)) (x : A → B) : A → C :=
  fun a => f a (x a)
  -- This is definitional: just function application
  -- No proof needed
```

Applicative laws like `pure id <*> v = v` are not theorems—they are **definitional equalities** of function application. The laws hold automatically because we *defined* the operation in the right way.

#### The proof-obligation cascade

This extends to axioms about shapes and compositions:

| Without Representability | With Representability | Why it works |
|---|---|---|
| Build matrix + prove dimensions match | Type signature enforces dimensions | Type constructor is the constraint |
| Prove `tabulate (lookup m) = m` | Isomorphism is definitional | The type is the lookup function |
| Prove binary operations commute (broadcast) | Use applicative `<*>` | `<*>` is *defined* as pointwise application |
| Prove composition is associative | Function composition associativity | Function composition *is* syntactically associative |
| Prove causality (no future reads in scans) | Encode in dependent type | Type forbids invalid state transitions |

Each "new" morphism no longer introduces *new* proof obligations. Instead, we instantiate **universal laws** that come from representability and applicative structure. These laws are *once-and-for-all proofs* about representable functors and applicative operations—they do not depend on problem-specific details. For concrete counts in this codebase, see [Measurable proof-effort reduction](#measurable-proof-effort-reduction).

---

### Categorical Integration: Naperian Structure in D-Graded PROPs

The mixins above are not decorative; they follow from a deep categorical principle. This subsection explains how Naperian typing refines the `D`-graded colored PROP formalism.

#### The point functor and representability

The heart of Naperian semantics is **representability** — a fundamental concept in category theory (see [References](#7-references)) where a functor is isomorphic to a hom-functor. In our setting, the isomorphism is:

$$F(X ⊛ P) \cong \text{El}(P) \to F(X)$$

(*Intuition:* a P-shaped container of X-values is completely determined by specifying, for each point (coordinate) in P, what the corresponding X-value is.)

Here we introduce the key notation:
- **⊛** is the **graded action** functor (from [graded_prop.md §3](#references)): the right action of objects in `D` on objects in `C`. This is the core structure of D-graded PROPs.
- **$\text{El}(P) := D(I_D, P)$** denotes the **points** (or **global elements**) of a shape object `P` in the index category `D`. Here `D(I_D, P)` is the **hom-set** — the set of all morphisms `I_D → P` in `D`, where `I_D` is the monoidal unit of `D` (written `I_D` throughout [graded_prop.md](#references); it is the unit for the `⊛` action, satisfying `X ⊛ I_D ≅ X`). Each morphism `I_D → P` picks out a single "coordinate" or "element" of `P`, hence the name. 
  
  In practice: In the concrete `StObj` setting, the full hom-set `D(I_D, P)` consists of all affine-stride morphisms `StMat [] P`. The `NaperianAxis StObj` instance uses a finite chosen representative set via `AxisPointData`, with `point_hom` providing the embedding. The categorical theory is stated for the full hom-set; the Lean implementation works with a finite representative (see [Connection to elementality](#connection-to-elementality) below for how this relates to elementality and separating families).
- **$F$** is a **semantic algebra**: a functor that interprets objects of `C` into concrete data structures (e.g., arrays, functions, or numerical values).
- **$\cong$** denotes a natural isomorphism: a natural transformation whose components are all isomorphisms. For this formula specifically, the isomorphism is natural in `X` (covariantly — the isomorphism commutes with morphisms `X → X'` in `C`) and natural in `P` (contravariantly) via `mapEl η : El(P) → El(Q)` for `η : P → Q` in `D` — a field of `NaperianAxis` satisfying functoriality and naturality with `point_hom`.

This representability is not coincidental. Following [Gibbons 2016](#7-references), for a **point functor** 
$$\text{El} : D \to \mathsf{FinSet}, \quad P \mapsto D(I_D, P)$$
with the **strong monoidal property** $\text{El}(P \otimes Q) \cong \text{El}(P) \times \text{El}(Q)$ (where $\otimes$ is the monoidal product in `D`), the point functor has the right product behavior for Naperian indexing. The equivalences `El(P ⊗ Q) ≃ El P × El Q` must satisfy standard monoidal coherence laws (associativity, left/right unit) to constitute a genuine strong monoidal functor; these are carried as explicit fields of `NaperianAxis`.

**Important caveat:** Strong monoidality of `El` is necessary but not sufficient for the isomorphism `F(X ⊛ P) ≅ (El(P) → F(X))`. The representability assumption — that the lifted functor `act(−, P)` is represented by `El P` — must be supplied explicitly via the `NaperianFamily` typeclass, which packages `lookup`/`tabulate` and their inverse laws. Strong monoidality ensures the point functor respects monoidal structure, making coherence laws like `El(P ⊗ Q) ≃ El(P) × El(Q)` well-behaved, but does not produce the isomorphism itself.

**Connection to elementality:** When the representability assumption holds, the evaluation transformations `ev_p := act(−, p)` (for $p \in \text{El}(P)$) form a jointly separating family for lifted `P`-indexed values. This Naperian joint monicity—where **lifted P-indexed families are determined by their values at each coordinate**—is a **different layer** from the **(Elem-C)** axiom of [graded_prop.md](#references). The (Elem-C) axiom says that **base C-morphisms are separated by C-points** (i.e., base operations are determined by their pointwise behavior). Naperian is a *separate assumption* about the **lifted** structure: it says that when we lift morphisms by a shape P, the resulting P-shaped families are determined by their coordinates. You can have (Elem-C) without Naperian representability; they are not derived from each other. Naperian typing makes the lifted dense family explicit in the type system by:
1. **Explicitly enumerating points**: The point functor `El` is finite and enumerable (for `D = St`, these are coordinate tuples).
2. **Typeclass-level monoidality**: Strong monoidality, with coherence laws, ensures the point functor respects the monoidal structure of `D`.
3. **Representability as first-class semantics**: Rather than deriving representability from `NaperianAxis`, Naperian typing records it in `NaperianFamily`, making `lookup`/`tabulate` laws explicit.

Additionally:
- **`lookup_p : X ⊛ P → X`** corresponds to the **evaluation transformation** $\text{ev}_p$ from [graded_prop.md §3](#references): for each point $p \in \text{El}(P)$, evaluating a `P`-indexed family at `p` yields a value in the base type. This evaluation is exactly the `ev_p` slice used in [graded_prop.md (line 97)](#references).
- **`tabulate`** (the inverse) is the **universal property** of representability: it reconstructs a `P`-indexed family from its pointwise evaluations. Because the `ev_p` are jointly separating under `naperian_jointly_monic`, this reconstruction is **unique** — a key fact used in the weave-uniqueness result ([graded_prop.md §3.3, Prop 8.2](#references)).

#### Fiber semantics and reindexing

From [graded_prop.md §3](#references), objects of `C` are **fibered over shapes in `D`** via the **shape map** 
$$\text{sh} : O_C \to \text{Ob}(D).$$
(Read: every object in the operation PROP `C` is assigned a shape/index object in `D`.) This is one of the core axioms of D-graded PROPs.

For a morphism $\eta : P \to Q$ in `D`, the existing D-graded PROP structure defines a **contravariant action** (§3, [graded_prop.md](#references))
$$[X, \eta] : X ⊛ Q \to X ⊛ P.$$
(Read: "applying `η` to `X` at shape `Q` produces a morphism to shape `P`".)

Under the Naperian isomorphism, this contravariant action is exactly **precomposition of families**:
$$(\text{El}(Q) \to A) \xrightarrow{\eta^*} (\text{El}(P) \to A)$$
where $\eta^* : \text{El}(P) \to \text{El}(Q)$ is the **induced map on points**, implemented as `mapEl η` and satisfying `point_hom p ≫ η = point_hom (mapEl η p)`. This exhibits **contravariance**: pulling back along $\eta : P \to Q$ reverses the direction. For `StObj`, `mapEl η p` applies the affine stride map `η : StMat P Q` to coordinates, so the concrete instance must ensure `toCoeff` values are valid coordinates.

This interpretation clarifies the existing structure:

- **`ReindexAction`** is not new algebra; it is the **contravariance of the action `act`** (from [graded_prop.md](#references)) made explicit via representability. Every reindexing is a precomposition in the index category `D`.
- **Affine index arithmetic** (e.g., $i \mapsto 2i+1$) is a **restricted class of `D`-morphisms**, particularly when `D = St` (the spatial index PROP). With `mapEl` and the relevant joint-monicity/extensionality theorem, reindexings can be classified by their action on points.

#### Applicative lifting and distributivity

The strong monoidal action gives the distributivity isomorphism
$$\delta_{X,Y,P} : (X \otimes Y) ⊛ P \cong (X ⊛ P) \otimes (Y ⊛ P)$$
This is exactly the applicative zipping law:
$$(\text{El}(P) \to X \times Y) \cong (\text{El}(P) \to X) \times (\text{El}(P) \to Y)$$

Consequently:

- **`PointwiseLift`** and `liftA2 : (a \to b \to c) \to (\text{El}(P) \to a) \to (\text{El}(P) \to b) \to (\text{El}(P) \to c)$ are the same construct, made categorical.
- **`BroadcastJoin`** constructs a canonical common shape in `D`, with coordinate projections back to each operand, before applying `δ`.
- Binary operations in the broadcastable fragment factor as: pointwise lifting of base ops, then reindexing to common shape.
- For **`α`**, the action order matters: `(X ⊛ P) ⊛ Q ≅ X ⊛ (Q ⊗ P)`, so the `El`-side uses `El(Q ⊗ P) ≃ El Q × El P`. For `StObj`, use `alphaElEquiv P Q := appendEquiv Q P`, not `appendEquiv P Q`.

#### Temporal grading as directed folding

The `TemporalGraded` mixin adds **causality**: output at time `t` depends only on times `≤ t`. This is a special case of the `D`-graded structure when the index category `D` is enriched with **directed structure** (see [See References](#7-references) in category theory literature). Categorically:

1. The **temporal axis** `L` is a **directed index object** in `D`, typically isomorphic to the finite ordinal $[0..N]$ with a **prefix order** (or **interval-order morphisms** in standard terminology). In [graded_prop.md](#references), this corresponds to a distinguished temporal color in the index PROP.

2. A **scan** is not a pointwise operation along `L`; it is a **finite fold** / **catamorphism** over the directed structure. Formally (from [See References](#7-references)), it is a morphism out of the initial algebra of the prefix-indexed functor, encoding stepwise iteration with feedback.

3. The **lift-fold distributivity law** states
   $$[\text{Scan}, P] \cong \text{Scan}([\text{step}, P])$$
   for `P` orthogonal to the temporal axis `L`. This is a **Beck-Chevalley / Fubini law** (see [graded_prop.md §3](#references)) on the **grading fibration** $C \to D$: the action and folding commute when dimensions are independent. This is not a coincidence but a consequence of the categorical structure.

The constraint that **no morphism can depend on future indices** is formalized as: only **monotone / prefix-respecting `D`-morphisms** are allowed along temporal axes. (A prefix-respecting morphism maps $[0..s]$ into $[0..s']$ for any `s`, preserving causality.) This causality constraint **cannot be expressed purely pointwise**; it requires the full directed structure of the grading fibration.

#### Broadcast generation as applicative normal form

The **broadcast-generation axiom** from [graded_prop.md §3](#references) states that every morphism in the **broadcastable fragment** of `C` factors as:
$$g = [f_{\text{base}}, P] ; \rho$$
where $\rho$ is a **reindexing alignment** (a sequence of precompositions in `D`) and $[f_{\text{base}}, P]$ is **pointwise lifting** of a base morphism $f_{\text{base}}$. This factorization is unique up to equivalence in the broadcastable fragment.

Categorically, this says:

> The **broadcastable fragment** is the **free applicative / reindexing completion** of the base fragment.

This is a **normal-form theorem** (following [Plotkin & Power 2003](#references) on algebraic computational effects and normal forms): morphisms are **determined by their pointwise action on evaluations**. The points $p \in \text{El}(P)$ and evaluation transformations $\text{ev}_p$ form a **dense family** (in the sense of [MacLane & Moerdijk](#references), a separating family in category-theoretic terms), so two morphisms that agree on all pointwise evaluations are equal once `naperian_jointly_monic` is available. This relies on representability or a broadcast-generation normal form; it does not follow from strong monoidality of `El` alone.

#### Reformulated view: Naperian D-graded PROPs

A cleaner categorical statement integrates all the pieces:

> A **Naperian D-graded colored PROP** is a D-graded colored PROP (as defined in [graded_prop.md](#references)) equipped with:
> 1. A **finite, strong-monoidal point functor** $\text{El} : D \to \mathsf{FinSet}$ with $\text{El}(I_D) = \{*\}$. (The point functor must satisfy $\text{El}(P \otimes Q) \cong \text{El}(P) \times \text{El}(Q)$, with associativity and unit coherence.)
> 2. A separate **representability assumption** for lifted families: `NaperianFamily.lookup`/`tabulate` witness `F(X ⊛ P) ≃ (El P → F X)`. This is not derived from `NaperianAxis` or strong monoidality.
> 3. **Jointly separating lifted evaluations** (`naperian_jointly_monic`): For lifted objects `X ⊛ P`, the evaluation transformations $\text{ev}_p$ (for $p \in \text{El}(P)$) form a separating family. This is not the same as full **(Elem-C)** from [graded_prop.md (Def 3.2)](#references), which separates arbitrary `C`-morphisms by `C`-points.
> 4. **Reindexings by precomposition**: For $\eta : P \to Q$ in `D`, reindexing $[X, \eta]$ is defined as postcomposition by the induced map `mapEl η : El(P) → El(Q)` on points, i.e., $[X, \eta](x)(p) = x(\text{mapEl}(\eta)(p))$. This is the contravariance of the action `act` from [graded_prop.md (Def 3.1.2)](#references), now indexed explicitly by elements.
> 5. **Reduction/fold as Kan extension**: Contractions are not pointwise; they are computed as **left Kan extensions** (or catamorphisms, for temporal structure) along index projections. This gives the Beck-Chevalley proof target and makes the lift-fold distributivity from [graded_prop.md](#references) a theorem of category theory once the required Kan-extension structure is supplied.

Under this definition:

- `NaperianAxis` is the **data of a finite index object** with **explicit point enumeration** (realized by the point functor `El`) and coherent maps on points.
- `BroadcastJoin`, `ReindexAction`, and `PointwiseLift` require their own data/laws or concrete derivations from the chosen action and alignment strategy; they are not consequences of strong monoidality alone.
- `AxisReduce` encodes **left Kan extensions**; `TemporalGraded` adds **directed temporal structure** with monotone-morphism restriction (a constraint on allowed reindexings).
- The **compiler pipeline becomes an enforcer** of applicative/representability laws: each pass verifies that operations respect strong monoidality, representability, and the Naperian separating family.

#### Benefits of this categorical clarity

1. **Soundness is structured** (from [See References](#7-references)): If mixins respect strong monoidality, its coherence laws, and the separate representability assumptions, then **coherence isomorphisms** ($\delta, \delta_0, \upsilon, \alpha$) from [See References](#7-references) have standard categorical proof targets. Constructor-level quotient and extensionality proofs are still required in the Lean implementation.

2. **Proof obligations shift** (from direct application to categorical library): Rather than proving individual lemmas about reindexing or pointwise lifting, Lean implementations **inherit theorems from the categorical library** (e.g., [See References](#7-references) for monoidal functor properties). This significantly reduces the sorry-count.

3. **Extension is systematic** (from functor extension principles): Adding a new type of graded operation (e.g., grouped reductions, higher-rank liftings) becomes an exercise in **extending the point functor or index category**, not ad-hoc axiom invention. The categorical framework constrains the design space and ensures consistency.

4. **Dependent types align with structure** (from [graded_prop.md](#references)): Type-level ranks (as in `StMatP dom cod`) **directly encode the fiber structure** of the grading; invariants become **unrepresentable violations** (rejected by the type system), not proof obligations. This is a consequence of internalizing the contravariant action into dependent types.

---

## 3. Compiler pipeline as law enforcement

### 3.1 Structural phases (`Structural.lean`)

| Current pass | Current behavior | PROP/Naperian meaning | Key invariant to enforce |
|---|---|---|---|
| `assignUIDs` | one UID per axis name | naming-level quotient seed | equal names imply same axis identity |
| `resolveDecls` | build `DeclEnv`; classify externals | separates declared structural objects from external generators | undeclared reads become typed external inputs |
| `checkReadRanks` | read arity checks | object-shape arity consistency in `D` | every read rank matches declared/inferred tensor rank |
| `checkDtypes` | axis kind/nonlin guards | sort discipline on colors and generators | predicate outputs remain identity/sum-only |
| `unifyAxes` | coequalize UIDs by name | canonicalization in the shape fibration | canonical representative per name; substitution total |
| `lowerArith` | affine LHS becomes scatter; overlap checks | detects non-invertible reindex writes | non-injective scatter requires explicit reducing behavior |
| `finalizeScans` | group base/recur by iteration axis; causality checks | `TemporalGraded`: temporal object, prefix discipline, recurrence typing | base/recur pairing, no look-ahead, single temporal UID per scan group |

### 3.2 Lowering/scheduling/routing (`Lowering.lean`)

| Current pass | Current behavior | PROP/Naperian meaning | Key invariant to enforce |
|---|---|---|---|
| `splitNonlins` | isolate nonlinearities into separate stmts | `PointwiseLift` separation from multilinear contraction | nonlinearity step preserves degree; contraction step is pure contraction |
| `schedule` | backward liveness DCE | preserves semantics in free SMC composition | only live morphisms remain; source ordering remains valid |
| `route` | build `ThreadedComposed`/`BrBaseP`; compute `degree`, `weaves`, `reindexings` | concrete realization of `BroadcastJoin` + `ReindexAction` + weave factorization | `degree` canonicalized (`lhs ++ contracted`), one affine `StMatP` per read, routing wires type-consistent |

### 3.3 Evaluation (`Eval/*.lean`)

| Current evaluator behavior | PROP/Naperian meaning | Key invariant |
|---|---|---|
| coordinate loops over shapes | Naperian point enumeration | axes reduced/evaluated must be finite at runtime |
| affine gathers via reindex matrices | reindex pullback action | out-of-bounds policy explicit and consistent |
| contraction over RHS-only axes | `AxisReduce` fold | contracted axis set exactly matches compile-time derivation |
| relu/softmax/normalize post-contraction | `PointwiseLift` | norm axis is well-defined and present |
| `evalScan` recurrence execution | `TemporalGraded` operational law | causal update (`l → l+1`) and state-shape stability |

### 3.4 Compile-time checks to centralize and add

1. **Naperian availability check**  
   Any axis used for reduction/scan/eager evaluation must have a finite concrete size by evaluation.

2. **Broadcast-join coherence**  
   Degree construction is canonical and stable under equivalent expression reorderings.

3. **Reindex soundness**  
   Every `IdxExpr` lowers to exactly one affine row over canonical degree axes; codomain rank matches source tensor rank.

4. **Scatter injectivity policy**  
   Non-injective writes require explicit reduction (currently conservative overlap checks).

5. **Reduction legality**  
   Aggregation choice is valid for output dtype and axis kind; norm-axis markers are coherent.

6. **Temporal safety**  
   Base/recur alignment, no future reads, single-scan-axis coupling unless explicitly supported.

1. **Reindex fusion**  
   `reindex η₂ (reindex η₁ x)  =>  reindex (η₁ ∘ η₂) x`.

2. **Pointwise-reindex commutation**  
   For true pointwise ops: `map f (reindex η x)  =>  reindex η (map f x)`.

3. **Lift associativity normalization**  
   `lift P (lift Q f)  =>  lift (Q ⊗ P) f`.

4. **Broadcast normalization**  
   normalize equivalent `degree` constructions to one canonical `BroadcastJoin` normal form.

5. **Reduction simplifications**  
   singleton/empty-axis fold eliminations where shape proves trivial contraction.

6. **Scan batching distributivity**  
   for batch axis independent of temporal axis: `lift P (scan step)  ≅  scan (lift P step)`.

7. **Softmax constant-drop (existing DSL behavior, formalized)**  
   constants independent of norm axis may be removed before softmax.

These should start as verifier-preserving rewrites, then become optimization passes.

---

## 4. Optimization and rewrite opportunities

### 4.1 Law-backed rewrites and optimizations

1. **Reindex fusion**  
   `reindex η₂ (reindex η₁ x)  =>  reindex (η₁ ∘ η₂) x`.

2. **Pointwise-reindex commutation**  
   For true pointwise ops: `map f (reindex η x)  =>  reindex η (map f x)`.

3. **Lift associativity normalization**  
   `lift P (lift Q f)  =>  lift (Q ⊗ P) f`.

4. **Broadcast normalization**  
   normalize equivalent `degree` constructions to one canonical `BroadcastJoin` normal form.

5. **Reduction simplifications**  
   singleton/empty-axis fold eliminations where shape proves trivial contraction.

6. **Scan batching distributivity**  
   for batch axis independent of temporal axis: `lift P (scan step)  ≅  scan (lift P step)`.

7. **Softmax constant-drop (existing DSL behavior, formalized)**  
   constants independent of norm axis may be removed before softmax.

These should start as verifier-preserving rewrites, then become optimization passes.

### 4.2 Staged rollout plan (low-risk implementation)

**Stage 0 — Documentation and naming alignment**
- Add this conceptual mapping (no runtime change).
- Keep DSL syntax unchanged.

**Stage 1 — Verifier hardening**
- Add explicit invariants around `CanonicalProgram`, `LoweredProgram`, `ThreadedComposed`.
- Fail loud on violated assumptions.

**Stage 2 — Capability inference**
- Introduce axis capability metadata (`naperian`, `temporal`, `reducible`, `broadcastable`) in compiler context.
- No syntax breakage.

**Stage 3 — Law-oriented internal APIs**
- Repackage existing logic as helpers aligned to `BroadcastJoin`, `ReindexAction`, `AxisReduce`, `PointwiseLift`, `TemporalGraded`.

**Stage 4 — Semantics-preserving rewrites**
- Enable reindex fusion and degree normalization first.
- Add scan/batch distributivity and scan-affine strategy selection behind flags.

**Stage 5 — Optional syntax sugar (opt-in)**
- Optional annotations (`temporal`, explicit `reduce`) if needed for readability; existing programs remain valid.

---

## 5. Lean implementation strategy with dependent-type optimization

When implementing the Naperian/applicative mixins in Lean, the following table maps Haskell concepts from Gibbons' paper to Lean equivalents (mostly in Mathlib). For each mixin, we show how dependent types simplify the implementation by encoding invariants in types rather than bundling them as separate proofs.

### 5.1 Haskell ↔ Lean translation and implementation patterns

| Haskell concept | Haskell signature | Lean equivalent | Location | Implementation note |
|---|---|---|---|---|
| **Applicative** | `class Applicative f` | `class Applicative (f : Type → Type)` | `Mathlib.Control.Applicative` | Base class for all lifting operations |
| `pure` | `pure : a → f a` | `Applicative.pure` or `pure` | `Mathlib.Control.Applicative` | Replication/broadcasting in our context |
| `<*>` | `(<*>) : f (a → b) → f a → f b` | `Applicative.seq` or `<*>` | `Mathlib.Control.Applicative` | Pointwise binary composition |
| `liftA2` | `liftA2 : (a → b → c) → f a → f b → f c` | `Applicative.liftA2` or `liftA2` | `Mathlib.Control.Applicative` | Used in `PointwiseLift` implementation |
| **Representable/Naperian** | `class Representable f where type Log f; ...` | `class Representable (f : Type u → Type v)` | `Mathlib.Data.Functor.Repr` | **Dependent optimization:** point types can live near axes, but representability still needs explicit `lookup`/`tabulate` laws |
| `lookup` | `lookup : Log f → f a → a` | `Representable.repr` (inverse) | --- | Evaluation at a point; becomes type-level for `Axis` |
| `tabulate` | `tabulate : (Log f → a) → f a` | `Representable.unrepr` | --- | Reconstruction; supplied by `NaperianFamily`, not by point enumeration alone |
| **Traversable** | `class Traversable t` | `class Traversable (t : Type u → Type u)` | `Mathlib.Data.Traversable.Basic` | Used in reduction/fold over axes |
| `traverse` | `traverse : Applicative f => (a → f b) → t a → f (t b)` | `Traversable.traverse` | `Mathlib.Data.Traversable.Basic` | Generalize pointwise ops to all traversable shapes |
| **Foldable** | `class Foldable t` | `class Foldable (t : Type u → Type v)` | `Mathlib.Data.Functor.Foldable` | Base for reductions |
| `fold` | `fold : Monoid a => t a → a` | `Foldable.fold` | `Mathlib.Data.Functor.Foldable` | Contract over monoidal aggregation |
| `foldl` | `foldl : (a → b → a) → a → t b → a` | `Foldable.foldl` | `Mathlib.Data.Functor.Foldable` | Used in scan iteration |
| `foldr` | `foldr : (a → b → b) → b → t a → b` | `Foldable.foldr` | `Mathlib.Data.Functor.Foldable` | Reverse-order reduction (for certain scan strategies) |
| **Array transpose** | `transpose :: f (g a) → g (f a)` | `Matrix.transpose : Matrix m n α → Matrix n m α` | `Mathlib.Data.Matrix.Transpose` | **Dependent optimization:** build from `Representable` isomorphism; encode axis permutation in type |
| **Isomorphism** | `(≅) : a → b` (type equivalence) | `Equiv : α ≃ β` or `Iso` in `CategoryTheory` | `Mathlib.Logic.Equiv.Basic` or `Mathlib.CategoryTheory.Iso` | Used for broadcast joins and reindex invertibility |
| **Monoidal functor** | `class MonoidalFunctor` | `class MonoidalFunctor` | `Mathlib.CategoryTheory.Monoidal.Functor` | Lift functor preserves monoidal structure |
| **Natural transformation** | `(⇒) : f → g` (between functors) | `Nat.Trans : F ⟹ G` | `Mathlib.CategoryTheory.Functor` | `ev_p` slices and evaluation transformations |
| **Symmetric monoidal** | `class SymmetricMonoidal` | `class SymmetricMonoidalCategory` | `Mathlib.CategoryTheory.Monoidal.Symmetric` | Permutation laws on colors |
| **Free monoid** | `Free f` or `[a]` | `List α` (linear free monoid) or `FreeMonoid α` | `Mathlib.Data.List.Basic` or `Mathlib.GroupTheory.FreeMonoid` | Object lists in colored PROP |

### 5.2 Mixin-specific implementation strategies (with dependent-type optimization)

**NaperianAxis: Fixed-size representability**

The core typeclass (in `LeanNCD/Core/Naperian.lean`):
```lean
class NaperianAxis (D : Type) [ColoredPROP D] where
  El : D → Type
  finite : ∀ P : D, Fintype (El P)
  point_hom : ∀ P : D, El P → SmallCategory.hom (ColoredPROP.unit : D) P
  mapEl : ∀ {P Q : D}, SmallCategory.hom P Q → El P → El Q
  mapEl_id : ∀ P (p : El P), mapEl (SmallCategory.id P) p = p
  mapEl_comp : ∀ {P Q R : D} (f : SmallCategory.hom P Q)
      (g : SmallCategory.hom Q R) p,
    mapEl (SmallCategory.comp f g) p = mapEl g (mapEl f p)
  point_hom_natural : ∀ {P Q : D} (η : SmallCategory.hom P Q) p,
    SmallCategory.comp (point_hom P p) η = point_hom Q (mapEl η p)
  strong_monoidal : ∀ P Q : D, El (ColoredPROP.tensor P Q) ≃ El P × El Q
  unit_point : El (ColoredPROP.unit : D) ≃ Unit
  -- Coherence: the equivalences are compatible with associativity and unit.
  strong_monoidal_assoc : ∀ P Q R,
    (strong_monoidal (ColoredPROP.tensor P Q) R).trans
      (Equiv.prodCongr (strong_monoidal P Q) (Equiv.refl _)) =
    (strong_monoidal P (ColoredPROP.tensor Q R)).trans
      (Equiv.prodCongr (Equiv.refl _) (strong_monoidal Q R)).trans
      (Equiv.prodAssoc _ _ _)
  strong_monoidal_unit_l : ∀ P,
    (strong_monoidal (ColoredPROP.unit : D) P).trans
      (Equiv.prodCongr unit_point (Equiv.refl _)) =
    Equiv.punitProd _
  strong_monoidal_unit_r : ∀ P,
    (strong_monoidal P (ColoredPROP.unit : D)).trans
      (Equiv.prodCongr (Equiv.refl _) unit_point) =
    Equiv.prodPUnit _
```

`El P` is the finite set of coordinate tuples of shape `P`. `point_hom` bridges to the existing `ev_p` API (converting a finite coordinate into a global element `I_D → P`), and `mapEl` records the functorial action of `D`-morphisms on finite points. `lookup` and `tabulate` — the Naperian isomorphism `F(X ⊛ P) ≅ (El(P) → F(X))` — live in a separate `NaperianFamily` class, keeping `NaperianAxis` focused on the index category `D` only. This requires an additional representability assumption — that the lifted functor `act(−, P)` is represented by `El P` — which is not a consequence of strong monoidality alone. This assumption is encoded in `NaperianFamily` as explicit `lookup`/`tabulate` fields.

**Important:** the existing `Axis` structure (`name : Option String; size : Numeric`) does not carry point/enumeration data, because `size : Numeric` is symbolic and cannot determine a `Fintype` alone. The `NaperianAxis StObj` instance is provided via an auxiliary `AxisPointData` typeclass in `LeanNCD/Instances/StNaperian.lean` that supplies concrete point types externally — without mutating `Axis` or breaking the existing `StMat`, DSL, or quotient code. See [NaperianTypingIntegrationPlan.md §3](NaperianTypingIntegrationPlan.md) for the full instance code.

Benefit: `El P` is a type-level invariant; finiteness is enforced by `Fintype` rather than bundled as a separate proof; the `strong_monoidal` fields ensure the point functor commutes coherently with the monoidal structure of `D`.

---

**BroadcastJoin: Canonical degree computation**

The mixin class (in `LeanNCD/Core/Naperian.lean`):
```lean
class BroadcastJoin (D : Type) [ColoredPROP D] [NaperianAxis D] where
  Join : D → D → D
  inl : ∀ P Q, SmallCategory.hom P (Join P Q)
  inr : ∀ P Q, SmallCategory.hom Q (Join P Q)
  -- Coordinate projection: points of the join project to points of each factor.
  projl : ∀ P Q, El (Join P Q) → El P
  projr : ∀ P Q, El (Join P Q) → El Q
  projl_natural : ∀ P Q p,
    SmallCategory.comp (point_hom (Join P Q) p) (inl P Q) =
      point_hom P (projl P Q p)
  projr_natural : ∀ P Q p,
    SmallCategory.comp (point_hom (Join P Q) p) (inr P Q) =
      point_hom Q (projr P Q p)
  -- Universal property: a point into the join is determined by its projections.
  join_point_sep : ∀ P Q (r : El (Join P Q)) (s : El (Join P Q)),
      projl P Q r = projl P Q s → projr P Q r = projr P Q s → r = s
```

For broadcast alignment, points of the common shape project to each operand shape; a coproduct-style point split is not the right general structure. The precise instance of `BroadcastJoin StObj` depends on the chosen axis-alignment strategy (append vs UID union) and is deferred.

---

**ReindexAction: Rank-safe reindexing**

Traditional approach:
```lean
class ReindexAction (D C : Type) [ColoredPROP D] [ColoredPROP C]
    [DGradedColoredPROP D C] where
  reindex : ∀ {X : C} {P Q : D}, SmallCategory.hom P Q →
    SmallCategory.hom (X ⊛ Q) (X ⊛ P)
```

Dependent-type optimization (encode codomain rank):
```lean
-- StMatP is typed so codomain rank is enforced at construction
def StMatP (from to : StObj) : Type :=
  {coeffs : Matrix (Fin to.length) (Fin from.length) Coeff //
   -- Invariant: to.length = rows, from.length = cols, built into the type}

def reindex (η : StMatP P Q) {X : C} : X ⊛ Q → X ⊛ P :=
  -- Type signature makes contravariance and codomain rank implicit
  sorry
```

Benefit: A pass that produces `η : StMatP P Q` *statically guarantees* codomain rank; no runtime check needed.

---

**PointwiseLift: Separation from contraction**

Traditional approach:
```lean
class PointwiseLift (D C : Type) [DGradedColoredPROP D C] where
  lift1 : ∀ {A B : C}, SmallCategory.hom A B → ∀ (P : Dᵒᵖ),
    SmallCategory.hom (A ⊛ P) (B ⊛ P)
  lift2 : ∀ {A B C0 : C}, SmallCategory.hom (A ⊗ B) C0 → ∀ (P : Dᵒᵖ),
    SmallCategory.hom ((A ⊗ B) ⊛ P) (C0 ⊛ P)
```

Dependent-type optimization (mark operations as pointwise at definition):
```lean
structure PointwiseOp (A B : C) where
  op : SmallCategory.hom A B
  isPointwise : True  -- Marker; type says "preserve all degrees"

-- Every PointwiseOp lifts automatically; no explicit lift1/lift2 needed
def instance_lift (f : PointwiseOp A B) (P : Dᵒᵖ) : SmallCategory.hom (A ⊛ P) (B ⊛ P) :=
  sorry
```

Benefit: Operations are pre-classified; the compiler knows which ops compose with lifts without checking.

---

**TemporalGraded: Iteration discipline**

Traditional approach:
```lean
class TemporalGraded (D C : Type) [ColoredPROP D] [ColoredPROP C]
    extends DGradedColoredPROP D C where
  L : D
  prefix : ℕ → D
  iotaTo : ∀ {m n : ℕ}, m ≤ n → SmallCategory.hom (prefix m) (prefix n)
  -- ... laws for prefix compatibility
```

Dependent-type optimization (encode temporal structure):
```lean
structure ScanState (L : ℕ) (X : C) where
  base : X
  step : ∀ (l : Fin L), X → X
  history : ∀ (l : Fin (L + 1)), X
  -- Equations history 0 = base and history (l+1) = step l (history l)
  -- are implicit from structure definition

def scan {L} {X} (s : ScanState L X) : X ⊛ (prefix L) :=
  sorry  -- Return type enforces output is lifted over temporal axis
```

Benefit: Causality and state-shape stability are implicit; the type system won't allow l+1 reads because `history` only takes `Fin (L+1)`.

---

### 5.3 Staged dependent-type introduction

To avoid rewriting everything at once:

1. **Stage A**: Introduce the new Naperian API without touching existing structures:
   - `LeanNCD/Core/Naperian.lean` with `NaperianAxis`, `NaperianFamily`, `BroadcastJoin`, `ReindexAction`, `PointwiseLift`
   - `LeanNCD/Instances/StNaperian.lean` with `AxisPointData` and the `NaperianAxis StObj` instance
   - `StMatP` with rank in type (new code only, alongside existing `StMat`)
   - `ScanState` for temporal structure (alongside existing `TemporalGraded`)

2. **Stage B**: Retroactively add dependent-type wrappers around existing `St`/`Br`:
   - Bridge old `StObj`-based code via `Equiv` (not breaking change)
   - New passes generate dependent types directly

3. **Stage C**: Gradually lift old code to native dependent types as it's touched:
   - `lowerArith` → produces `StMatP`, not loose matrices
   - `route` → already type-checks `degree` structure

This keeps existing tests and proofs valid while allowing new infrastructure to be cleaner.

### 5.4 Key LeanNCD custom types and their Haskell analogues

| LeanNCD type | Lean definition | Analogous Haskell pattern | Dependent-type benefit |
|---|---|---|---|
| `Axis` | `structure Axis where name : Option String; size : Numeric` | dimension metadata + size | Currently symbolic — point/enumeration data supplied separately via `AxisPointData` typeclass; avoids breaking `StMat`, DSL, and quotient code |
| `StObj` | `abbrev StObj := List Axis` | list of dimensions | Shape = ordered list of axes; rank is `List.length` |
| `StMat` | `structure StMat (dom cod : StObj)` | affine function | Can strengthen to `StMatP` with rank dependent on `cod.length` |
| `AxisSpec` | UID + name + kind | reference to a live axis | Keep operational; used during elaboration before dependent lifting |
| `IdxExpr` | `inductive IdxExpr` | integer-affine polynomial | Elaborates to `StMatP` rows; invariants checked via type |
| `TermTraversable` | `class TermTraversable α` | `Traversable`-like for UID substitution | Custom to LeanNCD; no dependent-type change needed |
| `Context` (UID coequalize) | UID equivalence classes | union-find or quotient monad | Produces canonical `Axis` uids before dependent stage |

### 5.5 Integration implementation plan

The strategies described in §5.1–5.4 are elaborated into a concrete, file-by-file implementation plan in [NaperianTypingIntegrationPlan.md](NaperianTypingIntegrationPlan.md). That document covers:

- **New files to create**: `LeanNCD/Core/Naperian.lean` (the `NaperianAxis`, `BroadcastJoin`, `ReindexAction`, and `PointwiseLift` typeclasses, plus `ev_naperian` and the `naperian_jointly_monic` statement) and `LeanNCD/Instances/StNaperian.lean` (the `NaperianAxis StObj` instance).
- **Modifications to existing files**: minimal changes to `Core/Graded.lean`, `Base/ColoredPROP.lean`, `Instances/StBr.lean`, and `LeanNCD.lean`; all backward-compatible.
- **The `NaperianAxis StObj` instance in full Lean 4**: using an auxiliary `AxisPointData` typeclass to supply concrete finite point types without mutating the existing `Axis` structure (necessary because `Axis.size : Numeric` is symbolic and does not determine a `Fintype` on its own).
- **Sorry-impact analysis**: which of the 16 open sorries become easier, which become closeable, and which remain independent. Key result: `δ`/`δ0`/`υ`/`α` get an intended pointwise proof strategy via coherent `El` equivalences once `act` and extensionality are available; `broadcast_gen` and `weave_unique` can be reframed as Naperian normal-form theorems without depending on `brCancelPoint`.
- **Milestone sequencing**: seven milestones from baseline confirmation through weave uniqueness, with a clearly identified minimum viable integration (three steps).
- **Risks**: the symbolic-size blocker, quotient-`Br` interaction, instance diamond risks, and the `α` tensor-order convention.

---

## 6. Payoff: benefits for users, maintainers, and formalization

### What this integration delivers

This proposal gives the LeanNCD tensor-logic pipeline and formalization effort:

- **Unified semantics:** broadcast/reindex/reduce/scan all live in one typed framework with explicit algebraic laws.
- **Better correctness:** fewer implicit conventions; more checkable invariants.
- **Optimization headroom:** law-backed rewrites (reindex fusion, scan batching, broadcast normalization) instead of heuristic transforms.
- **Proof/runtime alignment:** Lean proofs and executable lowering use the same conceptual primitives.
- **Backward compatibility:** rollout can preserve current DSL source surface and existing programs.

### Concrete benefits for Tensor Logic users

1. **Clearer error messages and earlier failures**  
   Many currently implicit consistency constraints become explicit law checks (degree alignment, reindex rank/codomain checks, scan causality constraints).

2. **More predictable compilation behavior**  
   Canonical broadcast/degree normal forms reduce dependence on incidental source ordering and make `route` decisions easier to reason about.

3. **Safer scan and recurrence evolution**  
   Temporal laws make it easier to extend scan features (affine scans, batched scans, coupled states) without fragile ad hoc conditions.

4. **Better optimization contracts**  
   Rewrites such as reindex fusion or pointwise-reindex commutation are justified by laws, reducing optimizer brittleness.

5. **Stronger bridge to executable backends**  
   The same abstractions describe both symbolic Tensor Logic equations and routed/evaluated execution, improving confidence in equivalence.

### Benefits for maintainers and formalization work

This framework directly reduces the formalization burden by restructuring proof obligations around type-level invariants.

**Immediate benefits:**

- **Lower proof burden drift:** implementation invariants mirror formal propositions, keeping theory and practice aligned.
- **Modular evolution:** new ops can declare capability requirements (`PointwiseLift`, `AxisReduce`, temporal compatibility) instead of patching multiple passes.
- **Better test design:** tests can target law-level properties (e.g., lift associativity normalization) in addition to example outputs.
- **Easier to document:** the framework provides a canonical vocabulary for axis operations, making both user docs and internal design docs clearer.

#### Measurable proof-effort reduction

Naperian typing restructures proof obligations by encoding invariants in types. Analysis of LeanNCD's open proof obligations (from `SORRY_INVENTORY.md`, Milestones A–F) projects:

| Proof family | Current | Projected | Impact |
|---|---:|---:|---|
| Representable/applicative laws (`instDGradedStBr` fields) | 10 | 2–4 | `δ`/`δ0`/`υ`/`α` get pointwise proof strategies from coherent `El` equivalences once `act` and extensionality are available; `act`, `broadcast_gen`, and coherence laws still require substantial work |
| Reindex identity/composition | scattered | ~0 | Dependent rank eliminates dimension checks |
| Weave uniqueness (`weave_unique`) | 1 | 0–1 | `weave_unique_naperian` (new theorem over `St` evaluations) avoids `brCancelPoint`; existing `weave_unique` via `Elemental Br` path unchanged |
| Scan/temporal laws | ~0 current | 0 future | Type-level causality barrier |
| Bridge/compilation (`E2b` bridge) | 6 | 1–2 | Well-formedness by construction |
| Point cancellation (`brCancelPoint`) | 1 | 0–1 | NbE/semantic Br rebuild; Naperian makes it less critical if consumers switch to `weave_unique_naperian` via `naperian_jointly_monic` (distinct from (Elem-C): this separates lifted morphisms by D-point evaluations, not C-morphisms by C-points) |
| **Subtotal** | **~18–20** | **3–6** | **estimated 50–70% reduction, contingent on `act` being defined and `naperian_jointly_monic` being provable without circularity** |

**Key proof simplifications:**

1. **Representable axioms (10 → 2–4):** Fields `δ`, `δ0`, `υ`, `α` get canonical pointwise proof strategies once the object-level `act` is defined, because their intended behavior is described by point-set equivalences — `El [] ≃ Unit` gives `υ`, `El (P ++ Q) ≃ El P × El Q` gives `δ`, and `alphaElEquiv P Q := appendEquiv Q P` gives the `α` order. However, the El-equivalences provide the *intended proof strategy* only once `act` is implemented and `naperian_jointly_monic` is available; without those, the equivalences do not constitute proofs of the coherence isos. The `act` functor itself still requires implementing the `BrMorph` quotient action; `broadcast_gen` and the coherence laws `act_unit_assoc`/`υ_nat`/`dist_coh` require substantial proof work and are not automatically closed by Naperian typing. `sh_act` simplifies once the object action is defined. See [NaperianTypingIntegrationPlan.md §4](NaperianTypingIntegrationPlan.md) for the per-field analysis.

2. **Reindex soundness (scattered → ~0):** Dependent `StMatP dom cod` encodes domain/codomain ranks in the type, making identity law `reindex id = id` structural (identity matrix on canonical rank) and composition law (`reindex (η ≫ θ) = reindex θ ≫ reindex η`) standard matrix algebra.

3. **Bridge correctness (6 → 1–2):** Typed `ThreadedComposed` carries well-formedness by construction; `realize` becomes structural recursion over a well-formed routed DAG. Validation for CSV input (external data) remains the main overhead.

4. **Scan laws (unchanged now, 0 future):** Type-level causality barrier (prefix-restricted inputs in type signature) prevents future-reads and makes trace shape explicit, eliminating scan correctness sorries as features evolve.

**Mechanism:** Rather than building syntax and then proving properties hold, Naperian typing constructs well-typed objects where properties hold by construction. Obligations migrate from scattered local sorries into reusable law bundles (`LeanNCD/Mixins/`), reducing local proof burden and centralizing verification effort.

**Tradeoffs:** Dependent transports (UID equality) introduce `eqToHom` casts; CSV input must validate before entering the dependent type system; symbolic sizes (`SizeExpr`) require runtime environments; free-SMC coherence laws remain until NbE is added (already planned).

**Recommended rollout:** Phase 1 (Naperian API + `StObj` instance) → Phase 2 (implement `act` object/morphism level; `δ`/`δ0`/`υ`/`α` get pointwise proof strategies, reducing from 10 to ~4–6 sorries if extensionality is available) → Phase 3 (dependent structures; `broadcast_gen`, `weave_unique_naperian`, bridge sorries, reducing to ~3–6) → Phase 4 (optional NbE rebuild for `brCancelPoint`, final 1–3 sorries). `Elemental BrObj` (`brCancelPoint`) remains independent and is deprioritised once `weave_unique_naperian` is available. `naperian_jointly_monic` is not the same as `Elemental C`; if it is proved from `broadcast_gen`, it must not be used as an input to prove `broadcast_gen`.

In short, this gives a tighter "theory ↔ compiler ↔ runtime" loop with measurable formalization payoff and minimal user-facing disruption.

---

## 7. References

**Key papers and primary sources:**

1. **[Gibbons 2016]** — Jeremy Gibbons, *[APLicative Programming with Naperian Functors](https://www.cs.ox.ac.uk/people/jeremy.gibbons/publications/aplicative.pdf)*.  
   University of Oxford, 2016. Foundational work on representable functors, applicative programming, and rank polymorphism in typed array languages.

2. **[MacLane]** — Saunders MacLane, *[Categories for the Working Mathematician](https://www.springer.com/gp/book/9780387984032)*.  
   Springer, 2nd edition. Standard reference for category theory, monoidal functors, coherence, and natural transformations.

3. **[MacLane & Moerdijk]** — Saunders MacLane and Ieke Moerdijk, *[Sheaves in Geometry and Logic: A First Introduction to Topos Theory](https://www.springer.com/gp/book/9780387977102)*.  
   Springer. Standard reference for dense families, separating families, and the Yoneda principle.

4. **[Plotkin & Power 2003]** — Gordon Plotkin and John Power, *[Algebraic Operations and Generic Effects](https://homepages.inf.ed.ac.uk/gdp/publications/alg_ops_gen_effects.pdf)*.  
   2003. Theory of normal forms and free completions for algebraic effects.

**LeanNCD-specific references:**

5. **[graded_prop.md](../papers/graded_prop.md)** — Graded Colored PROPs: A Categorical Synthesis of St, Br, Acsets, and Algebras.
   Located at: [`papers/graded_prop.md`](../papers/graded_prop.md)

6. **[tensor_logic_dsl.md](../docs/tensor_logic_dsl.md)** — The Tensor Logic DSL.
   Located at: [`docs/tensor_logic_dsl.md`](../docs/tensor_logic_dsl.md)

7. **LeanNCD pipeline implementation** (linked in source):
   - [Structural.lean](../../leanncd/LeanNCD/DSL/Pipeline/Structural.lean) — Axis naming, resolution, and unification passes
   - [Lowering.lean](../../leanncd/LeanNCD/DSL/Pipeline/Lowering.lean) — Nonlinearity splitting, scheduling, routing  
   - [Scan.lean](../../leanncd/LeanNCD/Eval/Scan.lean) — Scan evaluation and temporal structure

**Lean/Mathlib standard library equivalents:**

8. **[Mathlib.Control.Applicative](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Control/Applicative.html)** — Applicative functor typeclass and laws

9. **[Mathlib.Data.Functor.Repr](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Data/Functor/Repr.html)** — Representable functors and the `represent` / `unrepr` duality

10. **[Mathlib.Data.Traversable.Basic](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Data/Traversable/Basic.html)** — Traversable structures and traversals

11. **[Mathlib.Data.Functor.Foldable](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Data/Functor/Foldable.html)** — Catamorphisms and recursive fold patterns

12. **[Mathlib.CategoryTheory.Monoidal.Functor](https://leanprover-community.github.io/mathlib4_docs/Mathlib/CategoryTheory/Monoidal/Functor.html)** — Strong monoidal functor typeclass and coherence

13. **[Mathlib.CategoryTheory.Monoidal.Symmetric](https://leanprover-community.github.io/mathlib4_docs/Mathlib/CategoryTheory/Monoidal/Symmetric.html)** — Symmetric monoidal structures and braiding

**Category theory background (optional):**

14. **[nLab: Directed Objects](https://ncatlab.org/nlab/show/directed+object)** — Categorical treatment of directed structures and causality constraints.

15. **[nLab: Kan Extension](https://ncatlab.org/nlab/show/Kan+extension)** — Left/right Kan extensions, used to formalize reduction and fold operations.

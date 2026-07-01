# Naperian Typing for Tensor Logic DSL in a `D`-Graded Colored PROP

This note explains how to integrate Naperian/applicative array semantics into the existing LeanNCD tensor-logic pipeline and `D`-graded colored PROP formalism, with minimal disruption to current code paths.

It is meant as a practical bridge between:

- [`papers/graded_prop.md`](../papers/graded_prop.md) (categorical core),
- [`docs/tensor_logic_dsl.md`](../docs/tensor_logic_dsl.md) (user-facing DSL),
- [`leanncd/LeanNCD/DSL/Pipeline/*.lean`](../leanncd/LeanNCD/DSL/Pipeline/*.lean) (actual passes).

## Contents

1. [Motivation and alignment](#1-motivation-and-alignment)
2. [Semantic foundation: Naperian axes in a D-graded PROP](#2-semantic-foundation-naperian-axes-in-a-d-graded-prop)
   - [Gibbons' APLicative/Naperian results](#gibbons-aplicativenaperian-results)
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

The rollout is staged and backward-compatible (see [§4.2](#42-staged-rollout-plan-low-risk-implementation) for stages and [§5.3](#53-staged-dependent-type-introduction) for the dependent-type introduction order). One sequencing point matters up front:

> **Sequencing relative to in-flight work.** This is a forward-looking design; it is **not**
> scheduled to start now. The active LeanNCD proof effort is the multi-output `BrBase` +
> `compile_wellFormed` work (see `leanncd/docs/superpowers/plans/2026-06-26-multioutput-impl-plan.md`).
> This integration should be sequenced **after** that lands — it touches the categorical core
> (`Graded.lean`, `ColoredPROP.lean`, `StBr.lean`) and, honestly, in the near term it *adds*
> surface area (six new typeclasses with coherence-law fields) before it removes any sorries. Do
> not open this in parallel with the well-formedness proof; treat the [minimum viable
> integration](NaperianTypingIntegrationPlan.md) as the first thing to attempt once that work is
> complete.

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

**Concrete example:** Binary addition in the *`ZipList`* applicative (the pointwise/aligned one — which is the Naperian-relevant applicative for fixed-shape arrays):
```
liftA2 (+) (ZipList [1, 2, 3]) (ZipList [10, 20, 30])  →  ZipList [11, 22, 33]
```
Each element of the output is the sum of corresponding elements in the inputs. (Note: Haskell's *default* list applicative is the **Cartesian product**, so `liftA2 (+) [1,2,3] [10,20,30]` there yields the nine sums `[11,21,31,12,22,32,13,23,33]`, not the pointwise result. The pointwise/broadcasting semantics we want is `ZipList`, not the default list instance.)

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

*Note on terminology:* `Log f` (the "logarithm" of the functor) is the index type (same as the functor's **shape** or **dimensions**). The name comes from the exponential analogy: if a container `f a` is isomorphic to a function `Log f → a`, then `Log f` plays the role of the "exponent." Concretely: `Log(Pair) = Bool` (a pair is a function from `Bool`), `Log(Array n) = Fin n` (integers 0..n-1), and for a 2D array of shape `(m, n)`, `Log = Fin m × Fin n`. (Only *fixed-shape* functors are representable — `Maybe` and `List`, whose shape varies with the value, are **not** Naperian and have no `Log`.)

**For arrays:** A 2D array of shape `(m, n)` is isomorphic to a function from pairs `(i, j) ∈ [0..m) × [0..n)` to values. Indexing is retrieval; tabulation is construction.

This representability is the key difference from general applicatives: it makes **indexing a type-level property**, not a runtime convention. Shape mismatches become compile-time errors.

### Gibbons' APLicative/Naperian results

Jeremy Gibbons' paper (see [References](#7-references)) gives a typed account of APL-style rank polymorphism via fixed-shape representable functors:

1. **Arrays as nested dimensions.**  
   An `n`-dimensional array is modeled as nested functors (shape at the type level).

2. **Applicative lifting = pointwise semantics.**  
   `pure` gives replication/broadcasting; `<*>`/`zipWith` gives pointwise lifted operators.

3. **Naperian (representable) dimensions = indexability.**  
   `f a ≅ (Log f → a)`, witnessed by the inverse pair `lookup`/`tabulate`, so indexing is total and invertible (defined under [Representable Functors](#representable-functors-naperian-functors) above).

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
- coherence (`δ`, `δ0`, `υ`, `α`): the natural isomorphisms ensuring the action is well-behaved — `δ` is distributivity `(X ⊗ Y) ⊛ P ≅ (X ⊛ P) ⊗ (Y ⊛ P)`, `δ0` is the *nullary* distributivity `I ⊛ P ≅ I` (the tensor unit's action is trivial), `υ` is the unit-grading law `X ⊛ I_D ≅ X`, and `α` is associativity of nested actions `(X ⊛ P) ⊛ Q ≅ X ⊛ (Q ⊗ P)` (note the `Q ⊗ P` order — matches `graded_prop.md` Def 3.1),
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

Representability (defined [above](#representable-functors-naperian-functors): `F(X) ≅ (Log F → X)`) instantiates here as `Log(StMatP dom cod) = dom.point × cod.point`, so `lookup` extracts the element at `(i, j)` and `tabulate` rebuilds from `(i, j) ↦ value(i, j)`. Because we work with this **indexed-function view** (not raw arrays), composition becomes function composition:

```lean
def ap {A B : Type} (f : A → (B → C)) (x : A → B) : A → C :=
  fun a => f a (x a)
  -- This is definitional: just function application
  -- No proof needed
```

For the indexed-function (reader) applicative, laws like `pure id <*> v = v` reduce to `fun i => v i = v`, which holds **definitionally in Lean 4** by beta plus function eta — no explicit proof needed. (This is specific to the bare function/structure view and Lean 4's eta; it is not a claim that applicative laws are free for *every* applicative. It also relies on `StMatP` being definitionally the indexed function, so structure eta applies.)

#### The proof-obligation cascade

This extends to axioms about shapes and compositions:

| Without Representability | With Representability | Why it works |
|---|---|---|
| Build matrix + prove dimensions match | Type signature enforces dimensions | Type constructor is the constraint |
| Prove `tabulate (lookup m) = m` | Isomorphism is definitional | The type is the lookup function |
| Prove binary operations commute (broadcast) | Use applicative `<*>` | `<*>` is *defined* as pointwise application |
| Prove composition is associative | Function composition associativity | Function composition *is* syntactically associative |
| Prove causality (no future reads in scans) | Encode in dependent type | Type forbids invalid state transitions |

Each "new" morphism no longer introduces *new* proof obligations. Instead, we instantiate **universal laws** that come from representability and applicative structure. These laws are *once-and-for-all proofs* about representable functors and applicative operations—they do not depend on problem-specific details. **Scope caveat:** this mechanism eliminates *indexing/dimension* obligations (shape match, rank agreement, reindex identity/composition); it does **not** by itself discharge the coherence isos, the quotient-`Br` action, or `broadcast_gen`, which remain substantial (see [§4.2 What does not get solved automatically](NaperianTypingIntegrationPlan.md) and the projected — not measured — counts in [Projected proof-effort reduction](#projected-proof-effort-reduction-a-hypothesis-to-validate-not-a-measured-result)).

---

### Categorical Integration: Naperian Structure in D-Graded PROPs

The mixins above are not decorative; they follow from a deep categorical principle. This subsection explains how Naperian typing refines the `D`-graded colored PROP formalism.

#### The point functor and representability

The heart of Naperian semantics is **representability** — a fundamental concept in category theory (see [References](#7-references)) where a functor is isomorphic to a hom-functor. In our setting, the isomorphism is:

$$F(X ⊛ P) \cong \text{El}(P) \to F(X)$$

(*Intuition:* a P-shaped container of X-values is completely determined by specifying, for each point (coordinate) in P, what the corresponding X-value is.)

Here we introduce the key notation:
- **⊛** is the **graded action** functor (from [graded_prop.md §3](#7-references)): the right action of objects in `D` on objects in `C`. This is the core structure of D-graded PROPs.
- **$\text{El}(P) := D(I_D, P)$** denotes the **points** (or **global elements**) of a shape object `P` in the index category `D`. Here `D(I_D, P)` is the **hom-set** — the set of all morphisms `I_D → P` in `D`, where `I_D` is the monoidal unit of `D` (written `I_D` throughout [graded_prop.md](#7-references); it is the unit for the `⊛` action, satisfying `X ⊛ I_D ≅ X`). Each morphism `I_D → P` picks out a single "coordinate" or "element" of `P`, hence the name. 
  
  In practice: In the concrete `StObj` setting, the full hom-set `D(I_D, P)` consists of all affine-stride morphisms `StMat [] P`. The symbolic `NaperianAxis StObj` instance supplies `El P` as an abstract point *type* plus `point_hom` (the embedding into `D(I_D, P)`); the *finite* chosen representative set is supplied separately, post-solver, by `AxisPointData` (see [§5.2](#52-mixin-specific-implementation-strategies-with-dependent-type-optimization) for why finiteness is kept out of the symbolic class). The categorical theory is stated for the full hom-set; the Lean implementation works with a finite representative (see the **Connection to elementality** note below for how this relates to elementality and separating families).
- **$F$** is a **semantic algebra**: a functor that interprets objects of `C` into concrete data structures (e.g., arrays, functions, or numerical values).
- **$\cong$** denotes a natural isomorphism: a natural transformation whose components are all isomorphisms. For this formula specifically, the isomorphism is natural in `X` (covariantly — the isomorphism commutes with morphisms `X → X'` in `C`) and natural in `P` (contravariantly) via `mapEl η : El(P) → El(Q)` for `η : P → Q` in `D` — a field of `NaperianAxis` satisfying functoriality and naturality with `point_hom`.

This representability is not coincidental. Following [Gibbons 2016](#7-references), for a **point functor** 
$$\text{El} : D \to \mathsf{FinSet}, \quad P \mapsto D(I_D, P)$$
with the **strong monoidal property** $\text{El}(P \otimes Q) \cong \text{El}(P) \times \text{El}(Q)$ (where $\otimes$ is the monoidal product in `D`), the point functor has the right product behavior for Naperian indexing. The equivalences `El(P ⊗ Q) ≃ El P × El Q` must satisfy standard monoidal coherence laws (associativity, left/right unit) to constitute a genuine strong monoidal functor; these are carried as explicit fields of `NaperianAxis`.

**Important caveat:** Strong monoidality of `El` is necessary but not sufficient for the isomorphism `F(X ⊛ P) ≅ (El(P) → F(X))`. The representability assumption — that the lifted functor `act(−, P)` is represented by `El P` — must be supplied explicitly via the `NaperianFamily` typeclass, which packages `lookup`/`tabulate` and their inverse laws. Strong monoidality ensures the point functor respects monoidal structure, making coherence laws like `El(P ⊗ Q) ≃ El(P) × El(Q)` well-behaved, but does not produce the isomorphism itself.

**Connection to elementality:** When the representability assumption holds, the evaluation transformations `ev_p := act(−, p)` (for $p \in \text{El}(P)$) form a jointly separating family for lifted `P`-indexed values. This Naperian joint monicity—where **lifted P-indexed families are determined by their values at each coordinate**—is a **different layer** from the **(Elem-C)** axiom of [graded_prop.md](#7-references). The (Elem-C) axiom says that **base C-morphisms are separated by C-points** (i.e., base operations are determined by their pointwise behavior). Naperian is a *separate assumption* about the **lifted** structure: it says that when we lift morphisms by a shape P, the resulting P-shaped families are determined by their coordinates. You can have (Elem-C) without Naperian representability; they are not derived from each other. Naperian typing makes the lifted dense family explicit in the type system by:
1. **Explicitly enumerating points**: The point functor `El` is finite and enumerable (for `D = St`, these are coordinate tuples).
2. **Typeclass-level monoidality**: Strong monoidality, with coherence laws, ensures the point functor respects the monoidal structure of `D`.
3. **Representability as first-class semantics**: Rather than deriving representability from `NaperianAxis`, Naperian typing records it in `NaperianFamily`, making `lookup`/`tabulate` laws explicit.

Additionally:
- **`lookup_p : X ⊛ P → X`** corresponds to the **evaluation transformation** $\text{ev}_p$ from [graded_prop.md §3](#7-references): for each point $p \in \text{El}(P)$, evaluating a `P`-indexed family at `p` yields a value in the base type. This evaluation is exactly the `ev_p` slice used in [graded_prop.md §3](#7-references).
- **`tabulate`** (the inverse) is the **universal property** of representability: it reconstructs a `P`-indexed family from its pointwise evaluations. Because the `ev_p` are jointly separating under `naperian_jointly_monic`, this reconstruction is **unique** — a key fact used in the weave-uniqueness result ([graded_prop.md §3.3, Prop 8.2](#7-references)).

#### Fiber semantics and reindexing

From [graded_prop.md §3](#7-references), objects of `C` are **fibered over shapes in `D`** via the **shape map** 
$$\text{sh} : O_C \to \text{Ob}(D).$$
(Read: every object in the operation PROP `C` is assigned a shape/index object in `D`.) This is one of the core axioms of D-graded PROPs.

For a morphism $\eta : P \to Q$ in `D`, the existing D-graded PROP structure defines a **contravariant action** (§3, [graded_prop.md](#7-references))
$$[X, \eta] : X ⊛ Q \to X ⊛ P.$$
(Read: "applying `η` to `X` at shape `Q` produces a morphism to shape `P`".)

Under the Naperian isomorphism, this contravariant action is exactly **precomposition of families**:
$$(\text{El}(Q) \to A) \xrightarrow{\eta^*} (\text{El}(P) \to A)$$
where $\eta^* : \text{El}(P) \to \text{El}(Q)$ is the **induced map on points**, implemented as `mapEl η` and satisfying `point_hom p ≫ η = point_hom (mapEl η p)`. This exhibits **contravariance**: pulling back along $\eta : P \to Q$ reverses the direction. For `StObj`, `mapEl η p` applies the affine stride map `η : StMat P Q` to coordinates, so the concrete instance must ensure `toCoeff` values are valid coordinates.

This interpretation clarifies the existing structure:

- **`ReindexAction`** is not new algebra; it is the **contravariance of the action `act`** (from [graded_prop.md](#7-references)) made explicit via representability. Every reindexing is a precomposition in the index category `D`.
- **Affine index arithmetic** (e.g., $i \mapsto 2i+1$) is a **restricted class of `D`-morphisms**, particularly when `D = St` (the spatial index PROP). With `mapEl` and the relevant joint-monicity/extensionality theorem, reindexings can be classified by their action on points.

#### Applicative lifting and distributivity

The strong monoidal action gives the distributivity isomorphism
$$\delta_{X,Y,P} : (X \otimes Y) ⊛ P \cong (X ⊛ P) \otimes (Y ⊛ P)$$
This is exactly the applicative zipping law:
$$(\text{El}(P) \to X \times Y) \cong (\text{El}(P) \to X) \times (\text{El}(P) \to Y)$$

Consequently:

- **`PointwiseLift`** and `liftA2` with type $$\text{liftA2} : (a \to b \to c) \to (\text{El}(P) \to a) \to (\text{El}(P) \to b) \to (\text{El}(P) \to c)$$ are the same construct, made categorical.
- **`BroadcastJoin`** constructs a canonical common shape in `D`, with coordinate projections back to each operand, before applying `δ`.
- Binary operations in the broadcastable fragment factor as: pointwise lifting of base ops, then reindexing to common shape.
- For **`α`**, the action order matters: `(X ⊛ P) ⊛ Q ≅ X ⊛ (Q ⊗ P)`, so the `El`-side uses `El(Q ⊗ P) ≃ El Q × El P`. For `StObj`, use `alphaElEquiv P Q := appendEquiv Q P`, not `appendEquiv P Q`.

#### Temporal grading as directed folding

The `TemporalGraded` mixin adds **causality**: output at time `t` depends only on times `≤ t`. This is a special case of the `D`-graded structure when the index category `D` is enriched with **directed structure** (see [References](#7-references) in category theory literature). Categorically:

1. The **temporal axis** `L` is a **directed index object** in `D`, typically isomorphic to the finite ordinal $[0..N]$ with a **prefix order** (or **interval-order morphisms** in standard terminology). In [graded_prop.md](#7-references), this corresponds to a distinguished temporal color in the index PROP.

2. A **scan** is not a pointwise operation along `L`; it is a **finite fold** / **catamorphism** over the directed structure. Formally (from [References](#7-references)), it is a morphism out of the initial algebra of the prefix-indexed functor, encoding stepwise iteration with feedback.

3. The **lift-fold distributivity law** states
   $$[\text{Scan}, P] \cong \text{Scan}([\text{step}, P])$$
   for `P` orthogonal to the temporal axis `L`. This has the **shape of a Beck-Chevalley / Fubini law** (see [graded_prop.md §3](#7-references)) on the **grading fibration** $C \to D$: the action and folding commute when dimensions are independent. (This is an analogy that motivates the proof target, not a claim that the LeanNCD scan law is *literally* an instance of Beck-Chevalley — establishing that would itself be work; see the scope note under [Reformulated view](#reformulated-view-naperian-d-graded-props).)

The constraint that **no morphism can depend on future indices** is formalized as: only **monotone / prefix-respecting `D`-morphisms** are allowed along temporal axes. (A prefix-respecting morphism maps $[0..s]$ into $[0..s']$ for any `s`, preserving causality.) This causality constraint **cannot be expressed purely pointwise**; it requires the full directed structure of the grading fibration.

#### Broadcast generation as applicative normal form

The **broadcast-generation axiom** from [graded_prop.md §3](#7-references) states that every morphism in the **broadcastable fragment** of `C` factors as:
$$g = [f_{\text{base}}, P] ; \rho$$
where $\rho$ is a **reindexing alignment** (a sequence of precompositions in `D`) and $[f_{\text{base}}, P]$ is **pointwise lifting** of a base morphism $f_{\text{base}}$. This factorization is unique up to equivalence in the broadcastable fragment.

Categorically, this says:

> The **broadcastable fragment** is the **free applicative / reindexing completion** of the base fragment.

This is a **normal-form theorem** (following [Plotkin & Power 2003](#7-references) on algebraic computational effects and normal forms): morphisms are **determined by their pointwise action on evaluations**. The points $p \in \text{El}(P)$ and evaluation transformations $\text{ev}_p$ form a **dense family** (in the sense of [MacLane & Moerdijk](#7-references), a separating family in category-theoretic terms), so two morphisms that agree on all pointwise evaluations are equal once `naperian_jointly_monic` is available. This relies on representability or a broadcast-generation normal form; it does not follow from strong monoidality of `El` alone.

#### Reformulated view: Naperian D-graded PROPs

A cleaner categorical statement integrates all the pieces:

> A **Naperian D-graded colored PROP** is a D-graded colored PROP (as defined in [graded_prop.md](#7-references)) equipped with:
> 1. A **finite, strong-monoidal point functor** $\text{El} : D \to \mathsf{FinSet}$ with $\text{El}(I_D) = \{*\}$. (The point functor must satisfy $\text{El}(P \otimes Q) \cong \text{El}(P) \times \text{El}(Q)$, with associativity and unit coherence.)
> 2. A separate **representability assumption** for lifted families: `NaperianFamily.lookup`/`tabulate` witness `F(X ⊛ P) ≃ (El P → F X)`. This is not derived from `NaperianAxis` or strong monoidality.
> 3. **Jointly separating lifted evaluations** (`naperian_jointly_monic`): For lifted objects `X ⊛ P`, the evaluation transformations $\text{ev}_p$ (for $p \in \text{El}(P)$) form a separating family. This is not the same as full **(Elem-C)** (the elementality axiom of [graded_prop.md §3.2](#7-references); Def 3.2 there is the *Weave*, not (Elem-C)), which separates arbitrary `C`-morphisms by `C`-points.
> 4. **Reindexings by precomposition**: For $\eta : P \to Q$ in `D`, reindexing $[X, \eta]$ is defined as postcomposition by the induced map `mapEl η : El(P) → El(Q)` on points, i.e., $[X, \eta](x)(p) = x(\text{mapEl}(\eta)(p))$. This is the contravariance of the action `act` from [graded_prop.md (Def 3.1.2)](#7-references), now indexed explicitly by elements.
> 5. **Reduction/fold as Kan extension**: Contractions can be *read as* **left Kan extensions** (or catamorphisms, for temporal structure) along index projections. This framing suggests the Beck-Chevalley proof target for lift-fold distributivity — but see the scope note below: this is intuition, not a scheduled construction. LeanNCD's contraction is a concrete `foldl`, and proving it equals a Kan extension is itself substantial work that no milestone currently funds.

Under this definition:

- `NaperianAxis` is the **data of a finite index object** with **explicit point enumeration** (realized by the point functor `El`) and coherent maps on points.
- `BroadcastJoin`, `ReindexAction`, and `PointwiseLift` require their own data/laws or concrete derivations from the chosen action and alignment strategy; they are not consequences of strong monoidality alone.
- `AxisReduce` is *described* by **left Kan extensions**; `TemporalGraded` adds **directed temporal structure** with monotone-morphism restriction (a constraint on allowed reindexings).
- The **compiler pipeline becomes an enforcer** of applicative/representability laws: each pass verifies that operations respect strong monoidality, representability, and the Naperian separating family.

> **Scope of formalization commitment (framing vs. structures we will build).** Not all of the
> categorical language above is on the critical path to reducing sorries, and the document should
> not be read as a commitment to formalize all of it. We separate:
>
> - **Load-bearing — will be built and proved against:** `NaperianAxis` (finite point functor
>   `El` with strong-monoidal coherence), `NaperianFamily` (`lookup`/`tabulate`), the
>   `ev_p`/`naperian_jointly_monic` separating family, and typed `StMatP` reindexing. These
>   directly gate the projected proof savings.
> - **Vocabulary/framing only — adopted for understanding, NOT scheduled as Lean constructions:**
>   the Kan-extension reading of `AxisReduce`, the Beck–Chevalley/Fubini reading of scan, the
>   free-applicative-completion reading of `broadcast_gen`, and the dense-family/topos language.
>   These are correct and useful for reasoning, but building, e.g., a left-Kan-extension-based
>   contraction in Lean and proving it equal to the current `foldl` is a large amount of Mathlib
>   machinery for an operation the evaluator already performs directly. Unless a concrete milestone
>   explicitly funds one of these, the contraction/scan proofs should target the **existing
>   operational definitions**, using the categorical reading only as intuition.

#### Why the categorical framing helps

The practical payoff (and its caveats and *projected* — not measured — sorry counts) is collected in [§6](#6-payoff-benefits-for-users-maintainers-and-formalization); this note records only *why the framing* buys those things:

1. **Coherence isos get standard proof targets.** If the mixins respect strong monoidality and the separate representability assumptions, then $\delta, \delta_0, \upsilon, \alpha$ have the canonical categorical proof shapes. Constructor-level quotient and extensionality proofs are still required in Lean — the framing supplies the *target*, not the proof.
2. **Some obligations move to reusable library facts.** Instead of re-proving reindexing/pointwise-lifting lemmas per morphism, Lean can lean on monoidal-functor properties (see [§5.1](#51-haskell--lean-translation-and-implementation-patterns) for which Mathlib structures actually exist). How much this reduces the sorry count is a hypothesis to validate, not a delivered result — see [§6](#6-payoff-benefits-for-users-maintainers-and-formalization).
3. **Extension is systematic.** A new graded operation (grouped reductions, higher-rank liftings) becomes an exercise in extending the point functor or index category, not inventing ad-hoc axioms.
4. **Dependent types encode the fibration.** Type-level ranks (as in `StMatP dom cod`) directly encode the grading's fiber structure, so dimension/rank invariants become unrepresentable violations rather than proof obligations.

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

(The law-backed *rewrites* these checks enable — reindex fusion, broadcast normalization, scan
batching, etc. — are listed once in [§4.1](#41-law-backed-rewrites-and-optimizations).)

---

## 4. Optimization and rewrite opportunities

### 4.1 Law-backed rewrites and optimizations

*Notation:*
- `f` denotes a **base morphism** — an operation in `C` not yet lifted by any shape (e.g., `add`, `exp`, `softmax`). It acts on unlifted, scalar or base-typed values.
- `lift P f` is the **explicit-shape version of applicative lifting** (`liftA`): it lifts a base morphism `f` to work pointwise over a `P`-shaped family, corresponding to `[f, P]` in the PROP notation. Unlike `liftA`, the shape `P` is explicit because the compiler needs to track shapes concretely. Nested lifts can be collapsed: `lift P (lift Q f) => lift (Q ⊗ P) f` expresses action associativity `(X ⊛ P) ⊛ Q ≅ X ⊛ (P ⊗ Q)` as a rewrite rule.
- `reindex η x` applies a `D`-morphism `η : P → Q` contravariantly to a `Q`-shaped value `x`, producing a `P`-shaped value — the `ReindexAction` mixin in concrete form, corresponding to precomposition of the index function by `η`.

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

> **Verify module paths before relying on them.** The paths below were checked against the
> repo's pinned toolchain (`leanprover/lean4:v4.30.0` + the manifest-pinned Mathlib) on
> 2026-07-01. Two Haskell classes — `Representable`/`Log` and `Foldable` — have **no
> Mathlib equivalent** and must be defined locally; do not assume Mathlib supplies them. The
> "inherit theorems from the categorical library" payoff (§6) is contingent on the structures
> that *do* exist (monoidal functors, `Equiv`, `Traversable`), not on these two.

| Haskell concept | Haskell signature | Lean equivalent | Location | Implementation note |
|---|---|---|---|---|
| **Applicative** | `class Applicative f` | `class Applicative (f : Type → Type)` | `Mathlib.Control.Applicative` | Base class for all lifting operations |
| `pure` | `pure : a → f a` | `Applicative.pure` or `pure` | core Lean (`Pure`) | Replication/broadcasting in our context |
| `<*>` | `(<*>) : f (a → b) → f a → f b` | `Seq.seq` or `<*>` | core Lean (`Seq`) | Pointwise binary composition |
| `liftA2` | `liftA2 : (a → b → c) → f a → f b → f c` | `liftA2` | core Lean (prelude) | Used in `PointwiseLift` implementation |
| **Representable/Naperian** | `class Representable f where type Log f; ...` | **No Mathlib class** — define locally | (local) | Mathlib's `CategoryTheory.RepresentedBy`/Yoneda captures *categorical* representability, but NOT the Haskell `Representable` with `Log`/`index`/`tabulate`. That interface must be defined in `LeanNCD/Core/Naperian.lean`. |
| `lookup` | `lookup : Log f → f a → a` | local field on `NaperianFamily` | (local) | Evaluation at a point; becomes type-level for `Axis` |
| `tabulate` | `tabulate : (Log f → a) → f a` | local field on `NaperianFamily` | (local) | Reconstruction; supplied by `NaperianFamily`, not by point enumeration alone |
| **Traversable** | `class Traversable t` | `class Traversable (t : Type u → Type u)` | `Mathlib.Control.Traversable.Basic` | Used in reduction/fold over axes |
| `traverse` | `traverse : Applicative f => (a → f b) → t a → f (t b)` | `Traversable.traverse` | `Mathlib.Control.Traversable.Basic` | Generalize pointwise ops to all traversable shapes |
| **Foldable** | `class Foldable t` | **No Mathlib class** — fold via `Traversable`/`List` | `Mathlib.Control.Fold`, `Mathlib.Data.List.Fold` | Mathlib has no `Foldable` class; reductions go through `Traversable` folds, `Control.Fold`, or plain `List.foldl`. Do not cite a `Foldable` class. |
| `fold` | `fold : Monoid a => t a → a` | `Traversable`/`Control.Fold` helpers | `Mathlib.Control.Fold` | Contract over monoidal aggregation |
| `foldl` | `foldl : (a → b → a) → a → t b → a` | `List.foldl` / `Traversable` | core Lean / `Mathlib.Control.Fold` | Used in scan iteration |
| `foldr` | `foldr : (a → b → b) → b → t a → b` | `List.foldr` / `Traversable` | core Lean / `Mathlib.Control.Fold` | Reverse-order reduction (for certain scan strategies) |
| **Array transpose** | `transpose :: f (g a) → g (f a)` | `Matrix.transpose : Matrix m n α → Matrix n m α` | `Mathlib.Data.Matrix.Basic` | **Dependent optimization:** build from the local representability isomorphism; encode axis permutation in type |
| **Isomorphism** | `(≅) : a → b` (type equivalence) | `Equiv : α ≃ β` or `Iso` in `CategoryTheory` | `Mathlib.Logic.Equiv.Basic` or `Mathlib.CategoryTheory.Iso` | Used for broadcast joins and reindex invertibility |
| **Monoidal functor** | `class MonoidalFunctor` | `class MonoidalFunctor` | `Mathlib.CategoryTheory.Monoidal.Functor` | Lift functor preserves monoidal structure |
| **Natural transformation** | `(⇒) : f → g` (between functors) | `NatTrans : F ⟹ G` | `Mathlib.CategoryTheory.NatTrans` | `ev_p` slices and evaluation transformations |
| **Symmetric monoidal** | `class SymmetricMonoidal` | `class SymmetricCategory` | `Mathlib.CategoryTheory.Monoidal.Braided.Basic` | Permutation laws on colors |
| **Free monoid** | `Free f` or `[a]` | `List α` (linear free monoid) or `FreeMonoid α` | `Mathlib.Data.List.Basic` or `Mathlib.GroupTheory.FreeMonoid` | Object lists in colored PROP |

### 5.2 Mixin-specific implementation strategies (with dependent-type optimization)

**NaperianAxis: Fixed-size representability**

The core typeclass (in `LeanNCD/Core/Naperian.lean`):
```lean
class NaperianAxis (D : Type) [ColoredPROP D] where
  El : D → Type   -- abstract point TYPE (no finiteness here — see note below)
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

`El P` is the *type* of coordinate tuples of shape `P`. **Finiteness is deliberately NOT a field
of `NaperianAxis`** (an earlier draft had `finite : ∀ P, Fintype (El P)`; it was removed). The
reason is foundational: `Axis.size` is symbolic (`Numeric`/`SizeExpr`) at compile time and does
not determine a `Fintype`, so a monolithic class carrying `Fintype` could not be instantiated for
`StObj` during symbolic compilation. Finiteness therefore lives in a **separate, post-solver**
class (`AxisPointData`/`FiniteNaperian`), supplied only once `inferAxisSizes` yields concrete
extents. See [NaperianTypingIntegrationPlan.md §3.2](NaperianTypingIntegrationPlan.md) for the
symbolic-vs-concrete class split. `point_hom` bridges to the existing `ev_p` API (converting a
coordinate into a global element `I_D → P`), and `mapEl` records the functorial action of
`D`-morphisms on points. `lookup` and `tabulate` — the Naperian isomorphism
`F(X ⊛ P) ≅ (El(P) → F(X))` — live in a separate `NaperianFamily` class, keeping `NaperianAxis`
focused on the index category `D` only. This requires an additional representability assumption —
that the lifted functor `act(−, P)` is represented by `El P` — which is not a consequence of
strong monoidality alone. This assumption is encoded in `NaperianFamily` as explicit
`lookup`/`tabulate` fields.

**Important:** the existing `Axis` structure (`name : Option String; size : Numeric`) does not carry point/enumeration data, because `size : Numeric` is symbolic and cannot determine a `Fintype` alone. The `NaperianAxis StObj` instance is provided via an auxiliary `AxisPointData` typeclass in `LeanNCD/Instances/StNaperian.lean` that supplies concrete point types externally — without mutating `Axis` or breaking the existing `StMat`, DSL, or quotient code. See [NaperianTypingIntegrationPlan.md §3](NaperianTypingIntegrationPlan.md) for the full instance code.

Benefit: `El P` is a type-level invariant and the `strong_monoidal` fields ensure the point functor commutes coherently with the monoidal structure of `D`. Finiteness (`Fintype (El P)`) is enforced by the separate post-solver class rather than bundled here, so the symbolic instance stays constructible.

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
-- StMatP is typed so domain/codomain rank is enforced at construction.
-- (`from` is a reserved keyword in Lean 4, so we use `dom`/`cod`.)
-- Rows are indexed by the codomain, columns by the domain — the rank
-- invariant is carried by the Matrix index types, so no side predicate is needed:
def StMatP (dom cod : StObj) : Type :=
  Matrix (Fin cod.length) (Fin dom.length) Coeff

def reindex (η : StMatP P Q) {X : C} : X ⊛ Q → X ⊛ P :=
  -- Type signature makes contravariance and codomain rank implicit
  sorry
```

> **Note — three illustrative `StMatP` lenses, one authoritative form.** This document shows
> `StMatP` two ways for pedagogy: the *indexed-function* view (`structure … where lookup : …`,
> under [Intuition](#intuition-behind-why-this-works)) and the *dependent-matrix* view (above).
> Neither is the real definition — the executable code uses a `structure` with a `coeffs` field
> (see [NaperianTypingIntegrationPlan.md §3.5](NaperianTypingIntegrationPlan.md)). Treat both here
> as lenses on the same object, not literal source.

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

### 5.3.1 Pass ordering: symbolic Naperian typing vs affine size inference

One implementation question is whether the existing affine domain/size solver should run
before or after "Naperian typing." The answer is: **it depends on which layer of
Naperian structure we mean**.

The current LeanNCD pipeline already separates:

- **symbolic compilation** (`assignUIDs`, `unifyAxes`, `lowerArith`, `route`),
- **symbolic routed shapes** (`SizeExpr.var a.name` in `route`),
- **concrete size inference at evaluation time** (`inferAxisSizes` in
  `LeanNCD/Eval/Shape.lean`, called from `evalScheduled`),
- **concrete execution** (`evalPlain`, `evalScan`, scatter sizing/evaluation).

This separation should be preserved. In particular:

1. **Symbolic/law-level Naperian typing should happen before the affine size solver.**  
   This includes:
   - lowering each `IdxExpr` to a symbolic affine `St` row,
   - checking rank/codomain compatibility of reindexings,
   - constructing canonical `degree` objects and broadcast joins,
   - validating pointwise/reduction/scan discipline,
   - packaging these invariants in `ReindexAction`, `BroadcastJoin`, and related mixins.

   None of these require concrete axis extents; they only require canonical axis identity
   and symbolic shape structure.

2. **Concrete finite Naperian instantiation should happen after concrete sizes are known.**  
   This includes:
   - materializing finite point sets `El P`,
   - obtaining `Fintype` evidence for runtime-enumerable axes,
   - instantiating the coordinate data consumed by concrete `lookup`/`tabulate`,
   - running any point-enumerating evaluator or proof that depends on actual cardinalities.

   This stage cannot, in general, precede size inference because the current axis size data
   is symbolic (`Numeric` / `SizeExpr`) and does not by itself determine a `Fintype`.

So the implementation-guiding conclusion is:

> **Do not treat "Naperian typing" as a single pass.** Split it into:
> (a) a **symbolic typed reindexing/law-checking layer** before size solving, and
> (b) a **concrete finite-point instantiation layer** after size solving.

For this repository, the recommended staging is:

```text
assignUIDs
resolveDecls
checkReadRanks
checkDtypes
unifyAxes
lowerArith
finalizeScans
splitNonlins
schedule
elaborateAffineReindexings      -- new symbolic affine/St elaboration pass
checkNaperianSymbolic           -- new law-level Naperian/reindexing checks
route                           -- consumes symbolic typed reindexings
inferAxisSizes                  -- existing runtime/value-level affine solver
instantiateConcreteNaperian     -- new finite-point instantiation step
evalPlain / evalScan / scatter evaluation
```

The crucial design consequence is that the affine solver is **not** replaced by dependent
typing. Instead, dependent typing should constrain the solver's **inputs** (well-formed
symbolic affine reindexings) and its **outputs** (size environments used to instantiate
concrete point data). This keeps the current padded-semantics solver intact while making
the surrounding compilation pipeline more type-safe.

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
- **Pass ordering guidance**: symbolic affine/Naperian elaboration and law checks should occur before `inferAxisSizes`, while concrete finite-point instantiation should occur after size solving.
- **The `NaperianAxis StObj` instance strategy**: using an auxiliary `AxisPointData` typeclass to supply concrete finite point types without mutating the existing `Axis` structure (necessary because `Axis.size : Numeric` is symbolic and does not determine a `Fintype` on its own).
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

#### Circularity hazard and a required early spike

The entire payoff routes through **`naperian_jointly_monic`** (the statement that lifted
`P`-indexed families are separated by their coordinate evaluations `ev_p`). This lemma has a
known **circularity hazard**: it is tempting to prove it *from* `broadcast_gen`, but
`broadcast_gen` is itself one of the obligations we hope Naperian typing helps discharge — so if
`naperian_jointly_monic` is used as an input to `broadcast_gen`, neither is actually proved.

Because so much depends on this lemma, it must be **de-risked before the typeclass scaffolding is
built**, not discovered late. The integration plan schedules this as
[Milestone 1.5 — circularity spike](NaperianTypingIntegrationPlan.md): establish an
acyclic proof route for `naperian_jointly_monic` (either a direct proof over `St` evaluations
independent of `broadcast_gen`, or an explicit statement of the extra assumption it requires)
*before* committing to Milestones 2+. If no acyclic route exists, the projected reduction below
does not hold and the plan should stop at the minimum viable integration.

#### Projected proof-effort reduction (a hypothesis to validate, not a measured result)

**Read this table as a projection, not a measurement.** The numbers below are *estimates* of
how the open proof obligations (from `SORRY_INVENTORY.md`, Milestones A–F) *could* shift IF the
enabling structure is built and the key lemma holds. The baseline counts (the "~16–20" of
milestone-relevant obligations — §5.5 rounds this to 16, this table to ~18–20; reconcile against
the *current* `SORRY_INVENTORY.md` before quoting either, since the raw `sorry` total across the
repo is larger and drifts) are indicative, not audited. Almost all of the projected savings sit in
the first row, and that row's savings are gated on two things that are **not yet done**: the
object-level `act` functor being defined, and `naperian_jointly_monic` being provable without
circularity (see [the circularity note](#circularity-hazard-and-a-required-early-spike) and
[NaperianTypingIntegrationPlan.md §5, Milestone 1.5](NaperianTypingIntegrationPlan.md)). Dependent
types reliably remove the *easy* sorries (dimension/rank checks); they do **not** remove the hard
ones (quotient-`Br` action, coherence isos, `broadcast_gen`). Treat the headline percentage as a
claim to be confirmed or refuted at Milestone 2, and do not cite it as an achieved benefit until
the enabling spike has landed.

| Proof family | Current | Projected | Impact |
|---|---:|---:|---|
| Representable/applicative laws (`instDGradedStBr` fields) | 10 | 2–4 | `δ`/`δ0`/`υ`/`α` get pointwise proof strategies from coherent `El` equivalences once `act` and extensionality are available; `act`, `broadcast_gen`, and coherence laws still require substantial work |
| Reindex identity/composition | scattered | ~0 | Dependent rank eliminates dimension checks |
| Weave uniqueness (`weave_unique`) | 1 | 0–1 | `weave_unique_naperian` (new theorem over `St` evaluations) avoids `brCancelPoint`; existing `weave_unique` via `Elemental Br` path unchanged |
| Scan/temporal laws | ~0 current | 0 future | Type-level causality barrier |
| Bridge/compilation (`E2b` bridge) | 6 | 1–2 | Well-formedness *by construction* in a hypothetical typed redesign. **Caveat:** the in-flight `compile_wellFormed` effort (`2026-06-26-multioutput-impl-plan.md`) is currently proving well-formedness is a *substantial* obligation, not free — so this row assumes a redesign that has not happened and is the most speculative in the table. |
| Point cancellation (`brCancelPoint`) | 1 | 0–1 | NbE/semantic Br rebuild; Naperian makes it less critical if consumers switch to `weave_unique_naperian` via `naperian_jointly_monic` (distinct from (Elem-C): this separates lifted morphisms by D-point evaluations, not C-morphisms by C-points) |
| **Subtotal** | **~18–20** | **3–6** | **PROJECTED 50–70% reduction — unvalidated. Holds only if (a) the object-level `act` is defined and (b) the Milestone 1.5 circularity spike gives an acyclic proof of `naperian_jointly_monic`. If either fails, expect little net reduction (the easy dimension/rank sorries still go, but they are few).** |

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

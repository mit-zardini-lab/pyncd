# Scan and Route in a Graded PROP

This note refines the `Scan` discussion in [graded_prop.md](graded_prop.md) and
[iteration.md](iteration.md). Its goal is narrow: identify the smallest extension to the
`D`-graded colored PROP formalism that gives `Scan` and `Route` categorical foundations.

Every symbol is introduced before use. Notation follows [graded_prop.md](graded_prop.md):
composition is diagrammatic (`f ; g` means "first `f`, then `g`"); `→` is morphism type;
`×` is category product; `⊗` is monoidal product inside a category; `⊛` is the lifted
object `X ⊛ P`; `ev_{p,X}` is evaluation at point `p` of a `D`-object `P` on a
`C`-object `X`; when `X` is clear from context, write `ev_p`. The symbol `ω` denotes
ordinary weave reindexing metadata; `ρ` a runtime route parameter; `η` a fixed structural
route (`D`-morphism); `σ` a destination relabeling isomorphism. Para parameter objects are
written `Θ`; `P`, `Q` are reserved for index shapes in `D` such as `St` shapes.

---

## Contents

1. [Why formulate Scan and Route axiomatically?](#1-why-formulate-scan-and-route-axiomatically)
2. [Base setting](#2-base-setting)
3. [The weave criterion](#3-the-weave-criterion)
4. [`Scan`: fixed temporal coupling](#4-scan-fixed-temporal-coupling)
   1. [Obstruction to being a weave](#41-obstruction-to-being-a-weave)
   2. [Data](#42-data)
   3. [Axioms](#43-axioms)
   4. [Algebra semantics](#44-algebra-semantics)
   5. [Consequences](#45-consequences)
   6. [Broader instances and axiom stability](#46-broader-instances-and-axiom-stability)
5. [`Route`: value-dependent indexing](#5-route-value-dependent-indexing)
   1. [Obstruction to being a weave](#51-obstruction-to-being-a-weave)
   2. [Para and Route signature](#52-para-and-route-signature)
   3. [Axioms](#53-axioms)
   4. [Algebra semantics](#54-algebra-semantics)
   5. [Weighted multi-destination routing](#55-weighted-multi-destination-routing)
   6. [Consequences](#56-consequences)
   7. [Broader instances and axiom stability](#57-broader-instances-and-axiom-stability)
6. [Unified non-weave generator axioms](#6-unified-non-weave-generator-axioms)
   1. [Schema](#61-schema)
   2. [Axioms](#62-axioms)
   3. [What factors out](#63-what-factors-out)
   4. [Relation to categorical deep learning](#64-relation-to-categorical-deep-learning)
7. [What is deliberately not included](#7-what-is-deliberately-not-included)
8. [Lean formalization sketch](#8-lean-formalization-sketch)
   1. [DGradedPROP](#81-dgradedprop)
   2. [Routing envelope](#82-routing-envelope)
   3. [Para](#83-para)
   4. [DAlgebra](#84-dalgebra)
   5. [Formalizing Scan](#85-formalizing-scan)
   6. [Formalizing Route](#86-formalizing-route)
   7. [Formalizing the unified schema](#87-formalizing-the-unified-schema)
   8. [Proof targets](#88-proof-targets)
9. [Summary](#9-summary)
10. [Appendix A: pyncd code correspondence for `Scan`](#appendix-a-pyncd-code-correspondence-for-scan)

---

## 1. Why formulate Scan and Route axiomatically?

The ordinary `D`-graded PROP already explains a large class of pyncd programs: base
operations can be lifted over index shapes, reindexed by structural `D`-morphisms, and
compiled by algebras such as `construct()`. `Scan` and `Route` are important precisely
because they sit just outside that ordinary weave fragment.

**`Scan`** covers any computation where a fixed ordered axis threads state from one point
to the next: RNNs, state-space models, weight-tied depth, iterative refinement, learned
optimizers, fixed-round message passing, and sweep-style tensor-network algorithms. The
defining feature is that computing the output at step `l + 1` requires the output at step
`l`, so no pointwise lift over the temporal axis suffices.

**`Route`** covers value-dependent dispatch: sparse mixture-of-experts inference, dynamic
token-to-memory routing, hard attention, adapter selection, per-sample graph structure, and
stochastic resampling. The defining feature is that the destination index is a runtime
value, not a fixed `D`-morphism known at graph-construction time.

An axiomatic formulation is useful for four reasons.

1. It separates *what an operator means* from *how an implementation runs it*. A
   sequential loop and a checkpointed loop implement the same `Scan` when they satisfy the
   same finite-recurrence axioms. A dense dispatch kernel and a sparse dispatch kernel
   implement the same `Route` when they satisfy the same route laws.

2. It identifies the exact obstruction to being a weave. `Scan` fails because neighboring
   temporal points are coupled. `Route` fails because the reindexing map is a runtime
   value rather than a fixed `D`-morphism. The two failures are distinct.

3. Axioms give reusable compiler laws: prefix restriction, vectorization over orthogonal
   axes, fixed-route specialization, destination relabeling, and algebra preservation
   become theorems or checkable fields rather than ad hoc backend conventions.

4. An axiomatic boundary keeps the core small. The goal is not a general theory of loops,
   effects, routing, and dynamic shapes — only the equations that make finite scan and
   value-dependent route compositional inside the existing categorical framework.

This aligns with Gavranovic et al. (arXiv:2402.15332): architectures are presented by
generators and equations, while implementations are algebras, often in a `Para`-style
category of parameterized maps. pyncd adds a `D`-graded index action, so the Scan and
Route equations must specify how recurrence or value-dependent dispatch interacts with
lifted tensor shapes.

---

## 2. Base setting

**Definition 2.1 (Colored PROP).** A *colored PROP* is a strict symmetric monoidal
category whose objects are finite lists of colors. The monoidal product `⊗` is list
juxtaposition, the unit `I` is the empty list, and symmetry maps permit wire permutation.

Let `D` be the **index PROP**. Its objects are index shapes and its morphisms are
reindexings. In pyncd's implemented instance, `D = St`: objects are axis tuples, morphisms
are affine stride maps.

Let `C` be the **operation PROP**. Its objects are lists of typed arrays, morphisms are
tensor operations. In pyncd, `C = Br`, the broadcasted category.

**Definition 2.2 (`D`-graded colored PROP).** A *`D`-graded colored PROP* consists of
`C`, `D`, and the following structure.

1. A *shape map* `sh : colors(C) → Ob(D)`, sending each `C`-color to its underlying
   `D`-shape. For `C = Br` and `D = St`, the color `[a, A]` has `sh([a, A]) = A`.

2. A *right action* `act : C × D^op → C`. Write `X ⊛ P` for `act(X, P)`. On morphisms:
   - `[f, P] : X ⊛ P → Y ⊛ P` for a `C`-morphism `f : X → Y` (covariant in `C`);
   - `[X, η] : X ⊛ Q → X ⊛ P` for a `D`-morphism `η : P → Q` (contravariant in `D`).

3. Coherence laws: `act` is functorial and distributes over `⊗`. The key consequence is
   `[f ; g, P] = [f, P] ; [g, P]`.

**Definition 2.3 (Point and evaluation).** A *point* of a `D`-object `P` is a morphism
`p : I_D → P`. The action gives an *evaluation map*

```text
ev_{p,X} : X ⊛ P → X
```

derived by applying `act(id_X, p^op)` and composing with the unit isomorphism
`υ_X : X ⊛ I_D ≅ X`. Equivalently, `ev_{p,X}` is `[X, p]` followed by `υ_X`. When `X`
is clear, write `ev_p`.

**Definition 2.4 (Target actegory and algebra).** A *target actegory* is a symmetric
monoidal category `V` with a right `D`-action `⊛_V`. An *algebra* of `C` in `V` is a
strong symmetric monoidal functor `F : C → V` preserving the `D`-action up to coherent
isomorphism. In pyncd, `F = construct()`.

---

## 3. The weave criterion

**Definition 3.1 (Weave).** A morphism `g : X → Y` is a *weave* over a `D`-object `P`
when it factors as

```text
g = α ; [f, P] ; ω
```

where `f : X0 → Y0` is a base operation in `C`, `α : X → X0 ⊛ P` is structural input
bookkeeping, and `ω : Y0 ⊛ P → Y` is structural output bookkeeping. Both `α` and `ω` are
assembled from `D`-reindexing maps, associativity/unit isomorphisms, and symmetry
maps.[^per-point]

[^per-point]: **Per-point.** Fixing a point `p : I_D → P` and evaluating `X0 ⊛ P` at `p`
    via `ev_{p,X0} : X0 ⊛ P → X0` gives the tensor-wire type at one point. In pyncd with
    `C = Br`, if `P` is a batch axis `b` and the full input has axes `(b, i)`, fixing `b`
    leaves local axis `(i)`.

In the pyncd code, `α` and `ω` are represented by `Broadcasted.input_weaves`,
`Broadcasted.output_weaves`, and `reindexings` metadata. `Broadcasted.dom()` computes each
input object by `input_weave.imprint_to_degree(reindexing.cod())`; `Broadcasted.cod()`
computes each output object by `output_weave.imprint_to_degree(self.degree())`. The
PyTorch compiler uses the same metadata in `broadcast_vmap(...)` to choose vmap dimensions.

**Definition 3.2 (Point naturality).** A morphism satisfies *point naturality* over `P`
when, for every point `p : I_D → P`,

```text
[f, P] ; ev_{p,Y0} = ev_{p,X0} ; f.
```

This equation is Eq. 3 of [graded_prop.md](graded_prop.md) and holds automatically for
any lifted operation by functoriality of `act`. **(Elem-C)** (elements jointly separate
morphisms) then pins down the weave.

> **Weave test.** A morphism is a weave over `P` exactly when it is pointwise independent
> over `P`-coordinates: computing the output at any point `p` of `P` requires only the
> input at that same point.

---

## 4. `Scan`: fixed temporal coupling

### 4.1 Obstruction to being a weave

A *recurrence* is a computation

```text
H[l + 1] = step(H[l], U[l])
```

where `l` is the time or iteration index, `H[l]` is the state at step `l`, and `U[l]` is
the external input slice at step `l`. `Scan` computes the full history
`H[0], H[1], ..., H[N]` from an initial state `H[0]` and per-step inputs
`U[0], ..., U[N - 1]`.

`Scan` fails point naturality over the temporal axis: computing the output at time `l + 1`
requires the output at time `l`, so evaluating at a point does not commute with the
operation:

```text
Scan ; ev_{l+1,H}  ≠  ev_{l,U} ; step.
```

The obstruction is **not** data-dependence of the time reindexing — time is fixed and
structural. The obstruction is that the morphism *couples different positions of the time
axis*. This is different from `Route`, where the problem is a runtime-valued reindexing.

The examples in [future_ideas.md](future_ideas.md), [iteration.md](iteration.md), and
[graded_prop.md](graded_prop.md) show that this pattern is broader than RNN-style sequence
models. `Scan` is the right abstraction whenever a fixed structural axis orders repeated
applications of a state update, and each point depends on earlier points along that axis.

### 4.2 Data

**Definition 4.1 (Temporal object).** For each natural number `N`, let `L_N` denote the
`D`-object representing `{0, 1, ..., N - 1}`. In `St`, `L_N` is an axis of size `N`. For
`0 ≤ k < N`, let `pt_k : I_D → L_N` be the point selecting index `k`. For `0 ≤ M ≤ N`,
let

```text
ι_{M,N} : L_M → L_N
```

be the *prefix inclusion*, sending `{0, ..., M - 1}` into the first `M` points of
`{0, ..., N - 1}`. In `St` this is the affine inclusion of a prefix axis into a longer
axis.

**Definition 4.2 (Scan data).** Fix `C`-objects `H` (state at one step), `U` (per-step
external input), and `X` (input from which the initial state is computed). Fix
`C`-morphisms:

- `init : X → H` — computes the initial state;
- `step : H ⊗ U → H` — computes the next state from the current state and current input.

The sequence of per-step inputs is `U ⊛ L_N`.

**Definition 4.3 (Scan signature).** For each natural number `N`, `Scan` is a
distinguished generator:

```text
Scan_N(step, init) : X ⊗ (U ⊛ L_N) → H ⊛ L_{N+1}.
```

The domain pairs the initial-state source `X` with the `N` per-step inputs `U ⊛ L_N`. The
codomain is the history of `N + 1` states: the initial state `H[0]` plus one state after
each step.

In pyncd, this generator is the `Scan` dataclass in `data_structure/TensorDSL.py`; see
[Appendix A](#appendix-a-pyncd-code-correspondence-for-scan).

### 4.3 Axioms

The pointwise Scan laws are stated in an **ambient routing envelope** — a layer of
structural operations supplying projections and pairings for the external scan inputs,
matching the role played by `ThreadedComposed` in pyncd. A *projection*
`π_A : A ⊗ B → A` keeps the `A` component of a product input. A *pairing*
`⟨a, b⟩ : Z → A ⊗ B` feeds the same source `Z` to both morphisms `a : Z → A` and
`b : Z → B`, then juxtaposes their outputs. These belong to the routing envelope, not to
the bare linear PROP.

**(Scan-base)** The first output slice is the initial state. With `pt_0 : I_D → L_{N+1}`
selecting time `0` and `π_X : X ⊗ (U ⊛ L_N) → X` the projection keeping `X`:

```text
Scan_N(step, init) ; ev_{pt_0,H}
  =
π_X ; init.
```

**(Scan-step)** For `0 ≤ k < N`, with `pt_k : I_D → L_N` selecting input time `k`, and
`pt'_k, pt'_{k+1} : I_D → L_{N+1}` selecting output times `k` and `k + 1`, and
`π_U : X ⊗ (U ⊛ L_N) → U ⊛ L_N` keeping the input sequence:

```text
Scan_N(step, init) ; ev_{pt'_{k+1},H}
  =
⟨
  Scan_N(step, init) ; ev_{pt'_k,H},
  π_U ; ev_{pt_k,U}
⟩
; step.
```

This is a finite catamorphism law, requiring no fixpoint because `N` is finite.

**(Scan-lift)** Let `P` be a `D`-object with axes disjoint from those of `L_N`
(*orthogonal* to `L_N`). Then:

```text
[Scan_N(step, init), P]
  ≅
Scan_N([step, P], [init, P]).
```

The `≅` denotes a canonical isomorphism (definitional equality in a strict
implementation). This axiom is the categorical basis for vectorizing a recurrence across
batch coordinates.

### 4.4 Algebra semantics

Let `F : C → V` be an algebra into a target actegory. Supporting `Scan` requires:

**(Scan-algebra)** `F(Scan_N(step, init)) = fold_N(F(step), F(init))`,

where `fold_N(F(step), F(init))` is the concrete finite loop applying `F(step)` exactly
`N` times and returning the `N + 1`-state history.

Different implementations of `fold_N` may represent the same `V`-morphism:

| Implementation | Extra condition |
| --- | --- |
| Eager sequential loop | Always valid |
| Checkpointed loop | Step is pure |
| Batched loop over orthogonal axis | **(Scan-lift)** applies |
| Associative (parallel-prefix) scan | Step factors through a monoid |

A *monoid* means an object `M` with an associative multiplication `μ : M ⊗ M → M` and a
unit `e : I → M`. A step factors through a monoid when each step can be represented as a
monoid element and sequential execution is monoid multiplication; then a parallel-prefix
algorithm computes the scan.

### 4.5 Consequences

**Proposition 4.4 (Prefix restriction).** For `0 ≤ M ≤ N`,

```text
Scan_N(step, init) ; [H, ι_{M+1,N+1}]
  =
(id_X ⊗ [U, ι_{M,N}]) ; Scan_M(step, init).
```

Here `[H, ι_{M+1,N+1}] : H ⊛ L_{N+1} → H ⊛ L_{M+1}` restricts the output history, and
`[U, ι_{M,N}] : U ⊛ L_N → U ⊛ L_M` restricts the input sequence to its first `M`
entries. Restricting both inputs and output history of the `N`-step scan gives the
`M`-step scan. *Proof:* induction on `M` from **(Scan-base)** and **(Scan-step)**.

**Proposition 4.5 (Non-weave boundary).** The Scan axioms do not add point naturality for
`Scan` along the temporal axis. They specify how temporal points depend on predecessor
temporal points. `Scan` remains outside the image of `act(−, L_N)` — it is a generator
with fold laws, not a disguised weave.

**Proposition 4.6 (Scan fusion via algebra homomorphism).** An *algebra homomorphism*
from `step : H ⊗ U → H` to `step' : H' ⊗ U → H'` is a morphism `h : H → H'` satisfying

```text
(h ⊗ id_U) ; step' = step ; h.
```

When this holds, applying `h` to the history produced by `Scan_N(step, init)` agrees with
running `Scan_N(step', init ; h)`. Two scans with the same `init` and extensionally equal
`step` morphisms are equal by **(Scan-base)** and **(Scan-step)**.

### 4.6 Broader instances and axiom stability

The general Scan pattern: *fixed ordered axis + state threaded from one point to the
next.*

| Context | State `H` | Step input `U` |
| --- | --- | --- |
| Sequence modeling (RNNs, SSMs, autoregressive hidden-state) | hidden state | token/input slice |
| Weight-tied depth (Universal Transformer, ALBERT-style, iterative refinement) | layer representation | per-depth conditioning |
| Numerical and optimization loops (learned optimizers, fixed-point iterations) | iterate / optimizer state | gradients, residuals, forcing terms |
| Fixed-round message passing (GNN or belief-propagation) | node/edge features | graph-local messages |
| Tensor-network sweeps (DMRG/TEBD-style) | local variational state | local tensor or gate data |
| Search and inference (beam search, SMC, particle filters) | beam/particle state | proposal, likelihood, or accept signal |

These instances do **not** change the minimal Scan axioms when the recurrence is
first-order and fixed-length. What changes is the interpretation of `L_N`, `H`, `U`, or
the target algebra implementing `fold_N`.

Some variants require refinements:

1. *Multi-state or coupled scans*: replace `H` by a product of state objects, and use the
   same base/step/lift laws on that product.
2. *Multi-step lookback*: replace `H` by a window object such as `H ⊗ H`, or add several
   base cases and a higher-arity step law.
3. *Non-forward schedules*: replace `L_N` with a richer schedule object and corresponding
   predecessor maps.
4. *Dynamic or unbounded length*: requires a dependent, guarded, or coalgebraic account —
   finite `Scan_N` does not cover this.
5. *Search/SMC/speculative decoding*: usually compound — a Scan-like propagation step
   followed by a Route-like prune or resample step.

---

## 5. `Route`: value-dependent indexing

`Route` is not a recurrence. It solves a different problem: selecting where data goes
based on runtime values. The same obstruction appears whenever a computation must choose a
destination, handler, edge, block, slot, memory location, branch, or sample ancestor from
runtime values.

### 5.1 Obstruction to being a weave

A *fixed dense routed computation* sends every item to every destination via a fixed axis
structure `I ⊗ E`, where `I` is an item index and `E` is a destination set. This is a
weave when `E` is fixed structure.

A *dynamic routed computation* instead computes a runtime routing function

```text
ρ : I → E
```

from data-dependent values. Because `ρ` depends on tensor values, it is not a fixed
`D`-morphism; no reindexing `η : I → E` is available at graph-construction time.

Comparing the two obstructions:

- `Scan` has fixed reindexing but coupled temporal dependence.
- `Route` has uncoupled per-item execution but data-dependent reindexing.

The map `ρ` is a **value-level parameter** — data available to the concrete algebra (such
as a tensor of route decisions), not symbolic structure available to the graded PROP.

**The St/Br instance.** In the standard setting `D = St`, `C = Br`, a `D`-morphism is
an affine map with integer coefficients fixed at graph-construction time. The Route
obstruction is therefore precisely the gap between static affine access patterns and
runtime-determined indexing. A static gather with a fixed index (e.g., `h[..., 3]` or a
strided slice) is an ordinary `St` morphism — a weave. Dynamic gather (`X[idx]` where
`idx` is a runtime integer tensor) is the simplest Route instance: `I` is the output
position axis, `E` is the source axis, `handler_e = id`, and `ρ(i) = idx[i]`. Dense
attention is also a weave: the key and value axes are fixed structure, and the softmax
weights are tensor values — they distribute mass across all keys but do not select among
them as destinations. Hard (argmax) attention is Route: the argmax is a runtime choice of
which key coordinate to read.

### 5.2 Para and Route signature

To model value-level parameters, use **Para**. The category `Para(C)` has the same
objects as `C`. A morphism from `A` to `B` is a pair `(Θ, f : Θ ⊗ A → B)` where `Θ` is
a parameter object and `f` is a `C`-morphism. Composition threads parameters by tensoring
parameter objects together.

**Definition 5.1 (Route parameter object).** For single-destination routing, let
`R_{I,E}` be a parameter object whose concrete elements are functions from items to
destinations. An element `ρ : I → E` of `R_{I,E}` assigns each item `i` a destination
`ρ(i)`.

**Definition 5.2 (Handler family).** Let `A` be a `C`-object (one item feature vector)
and `B` a `C`-object (one item output vector). A *handler family* indexed by `E` is a
family of `C`-morphisms

```text
{handler_e : A → B}_{e : I_D → E},
```

or equivalently a single lifted morphism `handler : A ⊛ E → B ⊛ E` such that
`handler ; ev_{e,B} = ev_{e,A} ; handler_e` for each destination `e`.

**Definition 5.3 (Route signature).** For index object `I`, destination object `E`, item
object `A`, output object `B`, and handler family `{handler_e}`, add a parameterized
generator in `Para(C)`:

```text
Route_{I,E}({handler_e}) : A ⊛ I → B ⊛ I,
```

with parameter object `R_{I,E}`. As an ordinary `C`-morphism under the Para encoding:

```text
Route^C_{I,E}({handler_e}) : R_{I,E} ⊗ (A ⊛ I) → B ⊛ I.
```

### 5.3 Axioms

Write `Route_ρ({handler_e})` for `Route` specialized at a concrete value `ρ` of the
parameter object.

**(Route-dispatch)** For each item point `i : I_D → I` and route value `ρ` with chosen
destination `ρ(i) : I_D → E`:

```text
Route_ρ({handler_e}) ; ev_{i,B}
  =
ev_{i,A} ; handler_{ρ(i)}.
```

This is the Route analogue of **(Scan-step)**. For Scan, the point law relates neighboring
time points. For Route, the point law is item-local but depends on a value-level
parameter.

**(Route-static)** When the route is induced by a fixed `D`-morphism `η : I → E` (with
corresponding structural parameter value `ρ_η`):

```text
Route_{ρ_η}({handler_e}) ; ev_{i,B}
  =
ev_{i,A} ; handler_{η(i)}.
```

Because points of `I` jointly separate morphisms, this pointwise equality determines the
whole morphism: a static route is recoverable from ordinary `D`-graded structure.

**(Route-lift)** Let `P` be a `D`-object with axes disjoint from those of `I` and `E`.
Write `ρ ⊛ P` for the route that chooses, for each pair `(i, p)`, the same destination as
`ρ` chooses for `i`:

```text
[Route_ρ({handler_e}), P]
  ≅
Route_{ρ ⊛ P}({[handler_e, P]}).
```

**(Route-relabel)** Let `σ : E → E` be an isomorphism in `D`. Define the relabeled route
`σ ∘ ρ` by applying `ρ` then `σ`, and the relabeled family `{handler^σ_e}` by
`handler^σ_{σ(e)} = handler_e` (equivalently `handler^σ_e = handler_{σ^{-1}(e)}`). Then:

```text
Route_ρ({handler_e})
  =
Route_{σ ∘ ρ}({handler^σ_e}).
```

This prevents `Route` from depending on arbitrary names or storage order of destinations:
only the pairing between route decisions and handler implementations matters.

### 5.4 Algebra semantics

**(Route-algebra)** For an algebra `F : C → V`, the *Route preservation law* is:

```text
F(Route_ρ({handler_e}))(x)_i
  =
F(handler_{ρ(i)})(x_i).
```

Here `x` is a concrete value of `F(A ⊛ I)` and `x_i` is the slice at item `i`. This is
the runtime dispatch semantics, not expressible as a fixed `D`-morphism unless `ρ` is
known before graph construction.

### 5.5 Weighted multi-destination routing

Single-destination routing is the minimal categorical core. Weighted multi-destination
routing adds two pieces of structure.

Replace `R_{I,E}` by `R_{I,E,K,W}`, whose concrete values assign to each item `i` a
finite list

```text
((e_{i,1}, w_{i,1}), ..., (e_{i,K}, w_{i,K}))
```

of destination–weight pairs. Require `B` to carry a finite weighted-sum operation
`combine_K : (W ⊗ B)^{⊗ K} → B`. The axiom **(Route-dispatch)** becomes:

```text
Route_ρ({handler_e}) ; ev_{i,B}
  =
⟨
  w_{i,1} ⊗ (ev_{i,A} ; handler_{e_{i,1}}),
  ...,
  w_{i,K} ⊗ (ev_{i,A} ; handler_{e_{i,K}})
⟩
; combine_K.
```

The multi-destination variant inherits **(Route-static)**, **(Route-lift)**,
**(Route-relabel)**, and **(Route-algebra)**, with the selected handler replaced by the
weighted combination of selected handlers.

### 5.6 Consequences

**Proposition 5.4 (Static routes are ordinary structure).** When `ρ` is induced by a
fixed `D`-morphism `η : I → E`, **(Route-static)** identifies `Route_ρ` with the ordinary
graded construction. Dense or statically routed computations remain in the weave fragment.

**Proposition 5.5 (Dynamic routes are item-local but not structural).** The dispatch
axiom says each output item depends only on the corresponding input item and the handler
chosen by `ρ`. This is not ordinary point naturality, because `ρ(i)` is read from a
value-level parameter at runtime.

**Proposition 5.6 (Destination names are irrelevant).** **(Route-relabel)** implies that
optimization passes may reorder, pack, shard, or rename destinations as long as they
transform route labels and handler implementations together. The observable morphism is
unchanged.

**Proposition 5.7 (Independent batching is safe).** **(Route-lift)** implies that
vectorizing `Route` across an independent axis `P` does not change semantics — the exact
Route counterpart of the Scan batched-loop law.

Scan and Route therefore need different laws: Scan needs finite-fold laws for coupled
temporal dependence; Route needs Para laws for value-dependent indexing.

### 5.7 Broader instances and axiom stability

The general Route pattern: *runtime value decides which destination coordinate is read or
written.*

| Context | Item `I` | Destination `E` | `handler_e` | Route parameter `ρ` |
| --- | --- | --- | --- | --- |
| Dynamic gather (`X[idx]`, `torch.gather`) | output position axis | source axis | identity | `ρ(i) = idx[i]` (runtime integer tensor) |
| Sparse MoE (top-1 or top-k) | token axis | expert axis | expert FFN | router argmax or top-k |
| Adapter-bank / multi-LoRA serving | request axis | adapter axis | adapter forward pass | model/request → selected adapter |
| Hard (argmax) attention | query axis | key/value slot axis | value lookup | attention argmax |
| Adaptive filtering (spatially varying convolution) | spatial position axis | filter-bank axis | convolution with filter `e` | per-position filter choice |
| Conditional computation / Mixture of Depths | token axis | computation-depth axis | depth-`e` transformer block | difficulty predictor → active depth |
| Per-sample GNNs, meshes, molecules | sample/node axis | edge or neighbor-choice axis | message or feature lookup | sample/node → active neighborhood |
| Learned pooling, clustering, slot/capsule assignment | item axis | block, cluster, or slot axis | aggregation into slot `e` | item → learned block/slot |
| Particle filters, SMC, stochastic resampling | particle axis | ancestor axis | identity (copy ancestor state) | resampling draw |
| Beam search, speculative decoding accept/reject | draft token axis | beam or accept-branch axis | identity or rewrite | verification outcome |

These instances do **not** change the minimal deterministic Route axioms. What changes is
the interpretation of `E`, the shape of the parameter object, or the target algebra.

**The identity-handler case.** When `handler_e = id` for all `e`, Route reduces to
dynamic gather: `output[i] = input[ρ(i)]`. This is the simplest Route instance and the
one most directly visible in PyTorch (`torch.gather`, `X[idx]` advanced indexing,
nearest-neighbor retrieval with runtime keys). A static gather with a fixed index is a
plain `St` morphism and remains in the weave fragment; the Route obstruction appears
exactly when the index tensor itself is computed at runtime.

**What is not Route in the St/Br setting.** Dense softmax attention is a weave: the key
and value axes are fixed structure, and the softmax weights are tensor values that
distribute computation across all keys uniformly — no coordinate is selected at runtime.
Standard convolution, strided or dilated access, and any operation with a statically
known sparsity mask are similarly weaves (fixed `St` reindexings). Adaptive filtering is
Route only when the choice of *which* filter to apply at each position is a runtime
decision, not when the filter coefficients are merely learned parameters.

**Tensor network applications.** Standard TN algorithms — DMRG, TEBD, MERA,
belief propagation on fixed graphs — have static contraction topology: every axis
contraction is a fixed `St` morphism and the computation is a weave. Route does not
appear in this repertoire. The two exceptions are: (1) adaptive contraction path
selection, where a runtime measurement (singular value spectrum, entanglement entropy)
determines which pair of tensors to contract next — here `I` is the site axis, `E` is
the set of candidate contraction moves, and `ρ` is the runtime path decision; and (2)
batch-heterogeneous graph contractions (e.g., per-sample molecular graphs), which reduce
to the per-sample GNN case in the table above. Both are niche relative to the standard
fixed-graph TN algorithms.

---

## 6. Unified non-weave generator axioms

`Scan` and `Route` are both **non-weave generators**: distinguished operations whose
behavior is determined by pointwise equations that are not the ordinary naturality law for
a lifted base operation.

| Aspect | `Scan` | `Route` |
| --- | --- | --- |
| Weave obstruction | Fixed temporal axis; output points coupled across time | Item axis fixed; destination choice is a runtime value |
| Working category | `C` | `Para(C)`, specializing to `C` after fixing `ρ` |
| Output shape | `H ⊛ L_{N+1}` | `B ⊛ I` |
| Local law | predecessor state + current input | current input + runtime-selected handler |
| Output-point dependency | well-founded predecessor relation on time | none (empty relation) |
| Ordinary-structure boundary | base point uses `init`; prefix restriction induced by `D` | static `ρ_η` agrees with fixed `η : I → E` |
| Orthogonal batching | batch commutes with scan | batch commutes with route |
| Harmless symmetries | only order-preserving time automorphisms | destination relabeling |
| Algebra preservation | compile to `fold_N` | compile to dispatch |

### 6.1 Schema

Let `K` be the category in which the generator lives: `C` for Scan, `Para(C)` for Route
before specializing `ρ`.

**Definition 6.1 (Non-weave generator schema).** A *non-weave generator schema* consists
of:

1. An *output index object* `S` in `D`, with points `s : I_D → S`.

2. A *point-dependency relation* `t ≺ s` on points of `S`, which is **well-founded**
   (no infinite chain `... ≺ s_2 ≺ s_1 ≺ s_0`). For `Route`, this relation is empty.
   For `Scan`, `pt'_k ≺ pt'_{k+1}`.

3. A *full domain object* `Z` in `K`. For Scan: `Z = X ⊗ (U ⊛ L_N)`. For specialized
   Route: `Z = A ⊛ I`.

4. A *point output object* `Y` in `C`, so the full output is `Y ⊛ S`.

5. A generator `G : Z → Y ⊛ S` in `K`.

6. For each point `s : I_D → S`, a *local evaluator* `Φ_s` computing the output slice at
   `s` from external input data and from previously defined output slices `G ; ev_{t,Y}`
   for points `t ≺ s`. Local evaluators may use projections and pairings from the routing
   envelope.

### 6.2 Axioms

**(NW-point)** For every point `s : I_D → S`:

```text
G ; ev_{s,Y} = Φ_s({G ; ev_{t,Y}}_{t ≺ s}, external data).
```

This specializes to **(Scan-base)** and **(Scan-step)** (with `Φ_{pt_0} = π_X ; init`
and `Φ_{pt'_{k+1}}` using predecessor `pt'_k`), and to **(Route-dispatch)** (with empty
dependency and `Φ_i = ev_{i,A} ; handler_{ρ(i)}`).

**(NW-boundary)** When the non-weave obstruction degenerates to ordinary structure, the
generator agrees with the ordinary `D`-graded construction. For Scan this is the base
point **(Scan-base)**; for Route it is the static-route case **(Route-static)**.

**(NW-lift)** For `P` orthogonal to `S` and to all parameter index objects:
`[G, P] ≅ G^P`, where `G^P` is the generator built from lifted local data —
`Scan_N([step, P], [init, P])` or `Route_{ρ ⊛ P}({[handler_e, P]})`.

**(NW-relabel)** For an isomorphism `σ : S → S` preserving `≺` and transporting each
`Φ_s` to `Φ_{σ(s)}`, relabeling output points by `σ` does not change the observable
computation.

**(NW-algebra)** For an algebra `F : C → V`, `F(G) = semantic_G`, where
`semantic_G = fold_N(F(step), F(init))` for Scan and
`semantic_G(x)_i = F(handler_{ρ(i)})(x_i)` for Route.

### 6.3 What factors out

The shared structure is not "Scan and Route are the same operation" but the *interface for
adding generators that fail the ordinary weave criterion*:

1. a point-indexed output `Y ⊛ S`;
2. a well-founded point-presentation law **(NW-point)**;
3. a boundary case agreeing with ordinary `D`-graded structure **(NW-boundary)**;
4. orthogonal lift distribution **(NW-lift)**;
5. relabeling equivariance for symmetries preserving local laws **(NW-relabel)**;
6. algebra preservation **(NW-algebra)**.

The operation-specific part is the local evaluator `Φ_s`:

```text
Scan:  Φ_s may read earlier output slices (via the predecessor relation).
Route: Φ_s may read value-level routing parameters.
```

A future implementation or Lean development can share this scaffolding while keeping the
local equations specific to the generator being added.

### 6.4 Relation to categorical deep learning

The unified schema is a shaped-tensor specialization of the categorical deep learning
program (Gavranovic et al.): a non-weave generator is a new architecture generator, local
equations are its relations, and `F(G) = semantic_G` is the functorial implementation
condition.

For `Scan`, categorical deep learning models recurrent neural networks as algebras or
coalgebras for endofunctors lifted to `Para`; finite unrolling is an algebra homomorphism.
pyncd replaces the abstract list object with a finite `D`-index object `L_N`, adds point
evaluation `ev_{pt_k,H}`, and states an explicit orthogonal lift law:

```text
CDL recurrence algebra + finite D-indexed shape = pyncd Scan.
```

For `Route`, `Para` supplies the natural home for runtime parameters. The route parameter
`ρ`, handler family, and destination relabeling law are parameterized architecture data
plus a harmless reparameterization symmetry. pyncd adds the static index layer — item axis
`I`, destination object `E`, fixed-route specialization, and orthogonal batching:

```text
CDL parameterized maps + D-graded value-dependent indexing = pyncd Route.
```

The connection is useful but does not replace the present axioms. Ordinary categorical
deep learning explains why generators, equations, `Para`, algebra homomorphisms, and
reparameterizations are the right interfaces. The pyncd axioms add the `D`-graded shape
discipline: point evaluation, failure of ordinary weave naturality, orthogonal lift
distribution, and static-route boundary behavior.

---

## 7. What is deliberately not included

**Unbounded generation.** `Scan_N` is finite. Unbounded generation requires a coalgebraic
account (anamorphism or guarded corecursion) — a different foundation.

**Dynamic iteration count.** `N` is fixed by `L_N`. Runtime-dependent loop lengths need a
dependent or effectful extension.

**Multi-step lookback.** A recurrence `H[l + 1] = step(H[l], H[l - 1], U[l])` can be
represented by replacing `H` with a window object `H ⊗ H`, but this is not part of the
minimal account.

**Gauss-Seidel coupled updates.** The minimal account supports Jacobi-style updates, where
all right-hand sides read the old state and all left-hand sides write the new state.
Gauss-Seidel updates (where later equations in a single step read values already updated
in that step) make semantics order-dependent and are excluded.

**Route capacity management.** Per-destination capacity limits, overflow handling, padding,
and load-balancing losses are not part of the minimal `Route` generator. They can be
modeled by refining `R_{I,E}` to a richer parameter object, but the core **(Route-dispatch)**
axiom remains the semantic anchor.

**Router-to-route compilation.** The score-to-decision step (a neural gate, hash rule, or
top-k policy producing route decisions) is not the `Route` generator. In the minimal
account, `Route` begins once the value-level parameter `ρ` has been produced.

---

## 8. Lean formalization sketch

This section gives a Lean 4 sketch following the `SmallCategory`/`PROP` conventions of
[leanncd.md](leanncd.md). Key conventions: `X ⟶ Y` is `SmallCategory.hom X Y`; `comp f
g` is diagrammatic composition (first `f`, then `g`); `tensor` and `unit` are the
monoidal product and unit from `PROP`; `tensorHom f g : tensor A C ⟶ tensor B D` lifts
two morphisms; `id X : X ⟶ X` is the identity. The project does **not** use Mathlib's
`CategoryTheory`; `▸` rewrites propositional equalities from strictness axioms in term
mode. Every axiom is a field of its declaring class or structure. Helper functions named
in schematic positions (`foldN`, `Orthogonal`, `rhoOf`, `relabelHandlers`, etc.) require
separate definitions not spelled out here.

### 8.1 DGradedPROP

```lean
/-- A D-graded colored PROP: a PROP C equipped with a right action of D. -/
class DGradedPROP (ob_C ob_D : Type) [PROP ob_C] [PROP ob_D] where
  /-- Lift action on objects: X ⊛ P. -/
  actObj     : ob_C → ob_D → ob_C
  /-- Covariant action on morphisms: lift f : X ⟶ Y to actObj X P ⟶ actObj Y P. -/
  actHom     : {X Y : ob_C} → (X ⟶ Y) → (P : ob_D) → actObj X P ⟶ actObj Y P
  /-- Contravariant reindexing along η : P ⟶ Q. -/
  reindex    : {P Q : ob_D} → (X : ob_C) → (P ⟶ Q) → actObj X Q ⟶ actObj X P
  /-- Strict unit: actObj X unit = X. -/
  actUnit    : ∀ (X : ob_C), actObj X unit = X
  /-- Strict distributivity: actObj (tensor X Y) P = tensor (actObj X P) (actObj Y P). -/
  actDistrib : ∀ (X Y : ob_C) (P : ob_D),
    actObj (tensor X Y) P = tensor (actObj X P) (actObj Y P)
  /-- actHom preserves identity. -/
  actHom_id  : ∀ (X : ob_C) (P : ob_D), actHom (id X) P = id (actObj X P)
  /-- actHom preserves composition. -/
  actHom_comp : ∀ {X Y Z : ob_C} (f : X ⟶ Y) (g : Y ⟶ Z) (P : ob_D),
    actHom (comp f g) P = comp (actHom f P) (actHom g P)
  /-- reindex preserves identity. -/
  reindex_id  : ∀ (X : ob_C) (P : ob_D), reindex X (id P) = id (actObj X P)
  /-- reindex is contravariantly functorial. -/
  reindex_comp : ∀ {P Q R : ob_D} (X : ob_C) (η : P ⟶ Q) (θ : Q ⟶ R),
    reindex X (comp η θ) = comp (reindex X θ) (reindex X η)
```

Point evaluation is derived — not a class field — using `reindex` composed with `actUnit`:

```lean
/-- Evaluation at a D-point p : unit ⟶ P, giving actObj X P ⟶ X. -/
def evalAt [PROP ob_C] [PROP ob_D] [DGradedPROP ob_C ob_D]
    (X : ob_C) {P : ob_D} (p : (unit : ob_D) ⟶ P) : actObj X P ⟶ X :=
  -- reindex X p : actObj X P ⟶ actObj X unit
  -- actUnit X : actObj X unit = X, used to rewrite the codomain
  actUnit X ▸ reindex X p
```

### 8.2 Routing envelope

```lean
/-- Projections and pairing used to state local scan laws, not part of the bare PROP. -/
class RoutingEnvelope (ob : Type) [PROP ob] where
  projLeft  : {X Y : ob} → tensor X Y ⟶ X
  projRight : {X Y : ob} → tensor X Y ⟶ Y
  pair      : {Z X Y : ob} → (Z ⟶ X) → (Z ⟶ Y) → Z ⟶ tensor X Y
  pair_projLeft  : {Z X Y : ob} (f : Z ⟶ X) (g : Z ⟶ Y) →
    comp (pair f g) projLeft = f
  pair_projRight : {Z X Y : ob} (f : Z ⟶ X) (g : Z ⟶ Y) →
    comp (pair f g) projRight = g
```

### 8.3 Para

```lean
/-- A Para morphism from X to Y: parameter object plus a morphism using it. -/
structure ParaHom (ob : Type) [PROP ob] (X Y : ob) where
  Param : ob
  run   : tensor Param X ⟶ Y

/-- A reparameterization: a 2-cell in Para(C). -/
structure Reparam (ob : Type) [PROP ob] {X Y : ob} (f g : ParaHom ob X Y) where
  map      : g.Param ⟶ f.Param
  /-- Composing (map ⊗ id) with f.run gives g.run. -/
  commutes : comp (tensorHom map (id X)) f.run = g.run
```

### 8.4 DAlgebra

`DAlgebra` is a `structure` rather than a `class` so that its fields are accessible by
name when passed to `ScanAlgebra` and `RouteAlgebra`.

```lean
structure DAlgebra (ob_C ob_D ob_V : Type)
    [PROP ob_C] [PROP ob_D] [PROP ob_V]
    [DGradedPROP ob_C ob_D] [DGradedPROP ob_V ob_D] where
  mapObj       : ob_C → ob_V
  mapHom       : {X Y : ob_C} → (X ⟶ Y) → (mapObj X ⟶ mapObj Y)
  mapHom_id    : ∀ (X : ob_C), mapHom (id X) = id (mapObj X)
  mapHom_comp  : ∀ {X Y Z : ob_C} (f : X ⟶ Y) (g : Y ⟶ Z),
    mapHom (comp f g) = comp (mapHom f) (mapHom g)
  /-- D-action preservation (strict). -/
  preservesAct : ∀ (X : ob_C) (P : ob_D),
    mapObj (actObj X P) = actObj (mapObj X) P
```

### 8.5 Formalizing Scan

`HasScan` is a class whose fields include the scan generator and all three axioms. `L`
abbreviates the temporal object family; `timePt k` gives the D-point selecting index `k`.

```lean
class HasScan (ob_C ob_D : Type) [PROP ob_C] [PROP ob_D]
    [DGradedPROP ob_C ob_D] [RoutingEnvelope ob_C] where
  /-- The temporal object for length N. -/
  temporalObj : ℕ → ob_D
  /-- Time point pt_k : unit ⟶ temporalObj N for k : Fin N. -/
  timePt      : {N : ℕ} → Fin N → (unit : ob_D) ⟶ temporalObj N
  /-- Prefix inclusion ι_{M,N} : temporalObj M ⟶ temporalObj N, for M ≤ N. -/
  prefixIncl  : {M N : ℕ} → M ≤ N → temporalObj M ⟶ temporalObj N
  /-- The Scan generator. -/
  scan : {H U X : ob_C} (step : tensor H U ⟶ H) (init : X ⟶ H) (N : ℕ)
       → tensor X (actObj U (temporalObj N)) ⟶ actObj H (temporalObj (N + 1))
  /-- (Scan-base): output at time 0 equals projLeft ; init.
      Here ⟨0, Nat.zero_lt_succ N⟩ : Fin (N+1) is the zero point of the output axis. -/
  scan_base : {H U X : ob_C} (step : tensor H U ⟶ H) (init : X ⟶ H) (N : ℕ)
       → comp (scan step init N)
              (evalAt H (timePt (⟨0, Nat.zero_lt_succ N⟩ : Fin (N + 1))))
         = comp projLeft init
  /-- (Scan-step): output at k+1 equals step applied to output at k and input at k.
      k.castSucc : Fin (N+1) embeds k; k.succ : Fin (N+1) is k+1. -/
  scan_step : {H U X : ob_C} (step : tensor H U ⟶ H) (init : X ⟶ H) (N : ℕ)
       (k : Fin N)
       → comp (scan step init N) (evalAt H (timePt k.succ))
         = comp (pair
                   (comp (scan step init N) (evalAt H (timePt k.castSucc)))
                   (comp projRight (evalAt U (timePt k))))
                step
  /-- (Scan-lift): lifting scan over P orthogonal to temporalObj N equals scanning
      with lifted step and init. The LHS domain uses actDistrib to align types. -/
  scan_lift : {H U X : ob_C} (step : tensor H U ⟶ H) (init : X ⟶ H) (N : ℕ)
       {P : ob_D} (h : Orthogonal P (temporalObj N))
       → actDistrib X (actObj U (temporalObj N)) P ▸ actHom (scan step init N) P
         = scan (actHom step P) (actHom init P) N
```

The prefix theorem (Proposition 4.4) follows by induction on `M`, using `scan_base` at
zero and `scan_step` at successors.

Scan algebra extension:

```lean
structure ScanAlgebra (ob_C ob_D ob_V : Type)
    [PROP ob_C] [PROP ob_D] [PROP ob_V]
    [DGradedPROP ob_C ob_D] [DGradedPROP ob_V ob_D] [HasScan ob_C ob_D]
    [RoutingEnvelope ob_C]
    extends DAlgebra ob_C ob_D ob_V where
  /-- (Scan-algebra): the algebra maps scan to fold. -/
  preserve_scan : {H U X : ob_C} (step : tensor H U ⟶ H) (init : X ⟶ H) (N : ℕ)
       → preservesAct (tensor X (actObj U (temporalObj N))) (temporalObj N) ▸
         mapHom (scan step init N)
         = foldN (mapHom step) (mapHom init) N
         -- foldN : (mapObj H ⊗ mapObj U ⟶ mapObj H) → (mapObj X ⟶ mapObj H)
         --       → ℕ → tensor (mapObj X) (actObj (mapObj U) (temporalObj N))
         --           ⟶ actObj (mapObj H) (temporalObj (N + 1))
```

### 8.6 Formalizing Route

Route data types:

```lean
/-- A concrete route: assigns each item point a destination point. -/
structure RouteParam (ob_D : Type) [PROP ob_D] (I E : ob_D) where
  assign : ((unit : ob_D) ⟶ I) → ((unit : ob_D) ⟶ E)

/-- A handler family: a handler morphism for each destination point. -/
structure HandlerFamily (ob_C ob_D : Type) [PROP ob_C] [PROP ob_D]
    [DGradedPROP ob_C ob_D] (A B : ob_C) (E : ob_D) where
  atDest : ((unit : ob_D) ⟶ E) → (A ⟶ B)

/-- An isomorphism in D, used for destination relabeling. -/
structure DIso (ob_D : Type) [PROP ob_D] (E E' : ob_D) where
  fwd : E ⟶ E';  bwd : E' ⟶ E
  fwd_bwd : comp fwd bwd = id E;  bwd_fwd : comp bwd fwd = id E'
```

`HasRoute` axiomatizes the route generator with all four axioms as fields:

```lean
class HasRoute (ob_C ob_D : Type) [PROP ob_C] [PROP ob_D]
    [DGradedPROP ob_C ob_D] [RoutingEnvelope ob_C] where
  /-- The route generator: given handlers and a concrete route, produce a C-morphism. -/
  route : {A B : ob_C} {I E : ob_D}
        → HandlerFamily ob_C ob_D A B E → RouteParam ob_D I E
        → actObj A I ⟶ actObj B I
  /-- (Route-dispatch): evaluating output at i gives handler_{ρ(i)} applied to input at i. -/
  route_dispatch : {A B : ob_C} {I E : ob_D}
        (handlers : HandlerFamily ob_C ob_D A B E) (ρ : RouteParam ob_D I E)
        (i : (unit : ob_D) ⟶ I)
        → comp (route handlers ρ) (evalAt B i)
          = comp (evalAt A i) (handlers.atDest (ρ.assign i))
  /-- (Route-static): a route induced by a D-morphism η : I ⟶ E reduces to ordinary
      structure; the route point η(i) = comp i η. -/
  rhoOf  : {I E : ob_D} → (I ⟶ E) → RouteParam ob_D I E
  route_static : {A B : ob_C} {I E : ob_D}
        (handlers : HandlerFamily ob_C ob_D A B E) (η : I ⟶ E)
        (i : (unit : ob_D) ⟶ I)
        → comp (route handlers (rhoOf η)) (evalAt B i)
          = comp (evalAt A i) (handlers.atDest (comp i η))
  /-- (Route-lift): route over orthogonal P equals routing with lifted handlers.
      liftHandlers and liftRoute are auxiliary definitions. -/
  route_lift : {A B : ob_C} {I E : ob_D}
        (handlers : HandlerFamily ob_C ob_D A B E) (ρ : RouteParam ob_D I E)
        {P : ob_D} (hI : Orthogonal P I) (hE : Orthogonal P E)
        → actDistrib A I P ▸ actHom (route handlers ρ) P
          = route (liftHandlers handlers P) (liftRoute ρ P)
  /-- (Route-relabel): relabeling destinations with σ : E ≅ E is transparent. -/
  route_relabel : {A B : ob_C} {I E : ob_D}
        (handlers : HandlerFamily ob_C ob_D A B E) (ρ : RouteParam ob_D I E)
        (σ : DIso ob_D E E)
        → route handlers ρ = route (relabelHandlers handlers σ) (relabelRoute ρ σ)
```

Route algebra extension:

```lean
structure RouteAlgebra (ob_C ob_D ob_V : Type)
    [PROP ob_C] [PROP ob_D] [PROP ob_V]
    [DGradedPROP ob_C ob_D] [DGradedPROP ob_V ob_D] [HasRoute ob_C ob_D]
    [RoutingEnvelope ob_C]
    extends DAlgebra ob_C ob_D ob_V where
  /-- (Route-algebra): the interpreted route is pointwise dispatch. -/
  preserve_route : {A B : ob_C} {I E : ob_D}
       (handlers : HandlerFamily ob_C ob_D A B E) (ρ : RouteParam ob_D I E)
       (i : (unit : ob_D) ⟶ I)
       → comp (preservesAct A I ▸ mapHom (route handlers ρ)) (evalAt (mapObj B) i)
         = comp (evalAt (mapObj A) i) (mapHom (handlers.atDest (ρ.assign i)))
```

### 8.7 Formalizing the unified schema

```lean
/-- The prior-slice family: all output slices G ; evalAt Y t for t ≺ s. -/
def PriorSlices [PROP ob_C] [PROP ob_D] [DGradedPROP ob_C ob_D]
    {S : ob_D} (depends : ((unit : ob_D) ⟶ S) → ((unit : ob_D) ⟶ S) → Prop)
    {Z Y : ob_C} (gen : Z ⟶ actObj Y S) (s : (unit : ob_D) ⟶ S) : Type :=
  (t : (unit : ob_D) ⟶ S) → depends t s → (Z ⟶ Y)

/-- Schema for a non-weave generator: output index object S, well-founded dependency,
    and a local evaluator at each point. -/
structure PointPresentation (ob_C ob_D : Type) [PROP ob_C] [PROP ob_D]
    [DGradedPROP ob_C ob_D] where
  S         : ob_D
  Y         : ob_C
  Z         : ob_C
  depends   : ((unit : ob_D) ⟶ S) → ((unit : ob_D) ⟶ S) → Prop
  wf        : WellFounded depends
  /-- Local evaluator: computes output at s given prior slices. -/
  evalLocal : {G : Z ⟶ actObj Y S}
            → (s : (unit : ob_D) ⟶ S)
            → PriorSlices depends G s
            → Z ⟶ Y

/-- A non-weave generator satisfying the point-presentation law. -/
structure NonWeaveGenerator (ob_C ob_D : Type) [PROP ob_C] [PROP ob_D]
    [DGradedPROP ob_C ob_D]
    (pres : PointPresentation ob_C ob_D) where
  gen       : pres.Z ⟶ actObj pres.Y pres.S
  /-- (NW-point): each output slice equals the local evaluator applied to prior slices. -/
  point_law : (s : (unit : ob_D) ⟶ pres.S)
            → comp gen (evalAt pres.Y s)
              = pres.evalLocal s (fun t _ => comp gen (evalAt pres.Y t))
```

The five shared axiom interfaces:

```lean
/-- (NW-boundary): degenerate cases agree with ordinary D-graded structure. -/
class HasBoundary (ob_C ob_D : Type) [PROP ob_C] [PROP ob_D] [DGradedPROP ob_C ob_D]
    {pres : PointPresentation ob_C ob_D} (G : NonWeaveGenerator ob_C ob_D pres) where
  boundary_law : BoundaryCase G   -- defined per-generator (Scan: base point; Route: static ρ)

/-- (NW-lift): generator commutes with lifting over orthogonal P. -/
class HasOrthogonalLift (ob_C ob_D : Type) [PROP ob_C] [PROP ob_D] [DGradedPROP ob_C ob_D]
    {pres : PointPresentation ob_C ob_D} (G : NonWeaveGenerator ob_C ob_D pres) where
  lift_law : {P : ob_D} → Orthogonal P pres.S
           → actDistrib pres.Z (actObj pres.Y pres.S) P ▸ actHom G.gen P
             = liftedGenerator G P

/-- (NW-relabel): isomorphisms preserving the dependency relation are transparent. -/
class HasRelabeling (ob_C ob_D : Type) [PROP ob_C] [PROP ob_D] [DGradedPROP ob_C ob_D]
    {pres : PointPresentation ob_C ob_D} (G : NonWeaveGenerator ob_C ob_D pres) where
  relabel_law : {σ : DIso ob_D pres.S pres.S} → PreservesLocalLaw σ pres
              → relabelGenerator σ G = G

/-- (NW-algebra): the algebra maps the generator to its intended semantics. -/
class PreservesAlgebra (ob_C ob_D ob_V : Type)
    [PROP ob_C] [PROP ob_D] [PROP ob_V]
    [DGradedPROP ob_C ob_D] [DGradedPROP ob_V ob_D]
    {pres : PointPresentation ob_C ob_D} (G : NonWeaveGenerator ob_C ob_D pres)
    (F : DAlgebra ob_C ob_D ob_V) where
  semantic : F.mapObj pres.Z ⟶ actObj (F.mapObj pres.Y) pres.S
  preserve : F.preservesAct pres.Y pres.S ▸ F.mapHom G.gen = semantic
```

Instantiation lemmas stating that Scan and Route satisfy the schema:

```lean
def scanAsNonWeave {ob_C ob_D : Type} [PROP ob_C] [PROP ob_D]
    [DGradedPROP ob_C ob_D] [RoutingEnvelope ob_C] [HasScan ob_C ob_D]
    {H U X : ob_C} (step : tensor H U ⟶ H) (init : X ⟶ H) (N : ℕ)
    : NonWeaveGenerator ob_C ob_D (scanPresentation step init N) :=
  { gen := scan step init N, point_law := sorry }

def routeAsNonWeave {ob_C ob_D : Type} [PROP ob_C] [PROP ob_D]
    [DGradedPROP ob_C ob_D] [RoutingEnvelope ob_C] [HasRoute ob_C ob_D]
    {A B : ob_C} {I E : ob_D}
    (handlers : HandlerFamily ob_C ob_D A B E) (ρ : RouteParam ob_D I E)
    : NonWeaveGenerator ob_C ob_D (routePresentation handlers ρ) :=
  { gen := route handlers ρ, point_law := sorry }
```

### 8.8 Proof targets

1. **Point extensionality.** Two `NonWeaveGenerator`s for the same `PointPresentation`
   that agree on `comp gen (evalAt Y s)` for every `s` are equal as generators.

2. **Scan prefix theorem** (Proposition 4.4). Induction on `M`, with `scan_base` closing
   the zero case and `scan_step` closing the successor case.

3. **Scan orthogonal batching.** Follows from `scan_lift` as a field of `HasScan`.

4. **Route static specialization.** If `ρ = rhoOf η`, then
   `route handlers ρ = route handlers (rhoOf η)` and `route_static` applies.

5. **Route relabeling.** `route_relabel` states this directly; the key proof obligation
   is that `comp i (σ.fwd)` correctly composes item point `i` with the relabeling.

6. **Unified reuse.** Any instance of `NonWeaveGenerator` that also satisfies
   `HasOrthogonalLift` and `PreservesAlgebra` admits sound backend implementations
   whenever `F.mapHom G.gen = semantic`.

---

## 9. Summary

`Scan` and `Route` are the two smallest extensions to the `D`-graded PROP formalism
needed here, because both sit just outside the ordinary weave fragment.

For `Scan`, the obstruction is temporal coupling. The output at time `k + 1` depends on
the output at time `k`, so point naturality fails for any ordinary lifted operation. The
minimal extension adds the generator

```text
Scan_N(step, init) : X ⊗ (U ⊛ L_N) → H ⊛ L_{N+1}
```

with axioms **(Scan-base)**, **(Scan-step)**, **(Scan-lift)**, and **(Scan-algebra)**:

```text
F(Scan_N(step, init)) = fold_N(F(step), F(init)).
```

This is smaller than adding a new directed action or full temporal category: the existing
`D`-action already supplies point evaluation, prefix restriction, and ordinary batching;
`Scan` only adds the finite recurrence equations.

For single-destination `Route`, the obstruction is value-dependent indexing. The item axis
is fixed, but the destination choice `ρ : I → E` is a runtime value rather than a static
`D`-morphism. The minimal extension puts `Route` in `Para(C)`, adds the parameter object
`R_{I,E}`, and requires axioms **(Route-dispatch)**, **(Route-static)**,
**(Route-lift)**, **(Route-relabel)**, and **(Route-algebra)**:

```text
F(Route_ρ({handler_e}))(x)_i = F(handler_{ρ(i)})(x_i).
```

Both are **non-weave generators**: a point-indexed output `Y ⊛ S`, a well-founded
point-presentation law, a boundary case collapsing to ordinary `D`-graded structure,
orthogonal lift distribution, harmless relabeling equivariance, and algebra preservation.
`Scan` instantiates with a predecessor relation on time; `Route` with an empty output
dependency and a value-level routing parameter.

A Lean development should mirror the categorical layers: first the `D`-graded PROP, point
evaluation, routing envelope, `Para` morphisms, and `D`-compatible algebras; then
`HasScan` and `HasRoute` typeclasses with their pointwise laws; finally a
`NonWeaveGenerator` interface with proof targets for point extensionality, prefix
restriction, static specialization, relabeling, orthogonal batching, and backend
soundness.

[Appendix A](#appendix-a-pyncd-code-correspondence-for-scan) maps the abstract
`Scan_N(step, init)` laws to the `Scan` dataclass and `ConstructedScan` runtime.

---

## Appendix A: pyncd code correspondence for `Scan`

The categorical generator

```text
Scan_N(step, init) : X ⊗ (U ⊛ L_N) → H ⊛ L_{N+1}
```

is represented by the frozen dataclass `Scan` in `data_structure/TensorDSL.py`:

```python
@dataclass(frozen=True)
class Scan(fd.Term):
    step: object
    base: object
    N: nm.Numeric
    axis: sc.RawAxis
    affine: ScanAffine | None = None
    n_states: int = 1
    step_state_deps: tuple[tuple[int, ...], ...] = ()
    step_x_l_positions: tuple[int, ...] = ()
```

The names differ slightly: `base` is the implementation name for `init`, and `axis`
together with `N` represent the finite temporal object `L_N`.

| Categorical item | pyncd code item | Meaning |
| --- | --- | --- |
| `Scan_N(step, init)` | `Scan(step=..., base=..., N=..., axis=...)` | The term-level scan generator |
| `step : H ⊗ U → H` | `Scan.step` | The one-step update morphism |
| `init : X → H` | `Scan.base` | The base-case morphism computing `H_0` |
| `L_N` | `Scan.axis` and `Scan.N` | The recurrence axis and finite step count |
| `H ⊛ L_{N+1}` | output shape `step_out + (axis,)` | The full history: base plus `N` updates |
| orthogonal axes `P` | ordinary tensor axes ≠ `axis` | Batch/state axes carried through each loop slice |
| algebra `F` | `ConstructedScan` | The PyTorch interpretation of the `Scan` term |

### A.1 The `step` morphism in code

During `TensorDSL._finalize_iter()`, pyncd turns an equation of the form

```text
H[..., l + 1] = expr(H[..., l], U[..., l])
```

into the one-step morphism in three stages.

1. Rejects future-state reads on the RHS:

   ```python
   self._check_no_lnext_on_rhs(recur_value, l, name_str)
   ```

2. Strips the recurrence axis `l` from RHS factors and renames the recurrent tensor to an
   internal state proxy:

   ```python
   step_value = self._strip_iter_axis_from_value(
       recur_value, l, {state_name_dn: state_proxy_dn}
   )
   ```

   This turns `expr(H[..., l], U[..., l])` into a point-level expression
   `expr(H_state, U_step)`.

3. Compiles the stripped expression to a pyncd morphism:

   ```python
   step_morph = self._build_step_morph(None, step_out, step_value, step_ctx)
   ```

At runtime, `ConstructedScan` applies it in the sequential reference loop:

```python
sliced = tuple(x[..., l_idx] for x in step_xs)
H = to_tuple(self.step_module(H, *sliced))[0]
```

| Case | Representation of `Scan.step` |
| --- | --- |
| Ordinary single-state recurrence | morphism built from the stripped RHS expression |
| `TensorProxy.recur(axis, morphism)` | the prebuilt user-supplied morphism; bypasses TL RHS parsing |
| Coupled Jacobi scan, `n_states > 1` | tuple of step morphisms, one per state |

### A.2 Axiom-to-code correspondence

| Axiom | Concrete pyncd condition |
| --- | --- |
| **(Scan-base)** | `_finalize_iter()` requires a base equation and rejects `base_literal != 0`; `base` therefore computes `H_0`. |
| **(Scan-step)** | `_finalize_iter()` strips the recurrence axis from the RHS, rejects `l + 1` on the RHS, and builds `step` as the update from old state plus current input slice. `ConstructedScan._run_loop()` implements `H = step_module(H, *x[..., l_idx])`. |
| **(Scan-lift)** | `ConstructedScan` slices only along the recurrence axis; `step_x_l_positions` moves that axis to last dimension before slicing; all other axes are preserved as independent batch/state axes. |
| **(Scan-algebra)** | `ConstructedScan.forward()` returns the history `[H_0, ..., H_N]` stacked along `axis`; the sequential loop is the reference semantics. |
| Finite `Scan_N` | `ConstructedScan.__init__()` requires `N` to be an `nm.Integer`; dynamic iteration counts are outside the minimal account. |

Implementation refinements that are not new primitive axioms:

| pyncd field | Relation to the theory |
| --- | --- |
| `n_states > 1` | Coupled Jacobi-style Scan: updates several states from the old state tuple simultaneously, generalizing the single-state step law. |
| `step_state_deps` | Records which old states each coupled step reads, preserving a well-defined Jacobi dependency structure. |
| `affine : ScanAffine \| None` | Optional optimization evidence for an affine recurrence `H[l+1] = A_l H[l] + b_l`. Associative-scan lowering is valid only when observationally equal to the sequential recurrence, justified by **(Scan-algebra)** plus extra affine/associative structure. |
| `step_x_l_positions` | Runtime metadata ensuring the recurrence axis is sliced correctly regardless of its position in the user's tensor expression. |

The code-level validity criterion:

```text
A Scan dataclass is valid when:
1. N is concrete.
2. base defines H_0 at axis coordinate 0.
3. step defines H_{l+1} using only H_l and inputs at l, never l+1.
4. axis is the unique recurrence axis; other axes are orthogonal batch/state axes.
5. forward() returns the full history H_0 ... H_N stacked along axis.
6. any optimized lowering is observationally equal to the sequential recurrence.
```

The first three points are enforced while building the `Scan` term. The last three are
semantic obligations of `ConstructedScan` and any future backend algebra interpreting the
same generator.

### A.3 Test-case correspondence

Regression tests live in [`tests/test_torch_compile.py`](../tests/test_torch_compile.py).

| Theoretical obligation | Test | Structure |
| --- | --- | --- |
| **(Scan-base)** | `test_scan_base_case_is_output_slice_zero` | Builds `H[i, 0] = X[i]` and `H[i, l + 1] = H[i, l] + Delta[i, l]`; asserts output slice `0` equals `X`. |
| Base point must be `0` | `test_scan_rejects_nonzero_base_index` | Attempts `H[i, 1] = X[i]`; expects `tl.to_morphism()` to reject. |
| **(Scan-step)** cannot read future state | `test_scan_rejects_future_state_read_on_rhs` | Attempts a RHS read of `H[i, l + 1]`; expects construction failure. |
| Finite `Scan_N` requires concrete `N` | `test_scan_rejects_unsized_iteration_axis` | Uses unsized recurrence axis; expects rejection. |
| **(Scan-lift)** | `test_scan_orthogonal_batch_axis_runs_independent_loops` | Adds independent batch axis `b`; compares compiled batched scan with independent per-batch Python loops. |
| Orthogonal axes preserved; only `axis` sliced | `test_scan_step_axis_position_is_moved_to_last_before_slicing` | Writes step input as `Delta[l, i]`; checks that axis movement yields the same history as slicing over `l`. |
| **(Scan-algebra)** for optimized lowering | `test_scan_affine_fast_path_preserves_reference_recurrence` | Uses affine recurrence `H[i,l+1] = A[i,l] * H[i,l] + B[i,l]`; verifies affine fast path equals explicit sequential loop. |

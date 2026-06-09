# Scan and Route in a Graded PROP

This note refines the `Scan` discussion in [graded_prop.md](graded_prop.md) and
[iteration.md](iteration.md). Its goal is narrow: identify the smallest extension
to the `D`-graded colored PROP formalism that gives `Scan` and `Route`
categorical foundations.

Every symbol used below is introduced before it is used. The notation follows
[graded_prop.md](graded_prop.md): composition is written in diagrammatic order,
so `f ; g` means "first `f`, then `g`".

Notation is kept uniform throughout. A morphism type is written with `->`.
Category products are written with `×`; monoidal products inside a category are
written with `⊗`. The lifted object is `X ⊛ P`. Evaluation at a point `p` of a
`D`-object `P` on a `C`-object `X` is written `ev_{p,X}`. The symbol `ω` denotes
ordinary weave reindexing metadata, `ρ` denotes a runtime Route parameter,
`η` denotes a fixed structural route, and `σ` denotes a destination relabeling
isomorphism. Generic Para parameter objects are written `Θ`, not `P` or `Q`,
because `P` and `Q` are reserved for index shapes in `D` such as `St` shapes.

---

## Contents

1. [Why formulate Scan and Route axiomatically?](#1-why-formulate-scan-and-route-axiomatically)
2. [Base setting](#2-base-setting)
3. [The weave criterion](#3-the-weave-criterion)
4. [`Scan`: fixed temporal coupling](#4-scan-fixed-temporal-coupling)
   1. [Obstruction to being a weave](#41-obstruction-to-being-a-weave)
   2. [Minimal extension](#42-minimal-extension)
   3. [Axioms](#43-axiom-1-base-case)
   4. [Algebra semantics](#46-algebra-semantics)
   5. [Consequences](#47-consequences)
   6. [Broader Scan instances and axiom stability](#48-broader-scan-instances-and-axiom-stability)
5. [`Route`: value-dependent indexing](#5-route-value-dependent-indexing)
   1. [Obstruction to being a weave](#51-obstruction-to-being-a-weave)
   2. [Minimal extension](#52-minimal-extension)
   3. [Axioms](#53-axiom-1-itemwise-dispatch)
   4. [Algebra semantics](#57-algebra-semantics)
   5. [Weighted multi-destination routing](#58-weighted-multi-destination-routing)
   6. [Consequences](#59-consequences)
   7. [Broader Route instances and axiom stability](#510-broader-route-instances-and-axiom-stability)
6. [Unified non-weave generator axioms](#6-unified-non-weave-generator-axioms)
7. [What is deliberately not included](#7-what-is-deliberately-not-included)
8. [Lean shape](#8-lean-shape)
9. [Summary](#9-summary)

---

## 1. Why formulate Scan and Route axiomatically?

The ordinary `D`-graded PROP already explains a large class of pyncd programs:
base operations can be lifted over index shapes, reindexed by structural
`D`-morphisms, and compiled by algebras such as `construct()`. `Scan` and
`Route` are important precisely because they sit just outside that ordinary
weave fragment.

A machine learning task that requires `Scan` is autoregressive sequence modeling
with a recurrent state, such as evaluating a recurrent neural network or a
state-space model over a token sequence. The hidden state at token `l + 1`
depends on the hidden state at token `l`, so the computation cannot be expressed
as an independent pointwise lift over the sequence axis. The same pattern also
appears in weight-tied depth, iterative refinement, learned optimizers,
message-passing rounds, numerical solvers, and any fixed-length computation that
threads a state through ordered steps.

A machine learning task that requires `Route` is value-dependent dispatch, such
as sparse mixture-of-experts inference, dynamic token-to-memory routing, or
choosing a branch in an adaptive computation graph. A router computes, from each
item's activation values, which destination should process that item. The chosen
destination index is therefore a runtime value, not a fixed structural
reindexing known when the graph is built. The same pattern also appears in
per-sample graph structure, learned clustering or slot assignment, hard
attention, adapter selection, and stochastic resampling.

An axiomatic formulation is useful for four reasons.

First, it separates **what an operator means** from **how an implementation runs
it**. A sequential loop, a checkpointed loop, and an associative parallel prefix
can all implement the same `Scan` when they satisfy the same axioms. Likewise,
a dense dispatch kernel, a sparse dispatch kernel, and a fused routed-computation
kernel can all implement the same `Route` when they satisfy the same route laws.

Second, it identifies the exact obstruction to being a weave. `Scan` is not a
weave because it couples neighboring temporal points. `Route` is not a weave
because the reindexing map is a runtime value rather than a fixed `D`-morphism.
The two failures are different, so they need different minimal extensions.

Third, axioms give reusable compiler laws. Prefix restriction, vectorization over
orthogonal axes, fixed-route specialization, destination relabeling, and algebra
preservation become theorems or fields to check rather than ad hoc conventions
inside each backend.

Fourth, an axiomatic boundary keeps the core small. The goal is not to add a
general theory of loops, effects, routing, and dynamic shapes. The goal is to add
only the equations that make finite scan and value-dependent route compositional
inside the existing categorical framework.

The rest of the note therefore presents `Scan` and `Route` in parallel. Each is
introduced by its obstruction to the weave criterion, then given a signature,
axioms, algebra semantics, and derived consequences.

---

## 2. Base setting

A **colored PROP** is a strict symmetric monoidal category whose objects are
finite lists of colors. A **color** is a wire type. The monoidal product `⊗`
means juxtaposition of wires, the monoidal unit `I` is the empty list of wires,
and the symmetry maps permit wires to be permuted.

Let `D` be a colored PROP called the **index PROP**. Its objects are index
shapes and its morphisms are reindexings. In pyncd's implemented instance,
`D = St`, where an object is a tuple of axes and a morphism is an affine
stride/reindexing map between axis tuples.

Let `C` be a colored PROP called the **operation PROP**. Its objects are lists of
typed arrays, and its morphisms are tensor operations. In pyncd's implemented
instance, `C = Br`, the broadcasted category.

A **`D`-graded colored PROP** consists of `C`, `D`, and the following structure.

1. A **shape map**

   ```text
   sh : colors(C) -> Ob(D)
   ```

   sending each `C`-color to its underlying `D`-shape. For `C = Br` and
   `D = St`, the color `[a, A]` is an array with datatype `a` and axis tuple
   `A`, and `sh([a, A]) = A`.

2. A **right action**

   ```text
   act : C × D^op -> C
   ```

   where `D^op` is the opposite category of `D`. On objects, the action appends
   an index shape. We write:

   ```text
   X ⊛ P
   ```

   for `act(X, P)`, where `X` is a `C`-object and `P` is a `D`-object. In Br,
   this means adding the axes of `P` to every array in `X`.

   On morphisms, the action has two common forms:

   ```text
   [f, P] : X ⊛ P -> Y ⊛ P
   ```

   for a `C`-morphism `f : X -> Y`, and

   ```text
   [X, η] : X ⊛ Q -> X ⊛ P
   ```

   for a `D`-morphism `η : P -> Q`. The second form is contravariant in `D`,
   which is why the direction reverses.

3. Coherence laws saying that `act` is functorial and distributes over `⊗`.
   The most important law for this note is:

   ```text
   [f ; g, P] = [f, P] ; [g, P]
   ```

   for composable `C`-morphisms `f` and `g`.

A **point** of a `D`-object `P` is a `D`-morphism

```text
p : I_D -> P
```

where `I_D` is the monoidal unit of `D`. In `St`, a point of an axis is a
particular index value.

For every point `p : I_D -> P`, the action gives an **evaluation map**

```text
ev_{p,X} : X ⊛ P -> X
```

This map is derived from the same action functor, not added as a separate
generator. Since `act` is contravariant in its `D`-argument, the point
`p : I_D -> P` is used as an arrow `p^op : P -> I_D` in `D^op`, giving

```text
act(id_X, p^op) : X ⊛ P -> X ⊛ I_D.
```

Composing this with the action unit isomorphism

```text
υ_X : X ⊛ I_D ≅ X
```

gives `ev_{p,X}`. Equivalently, `ev_{p,X}` is `[X, p]` followed by
`υ_X`. When `X` is clear, we write this as `ev_p`.

A **target actegory** is a symmetric monoidal category `V` equipped with a right
action of `D`, written:

```text
⊛_V : V × D^op -> V.
```

For pyncd, `V` is the category of PyTorch tensor spaces and tensor functions;
`A ⊛_V P` appends the dimensions described by `P` to the tensor space `A`.

An **algebra** of `C` in `V` is a strong symmetric monoidal functor:

```text
F : C -> V
```

that preserves the `D`-action up to coherent isomorphism. In pyncd,
`F` is `construct()`: it maps abstract Br morphisms to concrete PyTorch modules.

---

## 3. The weave criterion

A **weave** is the data witnessing that a `C`-morphism is a lifted base operation
over an index shape. More explicitly, a morphism

```text
g : X -> Y
```

is a weave over a `D`-object `P` when it factors as

```text
g = [f, P] ; ω
```

where:

- `f : X0 -> Y0` is a base operation in `C`;
- `[f, P]` is the lift of `f` over `P`;
- `ω` is assembled from reindexing maps and symmetry maps.

The key test for being a weave is **point naturality**. A lifted operation must
satisfy, for every point `p : I_D -> P`,

```text
[f, P] ; ev_{p,Y} = ev_{p,X} ; f.
```

This equation says that evaluating a lifted operation at point `p` is the same
as evaluating the inputs at `p` and then running the base operation. In
[graded_prop.md](graded_prop.md), this is Eq. 3.

The intuition is simple: a weave is pointwise independent over its degree
coordinates. If a morphism mixes different points of the degree axis, it cannot
be a weave over that axis.

---

## 4. `Scan`: fixed temporal coupling

### 4.1 Obstruction to being a weave

A **recurrence** is a computation where the value at the next step depends on
the value at the current step. In tensor notation:

```text
H[l + 1] = step(H[l], U[l])
```

where:

- `l` is the time or iteration index;
- `H[l]` is the state at step `l`;
- `U[l]` is the external input slice at step `l`;
- `step` is the update operation.

`Scan` is the morphism that computes the full history

```text
H[0], H[1], ..., H[N]
```

from an initial state `H[0]` and per-step inputs `U[0], ..., U[N - 1]`.

`Scan` is not a weave over the temporal axis because point naturality fails. To
compute the output at time `l + 1`, one must first compute the output at time
`l`. Therefore evaluating `Scan` at a point does not commute with running the
operation:

```text
Scan ; ev_{l+1,H}  !=  ev_{l,U} ; step
```

as a pointwise lifted operation. The right side is not even well-typed without
the previously accumulated state.

The obstruction is **not** that time is data-dependent. The time reindexing is
fixed and structural. The obstruction is that the morphism couples different
positions of the time axis. This is different from `Route`, where the problem is
that the reindexing itself depends on runtime data.

The examples in [future_ideas.md](future_ideas.md),
[iteration.md](iteration.md), and [graded_prop.md](graded_prop.md) show that
this is broader than RNN-style sequence models. `Scan` is the right abstraction
whenever a fixed structural axis orders repeated applications of a state update,
and each point depends on earlier points along that axis.

---

### 4.2 Minimal extension

The minimal extension is to add `Scan` as a distinguished generator with three
axioms. No new structure is needed on `D`, and no new action functor is needed
beyond the existing `act : C × D^op -> C`.

#### 4.2.1 Temporal objects

For each natural number `N`, let

```text
L_N
```

denote the `D`-object representing the finite ordered set

```text
{0, 1, ..., N - 1}.
```

Thus `L_N` has `N` time points. In `St`, `L_N` is an axis of size `N`.

For `0 <= k < N`, let

```text
pt_k : I_D -> L_N
```

be the point selecting index `k`.

For `0 <= M <= N`, let

```text
ι_{M,N} : L_M -> L_N
```

be the prefix inclusion. It sends the ordered set `{0, ..., M - 1}` into the
first `M` points of `{0, ..., N - 1}`. In `St`, this is the affine inclusion of a
prefix axis into a longer axis.

#### 4.2.2 Step and initial morphisms

Let:

```text
H
```

be a `C`-object representing the state at one step.

Let:

```text
U
```

be a `C`-object representing the per-step external input.

Let:

```text
X
```

be a `C`-object representing the input from which the initial state is computed.

Let:

```text
init : X -> H
```

be a `C`-morphism computing the initial state.

Let:

```text
step : H ⊗ U -> H
```

be a `C`-morphism computing the next state from the current state and the
current input slice.

The full sequence of per-step inputs is the `C`-object:

```text
U ⊛ L_N.
```

This is `U` lifted over the `N` time points.

#### 4.2.3 Scan signature

For each natural number `N`, add a generator:

```text
Scan_N(step, init) : X ⊗ (U ⊛ L_N) -> H ⊛ L_{N+1}.
```

The domain has two parts:

- `X`, used to compute the initial state;
- `U ⊛ L_N`, the sequence of `N` per-step inputs.

The codomain is:

```text
H ⊛ L_{N+1},
```

the history of `N + 1` states: the initial state plus one state after each step.

### 4.3 Axiom 1: base case

The pointwise Scan laws are stated in an **ambient routing envelope** around the
linear PROP. The routing envelope supplies projections and pairings for the
external scan inputs, matching the role played by `ThreadedComposed` in pyncd.

If `Z`, `A`, and `B` are `C`-objects, a **projection**

```text
π_A : A ⊗ B -> A
```

keeps the `A` component of a product input. A projection is not a primitive of a
strictly linear PROP; it is routing metadata saying that a downstream subterm
uses one live input and ignores another.

If `a : Z -> A` and `b : Z -> B` are two morphisms with the same domain `Z`, a
**pairing**

```text
⟨a, b⟩ : Z -> A ⊗ B
```

feeds the same source `Z` to both `a` and `b`, then juxtaposes their outputs. A
pairing also belongs to the routing envelope, not to the bare linear PROP. In a
fully linear presentation, the same equations can be expressed by explicitly
duplicating the required external wires before entering the scan.

The **base-case axiom** says that the first output slice is the initial state.

Let:

```text
pt_0 : I_D -> L_{N+1}
```

be the point selecting time `0`. Then:

```text
Scan_N(step, init) ; ev_{pt_0,H}
  =
π_X ; init.
```

Here:

- `π_X : X ⊗ (U ⊛ L_N) -> X` is the projection that keeps the `X` component and
  discards the input-sequence component;
- `ev_{pt_0,H} : H ⊛ L_{N+1} -> H` evaluates the output history at time `0`.

The projection is the only reason this equation is not a bare PROP equation.
The categorical content is the base-case equality; the projection just adapts
the equality to pyncd's multi-input calling convention.

### 4.4 Axiom 2: inductive step

The **step axiom** says that output slice `k + 1` is obtained by applying
`step` to output slice `k` and input slice `k`.

For `0 <= k < N`, let:

```text
pt_k : I_D -> L_N
```

select input time `k`, and let:

```text
pt'_k : I_D -> L_{N+1}
pt'_{k+1} : I_D -> L_{N+1}
```

select output times `k` and `k + 1`, respectively.

Then:

```text
Scan_N(step, init) ; ev_{pt'_{k+1},H}
  =
⟨
  (Scan_N(step, init) ; ev_{pt'_k,H})
,
  (π_U ; ev_{pt_k,U})
⟩
; step.
```

Here:

- `π_U : X ⊗ (U ⊛ L_N) -> U ⊛ L_N` keeps the per-step input sequence;
- `ev_{pt_k,U} : U ⊛ L_N -> U` selects the input at time `k`;
- `ev_{pt'_k,H} : H ⊛ L_{N+1} -> H` selects the output state at time `k`;
- `ev_{pt'_{k+1},H} : H ⊛ L_{N+1} -> H` selects the output state at time `k + 1`;
- `⟨-, -⟩` pairs two reads from the same scan input so that their outputs can be
  passed together to `step : H ⊗ U -> H`.

This equation is a finite catamorphism law. It does not require an infinite
initial algebra or a fixpoint object because `N` is finite.

### 4.5 Axiom 3: orthogonal lift distribution

Let:

```text
P
```

be a `D`-object representing a non-temporal batch, head, sample, or spatial
degree.

Say that `P` is **orthogonal** to `L_N` when the axes or index components of `P`
are disjoint from the temporal components of `L_N`. In `St`, this means `P` and
`L_N` share no axis UID.

The **orthogonal lift-distribution axiom** says:

```text
[Scan_N(step, init), P]
  ≅
Scan_N([step, P], [init, P]).
```

The symbol `≅` denotes a canonical isomorphism in `C`; it may be definitional
equality in a strict implementation.

This axiom says that batching a scan over an independent axis is the same as
scanning the batched step. It is the categorical basis for vectorizing a
recurrence across batch coordinates.

---

### 4.6 Algebra semantics

Let `F : C -> V` be an algebra into a target actegory. To support `Scan`, the
algebra must satisfy one additional preservation law:

```text
F(Scan_N(step, init))
  =
fold_N(F(step), F(init)).
```

Here:

- `F(step) : F(H) ⊗ F(U) -> F(H)` is the concrete step function;
- `F(init) : F(X) -> F(H)` is the concrete initial-state function;
- `fold_N(F(step), F(init))` is the concrete finite loop that applies
  `F(step)` exactly `N` times and returns the `N + 1` state history.

Different implementations of `fold_N` may represent the same `V`-morphism:

| Implementation | Meaning | Extra condition |
| --- | --- | --- |
| Eager loop | Run steps sequentially | Always valid |
| Checkpointed loop | Recompute activations during backpropagation | Step is pure |
| Batched loop | Run independent scans across an orthogonal axis | Axiom 3 applies |
| Associative scan | Parallel prefix computation | Step factors through a monoid |

The word **monoid** means an object `M` with an associative multiplication
`μ : M ⊗ M -> M` and a unit `η : I -> M`. A step factors through a monoid when
each step can be represented as a monoid element and sequence execution is
monoid multiplication. In that case, the scan can be computed by a parallel
prefix algorithm.

---

### 4.7 Consequences

The three Scan axioms are enough to recover the useful laws without treating
`Scan` as an opaque generator.

#### 4.7.1 Prefix restriction

For `0 <= M <= N`, the prefix restriction of an `N`-step scan to its first
`M + 1` output states equals the `M`-step scan.

Formally, the prefix map:

```text
ι_{M+1,N+1} : L_{M+1} -> L_{N+1}
```

induces:

```text
[H, ι_{M+1,N+1}] : H ⊛ L_{N+1} -> H ⊛ L_{M+1}.
```

The theorem is:

```text
Scan_N(step, init) ; [H, ι_{M+1,N+1}]
  =
(id_X ⊗ [U, ι_{M,N}]) ; Scan_M(step, init)
```

Here `id_X : X -> X` is the identity morphism on `X`, and
`[U, ι_{M,N}] : U ⊛ L_N -> U ⊛ L_M` restricts the input sequence to its first
`M` entries. This equation says that restricting both the inputs and the output
history of the `N`-step scan gives the `M`-step scan.

This follows by induction on `M` from the base-case and step axioms.

#### 4.7.2 Why Scan still fails the weave criterion

The Scan axioms do not add point naturality for `Scan` along the temporal axis.
They instead specify how temporal points depend on previous temporal points.
Therefore `Scan` remains outside the image of the ordinary lift operation
`act(−, L_N)`.

This is the intended result: `Scan` is a generator with fold laws, not a disguised
weave.

#### 4.7.3 Equality and fusion of scans

Two scans with the same `init` and extensionally equal `step` morphisms are equal
by the base-case and step axioms. More generally, a morphism between state
objects that commutes with two step algebras induces a morphism between the
corresponding scans.

This is the ordinary **algebra homomorphism** principle. An algebra homomorphism
from `step : H ⊗ U -> H` to `step' : H' ⊗ U -> H'` is a morphism

```text
h : H -> H'
```

such that:

```text
(h ⊗ id_U) ; step' = step ; h.
```

When this equation holds, applying `h` to the history produced by `Scan_N(step,
init)` agrees with running `Scan_N(step', init ; h)`.

### 4.8 Broader Scan instances and axiom stability

The general Scan pattern is:

```text
fixed ordered axis + state threaded from one point to the next.
```

The ordered axis may mean different things in different applications.

| Source context | Scan instance | State object `H` | Step input `U` |
| --- | --- | --- | --- |
| Sequence modeling | RNNs, state-space models, autoregressive hidden-state evaluation | hidden state | token/input slice |
| Weight-tied depth | Universal Transformer, ALBERT-style repeated blocks, iterative refinement | layer representation | optional per-depth conditioning |
| Numerical and optimization loops | learned optimizers, fixed-point iterations, recurrent solvers | iterate / optimizer state | gradients, residuals, forcing terms |
| Message-passing rounds | fixed-round GNN or belief-propagation updates | node/edge features | graph-local messages or observations |
| Tensor-network algorithms | sweep-style updates such as DMRG/TEBD at a fixed schedule | local variational state / environment | local tensor or gate data |
| Search and inference | beam search, SMC, particle filters, speculative decoding | beam/particle/candidate state | proposal, likelihood, or accept signal |
| Compiler/runtime strategy | checkpointed loops, `vmap` over independent scans, associative-scan fast paths | activation/resource state | step body and resource annotations |

These broader examples do **not** change the minimal finite deterministic Scan
axioms when they are first-order, fixed-length recurrences. They change the
interpretation of `L_N`, `H`, `U`, or the target algebra implementing
`fold_N`.

For example:

- weight-tied depth uses `L_N` as a depth axis and `H` as the representation
  carried through the repeated block;
- iterative refinement uses `H` as the current estimate and `U` as any external
  forcing or observation;
- fixed-round message passing uses `H` as all node/edge features and the graph
  structure as fixed data inside `step`;
- checkpointed and associative implementations change the algebra's
  implementation of `fold_N`, not the abstract `Scan_N` equations.

Some variants do require refinements beyond the three minimal Scan axioms:

1. **multi-state or coupled scans** replace `H` by a product of state objects, or
   by a multi-sorted state object, and use the same base/step/lift laws on that
   product;
2. **multi-step lookback** replaces `H` by a window object such as `H ⊗ H`, or
   adds several base cases and a higher-arity step law;
3. **non-unit or non-forward schedules** replace the simple ordered object
   `L_N` with a richer schedule object and corresponding predecessor maps;
4. **dynamic or unbounded length** is not finite `Scan_N`; it needs a dependent,
   guarded, or coalgebraic/unfold account;
5. **Search/SMC/speculative decoding** are usually compound: a Scan-like
   propagation step followed by a Route-like prune, resample, or accept step.

Thus the core axioms remain stable for fixed finite first-order recurrences:
base case, inductive step, orthogonal lift distribution, and algebra
preservation. What changes across applications is the meaning of the temporal
object, the shape of the state object, and whether the recurrence is still
finite, first-order, and deterministic.

---

## 5. `Route`: value-dependent indexing

`Route` is not a recurrence. It solves a different problem: selecting where data
goes based on runtime values.

The examples in [future_ideas.md](future_ideas.md) and
[graded_prop.md](graded_prop.md) show that this is broader than gated experts.
Sparse mixture-of-experts is only the smallest worked example. The same
obstruction appears whenever a computation must choose a destination, handler,
edge, block, slot, memory location, branch, or sample ancestor from runtime
values.

### 5.1 Obstruction to being a weave

Let:

```text
E
```

be a finite set of destinations. A destination may be an expert, a memory bank, a
device shard, a branch of an adaptive computation, a storage slot, or any other
indexed place where an item can be sent.

Let:

```text
I
```

be an index object for items, such as tokens in a batch.

A fixed dense routed computation has a routing shape like:

```text
I ⊗ E.
```

Every item is sent to every destination, and a later fixed operation combines or
selects the destination outputs. This is a weave when the destination axis `E`
is fixed structure.

A dynamic routed computation instead computes a runtime routing function:

```text
ρ : I -> E
```

from data-dependent values. The selected destination for item `i` is `ρ(i)`.
Because `ρ` depends on tensor values, it is not a fixed `D`-morphism. Therefore
there is no single reindexing map

```text
η : I -> E
```

available at graph-construction time.

This is the `Route` obstruction:

- `Scan` has fixed reindexing but coupled temporal dependence.
- `Route` has uncoupled per-item execution but data-dependent reindexing.

The map `ρ` is therefore not an object of `D` and not a morphism of `D`.
Instead, it is a **value-level parameter**. A value-level parameter is data
available to the concrete algebra, such as a tensor of route decisions, rather
than symbolic structure available to the graded PROP.

### 5.2 Minimal extension

To model this, use **Para**. The category `Para(C)` has the same objects as
`C`. A morphism from an object `A` to an object `B` is a pair:

```text
(Θ, f : Θ ⊗ A -> B)
```

where `Θ` is a parameter object and `f` is a `C`-morphism using that parameter.
Composition in `Para(C)` threads parameters by tensoring the parameter objects
together.

#### 5.2.1 Route parameter object

For single-destination routing, let:

```text
R_{I,E}
```

be a parameter object whose concrete elements are functions from items to
destinations. In a target algebra, an element

```text
ρ : I -> E
```

of `R_{I,E}` assigns each item `i` a destination `ρ(i)`.

#### 5.2.2 Handler families

Let:

```text
A
```

be a `C`-object representing one item feature vector.

Let:

```text
B
```

be a `C`-object representing one item output vector.

A **handler family** indexed by `E` is a family of `C`-morphisms:

```text
handler_e : A -> B
```

one for each point `e : I_D -> E`. Equivalently, it can be represented as a
single lifted morphism over the destination axis:

```text
handler : A ⊛ E -> B ⊛ E
```

such that evaluating at destination `e` gives `handler_e`:

```text
handler ; ev_{e,B} = ev_{e,A} ; handler_e.
```

Here `ev_{e,A} : A ⊛ E -> A` and `ev_{e,B} : B ⊛ E -> B` are the evaluation maps
derived from the `D`-action.

#### 5.2.3 Route signature

For index object `I`, destination object `E`, item object `A`, output object
`B`, and handler family `{handler_e : A -> B}_{e : I_D -> E}`, add a parameterized
generator:

```text
Route_{I,E}({handler_e}) : A ⊛ I -> B ⊛ I
```

in `Para(C)`, with parameter object:

```text
R_{I,E}.
```

Equivalently, as an ordinary `C`-morphism under the Para encoding, write:

```text
Route^C_{I,E}({handler_e}) : R_{I,E} ⊗ (A ⊛ I) -> B ⊛ I.
```

The object `A ⊛ I` is the collection of item inputs. The object `B ⊛ I` is the
collection of item outputs. The parameter object `R_{I,E}` stores the runtime
route decision.

### 5.3 Axiom 1: itemwise dispatch

The **itemwise dispatch axiom** says that evaluating the routed output at item
`i` is the same as evaluating the input at item `i`, then applying the handler
chosen by the route parameter at `i`.

Let:

```text
i : I_D -> I
```

be a point selecting one item.

Let:

```text
ρ : I -> E
```

be a concrete value of the parameter object `R_{I,E}` in a target algebra. Let:

```text
ρ(i) : I_D -> E
```

be the point of `E` selected by `ρ` at item `i`.

Then the routed morphism specialized at `ρ`, written

```text
Route_ρ({handler_e}) : A ⊛ I -> B ⊛ I,
```

satisfies:

```text
Route_ρ({handler_e}) ; ev_{i,B}
  =
ev_{i,A} ; handler_{ρ(i)}.
```

Here:

- `ev_{i,A} : A ⊛ I -> A` selects the input item at `i`;
- `ev_{i,B} : B ⊛ I -> B` selects the output item at `i`;
- `handler_{ρ(i)} : A -> B` is the handler selected by the runtime route.

This is the Route analogue of the Scan step axiom. For Scan, the point law
relates neighboring time points. For Route, the point law is item-local but
depends on a value-level parameter.

### 5.4 Axiom 2: fixed-route specialization

The **fixed-route specialization axiom** says that when the route is known
statically, `Route` collapses to ordinary `D`-graded structure.

Let:

```text
η : I -> E
```

be a fixed `D`-morphism, available at graph-construction time. Let `ρ_η` be the
corresponding structural parameter value in `R_{I,E}`.

Then:

```text
Route_{ρ_η}({handler_e}) ; ev_{i,B}
  =
ev_{i,A} ; handler_{η(i)}
```

for every point `i : I_D -> I`, where `η(i) : I_D -> E` is the point obtained
by applying `η` to `i`.

Because the points of `I` jointly separate morphisms, this pointwise equality
determines the whole morphism. Thus a fixed route is not genuinely dynamic; it
is recoverable from the ordinary action, reindexing, and handler family.

### 5.5 Axiom 3: orthogonal lift distribution

Let:

```text
P
```

be a `D`-object representing an independent batch, head, sample, or spatial
degree.

Say that `P` is **orthogonal** to `I` and `E` when its axes or index components
are disjoint from the item and destination components. In `St`, this means `P`,
`I`, and `E` share no axis UID.

The route parameter can be lifted pointwise over `P`. Write:

```text
ρ ⊛ P
```

for the value-level route that chooses, for each pair `(i, p)`, the same
destination that `ρ` chooses for `i`. More general implementations may allow the
route to vary over `P`; then the parameter object is `R_{I ⊗ P,E}` instead of
`R_{I,E} ⊛ P`.

The **orthogonal lift-distribution axiom** says:

```text
[Route_ρ({handler_e}), P]
  ≅
Route_{ρ ⊛ P}({[handler_e, P]}).
```

The symbol `≅` denotes a canonical isomorphism in `Para(C)`; it may be
definitional equality in a strict implementation.

This axiom says that batching an independent route is the same as routing
batched handlers. It is the categorical basis for vectorizing sparse routing
across batch coordinates.

### 5.6 Axiom 4: destination relabeling equivariance

Let:

```text
σ : E -> E
```

be an isomorphism in `D`, such as a permutation of the destination axis.

Relabeling destinations should not change the computation, provided the route
labels and handler family are relabeled together. Define the relabeled route:

```text
σ ∘ ρ : I -> E
```

by first applying `ρ : I -> E`, then applying `σ : E -> E`.

Define the relabeled handler family `{handler^σ_e}` by:

```text
handler^σ_{σ(e)} = handler_e.
```

Equivalently:

```text
handler^σ_e = handler_{σ^{-1}(e)}.
```

The **destination relabeling axiom** says:

```text
Route_ρ({handler_e})
  =
Route_{σ ∘ ρ}({handler^σ_e}).
```

This axiom prevents `Route` from depending on arbitrary names or storage order
of destinations. Only the pairing between route decisions and handler
implementations matters.

### 5.7 Algebra semantics

Let `F : C -> V` be an algebra into a target actegory `V`. Extending `F` to the
parameterized setting interprets Para parameters as ordinary target values.

The **Route preservation law** says:

```text
F(Route_ρ({handler_e}))(x)_i
  =
F(handler_{ρ(i)})(x_i).
```

Here:

- `x` is a concrete value of `F(A ⊛ I)`;
- `x_i` is the slice of `x` at item `i`;
- `F(Route_ρ({handler_e}))(x)_i` is the slice of the routed output at item `i`;
- `F(handler_{ρ(i)})` is the concrete operation assigned to the selected
  destination.

This is the runtime dispatch semantics. It is not expressible as a fixed
`D`-morphism unless `ρ` is known before graph construction.

### 5.8 Weighted multi-destination routing

Single-destination routing is the minimal categorical core. Weighted
multi-destination routing adds two pieces of structure.

First, replace `R_{I,E}` by a parameter object:

```text
R_{I,E,K,W}
```

whose concrete values assign to each item `i` a finite list:

```text
((e_{i,1}, w_{i,1}), ..., (e_{i,K}, w_{i,K}))
```

where each `e_{i,j}` is a destination in `E` and each `w_{i,j}` is a scalar
weight in a scalar object `W`.

Second, require `B` to carry a finite weighted-sum operation:

```text
combine_K : (W ⊗ B)^{⊗ K} -> B.
```

Then the itemwise dispatch axiom becomes:

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

The multi-destination variant inherits fixed-route specialization, orthogonal
lift-distribution, destination relabeling equivariance, and algebra preservation,
with the selected handler replaced by the weighted combination of selected
handlers.

### 5.9 Consequences

The Route axioms are enough to recover useful laws without pretending that a
runtime route is a static `D`-morphism.

#### 5.9.1 Static routes are ordinary structure

If the route parameter `ρ` is induced by a fixed `D`-morphism `η : I -> E`, then
the fixed-route specialization axiom identifies `Route_ρ` with the ordinary
graded construction. Thus dense or statically routed computations remain in the
weave fragment.

#### 5.9.2 Dynamic routes are item-local but not structural

The itemwise dispatch axiom says that each output item depends only on the
corresponding input item and the handler chosen for that item:

```text
Route_ρ({handler_e}) ; ev_{i,B}
  =
ev_{i,A} ; handler_{ρ(i)}.
```

This is not point naturality in the ordinary `D`-action, because `ρ(i)` is read
from a value-level parameter. The computation is point-local in `I`, but the
choice of structural destination coordinate is not fixed before runtime.

#### 5.9.3 Destination names are irrelevant

Destination relabeling equivariance implies that optimization passes may reorder,
pack, shard, or rename destinations as long as they transform route labels and
handler implementations together. The observable morphism is unchanged.

#### 5.9.4 Independent batching is safe

Orthogonal lift distribution implies that vectorizing `Route` across an
independent axis `P` does not change semantics:

```text
[Route_ρ({handler_e}), P]
  ≅
Route_{ρ ⊛ P}({[handler_e, P]}).
```

This is the Route counterpart of Scan's batched-loop law.

Scan and Route therefore need different laws. Scan needs finite-fold laws for
coupled temporal dependence. Route needs Para laws for value-dependent indexing.

### 5.10 Broader Route instances and axiom stability

The general Route pattern is:

```text
runtime value decides which destination coordinate is read or written.
```

The destination coordinate may mean different things in different choices of the
index category `D`.

| Source context | Route instance | Destination object `E` | Route parameter `ρ` |
| --- | --- | --- | --- |
| Model-level lifts (`D = Br`) | sparse MoE, adapter-bank or multi-LoRA serving | experts, adapters, or sub-models | item/request -> chosen handler |
| Attention/retrieval | hard attention, dynamic token-to-memory routing | memory slots or key/value entries | query/item -> selected memory location |
| Graph-indexed `D` | per-sample GNNs, meshes, molecules | edges, neighbors, or incidence choices | sample/node -> active neighborhood |
| Partition-lattice `D` | learned pooling, clustering, slot/capsule assignment | blocks, clusters, slots, capsules | item -> learned block/slot |
| Markov or stochastic `D` | particle filters, SMC, sampling, stochastic routing | particles, ancestors, or sampled branches | item/particle -> sampled successor |
| Search/decoding | beam search, speculative decoding accept/reject | beams, candidates, or accept branches | step item -> retained candidate |

These broader examples do **not** change the minimal deterministic Route axioms.
They change the interpretation of `E`, the shape of the parameter object, or the
target algebra.

For example:

- sparse MoE uses `E` as an expert axis and `R_{I,E}` as top-1 or top-k routing
  decisions;
- learned clustering uses `E` as a block or slot axis and the same itemwise
  dispatch law;
- per-sample graph routing uses `E` as an edge or neighbor-choice object;
- stochastic routing refines `R_{I,E}` from deterministic functions `I -> E` to
  samples or kernels, and its algebra preservation law lives in a Markov or
  probabilistic target rather than in deterministic tensors.

Thus the core axioms remain:

1. **itemwise dispatch** after specializing a route parameter;
2. **fixed-route specialization** when the route is induced by a structural
   `D`-morphism;
3. **orthogonal lift distribution** for independent axes;
4. **destination relabeling equivariance**;
5. **algebra preservation** in the chosen target.

What changes across applications is not the logical shape of Route, but the
typing of `E`, the parameter object replacing `R_{I,E}`, and sometimes the
target semantics of "choose" or "combine".

---

## 6. Unified non-weave generator axioms

The preceding Scan and Route axioms suggest a more general perspective. `Scan`
and `Route` are not unrelated exceptions. They are both **non-weave generators**:
distinguished operations whose behavior is determined by pointwise equations,
but whose pointwise equations are not the ordinary naturality law for a lifted
base operation.

The overlap is clearest in this compressed table.

| Aspect | `Scan` | `Route` | Shared abstraction |
| --- | --- | --- | --- |
| Weave obstruction | Fixed temporal axis, but output points are coupled across time. | Item axis is fixed, but destination choice is a runtime value. | The ordinary point-naturality square is the wrong local law. |
| Working category | `C` | `Para(C)`, or a specialization back to `C` after choosing `ρ` | A category `K` containing the generator. |
| Output shape | `H ⊛ L_{N+1}` | `B ⊛ I` | An output object `Y ⊛ S` with point evaluations. |
| Local law | predecessor state plus current input | current input plus runtime-selected handler | Point slices are specified by local equations. |
| Dependency between output points | well-founded predecessor relation on time | no output-point dependency | A point-dependency relation `≺` on `S`. |
| Ordinary-structure boundary | base point `pt_0` uses `init`; prefix restriction is induced by `D`. | static `ρ_η` agrees with fixed `η : I -> E`. | Degenerate/static cases collapse to existing `D`-graded structure. |
| Orthogonal batching | batch commutes with scan | batch commutes with route | independent `D`-axes commute with the generator. |
| Symmetry | only order-preserving symmetries of time survive, usually just identity | destination relabeling is allowed | automorphisms preserving local laws are semantics-preserving. |
| Algebra preservation | compile to `fold_N` | compile to dispatch | algebras interpret the generator by the intended computation. |

### 6.1 Data for a non-weave generator

Let `K` be the category in which the generator lives. For `Scan`, `K = C`. For
`Route`, `K = Para(C)` before specializing the route parameter, and ordinary `C`
after specializing a concrete `ρ`.

A **non-weave generator schema** consists of the following data.

1. An output index object:

   ```text
   S
   ```

   in `D`. Its points are morphisms `s : I_D -> S`.

2. A point-dependency relation:

   ```text
   t ≺ s
   ```

   between points of `S`. The relation is **well-founded**, meaning there is no
   infinite chain

   ```text
   ... ≺ s_2 ≺ s_1 ≺ s_0.
   ```

   This ensures that point equations can define the generator by induction over
   output points. For `Route`, the relation is empty. For `Scan`, `pt'_k ≺
   pt'_{k+1}`.

3. A full domain object:

   ```text
   Z
   ```

   in `K`. For example, `Z = X ⊗ (U ⊛ L_N)` for `Scan`, while `Z = A ⊛ I`
   for a specialized `Route`.

4. A point output object:

   ```text
   Y
   ```

   in `C`, so the full output object is:

   ```text
   Y ⊛ S.
   ```

5. A generator:

   ```text
   G : Z -> Y ⊛ S
   ```

   in `K`, or in `C` after specializing any `Para` parameter.

6. For each point `s : I_D -> S`, a **local evaluator**

   ```text
   Φ_s
   ```

   which computes the output slice at `s` from the external input data and from
   previously defined output slices `G ; ev_{t,Y}` for points `t ≺ s`.

The local evaluators are allowed to use the same ambient routing envelope used
for `Scan`: projections, pairings, and duplicated reads of external inputs are
routing metadata, not new linear PROP primitives.

### 6.2 Unified axiom 1: well-founded point presentation

For every point `s : I_D -> S`, the generator satisfies:

```text
G ; ev_{s,Y} = Φ_s({G ; ev_{t,Y}}_{t ≺ s}, external data).
```

Here `{G ; ev_{t,Y}}_{t ≺ s}` denotes the family of already-defined output
slices on which point `s` depends.

This single axiom specializes to both Scan and Route.

For `Scan`, take:

```text
G = Scan_N(step, init)
S = L_{N+1}
Y = H.
```

At the base point `pt_0`, there are no predecessors, and:

```text
Φ_{pt_0} = π_X ; init.
```

At point `pt'_{k+1}`, the only predecessor is `pt'_k`, and:

```text
Φ_{pt'_{k+1}}
  =
⟨
  (G ; ev_{pt'_k,H}),
  (π_U ; ev_{pt_k,U})
⟩
; step.
```

For `Route`, take:

```text
G = Route_ρ({handler_e})
S = I
Y = B.
```

The dependency relation is empty, and for each item point `i : I_D -> I`:

```text
Φ_i = ev_{i,A} ; handler_{ρ(i)}.
```

So Route is the special case where the point presentation is non-recursive but
parameter-dependent.

### 6.3 Unified axiom 2: ordinary-structure boundary

A non-weave generator should agree with existing `D`-graded structure whenever
its obstruction disappears.

For `Scan`, the boundary is the set of points with no predecessors. In the
first-order finite case this is the base point:

```text
Scan_N(step, init) ; ev_{pt_0,H} = π_X ; init.
```

Prefix restriction is then a theorem: restricting an `N`-step scan to a prefix
gives the smaller scan.

For `Route`, the boundary is the case where the runtime route parameter is
induced by a fixed structural `D`-morphism:

```text
η : I -> E.
```

If `ρ_η` is the route parameter induced by `η`, then:

```text
Route_{ρ_η}({handler_e}) ; ev_{i,B}
  =
ev_{i,A} ; handler_{η(i)}.
```

Thus the unified boundary principle is:

```text
when the non-weave obstruction degenerates to ordinary structure,
the generator agrees with the ordinary D-graded construction.
```

### 6.4 Unified axiom 3: orthogonal lift distribution

Let `P` be a `D`-object orthogonal to the output index object `S` and to any
index objects used by the generator's parameters. Orthogonal means the axes or
index components are disjoint, as in the earlier Scan and Route sections.

The generator must commute with lifting over `P`:

```text
[G, P] ≅ G^P.
```

Here `G^P` is the same generator built from the lifted local data.

For `Scan`:

```text
G^P = Scan_N([step, P], [init, P]).
```

For `Route`:

```text
G^P = Route_{ρ ⊛ P}({[handler_e, P]}).
```

This is the most reusable shared axiom: independent batching is valid even when
the generator is not a weave along its distinguished axis.

### 6.5 Unified axiom 4: relabeling equivariance for harmless symmetries

Let:

```text
σ : S -> S
```

be an isomorphism in `D`. If `σ` preserves the dependency relation `≺` and
transports each local evaluator `Φ_s` to the corresponding evaluator
`Φ_{σ(s)}`, then relabeling output points by `σ` does not change the observable
computation.

For `Scan`, the temporal order on `L_N` is part of the local law. Most
permutations do not preserve predecessor structure, so this axiom is usually
trivial: only the identity, or very special schedule automorphisms, apply.

For `Route`, the item axis `I` is usually kept fixed while the destination axis
`E` is relabeled. This gives the earlier destination relabeling law:

```text
Route_ρ({handler_e})
  =
Route_{σ ∘ ρ}({handler^σ_e}).
```

So the same principle applies, but the nontrivial symmetries live in different
places.

### 6.6 Unified axiom 5: algebra preservation

Let:

```text
F : C -> V
```

be an algebra into a target actegory `V`, extended to `Para(C)` when the
generator is parameterized. The algebra must interpret the generator as the
intended target computation:

```text
F(G) = semantic_G.
```

For `Scan`:

```text
semantic_G = fold_N(F(step), F(init)).
```

For `Route`:

```text
semantic_G(x)_i = F(handler_{ρ(i)})(x_i).
```

This axiom is what lets backend rewrites and implementation strategies be
checked against one abstract meaning.

### 6.7 What actually factors out?

The factored common structure is not "Scan and Route are the same operation."
They are not. What factors out is the interface for adding generators that fail
the ordinary weave criterion:

1. a point-indexed output object `Y ⊛ S`;
2. a well-founded point-presentation law;
3. a boundary case that agrees with ordinary `D`-graded structure;
4. orthogonal lift distribution;
5. relabeling equivariance for symmetries preserving the local laws;
6. algebra preservation.

The part that remains operation-specific is the local evaluator `Φ_s`.

```text
Scan:  Φ_s may read earlier output slices.
Route: Φ_s may read value-level routing parameters.
```

This is the useful unification: a future implementation or Lean development can
share the scaffolding for non-weave generators while keeping the local equations
specific to the generator being added.

---

## 7. What is deliberately not included

The minimal Scan and Route extensions intentionally exclude several larger
features.

**Unbounded generation.** The present `Scan_N` is finite. An unbounded generator
would need a coalgebraic account, such as an anamorphism or guarded corecursion.
That is a different foundation.

**Dynamic iteration count.** The natural number `N` is fixed by the temporal
object `L_N`. Runtime-dependent loop lengths need a dependent or effectful
extension.

**Multi-step lookback.** A recurrence such as

```text
H[l + 1] = step(H[l], H[l - 1], U[l])
```

can be represented by replacing the state `H` with a window object
`H ⊗ H`, or by adding `k` base cases and a `k`-history step law. This is a
straightforward extension, but it is not part of the minimal account.

**Gauss-Seidel coupled updates.** In a Gauss-Seidel update, later equations in a
single step may read values already updated earlier in the same step. This makes
the semantics depend on equation order. The minimal account instead supports
Jacobi-style coupled updates, where all right-hand sides read the old state and
all left-hand sides write the new state.

**Route capacity management.** Practical routed systems often
add per-destination capacity limits, overflow handling, padding, and load-balancing
losses. These are not part of the minimal `Route` generator. They can be modeled
by refining the parameter object from `R_{I,E}` to a richer object that records
destination slots, dropped items, or auxiliary loss terms, but the core itemwise
dispatch axiom remains the semantic anchor.

**Router-to-route compilation.** A neural gate, hash rule, retrieval policy, or
controller may produce scores or choices, after which a top-k, sampling, or
decision rule produces route decisions. This score-to-decision step is not the
`Route` generator itself. In the minimal account, `Route` starts once the
value-level parameter `ρ` has been produced.

---

## 8. Lean shape

In Lean, the minimal extension can be represented by adding a `scan` constructor
to the inductive type of Br morphisms, plus three theorem fields or axioms.

The following is schematic Lean, not checked code.

```lean
-- H, U, and X are Br objects.
-- step is the state update.
-- init is the initial-state map.
-- N is the finite number of steps.
scan :
  (step : BrHom (H.tensor U) H) ->
  (init : BrHom X H) ->
  (N : Nat) ->
  BrHom (X.tensor (actObj U (L N))) (actObj H (L (N + 1)))
```

The base-case law states that evaluating at output time `0` gives the initial
state.

```lean
scan_base :
  evalOutputTime (scan step init N) 0 = projectX.comp init
```

The step law states that evaluating at output time `k + 1` gives the step
applied to output time `k` and input time `k`.

```lean
scan_step :
  k < N ->
  evalOutputTime (scan step init N) (k + 1) =
    pair
      (evalOutputTime (scan step init N) k)
      (projectInputs.comp (evalInputTime k))
    |>.comp step
```

The lift-distribution law states that independent batching commutes with scan.

```lean
scan_lift :
  Orthogonal P (L N) ->
  actHom (scan step init N) P =
    scan (actHom step P) (actHom init P) N
```

The algebra preservation field extends the existing algebra structure.

```lean
scan_preserve :
  F.map (scan step init N) =
    foldN (F.map step) (F.map init) N
```

Route is represented as a parameterized generator. The parameter object
`RouteParam I E` stores value-level assignments of items to destinations.

```lean
route :
  (handlers : HandlerFamily A B E) ->
  BrParaHom (actObj A I) (actObj B I)
```

The itemwise dispatch law states that specializing the route parameter to `rho`
and evaluating item `i` applies the handler selected by `rho i`.

```lean
route_item :
  evalItem (route handlers).specialize rho i =
    evalItemInput i |>.comp (handlers.at (rho i))
```

The fixed-route law states that a route induced by a static `D`-morphism reduces
to the corresponding ordinary reindexing construction.

```lean
route_fixed :
  (eta : DHom I E) ->
  (route handlers).specialize (rhoOf eta) =
    fixedRoute eta handlers
```

The lift-distribution law states that independent batching commutes with route.

```lean
route_lift :
  Orthogonal P I ->
  Orthogonal P E ->
  actParaHom (route handlers).specialize rho P =
    (route (handlers.act P)).specialize (rho.act P)
```

The relabeling law states that permuting destination names changes nothing when
the route labels and handler family are permuted together.

```lean
route_relabel :
  (sigma : DIso E E) ->
  (route handlers).specialize rho =
    (route (handlers.relabel sigma)).specialize (sigma.comp rho)
```

The algebra preservation field says that Route compiles to runtime dispatch.

```lean
route_preserve :
  F.map ((route handlers).specialize rho) x i =
    F.map (handlers.at (rho i)) (x i)
```

The prefix theorem is then proved by induction on the prefix length using
`scan_base` and `scan_step`.

---

## 9. Summary

The minimal foundation for `Scan` is:

1. Add `Scan_N(step, init)` as a generator with type
   `X ⊗ (U ⊛ L_N) -> H ⊛ L_{N+1}`.
2. Add the base-case axiom: the output at time `0` is `init`.
3. Add the step axiom: the output at time `k + 1` is `step` applied to output
   time `k` and input time `k`.
4. Add the orthogonal lift-distribution axiom:
   `[Scan_N(step, init), P] ≅ Scan_N([step, P], [init, P])`.
5. Extend algebras with the preservation law:
   `F(Scan_N(step, init)) = fold_N(F(step), F(init))`.

This is smaller than adding a new directed action or a full temporal category.
The existing `D`-action already supplies point evaluation, prefix restriction,
and ordinary batching. `Scan` only needs the finite fold laws that explain how
its temporal points depend on earlier temporal points.

The minimal foundation for single-destination `Route` is:

1. Put `Route` in `Para(C)`, with parameter object `R_{I,E}` storing runtime
   assignments `ρ : I -> E`.
2. Add the itemwise dispatch axiom:
   `Route_ρ({handler_e}) ; ev_{i,B} = ev_{i,A} ; handler_{ρ(i)}`.
3. Add the fixed-route specialization axiom: when `ρ` comes from a static
   `D`-morphism, `Route` agrees with the ordinary reindexing construction.
4. Add the orthogonal lift-distribution axiom:
   `[Route_ρ({handler_e}), P] ≅ Route_{ρ ⊛ P}({[handler_e, P]})`.
5. Add destination relabeling equivariance:
   `Route_ρ({handler_e}) = Route_{σ ∘ ρ}({handler^σ_e})`.
6. Extend algebras with the preservation law:
   `F(Route_ρ({handler_e}))(x)_i = F(handler_{ρ(i)})(x_i)`.

Thus `Scan` and `Route` require different minimal extensions. `Scan` is a
finite catamorphism over a fixed temporal object. `Route` is value-dependent
indexing over a fixed family of destination handlers, and its categorical home is
the parameterized `Para` layer rather than the ordinary `D`-graded lift.

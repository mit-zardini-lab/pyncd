# Scan and Route in a Graded PROP

This note refines the `Scan` discussion in [graded_prop.md](graded_prop.md) and
[iteration.md](iteration.md). Its goal is narrow: identify the smallest extension
to the `D`-graded colored PROP formalism that gives `Scan` a categorical
foundation, and contrast it with the different obstruction that leads to
`Route`.

Every symbol used below is introduced before it is used. The notation follows
[graded_prop.md](graded_prop.md): composition is written in diagrammatic order,
so `f ; g` means "first `f`, then `g`".

---

## Contents

1. [Base setting](#1-base-setting)
2. [The weave criterion](#2-the-weave-criterion)
3. [Why `Scan` is not a weave](#3-why-scan-is-not-a-weave)
4. [Minimal extension for `Scan`](#4-minimal-extension-for-scan)
5. [Algebra semantics](#5-algebra-semantics)
6. [Consequences](#6-consequences)
7. [`Route`: the other obstruction](#7-route-the-other-obstruction)
8. [What is deliberately not included](#8-what-is-deliberately-not-included)
9. [Lean shape](#9-lean-shape)
10. [Summary](#10-summary)

---

## 1. Base setting

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
   act : C x D^op -> C
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
ev_p,X : X ⊛ P -> X
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

gives `ev_p,X`. Equivalently, `ev_p,X` is `[X, p]` followed by
`υ_X`. When `X` is clear, we write this as `ev_p`.

---

## 2. The weave criterion

A **weave** is the data witnessing that a `C`-morphism is a lifted base operation
over an index shape. More explicitly, a morphism

```text
g : X -> Y
```

is a weave over a `D`-object `P` when it factors as

```text
g = [f, P] ; ρ
```

where:

- `f : X0 -> Y0` is a base operation in `C`;
- `[f, P]` is the lift of `f` over `P`;
- `ρ` is assembled from reindexing maps and symmetry maps.

The key test for being a weave is **point naturality**. A lifted operation must
satisfy, for every point `p : I_D -> P`,

```text
[f, P] ; ev_p,Y = ev_p,X ; f.
```

This equation says that evaluating a lifted operation at point `p` is the same
as evaluating the inputs at `p` and then running the base operation. In
[graded_prop.md](graded_prop.md), this is Eq. 3.

The intuition is simple: a weave is pointwise independent over its degree
coordinates. If a morphism mixes different points of the degree axis, it cannot
be a weave over that axis.

---

## 3. Why `Scan` is not a weave

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
Scan ; ev_{l+1}  !=  ev_l ; step
```

as a pointwise lifted operation. The right side is not even well-typed without
the previously accumulated state.

The obstruction is **not** that time is data-dependent. The time reindexing is
fixed and structural. The obstruction is that the morphism couples different
positions of the time axis. This is different from `Route`, where the problem is
that the reindexing itself depends on runtime data.

---

## 4. Minimal extension for `Scan`

The minimal extension is to add `Scan` as a distinguished generator with three
axioms. No new structure is needed on `D`, and no new action functor is needed
beyond the existing `act : C x D^op -> C`.

### 4.1 Temporal objects

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

### 4.2 Step and initial morphisms

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

### 4.3 Scan signature

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

### 4.4 Axiom 1: base case

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
Scan_N(step, init) ; ev_{pt_0}
  =
π_X ; init.
```

Here:

- `π_X : X ⊗ (U ⊛ L_N) -> X` is the projection that keeps the `X` component and
  discards the input-sequence component;
- `ev_{pt_0} : H ⊛ L_{N+1} -> H` evaluates the output history at time `0`.

The projection is the only reason this equation is not a bare PROP equation.
The categorical content is the base-case equality; the projection just adapts
the equality to pyncd's multi-input calling convention.

### 4.5 Axiom 2: inductive step

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
Scan_N(step, init) ; ev_{pt'_{k+1}}
  =
⟨
  (Scan_N(step, init) ; ev_{pt'_k})
,
  (π_U ; ev_{pt_k})
⟩
; step.
```

Here:

- `π_U : X ⊗ (U ⊛ L_N) -> U ⊛ L_N` keeps the per-step input sequence;
- `ev_{pt_k} : U ⊛ L_N -> U` selects the input at time `k`;
- `ev_{pt'_k} : H ⊛ L_{N+1} -> H` selects the output state at time `k`;
- `ev_{pt'_{k+1}} : H ⊛ L_{N+1} -> H` selects the output state at time `k + 1`;
- `⟨-, -⟩` pairs two reads from the same scan input so that their outputs can be
  passed together to `step : H ⊗ U -> H`.

This equation is a finite catamorphism law. It does not require an infinite
initial algebra or a fixpoint object because `N` is finite.

### 4.6 Axiom 3: orthogonal lift distribution

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

## 5. Algebra semantics

A **target actegory** is a symmetric monoidal category `V` equipped with a right
action of `D`, written:

```text
⊛_V : V x D^op -> V.
```

For pyncd, `V` is the category of PyTorch tensor spaces and tensor functions;
`A ⊛_V P` appends the dimensions described by `P` to the tensor space `A`.

An **algebra** of `C` in `V` is a strong symmetric monoidal functor:

```text
F : C -> V
```

that preserves the `D`-action up to coherent isomorphism. In pyncd,
`F` is `construct()`: it maps abstract Br morphisms to concrete PyTorch modules.

To support `Scan`, the algebra must satisfy one additional preservation law:

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

## 6. Consequences

The three Scan axioms are enough to recover the useful laws without treating
`Scan` as an opaque generator.

### 6.1 Prefix restriction

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

### 6.2 Why Scan still fails the weave criterion

The Scan axioms do not add point naturality for `Scan` along the temporal axis.
They instead specify how temporal points depend on previous temporal points.
Therefore `Scan` remains outside the image of the ordinary lift operation
`act(−, L_N)`.

This is the intended result: `Scan` is a generator with fold laws, not a disguised
weave.

### 6.3 Equality and fusion of scans

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

---

## 7. `Route`: the other obstruction

`Route` is not a recurrence. It solves a different problem: selecting where data
goes based on runtime values.

Let:

```text
E
```

be a finite set of experts or destinations.

Let:

```text
I
```

be an index object for items, such as tokens in a batch.

A fixed dense mixture-of-experts layer has a routing shape like:

```text
I ⊗ E.
```

Every item is sent to every expert, and a gate later combines the expert
outputs. This is a weave when the expert axis `E` is fixed structure.

A sparse top-1 mixture-of-experts layer instead computes a routing function:

```text
r : I -> E
```

from data-dependent gate values. The selected expert for item `i` is `r(i)`.
Because `r` depends on tensor values, it is not a fixed `D`-morphism. Therefore
there is no single reindexing map

```text
η : I -> E
```

available at graph-construction time.

This is the `Route` obstruction:

- `Scan` has fixed reindexing but coupled temporal dependence.
- `Route` has uncoupled per-item execution but data-dependent reindexing.

A minimal `Route` generator would have a type of the form:

```text
Route(r, experts) : Items -> Outputs
```

where:

- `Items` is the `C`-object carrying item features;
- `Outputs` is the `C`-object carrying routed outputs;
- `experts` is a family of `C`-morphisms indexed by `E`;
- `r` is a value-level routing parameter or gate, not a `D`-morphism.

The correct categorical home for `Route` is therefore not the ordinary
`D`-action. It belongs in the parameterized setting, where the route map is
carried as data in a `Para` morphism. Here **Para** means the category whose
morphisms from `A` to `B` are pairs:

```text
(P, f : P ⊗ A -> B)
```

where `P` is a parameter object and `f` is a morphism using that parameter.

For `Route`, the parameter object contains the gate values or routing decisions.
This is why `Route` is not made first-class by the Scan axioms. It requires a
separate account of value-dependent indexing.

---

## 8. What is deliberately not included

The minimal Scan extension intentionally excludes several larger features.

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

**Route laws.** This document identifies why `Route` is not a weave, but it does
not give a complete algebraic theory of value-dependent routing.

---

## 9. Lean shape

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

The prefix theorem is then proved by induction on the prefix length using
`scan_base` and `scan_step`.

---

## 10. Summary

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

`Route` remains separate. It is not a coupled temporal fold; it is a
value-dependent reindexing whose routing map is not a fixed `D`-morphism. Its
home is the parameterized `Para` layer, not the ordinary `D`-graded lift.

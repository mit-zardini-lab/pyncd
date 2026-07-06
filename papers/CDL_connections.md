# CDL Connections to the `D`-Graded Colored PROP

This note relates **Categorical Deep Learning** (**CDL**, arXiv:2402.15332) to
the pyncd/leanncd `D`-graded colored PROP framework. Write `D` for the
index/shape category, `C` for the operation category, and `V` for the concrete
target category used in compilation. The semantics functor `F : C -> V` is the
categorical content of `construct()`: it maps symbolic tensor programs to
executable objects while preserving the structure pyncd and leanncd require.

The useful split is:

- `D` records index geometry: axes, sizes, layouts, masks, permutations,
  windows, tiles, prefixes, and reindexings.
- `C` records computation: typed tensor wires, generators, composition, tensor
  product, symmetry, `Scan`, `Route`, and architecture equations.
- `V` executes the program: PyTorch modules/tensors today, or a Lean target
  category in leanncd.

Composition `f ; g` means "first `f`, then `g`". The monoidal product `⊗`
places wires or operations side by side. The symbol `I_D` is the unit object of
`D`, and `I_C` is the unit object of `C`.

## Table of contents

- [1) Index semantics (`D`)](#1-index-semantics-d)
  - [Symmetry-focused path](#symmetry-focused-path)
    - [Invariance/equivariance under a symmetry monad](#invarianceequivariance-under-a-symmetry-monad)
    - [Symmetry DSL and scoped semantics](#symmetry-dsl-and-scoped-semantics)
    - [What annotations add beyond raw indexing](#what-annotations-add-beyond-raw-indexing)
    - [Static symmetry checking in the pipeline](#static-symmetry-checking-in-the-pipeline)
    - [Useful `D`-side symmetry structures](#useful-d-side-symmetry-structures)
    - [Concrete symmetry DSL sketches](#concrete-symmetry-dsl-sketches)
  - [Other `D`-monad examples (non-symmetry)](#other-d-monad-examples-non-symmetry)
- [2) Computation semantics (`C`)](#2-computation-semantics-c)
  - [`Scan` as a parametric-monad algebra](#scan-as-a-parametric-monad-algebra)
- [3) Semantic bridge (`C -> V`)](#3-semantic-bridge-c---v)
  - [Equivariance-as-factorization theorem](#equivariance-as-factorization-theorem)
  - [`Para` and parameters](#para-and-parameters)
  - [LeanNCD practical next steps](#leanncd-practical-next-steps)
- [4) Compiler geometry and data plumbing](#4-compiler-geometry-and-data-plumbing)

## 1) Index semantics (`D`)

`D` is the **index semantics**. Its objects, written `P ∈ Ob(D)`, are shapes or
index spaces; its morphisms are structural maps between shapes. In pyncd's main
instance, `D = St`: objects are tuples of axes, and morphisms are stride/affine
reindexings, projections, permutations, and related layout maps.

A monad on `D` is an endofunctor with unit and multiplication,
`(T_D, η^D, μ^D)`, that packages reusable index-level structure: how grades are
transformed and how those transforms compose.

Possible `T_D` choices include symmetry monads (permutation/group/monoid
actions), locality monads (window/neighborhood), coarsening/quotient monads,
layout/reshape monads, and masking/subset monads.

The demarcation is by **what index structure the monad represents**: symmetry
monads encode index actions treated as "same up to transform," while the other
families encode non-symmetry structure (locality, quotienting, layout, masks,
etc.). Equivariance/invariance are then **properties of maps relative to a
chosen action**; they do not by themselves define whether a monad is a symmetry
monad.

In what follows, we go deeper on **symmetry-oriented** choices, since they
bridge directly to equivariance/invariance obligations and static checking.

### Symmetry-focused path

We start with monads that explicitly model symmetries (permutation, group,
monoid, fiber, and product actions), show concrete DSL declarations for them,
and then formalize those examples with local and monadic equivariance laws.
After that, we transition to other useful `D`-monad families that are
index-structural but not symmetry monads.

Companion container-indexed view: [generalized_tensors.md §10](../generalized_tensors.md#10-symmetry-actions-on-container-positions).
Useful crosswalk:
- core action-on-indices idea: [§10.1](../generalized_tensors.md#101-core-idea-symmetries-act-on-poscs),
- permutation/set symmetry: [§10.2](../generalized_tensors.md#102-setlist-symmetry-permutation-equivariance),
- translation/grid symmetry: [§10.3](../generalized_tensors.md#103-grid-symmetry-translation-equivariance),
- graph automorphism symmetry: [§10.4](../generalized_tensors.md#104-graphcontainer-automorphism-symmetry),
- scoped mask/sum constraints: [§10.5](../generalized_tensors.md#105-masksum-containers-and-scoped-symmetry),
- compiler obligations/checks: [§10.6](../generalized_tensors.md#106-compiler-obligations-from-the-combined-view).

#### Invariance/equivariance under a symmetry monad

**Without extra structure.** Without imposing a global target monad, a program
can still state equivariance or invariance for the concrete symmetries it
mentions. First choose the relevant `D`-side reindexings, such as automorphisms
`ρ(π) : P -> P` for an axis object `P`, and the induced action on the program's
tensor objects. The operation category `C` is acted on by `D` through a right
action

```text
act : C × D^op -> C,
```

where `×` is category product and `D^op` is the opposite category of `D`. For a
`C`-object `X` and a `D`-object `P`, write

```text
X ⊛ P := act(X, P)
```

for `X` lifted over `P`. For a `C`-morphism `f : X -> Y`, write
`[f, P] : X ⊛ P -> Y ⊛ P` for the lifted operation. For a `D`-morphism
`η : P -> Q`, write `[X, η] : X ⊛ Q -> X ⊛ P`; the direction reverses because
the action uses `D^op`. A category equipped with such a coherent action is a
`D`-actegory.

For a local symmetry check, use the concrete symmetry actions on the current
inputs and outputs. Here `F` denotes the map computed by the checked object
(a single statement, or the full composed program):

```text
equivariant: F(π · x) = π · F(x)
invariant:   F(π · x) = F(x).
```

This is often enough for DSL checking: axis declarations specify admissible
`D`-morphisms, tensor declarations specify how each object transforms, and a
statement assertion asks the generated `C`-morphism or compiled map in `V` to
commute with those chosen maps.

Connection to generalized containers: this is the same commuting-law notion used
in [generalized_tensors.md §10.1](../generalized_tensors.md#101-core-idea-symmetries-act-on-poscs),
where actions are phrased directly on `Pos(C,s)`.

**Extra structure.** To turn these local laws into a uniform monadic semantics,
impose extra structure. Let `T_D` be the chosen `D`-side symmetry monad. A
target-side `T_V : V -> V` is **not** automatic transport of `T_D`; it must be
supplied with compatible data:

1. a monad on grades `(T_D, η^D, μ^D)` on `D`;
2. a monad `(T_V, η^V, μ^V)` on `V`;
3. a `D`-action on `V`, written again as `X ⊛ P`;
4. a comparison, usually an isomorphism, natural in `X ∈ V` and `P ∈ D`,
   ```text
   λ_{X,P} : T_V(X ⊛ P) ≅ (T_V X) ⊛ T_D(P).
   ```

The comparison `λ` must satisfy distributive-law coherence: naturality in `X`
and `P`, compatibility with the units `η^V, η^D`, and compatibility with the
multiplications `μ^V, μ^D`. Because the action uses `D^op`, the `P` argument is
contravariant; orient `λ` accordingly, or use an equivalent op/comonad
presentation. Equivalently, `T_V` is a coherent lift of `T_D` through the
target `D`-action.

With that lift, a `T_V`-action on `X ∈ V` is a map

```text
α_X : T_V(X) -> X
```

satisfying the monad action laws. A map `f : X -> Y` is `T_V`-equivariant when
it commutes with the actions:

```text
f ∘ α_X = α_Y ∘ T_V(f) : T_V(X) -> Y.
```

Invariance is the special case where the codomain carries the trivial action.
When a fixed-point object for the induced symmetry action on `Y` is available,
outputs may instead be required to land in `Fix_T(Y)`, meaning the subobject
(or object of points) of `Y` that is unchanged by the chosen symmetry action.

**What this buys.** Local equations for particular `π` become
instances of one `T_V`-algebra law, compiled semantics can target the
Eilenberg--Moore category `V^{T_V}`, and factorization statements become
available: an algebra `F : C -> V` is equivariant exactly when it factors as
`F = U ∘ F~` through the forgetful functor `U : V^{T_V} -> V`, subject to the
same monoidal and `D`-actegory compatibility conditions.

One caveat belongs here: a monad on `D` does **not** automatically induce a
monad on `C` or `V`. To lift a `D`-side symmetry into operation- or
target-level semantics, one must supply coherent compatibility data such as

```text
T_C(X ⊛ P) ≅ T_C(X) ⊛ T_D(P),
```

and similarly on `V` and for the semantic functor `F : C -> V`. Without such
lift data, a `D`-side symmetry is only index metadata, not yet a theorem about
compiled maps. In simple cases, such as DeepSets, the group action may already
be internal to `D` via automorphisms `ρ(g) : P -> P`; then the broadcasted part
is equivariant by reindexing functoriality, while invariance still requires a
symmetric output aggregation such as sum, mean, or max.

With this in place, the DSL question is where users should declare `D`-side
symmetries and the required `C`-side equivariance/invariance behavior so these
lifting and reindexing semantics can be enforced automatically.

#### Symmetry DSL and scoped semantics

The surface DSL should stay generic, with symmetry claims scoped to the chosen
`D`-side structure. A useful split is:

- **Axis defaults constrain the index action.** They declare admissible `D`
  morphisms for an axis, such as permutations of a finite set axis.
- **Tensor overrides constrain the tensor action.** They say how a tensor's
  components transform under those index actions: inherited, trivial,
  invariant, equivariant, or with a named fiber action.
- **Statement assertions constrain the operation obligation.** They require a
  definition, contraction, reduction, `Scan`, or `Route` to commute with the
  declared actions and generate the proof/lowering side conditions.

This same scoped split is mirrored in the container-focused presentation:
[generalized_tensors.md §10.2](../generalized_tensors.md#102-setlist-symmetry-permutation-equivariance)
through [§10.5](../generalized_tensors.md#105-masksum-containers-and-scoped-symmetry),
with explicit treatment of permutation, translation, graph automorphisms, and
mask/sum scope boundaries.

For example, a permutation-scoped DSL could look like this. Write `ρ(π)`
for the `D`-morphism induced by a permutation `π`.

```lean
axis i : ℕ = n [symmetry permutation]     -- index action: ρ(π) : i -> i
axis d : ℕ = k                            -- no default symmetry
axis h : ℕ = m                            -- no default symmetry

tensor X(i, d)                            -- tensor action: inherits i ↦ π i
tensor W(d, h) [symmetry none]            -- tensor action: trivial in i
tensor Y(i, h) [equivariant i]            -- tensor action: output permutes on i

assert equivariant(i):                    -- operation obligation
  Y[i, h] := X[i, d] · W[d, h]

assert invariant(i):                      -- operation obligation
  z[h] := reduce_sum(i)(Y[i, h])
```

*What this constrains:* relabeling elements along axis `i` must relabel tensor
outputs consistently. The map `Y[i, h] := X[i, d] · W[d, h]` must commute with
permutations of `i`, while `z[h]` must be unchanged by those permutations. This
forbids hidden `i`-specific behavior unless explicitly declared.

*Learning impact:* the model is trained in a symmetry-constrained hypothesis
class. Shared parameters such as `W[d, h]` receive gradient contributions from
all `i` positions, improving statistical efficiency. Position-specific
memorization on `i` is excluded unless introduced with explicit
non-equivariant annotations.

The three layers are distinct and compositional:

1. **Axis declaration = index action in `D`.**  
   `axis i ... [symmetry permutation]` introduces admissible reindexings on the
   `i`-axis. For each permutation `π`, there is a map `ρ(π) : P -> P` in `D`.
   This layer defines *which coordinate relabelings exist*.

2. **Tensor declaration = representation on tensor spaces.**  
   A declaration such as `tensor X(i, d)` picks how `X` responds to `ρ(π)`.
   Standard indexed tensors use pullback:
   `(π · X)[i, d] = X[ρ(π)⁻¹(i), d]`.
   Trivial or invariant tensors satisfy `π · W = W`.  
   This layer defines *how each object transforms*.

3. **Statement assertion = map-level law in `C`.**  
   A statement induces a map `F`. `assert equivariant(i)` requires
   `F(π · x) = π · F(x)`. `assert invariant(i)` requires `F(π · x) = F(x)`.
   For equivariance, this is a commuting square:
   `F ∘ α_in(π) = α_out(π) ∘ F`.  
   This layer defines *what must be proved/checked about the operation*.

This is the elementwise form of the monadic statement above: we wrote
actions as `α_X : T_V(X) -> X` and equivariance as
`f ∘ α_X = α_Y ∘ T_V(f)`. Here, fixing one symmetry element `π`, the maps
`α_in(π)` and `α_out(π)` are the concrete action maps on input/output tensor
spaces, and `F(π · x) = π · F(x)` is the same commuting condition at the
program-equation level.

For the running example, let `F` be the morphism denoted by
`Y[i, h] := X[i, d] · W[d, h]`, and let `R` be the morphism denoted by
`z[h] := reduce_sum(i)(Y[i, h])`. Then `assert equivariant(i)` checks
`F(π · X, W) = π · F(X, W)`, while `assert invariant(i)` checks
`R(π · Y) = R(Y)`.

Ordinary syntax can stay uniform:

```lean
Y[i, j] := W[p, r] · X[i + p, 2 * j + r]          -- affine reindex/gather
Out[2 * i, 2 * j] := X[i, j]                      -- affine LHS scatter
A[q, s] := softmax(where s ≤ q)(Q[q, d] · K[s, d])
S[j, l + 1] := S[j, l] · A[j, k]                  -- prefix/scan syntax
```

Semantics, proofs, and lowering rules are attached per monad or mixin. The DSL
can therefore describe affine reindexing, masks, predicates, and prefix
recurrences uniformly, while permutation equivariance is checked only against
the chosen permutation structure and its compatibility data.

#### What annotations add beyond raw indexing

The current DSL already lets you write permutations and reindexings
syntactically. Symmetry annotations add a stronger claim: a **semantic
contract**.

- Without annotation: `i` is just an index variable.
- With annotation (for example permutation on `i`): the program must commute with
  that action, and this becomes a compile-time/proof obligation.

Plain-English reading of

```lean
axis i : ℕ = n [symmetry permutation]
tensor X(i, d)
```

under permutation symmetry on `i`: if examples are relabeled by a permutation
`π`, `X` must reorder along axis `i` in the same way; axis `d` is unaffected.

For learning, this changes the hypothesis class. In

```lean
Y[i, h] := X[i, d] · W[d, h]
```

`W[d, h]` is already shared across `i` in this equation. The symmetry annotation
turns that into an explicit contract: introducing an `i`-dependent parameter
(`W[i, d, h]`) would violate the declared equivariance unless separately justified.
During training, gradients from all `i` slices update that one shared `W`, so
updates remain inside the declared symmetry class.

This motivates the next step: make those contracts mechanically checkable at
compile time rather than leaving them as design intentions.

#### Static symmetry checking in the pipeline

A practical checker should be conservative and return one of:

- `PASS`: symmetry is proved preserved,
- `FAIL`: symmetry is proved violated,
- `UNKNOWN`: requires extra annotation/proof.

The checker runs on the **entire tensor logic program** (all declarations and
statements) and then discharges obligations statement-by-statement and
operator-by-operator.

Place it after structural lowering/unification and before final backend
lowering.

Direct companion checklist:
[generalized_tensors.md §10.6](../generalized_tensors.md#106-compiler-obligations-from-the-combined-view),
which enumerates typed action checks, commuting checks, invariance checks,
scatter coherence, scoped-branch checks, and diagnostics.

Typical local checks:

- elementwise ops: preserve when input/output actions match;
- reductions: invariant only with symmetry-compatible reduction axis/op;
- masks/predicates: `where`/Iverson conditions must be invariant under the action;
- reindex/scatter/slice: must commute with the declared action;
- learned weights: require tying/structure annotations for equivariance.

Typical failures:

- `First[] := X[0]` under permutation symmetry on `i`;
- causal mask `s ≤ q` claimed invariant under full sequence permutation.

Typical passes:

- `z[h] := reduce_sum(i)(Y[i, h])` under permutation on `i`;
- shared-weight tokenwise map `Y[i, h] := X[i, d] · W[d, h]`.

#### Useful `D`-side symmetry structures

Common symmetry-oriented choices for `T_D` are:

| `T_D : D -> D` | Index meaning | pyncd/leanncd use |
| --- | --- | --- |
| Permutation | reorder a finite axis | batch-equivariant layers, set pooling |
| Group action | attach invertible symmetries | translations, rotations, steerable CNNs |
| Monoid action | allow non-invertible actions | shifts, crops, padding, streaming updates |
| Fiber/action | move base indices and feature fibers together | gauge-equivariant or steerable features |
| Product-action | combine independent symmetries | batch permutation plus spatial translation |

#### Concrete symmetry DSL sketches

These snippets use proposed extension syntax; exact DSL spelling remains
unsettled.

For worked container-indexed examples in the same style (GNN permutation
equivariance/invariance and 2D CNN translation equivariance), see
[generalized_tensors.md §10.6](../generalized_tensors.md#106-compiler-obligations-from-the-combined-view).

**Group action (translations).**
```lean
axis x : ℤ = w [symmetry group_action ℤ]
axis c : ℕ = cin
axis h : ℕ = cout
axis r : ℤ = k [window centered]

tensor X(x, c) [equivariant x]
tensor K(r, c, h) [symmetry none]
tensor Y(x, h) [equivariant x]

assert equivariant(x):
  Y[x, h] := X[x + r, c] · K[r, c, h]
```
*What this constrains:* shifting input along `x` must shift output along `x`
by the same amount. The kernel `K` is shared across positions, so there is no
position-specific `K[x, r, c, h]`; implementations must commute with
translations.

*Learning impact:* optimization uses shared convolutional parameters. Gradients
from translated occurrences update the same kernel entries, improving sample
efficiency and preventing absolute-position shortcuts.

**Monoid action (one-sided shifts/crops).**
```lean
axis t : ℕ = n [symmetry monoid_action shift]
axis d : ℕ = k

tensor X(t, d) [equivariant t]
tensor H(t, d) [equivariant t]
tensor S(d) [symmetry none]

assert equivariant(t):
  H[t + 1, d] := S[d] · H[t, d] + X[t, d]
```
*What this constrains:* valid forward time shifts must map to matching shifts in
outputs and states. Because this is a monoid action, the model
respects composable forward transforms, not inverses, and `S[d]` cannot depend
on absolute time `t`.

*Learning impact:* recurrent update parameters are tied across time. Each step
contributes to the same parameter updates, and training cannot fit
time-index-specific weights unless explicitly declared.

**Fiber/action symmetry (base + feature transform).**
```lean
axis x : ℤ = w [symmetry group_action ℤ]      -- base action (e.g. translation)
axis c : ℕ = cin [symmetry fiber_action ρin]  -- feature-fiber representation
axis h : ℕ = cout [symmetry fiber_action ρout]
axis r : ℤ = k [window centered]

tensor X(x, c) [equivariant (x, c)]
tensor K(r, c, h) [equivariant_fiber (ρin -> ρout)]
tensor Y(x, h) [equivariant (x, h)]

assert equivariant((x, fiber)):
  Y[x, h] := X[x + r, c] · K[r, c, h]
```
*What this constrains:* symmetry acts on both the base index `x` and feature
channels `c, h`. Under a group transform, input features transform by `ρin`,
outputs by `ρout`, and kernel parameters must intertwine those actions.

*Learning impact:* optimization is restricted to representation-compatible
parameters. Learned filters respect the fiber geometry instead of relearning
equivalent channel transforms from data.

**Product-action symmetry (batch permutation × translation).**
```lean
axis b : ℕ = batch [symmetry permutation]
axis x : ℤ = width [symmetry group_action ℤ]
axis d : ℕ = k
axis h : ℕ = kout
axis r : ℤ = kr [window centered]

tensor X(b, x, d) [equivariant (b, x)]
tensor K(r, d, h) [symmetry none]
tensor Y(b, x, h) [equivariant (b, x)]

assert equivariant((b, x)):
  Y[b, x, h] := X[b, x + r, d] · K[r, d, h]
```
*What this constrains:* the program must commute with both actions at once:
permuting batch elements and shifting spatial indices.

*Learning impact:* one parameterization serves all batch orderings and spatial
translations, improving data efficiency and reducing dependence on sample order
or absolute position.

### Other `D`-monad examples (non-symmetry)

After symmetry monads, useful `D`-monads that are primarily structural (not
symmetry actions) include:

| `T_D : D -> D` | Index meaning | pyncd/leanncd use |
| --- | --- | --- |
| Prefix/causal | expose only past positions | causal attention, autoregressive scans |
| Window/neighborhood | expand each position to nearby positions | convolution, local attention, pooling |
| Block/tile | replace an axis by blocks and offsets | tiled matmul, block-sparse attention |
| Quotient/coarsening | collapse fine indices to coarse cells | patch pooling, graph pooling |
| Layout/reshape | track equivalent physical layouts | NHWC/NCHW (batch-height-width-channels vs. batch-channels-height-width), flatten/unflatten, fused kernels |
| Mask/subset | restrict admissible coordinates | sparse attention masks, valid-token masks |
| Graph-neighborhood | expand nodes by adjacency | message passing, k-hop GNN layers |

To call any of these a monad, one must actually specify an endofunctor
`T_D : D -> D` on objects and morphisms, together with unit `η^D` and
multiplication `μ^D`. The labels in the table are therefore **families of
possible monads**, not claims that each phrase names one unique canonical
construction. Typical readings are:

- **Prefix/causal.** Take `T_D(P)` to be a prefix object, down-set object, or
  object of causal admissible subsets of `P`. On morphisms, `T_D(f)` sends an
  admissible prefix/neighborhood along `f`, usually followed by causal closure
  if needed. The unit inserts a point as its trivial prefix; multiplication
  flattens a prefix-of-prefix by union/closure.

- **Window/neighborhood.** Take `T_D(P)` to attach a local offset or local
  neighborhood to each position, for example positions paired with an offset
  from a fixed window shape. On morphisms, `T_D(f)` transports the base
  position and the local coordinate data. The unit picks the zero offset or
  singleton neighborhood; multiplication composes offsets or flattens nested
  neighborhoods. When used purely for context extraction, this can also be more
  naturally comonadic.

- **Block/tile.** Fix a blocking scheme and let `T_D(P)` replace positions by
  coarse block coordinates together with in-block offsets. On morphisms,
  `T_D(f)` transports both coarse and fine coordinates. The unit views a point
  as a trivial block decomposition; multiplication flattens nested tilings by
  arithmetic composition of coarse/fine coordinates.

- **Quotient/coarsening.** Take `T_D(P)` to be a chosen quotient or coarsening
  of `P` (for example, pixels to patches, nodes to pooled clusters). On
  morphisms, `T_D(f)` is the induced map on equivalence classes or coarse
  cells. The unit is the quotient/coarsening map, and multiplication collapses
  repeated coarsenings. This is often an idempotent monad coming from a
  reflective subcategory of coarse objects.

- **Layout/reshape.** Take `T_D(P)` to record an alternative layout
  presentation of the same logical shape, or a canonical-layout wrapper. On
  morphisms, `T_D(f)` conjugates structural maps through the chosen layout
  conversions. The unit treats the current layout as the trivial presentation;
  multiplication composes consecutive layout changes or forgets intermediate
  presentations. If one only wants invertible reshapes, this may be better
  modeled directly inside `D` rather than by a monad.

- **Mask/subset.** Take `T_D(P)` to be the object of admissible subsets, masks,
  or support conditions on `P`. On morphisms, `T_D(f)` is direct image (or the
  appropriate pushforward of masks). The unit sends a point to its singleton
  mask; multiplication is union.

- **Graph-neighborhood.** Take `T_D(P)` to be a rooted neighborhood, adjacency
  expansion, or reachable-subgraph object over `P`. On morphisms, `T_D(f)`
  sends neighborhoods along graph homomorphisms. The unit picks the trivial
  neighborhood at a node; multiplication flattens neighborhood-of-neighborhood
  data to a single reachable neighborhood. As with windows, a comonadic reading
  is often also natural.

These are still valuable in the same DSL/actegory framework, but their primary
role is index locality, admissibility, or representation change—not symmetry in
the strict equivariance/invariance sense above.

Not every grade should be monadic. Ordinary broadcasting, slicing, and
compilation should still depend only on the basic `D`-graded PROP. Add a
leanncd mixin such as `SymmetryGraded D C T_D` only when a `D`-side structure
removes repeated equivariance proofs.

## 2) Computation semantics (`C`)

`C` is the **operation semantics**. It is a colored PROP: objects are finite
lists of typed tensor wires; morphisms are tensor programs; `⊗` is parallel
composition; `;` is sequential composition; and the symmetry `σ` permutes
wires. CDL and pyncd agree here: architectures are algebraic theories generated
by operations and equations.

Monads on `C` package operation-level structure: repeated computation,
recursion, parameter reuse, generated architecture fragments, or binding. They
are different from monads on `D`: `D` changes or constrains indices; `C`
changes how computations are assembled.

Typical `C`-side examples include:

- **free iteration/composition monads**, which package finite words or paths of
  composable `C`-morphisms, mapping a step to recipes for repeated application.
  The unit is the trivial recipe, and multiplication flattens nested repetitions.
  This is the CDL shape behind pyncd `Scan` and finite unrolling.
- **state-threading/recurrence monads**, which package computations that carry a
  hidden state wire through each `C`-morphism. The unit passes state through
  unchanged, and multiplication composes stateful stages while identifying the
  threaded state. This is the operation-level view of recurrent cells and scans.
- **parameterized or binding monads**, often most naturally expressed through
  `Para(C)` or related constructions, which package morphisms with a parameter
  or context object and map reparameterizations/substitutions. The unit adds no
  extra context, and multiplication merges nested contexts. This is relevant for
  pyncd weight tying, generated parameters, and Lean-side parameter laws.
- **macro/architecture-generation monads**, which package templates that expand
  a primitive into a structured subprogram and map generators to generated
  fragments. The unit treats a primitive as a degenerate macro, and
  multiplication inlines macro-of-macro expansions. This captures CDL-style
  architecture schemas before pyncd lowers them to concrete programs.

These are examples of what a monad on `C` may encode. `Scan` itself is usually
better understood not as the monad, but as an **algebra/fold for a monad of
repeated step application**. In this phrase, the monad is the formal recipe for
applying one shared parameterized step any finite number of times (categorically,
the free iteration monad generated by that step). Its unit is the zero-step
recipe (the monad unit), and its multiplication flattens nested recipes by
concatenating their iterations (the monad multiplication). The algebra/fold is
what runs that recipe (a monad algebra): it interprets the formal iterations as
actual `C`-morphisms, producing the sequence of states. Formally, for the
free-iteration monad `T` and target state-sequence computation object `A`, the
algebra is a map `α : T(A) -> A` from the free-iteration object to `A`,
satisfying the unit law `α ∘ η = id` and multiplication law
`α ∘ μ = α ∘ T(α)`. In `Scan` terms, the unit law says zero steps return the
initial state, while the multiplication law says nested or chunked iterations
produce the same state sequence as one flattened sequential composition of the
step.

Thus `Scan` is the fold that interprets the free repeated-step syntax, not the
syntax-generating monad itself. The parametric case makes this precise by
placing the shared step in `Para(C)`.

### `Scan` as a parametric-monad algebra

`Scan` is not a pointwise weave over time. Let `L ∈ Ob(D)` be a time axis, let
`t` be a point of `L`, let `H` be a state wire, let `X` be a per-step input
wire, and let `θ` be a shared parameter value in a parameter object `Θ`. If
`h_t ∈ H` is the state at time `t` and `x_t ∈ X` is the input slice at time
`t`, a recurrent cell has the form

```text
h_{t+1} = cell(θ, h_t, x_t).
```

Because `h_{t+1}` depends on `h_t`, evaluating one time point does not commute
with the operation.

CDL gives a clean presentation:

1. define one parameterized step `step : Θ ⊗ H ⊗ X -> H` in `C`;
2. view `step` as a morphism in `Para(C)`, the parametric category of `C`;
3. form the free parametric monad generated by repeated application of that step;
4. interpret `Scan` as the algebra/fold for that monad.

For a finite length `N`, where `N` indexes a prefix of `L`, the fold computes

```text
(h_0, x_0, ..., x_{N-1})  |->  (h_0, h_1, ..., h_N).
```

This separates responsibilities:

- `D` supplies the time grade `L`, prefix objects `[0..m]`, and prefix inclusions
  `ι_m : [0..m] -> L`.
- `C` supplies the step operation and the finite iteration/fold.
- `Para(C)` supplies the shared parameter object `Θ`, so weights are tied across
  all steps.
- `V` supplies the executable loop, checkpointed loop, associative scan, or
  custom kernel.

The executable pyncd operator `ConstructedScan` can remain the runtime
implementation. The CDL view explains its laws: prefix restriction, scan fusion,
weight tying, and backend independence. In leanncd, this belongs in a
`TemporalGraded D C` mixin with a temporal object `L`, prefix maps, finite
iteration, state-history trace, and the law that independent lifts commute with
scan.

For an independent batch grade `B ∈ Ob(D)`, the required lift-fold law is

```text
act(Scan(step), B)  ≅  Scan(act(step, B)).
```

It says that scanning a batched step is the same as independently scanning each
batch element. This is the bridge between `C`-side recurrence and `D`-side
batching.

## 3) Semantic bridge (`C -> V`)

`V` is the **target actegory**: a symmetric monoidal category with its own right
`D`-action. An algebra `F : C -> V` is a strong symmetric monoidal functor that
preserves the `D`-action up to coherent isomorphism. In pyncd, `F` is
implemented by `construct()`; in leanncd, this is the `Algebra` structure.

The common theme of this section is that the interesting structure discussed
earlier—symmetry on `D`, recurrence and parameterization on `C`, and explicit
parameter objects in `Para(C)`—only matters semantically once it is reflected
in the target interpretation `F : C -> V`. The subsections below therefore ask:
when does compiled semantics preserve symmetry in a principled way, when can
that be expressed as factorization through a symmetry-aware target, and how
should parameterized operations be organized so that this structure survives
compilation rather than being added informally afterward?

### Equivariance-as-factorization theorem

Plainly, compiled semantics are equivariant when compilation naturally lands
not just in ordinary target objects, but in target objects already equipped
with their symmetry action. In other words, one should be able to interpret a
layer or whole program directly as living between symmetry-aware targets
("equivariant objects"), not merely as an ordinary map that happens to commute
with a symmetry when checked afterward. The forgetful functor then drops this
extra action data and remembers only the underlying tensor/module/space. So
the usual compiled semantics are recovered by first building a symmetry-aware
semantic object and map, and then passing to the underlying ordinary map.

Let `T_V : V -> V` be a target-side symmetry monad induced by a compatible
`D`-side symmetry. Let `V^{T_V}` be the Eilenberg--Moore category of
`T_V`-algebras, let `U : V^{T_V} -> V` be the forgetful functor, let
`F~ : C -> V^{T_V}` denote a candidate symmetry-aware algebra, and let `∘`
denote functor composition. Then, for a `C`-algebra `F : C -> V`, the theorem
has the shape

```text
F is T_V-equivariant
if and only if
there exists F~ such that F = U ∘ F~.
```

Concretely, a layer is equivariant exactly when it can be interpreted directly
in the symmetry-aware target, not merely when tests observe that compiled tensors
happen to commute with a transformation.

Examples:

- For batch permutations, `F~` records the permutation action on inputs and
  outputs; `U` forgets that action and returns the ordinary compiled tensor map.
- For translations, `F~` lands in translation-equivariant objects, making
  convolutional weight sharing part of semantics.
- For steerable features, `F~` lands in representation-valued objects, so base
  motion and feature-fiber motion are tied together.

The caveat from Section 1 is essential: `V^{T_V}` must itself be a compatible
target `D`-actegory, and `U` must preserve the monoidal and `D`-action
structure. If the symmetry does not commute with the lift `⊛`, the theorem must
be reformulated; otherwise the factorization may not typecheck.

This leads directly to `Para(C)`. Earlier, `Scan` was described as repeated
application of one shared step, and that step typically carries parameters. The
question is therefore not only whether the compiled map factors through
symmetry-aware target objects, but also how the parameters themselves are
presented, shared, and transformed. `Para(C)` supplies that operation-level
parameter semantics: it separates runtime inputs from learnable parameter
objects, makes reparameterizations explicit, and gives a clean place to state
when parameter tying is part of the semantics rather than an implementation
accident. This also relates back to the factorization theorem: if a model is to
be equivariant in a genuinely semantic sense, its parameterized operations
should already be organized so that the symmetry-aware interpretation can see
how parameters behave under the symmetry, instead of treating weights as opaque
extras attached after compilation.

### `Para` and parameters

`Para(C)` is the **parameter category** of `C`: it has the same objects as `C`,
and a morphism `X -> Y` is a pair `(Θ, f)` where `Θ` is a **parameter object**
and `f : Θ ⊗ X -> Y` is an ordinary `C`-morphism. A **reparameterization** is a
map `r : Θ' -> Θ`; it turns `(Θ, f)` into `(Θ', (r ⊗ id_X) ; f)`. This is the
right place to model learnable weights, tied weights, generated weights, frozen
constants, and optimizer state without treating them as runtime input wires.

Practical pyncd implementation suggestions:

1. Add an explicit `ParamSpec` layer beside tensor-wire specs. A `ParamSpec`
   should record name, shape grade, dtype, initializer, trainable/frozen flag,
   and optional symmetry or sparsity constraints. `Linear`, convolution,
   attention projections, and `Scan` cells then expose parameter objects
   uniformly instead of ad hoc `nn.Parameter` fields.
2. Represent weight tying by stable parameter IDs plus alias groups, not Python
   object identity. For example, a recurrent `Scan` should lower to one ID
   `θ_cell`, while an untied unroll lowers to `θ_cell[0..N-1]`. The graph should
   make this distinction before PyTorch modules are built.
3. Make reparameterization maps first-class compiler nodes. Examples include
   low-rank factor maps `(A, B) |-> A B`, shared embedding/projection
   transposes, convolution kernel tying under a group action, and LoRA-style
   updates `(W, A, B) |-> W + A B`. These maps are morphisms into the parameter
   object, not post-hoc tensor mutations.
4. Lower `Para` composition by accumulating a parameter bundle. Composition
   `(Θ, f) ; (Φ, g)` should compile as one executable with parameter object
   `Θ ⊗ Φ`, modulo aliasing and reparameterization. This gives a deterministic
   source of truth for `state_dict` keys, optimizer groups, checkpoint loading,
   and serialization.
5. Add a parameter verifier before `construct()`: check that every symbolic
   parameter has exactly one owner, all aliases have compatible grades/dtypes,
   no frozen parameter appears in a trainable optimizer group, and every
   reparameterization target has the declared shape. These checks are cheaper
   and clearer before tensors are allocated.
6. Thread parameter metadata through `V`. In PyTorch, generated `nn.Module`s
   should expose both ordinary `parameters()` and a structured map from semantic
   parameter IDs to concrete tensors. In Lean, the algebra records which target
   morphism interprets each `ParamSpec`.

Formalism-level suggestions for pyncd/leanncd:

1. Define a `ParaAlgebra F` structure saying that an algebra `F : C -> V`
   extends to `Para(C) -> Para(V)`, preserves monoidal product of parameter
   objects, and respects reparameterization:

   ```text
   F_Para(reparam_r(Θ, f)) = reparam_{F(r)}(F_Para(Θ, f)).
   ```

2. Separate **parameter equality** from **parameter isomorphism**. Equality means
   the same learned value is reused; isomorphism means a change of coordinates,
   such as flattening a kernel or permuting channels. Lean laws should require
   tying to preserve equality and layout rewrites to preserve isomorphism,
   avoiding confusion between tied and merely reshaped weights.
3. State a scan tying law: `Scan(step)` uses the diagonal/copy map for one `Θ`,
   whereas an untied unroll uses a product `Θ^N`. This makes "shared recurrent
   cell" a theorem about the parameter object rather than a generated-code
   convention.

Experimental ideas:

- **Experimental: parameter lenses.** Attach to each compiled tensor a lens from
  the global parameter bundle to the local tensor slice. This could make
  checkpoint migration, sharded training, and partial freezing functorial:
  compiler rewrites would transform lenses, not just rename keys.
- **Experimental: symmetry-constrained parameter objects.** Let a `D`-side
  symmetry act on `Θ` itself, and define learnable parameters as fixed points or
  equivariant maps `Base -> Θ`. This would express convolutional sharing,
  steerable kernels, and some hypernetwork-generated weights through the same
  `Para` interface.

### LeanNCD practical next steps

1. **Make `ParaHom` the small core.**
   - Motivation: a parameterized morphism only needs a parameter object `Θ` and
     a map `Θ ⊗ X -> Y`.
   - Benefit: ordinary layers, tied layers, and scan cells use one Lean shape.

2. **Derive `paraMap` from `ParaHom`.**
   - Motivation: mapping an algebra over parameters should be functorial
     bookkeeping, not a second primitive API.
   - Benefit: there are fewer laws to maintain, and parameter composition has one
     source of truth.

3. **State weight tying as naturality.**
   - Motivation: shared weights should mean one parameter object is reused coherently
     under lift, scan, and algebra interpretation.
   - Benefit: recurrent cells, shared projections, and equivariant kernels become
     theorems instead of code conventions.

4. **Add an object-level `St`/`Br` lift API first.**
   - Motivation: early examples mostly need to say "lift this wire over this shape"
     before proving every reindexing law.
   - Benefit: engineers get usable constructors for batched, broadcasted, and
     strided objects while the full actegory laws mature.

5. **Use lightweight mixins before heavy Eilenberg--Moore machinery.**
   - Motivation: symmetry, temporal scan, and parameter laws can live as small
     structures on top of `DGradedColoredPROP`.
   - Benefit: the core stays stable, and each feature can be tested or removed
     independently.

6. **Prove tiny toy instances and tests first.**
   - Motivation: one-dimensional `St`, a small `Br`, and a two-step `Scan` expose
     most typing mistakes cheaply.
   - Benefit: Lean proofs and pyncd tests give quick feedback before larger CDL
     examples add noise.

## 4) Compiler geometry and data plumbing

### What CDL contributes here

CDL's main contribution is a **semantic discipline**:

- architectures are algebraic theories,
- implementations are structure-preserving interpretations,
- parameters live in `Para`.

That discipline helps compiler internals distinguish semantic rewrites, which
must preserve the algebra, from representational rewrites, which change
presentation but not meaning.

### CDL-aligned structure/data separation

The strongest overlap is the syntax/semantics split. Let `C♯` be the symbolic
operation category and `Dat : C♯ -> Set` assign concrete
sizes/dtypes/parameters. The Grothendieck construction

```text
C ≃ ∫Dat
```

matches the CDL mindset: keep theory-level structure separate from instantiated
data, then interpret functorially.

### What remains mostly outside CDL’s scope

Some compiler mechanisms are still essential but mostly CDL-adjacent rather than
CDL-derived:

- weave bookkeeping for broadcast/tiling layouts,
- pushout-style gluing/autoalignment of open components.

Keep these as implementation structures, not core CDL insights.

### Practical implementation rule

Keep four layers separate:

| Layer | Main role | pyncd/leanncd hook |
| --- | --- | --- |
| `D` monads | reusable index symmetry or locality | optional `SymmetryGraded D C T_D` |
| `C` monads | reusable computation such as finite recurrence | `TemporalGraded D C`, `ConstructedScan` |
| `C -> V` algebras | semantics and executable interpretation | `Algebra`, `construct()` |
| compiler plumbing | weaves, acsets, gluing, parameters | `Broadcasted`, `Weave`, `ParaAlgebra`, pushout composition |

The integration path is therefore conservative: keep the existing
`DGradedColoredPROP` as the core; add CDL monads as optional mixins with
explicit laws; present `Scan` both as the executable operator and as a
parametric-monad algebra; and preserve weaves, the Grothendieck split,
pushout/gluing, and `Para` as compiler-critical structures.

# generalized_tensors.md

## Generalized Containers as a Natural Generalization of Classical Tensors

This note explains generalized container-indexed tensors, why they strictly extend classical tensors, and how to extend the tensor logic DSL with backward compatibility; conceptually, it falls out of Gibbons-style Naperian/representable-functor semantics (the `lookup`/`tabulate` indexed-function view) generalized from fixed `Log` index spaces to container families with `Shape` and `Pos` (see [NaperianTyping.md](papers/NaperianTyping.md)). The main focus is tensors, but the same constructions extend to predicates; see [Appendix A. Generalized predicates in practice](#appendix-a-generalized-predicates-in-practice).

---

## Table of contents

- [Notation key (used throughout)](#notation-key-used-throughout)
- [1. Classical tensors are the all-ListDesc special case](#1-classical-tensors-are-the-all-listdesc-special-case)
- [2. Container-based tensor definition](#2-container-based-tensor-definition)
  - [2.1 Homogeneous containers (same container at every layer)](#21-homogeneous-containers-same-container-at-every-layer)
  - [2.2 Heterogeneous containers (different container per layer)](#22-heterogeneous-containers-different-container-per-layer)
  - [2.3 Scope and non-scope (explicit)](#23-scope-and-non-scope-explicit)
- [3. Container catalog and examples (including exotic ones)](#3-container-catalog-and-examples-including-exotic-ones)
  - [3.1 Core examples](#31-core-examples)
  - [3.2 Additional examples](#32-additional-examples)
- [4. Foldability and contraction legality](#4-foldability-and-contraction-legality)
- [5. ML use cases](#5-ml-use-cases)
  - [5.1 Ragged / variable-length batching](#51-ragged--variable-length-batching)
  - [5.2 Missing or optional features (`Maybe`)](#52-missing-or-optional-features-maybe)
  - [5.3 Multimodal routing (`Sum` / `Either`)](#53-multimodal-routing-sum--either)
  - [5.4 Structured sparsity (`SparseDesc`)](#54-structured-sparsity-sparsedesc)
  - [5.5 Hierarchical and tree-structured models (`TreeDesc`, W-types)](#55-hierarchical-and-tree-structured-models-treedesc-w-types)
  - [5.6 Locality-aware models (grid/graph/neighborhood)](#56-locality-aware-models-gridgraphneighborhood)
- [6. Convolution and Locality-aware tensors](#6-convolution-and-locality-aware-tensors)
  - [6.1 Regular stencils (affine grid convolution)](#61-regular-stencils-affine-grid-convolution)
  - [6.2 Common conv variants in the same form](#62-common-conv-variants-in-the-same-form)
  - [6.3 Irregular neighborhoods (graph/message passing)](#63-irregular-neighborhoods-graphmessage-passing)
  - [6.4 Gather/scatter view and DSL/compiler implications](#64-gatherscatter-view-and-dslcompiler-implications)
- [7. Proposed DSL syntax extensions (backward-compatible)](#7-proposed-dsl-syntax-extensions-backward-compatible)
  - [7.1 New declarations](#71-new-declarations)
  - [7.2 Container-schema declarations (`shape`, `pos`)](#72-container-schema-declarations-shape-pos)
  - [7.3 Patterned slots](#73-patterned-slots)
  - [7.4 Optional explicit reduce binder](#74-optional-explicit-reduce-binder)
  - [7.5 Gather and scatter surface forms](#75-gather-and-scatter-surface-forms)
- [8. Semantics](#8-semantics)
  - [8.1 Surface-to-core desugaring](#81-surface-to-core-desugaring)
  - [8.2 Reindex semantics](#82-reindex-semantics)
  - [8.3 Gather typing](#83-gather-typing)
  - [8.4 Scatter typing and legality](#84-scatter-typing-and-legality)
  - [8.5 Reduction semantics](#85-reduction-semantics)
  - [8.6 Pattern matching semantics (`some/none`, tags)](#86-pattern-matching-semantics-somenone-tags)
  - [8.7 Mask containers vs Iverson predicates](#87-mask-containers-vs-iverson-predicates)
  - [8.8 Symbolic vs concrete sizes](#88-symbolic-vs-concrete-sizes)
- [9. End-to-end mini examples](#9-end-to-end-mini-examples)
  - [9.1 Gather over neighborhood](#91-gather-over-neighborhood)
  - [9.2 Scatter with overlap reduction](#92-scatter-with-overlap-reduction)
  - [9.3 Non-contractible by default (Sum axis)](#93-non-contractible-by-default-sum-axis)
- [10. Symmetry actions on container positions](#10-symmetry-actions-on-container-positions)
  - [10.1 Core idea: symmetries act on `Pos(C,s)`](#101-core-idea-symmetries-act-on-poscs)
  - [10.2 Set/list symmetry (permutation equivariance)](#102-setlist-symmetry-permutation-equivariance)
  - [10.3 Grid symmetry (translation equivariance)](#103-grid-symmetry-translation-equivariance)
  - [10.4 Graph/container automorphism symmetry](#104-graphcontainer-automorphism-symmetry)
  - [10.5 Mask/sum containers and scoped symmetry](#105-masksum-containers-and-scoped-symmetry)
  - [10.6 Compiler obligations from the combined view](#106-compiler-obligations-from-the-combined-view)
- [11. Compatibility and migration](#11-compatibility-and-migration)
- [12. Summary](#12-summary)
- [Appendix A. Generalized predicates in practice](#appendix-a-generalized-predicates-in-practice)

---

## Notation key (used throughout)

- `R`: real numbers (the scalar value space in examples).
- `A -> B`: a function from type/set `A` to type/set `B`.
- `A × B`: product type (a pair containing one value from `A` and one from `B`).
- `Fin n`: finite index type with values `0,1,...,n-1`.
- `X :: XS`: list cons notation (head `X` followed by tail list `XS`).
- `~=`: isomorphic/equivalent representation (same information, different encoding).
- `Pos(C, s)`: position/index type for container `C` at shape `s`.
- `Shape(C)`: shape type for container `C` (all legal shape descriptors for `C`).
- `El(a)`: “elements/points” of axis or shape object `a` (Naperian point type).
- `RHS`: right-hand side of an assignment equation.
- Consistency convention:
  - use `List[n]` for DSL surface syntax,
  - use `ListDesc` for semantic/type-level container description.

---

## 1. Classical tensors are the all-ListDesc special case

A classical rank-`k` tensor with shape `[n1, ..., nk]` can be viewed as:

```text
T : Fin(n1) × ... × Fin(nk) -> R
```

Meaning: `T` is a function that takes a tuple of finite indices and returns a real scalar.

In container terms, each axis is `List[n]` at the DSL surface, corresponding to semantic `ListDesc` with positions `Fin n`.
So classical tensors are exactly generalized tensors where every axis container is `ListDesc`.

### Equivalent signatures

- Classical: `Tensor [3,2,2]`, or `Fin 3 → Fin 2 → Fin 2 → Float`
- Generalized: `GeneralizedTensor [ListDesc, ListDesc, ListDesc] [3,2,2]`

Here:
- `ListDesc` means the list container description (“shape is a length, positions are indices”).
- `Float` is a concrete machine numeric type; `R` is the abstract scalar notation.

---

## 2. Container-based tensor definition

### 2.1 Homogeneous containers (same container at every layer)

For a container `C`:
- shape type `Shape(C)` (the type of legal shape descriptors for `C`)
- position family `Pos(C, s)` for `s : Shape(C)` (the index set available at that shape)

```text
ContTensor(C, []) = R
ContTensor(C, s :: ss) = Pos(C, s) -> ContTensor(C, ss)
```

Definitions used above:
- `[]`: empty list of shape layers (rank 0 / scalar base case).
- `s :: ss`: one shape layer `s` followed by remaining layers `ss`.
- `ContTensor(C, s :: ss)`: at one layer, choose a position in `Pos(C, s)` and continue recursively.

`ContTensor` means **container tensor**: a generic representation of nested, indexable data using a single container `C` repeated across layers.

Connection to Naperian `lookup/tabulate` ([Representable Functors (Naperian Functors)](papers/NaperianTyping.md#representable-functors-naperian-functors)):

```text
lookup_{C,s::ss}
  : ContTensor(C, s::ss)
    -> (Pos(C, s) -> ContTensor(C, ss))

tabulate_{C,s::ss}
  : (Pos(C, s) -> ContTensor(C, ss))
    -> ContTensor(C, s::ss)
```

for each fixed `(C, s::ss)`, with laws
`lookup_{C,s::ss} (tabulate_{C,s::ss} f) = f` and
`tabulate_{C,s::ss} (lookup_{C,s::ss} x) = x`.
At one layer (`ss = []`), this is exactly `lookup/tabulate` at shape `s`.
This is still the homogeneous (non-dependent) case because `C` is fixed across layers.

Example (classical unrolling):

Set `C = ListDesc`, so `Pos(ListDesc, n) = Fin n`:

```text
ContTensor(ListDesc, []) = R
ContTensor(ListDesc, n :: ns) = Fin n -> ContTensor(ListDesc, ns)
```

Examples:

```text
ContTensor(ListDesc, [n])       = Fin n -> R
ContTensor(ListDesc, [n,m])     = Fin n -> Fin m -> R
ContTensor(ListDesc, [a,b,c])   = Fin a -> Fin b -> Fin c -> R
```

Naperian perspective (brief): this same rank-3 case can be read as

```text
T ~= El(a) × El(b) × El(c) -> V
```

and elementwise operations are lifted pointwise:

```text
(liftA  g  T)(i, j, k)     = g(T(i, j, k))
(liftA2 f T1 T2)(i, j, k)  = f(T1(i, j, k), T2(i, j, k))
```

### 2.2 Heterogeneous containers (different container per layer)

For heterogeneous layers:

```text
GeneralizedTensor([], []) = R
GeneralizedTensor(C :: Cs, s :: ss) =
  Pos(C, s) -> GeneralizedTensor(Cs, ss)
```

Important typing point: `s` is dependent on `C` (`s : Shape(C)`), so the shape list is heterogeneous.

Analogous `lookup/tabulate` construction (heterogeneous):

Each layer `(C, s)` contributes one representable step `Pos(C,s) -> (...)`.
Stacking heterogeneous layers composes these steps recursively.

Then for a full heterogeneous stack `(C :: Cs, s :: ss)`:

```text
lookup_{C::Cs,s::ss}
  : GeneralizedTensor(C::Cs, s::ss)
    -> (Pos(C, s) -> GeneralizedTensor(Cs, ss))

tabulate_{C::Cs,s::ss}
  : (Pos(C, s) -> GeneralizedTensor(Cs, ss))
    -> GeneralizedTensor(C::Cs, s::ss)
```

So this is the same representable idea as homogeneous containers, but indexed per layer by the dependent pair `(C, s)`.

Heterogeneous-layer index-path view:

```text
Ix([], []) = Unit
Ix(C :: Cs, s :: ss) = Pos(C, s) × Ix(Cs, ss)
```

Here `Ix(containers, shapes)` is the full dependent index-path type.
Any `GeneralizedTensor(containers, shapes)` can be viewed as:

```text
T : Ix(containers, shapes) -> V
```

and lifting remains pointwise with respect to that path:

```text
(liftA  g  T)(ix)      = g(T(ix))
(liftA2 f T1 T2)(ix)   = f(T1(ix), T2(ix))
```

where `ix : Ix(containers, shapes)`.

---

## 2.3 Scope and non-scope (explicit)

This document makes two different kinds of claims:

1. **Representation claim (broad):**
   - `ContTensor`/`GeneralizedTensor` describe how to represent tensor-like data for many container choices.
   - This requires only shape+position structure.

2. **Operation claim (conditional):**
   - Legal operations depend on extra capabilities.
   - Reindex/gather/scatter require lawful typed index maps, meaning:
     - **Typed:** each map has an explicit domain and codomain in position types (for example `idx : Pos(O,sO) × Pos(P,sP) -> Pos(X,sX)` and `tgt : Pos(O,sO) × Pos(P,sP) -> Pos(Y,sY)`).
      Operationally, these are index transformers: `idx` tells gather operations which `X` coordinate to read for each loop position, while `tgt` tells scatter operations which `Y` coordinate to write.
     - **Total:** map is defined on every input position in its declared domain (no partial “missing index” cases unless encoded explicitly in the type).
     - **Composable:** reindex maps obey identity/composition behavior expected of a lawful action (reindex by identity does nothing; chained reindexing equals composition of maps).
     - **Scatter-safe:** when used as write targets, either the map is injective on produced writes or an explicit collision policy (`reduce=`) is provided.
     - **Bounds-safe by construction:** mapped indices land in the codomain position type; out-of-bounds writes/reads are rejected or made explicit via declared semantics (e.g., mask/guard/fill policy).
   - Contraction/reduction requires contractible/foldable axis instances and an operator algebra, meaning:
     - **Contractible axis instance:** for the concrete axis `(C,s)`, there is a finite, well-defined position collection to reduce over.
     - **Fold law:** there is an implementation of traversal/fold over those positions (order-sensitive or order-insensitive, as declared).
     - **Operator algebra:** the reduction operator has declared combine/identity behavior appropriate for the fold (for example sum-with-zero, max-with-bottom, or another explicit monoidal/semiring policy).
     - **Determinism policy:** when fold order can vary (parallel evaluation), required algebraic properties (such as associativity, and commutativity when needed) are part of the contract.
     - **Empty-case policy:** behavior on empty position sets must be explicit (identity value, error, or forbidden by typing).
     - **No implicit default on non-contractible axes:** containers like `Maybe`/`Sum` are not reduced implicitly unless a user-supplied algebra/instance is provided.

Therefore, this note **does not** claim:
- that every container supports contraction by default,
- that every container supports every optimization pass,
- or that representation alone implies full tensor algebra.

It **does** claim:
- classical tensors are the all-`ListDesc` special case (surface: all `List[n]` axes),
- generalized containers extend representable index structures,
- and operation legality can be specified capability-by-capability.

---

## 3. Container catalog and examples (including exotic ones)
## 3.1 Core examples

- `ListDesc`: ordinary axes
- `MaybeDesc`: optional branch (`none` / `some`)
- `EitherDesc` / `SumDesc`: tagged branch
- `PairDesc`: fixed two-position layer

Definitions:
- `none`: empty/absent optional case.
- `some x`: present optional case with payload `x`.
- `SumDesc{a,b,...}`: disjoint tagged alternatives (exactly one active tag at a time).

Examples:

```text
GeneralizedTensor [MaybeDesc, ListDesc] [isNil, 5]   -- maybe list of length 5
GeneralizedTensor [SumDesc{text,image}, ListDesc] [tag, 768]
```

## 3.2 Additional examples

### RangeDesc (bounded interval)
```text
GeneralizedTensor [RangeDesc] [(start,end)]          -- positions in [start,end)
```

`[start,end)` means start-inclusive, end-exclusive interval.

Definition (one practical form):

```text
Shape(RangeDesc) = Nat × Nat                       -- (start, end), with start <= end
Pos(RangeDesc, (start, end)) = Fin (end - start)   -- logical index offset 0..(end-start-1)
```

Concrete coordinate at offset `k` is `start + k`.

### SparseDesc (active-index set)
```text
GeneralizedTensor [SparseDesc, ListDesc] [activeIdx, 64]
```

Definition (index-list sparse form):

```text
Shape(SparseDesc) = List Nat                        -- active global indices
Pos(SparseDesc, activeIdx) = Fin (length activeIdx) -- slot in sparse list
```

Lookup maps slot `p` to global coordinate `activeIdx[p]`.

### TreeDesc (finite rooted tree layer)
```text
GeneralizedTensor [TreeDesc, ListDesc] [treeShape, d]
```

Definition (finite rooted-tree form):

```text
Shape(TreeDesc) = FiniteRootedTree
Pos(TreeDesc, treeShape) = Nodes(treeShape)         -- finite set of node positions
```

`Nodes(treeShape)` can be encoded as node ids or root-to-node paths.

### SymmetricDesc (permutation-invariant finite positions)
```text
GeneralizedTensor [SymmetricDesc, ListDesc] [orbitShape, d]
```

`orbitShape` denotes a symmetry class/orbit descriptor rather than ordered coordinates.

Definition (orbit-representative form):

```text
Shape(SymmetricDesc) = (baseShape, groupAction)
Pos(SymmetricDesc, orbitShape) = OrbitReps(baseShape, groupAction)
```

`OrbitReps` picks one canonical representative per symmetry orbit.

### QuotientDesc (positions modulo equivalence)
```text
GeneralizedTensor [QuotientDesc, ListDesc] [quotShape, d]
```

“Modulo equivalence” means positions that are equivalent are treated as one canonical class.

Definition (quotient-set form):

```text
Shape(QuotientDesc) = (baseShape, equivRel)
Pos(QuotientDesc, quotShape) = Quotient(basePositions(baseShape), equivRel)
```

So positions are equivalence classes `[x]` rather than raw indices `x`.

### W-types (inductive) and M-types (coinductive) recursive descriptions
```text
GeneralizedTensor [WDesc, ListDesc] [wShape, d]
```

`WDesc` here denotes a well-founded (inductive) recursive container description (W-type style).
`MDesc` would denote the coinductive counterpart (M-type style) for potentially infinite unfolding structures.

Definitions (schematic):

```text
Shape(WDesc) = W-shape signatures (well-founded branching descriptions)
Pos(WDesc, wShape) = immediate branches/positions specified by wShape

Shape(MDesc) = M-shape signatures (coinductive unfolding descriptions)
Pos(MDesc, mShape) = one-step observable branches/positions of mShape
```

For `MDesc`, reductions typically require an explicit finite observation/truncation policy.

---

## 4. Foldability and contraction legality

Contraction/reduction on an axis is legal only when that container axis has a lawful fold for the chosen reduction algebra.

Definitions:
- “contraction/reduction”: combining all values along an index (for example by `sum`).
- “lawful fold”: a fold implementation satisfying expected algebraic behavior for the operator.
- “reduction algebra”: operator + identity/combination laws used by reduction (for example sum or max semantics).

Additional common contractible shape families (when finite and lawful folds are provided):
- fixed tuples / finite records (`Tuple k`, product/record containers),
- finite maps/dictionaries over bounded key sets,
- finite sets/multisets (with order-insensitive reduction semantics),
- sliding-window/patch containers (finite window positions),
- bucket/bin containers with finite bins,
- CSR/COO-style sparse row containers (finite stored entries per row),
- finite partition/block containers.

| Container | Contractible by default? | Notes |
|---|---|---|
| `ListDesc` | Yes | Standard sum/max reductions |
| `PairDesc` | Yes | Fixed finite fold over two positions |
| `RangeDesc` | Yes | Bounded finite fold |
| `SparseDesc` | Yes | Fold over active positions |
| `MaybeDesc` | No (default) | Optional; needs explicit algebra policy |
| `EitherDesc` / `SumDesc` | No (default) | Branch choice, not canonical multi-position fold |
| `TreeDesc` | Depends | Needs finite traversal and fold law |
| `Graph/Neighborhood` | Depends | Needs finite neighborhood enumerator |
| `QuotientDesc` | Depends | Needs canonical representative fold |
| `W-type` | Depends | Needs well-founded finite fold |
| `M-type` | Depends | Needs an explicit finite observation/truncation or other coinductive reduction policy |
| `SymmetricDesc` | Depends | Needs finite orbit fold |
| `DependentDesc` | Depends | Shape-dependent fold law |
| `ParametricDesc` | Depends | Parameter-specific fold law |

### Contraction rule

Implicit contraction of RHS-only indices is allowed iff the corresponding axis is contractible under the selected reduction operator.
Otherwise: compile-time error unless user provides explicit reduction algebra/instance.

`RHS-only index` = an index variable appearing only on the right side of an equation and not on the left side.

---

## 5. ML use cases

Many ML systems are forced into dense rectangular tensors even when the data is not rectangular (ragged sequences, optional fields, sparse neighborhoods, tree structure, modality branches). That mismatch leads to:

- padding and mask boilerplate,
- duplicated representation conversions (sparse <-> dense),
- implicit runtime conventions for legality (what may be reduced, where scatter may collide),
- and hard-to-debug shape/index errors.

Generalized containers move these structural assumptions into typed index/shape declarations. In practice this gives:

- earlier error detection (illegal reduce/reindex/scatter rejected sooner),
- explicit read/write maps for gather/scatter,
- explicit contraction legality per axis instance,
- explicit scatter collision policy (`reduce=`) instead of silent ambiguity,
- clearer equation intent because structure is in the type/index layer.

The use cases below show this concretely.

### 5.1 Ragged / variable-length batching

Background: language, speech, logs, and event streams naturally produce variable-length examples. Standard batching pads to max length and carries mask tensors everywhere.

Convenience:
- represent each sample at its natural length (ragged shape),
- avoid extra padded compute and memory traffic,
- define reductions on real positions instead of “valid token” masking conventions,
- keep sequence-length semantics local to shape declarations.

### 5.2 Missing or optional features (`Maybe`)

Background: tabular and sensor workloads often have missing readings or optional columns; sentinel values (`-1`, `NaN`) can leak into learning logic.

Convenience:
- optionality is explicit (`some`/`none`) instead of hidden in value hacks,
- branch behavior becomes part of equation structure,
- prevents accidental reductions over non-present data unless an explicit algebra is chosen,
- improves portability across datasets with different missingness patterns.

### 5.3 Multimodal routing (`Sum` / `Either`)

Background: modern pipelines combine text/image/audio/video and often need modality-specific branches with shared downstream heads.

Convenience:
- modality tags are first-class indices (`text`, `image`, ...),
- branch-local equations stay in one formalism rather than separate code paths,
- pattern coverage can be checked (missing branch handling is visible),
- easier composition of shared and branch-specific transforms.

### 5.4 Structured sparsity (`SparseDesc`)

Background: recommenders, graph models, retrieval, and sparse expert models work over active-index subsets, but many kernels still expect dense tensors.

Convenience:
- active indices are native positions of the axis,
- gather/scatter/reduce are expressed over active entries directly,
- avoids repeated densify/sparsify transforms around each operator,
- clearer separation between “logical shape” and “stored nonzero support”.

### 5.5 Hierarchical and tree-structured models (`TreeDesc`, W-types)

Background: parse trees, program ASTs, hierarchical taxonomies, and recursive latent structures are not naturally rectangular.

Convenience:
- node/path positions are first-class index positions,
- recursive structure is represented in the same equation language as dense tensors,
- traversal/fold legality is centralized at container definition,
- reduces one-off recursive plumbing in model code.

### 5.6 Locality-aware models (grid/graph/neighborhood)

Background: CNNs, GNNs, and message-passing systems all compute local aggregation but with different neighborhood encodings.

Convenience:
- one common operator shape: `reindex + pointwise + fold`,
- regular stencils (`v + p`) and irregular neighborhoods (`nbr(v,p)`) differ only in index map,
- shared optimization opportunities (fusion, legality checks, reduction policy),
- consistent semantics across conv-like and graph-like operators.

---

## 6. Convolution and Locality-aware tensors

Convolution can be expressed uniformly as:

1. **Reindex neighborhood** (choose source coordinates for each output position),
2. **Pointwise combine** (multiply by kernel/filter coefficients),
3. **Fold** (reduce over neighborhood slots and often channel slots).

This formulation is the same in regular grids and irregular graphs; only the index map changes.

Notation legend for this section:
- Data/activations: `X`, `Y`, `Z` (`X` input, `Y`/`Z` outputs or intermediates).
- Kernels/weights: `W`, `Wk`, `Wd`, `Wp`.
- Indexing structure (not data): `nbr(v,p)` and indices like `o, oh, ow, p, kh, kw, ci, co, c`.

### 6.1 Regular stencils (affine grid convolution)

Canonical 1D/2D forms:

```tl
Y[o] := W[p] · X[o + p]                                   -- 1D convolution
Y[oh, ow, co] := Wk[co, ci, kh, kw] · X[oh + kh, ow + kw, ci]  -- 2D convolution
```

Equivalent explicit-reduce form:

```tl
Y[o] := reduce(sum, p in K) (W[p] · X[o + p])                                      -- 1D convolution
Y[oh, ow, co] := reduce(sum, ci in Cin, kh in Kh, kw in Kw)                        -- 2D convolution
  (Wk[co, ci, kh, kw] · X[oh + kh, ow + kw, ci])
```

Interpretation:
- `o`, `oh`, `ow` index output coordinates,
- `p` / `kh,kw` index kernel offsets,
- `ci` is usually an RHS-only contraction axis (input channel sum),
- affine map (`o + p`, `oh + kh`, `ow + kw`) is the neighborhood reindex.

### 6.2 Common conv variants in the same form

#### Strided convolution

`stride` makes output index `o` advance input sampling by fixed steps, reducing output resolution and compute while keeping local kernel offsets relative to each sampled anchor.

Axes/shapes (example):

```tl
container O   = List[No]      -- output positions
container K   = List[Ks]      -- kernel offsets
container In  = List[Ni]      -- input positions
```

Canonical (implicit contraction over `p`):

```tl
Y[o] := W[p] · X[stride*o + p]
```

Explicit reduce:

```tl
Y[o] := reduce(sum, p in K) (W[p] · X[stride*o + p])
```

#### Dilated convolution

`dilation` spaces kernel offsets, increasing receptive field coverage without increasing kernel parameter count.

Axes/shapes (example):

```tl
container O   = List[No]
container K   = List[Ks]
container In  = List[Ni]
```

Canonical (implicit contraction over `p`):

```tl
Y[o] := W[p] · X[o + dilation*p]
```

Explicit reduce:

```tl
Y[o] := reduce(sum, p in K) (W[p] · X[o + dilation*p])
```

#### Depthwise convolution

Each channel is filtered independently: reduction is over spatial offsets only, so this step does not mix channels.

Axes/shapes (example):

```tl
container O   = List[No]
container K   = List[Ks]
container C   = List[Cin]
```

Canonical (implicit contraction over `p`):

```tl
Y[o, c] := Wd[c, p] · X[o + p, c]
```

Explicit reduce:

```tl
Y[o, c] := reduce(sum, p in K) (Wd[c, p] · X[o + p, c])
```

#### Pointwise (1x1) mixing after depthwise

This performs channel mixing/projection at fixed spatial location (no spatial neighborhood reduction), typically following depthwise filtering.

Axes/shapes (example):

```tl
container O    = List[No]
container Cin  = List[Cin]
container Cout = List[Cout]
```

Canonical (implicit contraction over `c`):

```tl
Z[o, co] := Wp[co, c] · Y[o, c]
```

Explicit reduce:

```tl
Z[o, co] := reduce(sum, c in Cin) (Wp[co, c] · Y[o, c])
```

Key point: stride/dilation/depthwise/pointwise are all parameter changes to index maps and contraction axes, not different semantic operators.

### 6.3 Irregular neighborhoods (graph/message passing)

```tl
Y[v, co] := W[co, ci, p] · X[nbr(v, p), ci]
```

where:
- `v`: center/output vertex,
- `p`: local neighbor-slot index,
- `nbr(v,p)`: neighborhood map from slot to concrete neighbor id.

Equivalent explicit reduction:

```tl
Y[v, co] := reduce(sum, p in Nbr(v), ci in Cin) (W[co, ci, p] · X[nbr(v,p), ci])
```

This is graph convolution/message passing in the same reindex+fold form as grid convolution.

### 6.4 Gather/scatter view and DSL/compiler implications

Most convolution equations are written as **gather**:

```tl
Y[o] := ... X[idx(o,p)] ...
```

Some operators (transpose/deconv-style or routing-style updates) are naturally **scatter**:

```tl
Y[tgt(o,p)] := W[p] · X[o]
```

Both are locality-aware transforms; they differ only in whether mapped indices are used for reads (gather) or writes (scatter).
When scatter writes are non-injective, the overlap policy (`reduce=...`) and default fill (`fill=...`) are supplied via existing scatter options/declarations, not inline RHS wrapper syntax.

Using one locality abstraction gives:
- one legality framework for bounds/typing/collision checks,
- one optimization vocabulary (reindex fusion, reduction scheduling, map canonicalization),
- shared semantics across CNN-, GNN-, and neighborhood-aggregation workloads.

---

## 7. Proposed DSL syntax extensions (backward-compatible)

> Status: proposal. Existing syntax stays valid.

Relation to existing notation:
- existing axis notation remains the canonical surface form for classical tensors,
- internally, an axis with size `n` is interpreted as a `List[n]` container axis,
- so current programs need no rewrite: this proposal adds non-`List` containers without replacing existing syntax.
- existing `axis` declarations remain valid (for example `axis i : ℕ = n`) and continue to provide explicit size/kind metadata.
- axis declarations are still optional for many plain contraction programs, just as today.
- declare axes explicitly when you need pinned sizes, stricter kind checks, or behavior that depends on concrete extents/iteration structure.
- container declarations complement (not replace) axis declarations: use them when you need non-`List` structure (`Maybe`, `Sum`, `Neighborhood`, ...).

## 7.1 New declarations

```tl
container Tok   = List[n]
container Time  = List[T]        -- T is the Time-axis size/extent
container Mask  = Maybe[has_mask]
container Mod   = Sum{text,image}
container Nbr   = Neighborhood[V]
```

Tensor declarations over named containers:

```tl
tensor X(Time, Tok, d)
tensor M(Time, Tok, Mask)
tensor F(Mod, Tok, d)
```

## 7.2 Container-schema declarations (`shape`, `pos`)

For nontrivial containers, the DSL can permit explicit schema declarations:

```tl
container C where
  shape := ...
  pos(s) := ...
```

where:
- `shape` defines the indexing object/domain for container instances,
- `pos(s)` defines legal position slots for a particular shape instance `s`.

GNN-specialized `Neighborhood` example (predicate-backed):

```tl
axis v : ℕ = n
axis p : ℕ = pmax
axis u : ℕ = n

predicate adj_slot(v, p, u)      -- slot p of v selects neighbor u

container Neighborhood[V] where
  shape := V
  pos(v) := { p | exists u, adj_slot[v,p,u] }
-- Well-formedness checks over `adj_slot`:
--   (1) existence: p is in pos(v) iff there exists at least one u with adj_slot[v,p,u]
--   (2) uniqueness: for fixed (v,p), at most one u satisfies adj_slot[v,p,u]
--   (3) closure: any selected u is a valid vertex index (u : V)
```

This makes neighborhood behavior explicit in the DSL itself while keeping tensor/predicate representations as the backing implementation.

CNN/locality-aware example (regular 2D stencil, predicate-backed):

```tl
axis ox : ℕ = nox        -- output x-position
axis oy : ℕ = noy        -- output y-position
axis px : ℕ = kx         -- kernel slot x-offset
axis py : ℕ = ky         -- kernel slot y-offset
axis ix : ℕ = nix        -- input x-position
axis iy : ℕ = niy        -- input y-position

predicate stencil2d_src(ox, oy, px, py, ix, iy)     -- kernel slot (px,py) at output (ox,oy) selects input (ix,iy)
stencil2d_src[ox,oy,px,py,ix,iy] :=
  [ix = ox + px] ·
  [iy = oy + py] ·
  [0 <= ix < nix] ·
  [0 <= iy < niy]

container Stencil2D[OX, OY] where
  shape := OX × OY
  pos(ox, oy) := { (px,py) | exists ix, iy, stencil2d_src[ox,oy,px,py,ix,iy] }
-- Well-formedness checks over `stencil2d_src`:
--   (1) existence: (px,py) is in pos(ox,oy) iff there exists at least one (ix,iy) with stencil2d_src[ox,oy,px,py,ix,iy]
--   (2) uniqueness: for fixed (ox,oy,px,py), at most one (ix,iy) satisfies stencil2d_src[ox,oy,px,py,ix,iy]
--   (3) closure: any selected (ix,iy) is a valid input index pair (ix : ℕ = nix, iy : ℕ = niy)
```

In standard contiguous conv, `stencil2d_src[ox,oy,px,py,ix,iy]` encodes
`ix = ox + px` and `iy = oy + py` (or stride/dilation variants),
but the same schema can also encode boundary handling and nonuniform local windows.

## 7.3 Patterned slots

```tl
M[t, i, some m]
M[t, i, none]
F[text, i, d]
F[image, i, d]
```

`text` and `image` are tags selecting a branch of a sum container.

## 7.4 Optional explicit reduce binder

```tl
Y[t, d] := reduce(sum, i in Tok) (A[t, i] · B[i, d])
```

RHS-only implicit contraction remains available for legal contractible axes.

Clarification versus current Lean DSL:
- current syntax already supports operator selection via forms like `maxreduce(expr)`,
- but that is **operator-only** (the reduced axes are still inferred implicitly from RHS-only indices),
- the proposed `reduce(op, i in Tok) (...)` adds **explicit axis/container selection** in addition to operator choice.

## 7.5 Gather and scatter surface forms

### Gather (pullback)
```tl
Y[out...] := W[..., p] · X[idx(out..., p), ...]
```

Definitions:
- `out...`: one or more output indices.
- `idx(...)`: index mapping function used for reading from `X`.
- gather/pullback: read from mapped source coordinates into current output coordinate.

Examples:
```tl
Y[v, co] := W[co, ci, p] · X[v + p, ci]
Y[v, co] := W[co, ci, p] · X[nbr(v, p), ci]
```

### Scatter (pushforward), existing mapped-LHS form
```tl
Y[tgt(out..., p), ...] := RHS
```

Definitions:
- `tgt(...)`: target index mapping function used for writing to `Y`.
- scatter/pushforward: write values into mapped target coordinates.
- overlap/default policies (for example `reduce=sum`, `fill=0`) are specified through existing scatter options/declarations.

Example:
```tl
Y[tgt(v, p), co] := W[co, ci, p] · X[v, ci]
```

Compatibility note: this can desugar to current-style affine/scatter lowering machinery (LHS mapped write + overlap policy), generalized from affine `St` maps to typed target maps.

---

## 8. Semantics

## 8.1 Surface-to-core desugaring

- Existing axis declarations/syntax are canonical for classical tensors and desugar to `List[n]` container axes.
- New `container` declarations introduce non-list axis kinds.
- Tensor slot expressions denote inhabitants of `Pos(C,s)` for the declared axis container/shape.

## 8.2 Reindex semantics

Given `eta : P -> Q`:

```text
reindex(eta)(x) = x ∘ mapPos(eta)
```

Plain-language reading:
- reindexing does not change stored values directly;
- it changes **how coordinates are interpreted**;
- to read the reindexed tensor at a position `p` of `P`, first map `p` into `Q`, then read `x` there.

Pointwise form:

```text
reindex(eta)(x)(p) = x(mapPos(eta)(p))
```

So reindexing is just function precomposition by an index map.

Definitions:
- `eta`: morphism/map between shape objects `P` and `Q`.
- `x`: value indexed by positions of `Q`.
- `mapPos(eta)`: induced map on positions.
- `∘`: function composition (“apply `mapPos(eta)` first, then `x`”).

Connection to current Lean `St` behavior:
- affine indexing is exactly one concrete instance of `mapPos(eta)`,
- e.g. `mapPos(eta)(i) = A*i + b`,
- so current affine reindexing is a special case of this general rule, not a separate semantic mechanism.

## 8.3 Gather typing

For
```tl
Y[o] := ... X[idx(o,p)] ...
```
`idx` must type-check as:
```text
idx : Pos(O, sO) × Pos(P, sP) -> Pos(XAxis, sX)
```
and be total on its domain.

Definitions:
- `O`, `P`: shape objects for output context and reduction/kernel context.
- `sO`, `sP`, `sX`: concrete shape parameters for those axes.
- “total”: defined for every input pair in the stated domain.

## 8.4 Scatter typing and legality

For
```tl
Y[tgt(o,p)] := RHS(o,p)
```
`tgt` must type-check as:
```text
tgt : Pos(O, sO) × Pos(P, sP) -> Pos(YAxis, sY)
```

Legality:
- if `tgt` is injective on produced writes, scatter is direct;
- if non-injective, explicit `reduce=` is required;
- `fill=` specifies untouched output values.

Definitions:
- “injective”: distinct source write events map to distinct targets.
- “non-injective”: multiple source writes can collide on one target.

## 8.5 Reduction semantics

```text
reduce(op, p in Pos(C, s)) f(p)
```

is defined only when container axis `(C,s)` has a lawful fold instance for `op`.

Definitions:
- `op`: reduction operator (for example `sum`, `max`).
- `p in Pos(C,s)`: iterate across all positions of that axis instance.

## 8.6 Pattern matching semantics (`some/none`, tags)

- Patterns refine index domains.
- Non-covered branches are compile-time errors unless an explicit default branch is supplied.
- Branch-specific equations can desugar to masked/guarded terms in core form.

“desugar” means rewrite surface syntax into an equivalent lower-level internal form.

## 8.7 Mask containers vs Iverson predicates

Mask containers and Iverson predicates can both gate computation, but they live at different semantic layers:

- **Mask container (`Maybe`-style): structural/index-level gating.**
  It changes which positions are part of the domain shape.
- **Iverson predicate (`[P]`): value-level gating.**
  It keeps the same domain and multiplies/filters values by a boolean condition.

Equivalent intuition:
- container masks affect "what coordinates exist",
- Iverson affects "what value is used at an existing coordinate."

Example (structural mask):

```tl
container Mask = Maybe[has_mask]
tensor M(Time, Tok, Mask)

-- Branches are structural:
M[t, i, some m] := ...
M[t, i, none]   := ...
```

Example (Iverson value gating):

```tl
Y[t, i] := [i <= t] · X[t, i]
```

Combined behavior (conceptually):
- effective contribution is present only when the position is structurally present **and** the predicate is true.

Compilation note:
- a structural mask may lower to guard logic at runtime, but semantically it remains stronger than a pure Iverson factor because it participates in shape/index legality.

## 8.8 Symbolic vs concrete sizes

When shapes/sizes are symbolic at compile time, contractibility/finiteness obligations may be deferred until concrete size instantiation (evaluation-time shape environment), consistent with existing symbolic-vs-concrete pipeline staging.

Definitions:
- “symbolic size”: an unknown size variable/expression at compile time.
- “concrete size instantiation”: runtime or late-stage resolution to specific integers.

---

## 9. End-to-end mini examples

### 9.1 Gather over neighborhood

`Neighborhood[V]` denotes a container family indexed by center vertex `v : V`, where each
instance enumerates local neighbor slots around `v`. In this example:
- `p` is a slot in the neighborhood of `v`,
- `nbr(v,p)` maps that slot to the concrete neighbor vertex id in `V`,
- `Nbr(v)` is therefore the position set being reduced/iterated for that center.

Concrete definition (adjacency-list/CSR style):

```text
Shape(Neighborhood[V]) = V
Pos(Neighborhood[V], v) = Fin(deg(v))
nbr(v, p) = adj[v][p]     -- p-th neighbor of v from adjacency storage
```

where:
- `deg(v)` is the out-degree (or chosen neighborhood size) of `v`,
- `adj[v]` is the stored neighbor list for `v`,
- `adj[v][p]` must be total/valid for all `p : Fin(deg(v))`.

```tl
container V   = List[num_nodes]
container Nbr = Neighborhood[V]
tensor X(V, Cin)
tensor W(Cout, Cin, Nbr)
tensor Y(V, Cout)

Y[v, co] := W[co, ci, p] · X[nbr(v,p), ci]
```

### 9.2 Scatter with overlap reduction
```tl
container V = List[num_nodes]
container K = List[k]
tensor X(V, Cin)
tensor W(Cout, Cin, K)
tensor Y(V, Cout)

-- Overlap/default policy supplied by existing scatter options/declarations:
-- reduce = sum, fill = 0
Y[v + p, co] := W[co, ci, p] · X[v, ci]
```

### 9.3 Non-contractible by default (Sum axis)
```tl
container Mod = Sum{text,image}
tensor F(Mod, Tok, d)
tensor Out(Tok, d)

-- Legal branchwise equations:
Out[i,d] := F[text, i, d] + F[image, i, d]

-- Illegal implicit contraction over Mod unless explicit algebra is provided.
```

---

## 10. Symmetry actions on container positions

This section makes the container/symmetry connection explicit.
For broader context and the original symmetry-oriented DSL discussion, see
[`papers/CDL_connections.md`](./papers/CDL_connections.md), especially:
["Symmetry DSL and scoped semantics"](./papers/CDL_connections.md#symmetry-dsl-and-scoped-semantics),
["Static symmetry checking in the pipeline"](./papers/CDL_connections.md#static-symmetry-checking-in-the-pipeline),
and ["Concrete symmetry DSL sketches"](./papers/CDL_connections.md#concrete-symmetry-dsl-sketches).

### 10.1 Core idea: symmetries act on `Pos(C,s)`

Containers provide index domains (`Pos(C,s)`), while symmetry declarations provide admissible actions on those domains.
For a symmetry element `g`, write:

```text
act_g : Pos(C,s) -> Pos(C,s)
```

A map is equivariant when it commutes with this action (up to the corresponding output action), and invariant when output is unchanged by the action.

Container view + symmetry view:
- containers answer: "what are valid positions and how do we traverse them?"
- symmetry answers: "which relabelings of those positions should be treated as structure-preserving?"

### 10.2 Set/list symmetry (permutation equivariance)

For `ListDesc`/set-like axes, a permutation `pi` acts by index relabeling.

```text
act_pi : Pos(ListDesc,n) -> Pos(ListDesc,n)   -- permutation of Fin n
```

Example (tokenwise shared map, equivariant in `i`):

```tl
container I = List[n]
tensor X(I, d)
tensor W(d, h)
tensor Y(I, h)

Y[i, h] := X[i, d] · W[d, h]
```

If indices on `I` are permuted, `Y` permutes the same way.
If we then reduce over `i` with a symmetric reducer (sum/mean/max), we obtain an invariant output.

### 10.3 Grid symmetry (translation equivariance)

For grid-like containers, translation acts on positions by offsets.

```text
act_t(pos) = pos + t
```

Convolution form:

```tl
Y[o] := W[p] · X[o + p]
```

Because the same kernel offsets are used at each `o`, shifting the input shifts the output accordingly.
This is the standard translation-equivariant conv property, expressed as action on container positions.

### 10.4 Graph/container automorphism symmetry

For graph/neighborhood containers, a graph automorphism `sigma` relabels vertices while preserving adjacency.

```text
act_sigma : V -> V
```

Neighborhood map must respect relabeling:

```text
sigma(nbr(v,p)) = nbr(sigma(v), p')
```

for a corresponding slot relabeling `p -> p'`.

Message-passing form:

```tl
Y[v, co] := W[co, ci, p] · X[nbr(v,p), ci]
```

Under a valid automorphism action, this remains equivariant if kernel sharing and slot/action coherence are respected.

### 10.5 Mask/sum containers and scoped symmetry

Symmetry should only be declared where structure supports it:

- `Maybe` mask axes: symmetry is usually inherited from underlying base axis, but "none/some" structure constrains which actions are admissible.
- `Sum`/branch axes: do not assume branch-permutation symmetry by default; branch tags often encode semantic roles.

Example:

```tl
container Mod = Sum{text,image}
tensor F(Mod, Tok, d)
```

A symmetry on `Tok` may be valid, while swapping `text` and `image` usually is not unless explicitly declared and semantically justified.

### 10.6 Compiler obligations from the combined view

Combining containers + symmetry gives concrete compiler obligations and failure checks:

1. **Action typing and closure checks**
   - Verify each declared action is well-typed on the target axis/container instance:
     `act_g : Pos(C,s) -> Pos(C,s)`.
   - Verify closure (mapped positions stay in-domain).
   - Fail if an action maps outside the declared position space.

2. **Action-law checks (where declared)**
   - Verify identity/composition constraints for declared symmetry families
     (for example permutation/group action consistency).
   - Fail if declared generators do not compose to valid position actions.

3. **Equivariance commuting checks for statements**
   - For statements marked/required equivariant, verify commutation:
     applying action before vs after the statement yields the same relabeling effect.
   - Practically, check gather/scatter index maps commute with `act_g`.
   - Fail with a diagnostic pointing to the offending index map or axis.

4. **Invariance checks for reductions**
   - For invariance claims, verify reducer/operator compatibility with symmetry
     (for example order-insensitive reducers for permutation invariance).
   - Verify reduced axes are contractible and symmetry-compatible.
   - Fail if reduction operator breaks declared invariance obligations.

5. **Scatter collision and symmetry coherence checks**
   - Reuse existing scatter legality checks (injective or explicit `reduce=` policy).
   - Additionally verify collision policy is compatible with symmetry claim
     (for example action should not change conflict-resolution meaning).
   - Fail if overlap handling is underspecified or symmetry-inconsistent.

6. **Scoped branch/mask checks**
   - Prevent symmetry assumptions from leaking across non-symmetric branches
     (`Sum`) or optional structure (`Maybe`) unless explicitly declared.
   - Verify branch/tag actions are explicitly justified if requested.
   - Fail on implicit branch-permutation assumptions.

7. **Stage placement and diagnostics**
   - Run these checks after axis/container resolution and before lowering optimizations.
   - Emit errors that name: axis/container, action symbol, statement, and violated law.
   - Prefer fail-loud semantics over silent fallback to non-symmetric behavior.

Concrete example (GNN equivariance + graph-level invariance):

```lean
axis v  : ℕ = n [symmetry permutation]   -- node axis; permutation acts here
axis p  : ℕ = pmax                       -- neighborhood slot axis
axis u  : ℕ = n                          -- neighbor node axis
axis c  : ℕ = cin                        -- input feature/channel axis
axis h  : ℕ = cout                       -- output feature/channel axis

container V   = List[n]
predicate adj_slot(v, p, u)      -- predicate-backed slot-to-neighbor relation

container Neighborhood[V] where
  shape := V
  pos(v) := { p | exists u, adj_slot[v,p,u] }
-- Well-formedness checks over `adj_slot`:
--   (1) existence: p is in pos(v) iff there exists at least one u with adj_slot[v,p,u]
--   (2) uniqueness: for fixed (v,p), at most one u satisfies adj_slot[v,p,u]
--   (3) closure: any selected u is a valid vertex index (u : V)
container Nbr = Neighborhood[V]

tensor X(V, c)            [equivariant v]   -- node features (input), permute with node relabeling
tensor W(c, h)            [symmetry none]   -- shared channel map / learnable weights
tensor M(V, h)            [equivariant v]   -- node-level messages/embeddings after neighborhood aggregation
tensor G(h)               [invariant v]     -- graph-level readout embedding (independent of node order)

assert equivariant(v):
  M[v, h] := adj_slot[v,p,u] · X[u, c] · W[c, h]   -- RHS-only p,u,c contracted

-- Note: `adj_slot` already gates slot/neighbor selection here, so no extra Iverson
-- factor is required in this equation.

assert invariant(v):
  G[h] := reduce_sum(v)(M[v, h])
```

Why this illustrates the synergy:
- container side (`Neighborhood[V]`) provides legal local index structure (`Pos(...,v)` from `adj_slot` existence, neighbor selection via `adj_slot`),
- symmetry side (`[symmetry permutation]`, `[equivariant v]`, `[invariant v]`) provides action/law obligations,
- together they let the compiler verify both indexing legality and equivariance/invariance claims in one pass.

Checks this example should trigger:
1. permutation action is well-typed on `V` positions and closed,
2. neighborhood map is coherent under relabeling (slot/neighbor compatibility),
3. message-passing equation commutes with node permutation (equivariance),
4. `reduce_sum(v)` discharges permutation invariance for graph readout,
5. diagnostics point to the exact axis/map/assertion on any violation.

Concrete example (2D CNN locality + translation equivariance):

```lean
axis ox : ℕ = nox [symmetry shift]   -- output x-position
axis oy : ℕ = noy [symmetry shift]   -- output y-position
axis px : ℕ = kx                     -- kernel slot x-offset
axis py : ℕ = ky                     -- kernel slot y-offset
axis ix : ℕ = nix [symmetry shift]   -- input x-position
axis iy : ℕ = niy [symmetry shift]   -- input y-position
axis c  : ℕ = cin                    -- input channel axis
axis h  : ℕ = cout                   -- output channel axis

predicate stencil2d_src(ox, oy, px, py, ix, iy)   -- slot (px,py) at output (ox,oy) selects input (ix,iy)
stencil2d_src[ox,oy,px,py,ix,iy] :=
  [ix = ox + px] ·
  [iy = oy + py] ·
  [0 <= ix < nix] ·
  [0 <= iy < niy]

container Stencil2D[OX, OY] where
  shape := OX × OY
  pos(ox, oy) := { (px,py) | exists ix, iy, stencil2d_src[ox,oy,px,py,ix,iy] }
-- Well-formedness checks over `stencil2d_src`:
--   (1) existence: (px,py) is in pos(ox,oy) iff there exists at least one (ix,iy) with stencil2d_src[ox,oy,px,py,ix,iy]
--   (2) uniqueness: for fixed (ox,oy,px,py), at most one (ix,iy) satisfies stencil2d_src[ox,oy,px,py,ix,iy]
--   (3) closure: any selected (ix,iy) is a valid input index pair (ix : ℕ = nix, iy : ℕ = niy)

tensor X(ix, iy, c)      [equivariant ix, iy]   -- input image/features
tensor K(px, py, c, h)   [symmetry none]        -- shared convolution kernel
tensor Y(ox, oy, h)      [equivariant ox, oy]   -- output feature map
tensor G(h)              [invariant ox, oy]     -- global pooled descriptor

assert equivariant(ox, oy):
  Y[ox, oy, h] := stencil2d_src[ox,oy,px,py,ix,iy] · X[ix, iy, c] · K[px, py, c, h]

assert invariant(ox, oy):
  G[h] := reduce_sum(ox, oy)(Y[ox, oy, h])
```

Checks this example should trigger:
1. shift action is well-typed on spatial axes and closed (respecting declared boundary semantics),
2. stencil map is coherent under shifts (source selection commutes with spatial relabeling),
3. convolution statement commutes with the shift action (translation equivariance),
4. global spatial reduction discharges invariance for `G[h]`,
5. diagnostics point to the exact axis/map/assertion on any violation.

This is the key synergy: containers provide the domain of legal indices; symmetry DSL provides legal relabelings and commuting obligations on that domain.

---

## 11. Compatibility and migration

1. Keep all existing programs valid.
2. Desugar existing axes to implicit `List[n]`.
3. Introduce container declarations incrementally.
4. Enforce contraction legality based on fold instances.
5. Reuse current affine/scatter compiler paths for the `ListDesc`/`St` specialization (surface: `List[n]`).

---

## 12. Summary

Generalized containers strictly extend classical tensor logic:

- Classical tensors = all-`ListDesc` generalized tensors (surface: all `List[n]` axes).
- Existing index notation remains the natural surface syntax for `List[n]`.
- Convolution/message passing are unified as reindex + pointwise + fold.
- Gather and scatter both generalize with mapped index slots.
- Contraction legality becomes explicit, typed, and compositional.

---

## Appendix A. Generalized predicates in practice

Generalized predicates follow the same indexing structure as generalized tensors.

```text
Pred(containers, shapes) := Ix(containers, shapes) -> Bool
```

So predicates can be homogeneous (`containers = [C, C, ...]`) or heterogeneous
(`containers = [C1, C2, ...]`) with the same `Shape/Pos` discipline.

Shape/Pos relationship to tensors:
- In many cases, a predicate and tensor share the same `Shape` and `Pos` (same coordinate system).
- The difference is codomain and role: tensors map indices to values, predicates map indices to truth/legality.
- Predicates can also refine availability by inducing a subset position family (for example valid slots/tokens only), while tensor payload semantics stay unchanged.
- Operationally: tensor says "what value is here?"; predicate says "is this index/route legal or active?".

Practical DSL patterns:

1. **Locality/stencil legality (CNNs)**

```tl
predicate stencil2d_src(ox, oy, px, py, ix, iy)
stencil2d_src[ox,oy,px,py,ix,iy] :=
  [ix = ox + px] ·
  [iy = oy + py] ·
  [0 <= ix < nix] ·
  [0 <= iy < niy]
```

Use: encodes legal local reads, including boundary behavior.
Corresponding `Shape/Pos`:
- `Shape(Stencil2D) = OX × OY` (output spatial locations),
- `Pos(Stencil2D, (ox,oy)) = { (px,py) | exists ix, iy, stencil2d_src[ox,oy,px,py,ix,iy] }`.

2. **Neighborhood selection (GNNs)**

```tl
predicate adj_slot(v, p, u)      -- slot p of v selects neighbor u
```

Use: encodes sparse/irregular connectivity while preserving typed legality checks
for existence, uniqueness per slot, and closure of selected neighbors.
Corresponding `Shape/Pos`:
- `Shape(Neighborhood) = V` (center node),
- `Pos(Neighborhood, v) = { p | exists u, adj_slot[v,p,u] }`.

3. **Ragged/variable-length validity**

```tl
predicate token_valid(b, t)
token_valid[b,t] := [t < len[b]]
```

Use: one program over padded batches while semantics are restricted to real tokens.
Corresponding `Shape/Pos`:
- `Shape(RaggedSeq) = B` (batch item),
- `Pos(RaggedSeq, b) = { t | token_valid[b,t] } = { t | t < len[b] }`.

4. **Branch/activity guards for multimodal `Sum`/`Maybe`**

```tl
predicate active_text(i)
predicate active_image(i)
```

Use: branch-specific equations can be gated without collapsing to an untyped global mask.
Corresponding `Shape/Pos` (one simple formulation):
- `Shape(ModalActive) = I` (item index),
- `Pos(ModalActive, i) = { text | active_text[i] } ∪ { image | active_image[i] }`.

5. **Scatter write legality as a predicate**

```tl
predicate write_target(o, p, y)
```

Use: captures where each write is allowed to land; compiler checks injectivity or
requires an explicit `reduce=` collision policy when overlaps are possible.
Corresponding `Shape/Pos`:
- `Shape(ScatterMap) = O` (output location),
- `Pos(ScatterMap, o) = { p | exists y, write_target[o,p,y] }`,
- selected write destination is recovered from `write_target[o,p,y]`.

Benefits of generalized predicates:
- **Unified sparsity model:** masks, raggedness, graph edges, and stencil legality share one typed representation.
- **Static legality proofs:** bounds, existence, uniqueness, and scatter collision obligations can be checked from predicate laws.
- **Symmetry-aware constraints:** equivariance/invariance conditions can be stated as predicate transport/commutation laws.
- **Learned structure:** adjacency/stencil predicates can be data-dependent while keeping `Shape/Pos` legality checks.
- **Program rewrites by logic:** compose/simplify predicates and eliminate dead regions before lowering.
- **Dependent routing:** branch activation for `Sum`/`Maybe` can be typed as predicate-governed position availability.
- **Certified optimization paths:** proven predicate properties justify sparse/local kernels instead of dense fallback.
- **Better diagnostics:** failures point to the exact predicate, axis, and violated law.

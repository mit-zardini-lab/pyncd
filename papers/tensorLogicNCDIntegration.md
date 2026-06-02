# Tensor Logic as an Interface to pyncd

This document examines the relationship between Tensor Logic (Domingos 2025) and the pyncd categorical framework. The central proposal is that tensor logic equations can be embedded into pyncd's `Term` hierarchy as a first-class `Operator` subclass. Each tensor equation becomes a `TensorEquation(Operator)` term, stored as the `operator` field of a `Broadcasted` base morphism; a `TensorProgram(Term)` collects equations and converts them to a `Composed` morphism via `Context`-mediated axis unification. Index variables in the embedding are `Axis(UTerm)` objects whose UIDs carry identity — tensor logic's implicit name-sharing becomes explicit UID identity, tracked by the same `Context` machinery used throughout pyncd.

---

## Contents

1. [Background: Tensor Logic](#1-background-tensor-logic)
   - [Core idea](#11-core-idea)
   - [The two-way equivalence](#12-the-two-way-equivalence)
   - [Tensor operations as logical operations](#13-tensor-operations-as-logical-operations)
   - [LHS index notation](#14-lhs-index-notation)
   - [Neural networks in tensor logic](#15-neural-networks-in-tensor-logic)
   - [Learning is free](#16-learning-is-free)
2. [Background: Einsum](#2-background-einsum)
   - [Syntax](#21-syntax)
   - [Formal semantics](#22-formal-semantics)
   - [Algebraic properties](#23-algebraic-properties)
   - [Delta tensors](#24-delta-tensors)
3. [Background: pyncd](#3-background-pyncd)
   - [The product category framework](#31-the-product-category-framework)
   - [Broadcasted: the base morphism](#32-broadcasted-the-base-morphism)
   - [Operators](#33-operators)
   - [UID and Context: axis identity](#34-uid-and-context-axis-identity)
4. [Coverage Analysis](#4-coverage-analysis)
   - [What tensor logic covers](#41-what-tensor-logic-covers)
   - [Gaps closed by the integration](#42-gaps-closed-by-the-integration-5)
   - [What tensor logic does not cover](#43-what-tensor-logic-does-not-cover)
   - [Summary](#44-summary)
5. [The Term-Based Integration](#5-the-term-based-integration)
   - [The integration boundary](#51-the-integration-boundary)
   - [`TensorEquation` as an `Operator` subclass](#52-tensorequation-as-an-operator-subclass)
   - [`bc_signature()`: conversion to `Broadcasted`](#53-bc_signature-conversion-to-broadcasted)
   - [Round-trip: equation and categorical structure stay in sync](#54-round-trip-equation-and-categorical-structure-stay-in-sync)
   - [`TensorProgram(Term)` and `to_morphism()`](#55-tensorprogramterm-and-to_morphism)
   - [Beyond `TensorProgram` — Extending the Tensor Logic Interface](#56-beyond-tensorprogram--extending-the-tensor-logic-interface)
6. [Iteration: Recurrent Tensor Equations](#6-iteration-recurrent-tensor-equations)
   - [DSL syntax](#61-dsl-syntax)
   - [The `Scan` term](#62-the-scan-term)
   - [Assembly in `_finalize_iter()`](#63-assembly-in-_finalize_iter)
   - [Affine fast path](#64-affine-fast-path)
   - [Compilation to PyTorch](#65-compilation-to-pytorch)
   - [Relationship to tensor logic's `*t` notation](#66-relationship-to-tensor-logics-t-notation)
7. [Summary](#7-summary)
8. [References](#references)

---

## 1. Background: Tensor Logic

For a detailed treatment of tensor logic and its foundations in einsum see [einsum_tensor_logic.md](einsum_tensor_logic.md). The subsections below summarise the concepts relevant to the pyncd integration.

### 1.1 Core idea

Pedro Domingos (2025) proposes **tensor logic** as a programming language whose sole primitive is the **tensor equation** — a named tensor defined as an einsum of other tensors with an optional nonlinearity applied to the result:

```text
Y[i, j] = relu(W[i, k] X[k, j])
```

Every statement is such an equation. A tensor logic *program* is a set of tensor equations. The language is intended to subsume both neural networks and logic programs: neural computation over the arithmetic semiring `(ℝ, +, ×)` and deductive reasoning over the Boolean semiring `(𝔹, ∨, ∧)` are the same structure under different semirings.

### 1.2 The two-way equivalence

The language rests on two formal equivalences.

**Relations are sparse Boolean tensors.** A relation `R(x, y)` is a Boolean matrix where `M_{xy} = 1` iff `(x, y) ∈ R`. The correspondence extends to n-ary relations and rank-n tensors; sparse storage is exactly tuple storage.

**A Datalog rule is an einsum over Boolean tensors.** The rule

```text
Aunt(x, z) ← Sister(x, y), Parent(y, z)
```

is equivalent to `A_{xz} = H(Σ_y S_{xy} P_{yz})` where `H` is the Heaviside step function converting a count of witnesses back to a Boolean. Without `H` this is a plain einsum over `(ℝ, +, ×)`. The join, projection, and existential quantification of relational algebra are all special cases of the einsum combine-then-aggregate structure.

### 1.3 Tensor operations as logical operations

| Logical / relational operation | Tensor logic operation |
| --- | --- |
| Natural join on shared variables | Elementwise product, summed over shared indices |
| Projection onto a subset of variables | Sum over non-output indices |
| Existential quantification | Projection + step function |
| Conjunction (AND) | Elementwise product |
| Disjunction (OR) | Elementwise sum |
| Negation (closed-world) | `1 − T` |

### 1.4 LHS index notation

Tensor equations use three kinds of LHS index. Two are relevant to this integration:

| Notation | Meaning |
| --- | --- |
| `i` | output index — stored in the result tensor |
| `t.` | normalization axis — function applied over the full slice along `t` |

The third notation, `*t` (virtual index — iterated in-place, enabling recurrence), is handled by the pyncd DSL via `.iteration_axis()` on a `TensorProxy`, which produces a `Scan` morphism — a first-class `Term` in the pyncd hierarchy. See §6 for a full description.

### 1.5 Neural networks in tensor logic

All standard architectures reduce to tensor equations. Selected examples from Domingos (2025):

```text
-- Perceptron
Y[i] = step(W[i,j] X[j])

-- Attention (softmax over normalization axis)
Comp[b, h, p, p'.] = softmax(Query[b,h,p,d] Key[b,h,p',d] / sqrt(D))

-- Output projection
Y[p, t.] = softmax(W_O[t,d] Stream[b,p,d])
```

A complete transformer fits in roughly a dozen tensor equations (Domingos 2025, Table 2).

### 1.6 Learning is free

The gradient of any tensor equation is itself a tensor equation. If `Y[...] = T[...] X₁[...] ... Xₙ[...]` then `∂Y[...]/∂T[...] = X₁[...] ... Xₙ[...]`. The full gradient of any loss is a sum of such equations and requires no special treatment beyond the equation language itself.

---

## 2. Background: Einsum

Tensor logic is built on einsum as its sole computational primitive. This section gives einsum a precise treatment since the mapping from tensor logic to pyncd runs through it.

### 2.1 Syntax

An einsum expression has the form:

```text
#(I₁, ..., Iₙ → I;  T₁, ..., Tₙ)
```

- `I₁, ..., Iₙ` — input index strings, one per tensor argument
- `I` — output index string
- `T₁, ..., Tₙ` — tensor arguments

Index symbols appearing only in inputs are **contracted** (summed over); symbols appearing on the output are **retained**.

| Operation | Linear algebra | Einsum |
| --- | --- | --- |
| Matrix product | `A · B` | `#(ij, jk → ik; A, B)` |
| Transposition | `Aᵀ` | `#(ij → ji; A)` |
| Inner product | `xᵀy` | `#(i, i → ; x, y)` |
| Outer product | `xyᵀ` | `#(i, j → ij; x, y)` |
| Trace | `tr(A)` | `#(ii → ; A)` |

### 2.2 Formal semantics

The result of an einsum over commutative semiring `(R, ⊕, ⊗)` is:

```text
T(x) = ⊕_{x̂: I → x}  ⊗ᵢ Tᵢ(x̂ ↾ Iᵢ)
```

For each output position `x`, aggregate over all global positions `x̂` projecting to `x` the combination of the corresponding input entries. This **combine-then-aggregate** structure unifies products, contractions, and reductions.

### 2.3 Algebraic properties

Wenig, Rump, Blacher, and Giesen (2025) formally prove commutativity, associativity, and distributivity of einsum. As a consequence:

- An n-tensor einsum decomposes into `n−1` binary einsums in any order (**contraction path**).
- Einsum distributes over elementwise aggregation, enabling algebraic simplification.

Together with Domingos (2025)'s identification of relations with Boolean tensors (§1.2), these properties imply that einsum rewriting rules correspond to query rewriting rules in relational algebra — a synthesis of the two frameworks, not a theorem stated by Wenig et al.

### 2.4 Delta tensors

Wenig et al. (2025) §7 introduce **delta tensors** as a first-class einsum primitive. The delta tensor `δ_o` over index structure `o` is the generalised identity matrix: `δ_o(x) = 1` if all components of `x` are equal, `0` otherwise. Trace (`#(ii → ; A)`) and diagonal extraction (`#(ii → i; A)`) are the canonical examples.

**Lemma 7.4** characterises delta tensors within einsum: `δ_o = #(I → II; 1_o)` — a delta tensor over `o` equals the einsum of the all-ones tensor `1_o` with a duplicated output index. Delta tensors are therefore expressible as ordinary einsums; they require no special primitive.

**Corollary 7.6** (delta removal) exploits this: any einsum expression containing a delta tensor can be rewritten into an equivalent delta-free einsum by substituting repeated indices. A delta tensor is a bookkeeping device, not a distinct operation.

**Correspondence with pyncd.** The diagonal-extraction case `Y[i] = X[i, i]` — a retained index appearing twice in one input — is the prototypical delta tensor pattern. In pyncd, `bc_signature()` handles this correctly without special-casing: when the same `Axis` object `i` appears twice in an input's `rhs` entry, `retained_uid_to_pos[i.uid]` is looked up twice, producing `Rearrangement(mapping=(0, 0), _dom=(i,))`. `Rearrangement.cod()` then returns `ProdObject((i, i))`, and `imprint_to_degree(cod)` fills both TILED slots with `i`, correctly recovering the `(i, i)` input shape. Delta tensor behaviour is a consequence of UID-based reindexing, not a separate mechanism — consistent with Corollary 7.6's result that delta tensors are eliminable.

---

## 3. Background: pyncd

For a detailed treatment of the pyncd categorical framework see [theory.md](theory.md). The subsections below summarise the concepts relevant to the tensor logic integration.

### 3.1 The product category framework

pyncd represents neural network computations as morphisms in a **product category** `ProdCategory[L, M]`. Morphisms take one of five forms:

```text
ProdCategory[L, M] = M                         -- atomic base morphism
                   | Composed[L, M]            -- sequential composition
                   | ThreadedComposed[L, M]    -- threaded composition with live-pool routing (see §5.6)
                   | ProductOfMorphisms[L, M]  -- parallel product
                   | Rearrangement[L]          -- permutation / copy / discard
                   | Block[L, M]               -- named sub-expression
                   | Scan[L, M]               -- iterative scan (see §6)
```

The framework specialises to two concrete categories.

**St (StrideCategory):** objects are lists of `Axis` terms (each carrying a `Numeric` size); morphisms are `StrideMorphism` objects — linear coordinate transforms mapping one list of axes to another. Composition is matrix multiplication. This is the **semantic** layer — it tracks how index spaces are linearly related.

**Br (BroadcastedCategory):** objects are lists of `Array[Datatype, Axis]`; base morphisms are `Broadcasted[B, A, O]` values. The full morphism type is the same recursive `ProdCategory[Array, Broadcasted]` union — `Broadcasted`, `Composed`, `ProductOfMorphisms`, `Rearrangement`, and `Block`. This is the **computational** layer — each `Broadcasted` is one operator application with its index structure fully specified.

### 3.2 Broadcasted: the base morphism

In pyncd, `Array[B, A]` is the object type in `BroadcastedCategory` — it is pyncd's name for what tensor logic calls a tensor: a typed, multi-dimensional data structure with datatype `B` and a shape given by a list of `Axis` objects. A `Broadcasted[B, A, O]` encodes a single operator application over one or more such arrays:

```python
@dataclass(frozen=True)
class Broadcasted[B: Datatype, A: Axis, O: Operator]:
    operator: O                            # what computation to perform
    input_weaves: Prod[Weave[B, A]]        # how input arrays map to degree axes
    output_weaves: Prod[Weave[B, A]]       # how output arrays map to degree axes
    reindexings: Prod[StrideCategory[A]]   # index rewriting for each input
```

A `Weave` is a **shape template** for one array (i.e. an `Array[B, A]`). Each position is one of two kinds:

- **Degree position** (`WeaveMode.TILED`) — a placeholder filled with a degree axis at runtime. The **degree** $P$ is the shared loop domain of all reindexings (`reindexings[i].dom()`, equal for all inputs). For `Einops`, $P$ equals the retained (output) index space of the einsum signature; for all other operators $P$ is empty and the output axes are all target positions in the output weave.
- **Target position** (a concrete `Axis` object) — an axis private to that array, not shared with the degree.

The domain of each input array is reconstructed by filling its weave's TILED positions with `reindexing.cod()` and leaving target positions in place. The **reindexing** is a `StrideCategory` morphism with `dom() = degree` whose mapping selects which subset of degree axes that input contributes — for `Y[i,j] = W[i,k] X[k,j]`, W's reindexing selects `i` and X's selects `j`. For pure einsums a `Rearrangement` suffices; strided convolutions require a full `StrideMorphism`. The contracted index `k` appears as a concrete target axis in each input's weave and is summed over by the operator.

| Index role | Weave position | Example (`Y[i,j] = W[i,k] X[k,j]`) |
| --- | --- | --- |
| Degree / output index | `WeaveMode.TILED` | `i`, `j` — TILED in W's and X's weaves respectively |
| Contracted input index | Concrete `Axis` in input weave | `k` — concrete Axis in both W's and X's weaves |
| Produced output index | Concrete `Axis` in output weave | embedding dim in `Embedding` |

### 3.3 Operators

`Operator` is an abstract `Term` base class. Every `Broadcasted` carries an `operator: O` field that specifies what computation the morphism performs. The existing subclasses cover the standard neural network vocabulary:

| Operator | Description |
| --- | --- |
| `Einops` | General einsum; degree = retained (output) indices |
| `Elementwise` (e.g. ReLU, σ) | Pointwise nonlinearities |
| `SoftMax` | Softmax along the `NormAxis` dimension; no masking |
| `MaskedSoftMax` | Softmax with Iverson predicate applied as $-\infty$ mask before exponentiation; TL DSL: `softmax(expr, where=pred)` |
| `Linear` | Weight matrix application |
| `Embedding` | Lookup table: `Natural → Reals` |
| `AdditionOp` | Elementwise sum of matching arrays |
| `Normalize` | Sum normalization: $x / \sum_\text{dim}(x)$; TL DSL: `normalize(expr)` |
| `MaskedNormalize` | Sum normalization with Iverson predicate applied as zero-mask before the denominator sum; TL DSL: `normalize(expr, where=pred)` |
| `WeightedTriangularLower` | Legacy causal mask (lower-triangular + sum normalization); superseded by `MaskedSoftMax` |

The following terms are proposed as part of the tensor logic integration and are discussed in detail in §5:

| Class | Superclass | Produces | Scope |
| --- | --- | --- | --- |
| `TensorEquation` | `Operator` | `Broadcasted` via `bc_signature()` | one equation = one base morphism |
| `TensorProgram` | `Term` | `ThreadedComposed` via `to_morphism()` | many equations = threaded sequential composition |

`TensorProgram` is not an `Operator`: `Operator` is the type parameter stored in `Broadcasted.operator` and represents a single atomic computation, whereas `TensorProgram` wraps multiple equations and `to_morphism()` produces a `ThreadedComposed` — a sequentially-composed morphism augmented with a live-pool routing table that delivers external tensors to every step that references them. See §5.5–5.6.

Because `Operator` is a `Term`, any `Operator` subclass participates in `deep_reconstruct` and `Context.apply` traversal. This is the hook that makes the tensor logic integration possible: `TensorEquation` is an `Operator` subclass, so its internal structure — including its `Axis` index fields — is reachable by the standard `Term` machinery.

### 3.4 UID and Context: axis identity

Every `Axis` is a `UTerm` — it carries a `UID` that acts as its unique identity across expressions. Two axes are the same if and only if they share a `UID`. A `Context` is a union-find structure over `UID` equality classes:

```python
@dataclass
class Context:
    equality_classes: list[EqualityClass]
    def append_iter(self, target: Iterable[UTerm]) -> None: ...
    def apply[T: GeneralTerm](self, target: T) -> T: ...
```

When two morphisms are composed, `Context.append_iter` unifies the codomain axes of the first with the domain axes of the second. `Context.apply` then substitutes canonical representatives throughout the expression, aligning the axes.

---

## 4. Coverage Analysis

This section maps tensor logic concepts onto pyncd and identifies where the correspondence holds, where it is partial, and where pyncd provides structure that tensor logic does not express.

### 4.1 What tensor logic covers

**Einsum → `Einops`.** A tensor logic equation `Y[i,j] = W[i,k] X[k,j]` corresponds directly to an `Einops` operator with index annotation `i k, k j -> i j`. The retained indices `i, j` form the degree and occupy `WeaveMode.TILED` positions in both input weaves and the output weave. The contracted index `k` occupies target (non-TILED) positions in each input weave, and the `Einops` operator performs the summation over it. The reindexings are `Rearrangement` terms selecting which degree axes each input contributes: W contributes `i` and X contributes `j`.

**Elementwise nonlinearity → `Elementwise` / `ReLU` / `SoftMax` / `Normalize`.** The optional nonlinearity in `Y[i] = relu(W[i,k] X[k])` selects the `Operator` subclass. The `.`-suffixed normalization axis maps to `SoftMax`, `MaskedSoftMax`, `Normalize`, or `MaskedNormalize` depending on whether a `where=` predicate is present.

**Contraction structure and execution layout → degree, weaves, and reindexings.** Retained indices (those on the LHS) form the degree of the `Broadcasted` and occupy `WeaveMode.TILED` positions in every weave where they appear. Contracted indices (those on the RHS but not the LHS) become concrete target positions in input weaves, and the `Einops` operator sums over them. Per-input `Rearrangement` reindexings follow directly from which degree axes each input carries. The full weave structure is therefore derivable from the equation's index structure — tensor logic covers execution layout implicitly. pyncd makes it explicit by reifying it as typed `Weave` and `Rearrangement` objects that code generation can consume directly.

**Sequential composition → `Composed`.** A tensor logic program where one equation feeds the next maps to `Composed([m1, m2, ...])` in pyncd. With the term-based integration, `TensorProgram.to_morphism()` constructs this automatically.

### 4.2 Gaps closed by the integration (§5)

The following gaps in tensor logic are addressed by the term-based embedding described in §5.

**Axis identity.** In tensor logic, index variables are syntactic: two equations sharing a letter `k` share that index by convention, with no semantic identity machinery. This is a gap at the language level. The term-based embedding closes it by representing index variables as `Axis(UTerm)` objects: sharing is object sharing tracked by UID, not name matching. The choice of `Axis(UTerm)` as the representation for index variables in §5 is what makes this work — `Context` can then operate directly on `TensorEquation` values without any name-to-UID translation step.

**Degree as a first-class object.** The `degree()` method on `Broadcasted` returns the contraction index space as a `ProdObject[A]`. This enables checking, at construction time, that all input reindexings agree on which axes are contracted. Tensor logic has no type-level representation of contraction structure; `bc_signature()` in §5.3 derives and reifies the degree from the equation's UID graph.

### 4.3 What tensor logic does not cover

**Parallel product.** Tensor logic has no notion of running two computations independently in parallel and combining their outputs as a product. In pyncd, `ProductOfMorphisms([m1, m2])` applies two morphisms to disjoint inputs and concatenates the outputs as a `ProdObject`. This is the categorical product structure — essential for expressing multi-head attention, where each head operates independently — and has no counterpart in tensor logic. A tensor logic program must either write one equation per head or add a head index to the single equation, flattening the structure.

**Datatypes.** pyncd distinguishes `Reals` (continuous-valued arrays) from `Natural(max_value=n)` (discrete token indices). The canonical example is an embedding layer: a lookup table that maps a discrete token index (an integer in `[0, vocab_size)`) to a continuous real-valued vector. The input type is fundamentally different from a real-valued tensor — it cannot be added or multiplied, only used to index into the table. pyncd captures this at the type level: `Embedding.template('vocab', output_size=d_model)` produces a morphism with domain type `Natural(max_value=vocab)` and codomain type `Reals`, carrying the vocabulary size as a `Numeric` expression through the type system. Tensor logic treats all tensors uniformly as elements of a semiring — the equation `Y[i, d] = W[i, d]` looks identical whether `i` is a discrete token index or a real-valued sequence position. `TensorEquation` carries no datatype information; datatypes must be supplied as arguments to `bc_signature()` when constructing the `Weave` objects.

**Symbolic shape inference.** Axis sizes in pyncd are `Numeric` expressions — formal terms subject to symbolic manipulation. The size of a composed expression is derived algebraically from its components. `TensorProgram.to_morphism()` produces a morphism with symbolically-typed domain and codomain; tensor logic programs carry no shape information.

**Block structure.** `Block[L, M]` in pyncd names a sub-expression with a `BlockTag` and optional aesthetics. This supports structured display and selective substitution. `TensorProgram` is flat; hierarchical grouping is imposed by the surrounding pyncd expression.

### 4.4 Summary

| Concept | Tensor logic | pyncd |
| --- | --- | --- |
| Single einsum | `Y[i,j] = W[i,k] X[k,j]` | `TensorEquation` → `Broadcasted` via `bc_signature()` |
| Elementwise nonlinearity | `relu(...)` in equation | `operator` field on `TensorEquation` |
| Normalization axis | `t.` suffix | `SoftMax` / `MaskedSoftMax` / `Normalize` / `MaskedNormalize` operator |
| Sequential composition | Feed-forward equation chain | `TensorProgram.to_morphism()` → `ThreadedComposed` |
| Parallel composition | None | `ProductOfMorphisms([...])` |
| Axis identity | Syntactic name sharing | `Axis(UTerm)` with UID — object sharing |
| Datatypes | None (uniform semiring) | `Reals` / `Natural(max_value)` |
| Symbolic shapes | None | `Numeric` expressions on `Axis._size` |
| Sub-expression naming | None | `Block` + `BlockTag` |

---

## 5. The Term-Based Integration

### 5.1 The integration boundary

The coverage analysis locates a natural boundary: **one tensor equation corresponds to one `Broadcasted`**. A tensor equation specifies exactly what `Broadcasted` encodes:

1. Which tensors are combined (the inputs)
2. Which indices are contracted (the degree)
3. Which indices are retained (the targets)
4. What nonlinearity is applied (the operator)

The term-based integration achieves this by making `TensorEquation` an `Operator` subclass — a `Term` in the pyncd hierarchy. The equation is stored as the `Broadcasted.operator` field and is never discarded. Because `Operator` is a `Term`, `deep_reconstruct` and `Context.apply` traverse into `TensorEquation`'s internal fields, keeping the equation and the categorical structure in sync as axes are aligned.

### 5.2 `TensorEquation` as an `Operator` subclass

```python
@dataclass(frozen=True)
class TensorEquation(Operator):
    lhs_name:    DynamicName | None = None
    lhs_indices: Prod[RawAxis] = ()                                    # retained — identified by UID
    rhs:         Prod[TensorRef | IversonBinOp | IversonUnaryOp] = ()  # factors; TensorRef for named tensors
    operator:    Operator | None = None                                # nonlinearity; None means Identity
```

The index variables in `lhs_indices` and `rhs` are `Axis` objects — `UTerm` subclasses carrying UIDs. The same `Axis` object appearing in multiple positions encodes index sharing. An index is **contracted** if its UID appears in any `rhs` entry but not in `lhs_indices`; it is **retained** if its UID appears in `lhs_indices`. No string matching is involved: identity is UID identity.

Each element of `rhs` is a `TensorRef(name, axes)` for a named input tensor, an `IversonBinOp`, or an `IversonUnaryOp` for predicate factors. The equation `Y[i, j] = W[i, k] X[k, j]` is constructed as:

```python
i = RawAxis.named('i')
j = RawAxis.named('j')
k = RawAxis.named('k')

eq = TensorEquation(
    lhs_name=DynamicName('Y'),
    lhs_indices=(i, j),
    rhs=(
        TensorRef(DynamicName('W'), (i, k)),
        TensorRef(DynamicName('X'), (k, j)),
    ),
    operator=Identity(),
)
```

The shared `k` object carries UID identity: its UID appears in both inputs but not in `lhs_indices`, making the contraction unambiguous.

For an equation with a nonlinearity,

```text
Y[b, p, t.] = softmax(W_O[t, d] Stream[b, p, d])
```

the `operator` field carries the `SoftMax` instance and the `.`-suffixed axis `t` is distinguished from plain retained indices by using `NormAxis` — a zero-field frozen dataclass subclass of `RawAxis` that marks the normalisation dimension. `NormAxis` is constructed via `norm_axis(name)` or `norm_axis(name, size)` in the DSL:

```python
b      = RawAxis.named('b')   # batch
p      = RawAxis.named('p')   # sequence position
d      = RawAxis.named('d')   # model dimension (contracted)
t_norm = norm_axis('t')       # NormAxis — marks the normalisation dimension

eq = TensorEquation(
    lhs_name=DynamicName('Y'),
    lhs_indices=(b, p, t_norm),
    rhs=(
        (DynamicName('W_O'), (t_norm, d)),
        (DynamicName('Stream'), (b, p, d)),
    ),
    operator=SoftMax(),
)
```

### 5.3 `bc_signature()`: conversion to `Broadcasted`

`Operator` declares `bc_signature() → Broadcasted` as the standard construction pathway. `TensorEquation` implements it by reading the contraction structure from the UID graph:

1. **Retained axes** (`Axis` values whose UIDs appear in `lhs_indices`) — these form the degree; they occupy `WeaveMode.TILED` positions in every weave where they appear.
2. **Contracted axes** (`Axis` values whose UIDs appear in `rhs` but not in `lhs_indices`) — these become target (non-TILED) positions in their respective input weaves; the operator sums over them.
3. **Per-input reindexing**: build a `StrideCategory` morphism with `dom()=degree` selecting which degree axes this input participates in (a `Rearrangement` suffices for pure einsum).
4. **Return** `Broadcasted(operator=self, input_weaves=..., output_weaves=..., reindexings=...)`.

The result is `Broadcasted[B, A, TensorEquation]` — the operator field IS the equation.

### 5.4 Round-trip: equation and categorical structure stay in sync

Because `TensorEquation` is a `Term`, `Context.apply` reaches into its `lhs_indices` and `rhs` fields via `deep_reconstruct` and substitutes canonical axis UIDs alongside the `Broadcasted`'s weaves and reindexings. When two `Broadcasted[B, A, TensorEquation]` morphisms are composed and their shared axes are unified, both the categorical structure and the embedded equation are updated consistently.

This enables:

- **Display**: render `Broadcasted.operator` as a tensor logic equation string via `eq.to_string()`.
- **Round-trip editing**: modify `TensorEquation` fields and call `bc_signature()` to re-derive the `Broadcasted`.
- **Axis tracing**: the `Axis` objects in `TensorEquation.lhs_indices` are the same objects as those in `Broadcasted.output_weaves`, so any UID substitution is automatically reflected in both representations.

### 5.5 `TensorProgram(Term)` and `to_morphism()`

A tensor logic program is a set of equations linked by shared tensor names. `TensorProgram` represents this as a `Term` and converts it to a pyncd morphism:

```python
@dataclass(frozen=True)
class TensorProgram(Term):
    equations: Prod[TensorEquation]

    def to_morphism(
        self,
        declarations: dict[DynamicName, tuple[RawAxis, ...]] | None = None,
        array_datatypes: dict[DynamicName, Datatype] | None = None,
    ) -> ThreadedComposed:
        ctx = Context()
        morphisms: list = []
        name_to_axes: dict[DynamicName | None, Prod[RawAxis]] = {}
        external_axes: dict[DynamicName, tuple[RawAxis, ...]] = {}
        internal_names = {eq.lhs_name for eq in self.equations}
        equations_sorted = topological_sort(self.equations)
        # Collect external tensor names in first-appearance order.
        external_order: list[DynamicName] = []
        external_name_set: set[DynamicName] = set()
        for eq in equations_sorted:
            for factor in eq.rhs:
                if isinstance(factor, TensorRef) and factor.name not in internal_names \
                        and factor.name not in external_name_set:
                    external_order.append(factor.name)
                    external_name_set.add(factor.name)
        n_external = len(external_order)
        ext_idx = {name: i for i, name in enumerate(external_order)}
        produced_idx: dict[DynamicName, int] = {}
        step_routing: list[tuple[int, ...]] = []
        for eq in equations_sorted:
            seen_in_eq: set[DynamicName | None] = set()
            for factor in eq.rhs:
                if not isinstance(factor, TensorRef):
                    continue
                name = factor.name
                if name in name_to_axes and name not in seen_in_eq:
                    # Internal tensor produced by a prior step: unify axes.
                    seen_in_eq.add(name)
                    for prior_axis, eq_axis in zip(name_to_axes[name], factor.axes):
                        ctx.append_iter((prior_axis, eq_axis))
                elif name not in internal_names and name not in seen_in_eq:
                    # External tensor: unify axes across equations that share it.
                    seen_in_eq.add(name)
                    if name in external_axes:
                        for prior_ax, eq_ax in zip(external_axes[name], factor.axes):
                            ctx.append_iter((prior_ax, eq_ax))
                    else:
                        external_axes[name] = factor.axes
                # self-joins: skip subsequent occurrences (each keeps its own UIDs)
            applied_eq = ctx.apply(eq)
            morphisms.append(_split_nonlinearity(applied_eq, array_datatypes=array_datatypes))
            name_to_axes[eq.lhs_name] = applied_eq.lhs_indices  # post-apply canonical axes
            # Build per-step routing: map each domain input to its live-pool slot.
            route = tuple(
                ext_idx[f.name] if f.name in ext_idx else n_external + produced_idx[f.name]
                for f in applied_eq.rhs if isinstance(f, TensorRef)
            )
            step_routing.append(route)
            produced_idx[eq.lhs_name] = len(produced_idx)
        return ThreadedComposed(
            content=tuple(morphisms),
            routing=tuple(step_routing),
            n_external=n_external,
        )
```

`topological_sort` orders equations so that each tensor is defined before it is used. The `name_to_axes` map translates tensor logic's implicit name-sharing into UID unification: when equation B refers to tensor `Hidden` that was defined by equation A, `ctx.append_iter` unifies A's `lhs_indices` with B's corresponding `rhs` `TensorRef` entry. `ctx.apply(eq)` then substitutes canonical UIDs into both the equation and its resulting `Broadcasted`.

#### Self-joins on computed intermediates

A **self-join** occurs when the same intermediate tensor appears more than once in a consuming equation's `rhs` — the canonical ML instance being the Gram matrix:

```text
H[a, b]  = W[a, k] X[k, b]       -- intermediate
Y[i, j]  = H[i, k] H[j, k]       -- Gram matrix: H self-joined on the row axis
```

The naïve unification loop iterates over every rhs entry for every intermediate tensor. When `H` appears twice, it unifies `name_to_axes['H'][0]` with both `i` and `j`, collapsing them into one equivalence class and silently producing `Y[i, i]` — a diagonal rather than a matrix. The self-join is destroyed without any error.

The fix tracks which intermediate tensors have already been seen within the current equation's rhs and skips `ctx.append_iter` for subsequent occurrences:

```python
seen_in_eq: set[DynamicName | None] = set()
for factor in eq.rhs:
    if isinstance(factor, TensorRef) and factor.name in name_to_axes \
            and factor.name not in seen_in_eq:
        seen_in_eq.add(factor.name)
        for prior_ax, eq_ax in zip(name_to_axes[factor.name], factor.axes):
            ctx.append_iter((prior_ax, eq_ax))
    # subsequent occurrences (self-joins): skip — each reference keeps its own axis UIDs
```

For the first occurrence of `H`, `i` is unified with `H`'s canonical row axis and `k` is unified with `H`'s canonical column axis as before. For the second occurrence, `j` is left untouched (independent UID) and `k` — being the same Python object as in the first occurrence — is already in the canonical class, so the coupled contraction `Σ_k H[i,k] H[j,k]` is correctly expressed. `bc_signature()` then produces a `Broadcasted` with two independent input weaves for `H`, one contributing degree axis `i` and one contributing `j`.

The one consequence of skipping subsequent unifications is that automatic size propagation does not reach axes that appear only in additional occurrences. In the Gram matrix example, `j`'s size is not inferred from `H`'s row size. The mitigation is an explicit declaration for `Y` (or `H`) — the existing `declarations` path in `to_morphism()` handles this and is unaffected by the change.

### 5.6 Beyond `TensorProgram` — Extending the Tensor Logic Interface

At and below `TensorProgram`, the tensor logic term representation covers sequential composition, einsum structure, nonlinearities, normalization axes, and axis identity. The remaining gaps — parallel product, datatypes, symbolic shapes, and block structure — are the caller's responsibility above `TensorProgram.to_morphism()` and are catalogued in §4.3.

This section describes what has been implemented in the Python DSL layer (`data_structure/TensorDSL.py`) and what remains as future work.

#### Python DSL: Implemented

`TensorDSL.py` provides a Python front-end for building pyncd morphisms directly from tensor-equation notation, without manually populating dataclass fields. The core of the DSL is the `TL` registry:

```python
tl = TL()
i, j, k = axes('i j k')
tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
morph = tl.to_morphism()     # Broadcasted — built eagerly at assignment time
eq    = tl.to_equation()     # TensorEquation — extracted from morph.operator
```

`TL.__getattr__` returns a `TensorProxy` for any tensor name. Subscripting a proxy (`tl.W[i, k]`) returns an `IndexedTensor`. The `*` operator accumulates factors into an `RHSExpression`. Assignment (`tl.Y[i,j] = ...`) **immediately builds a `Broadcasted` morphism** and stores it — no equation accumulation, no deferred conversion step.

`relu()`, `softmax()`, and `normalize()` wrap an expression with the corresponding `Operator`. Both normalizing functions operate over the axis in the LHS constructed with `norm_axis()` — that axis is typed as `NormAxis`, a frozen zero-field subclass of `RawAxis`, and its presence in `lhs_indices` signals the normalization dimension to downstream display and code generation. `norm_axis` accepts an optional concrete size:

```python
x = norm_axis('x', SEQ)      # NormAxis — marks the normalization dimension; size = SEQ
q, h, k = axes('q h k')
tl.Comp[h, q, x] = softmax(tl.Query[q, h, k] * tl.Key[x, h, k])
```

Both `softmax()` and `normalize()` accept an optional `where=` keyword argument taking an `IversonExpr`. When supplied, positions where the predicate is False are masked before the normalization denominator is computed — masked to $-\infty$ for softmax (so `exp(-∞) = 0`) and to $0$ for normalize — and those positions output zero:

```python
x = norm_axis('x', SEQ)
q = real_axis('q', SEQ)
tl.S[h, q, x] = softmax(tl.Q[q, h, k] * tl.K[x, h, k], where=(x <= q))  # causal mask
tl.A[q, m]    = normalize(tl.Attn[q, m] + tl.H[q, m])                    # plain normalize
tl.B[q, m]    = normalize(tl.Attn[q, m], where=(q <= some_axis))          # masked normalize
```

`_split_nonlinearity()` in `TensorLogic.py` detects a `SoftMax` or `Normalize` operator with a non-empty `where_predicate` and routes to `MaskedSoftMax.template()` or `MaskedNormalize.template()` respectively. The helper `_compute_mask_alignment(iverson_factor, lhs_indices)` computes the permutation and broadcast count needed to align a materialised Iverson mask with the score tensor at runtime; `ConstructedMaskedSoftMax` and `ConstructedMaskedNormalize` in `torch_compile.py` use these alignments in `forward()`.

**`normalize()` semantics.** `normalize()` compiles to sum-normalization: $x / \sum_\text{dim}(x)$ (clamped to avoid division by zero). It is **not** RMSNorm or LayerNorm. For a row of attention weights, `normalize()` renormalises so the unmasked positions sum to 1 — the correct operation after a causal mask has zeroed future positions.

**Additive expressions.** The `+` operator on `IndexedTensor` and `RHSExpression` produces a `SumExpr` — a flat list of `RHSExpression` terms. Assigning a `SumExpr` builds a `Composed(ProductOfMorphisms(terms), add_br)` where `add_br` is a `Broadcasted(AdditionOp(), ...)` that sums the term outputs element-wise:

```python
tl.Out[i] = tl.A[i] + tl.B[i]
tl.Out[i] = relu(tl.H[i, k] * tl.W[k]) + tl.Bias[i]
```

**Eager axis unification.** The `TL` instance holds a single `_ctx: Context` shared across all assignments. When a tensor appears on the RHS that was defined by a prior assignment, its axes are unified with the prior output axes via `_ctx.append_iter` before building the morphism. `tl.to_morphism()` builds a live-pool routing table from per-entry `input_names` and returns a `ThreadedComposed` — or falls back to `Composed` for coupled Scan entries. No topological sort is required; equations are already recorded in assignment order.

**Cross-`TL` composition.** When two morphisms built from separate `TL()` instances are composed via `@`, their boundary axes have different UIDs even if they share names and sizes. `composition()` handles this via shape-based matching: it matches elements of `dom(g)` to elements of `cod(f)` by axis shape signature (a tuple of (name, size) pairs). If the shared element is not at the positionally expected location, a `Rearrangement` is prepended to `g` so that matched elements come first before standard positional alignment proceeds. This allows `attn_res() @ ffn_res()` to work when the two functions were built from separate `TL()` instances.

For programs that need to inspect or re-process the underlying `TensorEquation` objects (e.g. for serialisation via `from_tensor_program`), `tl.to_program()` extracts them from the stored morphisms and wraps them in a `TensorProgram`. `TensorProgram.to_morphism()` remains available as an alternative build path and performs its own `Context`-mediated unification on the extracted equations.

**Tensor declarations** attach kind and shape metadata to a named tensor before it is used in equations:

```python
tl.W_in.tensor(real_axis('d_ff', 2048), real_axis('d', 512))
tl.Mask.predicate(real_axis('q'), real_axis('x'))
tl.E.selection(nat_axis('v', 50257), real_axis('d', 512))
```

Declarations are registered via `.tensor()`, `.predicate()`, and `.selection()` on a `TensorProxy` and stored in `TL._declarations`. They have two effects at indexing time: **arity checking** (the number of indices must match the declared shape) and **axis promotion** for SELECTION kinds. SELECTION tensors promote indices at positions declared as `NatAxis` to `NatAxis`. PREDICATE tensors carry `Bool` datatype metadata (used in `bc_signature()`) but do not promote axes. TENSOR declarations perform no promotion — they carry shape and size metadata only.

**Typed axes** encode the ℕ vs ℝ distinction and optional concrete sizes directly on `Axis` objects:

```python
nat_axis('v', 50257)   # NatAxis with _size = Integer(50257)
nat_axis('v')          # NatAxis with _size = FreeNumeric  (type only, no size)
real_axis('d', 512)    # RawAxis with _size = Integer(512)
real_axis('d')         # RawAxis with _size = FreeNumeric  (type only, no size)
```

`NatAxis` is a frozen zero-field dataclass subclass of `RawAxis`, following the same pattern as the existing `NormAxis`. The type distinction is carried by the Python class; size is carried by the `_size` field inherited from `Axis`. A declaration without concrete sizes is valid and serves as a kind annotation without binding sizes.

Declarations are **entirely optional**. When absent, all existing behaviour — contraction detection, UID-based axis unification, `bc_signature()`, `to_morphism()` — is completely unchanged.

#### Implicit threading: `ThreadedComposed` and the live-pool model

`TensorProgram.to_morphism()` and `TL.to_morphism()` produce a `ThreadedComposed` morphism — a subtype of sequential composition that carries an explicit per-step input routing table. This enables external tensors (weights, residual inputs) referenced by more than one step to be delivered to every step that needs them, rather than being consumed by the first step and dropped.

**Live-pool model.** `ThreadedComposed` maintains a *live pool* — a flat list of tensors indexed by position:

```text
live[0 .. n_external - 1]   initial caller inputs, in first-appearance order
live[n_external + i]        output of step i
```

The `routing` field is a tuple of tuples: `routing[i][j]` is the live-pool index for input slot `j` of step `i`. `n_external` is the count of initial caller inputs. `ConstructedThreadedComposed.forward()` executes this routing:

```python
def forward(self, *xs: torch.Tensor):
    live = list(xs)         # n_external slots
    for module, route in zip(self.chain, self.routing):
        last = to_tuple(module(*(live[i] for i in route)))
        live.extend(last)   # append this step's output(s)
    return last
```

**Residual connections without a fork.** Before `ThreadedComposed`, expressing `normalize(Attn + H)` where `H` also feeds Q/K/V projections required a `(0, 0)` rearrangement to copy `H` into two domain positions. With threading, `H` is assigned a single live-pool slot and the routing table delivers it to every step that needs it:

```python
# attn_res(): H is external; routing delivers it to projections and to the residual.
tl.Query[q, h, k]   = tl.W_Q[h, k, m] * tl.H[q, m]
tl.Key[x, h, k]     = tl.W_K[h, k, m] * tl.H[x, m]
tl.Value[x, h, k]   = tl.W_V[h, k, m] * tl.H[x, m]
# ... attention core steps ...
tl.A[q, m]          = normalize(tl.Attn[q, m] + tl.H[q, m])   # H threaded, no fork
```

**`TL._entries` 4-tuple.** Each completed assignment is stored as `(lhs_name, morph, out_axes, input_names)` where `input_names: tuple[DynamicName | None, ...]` records one name per domain slot of `morph` (`None` for unsized Iverson predicates). `to_morphism()` walks these names to build `external_order` and the routing table.

**`_compiled()` helper.** Within `to_morphism()`, a `_compiled()` helper applies `_split_nonlinearity` to any entry whose stored morphism is a `Broadcasted(operator=TensorEquation(operator=SoftMax|ReLU|Normalize|...))`. This ensures inline nonlinearities such as `softmax(Q * K)` are split into an einsum step followed by the nonlinearity template before being placed in the `ThreadedComposed` chain. The `_entries` list is left untouched; the split is applied only when building the final `content` tuple.

**Norm-axis-invariant term dropping.** Before `bc_signature()` is called for equations with a `NormAxis` in the LHS (softmax and normalize equations), `_register_entry()` runs a pre-pass over `SumExpr` operands: any additive term whose factor indices are disjoint from the `NormAxis` UID is a constant along the normalization axis and is dropped. This realises the algebraic identity `softmax(f + c) = softmax(f)` — the constant `c` does not change the output — without requiring `bc_signature()` to handle the additive structure.

**Dead code elimination.** `to_morphism()` runs a backward BFS from the final entry over `input_names` before building the routing table. Equations whose output is never consumed by any downstream step (dead code) are dropped from the compiled `content` tuple, so the resulting `ThreadedComposed` contains no unreachable steps.

**Scan in the live pool.** Uncoupled `Scan` entries (a single recurrent state, `n_states == 1`) now populate `input_names` via `_external_names_from_value()` and participate in the `ThreadedComposed` live-pool routing table like any other step. The plain `Composed` fallback in `to_morphism()` is only triggered for coupled Scans (`n_states > 1`) whose `input_names` remain empty.

#### Parallel product from dependency analysis

Not yet implemented. A `TensorProgram`'s dependency DAG encodes parallelism implicitly: two equations with no directed path between them are independent and could be composed as `ProductOfMorphisms` rather than sequentially. The full program decomposes into alternating sequential steps and parallel blocks by finding **fork-join pairs** in the DAG — a fork where one tensor feeds multiple independent chains, a join where those chains reconverge. Between fork and join, the chains become the arguments of a `ProductOfMorphisms`; finding these pairs is a standard dominators/post-dominators analysis on the DAG.

The main complication is shared inputs: parallel branches often read from the same tensor (both attention heads read from the same embedded sequence). Since `ProductOfMorphisms` requires disjoint domain objects, shared inputs must be fanned out via a `Rearrangement` (copy mode) before the parallel block and outputs concatenated via another `Rearrangement` at the join. The UID-based `Axis` representation makes shared-input detection straightforward — the same `Axis` objects appearing in multiple branches are immediately identifiable.

This analysis is the inverse of tensor logic's conventional head-index trick, which encodes multi-head attention as a single batched equation with an explicit head dimension `h`. The two representations are semantically equivalent; the DAG analysis recovers the `ProductOfMorphisms` structure automatically from either form, exposing the parallelism explicitly to the pyncd type system.

#### Symbolic Shape Inference

Implemented. Tensor logic equations carry no size information. The DSL addresses this through **explicit axis annotation** and **eager size propagation**: `nat_axis('v', 50257)` and `real_axis('d', 512)` attach `Integer(n)` directly to the `_size` field of each axis at construction time, and tensor declarations carry this information positionally:

```python
tl.W_in.tensor(real_axis('d_ff', 2048), real_axis('d', 512))
```

This records that `W_in`'s first axis has size 2048 and its second has size 512. Type-only declarations (no concrete size) are equally valid and record only the kind:

```python
tl.W_in.tensor(real_axis('d_ff'), real_axis('d'))
```

Declaration axes are separate objects from the equation-level axes (always created via `axes()`). Size propagation happens **eagerly at assignment time**: before building the morphism for an equation, `_register_entry` unifies each declaration axis with the corresponding LHS axis via `_ctx.append_iter((decl_ax, lhs_ax))`. This fires the same `Context` machinery as inter-equation unification, so declared sizes propagate through `bc_signature()` and downstream into `dom()` and `cod()` types. Axes of different subtypes (e.g. a `RawAxis` declaration slot paired with a `NormAxis` LHS index) are skipped — size propagation does not cross subtype boundaries.

#### Tensor kind declarations

The three-way distinction between contraction, selection, and predicate tensors is implemented in the Python DSL via tensor declarations. Each tensor is declared with a kind and a positional shape before it appears in equations:

```python
tl.W_in.tensor(real_axis('d_ff', 2048), real_axis('d', 512))   # ℝ: sum over shared indices
tl.E.selection(nat_axis('v', 50257), real_axis('d', 512))       # ℕ → ℝ: lookup by token ID
tl.Mask.predicate(real_axis('q'), real_axis('x'))               # 𝔹: existential over shared indices
```

Kind and shape together record what the tensor *is* independently of how it appears in any particular equation. This is the Python realisation of the arrow-notation design target — `.tensor()` for `ℝ → ℝ` weights, `.selection()` for `ℕ → ℝ` embedding tables, `.predicate()` for `(ℕ,ℕ) → 𝔹` relations — with size either concrete (`Integer(n)`) or left symbolic (`FreeNumeric`).

At indexing time, declarations enforce arity and promote axis types: SELECTION tensors promote indices at positions declared as `NatAxis` to `NatAxis`. PREDICATE and TENSOR declarations carry shape and size metadata only, with no axis promotion. The type distinction survives through `bc_signature()` and `Context`-mediated unification and is visible in `TensorEquation.rhs` axis types.

The three kinds map to the following equation-level notation and pyncd datatypes:

| Declaration | Equation syntax | Python mechanism | Semantics | pyncd datatype | Status |
| --- | --- | --- | --- | --- | --- |
| `.tensor(...)` | `T[i, j]` | `__getitem__` | contraction — sum over shared indices | `Reals` | implemented |
| `.predicate(...)` | `T[x, y]` | `__getitem__` | predicate — Bool-typed, existential over shared indices | `Bool` | implemented (axis promotion deferred) |
| `.selection(...)` | `T[d]` with NatAxis slot | `__getitem__` | selection — lookup row by token ID | `Natural(max_value=n)` | implemented |

All three kinds use `__getitem__` (`[]`) on `TensorProxy`. The declaration records the kind, and the DSL cross-validates usage at indexing time: arity is checked against the declared shape, and SELECTION tensors promote `NatAxis`-declared slots. The kind distinction is carried through `bc_signature()` via the `array_datatypes` argument: PREDICATE tensor names are mapped to `bc.Bool()`, and SELECTION tensor names to `bc.Natural(...)`, so the resulting `Weave` objects carry the correct datatype for downstream code generation.

**Selection (embedding lookup).** The core distinction for selection is that tensor logic's contraction `Σ_i A[i,...] B[i,...]` *sums* over `i`, whereas a lookup *selects* one row — a token index is a pointer into a table, not a summable weight. The `.selection()` declaration captures this at the type level: positional slots declared as `NatAxis` are promoted at indexing time, flagging the vocabulary dimension as ℕ rather than ℝ. The lookup equation itself is deferred: pyncd already encodes it correctly via `ops.Embedding.template(vocab_size)`, whose input weave has `Natural(vocab_size)` as its datatype and empty shape, encoding the vocabulary axis as a type rather than a shape axis. Extending the DSL to express this as a first-class equation requires no pyncd changes; the full scope of issues is recorded in [dsl_embedding_lookup_extension.md](../docs/dsl_embedding_lookup_extension.md).

**Predicate tensors.** The semiring distinction — Boolean `(𝔹, ∨, ∧)` vs arithmetic `(ℝ, +, ×)` — is partially implemented. `.predicate()` marks a tensor's datatype as `Bool` in `_array_datatypes()`, causing `bc_signature()` to emit `Weave(Bool(), ...)` for that tensor's input weave. However, contracted indices over predicate tensors are not yet distinguished from arithmetic summation — the `Einops` operator carries no `semiring` field. Realising the full semiring distinction requires adding a `semiring` field to `Einops`, updating contraction semantics for `Bool` inputs (∨ instead of +, ∧ instead of ×), and adding `Bool` rendering support in tsncd. The full design is recorded in [bool_semiring_extension.md](../docs/bool_semiring_extension.md).

---

## 6. Iteration: Recurrent Tensor Equations

Tensor logic's `*t` (virtual index) notation for in-place iteration is realised in the pyncd DSL through `.iteration_axis()` on a `TensorProxy`. Rather than a fixpoint combinator, this produces a `Scan` term — a new first-class construction rule in the pyncd morphism grammar, defined alongside `Broadcasted`, `Composed`, `ProductOfMorphisms`, `Rearrangement`, and `Block`.

### 6.1 DSL syntax

An iterative tensor requires three declarations in the `TL` registry:

```python
from data_structure.TensorDSL import TL, real_axis, axes

tl = TL()
i    = real_axis('i', 16)
l    = real_axis('l', 10)   # 10 recurrence steps; size must be concrete

tl.H.iteration_axis(l)                          # declare H as iteratively defined over l
tl.H[i, 0]   = tl.X[i]                         # base case: H[:, 0] = X
tl.H[i, l+1] = tl.H[i, l] + lr * tl.Grad[i, l]  # inductive step
```

- **`.iteration_axis(l)`** registers `l` as the recurrence axis for `H`. `l` must be a `real_axis` with a concrete integer size (`l._size` must be an `nm.Integer`); this is checked immediately.
- **Base case** (`tl.H[i, 0] = ...`): a literal integer at the `l` slot. `TensorProxy.__setitem__` detects the `int` and stores the equation in `TL._pending_iter[name]['base']`. The RHS is an ordinary tensor equation over the non-iteration axes.
- **Inductive step** (`tl.H[i, l+1] = ...`): `l+1` is produced by `RawAxis.__add__(int)`, returning an `IversonBinOp('+', l, 1)`. `__setitem__` detects this and stores the equation in `_pending_iter[name]['recur']`.

No special step is needed on the RHS: a tensor indexed with plain `l` (e.g. `tl.H[i, l]`) is automatically classified as the **running state** if it was declared with `.iteration_axis(l)`, or as a **per-step input** (pre-loaded, full `l` dimension) otherwise. Non-recurrent per-step inputs must have `N` as their last dimension.

### 6.2 The `Scan` term

`Scan` and its companion `ScanAffine` are defined in `data_structure/TensorDSL.py`:

```python
@dataclass(frozen=True)
class ScanAffine:
    A_morphism: object | None    # per-step A factor; None means identity
    b_morphism: object | None    # per-step bias; None means zero
    state_in_axes: tuple         # contracted state axes (non-empty iff matrix recurrence)
    a_positions: tuple[int, ...] # indices into step_xs for A_module inputs
    b_positions: tuple[int, ...] # indices into step_xs for b_module inputs

@dataclass(frozen=True)
class Scan(Term):
    step: object                      # step-body morphism: (H_state, *non_state_inputs) → H_next, l stripped
    base: object                      # base-case morphism: initial-condition inputs → H_0
    N: nm.Numeric                     # nm.Integer; N._value is the step count
    axis: RawAxis                     # recurrence axis l
    affine: ScanAffine | None = None  # affine decomposition for associative_scan, or None
```

`Scan` is a `Term` (frozen dataclass, not a `Morphism` subclass), but it satisfies the ProdCategory axioms and is a valid construction rule in the Br grammar. Its domain is all caller inputs (base-case inputs followed by per-step sequence inputs) and its codomain is the full state history of shape `(*state, N+1)` — scanl semantics.

`Scan` is NOT exported from `data_structure/Category.py` (which would create a circular import: `TensorDSL → Operators → Category → TensorDSL`). It is imported directly from `data_structure.TensorDSL`.

### 6.3 Assembly in `_finalize_iter()`

`TL` stores iterative equations in `_pending_iter` (keyed by tensor name) rather than in `_entries`. `_finalize_iter()` is called lazily by `to_morphism()` and runs the following steps for each iterative tensor:

1. **Coupled recurrence detection**: if two tensors share the same iteration axis uid, `NotImplementedError` is raised (not yet implemented).
2. **Check 4.1**: `l._size` must be `nm.Integer` (concrete size). Enforced at `.iteration_axis()` time; checked again in `_finalize_iter()`.
3. **Check 4.2**: base case literal must be `0` (matches the iteration lower bound).
4. **Check 4.4**: no `l+1` on the RHS (causality violation).
5. **Step body extraction** (`_strip_iter_axis_from_value`): strips `l` from all factor indices; renames the state tensor to a proxy (`H_state`); places the state factor first in the factor list. The stripped RHS is compiled via `_build_step_morph()` using a filtered context `step_ctx = _ctx.without(l.uid)` to prevent `l` from participating in step-body axis unification.
6. **Base morphism**: compiled from the base case equation via `_build_step_morph()` using the main `_ctx`.
7. **Affine recognition** (`_recognize_affine`): if the recurrence is of the form `H[l+1] = A_l·H[l] + b_l`, a `ScanAffine` is constructed and attached to the `Scan` term. See §6.4.
8. **Emit `Scan`**: always emits `Scan` (even pure-state recurrences that could use `Block`; `Block` emission is deferred as an optimisation).

The assembled `Scan` is appended to `_entries` like any other morphism entry.

### 6.4 Affine fast path

`_recognize_affine()` partitions the additive terms of the recurrence by whether they contain the state tensor:

- **State term**: the term containing `H[i, l]`. Must appear exactly once; its `operator` must be `Identity` (no nonlinearity wrapping the state).
- **Bias terms**: all remaining terms.

If the pattern matches, it extracts:

- `A_morphism`: the state term with the state factor removed, built with `lhs_indices = step_out + state_in_axes` so the contracted state axis survives as a free output axis.
- `b_morphism`: the bias terms compiled as a morphism over `step_out`.
- `state_in_axes`: axes of the state factor that are **contracted** (absent from `step_out`). This is filtered against `step_out_uids` to exclude free (tiled) axes that appear in both state and output — distinguishing scalar recurrences (`state_in_axes = ()`) from matrix recurrences (`state_in_axes = (k,)`).
- `a_positions`, `b_positions`: indices into the ordered non-state step inputs that select which tensors feed `A_module` and `b_module` respectively.

The affine property enables the parallel associative scan: the combine law `(A₂, b₂) ∘ (A₁, b₁) = (A₂·A₁, A₂·b₁ + b₂)` composes all N steps in O(log N) time.

### 6.5 Compilation to PyTorch

`ConstructedScan` (in `torch_compile/torch_compile.py`) handles both paths:

**Sequential path** (always correct):

```python
def _run_loop(self, H, step_xs):
    outputs = [H]
    for l_idx in range(self.N):
        sliced = tuple(x[..., l_idx] for x in step_xs)  # l-last convention
        H = step_module(H, *sliced)
        outputs.append(H)
    return torch.stack(outputs, dim=-1)   # (*state, N+1)
```

`_run_loop` is wrapped with `torch._dynamo.disable` to prevent `torch.compile` from unrolling the loop.

**Associative scan path** (when `Scan.affine is not None`):
All N copies of `A_l` and `b_l` are computed in one batched vmap pass (converting l-last to l-first layout), prefix-composed via `torch._higher_order_ops.associative_scan` with `combine_mode='generic'` (required for CPU tensors), then applied to `H0` to produce the full state sequence.

**Input convention**: `forward(*xs)` receives all caller inputs combined. `xs[:n_base]` go to `base_module` (initial-condition inputs); `xs[n_base:]` are the per-step sequence tensors, each with N as the last dimension. `n_base` is derived from `len(base_module._caller_positions)` when the base module has pre-materialised Iverson buffers, otherwise `len(target.base.dom())`.

**Output convention**: always returns shape `(*state, N+1)` — the initial state `H_0` followed by `H_1, …, H_N` stacked along the last dimension (l-last).

### 6.6 Relationship to tensor logic's `*t` notation

Domingos (2025) uses `*t` to annotate a virtual (iterated) index on the LHS of a tensor equation, making the recurrence implicit in the index notation. The pyncd approach differs in two respects:

1. **Declaration on the tensor, not the axis**. `.iteration_axis(l)` marks `H` as iteratively defined; `l` itself remains a plain `RawAxis` usable in non-recurrent equations. This avoids ambiguity when the same axis appears in multiple tensors with different roles.

2. **Explicit `l+1` on the LHS**. The inductive step writes `tl.H[i, l+1] = ...`, making the shifted-index convention visible at the definition site and allowing the compiler to reject malformed equations statically (checks 4.1–4.4).

Both approaches express the same mathematical object — a sequence `(H_0, H_1, …, H_N)` defined by a recurrence — but the pyncd form is more amenable to static validation and compiler analysis before any tensor is allocated.

Coupled recurrences (two tensors both declared with `.iteration_axis(l)`) are detected and raise `NotImplementedError`; the routing and step-body extraction for this case are not yet implemented.

## 7. Summary

Tensor Logic (Domingos 2025) provides a compact notation for individual operator applications in which the index structure is made explicit. `TensorEquation(Operator)` embeds this notation into pyncd's `Term` hierarchy: each equation is a frozen dataclass whose `Axis` index fields carry UID identity, whose `bc_signature()` method produces the corresponding `Broadcasted[B, A, TensorEquation]`, and whose structure remains accessible and traversable throughout the expression's lifetime. `TensorProgram(Term)` collects equations, topologically sorts them, and produces a `ThreadedComposed` morphism via `Context`-mediated axis unification — converting tensor logic's implicit name-sharing into pyncd's explicit UID identity. `ThreadedComposed` extends sequential composition with a live-pool routing table (`routing[i][j]` = live-pool index for input `j` of step `i`) that delivers external tensors to every step that references them, eliminating the need for explicit copy rearrangements at residual connections.

The integration boundary is clean: `TensorProgram.to_morphism()` produces a morphism in `BroadcastedCategory`; above that level, `ProductOfMorphisms`, type-level datatypes, symbolic shape propagation, and `Block` structure are the caller's responsibility — categorical structures that tensor logic deliberately omits.

Beyond this core integration, §5.6 describes the Python DSL layer implemented on top of `TensorProgram` and the remaining gaps. The `TL` registry builds `Broadcasted` morphisms eagerly as equations are assigned, using a single shared `Context` for axis unification across all assignments. Per-entry `input_names` tuples record the domain tensor names; `to_morphism()` uses these to build the live-pool routing table and returns `ThreadedComposed`, falling back to `Composed` only for coupled Scan entries. The `+` operator produces additive expressions (`SumExpr`) compiled to `Composed(ProductOfMorphisms(terms), AdditionOp Broadcasted)`. Tensor declarations (`.tensor()`, `.selection()`, `.predicate()`) record kind and positional shape, promoting axis subtypes (`NatAxis`) and enforcing arity at indexing time. `softmax(expr, where=pred)` and `normalize(expr, where=pred)` produce `MaskedSoftMax` and `MaskedNormalize` operators respectively; `_split_nonlinearity()` routes these via `_compute_mask_alignment()` to the appropriate masked compiled operator. `to_morphism()` includes a norm-axis-invariant additive term pre-pass (`softmax(f+c) = softmax(f)`) and a dead-code-elimination pass before building the routing table. Shape-based axis matching in `composition()` allows morphisms from separate `TL()` instances to compose correctly via `@`.

§6 describes how the DSL extends to iterative (recurrent) tensor equations via `.iteration_axis()`. A `Scan` term — a new construction rule alongside `Block` in the pyncd grammar — represents the recurrence and is compiled to a sequential PyTorch loop with an optional `associative_scan` fast path for affine recurrences. The `TL` registry defers assembly of scan morphisms until `to_morphism()`, where `_finalize_iter()` runs consistency checks and constructs the `Scan` term.

What remains: extracting parallel product structure from the program's dependency DAG; extending the DSL to express embedding lookups as first-class equations (deferred, design recorded separately); realising the full Boolean semiring distinction at the `Datatype` level; and implementing coupled recurrences (multiple tensors sharing the same iteration axis).

---

## References

- Domingos, P. *Tensor Logic: The Language of AI*. 2025. arXiv:2510.12269.
- Wenig, M., Rump, P.G., Blacher, M., Giesen, J. *The Syntax and Semantics of einsum*. 2025. arXiv:2509.20020.

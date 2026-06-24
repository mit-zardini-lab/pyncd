# Integrating CUTE Layouts with pyncd/leanncd and the `D`-Graded Formalism

This note proposes a concrete integration path for CUTE-style layout algebra into the `D`-graded colored PROP framework used by pyncd and leanncd.

The core recommendation is **hybrid integration**:

- keep the current `D`-graded semantics (broadcasting, weave laws, Eq. 3/point naturality, `Scan`/`Route` boundary),
- add a **layout-enriched index layer** (CUTE-style hierarchical layouts and algebra) underneath `St`/reindexing,
- optionally expose this enriched layer in the DSL as explicit layout annotations.

## Contents

1. [References](#1-references)
2. [CUTE layouts: compact background](#2-cute-layouts-compact-background)
3. [Integration with the `D`-graded colored PROP](#3-integration-with-the-d-graded-colored-prop)
   - [3.1 What should stay unchanged](#31-what-should-stay-unchanged)
   - [3.2 What changes in `St`](#32-what-changes-in-st)
   - [3.3 What changes in `Br`](#33-what-changes-in-br)
4. [What this unlocks (and what it does not)](#4-what-this-unlocks-and-what-it-does-not)
5. [Data interchange implications (pyncd/leanncd)](#5-data-interchange-implications-pyncdleanncd)
6. [Impact: simplifications and costs](#6-impact-simplifications-and-costs)
   - [6.1 Python](#61-python)
   - [6.2 Lean](#62-lean)
7. [DSL-level support](#7-dsl-level-support)
   - [7.0 What changes for me as a user?](#70-what-changes-for-me-as-a-user)
   - [7.1 No annotations (current inference style)](#71-no-annotations-current-inference-style)
   - [7.2 Flat annotations equivalent to current behavior](#72-flat-annotations-equivalent-to-current-behavior)
   - [7.3 Hierarchical tiled batched matmul](#73-hierarchical-tiled-batched-matmul)
   - [7.4 Transpose + blocked copy pattern](#74-transpose--blocked-copy-pattern)
   - [7.5 Convolution-like patch extraction (im2col-style)](#75-convolution-like-patch-extraction-im2col-style)
   - [7.6 Attention-style partitioning](#76-attention-style-partitioning)
   - [7.7 Expert routing (static table, not dynamic `Route`)](#77-expert-routing-static-table-not-dynamic-route)
8. [Categorical integration details](#8-categorical-integration-details)
   - [8.1 CUTE categories as index semantics](#81-cute-categories-as-index-semantics)
   - [8.2 Action of `D` on `C` with layout morphisms](#82-action-of-d-on-c-with-layout-morphisms)
   - [8.3 Weaves as internal factorizations](#83-weaves-as-internal-factorizations)
   - [8.4 Relationship to `C ≅ ∫Dat` and Para](#84-relationship-to-c--dat-and-para)
   - [8.5 Scope boundary: what CUTE cannot absorb](#85-scope-boundary-what-cute-cannot-absorb)
9. [Cost-based optimization perspective](#9-cost-based-optimization-perspective)
   - [9.1 Why the database analogy is strong](#91-why-the-database-analogy-is-strong)
   - [9.2 End-to-end walkthrough: Tensor Logic to optimized plan](#92-end-to-end-walkthrough-tensor-logic-to-optimized-plan)
   - [9.3 Optimizer architecture for Tensor Logic + CUTE](#93-optimizer-architecture-for-tensor-logic--cute)
   - [9.4 Cost model ingredients (GPU)](#94-cost-model-ingredients-gpu)
   - [9.5 Feasibility constraints from layout algebra](#95-feasibility-constraints-from-layout-algebra)
   - [9.6 Background material that is directly relevant](#96-background-material-that-is-directly-relevant)
10. [Minimal migration plan](#10-minimal-migration-plan)
11. [Bottom line](#11-bottom-line)

---

## 1. References

### External (CUTE)

- Cecka, **CuTe Layout Representation and Algebra**, arXiv:2603.02298v1  
  <https://arxiv.org/abs/2603.02298>
- Carlisle, Shah, Stern, VanKoughnett, **Categorical Foundations for CuTe Layouts**, arXiv:2601.05972v1  
  <https://arxiv.org/abs/2601.05972>

### Internal (pyncd/leanncd + graded formalism)

- [theory.md](theory.md) (St, Br, weaves, lifts, Eq. 3)
- [graded_prop.md](graded_prop.md) (`D`-graded colored PROP synthesis)
- [scan_route.md](scan_route.md) (`Scan` and `Route` as non-weave generators)
- [future_ideas.md](future_ideas.md) (roadmap and generalized index `D`)
- [tensorLogicNCDIntegration.md](tensorLogicNCDIntegration.md) (DSL/operator integration details)
- [weaves.md](weaves.md) (current `Weave`/`TILED` representation)
- [leanncd.md](leanncd.md) (Lean-facing design and formalization direction)
- [weavesWiresMorphisms.pdf](weavesWiresMorphisms.pdf) (formal framework paper / arXiv:2604.07242)

---

## 2. CUTE layouts: compact background

**Prerequisite: HTuples.** An `HTuple(T)` is defined recursively: either a scalar element of `T`, or a finite ordered list (a `Tuple`) of `HTuple(T)`s. For an HTuple `X`:

- **Rank** `rank(X)`: the number of top-level elements — length of `X` if it is a Tuple, 1 if it is a scalar.
- **Depth** `depth(X)`: 0 for a scalar; `1 + max(depth(X₀), …)` for a Tuple.
- **Size** `|X|` (when entries are positive integers): product of all scalar entries at every level.
- **Congruence** `S ~ D`: S and D have the same recursive structure — both scalars, or both Tuples of the same rank with pairwise congruent elements.

Two coordinate systems exist for a shape `S ∈ HTuple(ℤ⁺)`:

- **Integral coordinate** `c̄ ∈ ℤ_{|S|} = {0, …, |S|-1}`: a single flat integer ranging over all `|S|` elements.
- **Natural coordinate** `c̃ ∈ ℤ_S`: an HTuple *congruent to S*, with each component bounded by the corresponding mode size.

The bijection `idx2crd : ℤ_{|S|} → ℤ_S` converts integral → natural via colexicographic (mixed-radix) enumeration — the *first* component varies fastest; its inverse is `crd2idx`.

*Example.* For shape `S = (2, 3)`, size `|S| = 6`:

```text
Integral c̄  │  Natural c̃ = idx2crd(c̄)
─────────────┼──────────────────────────
      0      │  (0, 0)
      1      │  (1, 0)   ← first component cycles fastest
      2      │  (0, 1)
      3      │  (1, 1)
      4      │  (0, 2)
      5      │  (1, 2)
```

For a hierarchical shape such as `S = (2, (2, 3))`, size `|S| = 12`, natural coordinates have the form `(i, (j, k))` with `i ∈ {0,1}`, `j ∈ {0,1}`, `k ∈ {0,1,2}`; the integral coordinate is still a plain integer in `{0,…,11}`, with `idx2crd` unwrapping it into the nested structure. The layout stride map `D(c̃) = Σᵢ cᵢ dᵢ` is a simple dot product of the natural coordinate against the stride HTuple — it is *not* a simple function of the integral index `c̄` directly.

---

A CUTE **layout** `L = D ∘ S` is a function mapping a coordinate domain `ℤ(S)` to an offset codomain (typically `ℤ`), built from two congruent HTuples:

- **Shape** `S ∈ HTuple(ℤ⁺)`: defines the domain extents. Its size `|S| = ∏ₖ |Sₖ|`. Because S is an HTuple, it supports multi-level indexing: `(M, (N₀, N₁))` has rank 2 (two top-level modes) and admits both a 2D natural coordinate `(m, (n₀, n₁))` and a flat integral coordinate for the N₀·N₁-element combined second mode.
- **Stride** `D ∈ HTuple` with `S ~ D`: defines a linear inner-product map on natural coordinates: `D(c̃) = Σᵢ cᵢ dᵢ = D c̃` (a generalized matrix-vector product).
- **Semi-linearity**: `L(c) = D(S(c)) = d · c̃` is linear in the *natural* coordinates `c̃ ∈ ℤˢ`, but nonlinear in arbitrary coordinates `c ∈ ℤ(S)` because `idx2crd` is not linear. This semi-linearity is the key algebraic property: layouts are essentially generalized affine maps and their operations have algebraic closure.

The **layout algebra** provides operations that produce new CUTE layouts:

- **Concatenation**: `(L₀, …, Lₙ)` with `L(c) = Σᵢ Lᵢ(cᵢ)` — expresses a layout as a tuple of per-mode sublayouts.
- **Coalesce**: collapse a hierarchical layout to a shallower functionally equivalent one by merging adjacent modes with compatible (divisibility) strides.
- **Composition**: `B ∘ A` maps A-domain coordinates through A then B; requires A's cosize to divide into B's domain structure (admissibility).
- **Complement**: `comp(A, N) = A*` is a layout covering the `N/|A|` offsets in the ambient space of size N *not* reached by A.
- **Logical division**: `A ⊘ B` when B divides A; yields C such that `coalesce(C ∘ B) = coalesce(A)`. This is the primary operation for *deriving* tiled layouts from a flat layout and a tile shape.
- **Logical product**: `A ⊗ B = (A, A* ∘ B)` — tiles A over a grid described by B, using A's complement to compute per-tile offset shifts.
- **By-mode composition (tilers)**: `A ⋆ ⟨B, C⟩ = ⟨A₀ ⋆ B, A₁ ⋆ C⟩` — simultaneous per-mode tiling of each logical dimension independently.

Carlisle et al. (arXiv:2601.05972) identify a **tractable** fragment closed under all these operations. Their category **Nest** has objects = nested tuples of positive integers and morphisms `f : S → T` encoded as *diagrams of divisibility arrows* between the mode sizes of `S` and `T`. Non-degenerate tractable layouts correspond bijectively to **Nest**-morphisms of standard form (Theorem A), and every algebra operation has a proved morphism-level counterpart:

- `L_{g ∘ f} = L_g ∘ L_f` for composable f, g (Theorem B),
- `L_{coal(f)} = coal(L_f)` (Theorem C),
- `coal(L_{f^c}) = comp(L_f, N)` for injective f (Theorem D),
- `coal(L_{f ⊘ g}) = coal(L_f ⊘ L_g)` when g divides f (Theorem E),
- `L_{f ⊗ g} = L_f ⊗ L_g` when f, g are product-admissible (Theorem F).

### Tiny examples

1. **Flat row-major matrix layout**

```text
(M, N) : (N, 1)
```

CUTE shape `(M, N)` has rank 2 and size M·N.

Access pattern (logical coordinate → linear offset):

```text
addr(i, j) = i*N + j,   0 <= i < M, 0 <= j < N
```

Index-layout example (as in CUTE notation):

```text
L^{row} = (4, 8) : (8, 1)
```

Grid of offsets:

```text
[  0,  1,  2,  3,  4,  5,  6,  7 ]
[  8,  9, 10, 11, 12, 13, 14, 15 ]
[ 16, 17, 18, 19, 20, 21, 22, 23 ]
[ 24, 25, 26, 27, 28, 29, 30, 31 ]
```

So adjacent `j` entries are contiguous; stepping `i` jumps by `N`.

2. **Flat column-major matrix layout**

```text
(M, N) : (1, M)
```

CUTE shape `(M, N)` has rank 2 and size M·N.

Access pattern (logical coordinate → linear offset):

```text
addr(i, j) = i + j*M,   0 <= i < M, 0 <= j < N
```

Index-layout example:

```text
L^{col} = (4, 8) : (1, 4)
```

Grid of offsets:

```text
[  0,  4,  8, 12, 16, 20, 24, 28 ]
[  1,  5,  9, 13, 17, 21, 25, 29 ]
[  2,  6, 10, 14, 18, 22, 26, 30 ]
[  3,  7, 11, 15, 19, 23, 27, 31 ]
```

So adjacent `i` entries are contiguous; stepping `j` jumps by `M`.

3. **Hierarchical fold of an axis**

```text
(M, (N0, N1)) : (sM, (sN0, sN1))
```

CUTE shape `(M, (N0, N1))` has **rank 2** (two top-level modes) and size M·N0·N1. The second mode is itself an HTuple of depth 2, admitting both the 2D natural coordinate `(n0, n1)` and a flat 1D integral coordinate for the N0·N1 combined elements.

Folded-coordinate access pattern:

```text
addr(i, (n0, n1)) = i*sM + n0*sN0 + n1*sN1
```

Equivalent flattened view (when `j = n0*N1 + n1`):

```text
addr(i, j) = addr(i, (floor(j/N1), j mod N1))
```

Small fold illustration (`2 x (2 x 2)`):

```text
L^{fold,small} = (2, (2, 2)) : (8, (4, 1))

[ 0, 1, 4, 5 ]
[ 8, 9,12,13 ]
```

These are **offsets** (addresses), not logical element labels; this layout is non-compact, so offsets are not restricted to `0..7`.

Same idea at full `4 x 8` scale:

```text
L^{row,fold} = (4, (2, 4)) : (8, (4, 1))

[  0,  1,  2,  3,  4,  5,  6,  7 ]
[  8,  9, 10, 11, 12, 13, 14, 15 ]
[ 16, 17, 18, 19, 20, 21, 22, 23 ]
[ 24, 25, 26, 27, 28, 29, 30, 31 ]
```

For comparison, an interleaved tiled view over the same 32 elements is:

```text
L^{tiled} =
((2,2),(4,2)) : ((4,1),(8,2))

[  0,  2,  8, 10, 16, 18, 24, 26 ]
[  1,  3,  9, 11, 17, 19, 25, 27 ]
[  4,  6, 12, 14, 20, 22, 28, 30 ]
[  5,  7, 13, 15, 21, 23, 29, 31 ]
```

4. **Logical division: deriving a tiled layout**

Logical division `A ⊘ B` yields `C` such that `coalesce(C ∘ B) = coalesce(A)` when B *divides* A (B's extents evenly partition A's modes). Given a flat parent layout and a tile shape, logical division derives the tiled representation automatically — this is the primary CUTE tiling operation.

```text
L^{col}   = (4, 8) : (1, 4)            # column-major 4×8 matrix (Example 2)
T         = (2, 2) : (1, 4)            # 2×2 tile, column-major strides
L^{tiled} = L^{col} ⊘ T               # tiled result
           = ((2, 2), (2, 4)) : ((1, 4), (2, 8))
```

Access pattern: `L^{tiled}((r_in, c_in), (r_tile, c_tile)) = r_in + 4·c_in + 2·r_tile + 8·c_tile`

- First mode `(2,2):(1,4)` = coordinates *within* a tile (identical to T).
- Second mode `(2,4):(2,8)` = tile-grid (2 tile-rows × 4 tile-cols), with strides equal to T's per-mode extents multiplied by L^{col}'s corresponding strides (2×1 = 2 and 2×4 = 8).

Logical *product* `A ⊗ B = (A, A* ∘ B)` is the complementary operation: it tiles A over an independently specified grid layout B, computing shifts via A's complement rather than deriving them from a parent layout.

5. **Logical product (tile over grid)**

```text
A ⊗ B = (A, A* ◦ B)
```

where `A* = comp(A, |A|·|B|)` is the **complement** of A with respect to ambient size `|A|·|B|`. For compact A (image = `{0, …, |A|-1}`), A* has stride `|A|`, so `A*(k) = |A|·k` — it shifts each copy of A by A's own size. The composition `A* ◦ B` applies this shift at each grid coordinate, so tiles do not overlap.

The result `A ⊗ B` has rank rank(A) + rank(B) (concatenation of two sublayouts); for rank-2 A and B the output is rank-4 before optional coalescing.

Access pattern (tile coordinate `a`, grid coordinate `b`):

```text
addr(a, b) = A(a) + A*(B(b))
```

Index-layout example:

```text
A = (3, 4) : (4, 1)
B = (2, 5) : (1, 2)
A ⊗ B = ((3,4),(2,5)) : ((4,1),(12,24))
```

One tile (`b=(0,0)`) is the `A` grid:

```text
[ 0, 1, 2, 3 ]
[ 4, 5, 6, 7 ]
[ 8, 9,10,11 ]
```

Next tile in grid order (`b=(1,0)`) is shifted by `12`:

```text
[12,13,14,15]
[16,17,18,19]
[20,21,22,23]
```

Interpretation: each `b` selects a shifted copy of tile `A`; `A* ◦ B` provides the per-grid shift so tiles do not collide.

---

## 3. Integration with the `D`-graded colored PROP

### 3.1 What should stay unchanged

These are semantic invariants and should be preserved:

- `C` as operation category (`Br`-like) with broadcast/lift semantics,
- `D` as index category controlling reindexing action,
- Eq. 3 / point-naturality and elementality-based weave reasoning,
- distinction between weave fragment vs non-weave generators (`Scan`, `Route`),
- algebra-level interpretation (`construct()` / target actegory semantics).

In short, CUTE should enrich index/layout representation, not replace `D`-graded semantics.

### 3.2 What changes in `St`

Current `St` already gives useful affine canonicalization. CUTE integration extends it from affine reindexings over product axes (without first-class hierarchical layout algebra) to explicit hierarchical layout algebra.

Practical split:

- `St_aff` (**proposed label in this note**, not an existing module/type name): the current affine fragment (`StrideMorphism`-style behavior).
- `St_layout` (**proposed label in this note**, not an existing module/type name): a future layout-enriched fragment with hierarchical shape/stride and CUTE algebra operations.

Then provide:

- embedding/forgetful relation where affine morphisms are a subfragment,
- normalization + admissibility checks for composition/division/product/coalesce.

### Net new capabilities in `St`

- canonicalization over **hierarchical** equivalents (not only flat affine),
- explicit admissibility/divisibility failures as first-class static checks,
- principled modewise tiling/partition derivation,
- reusable inversion/complement/division lemmas for analysis/compilation.

### 3.3 What changes in `Br`

Today each input port effectively stores:

1. weave mask (`TILED` vs local concrete axis),
2. reindexing from degree to selected tiled subset.

With layout enrichment, represent each port by one structured map/factorization, e.g.

```text
λ_i : P × T_i → A_i
```

where:

- `P` = shared degree (broadcast space),
- `T_i` = local/target coordinates for operand `i`,
- `A_i` = full operand coordinates.

This single `λ_i` carries both old pieces (mask + reindexing).

The old view remains derivable:

- tiled dimensions = degree-dependent part of `λ_i`,
- local dimensions = non-degree part (`T_i` factor),
- reindexing = projection/composition of `λ_i` on `P`.

---

## 4. What this unlocks (and what it does not)

### Unlocks

- richer static legality checks for layout transforms,
- cleaner interoperability with hardware-oriented tiling/partitioning logic,
- canonical forms that improve rewrite matching/fusion opportunities,
- more direct bridge between mathematical layout derivation and backend lowering.

### Does **not** unlock by itself

- semantic treatment of recurrence (`Scan`) and value-dependent routing (`Route`),
- replacement of Eq. 3 / weave criterion proofs,
- replacement of `D`-graded algebra semantics.

CUTE layout algebra is an index/layout engine, not a full replacement for operation semantics.

---

## 5. Data interchange implications (pyncd/leanncd)

For acset/CSV/JSON-style interchange, the current `(weave mask, reindexing)` payload can be replaced or augmented by explicit layout objects:

- `layout_nodes`: hierarchical shape/stride expression DAG/tree,
- `layout_edges`: composition/refinement/coalesce operations,
- `port_layout_ref`: map operator ports to layout objects,
- `factorization` metadata identifying degree part (`P`) and local part (`T_i`),
- optional `normal_form_hash` for canonical-equivalence checks.

This is especially useful for Lean/Python roundtrip: one shared layout-normal form with explicit admissibility side conditions.

---

## 6. Impact: simplifications and costs

### 6.1 Python

Likely simplifications:

- cleaner internal representation (single per-port layout factorization instead of mask+reindexing split),
- better rewrite/optimization surface from first-class layout algebra operations,
- stronger static validation via admissibility/divisibility checks in compilation passes,
- more reusable lowering pipeline through shared layout normalization.

Likely costs:

- migration complexity while supporting old/new representations in parallel,
- richer IR objects increase debugging and tooling complexity,
- optimizer/search-space engineering required to realize performance gains,
- compatibility pressure from existing APIs/tests centered on `Weave` + `reindexings`.

### 6.2 Lean

Likely simplifications:

- proofs about index map composition/coalesce/complement can use dedicated layout lemmas,
- fewer ad hoc positional rewrites in morphism-level proofs,
- better proof reuse between backend code and formal model.

Likely costs:

- additional side conditions (admissibility/divisibility/refinement) appear in many statements,
- hierarchical typing obligations can increase local proof overhead.

Overall: **global simplification with localized side-condition burden**. Best practice is to centralize admissibility lemmas in `St_layout` and reuse throughout `Br`.

---

## 7. DSL-level support

The equation language can stay mostly unchanged, with optional layout annotations for explicit control.

### 7.0 What changes for me as a user?

Use layout annotations when:

- you care about concrete tiling/partitioning strategy (e.g., attention/blocking choices),
- you need stable lowering behavior across hardware backends,
- you want compiler-visible intent for layout-sensitive optimization.

Defaults are usually enough when:

- expressions are simple einsum-style contractions and broadcasts,
- no special memory-layout constraints are needed,
- performance is already acceptable without manual layout steering.

Typical new compiler diagnostics in a layout-enriched system:

- **admissibility/divisibility errors** (invalid composition/division under current shape factors),
- **hardware-legality errors** (tile/layout incompatible with target MMA/copy instruction requirements),
- **normalization/coalescing warnings** (annotation is legal but redundant or canonicalizes to an existing form),
- **cost-model hints** (annotation legal but predicted slower than an equivalent candidate).

Mental model: layouts behave like a **refinement (pseudo-)type system** for Tensor Logic. Base TL typing checks algebraic/index correctness; layout typing refines this with admissibility, coordinate-structure, and hardware-legality constraints.

### 7.1 No annotations (current inference style)

```tl
Y[b,i,j] = W[i,k] * X[b,k,j]
```

Compiler infers:

- degree `P=(b,i,j)` from retained indices,
- local contracted index `k`,
- per-input degree projections.

### 7.2 Flat annotations equivalent to current behavior

```tl
layout P     = shape((b,i,j))
layout W_deg = proj(P, (i))
layout X_deg = proj(P, (b,j))
layout Y_deg = id(P)

Y[b,i,j] = W@W_deg[i,k] * X@X_deg[b,k,j]
```

### 7.3 Hierarchical tiled batched matmul

```tl
layout P      = shape((b,(io,ii),(jo,ji)))
layout W_deg  = proj(P, ((io,ii)))
layout X_deg  = proj(P, (b,(jo,ji)))

Y[b, i=(io,ii), j=(jo,ji)] =
  W@W_deg[i,k] * X@X_deg[b,k,j]
```

Here `(io,ii)` and `(jo,ji)` are split/tiled coordinates:

- `i = i_outer * I_inner + i_inner`, with `io = i_outer`, `ii = i_inner`,
- `j = j_outer * J_inner + j_inner`, with `jo = j_outer`, `ji = j_inner`.

So `io`/`jo` index tiles, while `ii`/`ji` index positions inside each tile.

The math is unchanged; annotations make tiling/folding explicit.

### 7.4 Transpose + blocked copy pattern

Context (equivalent plain Tensor Logic):

```tl
# transpose only
Z[j,i] = A[i,j]

# transpose + explicit blocking (i = io*Ti + ii, j = jo*Tj + ji)
Z[jo,ji,io,ii] = A[io,ii,jo,ji]
```

The layout-annotated form below expresses the same semantics, but keeps transpose and blocking as explicit layout morphisms for legality checks and lowering.

```tl
layout Src = shape((i,j))
layout Tr  = compose(Src, permute((j,i)))
layout Tile = tiler((ti,tj))

Z = copy( A@compose(Tr, Tile) )
```

This exposes by-mode composition explicitly for lowering and legality checks.

### 7.5 Convolution-like patch extraction (im2col-style)

Context (equivalent plain Tensor Logic):

```tl
# Conceptually gather a local window around each output location:
Xpatch[b,oh,ow,kh,kw,c] = X[b, oh*sh + kh*dh, ow*sw + kw*dw, c]

# Then contract against kernel:
Y[b,oh,ow,co] = K[kh,kw,c,co] * Xpatch[b,oh,ow,kh,kw,c]
```

The layout-annotated form below makes that gather (`X → Xpatch`) explicit as a layout morphism.

```tl
layout ImgDeg = shape((b,oh,ow,c))
layout Patch  = compose(
  ImgDeg,
  affine_window((kh,kw), stride=(sh,sw), dilation=(dh,dw))
)

Y[b,oh,ow,co] = K[kh,kw,c,co] * X@Patch[b,oh,ow,kh,kw,c]
```

The key benefit is first-class expression of patch layout as an index morphism.

### 7.6 Attention-style partitioning

Context (equivalent plain Tensor Logic):

```tl
# Baseline score contraction:
S[b,h,q,k] = Q[b,h,q,d] * K[b,h,k,d]

# Partitioned form conceptually reindexes q,k into outer/inner tiles:
S[b,h,qo,qi,ko,ki] =
  Q[b,h,qo,qi,d] * K[b,h,ko,ki,d]
```

The layout-annotated form below expresses the same partitioning intent as explicit coordinate transforms.

```tl
layout QP = shape((b,h,q,d))
layout KP = shape((b,h,k,d))
layout ScoreTile = tiler((b,h,qo,ko,qi,ki))

S[b,h,q,k] = Q@compose(QP, ScoreTile)[b,h,q,d] * K@compose(KP, ScoreTile)[b,h,k,d]
```

Partition intent is explicit and can be validated/canonicalized before lowering.

### 7.7 Expert routing (static table, not dynamic `Route`)

Context (equivalent plain Tensor Logic):

```tl
# Static assignment matrix/table A[token,expert] is known at compile time.
# A one-hot case is a selector; weighted case is weighted combination.
Y[token,hidden] =
  A[token,expert] * E[expert,hidden] * X[token,hidden]   # sum over expert
```

Because `A` is compile-time/static here, this remains in the weave-compatible/static-reindexing fragment.

```tl
layout P = shape((token, expert))
layout StaticAssign = table_layout(assign_table)  # compile-time constant

Y[token,hidden] = E@StaticAssign[token,expert,hidden] * X[token,hidden]
```

`expert` is the contracted index (sum over experts). Static assignment is weave-compatible; dynamic assignment still belongs to `Route`.

---

## 8. Categorical integration details

### 8.1 CUTE categories as index semantics

From the `D`-graded perspective, the clean move is to treat CUTE’s **tractable layout fragment** — the **Nest**-morphism encoding of Carlisle et al. — as a richer index category:

```text
D_aff  ⊆  D_layout
```

where:

- `D_aff` is the current affine-reindexing fragment (`St_aff`), corresponding to flat-stride `StrideMorphism`-style morphisms;
- `D_layout` is the full tractable layout fragment. By Theorem A of Carlisle et al., non-degenerate tractable layouts are in bijection with **Nest**-morphisms of standard form, so `D_layout` inherits the complete CUTE algebra — composition, coalesce, complement, logical division, logical product — with admissibility conditions appearing as morphism-level structural constraints (Theorems B–F). A **Nest**-morphism `f : S → T` is concretely a diagram of divisibility arrows between the mode sizes of S and T; this makes admissibility structurally explicit rather than a run-time precondition.

Existing pyncd semantics is recovered by restricting to `D_aff ⊆ D_layout`.

From a typing viewpoint, this is a *refinement* of index types: `D_layout` carries stricter, constraint-rich layout judgments over the same underlying tensor equations.

### 8.2 Action of `D` on `C` with layout morphisms

The graded framework uses a right action:

```text
act : C × D^op → C
```

If `D = D_layout`, the Carlisle et al. Theorems B–F (morphism-level operations commute with realized layout operations) become reusable coherence facts for this action:

- Theorem B (composition) supports functoriality of reindexing composition,
- Theorem C (coalesce) supports normalization lemmas,
- Theorems D–F (complement/division/product) support structured factorization and rewrite lemmas.

So the `D`-action laws do not change in form; they gain a stronger library of admissible rewrites.

### 8.3 Weaves as internal factorizations

**Current representation** for each input port `i`:

1. **Weave mask** `Weave._shape = (w₁, …, wₙ)` — each `wⱼ` is either a concrete `Axis` (local to the base op) or `WeaveMode.TILED` (filled from the broadcast degree).
2. **Reindexing** `η_i : P → Q_i` — maps the shared degree P to the *subset* of degree axes that input `i` actually uses (`Q_i` = tiling shape, one axis per TILED slot in the weave).

These two pieces together specify where each coordinate of operand `i` comes from at every point in the broadcast loop.

**Translation to a single CUTE layout** `λ_i` over the combined domain `P × T_i`:

Let `L_i` be the base CUTE layout of the raw tensor for input `i` (encoding its memory strides). The combined-domain layout `λ_i` is assembled axis by axis:

- For each degree axis `p_k ∈ P`:
  - If `η_i` routes `p_k` to TILED slot `j` of the weave: **stride of `p_k` in `λ_i` = stride of position `j` in `L_i`** (the memory stride at that slot).
  - If `η_i` does *not* use `p_k` for this input: **stride = 0** (CUTE's broadcast convention — the tensor value does not change as `p_k` varies).
- For each local axis `t_l ∈ T_i`: **stride of `t_l` in `λ_i` = stride of the corresponding concrete-axis position in `L_i`**.

The shape of `λ_i` is the concatenation `(P_shape, T_i_shape)`.

**Worked example: matrix multiply `Y[i,j] = W[i,k] * X[k,j]`**

Degree `P = (i, j)`. Concrete shapes (row-major): `W` has layout `L_W = (I,K):(K,1)` and `X` has layout `L_X = (K,J):(J,1)`.

| Field | `W` | `X` |
| --- | --- | --- |
| Weave `_shape` | `(TILED, k)` | `(k, TILED)` |
| Reindexing `η` | `(i,j) → (i,)` | `(i,j) → (j,)` |
| Local axes `T_i` | `(k)` | `(k)` |

Applying the rule to build `λ_W` over domain `(I, J, K)`:

- `i` → maps to TILED slot 0 → stride = K (row stride of W)
- `j` → not used by W → stride = **0** (broadcast)
- `k` (local) → concrete slot 1 in weave → stride = 1 (column stride of W)

```text
λ_W = (I, J, K) : (K, 0, 1)
```

Applying the rule to build `λ_X` over the same domain `(I, J, K)`:

- `i` → not used by X → stride = **0** (broadcast)
- `j` → maps to TILED slot 1 → stride = 1 (column stride of X)
- `k` (local) → concrete slot 0 in weave → stride = J (row stride of X)

```text
λ_X = (I, J, K) : (0, 1, J)
```

**Recovering the old view.** The mask is the sign pattern of strides: TILED slots have non-zero stride (they vary with degree), concrete slots have their own stride, and zero strides identify projected-out degree axes. The reindexing `η_i` is recovered as the restriction of `λ_i` to the non-zero-stride degree axes. The factorization thus contains strictly more information (explicit strides, CUTE admissibility conditions) while making the mask+reindexing derivable rather than stored separately.

### 8.4 Relationship to `C ≅ ∫Dat` and Para

This integration does **not** require changing the structure/data split:

```text
C ≅ ∫Dat
```

still holds. What changes is the structure-side index morphism language (richer `D`). Data payloads (sizes/params/dtypes) remain in `Dat`.

Likewise, Para semantics (`construct()` as parameter-carrying algebra) remains the same at the architectural level; CUTE enrichment improves legality/canonicalization of index transformations before lowering.

### 8.5 Scope boundary: what CUTE cannot absorb

CUTE layout algebra strengthens the weave-compatible fragment, but does not absorb semantic operators whose obstruction is not layout expressivity:

- `Scan`: temporal coupling / failure of pointwise independence,
- `Route`: value-dependent runtime indexing.

So CUTE belongs in the index layer (`D`), while the non-weave generator boundary from `scan_route.md` remains essential.

---

## 9. Cost-based optimization perspective

### 9.1 Why the database analogy is strong

There is a direct analogy between Tensor Logic + CUTE and classical query optimization:

- **logical plan**: tensor equation + layout expression (including tiling/partitioning),
- **equivalence rules**: layout algebra rewrites (compose/coalesce/refine/divide/product under admissibility),
- **physical plan**: specific kernel schedule + memory layout + thread mapping,
- **optimizer objective**: minimize predicted runtime subject to correctness and hardware constraints.

This is especially natural once layout transforms are first-class morphisms rather than implicit codegen heuristics.

### 9.2 End-to-end walkthrough: Tensor Logic to optimized plan

Start from a plain Tensor Logic score contraction:

```tl
S[b,h,q,k] = Q[b,h,q,d] * K[b,h,k,d]
```

`d` is contracted (summed) because it appears on the RHS but not on the LHS.

Lift to explicit layout intent:

```tl
layout QP = shape((b,h,q,d))
layout KP = shape((b,h,k,d))
layout Tile = tiler((b,h,qo,ko,qi,ki))

S[b,h,q,k] =
  Q@compose(QP, Tile)[b,h,q,d] * K@compose(KP, Tile)[b,h,k,d]
```

Semantics are unchanged; only partitioning/layout intent is made explicit.

Generate equivalent candidate plans:

1. **Plan A (baseline contiguous)**: no explicit tiling.

```tl
layout QA = shape((b,h,q,d))
layout KA = shape((b,h,k,d))
layout SA = shape((b,h,q,k))
```

2. **Plan B (blocked tiling)**: block tiles `(Tq, Tk)` with shared-memory staging.

```tl
# q = qo*Tq + qi,  k = ko*Tk + ki
layout QB = compose(shape((b,h,q,d)), tile_q((qo,qi), Tq))
layout KB = compose(shape((b,h,k,d)), tile_k((ko,ki), Tk))
layout SB = shape((b,h,qo,ko,qi,ki))
```

3. **Plan C (blocked + interleaved tile)**: same contraction as B, but with an in-tile swizzle/permutation.

```tl
layout QC = compose(QB, swizzle_qi_d)   # e.g. interleave qi with d lanes
layout KC = compose(KB, swizzle_ki_d)   # e.g. interleave ki with d lanes
layout SC = SB                          # same logical score shape as B
```

So A/B/C differ only in layout/schedule choice (contiguous vs blocked vs blocked+swizzled), not in tensor algebra semantics.

Prune by legality:

- layout admissibility/divisibility,
- instruction-shape constraints,
- memory footprint limits.

Score remaining candidates (illustrative numbers):

| Plan | HBM bytes (est.) | Tensor-core eligible | Occupancy (est.) | Penalties | Score (lower better) |
| --- | ---: | --- | ---: | --- | ---: |
| A | 1.00x | Partial | High | high HBM traffic | 1.00 |
| B | 0.62x | Yes | Medium | sync overhead | 0.73 |
| C | 0.58x | Yes | Medium | lower coalescing penalty | 0.68 |

Pick best physical plan and emit kernel:

- chosen plan: **C**,
- emitted program preserves TL semantics by equivalence rewrites + legality checks,
- performance gain comes from lower predicted cost on the same mathematical computation.

This is exactly query-optimizer structure: equivalence-class search plus constrained cost minimization.

### 9.3 Optimizer architecture for Tensor Logic + CUTE

A practical architecture mirrors Cascades/Volcano-style systems:

1. **Normalize** expression into a canonical IR (`Br` morphism + explicit layout morphisms).
2. **Enumerate equivalent candidates** using legal rewrite rules (memoized e-classes or memo groups).
3. **Attach physical properties** to candidates (tile shapes, memory space, vector width, warp mapping).
4. **Prune by feasibility** (instruction constraints, divisibility/admissibility, memory limits).
5. **Score candidates** with a GPU cost model (analytic and/or learned).
6. **Emit best plan** and optionally keep top-K alternatives for autotuning.

In this setup, CUTE algebra gives the legal search space; the cost model selects among legal options.

### 9.4 Cost model ingredients (GPU)

A useful first-pass cost model should combine:

- **memory traffic terms**: bytes moved across HBM/shared/register levels,
- **compute terms**: FLOPs and tensor-core eligibility/utilization,
- **parallelism terms**: occupancy, warp efficiency, launch geometry,
- **synchronization terms**: barriers, staging overhead, pipeline bubbles,
- **access quality terms**: coalescing, bank conflicts, stride/pathological patterns.

A roofline-style bound (compute- vs memory-bound) is a good backbone, refined by architecture-specific penalties.

### 9.5 Feasibility constraints from layout algebra

Cost alone is insufficient; many candidates are invalid. CUTE-style admissibility conditions become hard constraints:

- divisibility requirements for composition/division,
- compatibility/refinement conditions for folded/blocked transforms,
- instruction-shape legality (e.g., MMA operand layout requirements),
- static memory footprint constraints for staging tiles.

So optimization is naturally:

```text
argmin_plan Cost(plan)
subject to  LayoutAdmissible(plan) and HardwareLegal(plan)
```

### 9.6 Background material that is directly relevant

The most relevant prior threads are:

- **query optimization**: Selinger-style cost-based planning; Volcano/Cascades memo rewrite frameworks,
- **tensor contraction optimization**: einsum contraction-path search (`opt_einsum`) as a direct analogue,
- **GPU performance modeling**: roofline, occupancy analysis, memory-hierarchy models,
- **schedule/autotune compilers**: Halide/TVM/Triton/MLIR Linalg-style logical-to-physical lowering,
- **IO-aware deep learning kernels**: FlashAttention-style traffic minimization as proof that layout/schedule rewrites + cost reasoning produce large wins.

In this document’s context: CUTE supplies the rewrite algebra; pyncd/leanncd `D`-graded semantics supply correctness; a GPU cost model closes the loop.

---

## 10. Minimal migration plan

1. **Add layout-enriched `St_layout` alongside existing `St_aff`**.
2. **Teach `Broadcasted` construction to accept explicit per-port layout maps** while retaining current weave/reindexing API.
3. **Add derivation adapters both ways**:
   - old `(weave,reindexing) → λ_i`,
   - new `λ_i → (weave,reindexing)` for compatibility.
4. **Expose optional DSL annotations** (`layout`, `compose`, `proj`, `tiler`, etc.) with current inference as default.
5. **Incrementally prove Lean lemmas**:
   - layout composition compatibility,
   - coalesce/refine normalization,
   - admissibility propagation.

This yields immediate practical value without breaking existing models or proofs.

---

## 11. Bottom line

CUTE is best integrated as a **layout algebra substrate for the index layer**, not as a replacement for the `D`-graded semantics.

- `St` gains expressive hierarchical layout morphisms and stronger static reasoning.
- `Br` keeps its semantics but can replace mask+reindexing internals with a cleaner per-port layout factorization.
- DSL remains familiar while gaining explicit, checkable layout intent where needed.

This is a conservative extension with high upside for compilation, verification, and Lean/Python alignment.

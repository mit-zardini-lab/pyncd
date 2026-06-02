# Separating Structure from Data in St and Br via Acsets

## Reference

Patterson, Lynch, Fairbanks (2022). *Categorical Data Structures for Technical Computing*. Compositionality 4(5). [arXiv:2106.04703](https://arxiv.org/abs/2106.04703)

Their central construction is the **acset**: a schema category $\mathcal{S}$ separates combinatorial structure (a C-set / copresheaf) from typed data attributes, with data migration as adjoint triples along schema morphisms.

---

## Contents

1. [Reference](#reference)
2. [Motivation](#motivation)
3. [St](#st)
   - [The Schema $\mathcal{S}_{St}$](#the-schema-mathcals_st)
4. [Br](#br)
   - [The Schema $\mathcal{S}_{Br}$](#the-schema-mathcals_br)
5. [The Acset Framework](#the-acset-framework)
   - [Two-Level Structure/Data Split](#two-level-structuredata-split)
   - [The Grothendieck Construction](#the-grothendieck-construction)
   - [Schema Morphisms and Data Migration](#schema-morphisms-and-data-migration)
6. [Instances as Functors: $\Phi_a$ at the Instance Category Level](#instances-as-functors-phi_a-at-the-instance-category-level)
   - [Schema Morphism Underlying $[a, \cdot]$](#schema-morphism-underlying-a-cdot)
   - [$\Phi_a$ as a Functor](#phi_a-as-a-functor)
   - [Connection to Contravariant Functoriality](#connection-to-contravariant-functoriality)
7. [Integration with Tensor Logic](#integration-with-tensor-logic)
   - [Tensor Logic as the Construction Interface](#tensor-logic-as-the-construction-interface)
   - [TensorEquation as a Proto-Acset](#tensorequation-as-a-proto-acset)
   - [The operator\_tag Attribute](#the-operator_tag-attribute)
   - [The Dual-View Pipeline](#the-dual-view-pipeline)
   - [What Remains in the Term World](#what-remains-in-the-term-world)
8. [Advantages of the acset Structure/Data Decomposition](#advantages-of-the-acset-structuredata-decomposition)
   - [Practical and Implementation Advantages](#practical-and-implementation-advantages)
   - [Theoretical Advantages](#theoretical-advantages)
9. [From SBrInstance to a Diagram in Br: A Lean Encoding](#from-sbrinstance-to-a-diagram-in-br-a-lean-encoding)
   - [Setting](#setting)
   - [Two Lean library shortcuts](#two-lean-library-shortcuts)
   - [The Index Category J](#the-index-category-j)
   - [The Denotation Functor D](#the-denotation-functor-d)
   - [Populating BrBase from Instance Tables](#populating-brbase-from-instance-tables)
   - [Proof Obligations](#proof-obligations)

---

## Motivation

The **acset pattern** separates combinatorial *structure* — which entities exist and how they connect — from typed *data* — values assigned freely to those entities. The same principle applies to both **St** and **Br**.

| | Structure | Data |
| --- | --- | --- |
| **St** morphism | support graph: which domain axes feed which codomain axes | axis sizes; transformation coefficients |
| **Br** morphism | arrays and their axes; reindexing support graph | axis sizes; reindexing coefficients; weave roles (`is_target`); dimension positions (`position`); input/output flags (`is_input`); array datatypes (`datatype_tag`); datatype bounds (`max_value`); operator tags (`operator_tag`); operator parameters (`bias`, `elementwise_fn`) |

In both cases the connectivity is fixed at graph-compile time — which axes exist, which inputs feed which outputs, which axes are looped over. The numeric and Boolean values vary across instances. The acset pattern encodes exactly this split, and exposes the data layer as queryable and migratable via adjoint triples along schema morphisms.

---

## St

### The Schema $\mathcal{S}_{St}$

$\mathcal{S}_{St}$ models a stride morphism as a weighted directed multigraph on axis nodes: each nonzero entry of the coefficient matrix becomes an edge.

$$\texttt{Entry} \underset{\texttt{tgt}}{\overset{\texttt{src}}{\rightrightarrows}} \texttt{Axis} \xrightarrow{\texttt{size}} \mathbb{N}_{>0} \qquad \texttt{Entry} \xrightarrow{\texttt{coeff}} \mathbb{N}$$

Two entity types (`Axis`, `Entry`), two attribute types ($\mathbb{N}_{>0}$, $\mathbb{N}$), four maps. A single schema covers both St objects and St morphisms.

**Attributes (data):**

| Attribute | Type | What it captures |
| --- | --- | --- |
| `size` | `Axis → ℕ_{>0}` | axis size |
| `coeff` | `Entry → ℕ` | transformation coefficient |

The **C-set part** (structure) = the directed multigraph on `Axis` with edges `Entry` — the support graph of the stride morphism, with both size and coefficient values erased.

**St objects** are instances with `Entry` = ∅ — a set of axes with sizes and no morphism data. The 28×28 RGB image:

| `Axis` | `size` |
| --- | --- |
| $a_0$ | 28 |
| $a_1$ | 28 |
| $a_2$ | 3 |

**St morphisms** are instances where `Entry` ≠ ∅. For $\Lambda = \begin{pmatrix} 2 & 0 \\ 1 & 3 \end{pmatrix}$ with domain sizes $(4, 6)$ and codomain sizes $(5, 7)$:

| `Axis` | `size` |
| --- | --- |
| $a_0$ | 4 |
| $a_1$ | 6 |
| $a_2$ | 5 |
| $a_3$ | 7 |

| `Entry` | `src` | `tgt` | `coeff` |
| --- | --- | --- | --- |
| $e_0$ | $a_0$ | $a_2$ | 2 |
| $e_1$ | $a_1$ | $a_2$ | 1 |
| $e_2$ | $a_1$ | $a_3$ | 3 |

The zero entry $\Lambda_{01} = 0$ is absent — support is sparse by default. Domain = image of `src` = $\{a_0, a_1\}$; codomain = image of `tgt` = $\{a_2, a_3\}$. Since both domain and codomain axes live in the same `Axis` table, sizes are readable from any `Entry` by following `src` or `tgt`.

---

## Br

### The Schema $\mathcal{S}_{Br}$

$\mathcal{S}_{Br}$ shares the `Axis`-centred reindexing structure of $\mathcal{S}_{St}$ but replaces `Entry` with `Sample` and adds two new entity types — `Array` and `ArrayAxis` — to carry the full reindexing structure of a `Broadcasted` morphism.

In $\mathcal{S}_{St}$, the analogous entity type is called `Entry` because each row is a nonzero entry of a coefficient matrix. In $\mathcal{S}_{Br}$, the same structure plays a different role: each row specifies one axis-component of how a particular input array is **sampled** at each step of the degree loop. The triple (`src`, `tgt`, `coeff`) says "for array $X$, the index into axis `tgt` equals `coeff` times the degree index along axis `src`" — it is a sampling rule, not a matrix entry in the abstract algebraic sense. The name `Sample` reflects this: each row is one component of the sampling pattern that determines which slice of an input the base operation sees at each loop iteration.

$$\texttt{Equation} \xleftarrow{\texttt{equation}} \texttt{Array} \xrightarrow{\texttt{norm\_axis}} \texttt{Axis}$$

$$\texttt{Sample} \underset{\texttt{tgt}}{\overset{\texttt{src}}{\rightrightarrows}} \texttt{Axis} \xleftarrow{\texttt{axis}} \texttt{ArrayAxis} \xrightarrow{\texttt{array\_slot}} \texttt{Array}$$

$$\texttt{Sample} \xrightarrow{\texttt{reindexing\_slot}} \texttt{Array}$$

$$\texttt{Equation} \xrightarrow{\texttt{lhs\_name}} \texttt{String} \qquad \texttt{Sample} \xrightarrow{\texttt{coeff}} \mathbb{N} \qquad \texttt{Axis} \xrightarrow{\texttt{size}} \mathbb{N}_{>0} \qquad \texttt{ArrayAxis} \xrightarrow{\texttt{is\_target}} \mathbb{B} \qquad \texttt{ArrayAxis} \xrightarrow{\texttt{position}} \mathbb{N}$$

$$\texttt{Array} \xrightarrow{\texttt{is\_input}} \mathbb{B} \qquad \texttt{Array} \xrightarrow{\texttt{datatype\_tag}} \texttt{DataTag} \qquad \texttt{Array} \xrightarrow{\texttt{max\_value}} \mathbb{N} \qquad \texttt{Array} \xrightarrow{\texttt{operator\_tag}} \texttt{OpTag}$$

$$\texttt{Array} \xrightarrow{\texttt{bias}} \mathbb{B} \qquad \texttt{Array} \xrightarrow{\texttt{elementwise\_fn}} \texttt{String} \qquad \texttt{Array} \xrightarrow{\texttt{slot}} \mathbb{N} \qquad \texttt{Array} \xrightarrow{\texttt{name}} \texttt{String}$$

Five entity types, six attribute types ($\mathbb{N}$, $\mathbb{N}_{>0}$, $\mathbb{B}$, $\texttt{OpTag}$, $\texttt{DataTag}$, $\texttt{String}$), twenty maps. `max_value` is a partial map defined only for `Natural`-typed arrays. `operator_tag`, `norm_axis`, `bias`, and `elementwise_fn` are partial maps defined only for output arrays (`is_input = False`); `norm_axis` is further restricted to operators in $\{\texttt{SoftMax}, \texttt{MaskedSoftMax}, \texttt{Normalize}, \texttt{MaskedNormalize}\}$; `bias` is restricted to `Linear` output arrays; `elementwise_fn` is restricted to `Elementwise` output arrays. `lhs_name` is partial — may be `None` for anonymous equations.

`norm_axis` is necessary even though the source representation (a `TensorEquation`) marks the normalisation axis via the `NormAxis` subtype of `RawAxis` in `lhs_indices`. Once converted to an instance, all axes in `lhs_indices` appear as `ArrayAxis` rows with `is_target = False` — degree axes and the norm axis are indistinguishable without the explicit pointer. `norm_axis` is what preserves this distinction in the standalone instance.

**Attributes (data):**

| Attribute | Type | What it captures |
| --- | --- | --- |
| `size` | `Axis → ℕ_{>0}` | axis size |
| `coeff` | `Sample → ℕ` | reindexing coefficient |
| `is_target` | `ArrayAxis → Bool` | For input arrays: True = axis consumed by the base operation (contracted for sum operations, normalized for SoftMax/MaskedSoftMax/Normalize); False = degree axis supplied by the outer loop. For output arrays: always False — every axis in `lhs_indices` is a degree axis produced by the outer loop; the normalization dimension is identified separately by `norm_axis` on `Array` |
| `position` | `ArrayAxis → ℕ` | 0-indexed dimension position within the array's physical layout; encodes the axis interleaving of `Weave._shape` |
| `is_input` | `Array → Bool` | input array (True) or output array (False) |
| `datatype_tag` | `Array → DataTag` | total map; `REALS` for floating-point arrays, `NATURAL` for discrete vocabulary arrays, `BOOL` for Boolean predicate arrays whose output is thresholded by $H$ |
| `max_value` | `Array → ℕ` | vocabulary size for `NATURAL` arrays; partial — undefined for `REALS` and `BOOL` arrays |
| `operator_tag` | `Array → OpTag` | base operation type; partial — defined only for output arrays (`is_input = False`) |
| `bias` | `Array → Bool` | whether `Linear` applies a bias term; partial — defined only for `Linear` output arrays |
| `elementwise_fn` | `Array → String` | name of the pointwise function; partial — defined only for `Elementwise` output arrays |
| `slot` | `Array → ℕ` | 0-indexed argument position within an equation: 0 for the output array, 1..N for input arrays in rhs order; primary key for `Array` within its equation, unambiguous under self-joins |
| `name` | `Array → String` | tensor name; metadata carried alongside `slot` for display and traceability; partial — may be `None` for anonymous arrays |
| `lhs_name` | `Equation → String` | output tensor name for the equation; metadata only; partial — may be `None` for anonymous equations |

The **C-set part** (structure) consists of:

- the `equation` map assigning each `Array` to its containing `Equation`
- the reindexing multigraph on `Axis` via `Sample ⇉ Axis` (analogous to `Entry ⇉ Axis` in $\mathcal{S}_{St}$)
- the bipartite graph linking each `ArrayAxis` to its `Array` and its `Axis` via `array_slot` and `axis`
- the `reindexing_slot` map assigning each `Sample` to the array whose reindexing it belongs to
- the `norm_axis` map (partial) pointing each SoftMax/MaskedSoftMax/Normalize/MaskedNormalize output array to its normalisation axis

The **degree** of a `Broadcasted` morphism is the tuple of loop axes — the axes iterated in the outer degree loop, shared across all inputs. In tabular form the degree is the image of `src` across all `Sample` rows.

**Example instance** — batched attention scores $Y[b,j] = \mathrm{SoftMax}_j\!\left(\sum_i X[b,i]\,W[b,i,j]\right)$, degree $P = (b, j)$, reindexing $\eta_X = \mathrm{id}_{(b)}$ ($X$ is sliced along $b$ only) and $\eta_W = \mathrm{id}_{(b,j)}$ ($W$ is sliced along both $b$ and $j$); the base operation is a SoftMax along $j$, which is contracted by the SoftMax (not by the outer loop) and identified by the `norm_axis` FK on the output array:

| `Axis` | `size` |
| --- | --- |
| $a_b$ | 32 |
| $a_i$ | 64 |
| $a_j$ | 128 |

| `Equation` | `equation_idx` | `lhs_name` |
| --- | --- | --- |
| $e_0$ | 0 | $Y$ |

| `Array` | `equation_idx` | `slot` | `is_input` | `operator_tag` | `norm_axis` | `datatype_tag` | `max_value` | `bias` | `elementwise_fn` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $Y$ | 0 | 0 | False | SOFTMAX | $a_j$ | REALS | — | — | — |
| $X$ | 0 | 1 | True | — | — | REALS | — | — | — |
| $W$ | 0 | 2 | True | — | — | REALS | — | — | — |

| `ArrayAxis` | `equation_idx` | `array_slot` | `axis` | `is_target` | `position` |
| --- | --- | --- | --- | --- | --- |
| $xa_b$ | 0 | 1 | $a_b$ | False | 0 |
| $xa_i$ | 0 | 1 | $a_i$ | True | 1 |
| $wa_b$ | 0 | 2 | $a_b$ | False | 0 |
| $wa_i$ | 0 | 2 | $a_i$ | True | 1 |
| $wa_j$ | 0 | 2 | $a_j$ | False | 2 |
| $ya_b$ | 0 | 0 | $a_b$ | False | 0 |
| $ya_j$ | 0 | 0 | $a_j$ | False | 1 |

| `Sample` | `equation_idx` | `src` | `tgt` | `coeff` | `reindexing_slot` |
| --- | --- | --- | --- | --- | --- |
| $s_0$ | 0 | $a_b$ | $a_b$ | 1 | 1 |
| $s_1$ | 0 | $a_b$ | $a_b$ | 1 | 2 |
| $s_2$ | 0 | $a_j$ | $a_j$ | 1 | 2 |

$X$ has one retained axis ($b$) → one Sample. $W$ has two retained axes ($b$ and $j$) → two Samples, both with `reindexing_slot = 2`. $s_1$ and $s_2$ share `reindexing_slot` but differ in `src`/`tgt` — together they express that at each degree step $(b, j)$, $W$ is indexed as $W[b, \cdot, j]$ with the remaining axis $i$ left to the base operation. The degree is $P = (b, j)$, confirmed by the image of `src` across all three rows. Tiling axes of each array are the `ArrayAxis` rows where `is_target = False`; their reindexing coordinates are supplied by the `Sample` rows with matching `reindexing_slot` via `tgt`. Note that $j$ is `is_target = False` on both $W$ and $Y$: it is a degree axis iterated by the outer loop; its role as the SoftMax normalization dimension is recorded separately by `norm_axis = a_j` on the $Y$ row in the `Array` table.

**Example instance** — stride-2 convolution $Y[p] = \sum_k W[k]\, X[2p + k]$, degree $P = (p)$, reindexing $\eta_X$ maps $p$ to $X$'s physical axis $n$ with coefficient 2; $W$ is not reindexed by the degree ($W[k]$ is the same slice at every step $p$):

| `Axis` | `size` |
| --- | --- |
| $p$ | 5 |
| $k$ | 3 |
| $n$ | 11 |

| `Equation` | `equation_idx` | `lhs_name` |
| --- | --- | --- |
| $e_0$ | 0 | $Y$ |

| `Array` | `equation_idx` | `slot` | `is_input` | `operator_tag` | `norm_axis` | `datatype_tag` | `max_value` | `bias` | `elementwise_fn` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $Y$ | 0 | 0 | False | IDENTITY | — | REALS | — | — | — |
| $W$ | 0 | 1 | True | — | — | REALS | — | — | — |
| $X$ | 0 | 2 | True | — | — | REALS | — | — | — |

| `ArrayAxis` | `equation_idx` | `array_slot` | `axis` | `is_target` | `position` |
| --- | --- | --- | --- | --- | --- |
| $ya_p$ | 0 | 0 | $p$ | False | 0 |
| $wa_k$ | 0 | 1 | $k$ | True | 0 |
| $xa_n$ | 0 | 2 | $n$ | True | 0 |

| `Sample` | `equation_idx` | `src` | `tgt` | `coeff` | `reindexing_slot` |
| --- | --- | --- | --- | --- | --- |
| $s_0$ | 0 | $p$ | $n$ | 2 | 2 |

$W$ has no `Sample` row — it does not depend on the degree. $X$ contributes one `Sample` with `coeff` $= 2$: at degree step $p$, $X$'s physical axis $n$ starts at $2p$. The contracted axis $k$ then adds positions $0, 1, 2$ within that slice, so the full access is $n = 2p + k$.

Three structural differences from the batched attention example stand out. First, `src` $\neq$ `tgt`: the degree axis $p$ and $X$'s physical axis $n$ are distinct — the reindexing is not a self-map on a shared axis. Second, `coeff` $= 2 \neq 1$: the stride. Third, $X$'s $n$-axis is `is_target = True` whereas in the batched attention example $X$'s $b$-axis is `is_target = False`: $b$ is shared directly between the degree and the input, so it is tiled and leaves `is_target = False`; $n$ is a target axis because the contracted $k$ also contributes to it ($n = 2p + k$ is not determined by $p$ alone).

**Example instance** — self-join (Gram matrix) $Y[i,j] = \sum_k H[i,k]\, H[k,j]$, where $H$ is a sequence of embeddings and $Y$ records their pairwise inner products. Degree $P = (i, j)$; both $i$ and $j$ are retained. The contracted axis $k$ is summed over. The same tensor $H$ appears **twice** in the rhs, once indexing row $i$ and once indexing row $j$.

| `Axis` | `size` |
| --- | --- |
| $a_i$ | 8 |
| $a_j$ | 8 |
| $a_k$ | 64 |

The two occurrences of $H$ in the rhs are assigned distinct slots (1 and 2). `name` is metadata; `(equation_idx, slot)` is the entity identifier.

| `Equation` | `equation_idx` | `lhs_name` |
| --- | --- | --- |
| $e_0$ | 0 | $Y$ |

| `Array` | `equation_idx` | `slot` | `is_input` | `operator_tag` | `norm_axis` | `datatype_tag` | `max_value` | `bias` | `elementwise_fn` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $Y$ | 0 | 0 | False | IDENTITY | — | REALS | — | — | — |
| $H$ | 0 | 1 | True | — | — | REALS | — | — | — |
| $H$ | 0 | 2 | True | — | — | REALS | — | — | — |

Two rows in the Array table share `name = H` with distinct `slot` values; `ArrayAxis` and `Sample` rows reference these by `(equation_idx, slot)`.

| `ArrayAxis` | `equation_idx` | `array_slot` | `axis` | `is_target` | `position` |
| --- | --- | --- | --- | --- | --- |
| $ya_i$ | 0 | 0 | $a_i$ | False | 0 |
| $ya_j$ | 0 | 0 | $a_j$ | False | 1 |
| $h1a_i$ | 0 | 1 | $a_i$ | False | 0 |
| $h1a_k$ | 0 | 1 | $a_k$ | True | 1 |
| $h2a_j$ | 0 | 2 | $a_j$ | False | 0 |
| $h2a_k$ | 0 | 2 | $a_k$ | True | 1 |

The first $H$ reference (slot 1) is indexed by degree axis $a_i$ (position 0) and contracted axis $a_k$ (position 1). The second $H$ reference (slot 2) is indexed by degree axis $a_j$ (position 0) and the same contracted axis $a_k$ (position 1). Crucially, $a_i$ and $a_j$ have distinct UIDs — they are different axis objects — while the single $a_k$ object appears in both: it is the axis that is summed over by the base operation, shared between both references.

| `Sample` | `equation_idx` | `src` | `tgt` | `coeff` | `reindexing_slot` |
| --- | --- | --- | --- | --- | --- |
| $s_0$ | 0 | $a_i$ | $a_i$ | 1 | 1 |
| $s_1$ | 0 | $a_j$ | $a_j$ | 1 | 2 |

Each degree axis contributes exactly one `Sample` row: $s_0$ assigns $a_i$ to the first $H$ reference (slot 1), and $s_1$ assigns $a_j$ to the second (slot 2). The contracted axis $a_k$ has no `Sample` row — it appears only as `is_target = True` in both `ArrayAxis` entries, indicating it is handled by the summation rather than the outer degree loop. `reindexing_slot` disambiguates the two occurrences without requiring tensor names to be unique, and corresponds directly to the positional structure of `Broadcasted`: the two input weaves for slots 1 and 2 are distinct positional entries in the same `Broadcasted` morphism.

---

## The Acset Framework

The acset framework operates at three categorical levels. At the **schema level**, a schema $\mathcal{S}$ is a small category whose maps are partitioned into *C-set maps* (combinatorial structure — which entities exist, how they connect) and *attribute maps* (typed data — numeric or Boolean values attached to entities). At the **instance level** (within a single schema), an instance of $\mathcal{S}$ is a functor $F : \mathcal{S} \to \mathbf{Set}$; morphisms between two instances of the same schema are natural transformations, so all $\mathcal{S}$-instances form a functor category $[\mathcal{S}, \mathbf{Set}]$. At the **functor-category level** (across schemas), a schema morphism $f : \mathcal{S} \to \mathcal{S}'$ automatically induces an adjoint triple between the corresponding functor categories of instances (defined in the Schema Morphisms subsection below); a map between two *different* such categories is a functor, not a natural transformation. The subsections below proceed from concrete to formal: the structure/data split first, then the Grothendieck construction that formalizes it, then schema morphisms and data migration.

### Two-Level Structure/Data Split

The structure/data distinction — C-set maps encoding connectivity, attribute maps encoding values — manifests at both the instance and functor-category levels. The two levels are complementary, not competing.

**At the instance level.** Within a single instance, the C-set part is graph connectivity only. All numeric and Boolean values are attributes. Two instances sharing the same connectivity have the same structural type regardless of their data values.

**At the functor-category level.** Across all instances, the structural skeleton $\mathbf{C}_\sharp$ (defined below) captures what is combinatorial — support graphs and connectivity patterns — while a data functor $D$ assigns to each structural object its valid set of data assignments. Functorial composition depends only on the structural layer.

| Category | Scope | Structural | Data |
| --- | --- | --- | --- |
| **St** | functor-category ($\mathbf{St}_\sharp$) | support + coefficients ($\mathbb{N}$-matrix) | axis sizes |
| **St** | instance ($\mathcal{S}_{St}$) | directed multigraph | sizes, coefficients |
| **Br** | functor-category ($\mathbf{Br}_\sharp$) | array–axis connectivity, reindexing support | sizes, coefficients, `is_target`, `position`, `is_input`, `operator_tag`, `datatype_tag`, `max_value`, `bias`, `elementwise_fn` |
| **Br** | instance ($\mathcal{S}_{Br}$) | Sample–Axis–ArrayAxis–Array connectivity graph | sizes, coefficients, `is_target`, `position`, `is_input`, `operator_tag`, `datatype_tag`, `max_value`, `bias`, `elementwise_fn` |

The schema $\mathcal{S}$ is the fixed template defining which maps are C-set maps (structural) and which are attribute maps (data). A morphism between two instances of the same schema is a natural transformation; a map between two *different* instance categories is a functor — the distinction is formalized by the Grothendieck construction below.

### The Grothendieck Construction

The Grothendieck construction formalizes the functor-category level of the structure/data split: it recovers the full category $\mathbf{C}$ from its structural skeleton $\mathbf{C}_\sharp$ and a data functor $D$. Its importance is that it explains why the adjoint triple of the next subsection is automatic for *any* schema morphism — not just the specific ones we anticipate. Without this structure you would need to construct $\Sigma_f \dashv f^* \dashv \Pi_f$ by hand for each $f$; because instance categories are Grothendieck integrals, the triple is guaranteed to exist in general.

**$\mathbf{C}_\sharp$ — structural skeleton.** Given a category $\mathbf{C}$ whose morphisms carry both combinatorial structure and numeric/Boolean data, $\mathbf{C}_\sharp$ (bold, with sharp subscript — a local notational convention, not from Patterson et al.) is the category obtained by erasing all data. Objects of $\mathbf{C}_\sharp$ are the label sets of $\mathbf{C}$-objects (axes, arrays) with sizes stripped; morphisms of $\mathbf{C}_\sharp$ are the combinatorial patterns of $\mathbf{C}$-morphisms (support graphs, connectivity) with all numeric and Boolean values erased. Composition is well-defined because it depends only on connectivity, never on values.

**Relationship to $\mathcal{S}$ instances.** An $\mathcal{S}_{St}$ instance $F$ has two parts: a C-set part (the `Axis`/`Entry` multigraph — one morphism of $\mathbf{St}_\sharp$) and an attribute part (`size`, `coeff`). Stripping the attributes from $F$ yields one morphism of $\mathbf{St}_\sharp$. The schema $\mathcal{S}$ is the fixed template for *one* instance; $\mathbf{C}_\sharp$ is the category that *all* such structural skeletons collectively form.

To recover $\mathbf{C}$ from $\mathbf{C}_\sharp$, we need to re-attach data — a compatible assignment of values to each structural object. Define a functor

$$D : \mathbf{C}_\sharp \to \mathbf{Set}$$

assigning to each object its set of valid data assignments. Because data values are independent across domain and codomain, $D$ acts trivially on morphisms — data is never constrained by structure. The **Grothendieck construction** $\int D$ recovers $\mathbf{C}$: objects are (structural object, data assignment) pairs and morphisms are structural morphisms with no compatibility condition on data.

For **St**: $\mathbf{St}_\sharp$ has finite index sets as objects and $\mathbb{N}$-matrices (support + coefficients, sizes erased) as morphisms; $D(I) = \mathbb{N}_{>0}^I$ (size assignments). $\int D$ recovers **St**.

For **Br**: $\mathbf{Br}_\sharp$ has bare array products as objects and stripped `Broadcasted` patterns (array–axis connectivity and reindexing support graph, with all numeric and Boolean values erased) as morphisms; $D$ assigns sizes, coefficients, `is_target`, `position`, `is_input`, `operator_tag`, `datatype_tag`, `max_value`, `bias`, and `elementwise_fn`. $\int D$ recovers **Br**.

### Schema Morphisms and Data Migration

Following Patterson et al., a **schema morphism** $f : \mathcal{S} \to \mathcal{S}'$ induces an adjoint triple

$$\Sigma_f \dashv f^* \dashv \Pi_f$$

between instance categories, for any schema:

- **Restriction** ($f^*$): reindex an $\mathcal{S}'$-instance along $f$ to produce an $\mathcal{S}$-instance; when $f$ is the C-set inclusion this forgets all attribute data
- **Left adjoint** $\Sigma_f$: freely extend an $\mathcal{S}$-instance to an $\mathcal{S}'$-instance (left Kan extension)
- **Right adjoint** $\Pi_f$: compute the tightest $\mathcal{S}'$-instance compatible with an $\mathcal{S}$-instance (right Kan extension)

For $\mathcal{S}_{St}$: $\Pi_f$ computes the tightest axis-size assignment on the codomain given a stride pattern — the shape-inference operation in a neural network compiler.

For $\mathcal{S}_{Br}$: the same triple migrates sizes, coefficients, weave roles, and datatype bounds across schema morphisms between broadcasting patterns.

---

## Instances as Functors: $\Phi_a$ at the Instance Category Level

Each $\mathcal{S}_{St}$ instance is a functor $F : \mathcal{S}_{St} \to \mathbf{Set}$ — a copresheaf assigning entity sets to the schema's object types and functions to its maps. Similarly each $\mathcal{S}_{Br}$ instance is a functor $G : \mathcal{S}_{Br} \to \mathbf{Set}$. In the acset framework, **instances are functors**, and a morphism between two instances of the same schema is a natural transformation between those functors.

The $[a, \cdot]$ construction maps every **St** instance to a **Br** instance and extends to a functor $\Phi_a : \mathcal{S}_{St}\text{-Inst} \to \mathcal{S}_{Br}\text{-Inst}$ between the respective functor categories — not a natural transformation, since $\Phi_a$ maps between two different instance categories rather than between two parallel functors sharing the same source and target.

### Schema Morphism Underlying $[a, \cdot]$

There is a schema morphism $\phi : \mathcal{S}_{St} \hookrightarrow \mathcal{S}_{Br}$ that sends the **St** entity types and maps into their counterparts in the **Br** schema:

| $\mathcal{S}_{St}$ | $\mathcal{S}_{Br}$ | Role |
| --- | --- | --- |
| `Axis` | `Axis` | identity — axis nodes are shared |
| `Entry` | `Sample` | reindexing coefficient rows |
| `src` | `src` | domain axis of each nonzero entry |
| `tgt` | `tgt` | codomain axis of each nonzero entry |
| `coeff` | `coeff` | linear coefficient value |
| `size` | `size` | axis size attribute |

Applying $\phi$ to an **St** instance $F$ via the restriction functor $\phi^* : \mathcal{S}_{Br}\text{-Inst} \to \mathcal{S}_{St}\text{-Inst}$ pulls back the `Axis`/`Sample` part of any **Br** instance to the `Axis`/`Entry` structure of an **St** instance. Going the other direction, the left Kan extension $\Sigma_\phi$ freely generates a **Br** instance from an **St** instance by:

1. Adding one input `Array` $[a, Q]$ (`is_input = True`) and one output `Array` $[a, P]$ (`is_input = False`)
2. Marking all `ArrayAxis` rows with `is_target = False` — every axis is a retained (degree) axis, since $[a, \Lambda]$ is a pure reindexing with no contraction
3. Setting `reindexing_slot` on every `Sample` to the input array $[a, Q]$
4. Setting `position` on each `ArrayAxis` row to the index of that axis within its containing axis tuple (0, 1, 2, ... in order of the domain or codomain sequence)

The result is exactly the `Broadcasted` acset structure of $[a, \Lambda] : [a, Q] \to [a, P]$.

### $\Phi_a$ as a Functor

Fix a datatype $a$. Define a map on instances:

$$\Phi_a : \mathcal{S}_{St}\text{-Inst} \to \mathcal{S}_{Br}\text{-Inst}, \qquad F \mapsto [a, F]$$

where $[a, F]$ is the **Br** instance constructed by $\Sigma_\phi$ extended with the two `Array` rows and `ArrayAxis`/`is_target`/`position` data above.

**$\Phi_a$ is a functor.** For any instance morphism $\alpha : F_1 \Rightarrow F_2$ — a natural transformation assigning, for each entity type $X \in \mathcal{S}_{St}$, a function $\alpha_X : F_1(X) \to F_2(X)$ commuting with all schema maps — $\Phi_a$ produces an instance morphism $[a, \alpha] : [a, F_1] \Rightarrow [a, F_2]$ whose `Axis` and `Sample` components are $\alpha_{\texttt{Axis}}$ and $\alpha_{\texttt{Entry}}$, extended by identity on the two `Array` rows. Functoriality requires:

$$[a,\, \alpha \circ \beta] = [a,\, \alpha] \circ [a,\, \beta] \qquad \text{and} \qquad [a,\, \mathrm{id}_F] = \mathrm{id}_{[a,F]}$$

Both hold because composition and identity of instance morphisms act componentwise, and the `Array`-row identity extension commutes with componentwise operations.

### Connection to Contravariant Functoriality

The instance-level functor $\Phi_a$ is the acset expression of the contravariant functor $[a, \cdot] : \mathbf{St}^{op} \to \mathbf{Br}$ from the theory. The two levels correspond:

| Level | **St** | **Br** | Connection |
| --- | --- | --- | --- |
| Category-theoretic | shape $A \in \text{Ob}(\mathbf{St})$ | array $[a,A] \in \text{Ob}(\mathbf{Br})$ | object action of $[a,\cdot]$ |
| Category-theoretic | morphism $\eta : P \to Q$ | reindexing $[a,\eta] : [a,Q] \to [a,P]$ | contravariant morphism action |
| Acset/instance | $\mathcal{S}_{St}$-instance $F$ (a functor) | $\mathcal{S}_{Br}$-instance $[a,F]$ (a functor) | $\Phi_a(F) = \Sigma_\phi F$ |
| Acset/instance | instance morphism $\alpha : F_1 \Rightarrow F_2$ | induced morphism $[a,\alpha] : [a,F_1] \Rightarrow [a,F_2]$ | functoriality of $\Phi_a$ |

The functoriality condition $[a, \alpha \circ \beta] = [a, \alpha] \circ [a, \beta]$ is the acset expression of the condition $[a, \Lambda \mathbin{;} M] = [a, M] \mathbin{;} [a, \Lambda]$ proved in `functor_proof.md`. Both say the same thing at different levels of abstraction: the pullback of a composed reindexing equals the composition of pullbacks in the reversed order.

---

## Integration with Tensor Logic

The pyncd categorical framework does not exist in isolation. `tensorLogicNCDIntegration.md` describes how tensor logic (Domingos 2025) serves as the primary user-facing interface for constructing pyncd morphisms. This connection clarifies the natural role for acset instances in the overall architecture and reveals a direct structural correspondence between `TensorEquation` and `SBrInstance`.

### Tensor Logic as the Construction Interface

Tensor logic is a programming language whose sole primitive is the tensor equation:

```text
Y[i, j] = relu(W[i, k] X[k, j])
```

In the pyncd integration, each equation becomes a `TensorEquation(Operator)` — a frozen dataclass embedded as the `operator` field of a `Broadcasted` morphism. A `TensorProgram` collects equations, topologically sorts them, and produces a `Composed` of `Broadcasted[B, A, TensorEquation]` morphisms via `Context`-mediated axis unification (`to_morphism()`). Users write tensor equations; the pyncd term hierarchy is the internal representation generated from them.

This places terms and acset instances in a symmetric position: both are views derived from the same `TensorProgram` source, each suited to different downstream consumers.

**Current scope.** The acset path (`from_tensor_program()`) terminates at the `SBrInstance` tables — it is an analytical and representational layer. Diagram generation and PyTorch code generation currently run through the term path (`to_morphism()`). The `SBrInstance` schema now defines all fields needed to represent the compilation context: `operator_tag`, `norm_axis`, `bias`, `elementwise_fn`, `datatype_tag`, and `max_value`. `from_tensor_equation` populates `operator_tag` and `norm_axis` directly from the `TensorEquation`; `datatype_tag`, `max_value`, `bias`, and `elementwise_fn` require caller-supplied datatype information (an `array_datatypes` parameter to `from_tensor_equation`) and are not yet populated by the current implementation.

`Weave._shape` is reconstructible from `position` on `ArrayAxisRow` for any named-axis morphism derived from `TensorEquation`. Two constructs remain exclusively in the term world by design: `ProductOfMorphisms` (parallel composition, no acset counterpart) and `Block` display metadata (presentation layer above both paths).

### TensorEquation as a Proto-Acset

The `TensorEquation` dataclass has four relevant fields: `lhs_name` (the output tensor name), `lhs_indices` (the tuple of output axes), `rhs` (a sequence of `(tensor_name, axes)` pairs for each input), and `operator` (the base nonlinearity, or `None` for identity). An axis appearing in `lhs_indices` is **retained** (also called a **degree** axis) — it loops in the outer degree loop and appears in the output. An axis appearing in `rhs` but not in `lhs_indices` is **contracted** — it is summed over by the base operation.

These fields map to the four tables of `SBrInstance`: `arrays: list[ArrayRow]`, `array_axes: list[ArrayAxisRow]`, `samples: list[SampleRow]`, and `axis_sizes: dict[UID, Numeric]`. Operator-specific parameters (`bias`, `elementwise_fn`) and array datatypes (`datatype_tag`, `max_value`) require additional caller-supplied information. The structural mapping is:

| `TensorEquation` field | `SBrInstance` table entry |
| --- | --- |
| The equation itself | One `Equation` row with `equation_idx = 0` (single equation) or the equation's topological position (multi-equation program), `lhs_name = lhs_name` |
| `lhs_name` | One `Array` row with `equation_idx` from above, `slot = 0`, `is_input = False`, `name = lhs_name` |
| Each `(name, indices)` in `rhs` at position $p$ (1-indexed) | One `Array` row with the same `equation_idx`, `slot = p`, `is_input = True`, `name = name` |
| `(tensor, axis)` pair where `axis ∈ lhs_indices` | `ArrayAxis` row with the same `equation_idx`, `is_target = False` (retained / degree axis) |
| `(tensor, axis)` pair where `axis ∉ lhs_indices` | `ArrayAxis` row with the same `equation_idx`, `is_target = True` (contracted) |
| Index of `axis` in its containing tuple (`lhs_indices` or `rhs` axes) | `position` attribute on the `ArrayAxis` row |
| `ArrayAxis` belonging to the $p$-th `Array` | `array_slot = p` on the `ArrayAxis` row |
| Retained axis $i \in$ `lhs_indices` appearing in input $X$ at slot $p$ | `Sample` row: `equation_idx` from above, `src` $= i$, `tgt` $= i$, `coeff` $= 1$, `reindexing_slot` $= p$ |
| `operator` field (`None` or `Identity()`) | `IDENTITY` for `operator_tag` on the output `Array` row |
| `operator = SoftMax()` with empty `where_predicate` | `SOFTMAX` for `operator_tag`; `norm_axis` FK set to the `NormAxis` axis |
| `operator = SoftMax(where_predicate=...)` | `MASKED_SOFTMAX` for `operator_tag`; `norm_axis` FK set; Iverson predicate serialised into `iverson_expr` on the output `Array` row |
| `operator = Normalize()` with empty `where_predicate` | `NORMALIZE` for `operator_tag`; `norm_axis` FK set to the `NormAxis` axis |
| `operator = Normalize(where_predicate=...)` | `MASKED_NORMALIZE` for `operator_tag`; `norm_axis` FK set; Iverson predicate serialised into `iverson_expr` on the output `Array` row |
| `operator.bias` (when `Linear`) | `bias` attribute on the output `Array` row |
| `operator.operator` (when `Elementwise`) | `elementwise_fn` attribute on the output `Array` row |
| Array datatype (from `array_datatypes` parameter) | `datatype_tag` attribute on the `Array` row |
| `Natural.max_value` (when datatype is `Natural`) | `max_value` attribute on the `Array` row |
| Bool datatype (predicate array) | `datatype_tag = BOOL`; `max_value` left absent |
| Every axis in `lhs_indices` and every axis in `rhs` | Entry in `axis_sizes: dict[UID, Numeric]` |

**Example: $Y[i,j] = W[i,k]\, X[k,j]$.** The degree is $(i, j)$ — the retained indices. The contracted axis $k$ appears in both $W$ and $X$ but not in the output. $W$ is indexed by degree axis $i$; $X$ by degree axis $j$.

**Axis table:**

| `Axis` | `size` |
| --- | --- |
| $i$ | `FreeNumeric` |
| $j$ | `FreeNumeric` |
| $k$ | `FreeNumeric` |

**Equation table:**

| `Equation` | `equation_idx` | `lhs_name` |
| --- | --- | --- |
| $e_0$ | 0 | $Y$ |

**Array table:**

| `Array` | `equation_idx` | `slot` | `is_input` | `operator_tag` | `norm_axis` | `datatype_tag` | `max_value` | `bias` | `elementwise_fn` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $Y$ | 0 | 0 | False | IDENTITY | — | REALS | — | — | — |
| $W$ | 0 | 1 | True | — | — | REALS | — | — | — |
| $X$ | 0 | 2 | True | — | — | REALS | — | — | — |

**ArrayAxis table:**

| `ArrayAxis` | `equation_idx` | `array_slot` | `axis` | `is_target` | `position` |
| --- | --- | --- | --- | --- | --- |
| $ya_i$ | 0 | 0 | $i$ | False | 0 |
| $ya_j$ | 0 | 0 | $j$ | False | 1 |
| $wa_i$ | 0 | 1 | $i$ | False | 0 |
| $wa_k$ | 0 | 1 | $k$ | True | 1 |
| $xa_k$ | 0 | 2 | $k$ | True | 0 |
| $xa_j$ | 0 | 2 | $j$ | False | 1 |

**Sample table:**

| `Sample` | `equation_idx` | `src` | `tgt` | `coeff` | `reindexing_slot` |
| --- | --- | --- | --- | --- | --- |
| $s_0$ | 0 | $i$ | $i$ | 1 | 1 |
| $s_1$ | 0 | $j$ | $j$ | 1 | 2 |

$W$ contributes degree axis $i$ (Sample $s_0$); $X$ contributes degree axis $j$ (Sample $s_1$). The contracted axis $k$ has no `Sample` rows — it is not part of any reindexing. Its `is_target = True` entries in the `ArrayAxis` table mark it as a target axis summed over by the base operation.

**Multi-equation programs.** A `TensorProgram` with multiple equations produces a single `SBrInstance` carrying all equations. Each equation is recorded as one `EquationRow` and assigned an `equation_idx` equal to its position in topological order. Every `Array`, `ArrayAxis`, and `Sample` row carries its equation's `equation_idx` as a foreign key; the composite primary key for `Array` within the instance is `(equation_idx, slot)`. The `axis_sizes` dict is naturally flat: axis UIDs are globally unique, so a shared axis used by multiple equations appears once with its single canonical size.

**Example: two-layer MLP**, $H[p, d_{ff}] = \mathrm{relu}(W_{in}[d_{ff}, d]\, X[p, d])$ followed by $Y[p, d] = W_{out}[d, d_{ff}]\, H[p, d_{ff}]$. Axis objects $p$, $d$, $d_{ff}$ are shared across both equations via UID identity.

| `Axis` | `size` |
| --- | --- |
| $p$ | `FreeNumeric` |
| $d$ | `FreeNumeric` |
| $d_{ff}$ | `FreeNumeric` |

| `Equation` | `equation_idx` | `lhs_name` |
| --- | --- | --- |
| $e_0$ | 0 | $H$ |
| $e_1$ | 1 | $Y$ |

| `Array` | `equation_idx` | `slot` | `name` | `is_input` | `operator_tag` | `elementwise_fn` |
| --- | --- | --- | --- | --- | --- | --- |
| | 0 | 0 | $H$ | False | ELEMENTWISE | relu |
| | 0 | 1 | $W_{in}$ | True | — | — |
| | 0 | 2 | $X$ | True | — | — |
| | 1 | 0 | $Y$ | False | IDENTITY | — |
| | 1 | 1 | $W_{out}$ | True | — | — |
| | 1 | 2 | $H$ | True | — | — |

(Columns `norm_axis`, `datatype_tag`, `max_value`, `bias` omitted; all `REALS` / `—`.)

| `ArrayAxis` | `equation_idx` | `array_slot` | `axis` | `is_target` | `position` |
| --- | --- | --- | --- | --- | --- |
| | 0 | 0 | $p$ | False | 0 |
| | 0 | 0 | $d_{ff}$ | False | 1 |
| | 0 | 1 | $d_{ff}$ | False | 0 |
| | 0 | 1 | $d$ | True | 1 |
| | 0 | 2 | $p$ | False | 0 |
| | 0 | 2 | $d$ | True | 1 |
| | 1 | 0 | $p$ | False | 0 |
| | 1 | 0 | $d$ | False | 1 |
| | 1 | 1 | $d$ | False | 0 |
| | 1 | 1 | $d_{ff}$ | True | 1 |
| | 1 | 2 | $p$ | False | 0 |
| | 1 | 2 | $d_{ff}$ | True | 1 |

| `Sample` | `equation_idx` | `reindexing_slot` | `src` | `tgt` | `coeff` |
| --- | --- | --- | --- | --- | --- |
| $s_0$ | 0 | 1 | $d_{ff}$ | $d_{ff}$ | 1 |
| $s_1$ | 0 | 2 | $p$ | $p$ | 1 |
| $s_2$ | 1 | 1 | $d$ | $d$ | 1 |
| $s_3$ | 1 | 2 | $p$ | $p$ | 1 |

$H$ appears as output in equation 0 (`equation_idx=0`, `slot=0`) and as input in equation 1 (`equation_idx=1`, `slot=2`): these are distinct `Array` rows despite sharing the name $H$. Axis $d_{ff}$ is retained in equation 0 (a tiling axis there, `is_target=False`) but contracted in equation 1 (summed over, `is_target=True`) — the same UID serving different roles across equations is expected and requires no renaming. The `axis_sizes` dict holds all three UIDs exactly once, shared across both equations.

**Strided convolutions.** For non-einsum reindexings (strided convolution, diagonal slice), `coeff` $\neq 1$ in the `Sample` rows, as illustrated by the stride-2 convolution example in the Br section. No schema extension is needed; `Sample.coeff` already captures arbitrary integer strides.

### The operator_tag Attribute

The `TensorEquation.operator` field — which distinguishes an `Identity` reindexing from a `SoftMax`, `MaskedSoftMax`, `Elementwise`, `Linear`, or `Embedding` — maps to the `operator_tag` attribute on output `Array` rows (part of $\mathcal{S}_{Br}$ as defined above). By convention, `operator = None` (the default in `TensorEquation`) is treated identically to `Identity()`: both map to `OpTag.IDENTITY`, meaning pure reindexing with no base computation. `operator_tag` is undefined for input arrays.

`SoftMax` maps to `SOFTMAX`; `SoftMax(where_predicate=(...))` maps to `MASKED_SOFTMAX`. `Normalize` maps to `NORMALIZE`; `Normalize(where_predicate=(...))` maps to `MASKED_NORMALIZE`. The `norm_axis` FK is populated for all four. The Iverson predicate from `where_predicate` is serialized via `_serialize_iverson` into the `iverson_expr` field of the output `Array` row — e.g. `"(x <= q)"` — and round-trips through `arrays.csv` for both masked variants. `mask_alignments` is not stored; it is recomputable from `_compute_mask_alignment(predicate, lhs_axes)` at reconstruction time.

| `OpTag` | Pyncd class | Role |
| --- | --- | --- |
| `IDENTITY` | `Identity()` | Pure reindexing — no base computation |
| `SOFTMAX` | `SoftMax()` | Softmax along `NormAxis`; no masking |
| `MASKED_SOFTMAX` | `MaskedSoftMax(iverson_factors, mask_alignments)` | Softmax with Iverson predicate applied as $-\infty$ mask before exponentiation; TL DSL: `softmax(expr, where=pred)` |
| `ELEMENTWISE` | `Elementwise()` | Pointwise nonlinearity |
| `NORMALIZE` | `Normalize()` | Sum normalization: $x / \sum_\text{dim}(x)$; no masking |
| `MASKED_NORMALIZE` | `MaskedNormalize(iverson_factors, mask_alignments)` | Sum normalization with Iverson predicate applied as zero-mask before the denominator sum; masked positions receive zero output; TL DSL: `normalize(expr, where=pred)` |
| `EMBEDDING` | `Embedding(...)` | Discrete lookup ($\mathbb{N} \to \mathbb{R}$) |
| `ADDITION_OP` | `AdditionOp()` | Elementwise addition |
| `WEIGHTED_TRIANGULAR_LOWER` | `WeightedTriangularLower()` | Legacy causal mask (lower-triangular + sum normalization); superseded by `MASKED_SOFTMAX` in the TL DSL |
| `LINEAR` | `Linear(...)` | Weight matrix application |

Adding a new operator subclass means adding one row to this table.

### CSV Serialization

`SStInstance` and `SBrInstance` can be written to and read from CSV via `acset.csv_io`:

```python
from acset.csv_io import write_sst, read_sst, write_sbr, read_sbr

write_sbr(inst, Path('out/my_program'))   # writes five CSV files into the directory
inst = read_sbr(Path('out/my_program'))   # reconstructs from those files
```

**SSt layout.** Two files, both inside the given directory:

| File | Columns |
| --- | --- |
| `axis_sizes.csv` | `axis_uid`, `size` |
| `entries.csv` | `src_uid`, `tgt_uid`, `coeff` |

**SBr layout.** Five files:

| File | Columns |
| --- | --- |
| `axis_sizes.csv` | `axis_uid`, `size` |
| `equations.csv` | `equation_idx`, `lhs_name` |
| `arrays.csv` | `equation_idx`, `slot`, `name`, `is_input`, `operator_tag`, `norm_axis`, `datatype_tag`, `max_value`, `bias`, `elementwise_fn` |
| `array_axes.csv` | `equation_idx`, `array_slot`, `axis_uid`, `is_target`, `position` |
| `samples.csv` | `equation_idx`, `reindexing_slot`, `src_uid`, `tgt_uid`, `coeff` |

**Encoding decisions.**

- *UIDs* are written as `TypeName:integer_id` (e.g. `NormAxis:1038335474`). The type tag preserves the axis subclass (`RawAxis`, `NormAxis`, `NatAxis`, `PredAxis`) so that downstream consumers can distinguish, for example, a normalization axis from a plain axis without inspecting the `norm_axis` foreign key. Files written by older versions of the library omit the tag; the reader falls back to `RawAxis` for untagged entries.
- *Numerics* are written as a plain integer for `Integer` values and as `?integer_id` for `FreeNumeric` variables.
- *Booleans* use `true`/`false` for required fields (`is_input`, `is_target`) and `true`/`false`/`` (empty) for optional fields (`bias`).
- *Names* (`lhs_name`, array `name`) use `_` as the subscript separator, matching `DynamicName.from_str`. Display metadata (`DynamicName.settings`: bold, overline, absolute) is not preserved — names serve as labels for Lean interop and display rendering is not needed.

**Known limitations.**

- `MaskedSoftMax` and `MaskedNormalize` Iverson predicates are serialized using the `iverson_expr` field on `Array` rows (shared with BOOL input array rows). `_operator_fields` in `convert.py` calls `_serialize_iverson` on the `where_predicate` and stores the result — e.g. `"(x <= q)"` — in `iverson_expr` on the output array row. `mask_alignments` (the compile-time permutation) is not stored; it is derivable from `_compute_mask_alignment(predicate, lhs_axes)` at reconstruction time. The `iverson_expr` field and its csv read/write were already in place from the BOOL-input-array path; no schema extension was needed for either masked operator.

- `FreeNumeric` sizes created by `RawAxis.named()` carry a display name on their `uid._name`. Only `uid._id` is serialized, so the display name is lost. As a consequence, `original_size == deserialized_size` returns `False` for such sizes: `FreeNumeric.__eq__` hashes `uid` including `_name`, and the reconstructed uid has `_name=None`. Internal consistency within a round-tripped instance is unaffected — all reconstructed UIDs are mutually consistent — but cross-instance comparison with the original fails. The correct fix is to base `FreeNumeric.numeric_hash()` on `uid._id` alone, which is a change to `data_structure/Numeric.py` deferred for now.
- Compound `Numeric` expressions (`Addition`, `Multiplication`, `Power`) cannot be serialized. These do not appear in acset instances produced by `convert.py`: axis sizes are always `Integer` or `FreeNumeric`, and all coefficients are `Integer`.

### The Dual-View Pipeline

With the tensor logic integration in place, the natural architecture is:

```text
              Tensor Logic DSL
                     │
                     ▼
              TensorProgram
         (TensorEquation objects
          with shared Axis UIDs)
              ╱             ╲
             ▼               ▼
          Terms          SBrInstance
  (Composed of          (one per TensorProgram;
   Broadcasted)          flat tables with
                         equation_idx FK;
                         Axis UIDs shared)
             │               │
  ┌──────────┴──────┐  ┌─────┴──────────────────┐
  │ compilation     │  │ shape inference (Πφ)   │
  │ rendering       │  │ structural matching    │
  │ parallel        │  │ data migration         │
  │   composition   │  │ serialization          │
  │ block structure │  └────────────────────────┘
  └─────────────────┘
```

| Terms | Acset instances |
| --- | --- |
| Compilation: operator types, weave structure, code generation | Shape inference ($\Pi_\phi$, right Kan extension) |
| Rendering: `Block` metadata, `ProductOfMorphisms` layout | Structural pattern matching: kernel selection, fusion rules |
| Type-level operator (`Broadcasted[B, A, TensorEquation]`) | Data migration: adjoint triple across schema morphisms |
| Generated by `TensorProgram.to_morphism()` | Generated by `acset.convert.from_tensor_program()` |

The two views are projections of `TensorProgram` optimised for different consumers. The same `Axis` UIDs appear in the term's `Weave` objects and in the acset's `ArrayAxis` rows: any `Context`-mediated unification during `to_morphism()` is automatically reflected in both, so the views remain consistent without a separate round-trip conversion.

### What Remains in the Term World

Two constructs sit above `TensorProgram.to_morphism()` and have no acset counterpart — by design, because they are precisely what tensor logic deliberately omits (`tensorLogicNCDIntegration.md §4.3`).

**Parallel product (`ProductOfMorphisms`).** Tensor logic has no notion of running two computations in parallel. In pyncd, `ProductOfMorphisms` applies two morphisms to disjoint inputs and concatenates the outputs — essential for multi-head attention where each head operates independently. `tensorLogicNCDIntegration.md §5.6` describes how this structure can be recovered automatically from the dependency DAG of a `TensorProgram` by identifying fork-join pairs; until that analysis is implemented, it is assembled by the caller above `to_morphism()`.

**Block structure.** `Block` carries display metadata (`title`, `fill_color`, `repetition`) that has no bearing on mathematical content. It belongs to the presentation layer — above both the term and the acset — and is added by the caller to group sub-expressions for rendering.

Both constructs sit above the tensor logic boundary cleanly. The term world handles them; the acset schema does not need to.

---

## Advantages of the acset Structure/Data Decomposition

Each practical advantage below is followed by an italicized note assessing whether the current `SBrInstance` definition is sufficient to realize it from acset data alone, or whether additional information — absent from `SBrInstance` but available from the term layer — would be required.

### Practical and Implementation Advantages

**Structure as a compile-time artifact.** The structural skeleton — which axes exist, which entries connect them, which arrays and array-axis rows exist, which samples belong to which reindexing — is fixed at graph-compile time and independent of numeric values. The kernel template (loop structure, memory access patterns, which axes are tiling vs. target) can therefore be compiled once from the structural layer and reused across many instantiations with different sizes or coefficients. The acset split makes this separation explicit and type-enforced rather than implicit in coding conventions. *With `position` now included, `SBrInstance` is sufficient to reconstruct `Weave._shape` for any named-axis morphism — the positional interleaving of tiling and target axes is fully captured by sorting `ArrayAxisRow`s by `position` within each array. The remaining limitation is anonymous `WeaveMode` markers in general `Broadcasted` morphisms not derived from `TensorEquation`; those have no named axis and therefore no `ArrayAxisRow` to carry a position.*

**Shape inference is $\Pi_\phi$.** The right Kan extension along a schema morphism is the canonical categorical notion of "tightest compatible extension." For **St**, given a set of size constraints (e.g., axis equalities imposed by `Context` during `@` composition), $\Pi_\phi$ computes the unique consistent size assignment — this is shape inference. Framing it as a Kan extension means it composes correctly across schema morphisms and is guaranteed to be canonical, replacing what would otherwise be a custom traversal written per operator. *`SBrInstance` is sufficient for axis-equality-based size propagation — shared UIDs carry the equality constraints and `axis_sizes` carries the values. Output shapes for `Linear` and other operators are recoverable from `position` and `axis_sizes`; `bias` and `elementwise_fn` cover the remaining operator parameters relevant to shape.*

**Model transformations are schema morphisms.** Common compiler operations — adding a batch dimension, fusing two operations, changing loop order, adding an output axis — are schema morphisms between instances, each with an automatically induced adjoint triple ($\Sigma_\phi \dashv \phi^* \dashv \Pi_\phi$). The left adjoint $\Sigma_\phi$ freely adds the new structure; the right adjoint $\Pi_\phi$ propagates constraints back. This replaces ad hoc mutation methods with a principled, composable vocabulary. The table below maps common operations to the appropriate adjoint:

| Operation | Adjoint | Direction |
| --- | --- | --- |
| Add a batch axis to all arrays | $\Sigma_\phi$ | extend freely |
| Shape inference from axis equalities | $\Pi_\phi$ | tightest compatible |
| Extract reindexing skeleton from a Br instance | $\phi^*$ | restriction |
| Forget sizes, keep connectivity | $\phi^*$ along size-forgetting schema morphism | restriction |

*`SBrInstance` is sufficient to represent the source and result of any such transformation. The schema now includes `bias`, `elementwise_fn`, `datatype_tag`, and `max_value`, covering operator parameters and array datatypes. `Weave._shape` is reconstructible from `position` for named-axis morphisms. When `array_datatypes` is supplied to `from_tensor_equation`, the acset is sufficient for type-level dispatch without consulting the term layer.*

**Serialization decouples architecture from weights.** The structural C-set (graph connectivity) is the computation graph; the attribute tables (sizes, coefficients, `is_target`, `position`, `is_input`, `operator_tag`, `max_value`) are the data. These can be serialized and transmitted independently: the graph is compiled once and the data values are streamed separately. This maps directly to standard model checkpointing practice, but the acset framework gives it a formal justification rather than leaving it as a design convention. *`SBrInstance` is sufficient for this split. The structural tables (`arrays`, `array_axes`, `samples`) and attribute tables (`axis_sizes`, `coeff`, `is_target`, `position`, `operator_tag`, `datatype_tag`, `max_value`, `bias`, `elementwise_fn`) are already separable. When populated via `array_datatypes`, all compilation-relevant parameters are independently serializable without consulting the term layer.*

**Testability.** Structural properties (correct connectivity, expected entity counts) and numeric properties (correct coefficient values, consistent sizes) can be tested independently. The structural skeleton is finite and enumerable for any given entity count, making it tractable for property-based testing and exhaustive verification at small scale. *`SBrInstance` is sufficient — both layers are directly accessible and independently queryable.*

**Pattern matching for optimization.** Operator fusion, kernel selection, and algebraic rewriting rules operate on the structural layer only: they need to know which axes feed which outputs, but not the sizes. The C-set representation makes this a graph-matching problem with a well-defined notion of isomorphism, rather than a bespoke traversal of the morphism term tree. Two instances are structurally isomorphic if and only if their entity sets are in bijection preserving all schema maps; this is decidable independent of attribute values. *`SBrInstance` is sufficient for pattern detection — `operator_tag`, `is_target`, and the sample graph provide all necessary structural information. Applying a rewrite to produce executable output requires reconstructing the term layer; the acset closes over the detection step but not the code-generation step.*

---

### Theoretical Advantages

**Grothendieck construction gives a fibered category.** The structure/data split is not a software pattern — it is the Grothendieck integral $\int D$ of the data functor $D$ over the structural skeleton. The categorical axioms of **St** and **Br** decompose accordingly: structural composition depends only on graph connectivity; numeric composition (matrix multiplication for **St**, reindexing coefficient assembly for **Br**) acts in the fiber over the structural layer. Canonical projection functors — from **St** to the structural skeleton (forget data) and from **St** to data (evaluate at a structural object) — are consequences of the fibration, not additional constructions.

**The adjoint triple is automatic for every schema morphism.** Any schema morphism $\phi : \mathcal{S} \to \mathcal{S}'$ induces $\Sigma_\phi \dashv \phi^* \dashv \Pi_\phi$ between instance categories. This means every structural transformation between **St** or **Br** instances — not just the ones anticipated in advance — comes equipped with three data migration functors for free. Shape inference, free extension, and restriction are not separate constructions defined per use case; they are instances of a single categorical pattern applied to different schema morphisms.

**Colimit-based composition has a universal property.** Composition of two instances can be expressed as a pushout — the smallest instance into which both factor compatibly — giving composition a universal property. This is the formal counterpart of what the `Context` / UID unification system does during `@` composition: it identifies the shared boundary axes and produces the smallest consistent composite. Framing composition as a colimit connects it to the general theory of limits, making associativity and identity immediate from categorical axioms rather than custom proofs.

**The Yoneda lemma applies.** Since instances are copresheaves, every instance decomposes canonically as a colimit of representables. For $\mathcal{S}_{St}$, the representable on `Axis` is the single-axis instance and the representable on `Entry` is the single-entry (one nonzero coefficient in a $1 \times 1$ stride morphism). Any instance decomposes into a colimit of these atoms. This gives a principled vocabulary of primitive instances and guarantees that any property preserved by colimits holds for complex instances whenever it holds for the atoms.

**Connection to categorical databases.** The acset framework is an instance of Spivak's functorial data models, which connect directly to categorical query language (CQL). Queries over **St** and **Br** instances — "which axes are shared between two operations?", "which samples have coefficient greater than 1?", "what is the degree of this broadcasted operation?" — are schema morphisms whose data migration yields the answer. Standard results in database theory (completeness of CQL, semantics of joins and projections) carry over: joins of **St** instances correspond to the pushout computing composition, giving a relational-algebraic account of morphism composition without additional proof.

**$\Phi_a$ is coherent with $[a, \cdot]$ across abstraction levels.** The category-theoretic contravariant functor $[a, \cdot] : \mathbf{St}^{op} \to \mathbf{Br}$ and the instance-level functor $\Phi_a : \mathcal{S}_{St}\text{-Inst} \to \mathcal{S}_{Br}\text{-Inst}$ are the same construction at two different levels: the pyncd category level (where **St** and **Br** are the objects of study) and the acset instance level (where $\mathcal{S}_{St}$- and $\mathcal{S}_{Br}$-instances are the objects). The Grothendieck construction is what connects these levels — it is what makes the instance-level functor $\Phi_a$ the precise counterpart of the category-theoretic $[a, \cdot]$. Any result proved at the category-theoretic level — such as the `pullback_comp` theorem — automatically implies the corresponding result at the instance level, and vice versa, making results transportable between the abstract and the concrete without additional proof work.

**Natural transformations are the correct morphism concept between same-schema instances.** When instances are copresheaves, the correct notion of map between two instances of the same schema is a natural transformation: a family of functions, one per entity type, commuting with all schema maps. This is more discriminating than term equality and more general than pointwise numeric equality. It is the notion under which composition, Kan extensions, and the adjoint triple are all well-behaved. Any weaker notion of morphism would break at least one of these properties. Maps between instances of *different* schemas — such as $\Phi_a : \mathcal{S}_{St}\text{-Inst} \to \mathcal{S}_{Br}\text{-Inst}$ — are functors between functor categories rather than natural transformations; the two cases are complementary, not contradictory.

**Connection to dependent type theory and formal verification.** The structure/data split is the categorical expression of the distinction between a type context (structural skeleton: which variables exist and how they relate) and a term (data assignment: values inhabiting those types). Schema morphisms are context morphisms (substitutions). This vocabulary maps directly onto Lean 4, where the type-theoretic and category-theoretic frameworks coincide. Proving properties of **St** and **Br** in Lean becomes a matter of instantiating general results about copresheaves and Kan extensions — the same framework used to establish `pullback_comp` — rather than developing bespoke proof strategies per construction.

---

## From SBrInstance to a Diagram in Br: A Lean Encoding

### Setting

An `SBrInstance` exported via `write_sbr` / `read_sbr` is a functor $G : \mathcal{S}_{Br} \to \mathbf{Set}$ — one point in the $\mathcal{S}_{Br}$-instance category. The CSV tables assign finite sets to entity types (`Axis`, `Equation`, `Array`, `ArrayAxis`, `Sample`) and functions to the foreign-key maps between them. Importing $G$ into Lean gives access to the relational data but not yet to the categorical structure of $\mathbf{Br}$.

The goal is to interpret $G$ as a **strict monoidal functor** $D : \mathcal{J} \to \mathbf{Br}$ from a finite index category $\mathcal{J}$ whose objects are the arrays of the program and whose morphisms are its equations. $\mathbf{Br}$ is the Lean 4 encoding in [`leanncd.md §5`](leanncd.md#5-br--free-category-over-broadcasted-base-morphisms).

**Running example.** A two-equation linear layer with activation:

$$Z[i,j] = W[i,k]\cdot X[k,j] \qquad\text{(equation 0: matmul, contracts }k\text{)}$$
$$Y[i,j] = \mathrm{relu}(Z[i,j]) \qquad\text{(equation 1: elementwise, no contraction)}$$

The two equations share the intermediate array $Z$; equation 1 takes the output of equation 0 as its sole input. This chain structure makes the J composition `JMor.comp (gen eq₀) (gen eq₁)` exact — no `tens` is needed — and the two equations show two distinct `op` values and two qualitatively different reindexings.

### Two Lean library shortcuts

Two Mathlib type choices give the construction access to ready-made lemmas.

**`MvPolynomial String ℕ` for `Numeric`** ([`leanncd.md §3`](leanncd.md#3-numeric)). Symbolic axis sizes and stride coefficients inhabit the free commutative semiring on a set of named generators — exactly `MvPolynomial String ℕ` (multivariate polynomials over $\mathbb{N}$ with `String`-named indeterminates):

```lean
abbrev Numeric := MvPolynomial String ℕ
-- free variable s  ↦  MvPolynomial.X s    (degree-1 monomial)
-- literal n        ↦  MvPolynomial.C ↑n   (constant polynomial)
-- instance : CommSemiring Numeric          -- from Mathlib
```

The `ring` tactic closes all `StMat` arithmetic goals over any `CommSemiring`, so all laws in [`leanncd.md §4`](leanncd.md#4-st--semantic-category-via-stride-matrices) discharge immediately.

**Mathlib's `Matrix` for `StMat.coeffs`** ([`leanncd.md §4`](leanncd.md#4-st--semantic-category-via-stride-matrices)). The coefficient block of `StMat` is `Matrix (Fin cod.length) (Fin dom.length) Numeric`, giving direct access to `Matrix.mul`, `Matrix.mul_assoc`, `Matrix.one_mul`, and `Matrix.fromBlocks`:

```lean
-- StMat (dom cod : StObj)   -- a record with:
--   coeffs : Matrix (Fin cod.length) (Fin dom.length) Numeric
--   bias   : Fin cod.length → Numeric
-- StMat.comp: coeffs := g.coeffs * f.coeffs   (Matrix.mul)
--             bias i := dotProduct (g.coeffs i) f.bias + g.bias i
-- StMat.assoc: coefficients by Matrix.mul_assoc; bias by ring
```

`reconstructProjection` below uses `Matrix.of` to build the selection matrix without manual index arithmetic.

### The Index Category J

The generating data is read from the instance:

```lean
structure ArrayKey where
  name : String   -- from ArrayRow.name, rendered via DynamicName

structure EquationGen where
  eqIdx  : Nat
  inputs : List ArrayKey   -- sorted by slot; is_input = true
  output : ArrayKey        -- slot = 0, is_input = false

def extractGens (inst : SBrInstance) : List EquationGen :=
  inst.equations.map fun eq =>
    let arrs := inst.arrays.filter (·.equation_idx == eq.equation_idx)
    { eqIdx  := eq.equation_idx
      inputs := arrs.filter (·.is_input) |>.sortBy (·.slot) |>.map nameOf
      output := arrs.find!  (!·.is_input) |> nameOf }
-- nameOf : ArrayRow → ArrayKey  (DynamicName → String)
```

**Python counterpart.** `EquationGen` corresponds to a single `TensorEquation`; `extractGens` reads the same structural data that `TensorProgram.equations` exposes — one entry per equation, with inputs sorted by their `rhs` slot position.

For the running example, `extractGens inst` returns two generators:

```lean
-- eq₀ = { eqIdx := 0, inputs := [⟨"W"⟩, ⟨"X"⟩], output := ⟨"Z"⟩ }
-- eq₁ = { eqIdx := 1, inputs := [⟨"Z"⟩],          output := ⟨"Y"⟩ }
```

Because $\mathbf{Br}$'s $\otimes$ is list concatenation ([`leanncd.md §2`](leanncd.md#2-prop)), it is natural to give $\mathcal{J}$ the same object type rather than a separate inductive:

```lean
abbrev JObj := List ArrayKey   -- ⊗ := (· ++ ·),  𝟙 := [],  atom k := [k]
```

The morphism type is then:

```lean
inductive JMor : JObj → JObj → Type
  | gen  : (eq : EquationGen) → JMor eq.inputs [eq.output]
  | id   : JMor A A
  | comp : JMor A B → JMor B C → JMor A C
  | tens : JMor A B → JMor C D → JMor (A ++ C) (B ++ D)
```

The `tens` constructor's index type uses `List.append` directly, matching `D.obj`'s definition below. No coherence constructors (associators, unitors) appear: because $\mathbf{Br}$ is a strict PROP, every coherence isomorphism in $\mathcal{J}$ would map to `rfl` under $D$ and is safely elided. This `JMor` is the **free strict monoidal category** on the `EquationGen` quiver. The universal property — formalised for the non-strict case by `CategoryTheory.FreeMonoidalCategory` in Mathlib, with the strict variant obtained by strictification — asserts that $D$ is the *unique* strict monoidal functor $\mathcal{J} \to \mathbf{Br}$ extending its action on generators. Specifying `D.genBase` therefore determines $D$ completely; the functor laws are a consequence, not a separate choice.

For the running example, the two generators give morphisms `gen eq₀ : JMor ["W", "X"] ["Z"]` and `gen eq₁ : JMor ["Z"] ["Y"]`. Because the codomain of `gen eq₀` equals the domain of `gen eq₁`, they compose directly:

```lean
JMor.comp (gen eq₀) (gen eq₁) : JMor ["W", "X"] ["Y"]
```

No `tens` or `id` wrapper is needed — this is the cleanest form of sequential composition in $\mathcal{J}$. When an equation has additional weight inputs that are not produced by a prior equation (as in the ffn's `W_out`), those are handled by tensoring with an identity: `JMor.comp (JMor.tens JMor.id (gen eq₀)) (gen eq₁)`.

### The Denotation Functor D

The helpers `nameOf`, `dtypeOf`, and `findUID` are straightforward: `nameOf` renders `DynamicName` to `String`, `dtypeOf` reads `datatype_tag` from an `ArrayRow`, and `findUID` looks up a UID in `axis_sizes`.

**D on objects.** `D.obj` is `List.map D.singleObj`, where `D.singleObj` assembles one `ArrayType` from the tables:

```lean
def D.singleObj (inst : SBrInstance) (key : ArrayKey) : ArrayType :=
  let arr  := inst.arrays.find! (fun a => nameOf a == key.name ∧ !a.is_input)
  let aaxs := inst.array_axes
    |>.filter (fun aa => aa.equation_idx == arr.equation_idx ∧ aa.array_slot == arr.slot)
    |>.sortBy (·.position)
  { dtype := dtypeOf arr
    shape := aaxs.map fun aa => ⟨none, findUID inst aa.axis_uid⟩ }

def D.obj (inst : SBrInstance) : JObj → BrObj := List.map (D.singleObj inst)
```

**Python counterpart.** `D.singleObj` assembles an `ArrayType` record from instance tables, corresponding to constructing `Array[B, A]` in the Python term layer (where `B` is the datatype and `A` is the shape). `D.obj` maps a `JObj` to a `BrObj`; this corresponds to the `ProdObject[Array[B,A]]` type used as domain/codomain in the term layer.

Monoidality: `D.obj (A ++ B) = D.obj A ++ D.obj B` by `List.map_append` (`simp`). For the running example, `D.obj ["W", "X"] = [W_type, X_type]` and `D.obj ["Z"] = [Z_type]` — lists of `ArrayType`, not axis unions.

**D on morphisms.**

```lean
def D.map (inst : SBrInstance)
    : ∀ {A B : JObj}, JMor A B → BrMorph (D.obj inst A) (D.obj inst B)
  | _, _, .id        => .nil _
  | _, _, .comp f g  => BrMorph.comp (D.map inst f) (D.map inst g)
  | _, _, .tens f g  => PROP.tensorHom (D.map inst f) (D.map inst g)
  | _, _, .gen eq    => .cons (D.genBase inst eq) (.nil _)
```

For the running example:

```lean
D.map (gen eq₀) : BrMorph [W_type, X_type] [Z_type]
  = BrMorph.cons matmul_base (.nil _)         -- single-element list

D.map (gen eq₁) : BrMorph [Z_type] [Y_type]
  = BrMorph.cons relu_base (.nil _)

D.map (JMor.comp (gen eq₀) (gen eq₁)) : BrMorph [W_type, X_type] [Y_type]
  = BrMorph.comp (D.map (gen eq₀)) (D.map (gen eq₁))
  = BrMorph.cons matmul_base (BrMorph.cons relu_base (.nil _))
```

**Python counterparts.** `D.map .id` → identity `Rearrangement` (permutation = `(0,1,...)`); `D.map (.comp f g)` → `Composed([b1, b2, ...])` assembled via `BrMorph.comp` (list concatenation); `D.map (.tens f g)` → `ProductOfMorphisms` running two sub-morphisms on disjoint inputs.

The morphism for a full `TensorProgram` is a single `BrMorph` — a list of `BrBase` operations in topological order assembled by `BrMorph.comp` (list concatenation).

### Populating BrBase from Instance Tables

`D.genBase` assembles the five fields of `BrBase` ([`leanncd.md §9.5`](leanncd.md#95-broadcasted-operations-paper-def-13)).

**`op`.** The `operator_tag` of the output `ArrayRow` (slot 0, `is_input = false`). For equation 0 (matmul): `op = "identity"` — a plain einsum has no nonlinearity. For equation 1 (relu): `op = "elementwise/relu"`.

**`degree` — outer-loop shape $P$** ([`leanncd.md §9.3`](leanncd.md#93-reindexing-and-batch-lift-paper-defs-1011)). All axes on the output array have `is_target = false`; sorted by `position` they give $P$:

```lean
def reconstructDegree (inst : SBrInstance) (eqIdx : Nat) : StObj :=
  inst.array_axes
    |>.filter (fun aa => aa.equation_idx == eqIdx ∧ aa.array_slot == 0)
    |>.sortBy (·.position)
    |>.map    (fun aa => ⟨none, findUID inst aa.axis_uid⟩)
```

**Python counterpart.** `reconstructDegree` recovers the same axes that `Broadcasted.degree()` computes lazily at runtime via `iallequals(m.dom() for m in reindexings)` — the shared domain of all reindexing `StrideMorphism`s.

For both equations: $P_0 = [i, j]$ (the axes of $Z$) and $P_1 = [i, j]$ (the axes of $Y$).

**`inputWeaves` and `outputWeaves`** ([`leanncd.md §9.4`](leanncd.md#94-weaves-paper-def-12)). Each `ArrayAxisRow.is_target` classifies the axis as a `WeaveSlot`. The naming is inverted relative to `convert.py`: `convert.py` calls contracted axes "target" (they are the contraction target); `leanncd.md` calls retained axes "fixed" (the reindexing selects a value for them at each step $p \in P$):

| `convert.py` / `is_target` | `WeaveSlot` (Lean) | Python `Weave._shape` slot | Role |
| --- | --- | --- | --- |
| retained / degree, `false` | `WeaveSlot.fixed a` | `WeaveMode.TILED` sentinel | reindexing supplies a specific value at each $p \in P$ |
| contracted, `true` | `WeaveSlot.tiled` | concrete `Axis` object | base op processes the full extent of this axis |

**Equation 0** ($Z[i,j] = W[i,k]\cdot X[k,j]$, degree $P_0 = [i,j]$):

- $W[i, k]$: $i$ retained → `fixed i`; $k$ contracted → `tiled`. `inputWeaves 0 = [.fixed i, .tiled]`, $Q_W = [i]$.
- $X[k, j]$: $k$ contracted → `tiled`; $j$ retained → `fixed j`. `inputWeaves 1 = [.tiled, .fixed j]`, $Q_X = [j]$.
- $Z[i, j]$ (output): both retained. `outputWeaves 0 = [.fixed i, .fixed j]`.

**Equation 1** ($Y[i,j] = \mathrm{relu}(Z[i,j])$, degree $P_1 = [i,j]$):

- $Z[i, j]$ (input): no contraction; both axes retained. `inputWeaves 0 = [.fixed i, .fixed j]`, $Q_Z = [i, j]$.
- $Y[i, j]$ (output): both retained. `outputWeaves 0 = [.fixed i, .fixed j]`.

```lean
def reconstructWeave (inst : SBrInstance) (eqIdx slot : Nat) : Weave :=
  inst.array_axes
    |>.filter (fun aa => aa.equation_idx == eqIdx ∧ aa.array_slot == slot)
    |>.sortBy (·.position)
    |>.map    fun aa => if aa.is_target then .tiled
                                        else .fixed ⟨none, findUID inst aa.axis_uid⟩
```

**`reindexings` — one `StMat` per input** ([`leanncd.md §9.3`](leanncd.md#93-reindexing-and-batch-lift-paper-defs-1011)). For `TensorProgram`-derived instances, `convert.py` always writes `src_uid == tgt_uid` and `coeff = Integer(1)` on every `SampleRow`. The reindexing $\eta_i : P \to Q_i$ is therefore a pure projection: each retained input axis shares its UID with the matching degree axis, and the coefficient is 1. This makes a direct UID comparison sufficient — no `SampleRow` scan needed:

```lean
-- For TensorProgram instances: Q_i axes and P axes share UIDs (src_uid == tgt_uid).
def reconstructProjection (degree fixed : StObj) : StMat degree fixed :=
  { coeffs := Matrix.of fun j k =>
        if degree[k].uid == fixed[j].uid then (1 : Numeric) else 0
    bias := fun _ => 0 }
-- fixed := (reconstructWeave inst eqIdx slot).targetAxes
-- Eq 0, W:  fixed=[i],    degree=[i,j],   η_W = [[1,0]]     (project onto i)
-- Eq 0, X:  fixed=[j],    degree=[i,j],   η_X = [[0,1]]     (project onto j)
-- Eq 1, Z:  fixed=[i,j],  degree=[i,j],   η_Z = [[1,0],[0,1]]  (identity: no contraction)
```

For the general case (strided convolutions, non-unit coefficients), the `SampleRow` scan recovers the full affine matrix:

```lean
def reconstructReindexing (inst : SBrInstance) (eqIdx slot : Nat)
    (degree fixed : StObj) : StMat degree fixed :=
  { coeffs := Matrix.of fun j k =>
        inst.samples.find? (fun s =>
            s.equation_idx    == eqIdx    ∧ s.reindexing_slot == slot
          ∧ s.tgt_uid == fixed[j].uid     -- input axis  → Q_i row j
          ∧ s.src_uid == degree[k].uid)   -- degree axis → P column k
        |>.map (·.coeff.toNumeric) |>.getD 0
    bias := fun _ => 0 }
```

When `coeff` is always 1 and `src_uid == tgt_uid`, `reconstructReindexing` reduces to `reconstructProjection`.

**Python counterpart.** Each call to `reconstructProjection` / `reconstructReindexing` produces one `StMat`, corresponding to one element of `Broadcasted.reindexings: Prod[StrideCategory[A]]` — the tuple of `StrideMorphism` objects (one per input array) stored inside a `Broadcasted` morphism.

Assembling all fields:

```lean
def D.genBase (inst : SBrInstance) (eq : EquationGen) : BrBase _ _ :=
  let outArr := inst.arrays.find! (fun a => a.equation_idx == eq.eqIdx ∧ !a.is_input)
  let inArrs := inst.arrays
    |>.filter (fun a => a.equation_idx == eq.eqIdx ∧ a.is_input) |>.sortBy (·.slot)
  let deg    := reconstructDegree inst eq.eqIdx
  let iWeave := fun i => reconstructWeave inst eq.eqIdx inArrs[i].slot
  { op           := opTagStr outArr.operator_tag
    degree       := deg
    inputWeaves  := iWeave
    outputWeaves := fun _ => reconstructWeave inst eq.eqIdx 0
    reindexings  := fun i => reconstructProjection deg (iWeave i).targetAxes }
```

**Python counterpart.** `D.genBase` assembles a `BrBase` that corresponds directly to `Broadcasted[B, A, O]`: `op` → `operator: O`; `degree` → `Broadcasted.degree()`; `inputWeaves` → `input_weaves: Prod[Weave[B,A]]`; `outputWeaves` → `output_weaves: Prod[Weave[B,A]]`; `reindexings` → `reindexings: Prod[StrideCategory[A]]`.

The two assembled records for the running example:

```lean
-- matmul_base : BrBase [W_type, X_type] [Z_type]
-- BrBase {
--   op           = "identity"
--   degree       = [i, j]
--   inputWeaves  = [ [.fixed i, .tiled],    -- W: retain i, contract k
--                    [.tiled,   .fixed j] ] -- X: contract k, retain j
--   outputWeaves = [ [.fixed i, .fixed j] ] -- Z: write to (i,j)
--   reindexings  = [ [[1,0]],               -- η_W: (i,j) ↦ i
--                    [[0,1]] ]              -- η_X: (i,j) ↦ j
-- }

-- relu_base : BrBase [Z_type] [Y_type]
-- BrBase {
--   op           = "elementwise/relu"
--   degree       = [i, j]
--   inputWeaves  = [ [.fixed i, .fixed j] ] -- Z: no contraction
--   outputWeaves = [ [.fixed i, .fixed j] ] -- Y: write to (i,j)
--   reindexings  = [ [[1,0],[0,1]] ]        -- η_Z: identity, (i,j) ↦ (i,j)
-- }
```

For `matmul_base`, the degree loop steps through all $(i,j) \in P_0$; at each step $\eta_W$ selects row $i$ of $W$, $\eta_X$ selects column $j$ of $X$, and the base op accumulates the dot product over the tiled axis $k$. For `relu_base`, the degree loop steps through all $(i,j) \in P_1$; $\eta_Z$ is the identity, so the base op reads $Z[i,j]$ and writes $\mathrm{relu}(Z[i,j])$ to $Y[i,j]$ — no summation. The two `BrBase` records compose into the two-element `BrMorph` shown in the D on morphisms section above.

### Proof Obligations

**Strict monoidality** ([`leanncd.md §2`](leanncd.md#2-prop)). `D.obj (A ++ B) = D.obj A ++ D.obj B` by `List.map_append` (`simp`). `D.map (.tens f g) = PROP.tensorHom ...` by the `tens` case of `D.map`. Both are one-liners.

**Functoriality** ([`leanncd.md §5`](leanncd.md#5-br--free-category-over-broadcasted-base-morphisms)). `D.map .id = .nil _` and `D.map (.comp f g) = BrMorph.comp ...` follow from the `id` and `comp` cases of `D.map` by structural induction on `JMor`. Both reduce to list lemmas requiring no arithmetic.

**StMat laws** ([`leanncd.md §9.6`](leanncd.md#96-what-lean-formalises-that-the-paper-leaves-informal)). The coefficient block of `StMat.comp` is `Matrix.mul`; associativity is `Matrix.mul_assoc` from Mathlib. The bias terms close by `ring` over `CommSemiring Numeric`. `BrMorph` laws hold structurally and require no arithmetic.

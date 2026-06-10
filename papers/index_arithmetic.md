<!-- markdownlint-disable MD013 -->
# Integer Constants and Affine Index Arithmetic on Axes

The tensor-logic DSL gives index slots first-class **integer constants** and **affine
arithmetic** on axes — e.g. `h[q, d_h, 3]`, `x[i+j]`, `x[2i+1]`, `Y[2i] = …`. Such an
index expression is a morphism of the axis-stride category **St**, compiled as a
reindexing composed around the einsum core. The Domingos tensor-logic formulation relies
on this — convolution `Y[i] = Σ_j W[j]·X[i+j]`, recurrences `h[…, l+1]`, slices
`h[…, 3]` — and St supplies the arithmetic; the DSL surfaces it at the equation layer.

A file-level implementation map is in [index_arithmetic_plan.md](index_arithmetic_plan.md).

---

## Contents

- [1. St is the arithmetic of axes](#1-st-is-the-arithmetic-of-axes)
- [2. Affine index expressions](#2-affine-index-expressions)
- [3. The per-tensor iteration gate](#3-the-per-tensor-iteration-gate)
- [4. Gather and scatter](#4-gather-and-scatter)
- [5. Lowering to St](#5-lowering-to-st)
- [6. Validity and bounds](#6-validity-and-bounds)
- [7. Limitations](#7-limitations)

---

## 1. St is the arithmetic of axes

The axis-stride category **St** (weaves paper, Def. 8) *is* the arithmetic of axes:

- lone objects are **axes**, each with size `|A| ∈ ℕ`; its elements `|i⟩` for
  `i ∈ |A|` are exactly the integer coordinates;
- root morphisms are **finite affine transforms** `η : Π_i A_i → Π_j B_j`, given by
  a matrix `Λ` and vector `v`, acting on coordinates as
  `(a_i) ↦ (v_j + Σ_i a_i Λ_ij)`.

In pyncd this is `StrideMorphism.from_matrix(*matrix)`
([StrideCategory.py:63](../data_structure/StrideCategory.py)). A **slice** is the
special case built from an element `⟨c|` and identities (weaves paper, Fig. 3:
"slices are reindexings built from elements and identities, and correspond to
Pythonic `x[i,:,j]`"). Every index expression — constants, offsets, dilations, sums of
axes — is an St morphism, and the DSL's index language is a **surface syntax for St
morphisms**.

## 2. Affine index expressions

An index slot accepts an **affine map over ℤ**

$$e \;=\; b + \textstyle\sum_k c_k\, a_k, \qquad b, c_k \in \mathbb{Z},\; a_k \text{ axes}.$$

| expression | form | example |
| --- | --- | --- |
| bare axis | `a` (c=1, b=0) | `x[i]` |
| constant | `b` (no axes) | `h[…, 3]` |
| offset | `a + b` | `h[…, l+1]`, `h[…, l-1]` |
| sum of axes | `a + a'` | `X[i+j]` (convolution) |
| dilated / strided | `c·a + b` | `x[2i+1]` |

An index slot value is therefore a `RawAxis` (bare axis), a Python `int` (constant), or
an affine combination. The affine combination **reuses the arithmetic AST**
(`IversonBinOp`/`IversonUnaryOp`), reinterpreted at the index position by
`affine_normal_form` — the slot's position tells the parser it is an index, not a
predicate, so there is no dedicated index-expression node and `+`/`-` need no new
overloads. A coefficient `c·a` rides `RawAxis.__rmul__(int)` (`2*i`, restricted to
integer coefficients so it does not collide with the tensor-product `__mul__`), with the
`imul(...)` helper as the explicit form.

Index expressions appear at **both read (RHS) and output (LHS) slots** — `x[i+j]`
gathers, `Y[2i] = …` scatters. §§ 3–4 treat the two symmetrically.

> **Affine boundary.** Expressions are restricted to the **affine** maps St provides
> (`b + Σ cₖ aₖ`, linear + constant). Offset, strided, dilated, multi-axis, and scatter
> access are all affine and in scope. **Binning by floor-division or modulo**
> (`Y[i//k]`, `Y[i mod k]`) is *not* affine — out of scope (§ 7). Strided **decimation**
> `Y[i] = X[2i]` *is* affine, so most downsampling is expressible; only `//`/`mod`
> binning is not.

## 3. The per-tensor iteration gate

The same syntax `l+1` denotes two categorically different things:

- on an **iteration** axis, `h[…, l+1] = f(h[…, l])` is a **recurrence** — the next
  state defined from the current one. Per
  [iteration.md § A.4](iteration.md#a4-interaction-with-weaves-and-reindexings) this is
  a **trace**: produced sequentially, *not* a static reindexing, and not expressible as
  a `StrideMorphism`.
- on a **non-iteration** axis, `i+1` (or `i+j`, `2i+1`) is a pure **affine reindexing**
  — both sides' data already exist; it *is* a `StrideMorphism`.

Same `+1`, opposite categorical content. So axis arithmetic is **gated on whether the
axis is iterative for the tensor being read** — exactly what the iteration declaration
provides.

### 3.1 Per-tensor declaration

Iteration is declared **on the tensor**, `tl.h.iteration_axis(l)`
([iteration.md § 2.1](iteration.md#21-declaration-on-the-tensor-not-the-axis)), not on
the axis — the `_iteration_axes` per-tensor registry feeds the Scan machinery. The
discrimination is therefore per **(tensor, axis)**: the same `l` may be a recurrence
variable in one tensor and a plain index in another. The interpreter resolves `l+1`
against the tensor whose slot it is, always in scope at the slot.

### 3.2 The discrimination rule

For each slot `s` of a tensor `T` carrying axis `a` and expression `e` (read = RHS,
output = LHS):

| slot | `a` iterative for `T`? | interpretation |
| --- | --- | --- |
| `h[…, l]` (RHS) | yes | read current state |
| `h[…, l+1]` (LHS) | yes | **recurrence** → Scan (trace) |
| `h[…, l-1]` (RHS) | yes | look-back → Scan with k-step memory |
| `h[…, 0]` (LHS) | yes | base case (history position 0) |
| `h[…, 3]` (RHS) | yes | slice the materialised **history `L'`** (size `N+1`) → St reindexing |
| `x[i+j]`, `x[2i+1]`, `x[3]` (RHS) | no | **gather** — affine read → `StrideMorphism` |
| `Y[2i] = …`, `Y[i+1] = …` (LHS) | no | **scatter** — affine write → `StrideMorphism` on the output |

The gate applied at each index slot:

```text
if a is iterative for T and e advances/reads the recurrence (l, l±k):
    → iteration semantics (Scan path)
else:   # constant; affine over non-iteration axes;
        # or a constant/affine slice of T's already-materialised history
    → St affine reindexing (StrideMorphism)
```

An `axis+int` LHS (`Y[i+1]`) is syntactically identical to a recurrence (`h[…, l+1]`),
and whether the tensor is iterative is not always known when the slot is parsed
(`.iteration_axis()` may come later). That one decision is therefore made at **finalize**:
`_reclassify_offset_scatters` routes such a write to a scatter unless the tensor carries
a base case, a `.recur()` morphism, an explicit `.iteration_axis()`/`.recur()`
declaration, or a self-referential body (an incomplete recurrence, which raises "no base
case").

### 3.3 Roles of the declaration

1. **Disambiguate** recurrence (`l+1` defining next state, a trace) from affine
   reindexing (`i+1`, a static map).
2. **Supply bounds**: the recurrence axis `l` has size `N`; the materialised history
   `L'` has size `N+1`. A constant slice `h[…, 3]` is interpretable and range-checkable
   (`0 ≤ 3 ≤ N`) given these.
3. **Route** the lowering: iteration-axis recurrence arithmetic → Scan; everything else
   → St reindexing.

## 4. Gather and scatter

St's affine morphisms run in both directions, so index arithmetic is symmetric:

- a non-trivial expression on a **read** slot is a **gather** — an input reindexing
  composed *before* the core (strided/dilated/offset reads, convolution `X[i+j]`,
  decimation `X[2i]`);
- a non-trivial expression on an **output** slot is a **scatter** — an output reindexing
  composed *after* the core (upsampling `Y[2i] = X[i]`, offset/padding `Y[i+1] = X[i]`,
  strided/blocked writes).

A gather is **total** — every output coordinate is defined by its read. A scatter is
**partial** and may conflict, so it carries coverage/fill/conflict obligations a gather
does not.

The iteration base case `H[i, 0]` (LHS constant) and recurrence `H[i, l+1]` (LHS affine)
are themselves LHS index arithmetic — the LHS cases where the axis is iterative for the
defined tensor. The general mechanism **subsumes** them: LHS affine on an iterative axis
is the recurrence (→ Scan); LHS affine on a non-iteration axis is scatter (→ St
reindexing on the output). Iteration is the special case, not a parallel path.

**Scatter coverage, fill, and conflict** are declared per output with
`tl.Y.scatter(fill=…, reduce=…)`:

- a scatter whose image does not cover the whole output axis leaves the uncovered
  coordinates at **`fill`** (zero by default); the runtime allocates
  `torch.full(out_shape, fill)`;
- a **non-injective** LHS map (overlapping writes) is **rejected at build time**
  (`_scatter_injective` enumerates the value domain) unless `reduce='sum'` is declared,
  in which case overlapping writes accumulate via `index_put_(…, accumulate=True)`.

A gather may also appear **inside a recurrence body** (a shift/conv per scan step): the
gather pre-pass runs on the recurrence/base bodies in `_finalize_iter`, gated so slots
referencing the iteration axis stay on the Scan path, and hoists a non-iteration gather
into a top-level `Reindex` feeding the scan.

## 5. Lowering to St

A read `T[e_1, …, e_n]` whose slots are affine expressions compiles to

```text
core_read(T over its plain axes)  ∘  reindexing
```

one row per slot:

- a **constant** `b` → an element `⟨b|` that fixes and drops that axis = a **slice**;
- an **offset** `a + b` → a shift row;
- `c·a + b`, `a + a'` → general affine rows (dilation, strides, convolution).

The **output (LHS)** direction is the mirror image — a scatter compiles to

```text
reindexing  ∘  core_write(over plain output axes)
```

with the reindexing placed *after* the core, mapping the core's plain output coordinates
onto the affine output positions, plus the fill/conflict handling of § 4. When the LHS
axis is iterative this output reindexing is exactly the recurrence/base-case wiring the
Scan performs.

einops never sees the arithmetic (it cannot express affine index maps); St carries it,
composed around the einsum core — the weaves architecture, where reindexings surround the
batch-lifted core operation.

**The normalization pass.** A single pass at equation-registration time
(`_extract_const_slices`) rewrites each affine-indexed factor into

```text
(factor over fresh plain axes)   +   (a reindexing carrying the arithmetic)
```

so every downstream site (`_build_linear_morphism`, `bc_signature`, the sum builder, the
iteration finalizer) sees only plain `RawAxis` indices and the contraction/broadcasting
machinery is untouched. A constant slot has no axis to unify, so the pass drops it from
the equation's free-axis set before unification.

**Runtime terms.** The runtime source of truth is a pair of dedicated terms — `Reindex`
(gather) and `Scatter` (write) — carrying the affine spec explicitly as per-slot rows
`(const, ((k, coeff), …))` plus axis sizes, realised by `ConstructedReindex` /
`ConstructedScatter` via `arange`-grid advanced indexing. They are the St affine
reindexings of this section; `StrideMorphism.from_matrix` is the categorical/diagram form
(the tsncd hexagon), kept in sync by construction. The constant-only case keeps a lighter
`Slice` term (`ConstructedSlice` = repeated `torch.select`).

## 6. Validity and bounds

- **Constants:** static check `0 ≤ b < |A|` (for a history slice, `0 ≤ b ≤ N`) in
  `_check_const_bound`; skipped when the axis size is unknown.
- **Affine gather maps:** `_check_gather_bounds` computes the read interval `[lo, hi]`
  from the slot's coefficients and the fan-out axis sizes; when the indexed input has a
  concrete **declared** size it requires `0 ≤ lo` and `hi < size`, rejecting out-of-range
  (including negative) reads. When the input size is undeclared the runtime advanced-index
  catches any residual overrun.
- **Affine scatter maps / output sizing:** when the output is **declared**
  (`tl.Y.tensor(o)`), that size fixes the scatter shape (the tail beyond the image is
  filled, not truncated) and the image must fit (`maxpos < size`); when undeclared, the
  tight size `maxpos+1` is **inferred** from the map. Scatter coefficients and constant
  must be non-negative.
- **Shift boundaries** in recurrences reuse the base-case / `max_lookback` rule from the
  iteration design.
- **Scatter coverage and conflicts:** uncovered output coordinates take the declared
  `fill` (zero default); a non-injective map is rejected unless a combine is declared
  (`reduce='sum'`). See § 4.

## 7. Limitations

- **Non-affine binning** `Y[i//k]`, `Y[i mod k]` is out of scope — these are not affine
  maps, and would need an index category richer than St's affine morphisms.
- **Constant-position scatter** `Out[0] = a` (bare-constant LHS) is routed to the
  iteration base case, not a non-iteration scatter; on a non-iterative tensor it errors.
- **Coupled scans** do not run the in-step gather pre-pass (§ 4).

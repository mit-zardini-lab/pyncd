<!-- markdownlint-disable MD013 -->
# Integer Constants and Affine Index Arithmetic on Axes

**Status:** design note. Defines the model; the implementation plan is derived
separately from this note.

This note proposes giving the tensor-logic DSL first-class **integer constants** and
**affine arithmetic** on axes at index positions — e.g. `h[q, d_h, 3]`,
`x[i+j]`, `x[2i+1]` — and specifies how such index expressions are interpreted and
compiled. It resolves a deficit surfaced while building the scan visualisation
([iteration.md § 7.6](iteration.md#76-implementation-status-in-tsncd--and-what-remains)):
reading a fixed slice `h[…, 3]` of an iterative tensor is rejected by the front-end.

---

## Contents

- [1. The deficit](#1-the-deficit)
- [2. St already provides the arithmetic](#2-st-already-provides-the-arithmetic)
- [3. Affine index expressions](#3-affine-index-expressions)
- [4. Interpretation is gated on the per-tensor iteration declaration](#4-interpretation-is-gated-on-the-per-tensor-iteration-declaration)
- [5. Lowering to St](#5-lowering-to-st)
- [6. Containing the blast radius](#6-containing-the-blast-radius)
- [7. Validity and bounds](#7-validity-and-bounds)
- [8. Subtleties and open questions](#8-subtleties-and-open-questions)
- [9. Phasing](#9-phasing)

---

## 1. The deficit

Two gaps, both confirmed in the code:

1. **No integer constants at an index slot.** `TensorRef.axes` is `Prod[RawAxis]`
   ([TensorExpr.py:29](../data_structure/TensorExpr.py)); a slot can only hold an
   axis. A literal `3` reaches the unification loop
   ([TensorDSL.py:305–310](../data_structure/TensorDSL.py)) as a raw `int` and
   throws `Elements are not all equal: RawAxis != int`.
2. **Index arithmetic is conflated with predicates.** `RawAxis.__add__/__sub__` are
   overloaded, but monkey-patched to build **Iverson predicates**
   (`_rawaxis_add → IversonBinOp`, [TensorExpr.py:220](../data_structure/TensorExpr.py)),
   and `__mul__` is deliberately excluded to avoid colliding with tensor-product. So
   `l+1` builds a predicate term, not an index offset; the iteration `l±k` case is a
   narrow special path (`IterNextRef`/`IterPrevRef`), not a general mechanism.

The Domingos tensor-logic formulation relies on arithmetic over integer-valued
indices (convolution `Y[i] = Σ_j W[j]·X[i+j]`, recurrences `h[…, l+1]`, slices). The
capability is fundamental, and St already has it — the DSL simply does not surface it
at the equation layer.

## 2. St already provides the arithmetic

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
Pythonic `x[i,:,j]`"). So every index expression we want — constants, offsets,
dilations, sums of axes — is already an St morphism. The fix is to make the DSL's
index language a **surface syntax for St morphisms**.

## 3. Affine index expressions

Introduce an **index expression** at an index slot: an affine map over ℤ

$$e \;=\; b + \textstyle\sum_k c_k\, a_k, \qquad b, c_k \in \mathbb{Z},\; a_k \text{ axes}.$$

Cases:

| expression | form | example |
| --- | --- | --- |
| bare axis | `a` (c=1, b=0) | `x[i]` (today) |
| constant | `b` (no axes) | `h[…, 3]` |
| offset | `a + b` | `h[…, l+1]`, `h[…, l-1]` |
| sum of axes | `a + a'` | `X[i+j]` (convolution) |
| dilated / strided | `c·a + b` | `x[2i+1]` |

An index slot value is therefore one of: a `RawAxis` (bare axis), a Python `int`
(constant), or an affine combination. The affine combination can **reuse the
existing arithmetic AST** (`IversonBinOp`/`IversonUnaryOp`) and be *reinterpreted at
the index position* — the slot's position already tells the parser it is an index,
not a predicate. No new operator overloads are needed for `+`/`-`; see § 8 for `·`.

Index expressions appear at **both read (RHS) and output (LHS) slots** — `x[i+j]`
gathers, `Y[2i] = …` scatters. §§ 4–5 treat the two symmetrically.

> **Affine boundary.** Expressions are restricted to the **affine** maps St provides
> (`b + Σ cₖ aₖ`, linear + constant). Offset, strided, dilated, multi-axis, and
> scatter access are all affine and in scope. **Binning by floor-division or
> modulo** (`Y[i//k]`, `Y[i mod k]`) is *not* affine — out of scope, a separate and
> larger extension. Note strided **decimation** `Y[i] = X[2i]` *is* affine (a strided
> read), so most downsampling is already expressible; only `//`/`mod` binning is not.

## 4. Interpretation is gated on the per-tensor iteration declaration

The same syntax `l+1` denotes two categorically different things:

- on an **iteration** axis, `h[…, l+1] = f(h[…, l])` is a **recurrence** — the next
  state defined from the current one. As established in
  [iteration.md § A.4](iteration.md#a4-interaction-with-weaves-and-reindexings) this
  is a **trace**: produced sequentially, *not* a static reindexing, and not
  expressible as a `StrideMorphism`.
- on a **non-iteration** axis, `i+1` (or `i+j`, `2i+1`) is a pure **affine
  reindexing** — both sides' data already exist; it *is* a `StrideMorphism`.

Same `+1`, opposite categorical content. So axis arithmetic cannot be lowered
uniformly — it must be **gated on whether the axis is iterative for the tensor being
read**. That is exactly what the iteration declaration provides.

### 4.1 Per-tensor declaration (chosen model)

Iteration is declared **on the tensor**, `tl.h.iteration_axis(l)`
([iteration.md § 2.1](iteration.md#21-declaration-on-the-tensor-not-the-axis)), not
on the axis. We keep this model — it is what the DSL already implements (the
`_iteration_axes` per-tensor registry and the Scan machinery), so the index-arithmetic
work layers on top with no migration. Consequence: the discrimination is per
**(tensor, axis)** — the same `l` may be a recurrence variable in one tensor and a
plain index in another. The interpreter resolves `l+1` against the tensor whose slot
it is, which is always in scope at the slot, so this is a single lookup, not a burden.

(The Domingos per-axis alternative marks iteration on the *axis* with `*`: equation
arithmetic is then locally readable without consulting the declaration, but an axis
cannot double as a plain index and the existing per-tensor machinery would have to be
migrated. We keep per-tensor; see § 8.)

### 4.2 The discrimination rule

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
| `Y[2i] = …`, `Out[0] = …` (LHS) | no | **scatter** — affine write → `StrideMorphism` on the output |

Pseudocode the front-end applies at each index slot:

```text
if a is iterative for T and e advances/reads the recurrence (l, l±k):
    → iteration semantics (existing Scan path)
else:   # constant; affine over non-iteration axes;
        # or a constant/affine slice of T's already-materialised history
    → St affine reindexing (StrideMorphism)
```

### 4.3 Three roles of the declaration

1. **Disambiguate** recurrence (`l+1` defining next state, a trace) from affine
   reindexing (`i+1`, a static map).
2. **Supply bounds** needed to interpret offsets and constants: the recurrence axis
   `l` has size `N`; the materialised history `L'` has size `N+1`. A constant slice
   `h[…, 3]` is only interpretable (and range-checkable, `0 ≤ 3 ≤ N`) given these.
3. **Route** the lowering: iteration-axis recurrence arithmetic → Scan;
   everything else → St reindexing.

### 4.4 Gather (RHS) and scatter (LHS) are symmetric

St's affine morphisms run in both directions, so index arithmetic is symmetric:

- a non-trivial expression on a **read** slot is a **gather** — an input reindexing
  composed *before* the core (strided/dilated/offset reads, convolution `X[i+j]`,
  decimation `X[2i]`);
- a non-trivial expression on an **output** slot is a **scatter** — an output
  reindexing composed *after* the core (upsampling `Y[2i] = X[i]`, constant-position
  construction `Out[0] = a`, strided/blocked writes).

**The iteration base case and recurrence are already LHS index arithmetic.** Today
the DSL writes `H[i, 0] = …` (LHS **constant**) and `H[i, l+1] = …` (LHS **affine**)
and handles them by a dedicated iteration path. Under this model they are simply the
LHS cases where the axis is iterative for that tensor — so the general mechanism
**subsumes and unifies** the base case and the recurrence rather than treating them
as special syntax. The per-tensor gate then reads symmetrically: LHS affine on an
axis iterative for the defined tensor is the recurrence (→ Scan); LHS affine on a
non-iteration axis is scatter (→ St reindexing on the output).

**Scatter semantics.** A scatter whose image does not cover the whole output axis
leaves the uncovered coordinates undefined; the output must specify a fill (zero by
default). Overlapping writes (a non-injective LHS map) require an explicit combine
(error, or a declared reduction). These coverage/fill/conflict rules are what make
general non-iteration scatter the genuinely harder part (§ 9).

## 5. Lowering to St

A read `T[e_1, …, e_n]` whose slots are affine expressions compiles to

```text
core_read(T over its plain axes)  ∘  reindexing
```

where the reindexing is the `StrideMorphism.from_matrix` affine map from the
equation's free (output) coordinates to `T`'s coordinates, one row per slot:

- a **constant** `b` → an element `⟨b|` that fixes and drops that axis = a **slice**;
- an **offset** `a + b` → a shift row;
- `c·a + b`, `a + a'` → general affine rows (dilation, strides, convolution).

The **output (LHS)** direction is the mirror image — a scatter compiles to

```text
reindexing  ∘  core_write(over plain output axes)
```

with the `StrideMorphism` placed *after* the core, mapping the core's plain output
coordinates onto the declared (affine) output positions, plus the fill/conflict
handling of § 4.4. When the LHS axis is iterative this output reindexing is exactly
the recurrence/base-case wiring the Scan already performs (§ 4.4), so iteration is
the special case, not a parallel path.

einops never sees the arithmetic (it cannot express affine index maps); St carries
it, composed around the einsum core — exactly the weaves architecture, where
reindexings surround the batch-lifted core operation.

## 6. Containing the blast radius

The literal-int index currently breaks ~20 sites that assume every index has a
`.uid` (`_build_linear_morphism`, `bc_signature`, the sum builder, the iteration
finalizer). Rather than thread int-handling through all of them, add **one
normalization pass** at equation-registration time that rewrites each affine-indexed
factor into:

```text
(factor over fresh plain axes)   +   (a StrideMorphism reindexing carrying the arithmetic)
```

After this pass, all downstream code sees only plain axes; the arithmetic lives
entirely in the reindexing. This localises the change to the front-end and the
new pass, leaving the contraction/broadcasting machinery untouched.

## 7. Validity and bounds

- **Constants:** static check `0 ≤ b < |A|` (for a history slice, `0 ≤ b ≤ N`).
- **Affine maps:** the image must land within the target axis range, or the read is
  declared partial (out-of-range handling is an explicit choice, not silent).
- **Shift boundaries** in recurrences reuse the existing base-case / `max_lookback`
  rule from the iteration design.
- A **constant slot has no axis to unify** — the normalization pass must drop it from
  the equation's free-axis set before unification (this is the precise fix to the
  loop that throws `RawAxis != int`).
- **Scatter coverage and conflicts (LHS):** an output map whose image misses some
  output coordinates needs a declared **fill** (zero default); a non-injective output
  map (overlapping writes) needs a declared **combine** (error or reduction). See
  § 4.4.

## 8. Subtleties and open questions

- **Disambiguation from Iverson** is positional (index slot vs. `[...]` predicate) —
  already how the DSL separates the two contexts. The reinterpretation of the
  arithmetic AST at index position must be unambiguous; a dedicated `IndexExpr` node
  is an alternative to reusing `IversonBinOp` if positional reuse proves fragile.
- **`__mul__` is globally excluded** (tensor product), so a coefficient `c·a` cannot
  ride `__mul__`. `2*i` dispatches to `RawAxis.__rmul__(int)`, which we can define
  restricted to integers without colliding with tensor-product; or expose a small
  `idx(...)` helper. Decide explicitly.
- **Per-tensor iteration** (§ 4.1): kept the existing per-tensor model
  (`.iteration_axis()` / `_iteration_axes`), so no migration of the working iteration
  machinery is needed and iteration.md § 2.1 stays correct. The Domingos per-axis
  alternative is more locally readable but would require migrating that machinery; not
  worth it here.
- **Affine boundary** (§ 3): only linear+constant maps are in scope. Floor-division
  and modulo binning (`Y[i//k]`, `Y[i mod k]`) are non-affine and explicitly out; a
  later extension would need a richer index category than St's affine morphisms.
- **Scatter is harder than gather:** a gather is total (every output coordinate is
  defined by its read); a scatter is partial and may conflict, so it carries the
  fill/combine obligations of § 4.4. This is why general non-iteration scatter is the
  last phase even though LHS *iteration* arithmetic (base case, recurrence) already
  works.

## 9. Phasing

LHS (output) indexing is **in scope**, not deferred — because the scan base case and
recurrence are *already* LHS index arithmetic (§ 4.4). The phases split by difficulty,
not by side:

| Phase | Scope | Unlocks |
| --- | --- | --- |
| **P1** | integer **constants** at read *and* output slots → element/slice reindexing; recognise the iteration base case (`H[…,0]`) and recurrence (`H[…,l+1]`) as the LHS special cases the mechanism subsumes | `h[…, 3]` (the scan output head); base/step unified |
| **P2** | general **affine gather (RHS)** and **affine scatter (LHS)** over non-iteration axes → `StrideMorphism`; subsumes `IterNextRef`/`IterPrevRef` | convolution, decimation/pooling (RHS); upsampling, constant-position / strided writes (LHS); scans uniformly |
| **P3** | scatter **coverage / fill / conflict** semantics (§ 4.4) + range inference from the affine maps | robust general scatter; static bounds |
| *out of scope* | non-affine binning `Y[i//k]`, `Y[i mod k]` | (separate extension beyond St's affine maps) |

P1 is the smallest increment and unblocks the slice we hit. P2 makes gather and
scatter symmetric and folds the iteration special-cases into the one mechanism. P3 is
the genuinely harder remainder — partial/overlapping output maps — which is why
general non-iteration scatter trails even though LHS iteration arithmetic already
works.

The detailed, file-level steps for each phase are in
[index_arithmetic_plan.md](index_arithmetic_plan.md).

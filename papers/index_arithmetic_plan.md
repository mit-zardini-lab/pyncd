<!-- markdownlint-disable MD013 -->
# Implementation Plan: Integer Constants and Affine Index Arithmetic

Derived from the design note [index_arithmetic.md](index_arithmetic.md). Model:
**per-tensor** iteration (no migration of the existing `_iteration_axes` / Scan
machinery). Phases follow design-note § 9.

---

## 0. Architecture (applies to all phases)

Three moving parts; the first two are new, the third already exists.

1. **Parse** — index slots accept more than `RawAxis`: a Python `int` (constant) and
   (P2) an affine combination. Represented as an `IndexExpr`.
2. **Normalize** — a single pass at equation-registration time rewrites each factor
   carrying non-trivial index exprs into *(factor over fresh plain axes)* + *(a
   `StrideMorphism` reindexing carrying the arithmetic)*. After this pass **every
   downstream site sees only plain `RawAxis` indices** — this is the containment
   invariant that keeps the ~20 `.uid`-assuming sites untouched.
3. **Lower** — the reindexing is `StrideMorphism.from_matrix(...)`
   ([StrideCategory.py:63](../data_structure/StrideCategory.py)), composed *before* a
   read (gather) or *after* a write (scatter). einops never sees the arithmetic.

**The gate** (design-note § 4.2), evaluated per slot with the slot's tensor `T` in
scope: `a` iterative for `T` and `e ∈ {l, l±k}` → existing Scan path; else → St
reindexing. The per-tensor lookup is `_iteration_axes.get(T_name)`.

Key files:

| File | Role |
| --- | --- |
| `data_structure/TensorExpr.py` | `TensorRef`, `IndexedTensor`, the `RawAxis` arithmetic monkey-patch; home of the new `IndexExpr` |
| `data_structure/TensorDSL.py` | `TensorProxy.__getitem__/__setitem__` (parse), `_register_entry`, `_build_rhs_morphism` (the unify loop ~305–310, `_build_linear_morphism` ~372–402, `bc_signature` call ~315), `_iteration_axes`, `_finalize_iter` |
| `data_structure/StrideCategory.py` | `StrideMorphism.from_matrix` — lowering target |
| `data_structure/TensorLogic.py` | `TensorEquation`, `bc_signature` |
| `torch_compile/torch_compile.py` | runtime: must execute a standalone slice/affine `StrideMorphism` in a `Composed` chain |

---

## Phase 1 — integer constants (unblocks `h[…, 3]`)

Goal: `Y[…] = … T[…, 3] …` and the full Example 4 build, compile, and run
end-to-end, replacing the term-level assembly used for the figure.

1. **`IndexExpr` type** (`TensorExpr.py`). Minimal union for P1: `AxisRef(axis)` and
   `Const(value: int)`. `IndexedTensor.indices` becomes `Prod[IndexExpr]` (bare
   `RawAxis` auto-wraps to `AxisRef`; `int` to `Const`). Keep a `free_axes` helper
   returning only the `AxisRef` axes.
2. **Parse** (`TensorProxy.__getitem__/__setitem__`). Accept `int` at any slot; wrap
   to `Const`. No change to the bare-axis path.
3. **Normalization pass** (new, called from `_register_entry` before
   `_build_rhs_morphism`). For each factor with ≥1 `Const` slot:
   - build the **slice** `StrideMorphism` (element `⟨c|` on the constant axes,
     identity on the rest) that drops the constant axes;
   - rewrite the factor to a fresh `TensorRef` over its `free_axes`;
   - record the slice to pre-compose on that factor's wire.
4. **Fix the unify loop** (~305–310): iterate `free_axes` only, so a `Const` slot is
   never unified (removes the `RawAxis != int` throw). The constant axis is dropped
   from the equation's free-axis set.
5. **Gate for iterative tensors**: a `Const` on an axis that is iterative for the
   *read* tensor is a **history slice** (axis `L'`, size `N+1`) — bounds `0 ≤ c ≤ N`;
   for any other axis, an ordinary slice — bounds `0 ≤ c < |a|`.
6. **Runtime** (`torch_compile.py`): ensure a slice `StrideMorphism` composed in a
   chain executes as a torch index/`select`. Add/extend a `Constructed*` for it if
   reindexings are currently only handled inside `Broadcasted`.
7. **LHS constants**: the iteration base case `H[…, 0]` (LHS `Const` on an iterative
   axis) already works via the Scan path — confirm it routes through the gate
   unchanged. Non-iteration LHS `Const` (block construction, `Out[0] = …`) is scatter
   → defer to P3; reject with a clear message for now.

**Verification (P1)**
- `Example 4` via the DSL (with `tl.y[q,c] = softmax(tl.W_out[c,dh] * tl.h[q,dh,3])`)
  builds, compiles, and matches a reference loop numerically.
- unit: `Y[i] = X[i, 3]` slices correctly; out-of-range constant → static error.
- regression: existing iteration tests unchanged; no new `tsc`/pytest failures.

---

## Phase 2 — affine gather (RHS) and scatter (LHS)

Goal: offsets, dilation, strides, multi-axis reads, and non-iteration scatter.

1. **Extend `IndexExpr`** to affine: `Affine(coeffs: dict[axis,int], const: int)`.
   Parse `a+b`, `a-b`, `a+a'` (reuse the `IversonBinOp` arithmetic AST, reinterpreted
   at index position); add `RawAxis.__rmul__(int)` for `c·a` (does not collide with
   the excluded `__mul__`). Decide `idx(...)` helper vs operators (design-note § 8).
2. **Normalization → general affine** rows via `StrideMorphism.from_matrix` (the
   matrix is the coefficient matrix from free output axes to the factor's axes;
   `const` is the offset vector). Gather composes before the read.
3. **Scatter (LHS, non-iteration)**: an affine LHS map composes a `StrideMorphism`
   *after* the core write. P2 handles the **total/injective** case (image covers the
   axis, no overlap); partial/overlap → P3.
4. **Subsume `IterNextRef`/`IterPrevRef`**: an offset `l±k` on an iterative axis is an
   `Affine`; route it to the Scan path as today. Optional: collapse the dedicated
   types into `IndexExpr` to remove the special path.
5. **Runtime**: affine `StrideMorphism` → torch gather/`index_select`/strided view;
   scatter → `index_copy`/`scatter`.

**Verification (P2)**
- conv `Y[i] = Σ_j W[j]·X[i+j]`; decimation `Y[i]=X[2i]`; dilation `X[2i+1]`.
- upsampling `Y[2i] = X[i]` (injective, full-cover).
- iteration tests still pass with offsets routed through the unified path.

---

## Phase 3 — scatter coverage / fill / conflict + bounds

Goal: robust general scatter and static range checking.

1. **Fill semantics**: an LHS map whose image misses coordinates requires a declared
   fill (zero default); thread through to the runtime allocation.
2. **Conflict semantics**: a non-injective LHS map requires a declared combine (error
   by default, or a reduction).
3. **Range inference**: validate every affine map lands in range; infer output axis
   sizes from the maps where determinable; clear errors otherwise.

**Verification (P3)**: partial scatter with zero-fill; conflicting writes rejected
or reduced as declared; out-of-range maps caught statically.

---

## Cross-cutting

- **Disambiguation from Iverson** is positional — the arithmetic AST is interpreted
  as an `IndexExpr` only in an index slot, never in a `[...]` predicate. Add a focused
  test that `[l+1 == k]` (predicate) and `T[…, l+1]` (index) stay distinct.
- **Containment**: a test asserting that, after normalization, no `IndexedTensor`
  reaching `bc_signature`/`_build_linear_morphism` carries a non-`AxisRef` index — so
  the `.uid` sites are provably unaffected.
- **Error quality**: out-of-range constants, non-affine expressions (`//`, `mod`),
  and unsupported scatter must fail with messages naming the equation and slot.

## Out of scope

Non-affine binning `Y[i//k]`, `Y[i mod k]` (design-note § 3) — needs an index
category richer than St's affine morphisms; a separate effort.

## Risk register

| Risk | Mitigation |
| --- | --- |
| The ~20 `.uid` sites break on non-axis indices | the normalization pass (0.2) runs *before* them; containment test enforces it |
| Standalone reindexing has no torch runtime | P1.6 — add/extend a `Constructed*`; verify numerically before relying on it |
| History (`L'`, `N+1`) vs recurrence (`l`, `N`) confusion for constant slices | gate keys on the read tensor's `_iteration_axes`; bounds `0 ≤ c ≤ N` for history |
| `__mul__` exclusion blocks `c·a` | use `__rmul__(int)` or `idx(...)` (P2.1) |
| Scatter partiality/overlap (silent wrong results) | deferred to P3 with explicit fill/combine; P2 restricted to injective full-cover |

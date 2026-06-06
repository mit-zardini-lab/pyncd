<!-- markdownlint-disable MD013 MD036 MD032 -->
# Implementation Plan: P2 — Affine Index Arithmetic (gather / scatter)

Builds on the merged P1 (integer constants) and its machinery. Design model:
[index_arithmetic.md](index_arithmetic.md); phase overview:
[index_arithmetic_plan.md](index_arithmetic_plan.md). Per-tensor iteration model.

P2 adds **affine** index expressions `b + Σ cₖ·aₖ` at read slots (**gather**) and
output slots (**scatter**), lowered to St affine reindexings. It is the
convolution / stencil / shift / dilation / pooling / upsampling family
([use cases](index_arithmetic.md)).

---

## 0. Architecture (extends P1)

P1 established the pattern: a pre-pass rewrites a non-plain index slot into a
**synthesised reindexing entry** over fresh axes, threaded by the normal
`ThreadedComposed` machinery, so the contraction builders only ever see plain axes
(the *containment invariant*). P2 generalises the single P1 case (constant → `Slice`)
to affine maps:

| direction | P1 | P2 |
| --- | --- | --- |
| read (RHS) | `Slice` (drop a constant axis) | **`Reindex`** (affine gather: `X_g[out] = X[b+Σcₖaₖ]`) |
| write (LHS) | base case `H[…,0]` (iteration only) | **`Scatter`** (affine write: `Y[b+Σcₖaₖ] = …`) |

The **gate** (design-note §4) is unchanged and essential: an affine expression on an
**iteration axis that advances/reads the recurrence** (`l`, `l±k`) stays on the Scan
path; only **non-iteration** affine expressions become reindexings.

Key files (all touched by P1, so the seams are known):

| File | P2 role |
| --- | --- |
| `data_structure/TensorExpr.py` | affine AST already exists (`IversonBinOp` for `a±k`, `a+a'`); add `RawAxis.__rmul__(int)` for `c·a`; add an **affine-normaliser** `b + Σ cₖaₖ` |
| `data_structure/TensorDSL.py` | generalise `_extract_const_slices → _extract_affine_indices` (gather pre-pass); add an LHS **scatter post-pass** in `_register_entry`/`__setitem__`; the per-tensor gate |
| `data_structure/StrideCategory.py` | `StrideMorphism.from_matrix` — the categorical reindexing (diagram + serialisation; renders as the tsncd hexagon) |
| `torch_compile/torch_compile.py` | `ConstructedReindex` (gather) and `ConstructedScatter` (write) runtimes + `case` dispatch |

**Term representation decision.** Mirror P1: dedicated `Reindex` / `Scatter` terms
carrying the explicit affine spec (per output axis: integer coeffs over input axes +
constant, plus axis sizes) — simplest for the runtime. They *are* the St affine
reindexings of the design note; record a `StrideMorphism.from_matrix` alongside for
the diagram/serialisation so tsncd draws a hexagon. (P1's `Slice` is the
rank-reducing constant special case; leave it, or fold into `Reindex` later.)

---

## Phase 2a — affine gather (RHS)

Goal: `X[i+j]`, `X[2i+1]`, `X[i+k]` reads compile to an affine gather, so convolution
= gather (im2col) + the existing contraction.

### 2a.1 Parse affine index expressions

- Index slots already receive `IversonBinOp` for `a+b`, `a-b`, `a+a'` (the TensorExpr
  monkey-patch). Add `RawAxis.__rmul__(int)` (and `IversonBinOp.__rmul__`) for `c·a`,
  restricted to integer coefficients (no collision with the excluded `__mul__`).
- Add `affine_normal_form(expr) -> (const: int, coeffs: dict[RawAxis,int]) | None`:
  flatten the AST to `b + Σ cₖaₖ`; return `None` for a bare axis (plain) or a literal
  (P1 constant); **raise** on non-affine (`a*a'`, `//`, `mod`) with the equation/slot
  named.

### 2a.2 Gather pre-pass (generalise `_extract_const_slices`)

Rename/extend to `_extract_affine_indices`. For each RHS factor `X` with ≥1 non-plain
slot:

- Skip iteration-recurrence slots (gate): an axis that is `_iteration_axes[X]` with an
  `l`/`l±k` expression stays for the Scan path. (A *constant* on the history axis is
  still P1's `Slice`.)
- Compute the **fan-out axes** = the union of axes appearing in the affine exprs
  (e.g. `X[i+j] → (i,j)`; `X[2i+1] → (i)`; `X[ci, i+k] → (ci, i, k)` — plain `ci`
  passes through).
- Build the affine map (fan-out coords → X coords): one row per X axis,
  `idx_p = const_p + Σ coeff_{p,k}·a_k`.
- Synthesise a `Reindex` entry `(X_g, reindex_morph, fan_out_axes, (X,))`; set
  `_name_to_axes[X_g] = fan_out_axes`; rewrite the factor to `X_g[fan_out_axes]`.

After the pass the equation is a plain read of `X_g` → existing contraction path
unchanged.

### 2a.3 `Reindex` term + `ConstructedReindex` runtime

- `Reindex(fd.Term)` fields: `out_axes` (fan-out, with sizes), `in_rows`
  (per input axis: `(const, tuple[(out_axis_pos, coeff)])`), and `in_rank`.
- Runtime: build per-input-axis index tensors from `arange` grids of the out-axis
  sizes (`idx_p = const_p + Σ coeff·grid_k`), broadcast to the out shape, then
  advanced-index `X[idx_0, …, idx_{r-1}]`. For the common single-axis shift/stride,
  optionally fast-path to `narrow`/strided slicing.
- **Bounds (P2):** require every produced index in range; raise at build time when
  statically provable (sizes known), else rely on a runtime check. Clamp/zero-fill
  for out-of-range is **P3**.

### 2a.4 Subsume `IterNextRef`/`IterPrevRef` (optional cleanup)

Once affine gather exists, `l±k` look-back on the *non-advancing* read side can route
through the same normaliser; keep the Scan path for the advancing (`l+1` LHS) case.

### 2a.5 Verify (numeric, vs torch references)

- shift `Y[i]=X[i+1]`; finite diff `Y[i]=X[i+1]-X[i]`.
- decimation `Y[i]=X[2i]`; dilation `X[2i+1]`.
- **conv1d** `Y[co,i]=Σ_{ci,k} W[co,ci,k]·X[ci,i+k]` vs `F.conv1d`.
- diagonal/band `Y[i]=X[i,i]`, `X[i,i+k]`.
- relative-position `bias[i,j]=R[i-j]`.
- regression: full suite green; containment test (no non-plain index reaches
  `bc_signature`).

---

## Phase 2b — affine scatter (LHS)

Goal: `Y[2i]=X[i]` (upsampling), `Y[i+pad]=X[i]` (padding), strided/constant-position
writes. P2b handles the **total / injective** case (image covers the axis, no
overlap); partial coverage and overlap are **P3**.

### 2b.1 Detect LHS affine (non-iteration)

Extend `__setitem__`: today it routes LHS `l+1`→recurrence, LHS `int`→base case, else
plain. Add: an LHS slot that is a **non-iteration** affine expr → scatter. (Guard so
iteration base/recurrence are untouched.)

### 2b.2 Scatter post-pass

Build the equation body over the plain free axes (producing a value tensor indexed by
the RHS axes), then synthesise a `Scatter` entry placing it at the affine output
positions. Output axis sizes come from the LHS tensor's declaration (or are inferred
where determinable).

### 2b.3 `Scatter` term + `ConstructedScatter` runtime

- `Scatter(fd.Term)` fields: output shape/axes, the affine map (value coords → output
  coords), fill value (zero default, P2).
- Runtime: `out = zeros(out_shape); out[idx…] = value` via `index_put_`
  (`accumulate=False`, injective only in P2). Build `idx` like the gather.

### 2b.4 Verify

- upsampling `Y[2i]=X[i]` (zeros between) vs `F.interpolate`/manual.
- padding `Y[i+1]=X[i]` into a larger zero tensor.
- strided/block placement; full suite green.

---

## Cross-cutting

- **Gating** (design-note §4): the pre/post passes must consult `_iteration_axes`
  (per the *read/write tensor*) and skip recurrence arithmetic. Add a test that
  `H[…,l+1]` (Scan) and `X[i+1]` (Reindex) in the same program stay distinct.
- **Affine inside a scan step** — a conv/shift *inside* a recurrence body. The pre-pass
  runs in `_register_entry`, but scan steps are built in `_finalize_iter` via
  `_build_step_morph`. Either run the affine pre-pass there too, or restrict P2 to
  non-scan equations first and lift the restriction in a follow-up. **Decide early.**
- **`__mul__`/`__rmul__`** — `c·a` via `RawAxis.__rmul__(int)`; document `idx(...)`
  as the escape hatch if operator overloading proves ambiguous.
- **Containment test** — assert no `Reindex`/`Scatter`-eligible factor reaches the
  contraction builders with a non-plain index.
- **Error quality** — non-affine (`a*a'`, `//`, `mod`), out-of-range static indices,
  and unsupported (partial/overlapping) scatter must name the equation and slot.

## Out of scope (→ P3)

- scatter **coverage / fill / conflict** (partial image, overlapping writes,
  reductions) and **range inference** of output sizes from the affine maps;
- non-affine binning `Y[i//k]`, `Y[i mod k]` (needs an index category beyond St's
  affine morphisms).

## Risk register

| Risk | Mitigation |
| --- | --- |
| General affine gather is `torch.gather`/advanced-indexing — memory blow-up for big fan-outs (im2col) | accept (it is the standard conv cost); fast-path pure shift/stride to strided views |
| Out-of-range affine indices read/write garbage | P2 requires in-range (static check where sizes known + runtime guard); clamp/mask is P3 |
| Affine inside a scan step not wired | restrict P2 to non-scan first; explicit follow-up to run the pre-pass in `_build_step_morph` |
| Scatter overlap silently wrong | P2 injective/full-cover only; reject overlap with a clear error; reductions are P3 |
| `StrideMorphism` runtime vs dedicated terms divergence | dedicated `Reindex`/`Scatter` are source of truth for the runtime; `from_matrix` only for diagram/serialisation, kept in sync by construction |

## Suggested order

1. **2a.1–2a.3 single-axis gather** (shift, stride, dilation) — smallest additive step
   on P1; unlocks shifts/decimation/dilation immediately.
2. **multi-axis gather** (`i+j`, `ci,i+k`) — unlocks **convolution** (verify vs
   `F.conv1d`), the headline capability.
3. **2b injective scatter** — upsampling/padding.
4. defer coverage/fill/conflict + range inference to **P3**.

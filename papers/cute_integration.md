# Integrating CUTE Layouts with pyncd/leanncd and the `D`-Graded Formalism

This note proposes a concrete integration path for CUTE-style layout algebra into the `D`-graded colored PROP framework used by pyncd and leanncd.

The core recommendation is **hybrid integration**:

- keep the current `D`-graded semantics (broadcasting, weave laws, Eq. 3/point naturality, `Scan`/`Route` boundary),
- add a **layout-enriched index layer** (CUTE-style hierarchical layouts and algebra) underneath `St`/reindexing,
- optionally expose this enriched layer in the DSL as explicit layout annotations.

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

At a high level, CUTE treats a layout as a compositional map from coordinates to offsets/data locations:

- hierarchical shapes (`HTuple`) and hierarchical strides,
- layout as composition of shape and stride maps,
- an algebra of transformations:
  - composition,
  - coalesce/refine,
  - complement,
  - inverse (left/right/full when admissible),
  - logical division and logical product,
  - by-mode composition via tilers.

The categorical foundations paper shows a tractable fragment represented by morphisms in categories (`Tuple`, `Nest`) and proves compatibility theorems:

- morphism composition corresponds to layout composition,
- morphism coalesce corresponds to layout coalesce,
- morphism complement corresponds to layout complement,
- morphism division/product correspond to layout division/product (under admissibility).

### Tiny examples

1. **Flat row-major matrix layout**

```text
(M, N) : (N, 1)
```

2. **Hierarchical fold of an axis**

```text
(M, (N0, N1)) : (sM, (sN0, sN1))
```

3. **Composition with a tiler**

```text
L_sub = L ◦ T
```

where `T` picks/partitions modes by construction, then can be sliced by thread/value coordinates.

4. **Logical product (tile over grid)**

```text
A ⊗ B = (A, A* ◦ B)
```

with `A*` the complement of `A`.

---

## 3. Integration with the `D`-graded colored PROP

## 3.1 What should stay unchanged

These are semantic invariants and should be preserved:

- `C` as operation category (`Br`-like) with broadcast/lift semantics,
- `D` as index category controlling reindexing action,
- Eq. 3 / point-naturality and elementality-based weave reasoning,
- distinction between weave fragment vs non-weave generators (`Scan`, `Route`),
- algebra-level interpretation (`construct()` / target actegory semantics).

In short, CUTE should enrich index/layout representation, not replace `D`-graded semantics.

## 3.2 What changes in `St`

Current `St` already gives useful affine canonicalization. CUTE integration extends it from mostly flat affine maps to hierarchical layout algebra.

Practical split:

- `St_aff` (existing fragment): current `StrideMorphism`-style behavior.
- `St_layout` (new fragment): layout morphisms with hierarchical shape/stride and CUTE algebra operations.

Then provide:

- embedding/forgetful relation where affine morphisms are a subfragment,
- normalization + admissibility checks for composition/division/product/coalesce.

### Net new capabilities in `St`

- canonicalization over **hierarchical** equivalents (not only flat affine),
- explicit admissibility/divisibility failures as first-class static checks,
- principled modewise tiling/partition derivation,
- reusable inversion/complement/division lemmas for analysis/compilation.

## 3.3 What changes in `Br`

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

## 6. Lean impact: likely simplifications and costs

### Likely simpler

- proofs about index map composition/coalesce/complement can use dedicated layout lemmas,
- fewer ad hoc positional rewrites in morphism-level proofs,
- better proof reuse between backend code and formal model.

### Likely harder

- additional side conditions (admissibility/divisibility/refinement) appear in many statements,
- hierarchical typing obligations can increase local proof overhead.

Overall: **global simplification with localized side-condition burden**. Best practice is to centralize admissibility lemmas in `St_layout` and reuse throughout `Br`.

---

## 7. DSL-level support

The equation language can stay mostly unchanged, with optional layout annotations for explicit control.

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

The math is unchanged; annotations make tiling/folding explicit.

### 7.4 Transpose + blocked copy pattern

```tl
layout Src = shape((i,j))
layout Tr  = compose(Src, permute((j,i)))
layout Tile = tiler((ti,tj))

Z = copy( A@compose(Tr, Tile) )
```

This exposes by-mode composition explicitly for lowering and legality checks.

### 7.5 Convolution-like patch extraction (im2col-style)

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

```tl
layout QP = shape((b,h,q,d))
layout KP = shape((b,h,k,d))
layout ScoreTile = tiler((b,h,qo,ko,qi,ki))

S[b,h,q,k] = Q@compose(QP, ScoreTile)[b,h,q,d] * K@compose(KP, ScoreTile)[b,h,k,d]
```

Partition intent is explicit and can be validated/canonicalized before lowering.

### 7.7 Expert routing (static table, not dynamic `Route`)

```tl
layout P = shape((token, expert))
layout StaticAssign = table_layout(assign_table)  # compile-time constant

Y[token,hidden] = E@StaticAssign[token,expert,hidden] * X[token,hidden]
```

This remains weave-compatible if assignment is static; dynamic assignment still belongs to `Route`.

---

## 8. Minimal migration plan

1. **Add layout-enriched `St_layout` alongside existing `St_aff`**.
2. **Teach `Broadcasted` construction to accept explicit per-port layout maps** while retaining current weave/reindexing API.
3. **Add derivation adapters both ways**:
   - old `(weave,reindexing) -> λ_i`,
   - new `λ_i -> (weave,reindexing)` for compatibility.
4. **Expose optional DSL annotations** (`layout`, `compose`, `proj`, `tiler`, etc.) with current inference as default.
5. **Incrementally prove Lean lemmas**:
   - layout composition compatibility,
   - coalesce/refine normalization,
   - admissibility propagation.

This yields immediate practical value without breaking existing models or proofs.

---

## 9. Bottom line

CUTE is best integrated as a **layout algebra substrate for the index layer**, not as a replacement for the `D`-graded semantics.

- `St` gains expressive hierarchical layout morphisms and stronger static reasoning.
- `Br` keeps its semantics but can replace mask+reindexing internals with a cleaner per-port layout factorization.
- DSL remains familiar while gaining explicit, checkable layout intent where needed.

This is a conservative extension with high upside for compilation, verification, and Lean/Python alignment.

# Weaves in pyncd

A **weave** is the per-tensor record of which axes belong to the base
operation and which belong to the broadcast loop. It is the mechanism that
separates *what an operation computes on one tile* from *how that tile is
broadcast over a larger tensor*.

---

## Contents

1. [Data Structures](#1-data-structures)
2. [TILED vs Concrete Axes](#2-tiled-vs-concrete-axes)
3. [Degree and Reindexings](#3-degree-and-reindexings)
4. [Key Methods](#4-key-methods)
5. [Worked Example: Matrix Multiply](#5-worked-example-matrix-multiply)
6. [Building Weaves](#6-building-weaves)
7. [Compilation](#7-compilation)
8. [Datatypes and Iverson Predicates](#8-datatypes-and-iverson-predicates)

---

## 1. Data Structures

```python
class WeaveMode(Enum):
    TILED = 'TILED'

@dataclass(frozen=True)
class Array[B: Datatype, A: Axis](Term):
    datatype: B
    _shape:   tuple[A, ...]
    iverson_expr: str | None = None

@dataclass(frozen=True)
class Weave[B: Datatype, A: Axis](Term):
    datatype: B
    _shape:   tuple[A | WeaveMode, ...] = ()
    iverson_expr: str | None = None

@dataclass(frozen=True)
class Broadcasted[B: Datatype, A: Axis, O: Operator](Morphism[Array[B, A]]):
    operator:      O
    input_weaves:  tuple[Weave[B, A], ...]
    output_weaves: tuple[Weave[B, A], ...]
    reindexings:   tuple[StrideCategory[A], ...]
```

**`Array`** is a fully-shaped object in **Br**: a typed tensor with concrete
axes only. **`Weave`** is a shape *template*: each position is either a
concrete `Axis` (seen directly by the base operation) or `WeaveMode.TILED`
(a placeholder filled at runtime from the broadcast degree). `Array` and
`Weave` share the same `datatype` and `iverson_expr` fields — `Weave` is
`Array` with some axes left open.

**`Broadcasted`** packages an operator with its weaves and reindexings:

| Field | Meaning |
| --- | --- |
| `operator` | Base computation: `TensorEquation`, `Linear`, `SoftMax`, etc. |
| `input_weaves` | One weave per input; encodes which axes are local and which are broadcast. |
| `output_weaves` | One weave per output; typically all-TILED (output shape = full degree). |
| `reindexings` | One `Rearrangement` per input; maps the shared degree to the subset this input uses. |

---

## 2. TILED vs Concrete Axes

Each position in `Weave._shape` encodes one axis of the tensor:

- **Concrete `Axis`** — a *target axis*. The base operation sees it directly;
  depending on the operator it may be contracted (summed), passed through, or
  produced as a local output.
- **`WeaveMode.TILED`** — a *tiling axis*. Filled at runtime by the broadcast
  loop; the base operation does not see it directly.

For a batch-head-feature tensor `(b, h, k)` where `b` is the broadcast axis:

```python
Weave(Reals(), (WeaveMode.TILED, h, k))
#              └── b supplied by degree    └── local to operator
```

The TILED/concrete split exactly encodes the paper's Boolean family `w_i ∈
{0, 1}` (§2.2 of the categorical framework), but stores it inline rather than
as a separate tuple.

---

## 3. Degree and Reindexings

The **degree** `P` is the shared loop domain of all reindexings — the set of
index tuples over which the broadcast loops. It is derived as the common
domain of all reindexings:

```python
def degree(self) -> ProdObject[A]:
    return iallequals(morphism.dom() for morphism in self.reindexings)
```

Each reindexing `η_i : P → Q_i` maps the shared degree to the tiling shape
of input `i` (the subset of degree axes that input actually uses). The weave
says *where* the TILED holes are; the reindexing says *what coordinate fills
each hole*.

| Case | Meaning |
| --- | --- |
| `η = id` | Input uses all degree axes in order. |
| `η = ()` | Input has no TILED axes — reused at every degree coordinate. |
| Projection | Input reads a strict subset, e.g. only `i` from degree `(i, j)`. |
| Duplication | One degree axis fills multiple TILED positions (diagonal slice). |
| Affine map | `StrideMorphism` with strides, e.g. for convolution or index arithmetic. |

In practice, reindexings are `Rearrangement` objects (integer-mapping
projections) for pure einsums, and full `StrideMorphism` objects only when
affine index arithmetic is involved.

---

## 4. Key Methods

All four methods walk `Weave._shape` and handle each slot by type.

### `target() → Array[B, A]`

Strips TILED slots, returning the shape visible to the base operator:

```python
Weave(Reals(), (TILED, h, q, k)).target()
  → Array(Reals(), (h, q, k))
```

### `imprint_to_degree(other) → Array[B, A]`

Fills TILED slots from `other` in order, leaving concrete axes in place.
Returns a fully-concrete `Array`:

```python
Weave(Reals(), (TILED, k)).imprint_to_degree((i,))
  → Array(Reals(), (i, k))
```

Used by `dom()` and `cod()`:

```python
def dom(self):
    return ProdObject.from_iter(
        weave.imprint_to_degree(reindexing.cod())
        for weave, reindexing in zip(self.input_weaves, self.reindexings)
    )

def cod(self):
    return ProdObject.from_iter(
        weave.imprint_to_degree(self.degree())
        for weave in self.output_weaves
    )
```

`dom()` fills each input weave with the *subset* of degree axes its
reindexing selects. `cod()` fills each output weave with the *full* degree.

### `imprint(tiling_imprint) → tuple[A | T]`

Fills only TILED slots; concrete axes are kept as-is. Returns a mixed
tuple. Used when partial filling is needed.

### `imprint_axes(tiling_imprint, axes_imprint) → tuple[T]`

Dual-source fill: TILED slots come from `tiling_imprint`, concrete axes come
from `axes_imprint`. Returns a uniformly-typed tuple. Used by the compiler to
substitute degree tags for TILED slots and contraction tags for concrete axes
simultaneously.

---

## 5. Worked Example: Matrix Multiply

```text
Y[i, j] = W[i, k] * X[k, j]
```

`i` and `j` appear on the LHS → retained → degree axes.
`k` appears only on the RHS → contracted → concrete target axis.

```text
degree = (i, j)
```

| Tensor | Full shape | Weave `_shape` | Notes |
| --- | --- | --- | --- |
| `W[i, k]` | `(i, k)` | `(TILED, k)` | `i` from degree; `k` contracted |
| `X[k, j]` | `(k, j)` | `(k, TILED)` | `k` contracted; `j` from degree |
| `Y[i, j]` | `(i, j)` | `(TILED, TILED)` | output = full degree |

Reindexings are projections out of `(i, j)`:

```text
η_W(i, j) = (i,)    η_X(i, j) = (j,)
```

Verified by the test suite:

```python
assert br.input_weaves[0]._shape == (WeaveMode.TILED, k)
assert br.input_weaves[1]._shape == (k, WeaveMode.TILED)
assert all(p is WeaveMode.TILED for p in br.output_weaves[0]._shape)
```

---

## 6. Building Weaves

### TensorLogic path (primary)

`TensorEquation.bc_signature()` in `data_structure/TensorLogic.py` derives
weaves from UID identity:

- An axis whose UID is in `lhs_indices` → **retained** → `WeaveMode.TILED` in
  the input weave
- An axis whose UID is absent from `lhs_indices` → **contracted** → concrete
  `Axis` in the input weave

The central translation:

```python
tuple(
    WeaveMode.TILED if ax.uid in retained_uid_to_pos else ax
    for ax in _factor_axes(factor)
)
```

The output weave is all-TILED (the output shape equals the retained degree):

```python
output_weave = Weave(out_dt, tuple(WeaveMode.TILED for _ in degree))
```

### String signature path (Einops operators)

`construction_helpers/einops.py` provides `signature_to_buckets()` and
`bucketed_to_broadcast()`. These parse an einops-style string
(`"i k, k j -> i j"`) and produce the same `(input_weaves, output_weaves,
reindexings)` triple, using negative indices for retained axes (→ TILED) and
non-negative indices for contracted axes (→ concrete). The two paths converge
on the same `Broadcasted` structure.

### Batch lifting

`construction_helpers/lift.py:broadcasted_stride_lift()` extends an existing
`Broadcasted` over new batch axes by prepending TILED slots:

```python
input_weaves = tuple(
    Weave(weave.datatype,
          (WeaveMode.TILED,) * len(lift_by_morphism.cod()) + weave._shape)
    for weave in base.input_weaves
)
```

This realises the paper's batch-lift law: slicing after the lifted operation
is the same as slicing first and then running the original.

---

## 7. Compilation

`generate_tensor_equation_signature()` in `torch_compile/torch_compile.py`
converts weave structure to an `einops.einsum` string:

- TILED slots → degree tags `y0, y1, …`
- Concrete axes → contraction tags `x0, x1, …`, assigned by UID so the same
  axis in two weaves gets the same tag (causing einsum contraction)

The key loop assigns contraction tags by scanning concrete slots:

```python
for weave in target.input_weaves:
    for slot in weave._shape:
        if not isinstance(slot, WeaveMode):
            if slot.uid not in contracted_tag:
                contracted_tag[slot.uid] = f'x{tag_counter}'
                tag_counter += 1
```

Each input segment is then formed with `weave.imprint_axes(degree_tags,
contracted_tags)`, producing the full einops string. For matrix multiply:

```text
... y0 x0, ... x0 y1 -> ... y0 y1
```

For simpler broadcastable operators, `torch_compile/bcast.py` uses the same
weave structure to select between `torch.vmap`, semantic reshape, or a direct
`dim=` argument.

---

## 8. Datatypes and Iverson Predicates

A weave's `datatype` field is independent of its TILED/concrete structure:

```python
Weave(Reals(), ...)    # real-valued tensor
Weave(Bool(), ...)     # binary predicate
Weave(Natural(n), ...) # discrete token indices
```

Inline Iverson predicates (`q <= x`) become Bool-typed input weaves during
`bc_signature()`:

```python
Weave(
    Bool(),
    tuple(WeaveMode.TILED if ax.uid in retained_uid_to_pos else ax
          for ax in _factor_axes(factor)),
    iverson_expr=_serialize_iverson(factor),
)
```

The `iverson_expr` string (e.g. `"q <= x"`) is carried for display by tsncd.

**Sized predicates** (all axes carry concrete `Integer` sizes) are
auto-materialised as constant `nn.Module` buffers at compile time and removed
from the morphism's domain. **Unsized predicates** remain as Bool input weaves
— the caller must supply the mask tensor at runtime.

The output weave's datatype drives the Heaviside demotion in
`ConstructedTensorEquation`:

```python
self.demote = isinstance(target.output_weaves[0].datatype, Bool)
# forward(): if self.demote: return (result > 0).to(result.dtype)
```

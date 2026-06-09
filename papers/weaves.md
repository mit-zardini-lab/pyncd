# Weaves in pyncd

This note explains how **weaves** work in pyncd: what they mean categorically,
how they are represented in code, how they are built from tensor equations and
signatures, and how the compiler uses them.

The short version is:

> A weave is the per-tensor record of which axes belong to the base operation
> and which axes belong to the broadcast loop.

This is the mechanism that lets pyncd separate *what an operation computes on
one tile* from *how the operation is tiled, broadcast, and indexed over a larger
tensor*.

---

## Contents

1. [The Core Idea](#the-core-idea)
2. [Arrays, Weaves, and Broadcasted Morphisms](#arrays-weaves-and-broadcasted-morphisms)
3. [TILED vs Concrete Axes](#tiled-vs-concrete-axes)
4. [Degree and Reindexings](#degree-and-reindexings)
5. [How Weaves Reconstruct Array Shapes](#how-weaves-reconstruct-array-shapes)
6. [Worked Example: Matrix Multiply](#worked-example-matrix-multiply)
7. [How TensorLogic Builds Weaves](#how-tensorlogic-builds-weaves)
8. [How String Signatures Build Weaves](#how-string-signatures-build-weaves)
9. [Lift and Batch Axes](#lift-and-batch-axes)
10. [Compilation](#compilation)
11. [Datatypes and Iverson Predicates](#datatypes-and-iverson-predicates)
12. [Mental Model](#mental-model)

---

## The Core Idea

The paper describes broadcasting as a categorical way to run a base operation
independently over a loop domain called the **degree**:

```text
P = broadcast / tiled / loop shape
```

Here `P` is an object of the axis-stride category **St**: an ordered product of
axes. If `P = (b, s)`, then `b` and `s` are axes, for example a batch axis and a
sequence axis. The broadcast loop has one coordinate for each pair of index
values `(b_idx, s_idx)` drawn from those two axes.

For each input tensor, a weave records two choices for every axis position:

- **target axes** are seen directly by the base operation;
- **tiling axes** are supplied by the outer broadcast loop.

If an array has ordered shape

```text
S = (S_0, S_1, ..., S_{n-1})
```

then the index `i` in `w_i` means "the `i`-th axis position of this array
shape", with `0 <= i < n`.

The paper presents a weave as a Boolean family:

```text
w_i = 1  means S_i is a target axis
w_i = 0  means S_i is a tiling axis
```

In pyncd, this Boolean family is not stored as a separate tuple of `0` and `1`
values. Instead, the classification is stored directly in `Weave._shape`.
Each slot of `Weave._shape` is either:

```python
Axis              # target axis
WeaveMode.TILED  # tiling axis
```

For example, suppose a tensor has shape:

```text
S = (b, h, k)
```

where `b`, `h`, and `k` are axes. In the paper notation, the weave

```text
w = (0, 1, 1)
```

means:

```text
b is a tiling axis
h is a target axis
k is a target axis
```

pyncd represents the same weave as:

```python
Weave(Reals(), (WeaveMode.TILED, h, k))
```

This call has two positional arguments:

- `Reals()` is the weave's datatype: the tensor stores real-valued entries.
- `(WeaveMode.TILED, h, k)` is the weave's `_shape` tuple: one entry for each
  axis position of the tensor shape `S = (b, h, k)`.

The entries of the `_shape` tuple line up with the original shape positions:

```text
original shape S:      (b,                h, k)
pyncd weave _shape:    (WeaveMode.TILED,  h, k)
```

The first `_shape` entry is `WeaveMode.TILED` because the original first axis
`b` is filled by the broadcast degree. The second and third `_shape` entries are
the concrete axes `h` and `k` because those axes are seen by the base operation.

The relevant implementation is in
[`data_structure/BroadcastedCategory.py`](../data_structure/BroadcastedCategory.py).

---

## Arrays, Weaves, and Broadcasted Morphisms

An `Array` is a fully shaped object in the broadcasted category **Br**:

```python
@dataclass(frozen=True)
class Array[B: Datatype, A: Axis](Term):
    datatype: B
    _shape: tuple[A]
    iverson_expr: str | None = None
```

In this type annotation, `B` is the datatype parameter, such as `Reals()` or
`Bool()`, and `A` is the axis type parameter, usually `RawAxis` in this
codebase.

An `Array` says:

```text
values of datatype B indexed by a shape
```

For example:

```text
Array(Reals(), (b, s, d))
```

represents a real-valued tensor with shape `(b, s, d)`, where `b`, `s`, and `d`
are axis objects. A typical reading is batch axis `b`, sequence axis `s`, and
feature axis `d`.

A `Weave` is not quite a full array shape. It is a shape pattern with some slots
left to be filled by the broadcast degree:

```python
class WeaveMode(Enum):
    TILED = 'TILED'

@dataclass(frozen=True)
class Weave[B: Datatype, A: Axis](Term):
    datatype: B
    _shape: tuple[A | WeaveMode] = ()
    iverson_expr: str | None = None
```

So:

```text
Weave(Reals(), (TILED, TILED, d))
```

means:

```text
The first two axes are supplied by the broadcast loop.
The axis d is seen by the base operation.
```

A `Broadcasted` morphism packages an operator with its weaves and reindexings:

```python
@dataclass(frozen=True)
class Broadcasted[B: Datatype, A: Axis, O: Operator](Morphism[Array[B, A]]):
    operator: O
    input_weaves: tuple[Weave[B, A]]
    output_weaves: tuple[Weave[B, A]]
    reindexings: tuple[StrideCategory[A]]
```

The four fields correspond to Definition 13 in the paper:

| Field | Meaning |
| --- | --- |
| `operator` | The base computation, e.g. `Einops`, `Linear`, `SoftMax`, `TensorEquation`. |
| `input_weaves` | One weave per input tensor. |
| `output_weaves` | One weave per output tensor. |
| `reindexings` | One stride morphism per input, telling that input how to read from the shared degree. |

---

## TILED vs Concrete Axes

Each slot of a weave has one of two meanings. A **slot** is one position in the
ordered tuple `Weave._shape`; for example, `(TILED, h, q, k)` has four slots.

### Concrete `Axis`

A concrete `Axis` is a **target axis**. The base operation sees this axis
directly. Depending on the operator, it may be:

- contracted over, as in the `k` axis of matrix multiplication;
- passed through as a free local axis;
- produced as a local output axis.

For example:

```text
Weave(Reals(), (h, q, k))
```

says that the axes `h`, `q`, and `k` are all local to the base operator.

### `WeaveMode.TILED`

`TILED` marks a **tiling axis**. The base operation does not see this axis
directly; it is filled by the broadcast degree at runtime.

For example:

```text
Weave(Reals(), (TILED, h, q, k))
```

says:

```text
The first axis is a loop/broadcast axis.
The h, q, and k axes are local target axes.
```

In an attention score computation,

```text
S[b, h, q, x] = sum_k Q[b, h, q, k] K[h, x, k]
```

the axes are:

- `b`: batch;
- `h`: attention head;
- `q`: query position;
- `x`: key/value position;
- `k`: head feature dimension, contracted by `sum_k`.

The `Q` weave might be:

```text
(TILED, h, q, k)
```

when `b` is the broadcast degree. At each `b`, the base operation sees a local
slice:

```text
Q[b, :, :, :]
```

The `K` input might have weave:

```text
(h, x, k)
```

meaning `K` is reused for every batch coordinate.

---

## Degree and Reindexings

The **degree** is the shared loop shape of a `Broadcasted` morphism.

In code:

```python
def degree(self) -> ProdObject[A]:
    return iallequals(
        morphism.dom()
        for morphism in self.reindexings
    )
```

All input reindexings must have the same domain. That common domain is the
degree:

```text
eta_i : P -> Q_i
```

where:

- `P` is the shared degree;
- `i` ranges over input positions, so `i = 0` means the first input tensor,
  `i = 1` means the second, and so on;
- `Q_i` is the tiling shape of input `i`, i.e. the shape obtained by keeping
  only the `TILED` slots of input weave `i`;
- `eta_i` is the reindexing for input `i`;
- `p` is one concrete coordinate of the degree `P`;
- `eta_i(p)` tells input `i` which tiled coordinate in `Q_i` to read at loop
  coordinate `p`.

This is how pyncd represents broadcasting, reuse, projections, diagonal slices,
and affine indexing.

In this table, `eta` is a generic reindexing map; concrete inputs usually have
names such as `eta_W` or `eta_X`.

| Case | Meaning |
| --- | --- |
| `eta = id` | Input is indexed by the same degree coordinate. |
| `eta = ()` | Input has no tiled axes and is reused everywhere. |
| projection | Input reads only some axes from a larger degree. For example, from degree `(i, j)`, an input tensor `U[i]` reads only the `i` coordinate. |
| duplication | One degree axis fills multiple input positions. For example, with degree coordinate `p`, an input can read `X[p, p, j]`. |
| affine map | Input reads coordinates like `s * p + w`, where `s` is a fixed stride, `p` is a loop coordinate, and `w` is a filter/window coordinate. This is used by convolution/index arithmetic. |

The weave says *where* the tiling holes are. The reindexing says *what goes in*
those holes.

---

## How Weaves Reconstruct Array Shapes

The key methods on `Weave` are:

```python
def target(self) -> Array[B, A]:
    ...

def imprint(self, tiling_imprint: Iterable[T]) -> tuple[A | T]:
    ...

def imprint_axes(self, tiling_imprint: Iterable[T], axes_imprint: Iterable[T]) -> tuple[T]:
    ...

def imprint_to_degree(self, other: Iterable[A]) -> Array[B, A]:
    ...
```

Here `T` is an arbitrary placeholder type used by the method: the same weave
logic works whether the imprint items are axes, compiler tags such as `y0`, or
integer vmap locations.

They all use the same idea: walk over `Weave._shape`; keep concrete axes, and
replace `TILED` slots with items from an external tiling sequence.

### `target()`

`target()` strips out `TILED` slots:

```text
Weave(Reals(), (TILED, h, q, k)).target()
  = Array(Reals(), (h, q, k))
```

Here `h`, `q`, and `k` are concrete axes, while `TILED` is the single tiling
slot removed by `target()`.

This is the shape visible to the base operator.

### `imprint_to_degree()`

`imprint_to_degree()` fills `TILED` slots with concrete axes:

```text
Weave(Reals(), (TILED, k)).imprint_to_degree((i,))
  = Array(Reals(), (i, k))
```

Here `i` is the axis supplied for the one `TILED` slot, and `k` is the concrete
target axis already stored in the weave.

`Broadcasted.dom()` uses this method with each input reindexing's codomain:

```python
def dom(self):
    return ProdObject.from_iter(
        weave.imprint_to_degree(reindexing.cod())
        for weave, reindexing in zip(self.input_weaves, self.reindexings)
    )
```

`Broadcasted.cod()` is similar, but output weaves are imprinted with the full
degree:

```python
def cod(self):
    return ProdObject.from_iter(
        weave.imprint_to_degree(self.degree())
        for weave in self.output_weaves
    )
```

---

## Worked Example: Matrix Multiply

Consider:

```text
Y[i, j] = W[i, k] * X[k, j]
```

Here:

- `i` and `j` occur on the left-hand side (LHS), so they are retained/output
  axes.
- `k` occurs only on the right-hand side (RHS), so it is contracted.
- `W`, `X`, and `Y` are tensor names; `W[i, k]` means the entry of tensor `W`
  indexed by axes `i` and `k`.

The degree is:

```text
P = (i, j)
```

The input and output weaves are:

| Tensor | Full shape | Weave | Meaning |
| --- | --- | --- | --- |
| `W[i, k]` | `(i, k)` | `(TILED, k)` | `i` comes from the degree; `k` is local/contracted. |
| `X[k, j]` | `(k, j)` | `(k, TILED)` | `k` is local/contracted; `j` comes from the degree. |
| `Y[i, j]` | `(i, j)` | `(TILED, TILED)` | output shape is exactly the degree. |

The reindexings are projections out of `(i, j)`:

```text
eta_W(i, j) = i
eta_X(i, j) = j
```

Here `eta_W` is the reindexing for the `W` input and `eta_X` is the reindexing
for the `X` input. The argument `(i, j)` denotes one coordinate of the degree
`P = (i, j)`.

At each output coordinate `(i, j)`, the base operation sees:

```text
W[i, k]
X[k, j]
```

and contracts over `k`.

This shape is asserted directly in
[`tests/test_tensor_logic.py`](../tests/test_tensor_logic.py):

```python
assert br.input_weaves[0]._shape == (WeaveMode.TILED, k)
assert br.input_weaves[1]._shape == (k, WeaveMode.TILED)
assert all(p is WeaveMode.TILED for p in br.output_weaves[0]._shape)
```

---

## How TensorLogic Builds Weaves

The TensorLogic path is implemented in
[`data_structure/TensorLogic.py`](../data_structure/TensorLogic.py), especially
`TensorEquation.bc_signature()`.

TensorLogic uses **axis UID identity**:

- passing the same `Axis` object in multiple positions means the same index;
- an axis whose UID appears on the LHS is retained;
- an axis whose UID appears only on the RHS is contracted.

The central translation is:

```python
tuple(
    WeaveMode.TILED if ax.uid in retained_uid_to_pos else ax
    for ax in _factor_axes(factor)
)
```

That is:

```text
RHS axis appears on LHS     -> TILED
RHS axis absent from LHS    -> concrete Axis
```

In the code snippet, `ax` is one axis found in an RHS factor, and
`retained_uid_to_pos` maps each retained/LHS axis UID to its position in the
degree.

The output weave is all `TILED`:

```python
output_weave = Weave(
    out_dt,
    tuple(WeaveMode.TILED for _ in degree),
)
```

because the output shape is exactly the retained degree.

For:

```python
tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
```

TensorLogic produces:

```text
degree        = (i, j)
input_weaves  = ((TILED, k), (k, TILED))
output_weaves = ((TILED, TILED),)
```

Here `tl` is a `TL` builder, and `i`, `j`, and `k` are `Axis` objects.

---

## How String Signatures Build Weaves

The older string-signature path is used by operators such as `Einops`.

The path is:

```text
Operators.Einops.template()
  -> construction_helpers.einops.signature_to_buckets()
  -> construction_helpers.einops.bucketed_to_broadcast()
```

In `bucketed_to_broadcast()`, positions are classified using integer buckets:

```python
axes[index] if 0 <= index else WeaveMode.TILED
```

The convention is:

- negative indices represent output/degree axes and become `TILED`;
- non-negative indices represent absorbed/contracted axes and remain concrete.

Here `index` is an integer label assigned to one symbolic axis name in the
signature parser, and `axes` maps those integer labels to concrete `RawAxis`
objects.

So the string route and TensorLogic route converge on the same structure:

```python
Broadcasted(
    operator=...,
    input_weaves=...,
    output_weaves=...,
    reindexings=...,
)
```

---

## Lift and Batch Axes

Batch lifting extends an operation over new axes without changing what the base
operation computes at a single coordinate.

The implementation is in
[`construction_helpers/lift.py`](../construction_helpers/lift.py):

```python
input_weaves = tuple(
    Weave(
        weave.datatype,
        (WeaveMode.TILED,) * len(lift_by_morphism.cod()) + weave._shape
    )
    for weave in base.input_weaves
)
```

The same happens for output weaves.

So lifting a local operation over a batch shape prepends `TILED` slots:

```text
original weave:   (d_in)
lift over (b, s): (TILED, TILED, d_in)
```

Here `d_in` is a local feature/input axis, while `b` and `s` are the new batch
or sequence axes supplied by the lift.

This exactly matches the paper's batch-lift law:

```text
[f, P] ; [Y, p] = [X, p] ; f
```

Here:

- `f : X -> Y` is the original, unlifted morphism;
- `X` is the domain object of `f`;
- `Y` is the codomain object of `f`;
- `P` is the shape being lifted over;
- `p` is one coordinate of `P`;
- `[X, p]` and `[Y, p]` are slice/index morphisms selecting the `p`-th slice.

Slicing after the lifted operation is the same as slicing first and then running
the original operation. In other words, the lifted operation has no interaction
between different coordinates of `P`.

---

## Compilation

The compiler reads weaves to generate tensor code.

For TensorLogic equations, `generate_tensor_equation_signature()` in
[`torch_compile/torch_compile.py`](../torch_compile/torch_compile.py) builds an
`einops.einsum` signature from the weave structure.

The rule is:

- `TILED` slots become degree tags like `y0`, `y1`, ...
- concrete axes become local/contraction tags like `x0`, `x1`, ...
- shared concrete axis UIDs get the same tag, causing einsum contraction.

The names `y0`, `y1`, ... are compiler-generated labels for degree axes; the
names `x0`, `x1`, ... are compiler-generated labels for concrete target axes
that appear in input weaves. They are not user tensor names.

The key loop is:

```python
for weave in target.input_weaves:
    for slot in weave._shape:
        if not isinstance(slot, WeaveMode):
            uid = slot.uid
            if uid not in contracted_tag:
                contracted_tag[uid] = f'x{tag_counter}'
                tag_counter += 1
```

Here `target` is the `Broadcasted` morphism being compiled, `weave` ranges over
its input weaves, `slot` ranges over one position in `weave._shape`, `uid` is the
identity of a concrete axis, and `contracted_tag` maps axis UIDs to einsum tag
strings.

Then each input segment is formed by:

```python
weave.imprint_axes(
    degree_tags_for_this_input,
    contracted_tags_for_concrete_axes,
)
```

For matrix multiply, this produces the usual einsum shape:

```text
... y0 x0, ... x0 y1 -> ... y0 y1
```

where:

- `y0` is `i`;
- `y1` is `j`;
- `x0` is `k`;
- `...` is the einops ellipsis, used for any extra leading batch dimensions not
  named explicitly by pyncd.

For simpler broadcastable operators, [`torch_compile/bcast.py`](../torch_compile/bcast.py)
uses the same weave information to determine `torch.vmap` locations and semantic
broadcasting reshapes.

---

## Datatypes and Iverson Predicates

A weave also carries a datatype:

```python
Weave(Reals(), ...)
Weave(Bool(), ...)
Weave(Natural(...), ...)
```

This is independent of the target/tiling classification. The same `_shape`
pattern can carry real-valued data, boolean masks, or natural-valued selections.

TensorLogic uses this for predicates and Iverson brackets.

An inline predicate such as:

```python
q <= x
```

where `q` and `x` are axes, becomes an `IversonBinOp` in the TensorLogic AST.
During `bc_signature()`, an unsized predicate factor becomes a Bool-typed input
weave:

```python
Weave(
    Bool(),
    tuple(WeaveMode.TILED if ax.uid in retained_uid_to_pos else ax
          for ax in _factor_axes(factor)),
    iverson_expr=_serialize_iverson(factor),
)
```

In this snippet, `factor` is the Iverson predicate factor, such as the AST for
`q <= x`; `_factor_axes(factor)` lists the axes mentioned by that predicate; and
`iverson_expr` stores a display string such as `"q <= x"`.

Sized Iverson factors are auto-materialized as constant buffers at compile time
and omitted from the morphism's domain. Unsized Iversons remain Bool input
weaves, so the caller must supply the mask tensor.

Downstream code reads `weave.datatype` per weave. For example,
`ConstructedTensorEquation` checks the output weave datatype:

```python
self.demote = isinstance(target.output_weaves[0].datatype, Bool)
```

When the output is Bool, the real-valued contraction result is thresholded with
`H(r) = (r > 0)`, where `r` is the numeric contraction result.

---

## Mental Model

Read a weave slot-by-slot:

```text
Concrete Axis  = this dimension belongs to the base operation.
TILED          = this dimension belongs to the broadcast loop.
Reindexing     = how the shared loop coordinate fills this input's TILED slots.
Degree         = the common loop space shared by all inputs and outputs.
```

A `Broadcasted` morphism is therefore:

```text
base operator
+ input/output weave patterns
+ one reindexing per input
= a full tensor operation
```

This gives pyncd a compact representation that is simultaneously:

- categorical, because it is a morphism in **Br**;
- executable, because the compiler can derive `einsum`, `vmap`, or reshape code;
- compositional, because `dom()` and `cod()` recover full array types;
- visualizable, because each wire bundle knows its datatype and axis structure.

The weave is the bridge between tensor notation and tiled execution. It says how
a large tensor operation decomposes into small local operations while preserving
enough structure for composition, alignment, rendering, and compilation.

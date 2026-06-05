<!-- markdownlint-disable MD024 -->
# Tensor Logic DSL — Worked Examples

Each example is shown five ways:

1. **Mathematical description** — the tensor-logic / einsum form.
2. **PyRel** — the equivalent [PyRel](https://docs.relational.ai) program. PyRel is a
   Datalog extension that subsumes tensor logic; tensors are relations and contraction is
   `sum(product).per(output_indices)`. See [tensor_logic_in_pyrel.md](../papers/tensor_logic_in_pyrel.md).
3. **TL DSL** — the equivalent `data_structure/TensorDSL.py` program (runnable as-is).
4. **Visualization** — the string diagram. pyncd's text renderer (`display.print_category`)
   draws the same node graph that the tsncd TypeScript frontend renders as an interactive
   SVG; the ASCII form is shown here, with a note on launching the live graphical view.
5. **Generated PyTorch** — pyncd does **not** emit `.py` source; `ConstructedModule.construct`
   builds an `nn.Module`. The faithful artifacts are that module's structure (`repr`), the
   `einops.einsum` signature strings its `forward` runs, and its buffers/parameters.

All captured output below is real (produced by running the code, not hand-written).

For the DSL itself see [tensor_logic_dsl.md](tensor_logic_dsl.md); for the Boolean
semiring / rendering see [bool_semiring_extension.md](bool_semiring_extension.md).

---

## Example 1 — Basic Factor Graph Contraction

A factor graph over five binary factors with the topology:

```text
A – B – C
    |
    D – E
```

Factors A and E are rank-2; B, C, and D are rank-3. Shared variables — the edges — are
contracted; four external variables (`a`, `c_1`, `c_2`, `e`) are retained.

### 1. Mathematical description

Label the variables: `x` is shared by A and B; `y` is shared by B, C, and D; `w` is
shared by B and D; `z` is shared by D and E.

$$
\text{Result}[a, c_1, c_2, e] = A[a, x]\; B[x, y, w]\; C[y, c_1, c_2]\; D[y, w, z]\; E[z, e]
$$

`x`, `y`, `w`, `z` are absent from the left-hand side and therefore contracted; `a`, `c_1`, `c_2`, `e`
appear on both sides and are retained.

### 2. PyRel

Each factor is declared as a `Relationship`; shared index variables are unified across
patterns in `where(...)`. All four contracted indices are summed by `.per(a, c1, c2, e)`.

```python
from relationalai.semantics import Model, Float, Integer, where
from relationalai.semantics.std.aggregates import sum

model = Model("factor_graph")

A      = model.Relationship(f"{Integer:a} {Integer:x} {Float:val}")
B      = model.Relationship(f"{Integer:x} {Integer:y} {Integer:w} {Float:val}")
C      = model.Relationship(f"{Integer:y} {Integer:c1} {Integer:c2} {Float:val}")
D      = model.Relationship(f"{Integer:y} {Integer:w} {Integer:z} {Float:val}")
E      = model.Relationship(f"{Integer:z} {Integer:e} {Float:val}")
Result = model.Concept("result")

a, x, y, w, z = Integer.ref(), Integer.ref(), Integer.ref(), Integer.ref(), Integer.ref()
c1, c2, e = Integer.ref(), Integer.ref(), Integer.ref()
va, vb, vc, vd, ve = Float.ref(), Float.ref(), Float.ref(), Float.ref(), Float.ref()

where(
    A(a, x, va), B(x, y, w, vb), C(y, c1, c2, vc), D(y, w, z, vd), E(z, e, ve)
).define(
    Result.new(a=a, c1=c1, c2=c2, e=e, val=sum(va*vb*vc*vd*ve).per(a, c1, c2, e))
)
```

### 3. TL DSL

```python
import torch
from data_structure.TensorDSL import TL, real_axis
from torch_compile.torch_compile import ConstructedModule

a  = real_axis('a', 4)
c1 = real_axis('c_1', 2)
c2 = real_axis('c_2', 2)
e  = real_axis('e', 3)
x  = real_axis('x', 2)   # shared: A–B
y  = real_axis('y', 2)   # shared: B–C and B–D
w  = real_axis('w', 2)   # shared: B–D
z  = real_axis('z', 3)   # shared: D–E

tl = TL()
tl.A.tensor(a, x)
tl.B.tensor(x, y, w)
tl.C.tensor(y, c1, c2)
tl.D.tensor(y, w, z)
tl.E.tensor(z, e)
tl.Result[a, c1, c2, e] = tl.A[a, x] * tl.B[x, y, w] * tl.C[y, c1, c2] * tl.D[y, w, z] * tl.E[z, e]

morph = tl.to_morphism()
module = ConstructedModule.construct(morph)

A_t = torch.randn(4, 2)      # a  x
B_t = torch.randn(2, 2, 2)   # x  y  w
C_t = torch.randn(2, 2, 2)   # y  c1 c2
D_t = torch.randn(2, 2, 3)   # y  w  z
E_t = torch.randn(3, 3)      # z  e
out = module(A_t, B_t, C_t, D_t, E_t)
out = out[0] if isinstance(out, tuple) else out
assert out.shape == (4, 2, 2, 3)
```

### 4. Visualization

Read the diagram left-to-right. Five input factor groups enter from the left, each
labelled by their two axes. The **`Σ`** box contracts all shared variables simultaneously,
producing the four retained output wires `a(4)`, `c_1(2)`, `c_2(2)`, `e(3)`. Dashed separators
delimit the five factor inputs.

![Factor graph string diagram](factor_graph_render.png)

### 5. Generated PyTorch

The whole contraction compiles to a single `ConstructedTensorEquation`:

```text
ConstructedThreadedComposed(
  (chain): ModuleList(
    (0): ConstructedTensorEquation()
  )
)
```

Its `forward` runs one `einops.einsum` call whose signature reflects the five factors and
four retained axes:

```python
module.chain[0].signature
# '... y0 x0, ... x0 x1 x2, ... x1 y1 y2, ... x1 x2 x3, ... x3 y3 -> ... y0 y1 y2 y3'
# i.e.  a  x ,   x   y   w ,   y  c1 c2  ,   y   w   z ,   z   e   ->  a  c1 c2  e
```

The module has no learned parameters — all five factors are caller inputs.

---

## Example 2 — Multi-Layer Perceptron

A two-layer MLP (one hidden layer with a ReLU): project the input up to a hidden
dimension, apply ReLU, project back down.

### 1. Mathematical description

With batch/position index `q`, model dimension `d`, and hidden dimension `f`,
in **tensor-logic (Einstein) notation** — a repeated index on the right that
does not appear on the left is implicitly summed:

$$
H[q, f] = \mathrm{relu}\!\big( W_{\text{in}}[f, d]\; X[q, d] \big)
\qquad
\mathrm{Out}[q, d] = W_{\text{out}}[d, f]\; H[q, f]
$$

So `d` (repeated on the right, absent on the left of the first equation) is
contracted, and likewise `f` in the second; the left-hand indices are retained.
This implicit-summation convention is exactly the DSL's rule — contraction is
read from axis (UID) identity, never written as an explicit `Σ`.

### 2. PyRel

Two rules, one per equation. Shared index `d` is contracted in the first rule via
`.per(q, f)`; shared `f` is contracted in the second. The intermediate `H` is a
`Concept` populated by the first rule and queried by the second.

```python
from relationalai.semantics import Model, Float, Integer, where
from relationalai.semantics.std.aggregates import sum
from relationalai.semantics.std.math import relu

model = Model("mlp")

# W_in and W_out are caller-supplied relations here; PyRel has no notation
# for learned relations — new syntax would be needed to express the
# .linear() form where weights are parameters rather than inputs.
W_in  = model.Relationship(f"{Integer:f} {Integer:d} {Float:val}")
W_out = model.Relationship(f"{Integer:d} {Integer:f} {Float:val}")
X     = model.Relationship(f"{Integer:q} {Integer:d} {Float:val}")
H     = model.Concept("H")
Out   = model.Concept("Out")

q, d, f = Integer.ref(), Integer.ref(), Integer.ref()
vw, vx, vh = Float.ref(), Float.ref(), Float.ref()

where(W_in(f, d, vw), X(q, d, vx)).define(
    H.new(q=q, f=f, val=relu(sum(vw*vx).per(q, f)))
)

where(W_out(d, f, vw), H(q, f, vh)).define(
    Out.new(q=q, d=d, val=sum(vw*vh).per(q, d))
)
```

### 3. TL DSL

We build it from `.linear()` layers (§4 of [tensor_logic_dsl.md](tensor_logic_dsl.md)):
each weight is declared a Linear layer, so the weight-multiplies become `Linear`
operators whose weights are the layers' **internal parameters** — you pass only `X`.

```python
import torch
from data_structure.TensorDSL import TL, real_axis, relu
from torch_compile.torch_compile import ConstructedModule

q   = real_axis('q', 2)      # batch / sequence position
d   = real_axis('d', 4)      # model dimension
dff = real_axis('dff', 8)    # hidden dimension

tl = TL()
tl.W_in.linear(out_axes=(dff,), in_axes=(d,))     # Linear layer  d -> dff
tl.W_out.linear(out_axes=(d,), in_axes=(dff,))    # Linear layer  dff -> d
tl.H[q, dff] = relu(tl.W_in[dff, d] * tl.X[q, d])
tl.Out[q, d] = tl.W_out[d, dff] * tl.H[q, dff]

morph = tl.to_morphism()                  # ThreadedComposed (two layers, H threaded)
module = ConstructedModule.construct(morph)

out = module(torch.randn(2, 4))           # only X; weights live inside as parameters
out = out[0] if isinstance(out, tuple) else out
assert out.shape == (2, 4)
```

> **Alternative — explicit contraction (weights as data).** Drop the two `.linear()`
> declarations and the *same* equations express plain tensor contractions, where
> `W_in`/`W_out` are ordinary input tensors you supply: `module(W_in, X, W_out)`. That
> form draws `Σ` (einsum) boxes with the weights as input wires instead of `L` boxes —
> the tensor-logic philosophy of weights-as-explicit-data. `.linear()` is the opt-in
> that reframes a weight as a trainable layer.

### 4. Visualization

Read the diagram left-to-right. Wires are labelled `axis(size)` and carry tensors
between boxes. **`L`** boxes (subscripted by weight name) are learned linear layers.
**`▶relu▶`** is a pointwise nonlinearity. The
upper wire (`q`) is the batch dimension, which passes through unchanged; the lower wire
(`d`/`dff`) is the feature dimension transformed by each layer.

![MLP string diagram](mlp_render.png)

### 5. Generated PyTorch

The compiled module is a `ThreadedComposed` of the two layers; the ReLU is folded into
the first as a `Lambda`. Each layer is a `ConstructedLinear` wrapping a `Multilinear`
(the learned weight):

```text
ConstructedThreadedComposed(
  (chain): ModuleList(
    (0): ConstructedComposed(
      (chain): Sequential(
        (0): ConstructedLinear(
          (module): Multilinear((4,) -> (8,) (weights): Weights(size=(4, 8)))
        )
        (1): Lambda(ReLU(name=DynamicName(body='\\mathrm{relu}', ...), operator='relu'))
      )
    )
    (1): ConstructedLinear(
      (module): Multilinear((8,) -> (4,) (weights): Weights(size=(8, 4)))
    )
  )
)
```

Unlike the explicit-contraction form, the weights here **are learned parameters**, not
caller inputs — one per layer:

```python
[(n, tuple(p.shape)) for n, p in module.named_parameters()]
# [('chain.0.chain.0.module.weights.weight', (4, 8)),   # W_in : d -> dff
#  ('chain.1.module.weights.weight',         (8, 4))]   # W_out: dff -> d
```

(`bias=True` on a `.linear()` declaration adds a matching bias parameter; multi-axis
feature blocks like `out_axes=(h, k)` give a weight whose shape is `in_size + out_size`.)

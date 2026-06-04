# Tensor Logic DSL — Worked Examples

Each example is shown four ways:

1. **Mathematical description** — the tensor-logic / einsum form.
2. **TL DSL** — the equivalent `data_structure/TensorDSL.py` program (runnable as-is).
3. **Visualization** — the string diagram. pyncd's text renderer (`display.print_category`)
   draws the same node graph that the tsncd TypeScript frontend renders as an interactive
   SVG; the ASCII form is shown here, with a note on launching the live graphical view.
4. **Generated PyTorch** — pyncd does **not** emit `.py` source; `ConstructedModule.construct`
   builds an `nn.Module`. The faithful artifacts are that module's structure (`repr`), the
   `einops.einsum` signature strings its `forward` runs, and its buffers/parameters.

All captured output below is real (produced by running the code, not hand-written).

For the DSL itself see [tensor_logic_dsl.md](tensor_logic_dsl.md); for the Boolean
semiring / rendering see [bool_semiring_extension.md](bool_semiring_extension.md).

---

## Example 1 — Multi-Layer Perceptron

A two-layer MLP (one hidden layer with a ReLU): project the input up to a hidden
dimension, apply ReLU, project back down.

### 1. Mathematical description

With batch/position index `q`, model dimension `d`, and hidden dimension `f`:

$$
H[q, f] = \mathrm{relu}\!\left( \sum_{d} W_{\text{in}}[f, d]\; X[q, d] \right)
\qquad
\mathrm{Out}[q, d] = \sum_{f} W_{\text{out}}[d, f]\; H[q, f]
$$

`d` is contracted in the first equation (summed away); `f` is contracted in the
second. Everything else is retained on the left.

### 2. TL DSL

```python
import torch
from data_structure.TensorDSL import TL, real_axis, relu
from torch_compile.torch_compile import ConstructedModule

q   = real_axis('q', 2)      # batch / sequence position
d   = real_axis('d', 4)      # model dimension
dff = real_axis('dff', 8)    # hidden dimension

tl = TL()
tl.H[q, dff] = relu(tl.W_in[dff, d] * tl.X[q, d])   # d contracted, then ReLU
tl.Out[q, d] = tl.W_out[d, dff] * tl.H[q, dff]      # dff contracted

morph = tl.to_morphism()                  # ThreadedComposed (two steps, H threaded)
module = ConstructedModule.construct(morph)
```

Running it (external inputs are passed in first-appearance order — `W_in`, `X`,
`W_out`; the intermediate `H` is threaded internally):

```python
X     = torch.randn(2, 4)
W_in  = torch.randn(8, 4)
W_out = torch.randn(4, 8)

out = module(W_in, X, W_out)
out = out[0] if isinstance(out, tuple) else out
assert out.shape == (2, 4)
# equivalent to the reference computation:
ref = (X @ W_in.T).clamp(min=0) @ W_out.T
assert torch.allclose(out, ref, atol=1e-5)
```

### 3. Visualization

`display.print_category(morph)` renders the string diagram as text. Read it
left-to-right: the first `TensorEquation` (einsum) contracts `d`, the `Elementwise`
box applies the nonlinearity, and the second `TensorEquation` contracts `dff`. Left-edge
wires are inputs labelled `axis-size` with their datatype anchor (`Re` = ℝ); `~~~`
marks a contracted (summed) axis, `─<n` a tiled/retained wire, and `( … )` groups the
output degree.

```text
8 ─<1                                                    4 ─<1
4 ~~~                                                    8 ~~~
Re           (2          ── 2 2 ─<0       (2        ── 2 Re           (2          ── 2
-----         8)         ── 8 8 ─<1        8)       ── 8 -----         4)         ── 4
2 ─<0 > TensorEquation >   Re Re    > Elementwise >   Re 2 ─<0 > TensorEquation >   Re
4 ~~~                                                    8 ~~~
Re                                                       Re
```

> The tsncd frontend draws this same node graph as an interactive Canvas/SVG (pan,
> zoom, KaTeX labels). To view it live: run `uv run run_server.py` (websocket on
> `ws://localhost:8765`), start the tsncd frontend (`cd ../tsncd && npm run dev`), and
> send the morphism with `await websocket_transfer…send_term(morph)`. The ASCII above
> is the text form of that diagram.

### 4. Generated PyTorch

The compiled module is a `ThreadedComposed` of the two equation steps, with the ReLU
folded into the first step as a `Lambda`:

```text
ConstructedThreadedComposed(
  (chain): ModuleList(
    (0): ConstructedComposed(
      (chain): Sequential(
        (0): ConstructedTensorEquation()
        (1): Lambda(Elementwise(name=DynamicName(body='\\sigma', subscript=None, settings=None), operator='sigmoid'))
      )
    )
    (1): ConstructedTensorEquation()
  )
)
```

> The nonlinearity node reprs as `Elementwise(operator='sigmoid')` — that string is
> the operator template's default label. The *executed* function is whatever is
> registered for `ops.Elementwise`, which is `torch.relu`
> (`ConstructedModule.add_function(ops.Elementwise, torch.relu)`); the assertion above
> confirms ReLU semantics.

Each `ConstructedTensorEquation.forward` runs one `einops.einsum`. The signature
strings (the actual generated tensor program) are identical for both layers — a
matmul contracting one axis:

```python
# chain.0.chain.0  (H = W_in · X, contracting d)
einops.einsum(W_in, X, '... y1 x0, ... y0 x0 -> ... y0 y1')
# chain.1          (Out = W_out · H, contracting dff)
einops.einsum(W_out, H, '... y1 x0, ... y0 x0 -> ... y0 y1')
```

Here `x0` is the contracted axis (`d`, then `dff`) and `y0 y1` are the retained
output axes; the leading `...` carries any batch dimensions.

No buffers and no parameters are registered:

```python
list(module.named_buffers())     # []
list(module.named_parameters())  # []
```

This reflects a core design choice: **weights are inputs, not learned parameters.**
`W_in` and `W_out` are passed to `forward` like any other tensor rather than stored as
`nn.Parameter`s. (Pre-materialised Iverson predicate masks would appear as buffers, and
`ops.Linear`/`ops.Embedding` layers would introduce parameters — neither occurs in this
pure-contraction MLP.)

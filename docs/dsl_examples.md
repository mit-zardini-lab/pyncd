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

### 2. TL DSL

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

### 3. Visualization

`display.print_category(morph)` renders the string diagram as text. Read it
left-to-right: `X[q,d]` enters the first **`Linear`** box (`d → dff`), then `ReLU`,
then the second `Linear` box (`dff → d`). Left-edge wires are labelled `axis-size`
with their datatype anchor (`Re` = ℝ); `~~~` marks the feature axis the layer
transforms, `─<n` a tiled/batch wire, and `( … )` groups the output degree.

```text
2 ─<0    (2)     ── 2 2 ─<0   (2     ── 2 2 ─<0    (2)     ── 2
4 ~~~ > Linear > ~~ 8 8 ─<1    8)    ── 8 8 ~~~ > Linear > ~~ 4
Re                 Re Re    > ReLU >   Re Re                 Re
```

> The tsncd frontend draws this same node graph as an interactive Canvas/SVG (pan,
> zoom, KaTeX labels — wires labelled by axis name and size, `L` boxes subscripted by
> the weight name). To view it live: point `tsncd/src/data_transfer/json.ts` at a JSON
> exported with `data_transfer.json.TermJSONConverter.export_to_json(morph)`, then
> `cd ../tsncd && npm run dev`. The ASCII above is the text form of that diagram.

### 4. Generated PyTorch

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
        (1): Lambda(ReLU(name=DynamicName(body='\\mathrm{relu}', ...), operator='sigmoid'))
      )
    )
    (1): ConstructedLinear(
      (module): Multilinear((8,) -> (4,) (weights): Weights(size=(8, 4)))
    )
  )
)
```

> The nonlinearity reprs as `ReLU(..., operator='sigmoid')` — the `'sigmoid'` operator
> string is a vestigial default; the *executed* function is `torch.relu`, registered for
> `ops.ReLU` (`ConstructedModule.add_function(ops.ReLU, torch.relu)`), and the box
> renders its name `\mathrm{relu}`.

Unlike the explicit-contraction form, the weights here **are learned parameters**, not
caller inputs — one per layer:

```python
[(n, tuple(p.shape)) for n, p in module.named_parameters()]
# [('chain.0.chain.0.module.weights.weight', (4, 8)),   # W_in : d -> dff
#  ('chain.1.module.weights.weight',         (8, 4))]   # W_out: dff -> d
```

(`bias=True` on a `.linear()` declaration adds a matching bias parameter; multi-axis
feature blocks like `out_axes=(h, k)` give a weight whose shape is `in_size + out_size`.)

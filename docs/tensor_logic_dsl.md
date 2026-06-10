# The Tensor Logic DSL

**Status:** Reference for the Python tensor-logic DSL (`data_structure/TensorDSL.py`) and its compilation to executable PyTorch modules (`torch_compile/`).

The tensor logic DSL is a declarative interface for writing tensor contractions —
matmuls, attention, normalisations, masked reductions, and recurrences — as Python
assignment statements, then compiling them to `torch.nn.Module`s. Contraction
structure is read from **axis identity**, not from an einsum string: an axis object
that appears on both sides of an equation is retained; one that appears only on the
right is summed over.

Every code block below is runnable as-is (they are lifted from the test suite and a
verification script). Imports used throughout:

```python
from data_structure.TensorDSL import (
    TL, axes, real_axis, norm_axis, nat_axis, NormAxis, NatAxis,
    relu, softmax, normalize, ieq, imul, iabs, inot,
)
from data_structure.TensorExpr import IversonConst
from data_structure.Numeric import Integer
from torch_compile.torch_compile import ConstructedModule
from torch_compile.materialise import materialise_iverson
import torch
```

---

## Contents

1. [Quick start](#1-quick-start)
2. [Axes](#2-axes)
3. [Tensors and equations](#3-tensors-and-equations)
4. [Declarations](#4-declarations)
5. [Predicates (Iverson brackets)](#5-predicates-iverson-brackets)
6. [Normalisations and nonlinearities](#6-normalisations-and-nonlinearities)
7. [Iteration and recurrence](#7-iteration-and-recurrence)
8. [Index arithmetic](#8-index-arithmetic)
9. [Compilation and execution](#9-compilation-and-execution)
10. [Worked example: a small attention block](#10-worked-example-a-small-attention-block)
11. [Further reading](#11-further-reading)

---

## 1. Quick start

Matrix multiply, end to end:

```python
tl = TL()
i, j, k = axes('i j k')
tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]      # k appears only on the RHS → contracted

module = ConstructedModule.construct(tl.bc_signature())
W, X = torch.randn(2, 3), torch.randn(3, 4)
result = module(W, X)                       # inputs in first-appearance order: W, X
out = result[0] if isinstance(result, tuple) else result
assert torch.allclose(out, W @ X, atol=1e-5)
```

The three stages — **define axes → write equations on a `TL` registry → compile and
call** — are the whole workflow. The rest of this document fills in each piece.

---

## 2. Axes

Axes are the index variables of an equation. Identity is by object (UID): the same
Python axis object used in two places means the same index. Four constructors:

| Constructor | Returns | Use |
|---|---|---|
| `axes('i j k')` | tuple of `RawAxis` (free size) | quick, unsized index variables |
| `real_axis('d', 512)` | `RawAxis` (optional size) | named axis, typically a feature dimension |
| `norm_axis('x', N)` | `NormAxis` | the axis a softmax/normalize reduces over |
| `nat_axis('t', 50000)` | `NatAxis` | named axis, typically a discrete index dimension |

```python
i, j, k = axes('i j k')          # variadic-in-one-string
i, j, k = axes('i', 'j', 'k')    # or variadic args
(d_ff,) = axes('d_{ff}')         # LaTeX-style names are fine

d = real_axis('d', 512)          # concrete size 512  → d._size == Integer(512)
d = real_axis('d')               # free size          → d._size is FreeNumeric
```

**Sizes matter for three features:** auto-materialising predicates ([§5](#5-predicates-iverson-brackets)), iteration
([§7](#7-iteration-and-recurrence)), and index-arithmetic range validation ([§8](#8-index-arithmetic)) all require concrete integer sizes on
the relevant axes. Plain contractions do not — `axes(...)` is enough.

**`norm_axis`** marks the normalisation dimension on the *left-hand side* of a
softmax/normalize equation. It also lets the compiler drop additive terms that are
constant along that axis (`softmax(f + c) == softmax(f)` when `c` does not depend on
the norm axis).

**`nat_axis`** marks an index dimension.

**On the `real_` and `nat_` qualifiers.** Despite the names, neither `real_axis`
nor `nat_axis` attaches a datatype to the axis. Datatypes (`Reals()`, `Natural()`,
`Bool()`) live on `Weave` objects and are determined by tensor declarations
(`.predicate()`, `.linear()`, etc.), not by axis constructors. The qualifiers are
advisory labels: `real_axis` is identical to `axes()` except for accepting a name
and optional concrete size; `nat_axis` produces a `NatAxis` subclass that the
compiler treats as a plain `RawAxis` for all purposes except acset serialisation
metadata. The distinction is useful as documentation — ℝ feature dimensions vs.
ℕ index dimensions — but the compiler does not enforce it.

---

## 3. Tensors and equations

A tensor name is any attribute access on a `TL` instance — `tl.W` returns a
`TensorProxy`. Subscripting gives an `IndexedTensor`; assignment captures an equation
into the registry.

```python
tl = TL()
i, j, k = axes('i j k')
tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
```

**Multiplication `*`** accumulates factors (a contraction):

```python
expr = tl.A[a, b] * tl.B[b, c] * tl.C[c, d]   # three-factor chain, RHSExpression
```

**Addition `+`** forms a sum of terms; each term compiles to its own einsum and the
results are added:

```python
tl.Out[q, dm] = tl.W2[dm, dh] * tl.Y[q, dh] + tl.H[q, dm]   # contraction + residual
```

**Contraction structure from UID identity.** Axes in the left-hand index list are
*retained* (they appear in the output); axes that occur only in `rhs` factors are
*contracted* (summed). The same axis object carries the same UID everywhere it is
used, which is how shared indices are matched:

```python
tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
eq = tl.to_equation()
# k is the same object in W's 2nd slot and X's 1st slot:
assert eq.rhs[0].axes[1].uid == eq.rhs[1].axes[0].uid    # contracted
```

---

## 4. Declarations

Declarations are optional metadata attached to a tensor name *before* it is used.
They enforce arity, control datatype, drive axis promotion, or pick a different
operator. All return the proxy for chaining.

```python
i = real_axis('i', 64); k = real_axis('k', 64); j = real_axis('j', 64)
tl = TL()
tl.W.tensor(i, k)            # ordinary ℝ-valued tensor
tl.Mask.predicate(i, j)      # 𝔹-valued predicate (output typed Bool)
tl.Wq.linear(out_axes=(i,), in_axes=(k,))                    # a Linear layer weight
```

- **`.tensor(*shape)`** — default contraction semantics, no promotion.
- **`.predicate(*shape)`** — marks the name `Bool`-typed. When such a tensor is the
  *output* of an equation, the einsum result is demoted with the Heaviside step
  $H(x)=\mathbf 1[x>0]$ to `{0,1}` (∃/∧ semantics — see [§6](#6-normalisations-and-nonlinearities)). Axes are not promoted.
- **`.linear(*, out_axes, in_axes, bias=False)`** — marks the name as the weight of a
  **Linear/affine layer**. When it multiplies an activation in an equation, the
  contraction compiles to an `ops.Linear` operator (an `L` box) instead of an einsum:
  the weight becomes the layer's **internal parameter** (so it is *not* a caller
  input — only the activation is), and `bias=True` makes it affine (`Wx + b`).
  `out_axes`/`in_axes` are the output/input feature blocks and may each be multi-axis,
  e.g. a QKV projection `tl.Wq.linear(out_axes=(h, k), in_axes=(d,))` used as
  `tl.Q[q, h, k] = tl.Wq[h, k, d] * tl.X[q, d]` maps `d → (h, k)`. See
  [dsl_examples.md](dsl_examples.md) Example 1 for an MLP built from `.linear()` layers.
- **`.scatter(*, fill=0.0, reduce=None)`** — configures output behaviour for a tensor
  whose LHS uses an affine expression (`tl.Y[2*i]`, `tl.Y[i+j]`). `fill` sets the
  value written to output coordinates the affine map does not reach (default `0`).
  `reduce='sum'` permits non-injective maps by accumulating overlapping writes; without
  it, a non-injective LHS is rejected at build time. See [§8](#8-index-arithmetic) for
  examples (custom fill, upsampling, convolution-coefficient scatter).
- **`.iteration_axis(l)`** — registers a tensor as iterative and records `l` as its
  recurrence axis. The iteration axis is a plain `real_axis` whose concrete integer size
  is the step count `N`; there is no special iteration-axis constructor. This call is
  required for coupled recurrences (where a step equation reads a sibling state that has
  not yet been written) and optional for uncoupled ones. It also gates the LHS syntax:
  without it (and without a base case), an `axis+int` write — `tl.Y[i+1]` — is
  reclassified as an offset scatter at finalize time. See [§7](#7-iteration-and-recurrence)
  for the full iteration model, calling convention, and coupled vs. uncoupled examples.

---

## 5. Predicates (Iverson brackets)

A predicate on indices — the Iverson bracket `[P]` — is a `{0,1}`-valued factor.
Comparison and arithmetic operators on axes build a predicate expression tree.

### Construction

```python
q, x, k = axes('q x k')

q < x        # IversonBinOp('<',  q, x)
q <= x       # IversonBinOp('<=', q, x)   — causal mask
q > x ; q >= x
ieq(q, x)    # equality  [q == x]   (== can't be overridden on axes; use ieq)
imul(q, x)   # arithmetic product inside a predicate
iabs(q - x)  # absolute value
q + 1 ; q - 2                       # index arithmetic

(q < x) & (x < k)   # conjunction (AND)
(q < x) | ieq(q, x) # disjunction (OR)
~(q < x)            # negation (NOT); also inot(q < x)
```

Numeric literals are wrapped with `IversonConst(Integer(n))`:

```python
banded = iabs(q - x) < IversonConst(Integer(2))   # |q - x| < 2  → tri-diagonal band
```

Boolean operators (`&`, `|`, `~`) give a complete set, so De Morgan rewrites are
expressible by hand. `~` is defined on predicates, sub-expressions, and axes.

### As a multiplicative mask

Multiply a predicate into an equation to gate a contraction. Future/excluded
positions are zeroed:

```python
tl = TL()
q, x = axes('q x')
tl.Attn[q, x] = tl.Score[q, x] * (q <= x)     # causal mask, Real output
```

### Materialisation

`materialise_iverson` evaluates a predicate (whose axes are all sized) to a `{0,1}`
float tensor:

```python
q = real_axis('q', 5); x = real_axis('x', 5)
materialise_iverson(iabs(q - x) < IversonConst(Integer(2)))
# tensor([[1,1,0,0,0],
#         [1,1,1,0,0],
#         [0,1,1,1,0],
#         [0,0,1,1,1],
#         [0,0,0,1,1]])

a = real_axis('a', 4); b = real_axis('b', 4)
materialise_iverson(~(a < b))      # 1 - [a<b]  == lower-triangular incl. diagonal
```

Operator semantics: comparisons → `{0,1}`; `&`/`|` → logical and/or; `~` → `1 - v`;
`abs`/`+`/`-`/`*` → arithmetic.

**Auto vs. caller-supplied.** When a predicate factor's axes are all sized, the
compiler materialises it once and stores it as a module buffer — the caller never
passes it. When axes are unsized, the caller must supply the `{0,1}` tensor in RHS
factor order:

```python
# Sized → auto-buffered; caller passes only Score
q = real_axis('q', 4); x = real_axis('x', 4)
tl = TL(); tl.Attn[q, x] = tl.Score[q, x] * (q <= x)
m = ConstructedModule.construct(tl.bc_signature())
assert '_mask_1' in dict(m.named_buffers())
assert torch.allclose(m(torch.ones(4, 4)), torch.triu(torch.ones(4, 4)))

# Unsized → caller supplies the mask
q, x = axes('q x')
tl = TL(); tl.Attn[q, x] = tl.Score[q, x] * (q <= x)
m = ConstructedModule.construct(tl.bc_signature())
m(torch.ones(4, 4), torch.tril(torch.ones(4, 4)))   # Score, Mask
```

> A predicate that repeats one axis (e.g. `(x > 3) | ieq(x, 3)`, with `x` twice)
> materialises in an *expanded* form — one dimension per leaf — and the einsum
> contracts the diagonal. This is a deliberate space/uniformity tradeoff documented in
> `materialise_iverson`'s docstring (`torch_compile/materialise.py`).

---

## 6. Normalisations and nonlinearities

Wrap an RHS expression with one of three functions to attach an operation. Each
accepts an `IndexedTensor`, an `RHSExpression` (a `*` chain), or a `SumExpr` (a `+`
sum).

| Function | Operation | Reduction axis |
|---|---|---|
| `relu(expr)` | elementwise `max(x, 0)` | — |
| `softmax(expr, where=None)` | `softmax` over the norm axis | norm axis / last dim |
| `normalize(expr, where=None)` | **sum-normalisation** `x / x.sum(dim)` | norm axis / last dim |

`normalize` is sum-normalisation (`x / x.sum(dim).clamp(min=1e-8)`), **not** LayerNorm.

### ReLU

```python
i, j, k = axes('i j k')
tl = TL(); tl.Y[i, j] = relu(tl.W[i, k] * tl.X[k, j])
m = ConstructedModule.construct(tl.to_morphism())
W, X = torch.randn(2, 3), torch.randn(3, 4)
out = m(W, X); out = out[0] if isinstance(out, tuple) else out
assert torch.allclose(out, (W @ X).clamp(min=0), atol=1e-5)
```

### Softmax

The reduction axis is the `norm_axis` on the LHS; with plain `real_axis` it defaults
to the last dimension. Rows over the reduced axis sum to 1:

```python
q = real_axis('q', 4); x = real_axis('x', 4)
tl = TL(); tl.Out[q, x] = softmax(tl.X[q, x])
m = ConstructedModule.construct(tl.bc_signature())
out = m(torch.randn(4, 4)); out = out[0] if isinstance(out, tuple) else out
assert torch.allclose(out.sum(-1), torch.ones(4), atol=1e-5)
```

### Normalize

```python
p = real_axis('p', 8); m_ax = real_axis('m', 16)
tl = TL(); tl.Out[p, m_ax] = normalize(tl.X[p, m_ax])
mod = ConstructedModule.construct(tl.bc_signature())
out = mod((torch.rand(8, 16) + 0.1)); out = out[0] if isinstance(out, tuple) else out
assert torch.allclose(out.sum(-1), torch.ones(8), atol=1e-5)
```

### Masked variants (`where=`)

A `where=` predicate gates the reduction itself (it is not a multiplicative factor of
the einsum). For **softmax**, excluded positions are set to $-\infty$ before
exponentiation (zero attention weight); for **normalize**, they are zeroed before the
sum, so they stay zero in the output. The norm axis must be sized.

```python
# Masked (causal) attention softmax
SEQ, H = 5, 2
q = real_axis('q', SEQ); h = real_axis('h', H); k = real_axis('k', 3); x = norm_axis('x', SEQ)
tl = TL()
tl.S[h, q, x] = softmax(tl.Q[q, h, k] * tl.K[x, h, k], where=(x <= q))
m = ConstructedModule.construct(tl.to_morphism())
S = m(torch.randn(SEQ, H, 3), torch.randn(SEQ, H, 3))   # Q, K

# Masked normalize: [(m > 2) | (m == 2)] == [m >= 2] keeps m=2,3,4
SEQ = 5
p = real_axis('p', SEQ); m_n = norm_axis('m', SEQ); two = IversonConst(Integer(2))
tl = TL()
tl.Out[p, m_n] = normalize(tl.X[p, m_n], where=((m_n > two) | ieq(m_n, two)))
mod = ConstructedModule.construct(tl.to_morphism())
out = mod(torch.ones(SEQ, SEQ)); out = out[0] if isinstance(out, tuple) else out
expected = torch.zeros(SEQ, SEQ); expected[:, 2:] = 1 / 3
assert torch.allclose(out, expected, atol=1e-6)
```

### Boolean output (∃/∧)

Declaring the LHS a `.predicate()` makes the equation a Boolean contraction: the
einsum runs in `{0,1}` arithmetic and the result is thresholded with the Heaviside
step to `{0,1}`. `Y(i,j) = ∃k: A(i,k) ∧ B(k,j)`:

```python
i, j, k = axes('i j k')
tl = TL()
tl.Out.predicate(i, j)
tl.Out[i, j] = tl.A[i, k] * tl.B[k, j]
m = ConstructedModule.construct(tl.bc_signature())
out = m(torch.ones(3, 4), torch.ones(4, 5)); out = out[0] if isinstance(out, tuple) else out
assert torch.all((out == 0) | (out == 1))
```

See [bool_semiring_extension.md](bool_semiring_extension.md) for the full Boolean
semiring story (the ι/H retraction, masked reductions, rendering).

---

## 7. Iteration and recurrence

A recurrence is written as two assignments: a **base case** (literal `0` in the
iteration slot) and a **step** (`l + 1` in that slot), where `l = real_axis('l', N)`
is the iteration axis with a concrete step count `N`.

> **The iteration axis is a plain `real_axis`** — there is no special iteration-axis
> constructor (the only constructors are `axes`, `real_axis`, `norm_axis`, `nat_axis`;
> `norm_axis` is for the softmax/normalize reduction dimension, [§6](#6-normalisations-and-nonlinearities), not iteration). The
> similarly named `tl.H.iteration_axis(l)` is a *method*, not a constructor: it
> *registers* `H` as iterative and records `l` as its recurrence axis so forward
> references resolve (required for coupled recurrences; optional for uncoupled ones when
> both the base and step are written before `to_morphism()`). It also explicitly marks
> the tensor as iterative — without it (and without a base case), an `axis+int` LHS is
> reclassified as an offset scatter at finalize time ([§8](#8-index-arithmetic)). The axis needs a concrete
> integer size because that size is the step count `N`.

### Uncoupled (single state)

```python
i = real_axis('i', 3)
l = real_axis('l', 4)                       # N = 4 steps
tl = TL()
tl.H[i, 0]     = tl.X[i]                     # base:  H[i,0] = X[i]
tl.H[i, l + 1] = tl.H[i, l] + tl.Delta[i, l] # step:  H[i,l+1] = H[i,l] + Delta[i,l]

m = ConstructedModule.construct(tl.to_morphism())
X = torch.tensor([1., 2., 3.])
Delta = torch.zeros(3, 4); Delta[0, :] = 1.0     # per-step input: N is the LAST dim
result = m(X, Delta)
H = result[0] if isinstance(result, tuple) else result
assert H.shape == (3, 5)                          # (*state, N+1): base + N steps
assert torch.allclose(H[0], torch.tensor([1., 2., 3., 4., 5.]))
```

**Calling convention.** Base-case inputs come first, then per-step inputs. Each
per-step input carries the step count `N` as its **last** dimension. The output has
shape `(*state, N+1)` — the base value plus one slice per step.

### Coupled (multiple states, Jacobi)

Several states that read each other update **simultaneously** — each step sees the
*old* value of every state. Pre-register the shared axis with `iteration_axis` so
forward references resolve:

```python
i = real_axis('i', 3); l = real_axis('l', 4)
tl = TL()
tl.H.iteration_axis(l); tl.G.iteration_axis(l)
tl.H[i, 0] = tl.X[i];   tl.G[i, 0] = tl.Y[i]
tl.H[i, l + 1] = tl.H[i, l] + tl.G[i, l]      # H reads old H and old G
tl.G[i, l + 1] = tl.G[i, l] * tl.H[i, l]      # G reads old G and old H

m = ConstructedModule.construct(tl.to_morphism())
# Inputs and outputs are in CANONICAL (name-sorted) order: G before H.
G_out, H_out = m(torch.tensor([1., 1., 1.]),   # Y  (G's base)
                 torch.tensor([1., 2., 3.]))   # X  (H's base)
```

Coupled scans return a **tuple** of states, each `(*state_k, N+1)`, in name-sorted
order; base and per-step inputs are likewise grouped per state in that order.

### Pre-built step morphism

When a step is too complex for one equation (e.g. a whole transformer layer built
from another `TL` session), register the morphism directly with `recur`. The step
**must be a morphism that takes the current state as its sole input and returns the
next state** (same shape); it is invoked once per step as `step(H)`, with no per-step
external inputs. The base case is still given by `tl.H[..., 0] = …`.

Here the step is a self-contained block built in its own `TL` session — its single
external tensor (`In`) is the state slot:

```python
def relu_step():
    """A state -> state step morphism: applies ReLU to the current state."""
    inner = TL()
    x, m = axes('x m')
    inner.Out[x, m] = relu(inner.In[x, m])   # 'In' is the (sole) state input
    return inner.to_morphism()

l = real_axis('l', 3)
x, m = real_axis('x', 2), real_axis('m', 2)
tl = TL()
tl.H[x, m, 0] = tl.X[x, m]            # base:  H[...,0] = X
tl.H.recur(l, relu_step())            # step:  H[...,l+1] = relu(H[...,l])

mod = ConstructedModule.construct(tl.to_morphism())
X = torch.tensor([[-1., 2.], [3., -4.]])
out = mod(X)
H = out[0] if isinstance(out, tuple) else out
assert H.shape == (2, 2, 4)                       # (*state, N+1)
assert torch.allclose(H[..., 0], X)               # base
assert torch.allclose(H[..., 1], X.clamp(min=0))  # one ReLU step
```

A real transformer layer follows the same contract: a pre-built morphism whose only
input is the state and whose weights are already bound inside it (so it exposes no
extra per-step inputs) — substitute it for `relu_step()` above.

### History slices

A downstream equation can read any fixed step of the scan output with a constant
index. The materialised history has shape `(*state, N+1)` (base at index 0, then N
steps), so valid indices are `0 ≤ c ≤ N`; an out-of-range index is rejected at build
time. This is a special case of the constant-index gather in [§8](#8-index-arithmetic).

```python
i = real_axis('i', 3); l = real_axis('l', 4)
tl = TL()
tl.H[i, 0]     = tl.X[i]
tl.H[i, l + 1] = tl.H[i, l] + tl.Delta[i, l]
tl.Y[i]        = tl.H[i, 3]   # read step 3 (base is index 0; N=4 so 0..4 are valid)

mod = ConstructedModule.construct(tl.to_morphism())
X = torch.tensor([1., 2., 3.]); Delta = torch.ones(3, 4)
out = mod(X, Delta); out = out[0] if isinstance(out, tuple) else out
assert torch.allclose(out, torch.tensor([4., 5., 6.]))
```

### Associative-scan fast path

When the recurrence is **affine in the state** — `H[l+1] = A_l · H[l] + b_l` — the
uncoupled path is automatically lowered to `torch.associative_scan` (parallel prefix)
instead of a sequential loop; otherwise it runs as a Python loop. This is detected at
compile time; no user action is needed.

---

## 8. Index arithmetic

An index slot accepts an **affine expression** `b + Σ cₖ aₖ` (b, cₖ ∈ ℤ, aₖ axes) —
not just a bare axis:

| expression | form | example |
|---|---|---|
| bare axis | `a` | `X[i]` — unchanged |
| constant | `b` (no axes) | `X[i, 3]` — fixed slice |
| offset | `a + b` | `X[i + 1]` — shift right by 1 |
| strided / dilated | `c·a + b` | `X[2*i + 1]` — every-other with offset |
| sum of axes | `a + a'` | `X[i + k]` — convolution window |

A coefficient is written `c * axis` — `int * RawAxis` resolves through `__rmul__`;
`__mul__` is reserved for tensor product and is not overloaded. Floor-division and
modulo (`//`, `mod`) are outside the affine boundary and are not supported.

An affine expression on a **read (RHS) slot** is a **gather** — an input reindexing
composed before the contraction. An affine expression on a **write (LHS) slot** is a
**scatter** — an output reindexing composed after the contraction, zero-filling
coordinates the image does not reach by default.

**The iteration gate.** An `axis+int` LHS (`Y[i+1]`) has the same syntax as a
recurrence step (`H[i, l+1]`). The two are distinguished at finalize time: if the
tensor carries a base case, a `.recur()` morphism, or an explicit `.iteration_axis()`
call it remains a recurrence; otherwise it is reclassified as a scatter ([§7](#7-iteration-and-recurrence) explains
the full iteration model).

### Constant reads

A literal integer in an index slot selects one coordinate and drops that axis from the
equation:

```python
i = real_axis('i', 4)
tl = TL()
tl.Y[i] = tl.X[i, 3]      # Y[i] = X[i, 3]; the column axis is dropped

mod = ConstructedModule.construct(tl.to_morphism())
X = torch.arange(20.).reshape(4, 5)
out = mod(X); out = out[0] if isinstance(out, tuple) else out
assert torch.allclose(out, X[:, 3])
```

Multiple constant slots work simultaneously — `X[3, j, 2]` selects row 3, column 2,
leaving only the `j` axis free. When the input is declared (`.tensor(*shape)`),
constants are range-checked at build time (`0 ≤ c < |axis|`).

### Affine gather (RHS)

**Shift.** `X[i + 1]` reads one position ahead; `X[i - 1]` one behind:

```python
i = real_axis('i', 5)
tl = TL()
tl.Y[i] = tl.X[i + 1]     # Y[i] = X[i+1]; reads positions 1..5 of X

mod = ConstructedModule.construct(tl.to_morphism())
X = torch.arange(7.)
out = mod(X); out = out[0] if isinstance(out, tuple) else out
assert torch.allclose(out, X[1:6])
```

**Stride / decimation.** A coefficient strides over the input:

```python
i = real_axis('i', 5)
tl = TL()
tl.Y[i] = tl.X[2 * i]     # Y[i] = X[2i]; every other element starting at 0

mod = ConstructedModule.construct(tl.to_morphism())
X = torch.arange(10.)
out = mod(X); out = out[0] if isinstance(out, tuple) else out
assert torch.allclose(out, X[::2])
```

**Dilation.** Combine stride and offset (`X[2*i + 1]` = odd-indexed elements):

```python
i = real_axis('i', 5)
tl = TL()
tl.Y[i] = tl.X[2 * i + 1]

mod = ConstructedModule.construct(tl.to_morphism())
X = torch.arange(11.)
out = mod(X); out = out[0] if isinstance(out, tuple) else out
assert torch.allclose(out, X[1::2])
```

**Multi-axis gather / convolution window.** When a slot combines two axes with `+`,
the gather builds a full index grid. Contracting over the kernel axis gives conv1d:

```python
import torch.nn.functional as F
Cout, Cin, K, Lout = 2, 3, 3, 6
co = real_axis('co', Cout); ci = real_axis('ci', Cin)
i  = real_axis('i',  Lout); k  = real_axis('k',  K)

tl = TL()
tl.Y[co, i] = tl.W[co, ci, k] * tl.X[ci, i + k]   # sum over ci and k

mod = ConstructedModule.construct(tl.to_morphism())
W = torch.randn(Cout, Cin, K); X = torch.randn(Cin, Lout + K - 1)
# X appears first in the equation (i+k slot), so it is the first external input.
out = mod(X, W); out = out[0] if isinstance(out, tuple) else out
assert torch.allclose(out, F.conv1d(X.unsqueeze(0), W).squeeze(0), atol=1e-4)
```

**Range validation.** When the input is declared with `.tensor(*shape)`, the compiler
checks that the entire read interval `[lo, hi]` lands within `[0, size)`. Out-of-range
reads — including negative ones — are rejected at build time with a message naming the
axis:

```python
a = real_axis('a', 4); i = real_axis('i', 4)
tl = TL()
tl.X.tensor(a)
# tl.Y[i] = tl.X[i + 2]   # would raise: reads positions 2..5, but size is 4
# tl.Y[i] = tl.X[i - 1]   # would raise: reads position -1 at i=0
```

### Affine scatter (LHS)

An affine expression on the LHS places the computed value at affine output positions.
Coordinates the image does not reach take the fill (zero by default).

**Offset scatter.** Shift right by 1; position 0 stays at zero:

```python
i = real_axis('i', 4)
tl = TL()
tl.Y[i + 1] = tl.X[i]     # Y[1..4] = X[0..3]; Y[0] = 0

mod = ConstructedModule.construct(tl.to_morphism())
X = torch.tensor([1., 2., 3., 4.])
out = mod(X); out = out[0] if isinstance(out, tuple) else out
exp = torch.zeros(5); exp[1:] = X
assert torch.allclose(out, exp)
```

The output size is **inferred** from the map when the output is undeclared
(`maxpos + 1`, where `maxpos = const + Σ coeff · (axis_size − 1)`). Declaring the
output with `.tensor(o)` uses that fixed size — so the tail beyond the image is filled
rather than truncated — and requires the image to fit within it:

```python
i = real_axis('i', 4); o = real_axis('o', 10)
tl = TL()
tl.Y.tensor(o)             # output fixed at size 10; positions 7..9 stay zero
tl.Y[2 * i] = tl.X[i]

mod = ConstructedModule.construct(tl.to_morphism())
X = torch.tensor([1., 2., 3., 4.])
out = mod(X); out = out[0] if isinstance(out, tuple) else out
assert out.shape == (10,) and torch.allclose(out[0::2][:4], X)
```

**Upsampling.** Stride the LHS to leave gaps (zero-filled):

```python
i = real_axis('i', 4)
tl = TL()
tl.Y[2 * i] = tl.X[i]     # even positions get X; odd positions stay zero

mod = ConstructedModule.construct(tl.to_morphism())
X = torch.tensor([1., 2., 3., 4.])
out = mod(X); out = out[0] if isinstance(out, tuple) else out
exp = torch.zeros(7); exp[0::2] = X
assert torch.allclose(out, exp)
```

### Scatter fill and conflict

**Custom fill.** Override the default zero fill with `.scatter(fill=…)`:

```python
i = real_axis('i', 4)
tl = TL()
tl.Y.scatter(fill=-1.)     # uncovered positions get -1
tl.Y[2 * i] = tl.X[i]

mod = ConstructedModule.construct(tl.to_morphism())
X = torch.tensor([1., 2., 3., 4.])
out = mod(X); out = out[0] if isinstance(out, tuple) else out
exp = torch.full((7,), -1.); exp[0::2] = X
assert torch.allclose(out, exp)
```

**Conflicting writes.** A non-injective LHS map — where multiple input coordinates map
to the same output position — is **rejected at build time** unless a reduction is
declared. Declare `reduce='sum'` to accumulate overlapping writes:

```python
i = real_axis('i', 3); j = real_axis('j', 3)
tl = TL()
tl.Y.scatter(reduce='sum')
tl.Y[i + j] = tl.X[i, j]   # Y[k] = Σ_{i+j=k} X[i,j]  (polynomial product coefficients)

mod = ConstructedModule.construct(tl.to_morphism())
X = torch.randn(3, 3)
out = mod(X); out = out[0] if isinstance(out, tuple) else out
exp = torch.zeros(5)
for a in range(3):
    for b in range(3):
        exp[a + b] += X[a, b]
assert torch.allclose(out, exp)
```

Without the `.scatter(reduce='sum')` declaration, `tl.Y[i + j] = tl.X[i, j]` would
raise a `ValueError` at assignment time naming the overlapping positions.

### Affine gather inside a scan step

A gather over a **non-iteration** axis works inside a recurrence body. The compiler
hoists it into a top-level `Reindex` entry that feeds the scan; slots referencing the
iteration axis (`l`) stay on the Scan path.

```python
i = real_axis('i', 4); l = real_axis('l', 3)
tl = TL()
tl.H[i, 0]     = tl.X[i]
tl.H[i, l + 1] = tl.H[i, l] + tl.D[i + 1, l]   # D[i+1, l]: gather over i, pass-through on l

mod = ConstructedModule.construct(tl.to_morphism())
X = torch.zeros(4); D = torch.randn(5, 3)
# The hoisted Reindex for D is ordered before the Scan in topological order,
# making D the first external input.
out = mod(D, X); out = out[0] if isinstance(out, tuple) else out
H = X.clone(); hist = [H.clone()]
for s in range(3):
    H = H + D[1:5, s]; hist.append(H.clone())
assert torch.allclose(out, torch.stack(hist, dim=-1))
```

---

## 9. Compilation and execution

Pick the entry point by what you built:

| Method | Input | Returns | Use |
|---|---|---|---|
| `tl.bc_signature()` | exactly one equation | `Broadcasted` (or `Composed` if it has `+`/a nonlinearity to split) | single equation |
| `tl.to_equation()` | exactly one equation | `TensorEquation` | introspection |
| `tl.to_program()` | any number | `TensorProgram` | grouping equations |
| `tl.to_morphism()` | any number, incl. recurrences | `ThreadedComposed` (live-pool routed) | the general path |
| `ConstructedModule.construct(morphism)` | any morphism | `nn.Module` | final lowering |

The usual path is `construct(tl.to_morphism())`. For a single plain equation,
`construct(tl.bc_signature())` is equivalent and slightly more direct.

**Inspecting the einops signature.** `generate_tensor_equation_signature(bc)` takes a
single-equation `Broadcasted` and returns the einops contraction string that
`ConstructedTensorEquation` will use — useful for debugging contraction structure:

```python
from torch_compile.torch_compile import generate_tensor_equation_signature

i, j, k = axes('i j k')
tl = TL(); tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
print(generate_tensor_equation_signature(tl.bc_signature()))
# '... y0 x0, ... x0 y1 -> ... y0 y1'
```

Degree axes get tags `y0, y1, …`; contracted axes get `x0, x1, …`. The same axis
object always receives the same tag, so shared UIDs across weaves contract correctly.

```python
tl = TL(); i, j, k = axes('i j k')
tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]

eq   = tl.to_equation()        # TensorEquation (introspect lhs_indices, rhs, operator)
sig  = tl.bc_signature()       # Broadcasted
mod  = ConstructedModule.construct(tl.to_morphism())
```

**Calling the module.** External tensors are passed positionally in **first-appearance
(topological) order**. Sized predicate factors are auto-buffered and skipped. Multi-equation
and scan modules may return a **tuple** of outputs; unwrap with
`out[0] if isinstance(out, tuple) else out`, or unpack a coupled scan's states
directly.

```python
# multi-equation FFN: project down, project up + residual, normalize
q = real_axis('q', 3); dm = real_axis('dm', 8); dh = real_axis('dh', 4)
tl = TL()
tl.Y[q, dh]  = tl.W1[dm, dh] * tl.H[q, dm]
tl.Out[q, dm] = normalize(tl.W2[dm, dh] * tl.Y[q, dh] + tl.H[q, dm])
mod = ConstructedModule.construct(tl.to_morphism())
W1, Hh, W2 = torch.randn(8, 4), torch.randn(3, 8), torch.randn(8, 4)  # topo order
out = mod(W1, Hh, W2)
out = out[0] if isinstance(out, tuple) else out
assert out.shape == (3, 8)
```

---

## 10. Worked example: a small attention block

Multi-head scaled-dot-product attention with a causal mask, compiled and run:

```python
SEQ, H, DK = 5, 2, 3
q = real_axis('q', SEQ)      # query position
h = real_axis('h', H)        # head
k = real_axis('k', DK)       # key/query feature (contracted)
x = norm_axis('x', SEQ)      # key position — softmax reduces over this

tl = TL()
# scores[h,q,x] = sum_k Q[q,h,k] * K[x,h,k], softmax over x, causal mask x<=q
tl.Attn[h, q, x] = softmax(tl.Q[q, h, k] * tl.K[x, h, k], where=(x <= q))

mod = ConstructedModule.construct(tl.to_morphism())
Q = torch.randn(SEQ, H, DK)
K = torch.randn(SEQ, H, DK)
Attn = mod(Q, K)
Attn = Attn[0] if isinstance(Attn, tuple) else Attn
# Attn[h, q, :] is a probability distribution over key positions, zero for x > q.
```

To carry it further — value aggregation `Out[q,h,d] = Attn[h,q,x] * V[x,h,d]` — add a
second equation reading `Attn`; `to_morphism()` topologically orders the two and threads
`Attn` from the first into the second.

---

## 11. Further reading

- **[bool_semiring_extension.md](bool_semiring_extension.md)** — the Boolean semiring
  `(𝔹, ∨, ∧)`: predicate datatypes, the ι/H promotion–demotion retraction, masked
  reductions, acset serialisation, and tsncd rendering.
- **[index_arithmetic.md](../papers/index_arithmetic.md)** — the categorical model (St
  affine morphisms), the per-tensor iteration gate, and implementation notes for
  constant reads, affine gathers, scatters, range validation, and the normalization pass.
- **`torch_compile/materialise.py`** — `materialise_iverson` and the expanded-buffer
  performance tradeoff for repeated predicate axes.
- **Tests as examples** — `tests/test_tensor_dsl.py` (DSL construction) and
  `tests/test_torch_compile.py` (end-to-end compiled behaviour) are the authoritative,
  always-current example set.

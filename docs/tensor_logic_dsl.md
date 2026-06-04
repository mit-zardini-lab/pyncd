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
8. [Compilation and execution](#8-compilation-and-execution)
9. [Worked example: a small attention block](#9-worked-example-a-small-attention-block)
10. [Further reading](#10-further-reading)

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
| `real_axis('d', 512)` | `RawAxis` (ℝ dimension, optional size) | a real-valued dimension |
| `norm_axis('x', N)` | `NormAxis` | the axis a softmax/normalize reduces over |
| `nat_axis('t', 50000)` | `NatAxis` | a natural-number / index dimension (ℕ) |

```python
i, j, k = axes('i j k')          # variadic-in-one-string
i, j, k = axes('i', 'j', 'k')    # or variadic args
(d_ff,) = axes('d_{ff}')         # LaTeX-style names are fine

d = real_axis('d', 512)          # concrete size 512  → d._size == Integer(512)
d = real_axis('d')               # free size          → d._size is FreeNumeric
```

**Sizes matter for two features:** auto-materialising predicates (§5) and iteration
(§7) both require concrete integer sizes. Plain contractions do not — `axes(...)`
is enough.

**`norm_axis`** marks the normalisation dimension on the *left-hand side* of a
softmax/normalize equation. It also lets the compiler drop additive terms that are
constant along that axis (`softmax(f + c) == softmax(f)` when `c` does not depend on
the norm axis).

**`nat_axis`** marks an index dimension; `selection` declarations (§4) promote their
`NatAxis` slots so the morphism carries the `Natural` datatype.

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
They enforce arity, control datatype, and drive axis promotion. All three return the
proxy for chaining.

```python
i = real_axis('i', 64); k = real_axis('k', 64); j = real_axis('j', 64)
tl = TL()
tl.W.tensor(i, k)            # ordinary ℝ-valued tensor
tl.Mask.predicate(i, j)      # 𝔹-valued predicate (output typed Bool)
tl.Emb.selection(nat_axis('t', 50000), real_axis('m', 512))  # lookup table
```

- **`.tensor(*shape)`** — default contraction semantics, no promotion.
- **`.predicate(*shape)`** — marks the name `Bool`-typed. When such a tensor is the
  *output* of an equation, the einsum result is demoted with the Heaviside step
  $H(x)=\mathbf 1[x>0]$ to `{0,1}` (∃/∧ semantics — see §6). Axes are not promoted.
- **`.selection(*shape)`** — slots declared as `NatAxis` promote their index axes to
  `NatAxis` (so the morphism is `Natural`-typed). Used for embedding/lookup tables.

```python
tl = TL()
t = nat_axis('t', 50000); d = real_axis('d', 512)
tl.Emb.selection(t, d)
it = tl.Emb[axes('a')[0], axes('b')[0]]
assert isinstance(it.indices[0], NatAxis)        # 't' slot promoted
assert not isinstance(it.indices[1], NatAxis)    # 'd' slot unchanged
```

> Note: the full embedding-as-masked-contraction lowering is not yet wired through the
> DSL (`ConstructedEmbedding` uses `torch.nn.Embedding` directly). See
> [bool_semiring_extension.md](bool_semiring_extension.md) §8.

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
from another `TL` session), register the morphism directly with `recur`:

```python
l = real_axis('l', L)
x, m = axes('x m')
tl = TL()
tl.H[x, m, 0] = tl.X[x, m]            # base
tl.H.recur(l, transformer_layer())    # step: a pre-built morphism (state in → state out)
morphism = tl.to_morphism()
```

### Associative-scan fast path

When the recurrence is **affine in the state** — `H[l+1] = A_l · H[l] + b_l` — the
uncoupled path is automatically lowered to `torch.associative_scan` (parallel prefix)
instead of a sequential loop; otherwise it runs as a Python loop. This is detected at
compile time; no user action is needed.

---

## 8. Compilation and execution

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

## 9. Worked example: a small attention block

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

## 10. Further reading

- **[bool_semiring_extension.md](bool_semiring_extension.md)** — the Boolean semiring
  `(𝔹, ∨, ∧)`: predicate datatypes, the ι/H promotion–demotion retraction, masked
  reductions, acset serialisation, and tsncd rendering.
- **`torch_compile/materialise.py`** — `materialise_iverson` and the expanded-buffer
  performance tradeoff for repeated predicate axes.
- **Tests as examples** — `tests/test_tensor_dsl.py` (DSL construction) and
  `tests/test_torch_compile.py` (end-to-end compiled behaviour) are the authoritative,
  always-current example set.

---
tags: [layer/para, concept]
code: para/algebra/pathway_collapse.py, algebra/operator_expansion.py
status: evolving
---

# Recomputing the Exponent in the Backward Pass

## What it is

The attention backward pass reads the probabilities $P = \mathrm{softmax}_x(QK^{\top})$
three times: twice to form $\mathrm{d}S = P \odot (\mathrm{d}P - D)$ and once to form
$\mathrm{d}V = P^{\top}\mathrm{d}O$. There are two ways to have $P$ available. Option A
computes $QK^{\top}$ and the exponent in the forward pass and saves the result, an array
of size $q \times x$ per head. Option B saves nothing of that size and recomputes
$QK^{\top}$ and the exponent in the backward pass from $Q$, $K$ and a row statistic.
Option B is what FlashAttention 1, 2 and 3, xformers' memory-efficient attention and
PyTorch's flash and efficient SDPA backends do, and it is what training uses in practice.
Option A is what eager attention does, because autograd saves the softmax output, and it
is what bounded the context length before FlashAttention.

`pathway_collapse.dedup_and_collapse` produces option A. This note records the trade-off,
why the row statistic stays on the tape under both options, and the rewrite that would
produce option B.

## The tape the derivation produces beside FlashAttention's

Deriving the pair from the expanded softmax and simplifying it gives a tape which, beside
FlashAttention's, is:

| value | shape | derived tape | FlashAttention |
|---|---|---|---|
| $Q$, $K$ | $[q, d]$, $[x, d]$ | `s0`, `s1` | stored |
| $V$ | $[x, v]$ | `s5` | stored |
| $O$ | $[q, v]$ | `s8` | stored |
| row statistic | $[q]$ | `s6`, the reciprocal $r = 1 / \sum_x e$ | stored, as $L = m + \log l$ |
| exponent $e = \mathrm{e}^{QK^{\top}}$ | $[q, x]$ | `s4` | recomputed |

The one difference is the exponent. $D = \langle \mathrm{d}O, O\rangle$ is computed from
$O$ in both, in FlashAttention's preprocessing step and in the hoisted pairing the
collapse derives.

## Storing the exponent against recomputing it

Five things weigh on the choice. The first three all favour recomputing, on a GPU, at
any sequence length that matters.

**Memory capacity.** Option A keeps $b \cdot h \cdot q \cdot x$ elements per layer until
the backward pass runs, and every layer's copy is held at once. At $q = x = 8192$,
32 heads and bf16 that is 4 GiB per layer per sequence, and it grows quadratically with
sequence length. Option B keeps $b \cdot h \cdot q$. Megatron's selective activation
recomputation (Korthikanti et al., 2022) recomputes exactly this part of the layer,
because the attention matrices hold most of the activation memory and cost a few percent
of the FLOPs.

**Arithmetic.** Option B adds one matmul, $2\,q\,x\,d$ FLOPs, and one exponent per
element to the backward pass. The FlashAttention backward runs five matmuls where option
A runs four, so the backward costs about 25% more and forward plus backward about 17%
more.

**Bandwidth.** Option A writes $e$ to HBM in the forward pass and reads it back in the
backward pass, 4 bytes per element in bf16. An H100 performs roughly 300 FLOPs in the
time it moves one byte, so those 4 bytes buy about 1200 FLOPs, and the recompute costs
$2d$ FLOPs per element, 128 to 256 at $d = 64$ to $128$. Recomputing is cheaper than
storing by 5 to 10 times even with unlimited memory. The backward kernel already holds
the $Q$ and $K$ tiles in shared memory for the $\mathrm{d}Q$ and $\mathrm{d}K$ matmuls,
so the extra matmul moves no data. This is the argument of Dao et al. (2022).

**Precision.** Option A stores $e$ or $P$ in the activation datatype, usually bf16.
Option B recomputes it in fp32 registers from bf16 $Q$ and $K$, and forms
$\mathrm{d}S = P \odot (\mathrm{d}P - D)$ at fp32.

**The statistic.** Option B needs the row statistic saved, which the next section
states.

Option A wins when $q \cdot x$ is small next to $d$, or when the attention matrix is
wanted for its own sake, as when attention maps are being inspected.

## The row statistic stays on the tape

The row statistic is saved under both options, and the tape in the table above holds it
as `s6`. Two things need saying about it, because a reader who lists FlashAttention's tape
as $Q$, $K$, $V$ and $O$ has left it out.

It cannot be dropped cheaply. Recomputing $P$ from $Q$ and $K$ gives the exponent, and
turning the exponent into $P$ needs the row's normaliser. The backward kernel loops over
key tiles and forms $\mathrm{d}S$ one tile at a time, and the normaliser is a sum over
every key tile of the row. Recomputing it would need a whole extra sweep of $K$ per query
tile before the main loop, another $2\,q\,x\,d$ FLOPs, to save $q$ floats. FlashAttention
stores the $q$ floats.

Its form differs. FlashAttention keeps the maximum shift $m$ and the row sum $l$ in one
number $L = m + \log l$, so that $P = \mathrm{e}^{S - L}$ is one subtraction and one
exponent, and neither the shift nor the sum can overflow.
`algebra/operator_expansion.py` holds both expansions. `expand_softmax` has no shift, so
the statistic of the pair in the table is the reciprocal $r = 1/l$ on its own.
`expand_shifted_softmax` subtracts the row maximum before the exponent, and the pair
derived from it carries $m$ and $r$ as two slots, both at $[q]$. The maximum has a zero
cotangent, because a softmax is unchanged by a shift of its scores, so the backward pass
reads $m$ only where it rebuilds the exponent. A pass that rebuilds the exponent takes $m$
as a side operand of the rebuilt chain and computes $\mathrm{e}^{QK^{\top} - m}$ from $Q$,
$K$ and $m$, which is as stable as the forward pass. Folding $m$ and $r$ into one $L$ needs
a collapse rule for a maximum and a reciprocal feeding one exponential, and no such rule
exists.

The algebra put $r$ there rather than the row sum. `backprop` tapes the row sum $z$ as the
reciprocal's residual, because the numeric derivative of $x^{-1}$ is $-z^{-2}$.
`split_inverse_squares` writes $-z^{-2}$ as $r \cdot (-r)$, and `migrate_drops` moves
the slot from $z$ to $r$, because the backward pass reads $z$ only through the reciprocal.
The tape holds the value the backward pass reads, which is the form FlashAttention stores.

## The rewrite that would produce option B

No pass in `pathway_collapse` replaces a `Grab` by a recomputation through an `Einops`.
`migrate_drops` spends memory to remove a recomputation, and
`recompute_elementwise_slots` spends a recomputation to remove a slot, across a pointwise
map alone. The exponent sits one pointwise map behind a contraction of two taped values,
so neither reaches it.

The rule is: where the forward pass drops $f(\mathrm{Einops}(a, b))$ and $a$ and $b$ are
already on the tape, delete the drop and replace every grab of it by
$\mathrm{Grab}(a), \mathrm{Grab}(b) \to \mathrm{Einops} \to f$. On attention it removes
`s4`, leaves the five slots FlashAttention stores, and adds one `Einops` and one
`Arithmetic` to the backward pass, which is the FlashAttention backward with its
recomputed $QK^{\top}$.

Which of the two policies is right is the number of bytes kept on the tape against the
arithmetic a rebuild costs, and the ratio of the two on the hardware the pair will run on.
`migrate_drops` and `recompute_elementwise_slots` leave that decision to the person
calling them, and no pass in this repository makes it.

## Detecting a quadratic slot from the shape

Attention written on two axes, $q$ and $x$, does not state that the two have the same
size, so a rule on the tape cannot read from the shape `[q, x]` that the slot is
quadratic. Self-attention written from one input copied three ways puts the token axis
at both positions of the scores, `[x, x]`, and the shape then states it. A legality rule
follows: an array whose shape carries one axis twice may not be stored, and the value is
recomputed in the backward pass from what is stored. The rule needs no declaration about
the axis, and it misses cross-attention, where the two sequence axes are distinct.
Marking an axis as one that scales with the input rather than with the model is the
declared-a-priori form that would cover that case.

The derivative rules and the collapse read an operator by position since 2026-09-05, so a
pair derived from the copied form carries the exponent at `[x, x]` and the rule above can
be read off the shape alone.

## Gaps

- **The rewrite above is not implemented.** `migrate_drops` and
  `recompute_elementwise_slots` reach a pointwise map and nothing further, so the exponent
  stays on the tape.
- **The tape holds $m$ and $r$ as two slots where FlashAttention holds one $L$.** Folding
  the two into $L$ needs the collapse rule the section above names.
- **A slot produced by a `Linear` is not rebuilt**, because the chain finder reads an
  `Einops` and a pointwise map and nothing else.

## See also

- [[Pathway Collapse]] — the rewrites that produce the tape in the table
- [[Backpropagation]] — how each operator declares its residual
- [[Expression Simplification]] — the two expansions of a softmax
- [[Notebooks]] — what each notebook of the repository demonstrates

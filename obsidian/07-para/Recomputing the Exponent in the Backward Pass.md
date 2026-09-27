---
tags: [layer/para, concept]
code: para/algebra/pathway_collapse.py, para/algebra/recompute_contraction_slots.py, para/algebra/store_operands_of_views.py, algebra/operator_expansion.py, notebooks/website/tutorial/derive_training_step.py
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
why the row statistic stays on the tape under both options, the rewrite that produces
option B, and the rewrite that stores the keys and the values of a causal attention
once per token.

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
time it moves one byte, so moving those 4 bytes takes the time of about 1200 FLOPs, and
the recompute costs
$2d$ FLOPs per element, 128 to 256 at $d = 64$ to $128$. Recomputing is cheaper than
storing by 5 to 10 times even with unlimited memory. The backward kernel already holds
the $Q$ and $K$ tiles in shared memory for the $\mathrm{d}Q$ and $\mathrm{d}K$ matmuls,
so the extra matmul moves no data. Dao et al. (2022) make the same argument.

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

## The rewrite that produces option B

`migrate_drops` in `pathway_collapse` keeps a value on the tape to remove a
recomputation, and `recompute_elementwise_slots` recomputes a value to remove a slot,
across a pointwise map alone. The exponent sits one pointwise map behind a contraction of
two taped values, so neither reaches it.

`para.algebra.recompute_contraction_slots.recompute_contraction_slots` reaches one
contraction further. Where the forward pass drops $f(\mathrm{Einops}(a, b))$, with $f$ a chain of pointwise maps and additions, and $a$ and $b$ are already on the tape, it deletes the drop and replaces every grab of it by $\mathrm{Grab}(a), \mathrm{Grab}(b) \to \mathrm{Einops} \to f$, rebuilt onto the wire the grab produced so that its readers do not move. On attention it removes `s4`, leaves the five slots FlashAttention stores, and adds one `Einops` and one `Arithmetic` to the backward pass, which is the FlashAttention backward with its recomputed $QK^{\top}$.

An addition in the chain has side operands, such as the negated maximum a shifted
softmax adds to its scores before the exponent. A side operand is rebuilt from a slot of
its own. The wire it is a pointwise image of is grabbed where the forward pass drops it
already, and dropped to a new slot otherwise. The new slot is small, because a side
operand is broadcast over the axis the chain's contraction produced. On the shifted
expansion the exponent is $\mathrm{e}^{QK^{\top} - m}$, and the backward pass rebuilds it
from $Q$, $K$ and the maximum $m$, which joins the tape at $[q]$, so the recomputation is
as stable as the forward pass.

Which of the two policies is right is the number of bytes kept on the tape against the
arithmetic a rebuild costs, and the ratio of the two on the hardware the pair will run on.
`migrate_drops`, `recompute_elementwise_slots` and `recompute_contraction_slots` leave
that decision to the person calling them, and a caller that names no slots has
`recompute_contraction_slots` rebuild every slot it can. The training step of the
tutorial pages rebuilds every such slot.

## Storing the operand of a view

Causal attention reads the keys and the values through a mask, a view that holds token
$i_{x} - i_{w}$ at slot $i_{w}$ of token $i_{x}$. The collapse saves the arrays returned
by the mask, at $[x, w|x, d]$, because the backward pass reads them there. A view
computes nothing, so `para.algebra.store_operands_of_views.store_operands_of_views` moves
such a drop to the array the chain of views reads, where that array has fewer axes, and
replays the views after every grab of the slot in the backward pass. The slot keeps its
name and holds the array the views read. Where that array is on the tape already, the
drop of the result is deleted and its grabs read the slot that holds the array. A view
left with no reader once its drop is gone, such as the view of the state that the
CausalSlide computes only to save it, is deleted with the drop.

The recomputation of the exponent runs first. `recompute_chain_of` needs both operands of
the contraction on the tape, and the masked keys leave the tape once their drop has moved
to the keys.

## The tutorial training step

`notebooks/website/tutorial/derive_training_step.py` collapses the derived pair with
`dedup_and_collapse`, then applies `recompute_contraction_slots` to every slot it can
rebuild, then `store_operands_of_views`, then `pathway_collapse.dedup_roots`, which merges
the grabs and the views the two rewrites repeat. `TrainingStep.recomputed` holds the
result, and the training variant of each tutorial page draws it, per
[[Website Notebooks]].

The tape of scaled dot-product attention then holds $Q$, $K$, $V$, $O$ and $r$, the five
arrays FlashAttention stores. The tape of causal self-attention with weights and a
residual connection holds six arrays: the state at $[x, m]$, $Q$, $K$ and $V$ at
$[x, d]$, the result of the attention before $W^{O}$ at $[x, d]$, and $r$ at $[x]$. The
state and the result before $W^{O}$ are the inputs of the learned matrices, which the
gradient of each matrix reads. The weighted model is derived as built, with the mask after
the projections, because the CausalSlide runs $W^{K}$ and $W^{V}$ over every slot, and its
tape then holds the keys and the values at $[x, w|x, d]$. The forward variant of its page
still draws the CausalSlide.

`validate_attention.py` checks that the collapsed tape holds the exponentials of the
scores, that the backward pass rebuilds them from the queries and the keys, and that the
forward pass saves five arrays. `validate_attention_with_weights_and_residual.py` checks
that the training step is derived from the model as built, that a training step of the
CausalSlide would save keys and values for every slot, that the forward pass saves six
arrays, and that the backward pass reads the keys and the values through the mask. Both
compare the gradients of the training step with `torch.autograd`.

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

- **The tape holds $m$ and $r$ as two slots where FlashAttention holds one $L$.** Folding
  the two into $L$ needs the collapse rule the section above names.
- **A slot produced by a `Linear` is not rebuilt**, because the chain finder reads an
  `Einops` and a pointwise map and nothing else.
- **The chain finder does not read through a view.** `recompute_chain_of` does not
  follow an operand back through a view to a taped array, so the recomputation has to
  run before `store_operands_of_views`. A chain finder that reads through views would
  let the two rewrites run in either order.
- **Nothing chooses which slots to rebuild.** The public code has no cost of a slot,
  so a caller names the slots or rebuilds every one the rule reaches.

## See also

- [[Pathway Collapse]] — the rewrites that produce the tape in the table
- [[Backpropagation]] — how each operator declares its residual
- [[Expression Simplification]] — the two expansions of a softmax
- [[Notebooks]] — what each notebook of the repository demonstrates
- [[Website Notebooks]] — the tutorial pages that draw the training step

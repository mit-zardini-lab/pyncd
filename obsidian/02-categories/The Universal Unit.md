---
tags: [layer/categories, concept]
code: data_structure/BroadcastedCategory.py, data_structure/Numeric.py, algebra/registries/accumulator.py, deepseek/data_structure.py
status: speculative
agent: Claude (Fable 5.1, high effort, 2026-09-11)
---

# The Universal Unit

## What it is

The real datatype is read as a union. A position of an array on `cat.Reals` holds either
a real number or the universal unit, written $\mathbb{1}$ with a blackboard-bold one. The
universal unit is the value a position holds when it holds nothing. An empty slot of a
[[Sparse Axes|sparse axis]] holds it, and so does a position that a padded read lands on
outside the input, per [[Padding and Masks as Sparse Axes]].

Two laws define the unit, and every concrete value that a hand-written kernel writes into
an empty position follows from them.

**A fold over an axis ignores the unit.** For an associative operation $f$ applied along
an axis, $f(x, y, \mathbb{1}, z) = f(x, y, z)$. Feeding the unit at a position is the same
as feeding nothing at that position, wherever the position sits.

**A pointwise operation preserves the unit.** An elementwise map sends $\mathbb{1}$ to
$\mathbb{1}$. An empty position stays empty through an exponential, a scale, a ReLU or a
product with a weight, so the structure of a sparse axis survives every elementwise
operation applied along it.

The two laws together decide what the unit is worth at every operator. The worth differs
from one operator to the next, and the laws are what say which value is right where.

| the operation reading the position | the value the unit stands for | why |
|---|---|---|
| a sum over a group, in an `Einops` or a `Linear` | $0$ | the unit of addition |
| a fold by `ops.Maximum` | $-\infty$ | the unit of `max` |
| a contraction, meaning a product feeding a sum | $0$ | the product keeps the unit at that position by the second law, and the sum drops it by the first. Feeding $0$ drops the same term |
| a softmax, meaning an exponential feeding a sum, a reciprocal and a scale | $-\infty$ | the exponential keeps the unit, the denominator drops it, and the output at that position is the unit. Feeding $-\infty$ puts a $0$ at that position, which the contraction after the softmax drops alike |

The score of a masked key is therefore $-\infty$ before a softmax and $0$ after it, and
the two are one value read through two operators. A kernel written by hand chooses the
fill per operator, and a $0$ written into a score before the exponential is a key that
attends with weight $e^{0}$. The unit removes the choice, because the fill is a property
of the operator the position reaches rather than of the array.

## Where it lives

Nothing in the package represents the unit yet. `cat.Reals` in
`data_structure/BroadcastedCategory.py` is a datatype with no fields, and no array carries
a fill value. What the package holds is the unit of each terminal operation, in three
places, and each of them is one row of the table above.

- `nm.Associative` in `data_structure/Numeric.py` declares `unit` per operation,
  `Integer(0)` for `Addition` and `Integer(1)` for `Multiplication`, and `template` drops
  the unit from an operation's operands, per [[Numerics]]. That is the first law, applied
  to numerics.
- `algebra/registries/accumulator.py` registers, per terminal operation, the operator
  that folds two partial results into one, being `ops.AdditionOp` for an `Einops` and for
  a `Linear` and `ops.Maximum` for a `Maximum`. `ops.ConstantOp` is the value such a fold
  starts from, being `0` for a sum, and that starting value is the unit of the terminal.

Each states the unit of one operation. The universal unit is the statement that one value
stands for all of them, made once at the datatype, so that a model says a position is
empty and no operator has to be asked which number that means.

## The mathematics

A monoid $(M, \cdot, e)$ has one unit $e$, the element its fold ignores. The reals carry
several monoids, being $(\mathbb{R}, +, 0)$, $(\mathbb{R}, \times, 1)$ and
$(\mathbb{R} \cup \{-\infty\}, \max, -\infty)$, and each has its own unit. The universal
unit adjoins one fresh element to the carrier, giving $\mathbb{R} \cup \{\mathbb{1}\}$,
and extends every fold so that $\mathbb{1}$ is its unit and every pointwise map so that it
fixes $\mathbb{1}$.

The extension is consistent because the operations of this package meet the unit in one
order. A pointwise map is applied first and a fold is applied after it. The second law
carries $\mathbb{1}$ through the maps unchanged, and the first law removes it at the fold.
A contraction is the case where the two meet. The product of $\mathbb{1}$ with a weight is
$\mathbb{1}$, and the sum over the contracted group drops it. A zero pad under a
convolution and a $-\infty$ mask under a softmax are for that reason one statement, and
the number a kernel writes depends on where it writes it.

An `AdditionOp` is read under both laws, and the two readings differ. As the accumulator
of a fold it combines two partial results, so a partial that holds the unit, from a part
that read nothing, is ignored and the running value is unchanged. As an operation between
two wires of a model, such as a residual connection or a bias, it is pointwise, so a
position that is empty on one wire is empty in the sum. The second reading is what a
padded token needs. Every elementwise operation keeps the token empty through the residual
stream, and the fold at the loss ignores it. The accumulator registry already tells the two
apart, because a row is registered per terminal operation, and an implementation of the
unit has to keep them apart.

A fold that also reads the size of its axis reads the activity. A mean over a sparse axis
divides by the number of active positions, and `ops.Normalize` over a padded axis
normalises by the positions that hold a real. A `SparseAxis` carries both numbers, its
`_size` and its `activity`, and the second is the one a mean reads.

## The rules

- **A sparse axis's empty slots hold $\mathbb{1}$.** [[Sparse Axes]] says the slots hold
  nothing, and this note names the nothing. On $[\mathbb{R},\, x, k/n]$ every slice along
  $x$ holds $k$ positions with a real and $n - k$ with the unit.
- **Write the unit and let the operator decide the number.** A model states that a
  position is empty. It states neither $0$ nor $-\infty$. The number is derived at the
  operator that reads the position, from the table above, and the lowering in
  [[Torch Compile]] would write it.
- **A multiplicative zero-one mask states something else.** Multiplying a score by $0$
  before a softmax gives the position the weight $e^{0}$. The notebook that draws a
  candidate pool as a mask, `notebooks/sota/DeepSeekV41Flash/candidate_pool.py`, records under its
  omissions that the reference writes $-\infty$ where the drawing multiplies. Under the
  unit the pool is a sparse axis over the entries, the positions outside it hold
  $\mathbb{1}$, and the softmax reads $-\infty$ from the table without the model saying
  so.

## Gaps

None of this is implemented. The items below are what an implementation has to settle,
and each needs the checks that [[Padding and Masks as Sparse Axes]] lists under *Checks an
implementation owes*. [[Open Gaps]] carries the item.

- **The datatype does not carry the unit.** `cat.Reals` would state that its positions may
  be empty, or a second datatype would, and every consumer of `Reals` would have to accept
  the statement.
- **The lowering has no fill.** [[Torch Compile]] compiles a `Broadcasted` to a module
  with no notion of an empty position. The fill for each operator comes from the table
  above, and a masked softmax needs $-\infty$ where a masked contraction needs $0$.
- **The cotangent of an empty position is undecided.** A position holding $\mathbb{1}$
  contributed nothing forward, so nothing flows back to it. Whether its cotangent is $0$
  or $\mathbb{1}$ matters for an `Inject` writing into a sparse parent, per
  [[Selection and the Reverse Pass]], and for the transposed convolution of a padded
  convolution, per [[Derivatives]].
- **The cost of an empty position.** A fold over a run of positions that hold only units
  returns the unit, and nothing in the expression says that the run can be skipped.
  [[Padding and Masks as Sparse Axes]] takes that up, because it is what makes causal
  attention cost half of dense attention.

## See also

- [[Sparse Axes]] — the axis whose empty slots hold the unit
- [[Padding and Masks as Sparse Axes]] — the reads that produce empty positions by position alone
- [[Sparse Expansion]] — the rewrite that names the active positions of a selection
- [[Numerics]] — `Associative.unit`, the first law applied to numerics
- [[Broadcasted Category]] — where `Reals` and `Natural` are declared

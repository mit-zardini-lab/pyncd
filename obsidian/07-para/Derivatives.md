---
tags: [layer/para, concept]
code: algebra/node_expansion.py, para/data_structure/Contravariant.py, para/processing/backprop.py
status: working
---

# Derivatives

## What it is

[[Training]] states what a reverse pass is: a second morphism coupled to the first by a tape.
This note is the method for building one, in four steps, with the division of labour that
makes it tractable. A `Broadcasted` mixes two things whose duals are computed in completely
different ways, so the first step is to stop mixing them.

The note is written to be implemented from, and it has been. [[Backpropagation]] states what
exists. Every step names the function that performs it.

## Where it lives

| | |
|---|---|
| `algebra/node_expansion.py` | `expand_to_nodes`, which is step 1. It moved out of `algebra/linear_expansion.py`, which now re-exports it, because two consumers need it for opposite reasons |
| `para/data_structure/Para.py` | `TapeSlot`, `Grab` and `Drop`, which are the tape, per [[Training]] |
| `para/data_structure/Contravariant.py` | `Contravariant`, the construction rule that records steps 2 and 4, the syntactic transpose. Its `body` is the forward expression, and reading `body` inverts it |
| `para/registries/derivative.py` | one `Rule` per operator, covering steps 3 and 4 at once |
| `para/data_structure/contraction.py` | `contract`, the einsum over axis objects that steps 2 and 4 are written in |
| `para/data_structure/transpose.py` | `Transpose` and `ReindexTranspose`, which are steps 2 and 4 for a morphism that is already linear |
| `para/processing/backprop.py` | `forward_backward`, the whole transform, per [[Backpropagation]] |

## The method

### 1. Expand to nodes

`expand_to_nodes(broadcasted)` returns `(nodes, core)`. The nodes are a product of `View`
morphisms, one per input, each carrying that input's reindexing and nothing else. The core is
the same operator with its reindexings replaced by the degree identity.

Expand before differentiating anything. The two halves have duals of different kinds.

- A **reindexing** is affine, so as a linear map it is a 0/1 matrix and its transpose is
  completely determined. Writing it down needs no knowledge of any operator.
- An **operator** needs a declared derivative, and once its reindexings are trivial it is
  cleanly broadcast. Every input already sits in the degree, so its derivative is the
  pointwise one lifted over the degree, and no broadcasting logic appears in it at all.

Mixed together, every operator's derivative would have to re-derive the broadcasting.
Separated, the broadcasting is itself a morphism, differentiated once for all operators.

A node is fully tiled on both sides. Its own degree is the core's degree followed by that
input's target axes, which is where `nodes_reindexings` puts them, and putting them there is
what makes the reindexing a plain `Rearrangement`. Its output weave therefore has a TILED slot
for every one of them and no target of its own. The array it produces is the same either way,
and a weave that kept the target axes literal would disagree with `degree()` about how many
TILED positions exist, which is the one thing a weave may not do, per [[Weaves and Degree]].

### 2. Transpose the nodes

A reindexing $\rho$ takes an output index to the input index it reads, so a node computes
$y[i] = x[\rho(i)]$, and its transpose sums over the fibre:

$$x^{*}[j] \;=\; \sum_{i \,:\, \rho(i) = j} y^{*}[i]$$

There are five cases, and only the first stays inside [[Stride Category|St]]:

| the node is | its transpose is | still affine? |
|---|---|---|
| a permutation | the inverse permutation | yes, a `Rearrangement` |
| a **broadcast**, meaning a TILED position | a **sum** over those positions | no, it is a contraction |
| a **diagonal**, meaning a copy-`Rearrangement` | read the diagonal, then sum | a reindexing and a contraction |
| a **projection**, meaning a deleted input | zero | trivially |
| genuinely **strided**, such as a convolution window | a `transpose.ReindexTranspose` carrying $\rho$ | affine, and not a contraction |

The first four are a contraction against the same affine map, which is an ordinary `Einops`
that the algebra already has, and which `derivative.identity` builds through
`contraction.contract`. Nothing new is needed while $\rho$ is a `Rearrangement`. The fifth is
the row that used to be empty: $x = x' + w$ sends two domain axes to one codomain axis, so
the fibre is a diagonal rather than a set of axes, `contract` has nothing to name, and the
transpose is a scatter-add. The line worth holding on to reads:

> An **affine** index map transposes into a morphism the algebra can write down from the map
> alone: a contraction where the map is a `Rearrangement`, and a `ReindexTranspose` carrying
> the map itself where it is strided. An index map that depends on **data** transposes into
> an `Inject` that takes the map as an operand, because the map arrives on a wire rather
> than in the term, followed by the sum over the degree that accumulates it.

### Why the transpose needs an operator rather than a reindexing

Write $\rho : P \to Q$ as the user of this note did. A reindexing in a context $X$ is an
array morphism $[X;\rho] : [X, Q] \to [X, P]$, with $[X;\rho];[X,i] = [X,\rho(i)]$, so the
transpose is $[X,P] \to [X,Q]$, and it cannot be a `reindexings` entry. `Broadcasted.dom()`
reads the domain off `reindexing.cod()`, hard-wired, so a reindexing always points from the
output back to the input. The only other place to carry $\rho$ is the operator, which is what
`ReindexTranspose(name, reindexing)` is. The argument settles nothing about drawing it:
[[Diagram Display]] draws the node figure either way, unreversed, so that the pentagon's point
lands on the axis the fibre is summed onto.

The implementation does not factor the context $X$ out. The transpose is built with an empty
degree and the whole of $P$ and $Q$ in the targets of its weaves, so the transpose of a
batched convolution consumes the batch axis rather than being broadcast over it.
 states why, and what changing it would take.

The same argument is why a `cat.Rearrangement` under a `Contravariant` can never be read as
a `Rearrangement` in the value direction. Reversing a copy is addition, and addition is not a
rearrangement. `Contravariant` gets its `dom` and `cod` right and round-trips, and the value
semantics of everything in its body are supplied wherever the body is interpreted.

The reading belongs to the enclosing `Contravariant` rather than to a class of its own.
`ReversedRearrangement` was that class until 2026-09-01, when it was removed. A root term in
the sense of *Weaves, Wires, and Morphisms* carries the data its properties are derived from,
and `ReversedRearrangement` carried a mapping it never interpreted.

### 3. Differentiate the core

`Derivative(f)` is the linearisation presented as a morphism, taking the point it is taken at
and then the tangent it acts on:

```
Derivative(f).dom() = residual(f) * tangent(f.dom())     # `*`, the parallel product
Derivative(f).cod() = tangent(f.cod())
```

It needs two pieces that did not exist when the note was written.

**`tangent` is a functor on objects, and it is partial.** It is now `para/algebra/tangent.py`.
`Reals` maps to `Reals`. `Natural(n)`, which is an index, has no tangent object, so the factor
is dropped. Applying it on the domain as well as the codomain is not optional, because
`IndexSelect` and `Embedding` both take an index and their tangent domain must not contain it.
That one definition disposes of the whole top-k problem, per
[[Selection and the Reverse Pass]].

**`residual(f)` is declared a-priori, per operator.** It is now `derivative.Residual`, one per
rule, and it states what the derivative needs off the tape. `f.dom()` is the safe default,
meaning store the input, and it is the worst policy for memory and cannot express several
standard backward passes at all:

| operator | the residual it needs |
|---|---|
| `SoftMax` | its **output**, because $dx = y \odot (dy - \langle dy, y \rangle)$ is unwritable from the input without recomputing |
| `Einops` | the other operand |
| `Linear` | nothing, because the weight is not an operand, so the transpose is the whole of it |
| `ReLU` | a sign mask, which is one bit per element rather than one float |
| `IndexSelect` | the index, which is a few integers rather than the payload |
| attention | the log-sum-exp |

The residual is the $M$ of the optic, and it is generally not the domain. It is also the knob
for activation checkpointing: storing the index against recomputing it from the scores is a
choice of residual, and with the residual declared, recomputation is a rewrite that replaces a
`Grab` from a slot with the subgraph that rebuilds it. Hardwired to `dom`, there is no choice
to make.

### 4. Transpose the core

The reverse of a derivative transposes the linear legs alone, because the residual is not a
linear input and has no cotangent:

$$R[f] \;:\; \mathrm{residual}(f) \otimes T^{*}(\mathrm{cod}) \;\longrightarrow\; T^{*}(\mathrm{dom})$$

which is the reverse-derivative-category axiom of the reference [[Para Category]] cites. The
generic `Contravariant` cannot do it alone, because `Contravariant(Derivative(f))` swaps
the whole domain for the whole codomain and demands a cotangent of the point. `Derivative` needs its
own transpose, or the reversal needs an argument stating which legs are linear.

**What was done instead.** `derivative.py` declares $R[f]$ directly and never builds $D[f]$,
which collapses this step into the one above. Transposing a Jacobian is linear algebra rather
than a rewrite, since there is no syntactic route from $D[\mathrm{SoftMax}]$ to
$R[\mathrm{SoftMax}]$, so a transposer would have needed per-operator help at every
operator of consequence anyway. Declaring the composite avoids inventing one that does not pay
for itself. The cost is that forward mode is unavailable, and nothing needs it yet.

## The reverse pass assumes a locally Lipschitz definable function

Every rule in the registry writes $R[f]$ as though $f$ were differentiable. `ReLU` is the
operator where it is not, and the same question arises at `Maximum`, at an absolute value,
at a clamp and at a `TopK` tie. The condition the rules need is stated here once, and each
rule is checked against it rather than against differentiability.

### Almost-everywhere differentiability is not the condition

`ReLU` fails to be differentiable on $\{0\}$ in one dimension, and the elementwise map on
$\mathbb{R}^n$ fails on a union of $n$ coordinate hyperplanes. For a network of `ReLU`s the
failure set is a finite union of faces of a polyhedral arrangement. The set has Lebesgue
measure zero in every case, and it stops being a finite set of points as soon as $n > 1$,
so counting bad points is not a condition that survives past one dimension.

Measure zero is not sufficient either, because the class of almost-everywhere
differentiable functions is not closed under the composition that $R$ performs. Take

$$h(x) \;=\; \mathrm{ReLU}(x) - \mathrm{ReLU}(-x) \;=\; x \qquad \text{for every } x .$$

$h$ is linear and $h'(0) = 1$. The registry's rule gives
$R[h](0) = 0 \cdot 1 - 0 \cdot (-1) = 0$, so the reverse pass of a smooth function returns
a value that is not its derivative at a point where the function has one. Two expressions
denoting the same function have different reverse passes, which makes $R$ an operation on
expressions rather than on functions. [[Para Category]] states what that costs when the
construction is read as a reverse derivative category.

### The Clarke subdifferential covers one operator and not a composite

For a locally Lipschitz $f$, the Clarke subdifferential $\partial f(x)$ is the convex hull
of the limits of $\nabla f$ at nearby points where the gradient exists. At a single `ReLU`,
$\partial\,\mathrm{ReLU}(0) = [0,1]$, so the three framework conventions of $0$, $1/2$ and
$1$ all lie inside it and each is a legitimate subgradient.

The chain rule for $\partial$ is an inclusion rather than an equality. A reverse pass built
by composing subgradients can therefore land outside the subdifferential of the composite.
In the example above $\partial h(0) = \{1\}$ and the reverse pass returns $0$. Nonsmooth
analysis licenses each rule in the registry and licenses no composite of them.

### Conservative fields license the composite

Bolte and Pauwels supply the condition that does compose. A set-valued map $D$ with a
closed graph and nonempty locally bounded values is **conservative** for a locally
Lipschitz $f$ when, along every absolutely continuous curve $\gamma$,

$$\frac{d}{dt} f(\gamma(t)) \;=\; \langle v, \dot\gamma(t) \rangle
\qquad \text{for almost every } t \text{ and every } v \in D(\gamma(t)) ,$$

and $f$ admitting such a $D$ is **path-differentiable**. Conservative fields are closed
under sum and under composition, which is the property almost-everywhere differentiability
lacks, so a reverse pass assembled from conservative rules is conservative for the
composite forward map whatever value each rule chose at its kinks. Stochastic gradient
descent driven by a conservative field converges to the points where $0 \in D$.

The value at a kink is harmless to the dynamics because the training loop integrates the
field along a curve. A curve either crosses the kink in zero time or stalls on it with
$\dot\gamma = 0$, and both sides of the identity vanish in the second case. The example
above is the worked case. $D(x) = \{1\}$ off zero with $D(0) = \{0\}$ is conservative for
$h(x) = x$ while being wrong at zero as a derivative. Conservative is the weaker property,
and it is the one the training loop needs.

### The class is local Lipschitzness together with definability

The concrete sufficient condition is that $f$ is locally Lipschitz and definable in an
o-minimal structure. Semialgebraic functions qualify, and the globally subanalytic
structure with $\exp$ covers `exp`, `log`, `tanh` and the logistic function. A definable
locally Lipschitz function is path-differentiable, and definability supplies the finite
statement that counting points was reaching for. The domain admits a finite partition into
smooth manifolds, a Whitney stratification, on which the restriction of $f$ is $C^p$.
`ReLU` on $\mathbb{R}$ has the three strata $\{x<0\}$, $\{0\}$ and $\{x>0\}$, and a `TopK`
has one stratum per winning set.

Local Lipschitzness carries the part of the condition that measure zero was doing no work
on. The square root on $[0,\infty)$ fails to be differentiable at one point alone, which
satisfies every measure-theoretic condition above, and its reverse pass at that point is
$+\infty$, which is where a training run produces NaN. `ReLU` is admissible because it is
1-Lipschitz. `ReLU`, `Maximum`, an absolute value, a clamp and a `TopK` tie all satisfy the
condition. The rule to check is `arithmetic`, which differentiates the formula through
`solver.algebra.differentiate_numeric` without asking whether the result is bounded, so a
negative power or a root reaches the reverse pass unexamined. A clamp written as a root
of a square, $\tfrac{1}{2}(v + L - \sqrt{(v - L)^2})$, is such a case. Its value is the
clamp's, and its derivative is $0 \cdot \infty$ at $v = L$. [[Representing Models]]
therefore writes a clamp with `nm.RectifiedLinear`, whose derivative is a sum of
`nm.IsPositive` terms and is bounded, per [[Numerics#Expandable numerics]].

### References

- Clarke, **Optimization and Nonsmooth Analysis**, Wiley 1983. The subdifferential
  $\partial f$ and the calculus that makes its chain rule an inclusion.
- Bolte, Pauwels, **Conservative set valued fields, automatic differentiation, stochastic
  gradient methods and deep learning**, Mathematical Programming 2021;
  [arXiv:1909.10300](https://arxiv.org/abs/1909.10300). Conservative fields, their closure
  under composition, and the convergence of stochastic gradient descent driven by one.
  The source for this section.
- Davis, Drusvyatskiy, Kakade, Lee, **Stochastic subgradient method converges on tame
  functions**, Foundations of Computational Mathematics 2020;
  [arXiv:1804.07795](https://arxiv.org/abs/1804.07795). Tame meaning definable in an
  o-minimal structure, and the convergence result that definability is chosen for.
- Lee, Yu, Rival, Yang, **On correctness of automatic differentiation for
  non-differentiable functions**, NeurIPS 2020;
  [arXiv:2006.06903](https://arxiv.org/abs/2006.06903). Piecewise-analytic functions under
  an analytic partition, and the sense in which automatic differentiation is correct almost
  everywhere on them.
- Kakade, Lee, **Provably correct automatic subdifferentiation for qualified programs**,
  NeurIPS 2018; [arXiv:1809.08530](https://arxiv.org/abs/1809.08530). The condition on a
  program under which its reverse pass returns a genuine Clarke subgradient.
- Bertoin, Bolte, Gerchinovitz, Pauwels, **Numerical influence of ReLU'(0) on
  backpropagation**, NeurIPS 2021; [arXiv:2106.12915](https://arxiv.org/abs/2106.12915).
  The measured effect of the three conventions, visible in float16 and close to invisible
  in float32.

## TopK, worked through

Factor the selection into its two halves:

$$\mathrm{TopK} \;=\; \mathrm{ArgTopK} \,;\, \mathrm{IndexSelect}
\qquad \mathrm{ArgTopK} : [\mathbb{R}; n] \to [\mathrm{Nat}(n); k]$$

- **`ArgTopK`** is the only genuinely non-smooth piece, and its codomain has no tangent, so
  `Derivative(ArgTopK)` has an empty codomain and drops out of the reverse pass entirely. It
  is absent rather than zeroed and rather than straight-through.
- **`IndexSelect`** is linear in its payload, so it is its own derivative, with the index as
  its residual: $D\,\mathrm{TopK}|_x = \mathrm{IndexSelect}(S, -)$. The identity holds on the
  open cell where the winning set is fixed, and the non-smoothness is confined to ties, which
  is `ReLU`'s status at 0. Both are
  covered by the stratification condition above.
- **$R[\mathrm{IndexSelect}]$ is `Inject` at the payload's degree**, and no rule writes it
  yet. $R$ of a selecting `Linear` and of a complete `TopK` are written, per
  [[Selection and the Reverse Pass]].

The factorisation is not a device for the derivative's benefit. The models need it anyway.
DeepSeek-V4's router selects on $s + b$, which is the score plus the auxiliary-loss-free bias,
and gates on $s$, so the indices come from one wire and the values from another. The single
input pair of `TopK.complete` cannot state that, and `ArgTopK ; IndexSelect` can.
[[Sparse Axes]] covers the two presentations and why only the expanded one can be reversed.

## The rules

- **Every operator's rule assumes its map is locally Lipschitz and definable.** A kink is
  admissible and an unbounded derivative is not, per the function-class section above.
  `arithmetic` is the rule that does not check.
- **Expand to nodes before differentiating.** A core with non-trivial reindexings should never
  reach `Derivative`.
- **The tangent functor acts on datatypes rather than on shapes.** An index wire is absent
  from the reverse graph. If one appears there, something upstream is wrong.
- **A residual is declared a-priori, not derived.** Defaulting it to `dom` is a policy, and it
  should be written as one.
- **Transpose the linear legs alone.**
- **A morphism that is already linear needs no residual.** A `Linear`, or a node, a repeat
  included, propagates the cotangent on its own, so taping the input is pure cost. The
  residual is `()` and should be written as `()`.
- **A repeat need not survive into the result.** It is a `View` whose reindexing drops the
  repeated axis, which makes it a node, so whatever reads it can broadcast on its own, and
  `reindexing_absorption.absorb` folds it into that reader's reindexing, per
  [[Expression Simplification]]. It is the last thing standing between the backward pass of
  the expanded softmax and a listing in which every axis is either consumed or broadcast over
  by some operation.
- **An affine map transposes into something the term already contains**, being a contraction
  or the map itself. **A map that depends on data transposes into `Inject`**, which takes the
  map as an operand.
- **$R$ sends a `Grab` to a `Drop` on the gradient slot**, per [[Training]], and
  `backprop._para_seed` performs the substitution. A gradient slot is held beside the term
  rather than in it, so the substitution is part of interpreting a leaf. `Grab` and `Drop`
  are therefore not each other's reverse in the contravariant category, and
  `Contravariant.Contravariant` holds both without reading either.

## Gaps

Four of the five below moved on 2026-08-20, and [[Backpropagation]] carries the current list.

- ~~`Scatter` does not exist~~. Closed 2026-09-04 as `Inject` followed by a sum, for a
  selecting `Linear` and a complete `TopK`, per [[Selection and the Reverse Pass]]. The
  strided-reindexing case closed on 2026-08-20 as `transpose.ReindexTranspose`, affine and
  therefore carrying its whole map. $R[\mathrm{IndexSelect}]$ is the same `Inject` and has no
  rule yet.
- ~~`tangent` does not exist~~. It is `para/algebra/tangent.py`, and nothing has yet exercised
  the case where it drops a factor.
- ~~`residual` does not exist~~. It is `derivative.Residual`, declared per rule. `SoftMax`
  tapes its output, and an `Einops` with one operand tapes nothing.
- ~~`reverse` still has no case for the tape~~. Closed on 2026-08-31 as no gap. The reversed
  category exchanges the two ends of a morphism and states nothing about a slot, so a `Grab`
  and a `Drop` held in a `Contravariant` is the whole of what the construction says about
  them.
  `backprop` interprets the leaves and performs the swap.
- **Nothing interprets a `Rearrangement` under a `Contravariant`.** `backprop._rearrangement` now
  writes the addition that reversing a copy means, so the meaning exists twice over.

## See also

- [[Notebooks]] — the notebooks that build a two layer model,
  reverses it, composes a `MultiCategory` whose two rows read in opposite directions,
  and draws the reversed expression as its body mirrored, per [[Diagram Display]]

- [[Backpropagation]] — the implementation of this note, and what it settled
- [[Training]] — what a reverse pass is, and the four writers of a parameter slot
- [[Selection and the Reverse Pass]] — why `Scatter`, and where a selector's signal comes from
- [[Para Category]] — the construction and its reference
- [[Expression Simplification]] — undoing step 1 once the derivative is taken
- [[Linear Expansion]] — the other consumer of `expand_to_nodes`, which needs the core
- [[Weaves and Degree]] — what a degree is, and what "cleanly broadcast" means
- [[Stride Category]] — affine reindexings, whose transpose is free

---
tags: [layer/para, reference]
code: para/data_structure/Para.py
status: evolving
---

# Para Category
The para- category involves equipping morphisms with composition-neutral load and save operations that generates a tape. This tape can then be extracted via an additional function.

## A value a stream loop carries has its own two seeds

A repeated `cat.Block` overwrites some of its values on every iteration. `Para.StreamGrab`
reads the value a slot held when the iteration started and `Para.StreamDrop` writes the
value the next iteration starts from. Both are subclasses of `Grab` and `Drop`, so every
pass with two cases still has two, and the repetition is a property of the block, so the
loop variable is stated at the level of the loop. Within one iteration every operation
reads wires, and the loop drop is the one operation whose effect reaches past the body's
end, so where it stands in the body carries no meaning. A `ParaWrap` holds a
`Para.StreamSlot` for either, and `tie_tapes` refuses both, because a wire from the drop to
the grab would close a cycle. The three classes were named `Para.LoopGrab`,
`Para.LoopDrop` and `Para.LoopSlot` until 2026-09-15, when the indexed seeds below took
those names.

## A slot indexed by an iteration has its own two seeds

A repeated `cat.Block` also reads and writes one member of the tape per iteration, and
two further subclasses name the member, added 2026-09-15. `Para.LoopGrab` reads the
member of its slot that the iteration `index` wrote and `Para.LoopDrop` writes the member
the iteration `index` holds, where `index` is a numeric over the counter of a repeated
block around the seed. Both carry `tape` and `size` in the positions the plain seeds carry
them and `index` after, because tsncd constructs a term positionally, per
[[Terms Mirrored in tsncd]]. A `ParaWrap` holds a `Para.LoopSlot`, which carries the slot
and the index, and the diagram draws the slot's name with the index in its subscript,
`e_i` on the drop and `e_{3-i}` on the grab.

The counter of a block is the name `cat.Block.template` gives its tag through `index_name`,
and `nm.FreeNumeric.named` reads that name as a numeric. A `FreeNumeric` derives its id
from its name through `fd.hash_id`, per the axis identity invariant, so
`nm.FreeNumeric.named('i')` is the same numeric wherever it is written, and a drop and a
grab that carry it name the member one iteration writes and that same iteration reads. A
layer stack whose first layer drops a value and whose later layers grab it is written that
way, with the whole group of layers one repeated block. An index that is an expression of
the counter names the member of another iteration, and the expansion path of a UNet reads
the skip the contraction path dropped at `N - 1 - i`.

`tie_tapes` refuses both seeds, because a repeated block is one morphism and no wire
inside it carries the member of another iteration. `tape_members` reports one member per
iteration of the block, as it does for a plain grab inside a loop, and indexes the member
by the loops around the seed rather than by the seed's own index, so a grab at `N - 1 - i`
is grouped under the loop it stands in. `para/validate_loop_seeds.py` checks the seeds,
the entry, the wrap, the refusal and the listing.

A reduction across processors is exchanged in rounds, and two further subclasses pass
a value between processors, added 2026-09-13, where the stream seeds load from and save
to a tape that persists across one processor's iterations. `Para.ReductionDrop` sends
this processor's partial to the processors that read it in the round, and
`Para.ReductionGrab` receives a partner's partial, so the slot the two share names the
value exchanged rather than a place it is kept, and the partial itself is carried round
the loop on its wire. A `ParaWrap` holds a `Para.ReductionSlot` for either, drawn `Rdx0*`
on the operand received and `Rdx0'` on the copy sent, and `tie_tapes` refuses both,
because no wire on one processor carries what a partner sent.

## A reverse derivative category axiomatises $R$ on expressions

A Cartesian reverse differential category is a Cartesian left-additive category
$\mathcal{C}$ carrying, for every $f : A \to B$, a morphism

$$R[f] : A \times B \longrightarrow A$$

subject to seven axioms. Cockett, Cruttwell, Gallagher, Lemay, MacAdam, Plotkin and Pronk
state them, and the four that this package implements are these.

| axiom | statement | what implements it |
|---|---|---|
| RD.1 | $R[f]$ is additive in its second argument, so $R[f](a, b_1 + b_2) = R[f](a,b_1) + R[f](a,b_2)$ and $R[f](a, 0) = 0$ | `backprop._rearrangement`, which writes the addition that reversing a copy means |
| RD.3 | $R[\mathrm{id}] = \pi_2$, and a projection reverses to an injection | `Contravariant.Contravariant` on a `Rearrangement`, which exchanges the two ends of a permutation, copy or deletion |
| RD.4 | $R$ of a pairing sums the reverses of its components | `Contravariant.Contravariant` on a `ProductOfMorphisms`, whose body keeps its order, so each factor reverses in place |
| RD.5 | the chain rule, $R[f\,;g](a,c) = R[f]\big(a,\; R[g](f(a), c)\big)$ | `Contravariant.Contravariant` on a `Composed`, whose body keeps its order, so the factors compose in the opposite order when the body is read. `MultiCategory.compose_column` applies the flip |

RD.2 is the additivity of $R$ itself, and RD.6 and RD.7 constrain the second derivative.
The signature $A \times B \to A$ is the residual-and-cotangent shape that
`derivative._pointwise_reverse` returns, and `derivative.Residual` narrows the $A$ port to
the part of the input the rule needs, which the axioms permit and do not require. The
registry supplies $R$ on the generators and the axioms determine it on every composite, so
`para/` is a presentation of the free Cartesian reverse differential category on the
operator set. [[Derivatives]] is the presentation, one row per generator.

### The morphisms have to be expressions rather than functions

RD.5 is an equality, which makes $R$ an operation on morphisms. Taking the morphisms to be
functions therefore fails as soon as a generator is non-smooth. The function
$h(x) = \mathrm{ReLU}(x) - \mathrm{ReLU}(-x)$ is the identity, and the axioms compute
$R[h](0) = 0$ against $R[\mathrm{id}](0) = 1$, so $R$ is not well-defined on the function
$h$ denotes. [[Derivatives]] works the example through.

The three base categories the paper instantiates over are smooth maps, polynomial circuits
over a commutative ring, and the free category on a set of generators with a chosen $R$ for
each. A category of neural networks is the third, because a `ReLU` is admissible in neither
of the first two. Every morphism here is a `Term`, so the syntactic reading is the one the
package already has, and it is a consequence of the axioms rather than an implementation
convenience.

### Conservative fields are what make the axioms sound over a non-smooth generator

The free category is a Cartesian reverse differential category by construction, since it is
quotiented by RD.1 to RD.7. What the construction does not supply is a semantics. The
forward part of a term denotes a function, and the example above shows that the denotation
is not faithful, so $R$ cannot denote the derivative of that function.

The property that does survive is Bolte and Pauwels' conservativity, stated in
[[Derivatives]]. Conservative fields are closed under sum and under composition, which are
the two operations RD.4 and RD.5 build $R$ out of, so the assignment sending a term to its
forward function and $R$ to a conservative field for that function respects the axioms that
compose. Whether that assignment is a functor out of the free category, and in particular
whether RD.6 and RD.7 hold of it at a kink, is unresolved here. RD.6 and RD.7 hold on each
stratum of the Whitney stratification and say nothing on the kink set itself.

### A reverse derivative operator is single-valued and the Clarke subdifferential is not

$R[f]$ is a morphism, so it is total and single-valued. Clarke's $\partial f$ is set-valued,
returning $[0,1]$ at a `ReLU`'s zero. The axioms have no room for the set without moving to
a category of relations, so a reverse derivative category over non-smooth generators is
committed to a selection from $\partial f$ at every generator. The single-valuedness is not
a limitation of the axioms. It is the commitment that has to be discharged analytically,
and conservativity is what discharges it.

## Probabilistic Extension
We have a measurable space $(Z, \mathcal{Z})$, a family ${P_\theta: \theta \in \Theta}$ with open $\Theta \subset \mathbb{R}^d$, and $f \in L^1(P_\theta)$;
$\mathcal{L}(\theta) = E_{z \sim P_\theta} [f(z)] = \integral_Z f(z) P_\theta (dz)$

### Lift 1
*Domination*, there exists $\sigma$-finite with $P_\theta << \mu$ for all $\theta$ and write $\rho_\theta = \frac{dP_\theta}{d\mu}$.

*Regularity*, for $\mu$-a.e. $z$, $\theta \mapsto \rho_\theta(z)$ is differentiable on a neighbourhood $\theta_0 \in N$, and there exists $G \in L^1(\mu)$ with;

$|f(z)| \abs

**Reparameterization**
For an $\omega$ generated by an unchanging distribution and a function,
$z = g(\omega, \theta)$
We can differentiate over $\theta$, using the taped $\omega$ value.

**Score Function Estimates**
Using,
$\nabla_\theta p_\theta = p_\theta \nabla_\theta \log p_\theta$, the score $\log p_\theta = \ell_\theta$ is the quantity with algebraic value. It gives $\nabla_\theta E_\theta [f(z)] = E_{\theta} [f(z) \nabla_\theta \log]$

$f()

## References

- Cockett, Cruttwell, Gallagher, Lemay, MacAdam, Plotkin, Pronk, **Reverse Derivative
  Categories**, CSL 2020 (LIPIcs 152, 18:1–18:16);
  [arXiv:1910.07065](https://arxiv.org/abs/1910.07065).
  The seven axioms above, the equivalence with a Cartesian differential category carrying
  a dagger on its linear maps, and the free construction on a set of generators with a
  chosen $R$ for each. The free construction is what `para/registries/derivative.py` and
  `para/data_structure/Contravariant.py` present between them.

- Cruttwell, Gavranović, Ghani, Wilson, Zanasi, **Categorical Foundations of
  Gradient-Based Learning**, ESOP 2022 (LNCS 13240, 1–28);
  [arXiv:2103.01931](https://arxiv.org/abs/2103.01931).
  The reference construction for this note. A learner is a morphism of
  $\mathbf{Para}(\mathbf{Lens}(\mathcal{C}))$ over a *reverse derivative category*: the
  forward map and its backward are one object, so composing learners is just composition
  in $\mathbf{Para}$ and the tape is the lens residual. Two things follow directly.
  First, the optimiser lives in the same structure. SGD, momentum, AdaGrad and Adam
  are all lenses on the parameter port, which is the slot a forward-driven parameter has
  to occupy here. The paper's port is written by the reverse pass alone, and the load and save
  tape above is the generalisation that lets the forward pass write to it too. Second, it
  instantiates over **discrete** base categories (boolean circuits), which is the nearest
  thing in the literature to the non-smooth selection problem, meaning the top-$k$ of a router or
  a sparse-attention indexer, where the derivative is $0$ a.e. and the decision has to be
  carried as residual rather than differentiated through.

## See also

- [[Derivatives]] — the presentation of $R$, one rule per generator, and the function
  class the rules assume
- [[Training]] — the tape in place of the lens residual, and the four writers of a
  parameter slot

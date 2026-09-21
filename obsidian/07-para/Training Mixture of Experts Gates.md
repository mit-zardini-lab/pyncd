---
tags: [layer/para, concept]
code: para/data_structure/Para.py, deepseek/data_structure.py
status: speculative
---

# Training Mixture of Experts Gates

## What it is

[[Training]] classifies a parameter by which pass writes it, and names the
auxiliary-loss-free routing bias as the case no lens can hold. This note is the literature
behind that case. It states how the published mixture-of-experts models train the router:
which channels carry a gradient into the router weight, which of those channels reach an
expert the token did not select, and what each channel's update depends on.

The answer to the second question is that no channel carries a counterfactual. A row of the
router weight belonging to an expert outside the selected set is updated by how loaded that
expert is, by how large its own logit is, and in some formulations by a suppression term
proportional to its own probability. No term in the update depends on what that expert
would have produced. Balance is therefore imposed by a term written for the purpose, and
that term is what moves the unselected rows.

## Where it lives

Nothing in this repository trains anything. `deepseek/data_structure.py` holds the `TopK`
the routing is written with, `notebooks/sota/DeepSeekV41Flash.ipynb`
builds the forward routing of DeepSeek-V4.1-Flash, and `para/data_structure/Para.py` holds
the tape the updates below would be written through. The note is here because the four
writers in [[Training]] were classified against a router, and this is the detail that
classification compresses.

The smallest mixture that still has a gate is one `Linear` for the router and one `Linear`
holding every expert, and its reward $r_i$ and shared softmax denominator are the two
quantities the sections below are written in.

The selecting form puts a `TopK` between the router and the softmax, which is the
normalisation over $S$ of the table below. The reverse of the selection is `Inject`, per
[[Selection and the Reverse Pass]], which writes zero at every row the token did not
select, so a derived pass gives an unselected row of $W^g$ nothing, which is the claim the
table below argues. [[DeepSeek-V3 Backward Pass]] derives the same for a whole layer with
a sigmoid router and normalised gates.

## The routing morphism

For a token $u \in \mathbb{R}^d$, a router weight $W \in \mathbb{R}^{n \times d}$ and $k$
of the $n$ experts selected:

$$\ell_i = \langle u, W_i \rangle, \qquad s = \varphi(\ell), \qquad
S = \mathrm{TopK}_k(s + b), \qquad g_i = \frac{s_i}{\sum_{j \in S} s_j}, \qquad
y = \sum_{i \in S} g_i E_i(u)$$

Published models differ in $\varphi$, in the set the normalisation of $g$ is taken over, and
in what balances the load.

| model | $\varphi$ | $g$ normalised over | balance |
|---|---|---|---|
| Shazeer et al. 2017 | softmax after masking to $S$ | $S$ | importance loss and load loss, both on the coefficient of variation |
| GShard, Switch | softmax over all $n$ | all $n$ | auxiliary loss $\alpha n \sum_i f_i P_i$ |
| ST-MoE | softmax over all $n$ | all $n$ | the same auxiliary loss and a router z-loss |
| Mixtral | softmax after masking to $S$ | $S$ | auxiliary loss |
| DeepSeek-V2 | softmax over all $n$ | all $n$ | expert-level, device-level and communication-level auxiliary losses |
| DeepSeek-V3 | sigmoid, per expert | $S$ | a bias updated by a sign rule, and a sequence-wise auxiliary loss at $\alpha = 10^{-4}$ |
| GLM-4.5 | sigmoid, per expert | $S$ | a bias updated by a sign rule |
| Qwen3 | softmax over all $n$ | $S$ | auxiliary loss, over the global batch |
| DeepSeek-V4 | $\sqrt{\vphantom{x}\smash{\mathrm{softplus}}}$ | $S$ | a bias updated by a sign rule, at speed 0.001 |

The DeepSeek-V4 row is the one [[Training]] works against, at arXiv 2606.19348. Two columns
of the table decide everything below. The normalisation column decides whether the main loss
reaches an unselected row at all. The balance column is what reaches it when the main loss
does not.

## The gradient into a selected row

Write $r_i = \langle \partial L / \partial y, \; E_i(u) \rangle$ for the inner product of the
incoming cotangent with expert $i$'s output. It is the only quantity in the whole scheme that
depends on what an expert computed. It exists for $i \in S$ and for no other expert, because
$E_j(u)$ was never evaluated for $j \notin S$.

$r_i$ is a bandit reward on the arm that was pulled. It is measured after the choice, it has
no exploration term, and there is no second evaluation to compare it against. The update to
$W_i$ is $r_i$ times a scalar times $u$, so it is an outer product on the $k$ rows of $W$ that
$S$ names, of which DeepSeek-V4 selects 6 of 384.

## The gradient into an unselected row

The question has a different answer for each of the three normalisations in the table.

**Normalised over all $n$.** With $g = p = \mathrm{softmax}(\ell)$ and the entries outside $S$
masked to zero after the softmax, the denominator $\sum_{j=1}^{n} e^{\ell_j}$ contains every
logit, so every logit has a derivative. Using $\partial p_i / \partial \ell_j = p_i(\delta_{ij} - p_j)$,

$$\frac{\partial L}{\partial \ell_j} = -\,p_j \sum_{i \in S} p_i r_i \qquad (j \notin S)$$

The factor $\sum_{i \in S} p_i r_i$ is the same for every unselected expert, so the term ranks
the unselected rows by their current probability and by nothing else. When the selected
experts were useful, gradient descent decreases every unselected logit, most for the expert
that came closest to being selected. The term is a competition between the rows of a softmax
and it does not depend on $E_j$.

**Normalised over $S$.** Mixtral masks to $S$ before the softmax, and DeepSeek-V3 and V4 divide
by $\sum_{j \in S} s_j$. Every quantity in the expression is then a function of the rows in $S$
alone, and $\partial L / \partial W_j = 0$ exactly for $j \notin S$. [[Training]] states the same
fact structurally: the index wire carrying $S$ has no cotangent, so the rows $S$ did not name are
absent from the reverse graph. Moving from V2's normalisation to V3's removed the last channel
from the main loss to an unselected row.

**Through the auxiliary loss.** Switch's loss is $L_{\mathrm{aux}} = \alpha n \sum_{i=1}^{n} f_i P_i$,
where $f_i$ is the fraction of the batch's tokens dispatched to expert $i$ and
$P_i$ is the mean over the batch of $p_i$. $f_i$ is a count, so it is a constant in the
derivative, and $P_i$ is differentiable in every logit. The result is

$$\frac{\partial L_{\mathrm{aux}}}{\partial \ell_j} = \frac{\alpha n}{T}\, p_j
\Big( f_j - \sum_{i} f_i p_i \Big)$$

for every $j$, selected or not. An expert whose load is above the load-weighted mean has its
logit decreased and one below it has its logit increased. The update depends on the load of
expert $j$ and on nothing that expert $j$ computed. It is the channel that the models in the
first four rows of the table rely on to move an unselected row.

**Through the router z-loss.** ST-MoE adds
$L_z = \frac{1}{T} \sum_t \big(\log \sum_i e^{\ell_{t,i}}\big)^2$, whose derivative is
$\frac{2}{T} (\log \sum_i e^{\ell_i})\, p_j$ for every $j$. It reaches every row and it
decreases every logit. Its purpose is the numerical range of the exponentials in bfloat16
rather than balance.

**Through a smooth estimate of selection.** Shazeer's noisy top-k adds Gaussian noise scaled by
a second learned projection before the selection, and then defines
$\mathrm{Load}_i = \sum_t \Pr[\,\ell_i(t) + \text{noise lands in the top } k\,]$, in closed form
through the normal cumulative distribution function. The probability is differentiable in
$\ell_i$ even for a token where the sample missed, so $L_{\mathrm{load}} = w \cdot \mathrm{CV}(\mathrm{Load})^2$
gives an unselected row a term measuring how close it came to selection. Among the channels
listed here it is the only one whose update to an unselected row depends on the selection
boundary rather than on a magnitude.

**Through an estimator for the discrete choice.** The selection itself has no derivative, and
the ordinary construction trains it only because the gate $g_i$ multiplies the expert output.
SparseMixer and SparseMixer-v2, used in GRIN MoE, replace that surrogate with a mid-point
estimate of the derivative through the discrete choice, so the routing decision is trained
rather than the scalar alone. Straight-through Gumbel-softmax is the cruder version of the
same idea, with a hard sample forward and the soft Jacobian back.

**Through no gradient at all.** Loss-Free Balancing, adopted in DeepSeek-V3 and in GLM-4.5,
updates $b_i \leftarrow b_i + \gamma \cdot \mathrm{sign}(\bar{c} - c_i)$ after each step, where
$c_i$ is the observed count for expert $i$ and $\bar{c}$ the mean count. The bias enters the
selection score $s + b$ and is excluded from the gate value $g$, so $L$ is not a function of
$b$ and $\partial L / \partial b$ does not exist. The stated reason for excluding it is that an
auxiliary loss adds a term whose minimum is not the language-modelling minimum, and the sign
rule balances without adding one. It is class 2 of [[Training]]'s four writers, and it is why a
tape is needed where a lens would do.

## What reaches an unselected row

| channel | reaches a row outside $S$ | the update depends on |
|---|---|---|
| the gate path | no | $r_i$, for $i \in S$ |
| the shared softmax denominator | only when $g$ is normalised over all $n$ | $p_j$ and $\sum_{i \in S} p_i r_i$ |
| the auxiliary balance loss | yes, every row | $p_j (f_j - \sum_i f_i p_i)$ |
| the router z-loss | yes, every row | $p_j$ and the log-sum-exp |
| the noisy top-k load term | yes, every row | the probability that $j$ enters the top $k$ |
| a discrete-choice estimator | the rows near the boundary | the loss under a flipped decision |
| the bias sign rule | it has no gradient | $c_i$ against $\bar{c}$ |

## Routing schemes that remove the question

Three published schemes balance by construction, so no row is unselected in the sense above.

- **Expert choice.** Each expert takes its own top-$k$ tokens from the batch, so every expert
  is used exactly to capacity and no auxiliary loss is needed. The assignment reads the whole
  batch at once, so a decoder trained this way sees later positions when routing an earlier
  one, which is why it is used in encoders and in vision.
- **BASE layers.** The assignment of tokens to experts is a linear assignment problem, solved
  with an auction algorithm at training time and replaced by an argmax at inference. The
  balance constraint is in the solver.
- **Sinkhorn routing.** The score matrix is normalised towards doubly stochastic by alternating
  row and column scalings, which is entropic optimal transport, and the assignment is read off
  the result.

## The rules

- **Ask which set the gate is normalised over before asking what trains the router.** The two
  denominators differ by the terms outside $S$, and the difference decides whether an
  unselected row appears in the reverse graph at all.
- **A balance term carrying a factor of $p_j$ cannot revive a dead expert.** Both the auxiliary
  loss and the z-loss scale their update to row $j$ by that expert's own probability, so an
  expert whose probability has already collapsed receives an update near zero and stays
  collapsed. The bias sign rule has no such factor, because it acts on the selection score
  directly.
- **Do not read the gate gradient as a comparison between experts.** $r_i$ is measured on the
  expert that ran. Two experts are never evaluated on the same token, so no term in the
  gradient states that a different choice would have been better.
- **Count the load over the global batch.** A micro-batch of a few thousand tokens is a small
  sample of the domain, and forcing balance within it penalises a router that has specialised
  correctly. Qwen3 and OLMoE both report the global-batch statistic as the version that works.
- **A dropped token trains nothing.** With a capacity factor, a token beyond an expert's
  capacity contributes to no expert's output, so its selected rows receive no gradient either.
  MegaBlocks removes the drop by replacing the padded batched matrix multiply with a
  block-sparse one.
- **Keep the router in float32.** Switch's selective precision applies to the router alone, and
  ST-MoE's z-loss exists for the same reason. The selection is a comparison of close numbers,
  and a change of selection is discrete.

## Gaps

- **The loss and the optimiser have no representation**, per [[Training]]. Every update in this
  note is a morphism that would have to be written before any of it could be derived.
- **A balance statistic is a reduction over the batch axis.** $f_i$ and $c_i$ are counts
  over tokens, so the statistic is a fold over the token axis and the sign rule reads its
  result. Nothing in this repository writes that fold.
- **The count $c_i$ is a forward `Drop` with no reverse `Grab`.** It is the concrete instance of
  the slot that [[Para Category]] allows and a lens does not.
- **Expert choice is a `TopK` along the other axis of the same score matrix.** [[Sparse Expansion]]
  should express it without a new operator, and that has not been tried.
- **The reverse of the gather is a scatter**, per [[Selection and the Reverse Pass]], and the
  operator set is not closed under $R$ until one is written.

## See also

- [[Training]] — the four writers of a parameter slot, of which the routing bias is class 2
- [[Selection and the Reverse Pass]] — what top-$k$ does to the reverse functor
- [[Para Category]] — why a named slot is needed where a lens has a residual
- [[Sparse Axes]] and [[Sparse Expansion]] — the two presentations of the selection
- [[Representing Models]] — the rules a SOTA notebook is written to
- [[Open Gaps]] — item 2, the reverse functor's missing rules

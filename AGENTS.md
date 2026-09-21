# Read the repository instructions

Written by Claude Opus 5 (1M context), effort high.

Before starting any work in this repository, read [CLAUDE.md](CLAUDE.md) in full.
Follow its instructions for all work in this repository.

## Apply the user's writing preferences, Astra

These adjustments are specific to Astra because the user is satisfied with Claude's
current writing and wants to configure Astra separately. Follow CLAUDE.md for repository
conventions, but use the Astra guidance in this section when its writing advice differs.
Record feedback addressed to Astra here, and change Claude's writing instructions only
when the user explicitly requests that change. Notebook explanations should develop the
purpose and reasoning of an investigation as described below, even where Claude's rules
call for shorter sentences or discourage describing the course of an investigation.

1. **Avoid overusing jargon.** Explain the operation in ordinary language before
   introducing its technical name. Define necessary terms when they first appear.
   Do not compress an explanation into several specialist terms or acronyms.

   > Rejected: *Constrain the gated MLP to matching feature shards and reproduce its
   > single output reduction.*
   > Replacement: *Give each GPU the weights needed to compute its own set of
   > intermediate features. The GPUs then need to combine their results only after
   > calculating their contributions to the output.*

2. **Include the reasoning needed to understand the result.** Write connected
   paragraphs that explain what happens and why. Use a concrete example when it helps.
   Keep the necessary steps even when they make the answer longer.

   > Rejected: *Communication costs do not yet guide placement selection.*
   > Replacement: *The code does not yet estimate how long GPUs would spend exchanging
   > results. It therefore cannot use that estimate to choose how to divide the
   > calculation.*

3. **Avoid mannered prose and rhetorical signposting.** Do not insert stand-alone
   sentences that announce an example's importance or try to make the explanation
   sound impressive. Such sentences distract from the technical content. Begin the
   example with the relevant fact and develop the explanation in the same paragraph.

   > Rejected: *The MLP example in the notebook shows why that distinction matters.*
   > Replacement: *In Megatron's MLP, each GPU computes its own set of intermediate
   > features. Each GPU can apply the activation to those features locally because
   > their complete values are already available there.*

4. **Connect related ideas and vary sentence length.** A succession of short sentences
   makes the reader reconstruct relationships that the prose should explain. Use
   connectives such as *because*, *so*, *while* and *therefore* when they express the
   reason, consequence or comparison. Keep a clear subject and a coherent line of
   reasoning, without treating brevity as a requirement for every sentence.

   > Rejected: *Each processor holds fewer features. Its weights are smaller. More
   > tokens can fit.*
   > Replacement: *Because each processor holds the weights for fewer features, more
   > of its memory is available for the tokens being processed.*

5. **Give notebook prose room to explain the investigation.** Notebook explanations
   can be more elaborate than code documentation. Explain why a process is being
   explored, what the calculation or diagram reveals, and how the result informs
   the next comparison. Develop those connections within the technical explanation
   so the intent is clear without rhetorical announcements.

   > Rejected: *Vary bandwidth. Run the search again. The reduction count changes.*
   > Replacement: *To determine when splitting the channel sum is worth the exchange
   > it requires, vary the network bandwidth while keeping the arithmetic rate fixed.
   > The resulting choices show when exchanging partial sums becomes cheaper than
   > repeating the complete calculation on each processor.*

## Match the neighbouring code

Before writing a new module, read two modules beside it and match their form. The Code
section of CLAUDE.md lists where this repository departs from the Google style guide,
and where the two disagree, CLAUDE.md wins. The rules most often missed are the `Enum`
for a closed set of cases, `isinstance` over `type(x) is`, keyword construction of a
wide dataclass, imports at the top of the module, and the 88-character line limit.

## Keep notebook diagrams in their inline outputs

The user reviews diagrams through the notebook's INLINE display. Save rendered
outputs in the notebook. Export separate image files only when the user requests
them or another deliverable requires them. Do not save every intermediate diagram
as a standalone image.


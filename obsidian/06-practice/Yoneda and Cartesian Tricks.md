---
tags: [layer/practice, concept]
code: graphs/processing/Hypergraph2Morphism.py, advanced_axis_dynamics/algebra/move_reads_backwards.py, advanced_axis_dynamics/algebra/slide_causal_reads_backwards.py
status: stable
---

# Yoneda and Cartesian Tricks

## What they are

Two semantics-preserving rewrites, named after their justifications in
*Weaves, Wires and Morphisms*, simplify how an expression is drawn.

The Yoneda trick rests on naturality. A pointwise operator commutes with an
affine reindexing. An operator broadcast over its degree is the *same
function at every index*, and a reindexing only renames which index it is
performed at, so `operator ; reindexing = reindexing ; operator`, with the
operator's degree re-read through the reindexing. In V4-Flash's compressor,
`W^Z` (pointwise in the position) slides underneath the window view
`x = 4b + w + shift`: the phase reads `reindexing ; linear`, and the
positional bias then rides the `Linear`'s own `bias` flag instead of an
explicit parameter array plus an addition.

The Cartesian trick rests on the comonoid law. Copying is the comonoid of a
Cartesian category: $\Delta ; (f \times f) = f ; \Delta$. A copy that feeds
two *equal* morphisms is that morphism performed once and copied after. In
the compressor, once `W^Z` moved, both branches began with the same window
view, so it is one window view followed by the copy.

Applied in sequence they compound: the first rewrite is what *creates* the
two equal reindexings the second collapses.

## The rule

**Draw the smaller form. The performance is worked out
afterwards, with other tools.** Both tricks move where computation is
*nominally* performed. `W^Z` per window slot instead of per token reads the
overlapping phases' shared positions twice, and the expression is not the
schedule. What is materialised, reused or recomputed is settled after the
expression is written, by whatever derives the schedule from it. Choosing the
presentation to look cheap would fix a scheduling decision in the expression.
[[Sparse Axes]] separates a selection's presentation from its cost in the same
way, and [[Linear Expansion]] keeps both forms of a weight for the same reason.

## The mechanical forms of the two tricks

`hypergraph_to_morphism`'s `hoist_rearrangements`
([[Hypergraph to Morphism]]) is the Cartesian trick run as a normalization:
a product child's leading rearrangement hoists out and merges forward, so
cascaded copies merge into one fan at the earliest point (found when the
expanded MoE drew its router input being copied at the router).

`move_reads_backwards` in `advanced_axis_dynamics/algebra/` runs the Yoneda trick
mechanically, per [[Advanced Axis Dynamics]]. It carries a read of an expression's
result back towards its inputs and composes the read into every view on its path. At a
copy whose branches ask for one read, it writes that read once, which is the Cartesian
trick. `slide_causal_reads_backwards` starts the same crawl at every causal read, which
gives the CausalSlide form below. `caching.algebra.derive_cached_pass` carries the read
of the new tokens of a pass of generation, and places a cache wherever a causal read
reads earlier tokens.

## The CausalSlide is the standard form of a displayed expression

The user ruled on 2026-09-26 that an expression is displayed with every causal read
moved back as far as it goes, and named that form the CausalSlide. The causal mask of
attention reads token `i_x - i_w` for every slot `i_w`. Each operator between the mask
and the copy that feeds the queries is broadcast over the tokens, so the mask can stand
before it. In the CausalSlide the mask reads the state at that copy, and the key and
value projections are drawn over the tokens and the slots. The form does not change the
function. It is also the smaller form here, because the key and value branches share one
read where the model as built holds two. The projections are written once per token and
slot. The CausalSlide states what each slot reads, and the schedule is worked out
afterwards, per the rule above.

The position of the causal read on its path is the placement of a cache, as
[[Deriving Caches by Dragging the New Tokens]] shows. The operators before the read
compute once per token and their result can be cached, and the operators after it
compute once per token and slot. The CausalSlide puts every operator after the read, so
the cache implied by the CausalSlide holds the input of the sublayer. Each cheaper
placement slides the read forward past the operators that should compute once per token.
`slide_causal_reads_back_past` places the read after a given set of operators.

The notebooks under `notebooks/website/classic/` and `notebooks/website/modern/` draw
their models in the CausalSlide.
`notebooks/website/classic/validate_attention_is_all_you_need.py` slides the masked
sublayer of the transformer, and checks that the slid sublayer reads its input through
one mask and that the slide changes no operator other than the mask.

## See also

- [[Representing Models]] — the modelling rules these join
- [[Deriving Caches by Dragging the New Tokens]] — the read of the new tokens carried
  by the same crawl, and the placements of a cache as positions of the causal read
- [[Crawlers]] — the reverse crawl that carries a read
- [[Stride Category]] — why affineness is what makes the commuting legal
- [[Broadcasted Category]] — what "pointwise over the degree" means

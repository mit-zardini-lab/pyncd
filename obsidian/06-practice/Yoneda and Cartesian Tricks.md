---
tags: [layer/practice, concept]
code: graphs/processing/Hypergraph2Morphism.py
status: stable
---

# Yoneda and Cartesian Tricks

## What they are

Two semantics-preserving rewrites, named after their justifications in
*Weaves, Wires and Morphisms*, for simplifying how an expression is drawn:

**The Yoneda trick (naturality).** A pointwise operator commutes with an
affine reindexing. An operator broadcast over its degree is the *same
function at every index*, and a reindexing only renames which index it is
performed at, so `operator ; reindexing = reindexing ; operator`, with the
operator's degree re-read through the reindexing. In V4-Flash's compressor,
`W^Z` (pointwise in the position) slides underneath the window view
`x = 4b + w + shift`: the phase reads `reindexing ; linear`, and the
positional bias then rides the `Linear`'s own `bias` flag instead of an
explicit parameter array plus an addition.

**The Cartesian trick (the comonoid law).** Copying is the comonoid of a
Cartesian category: $\Delta ; (f \times f) = f ; \Delta$. A copy that feeds
two *equal* morphisms is that morphism performed once and copied after. In
the compressor, once `W^Z` moved, both branches began with the same window
view, so it is one window view followed by the copy.

Applied in sequence they compound: the first rewrite is what *creates* the
two equal reindexings the second collapses.

## The rule

**Draw the smaller form. The performance is worked out
afterwards, with other tools.** Both tricks move where computation is
*nominally* performed — `W^Z` per window slot instead of per token reads the
overlapping phases' shared positions twice, and the expression is not the
schedule. What is materialised, reused or recomputed is settled after the
expression is written, by whatever derives the schedule from it. Choosing the
presentation to look cheap would fix a scheduling decision in the expression,
which is the same separation [[Sparse Axes]] draws between a selection's
presentation and its cost, and the reason both routes of a
`Linear` agree.

## Where they are applied mechanically

`hypergraph_to_morphism`'s `hoist_rearrangements`
([[Hypergraph to Morphism]]) is the Cartesian trick run as a normalization:
a product child's leading rearrangement hoists out and merges forward, so
cascaded copies merge into one fan at the earliest point (found when the
expanded MoE drew its router input being copied at the router). The Yoneda
trick has no mechanical form yet, and in the notebooks it is applied by hand,
as a modelling choice.

## See also

- [[Representing Models]] — the modelling rules these join
- [[Stride Category]] — why affineness is what makes the commuting legal
- [[Broadcasted Category]] — what "pointwise over the degree" means

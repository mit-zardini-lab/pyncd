---
tags: [layer/hypergraphs, concept]
code: graphs/processing/hypergraph_crawler.py
status: stable
---

# Crawlers

## What it is

A crawler propagates a guide alongside an expression. Where a [[Functors|functor]] applies
one rule everywhere, a crawler carries state through the traversal, so the rule at each
morphism can depend on what arrived from the previous one.

There are two directions:

| class | direction | what the guide is |
|---|---|---|
| `ForwardCrawler` | `dom` to `cod`, top-down | what the inputs are |
| `ReverseCrawler` | `cod` to `dom`, bottom-up | what the outputs are required to be |

`BuiltForwardCrawler` and `BuiltReverseCrawler` add the machinery for rebuilding the
expression as they go. The plain versions observe alone.

A subclass supplies `object_processor` and `root_processor`. The base provides the
traversals over `Composed`, `ProductOfMorphisms`, `Block`, `Rearrangement` and the
hypergraph forms.

## Two reverse crawls carry an index or a read towards the inputs

A rewrite whose rule depends on what the surrounding expression requires cannot be written
as a [[Functors|functor]], which applies one rule everywhere. A rewrite that reads the
array each codomain wire is required to hold, and derives what the domain wires therefore
have to hold, is a `ReverseCrawler`. Two such crawls are defined in
`advanced_axis_dynamics/algebra/`, per [[Advanced Axis Dynamics]].

### `drag_index_backwards.IndexPinCrawler`

The guide of `IndexPinCrawler` holds one pin for each axis of a wire. A pin is the one
index of the axis that is read, or `None` where no single index is read. The crawl
carries one index of a result back to the inputs, and it rebuilds the expression with a
pinned axis at every position reached by the index. Inside each root it runs
`ReindexingPinCrawler`, a `ForwardCrawler` over the stride category, because a reindexing
maps the degree of the result to the degree of the operand. Inside a block operator it
runs itself on the body with the pins of the box's targets.

### `move_reads_backwards.ReadCrawler`

The guide of `ReadCrawler` is the pending read of each wire. A pending read is the
reindexing of the view that would stand on the wire, or `None` where nothing is read. The
crawl carries a read of a result back towards the inputs. It composes the read into every
view on its path and carries it through every operator broadcast over the axes read by it,
and it rebuilds the expression over the arrays returned by the reads. Where the branches
of a copy ask for different reads, the reads stop at the copy, and each distinct read is
written once after it as a view. The crawl runs itself on the body of a block operator
with the read of the box's targets, and it turns a read of an `ops.Arrange` into the
values of the index.

`slide_causal_reads_backwards.CausalReadCrawler` subclasses `ReadCrawler` and starts the
crawl at every causal read, which gives the CausalSlide of
[[Yoneda and Cartesian Tricks]]. `caching/algebra/derive_cached_pass.py` subclasses it as
`NewTokenCrawler`, which carries the read of the new tokens of a pass, per
[[Deriving Caches by Dragging the New Tokens]].

## The guides of a wire read twice are merged by `merge_guides`

`Crawler.merge_guides` returns the one guide carried by a wire read by several consumers.
The default is `util.iallequals`, so every consumer must have given the wire the same
guide. `IndexPinCrawler` overrides it with `agreed_pins`, which
keeps the pin of an axis where every consumer gave the same pin and frees the axis
elsewhere. An index that reaches a copy on one branch alone therefore stops there rather
than raising. `ReadCrawler` keeps a read where every consumer asked for the same read and
hands the wire no read otherwise. `realign_guide` takes the same merge, so a wire entering
a block twice is merged by the crawler's rule at both crossings. The method was added on
2026-09-25.

## The rule to internalise

> [!important] The guide handed upstream is read off the rebuilt morphism
> `BuiltReverseCrawler.root_processor` rebuilds the morphism, then reads a fresh guide off
> the domain of the rebuilt morphism through `_object_to_guide`. The guide that arrived
> from downstream is not passed on unchanged, so a rewrite that changes what a wire holds
> hands the changed form upstream without doing anything further.

## A guide crosses into a block by node rather than by position

> [!important]
> `HypergraphBlock.template` deduplicates the block's `dom` by node, and the body's `dom` is
> not deduplicated. A wire that enters a block twice is therefore listed once on the block's
> `dom` and twice on the body's, and a guide zipped positionally across the two is shifted
> by one for every wire after the duplicate.
>
> `Crawler.propagate_graph` therefore realigns the guide with `realign_guide` at both
> crossings, on the way in through `entry_nodes` and on the way out through `exit_nodes`,
> which are `dom` and `cod` for the forward direction and `cod` and `dom` for the reverse. The
> two guides given to a wire listed twice are merged by the crawler's `merge_guides`, which
> by default requires them to be equal.
>
> Before the fix, on 2026-08-21, the shift was silent whenever the duplicate happened to
> be last, so a graph whose repeated wire stood anywhere else handed each later wire the
> guide of its neighbour.

## A repeated block returns its guide unchanged

A block whose `repetition` is not one denotes a loop, so the guide leaving its body is
the guide handed to the next iteration. The body must therefore return its guide
unchanged. `require_the_guide_kept_by_a_loop` checks the condition for a `cat.Block` and a
`HypergraphBlock` in both directions, and raises `RepeatedBlockChangesTheGuide` with the
repetition and the number of wires where the body changes the guide. Three bare `assert`
statements stood in its place until 2026-09-26.

## See also

- [[Functors]] — the uniform alternative, for a rule that needs no context
- [[Hypergraph Analysis]] — connectivity, which a crawler consults
- [[Hypergraphs]] — the form a crawl walks
- [[Advanced Axis Dynamics]] — the index and the read carried backwards by the two crawls

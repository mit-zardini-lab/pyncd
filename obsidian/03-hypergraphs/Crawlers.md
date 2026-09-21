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

A subclass supplies `object_processor` and `morphism_processor`. The base provides the
traversals over `Composed`, `ProductOfMorphisms`, `Block`, `Rearrangement` and the
hypergraph forms.

## Where a crawl is used

No pass in this repository constructs a crawler today. The machinery is here because a
rewrite whose rule depends on what the surrounding expression requires cannot be written
as a [[Functors|functor]], which applies one rule everywhere. A rewrite that reads the
array each codomain wire is required to hold, and derives what the domain wires therefore
have to hold, is a `ReverseCrawler`.

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
> which are `dom` and `cod` for the forward direction and `cod` and `dom` for the reverse. A
> wire listed twice has to carry the same guide both times, which `iallequals` checks.
>
> Before the fix, on 2026-08-21, the shift was silent whenever the duplicate happened to
> be last, so a graph whose repeated wire stood anywhere else handed each later wire the
> guide of its neighbour.

## See also

- [[Functors]] — the uniform alternative, for a rule that needs no context
- [[Hypergraph Analysis]] — connectivity, which a crawler consults
- [[Hypergraphs]] — the form a crawl walks
